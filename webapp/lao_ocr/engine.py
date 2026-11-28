"""Lao + Thai + English OCR engine: one object, no PyTorch needed.

    from lao_ocr.engine import LaoOCR
    ocr = LaoOCR()                       # loads recognizer + text-line detector + Lao word helper
    print(ocr.read("page.jpg").text)     # a photo / scan of a whole page, or one cropped line
    for label, res in ocr.read_file("book.pdf"):
        print(label, res.text)

Pipeline (the same as the web app):
  1. text-line detector (PP-OCRv6 DB, models/OpenThai2.1-OCR-Text-*/det.onnx) finds every text line,
     straightens tilted / perspective lines and sorts them in reading order
     (falls back to projection cutting, lao_ocr/layout.py, when the detector is missing)
  2. recognizer (models/rec/rec.onnx, trained here) reads each line
  3. Lao character language model (models/lm/*.npz) re-ranks look-alike letters / lost tone marks
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .layout import is_junk_line, is_single_line, segment_page, to_tensor

PKG_ROOT = Path(__file__).resolve().parents[1]

LANGS = {
    'auto': 'Lao / Thai / English (automatic)',
    'lao+en': 'Lao + English',
    'th+en': 'Thai + English',
    'lao': 'Lao only',
    'th': 'Thai only',
    'en': 'English only',
}


class Codec:
    """CTC alphabet: index 0 = blank, then rec_dict.txt, then space (same as lao_ocr.core.Codec)."""

    def __init__(self, dictionary: list[str]):
        self.dictionary = list(dictionary)
        self.chars = [''] + self.dictionary + [' ']

    def decode(self, seq) -> str:
        out, prev = [], None
        for v in seq:
            i = int(v)
            if i and i != prev:
                out.append(self.chars[i])
            prev = i
        return ''.join(out)


@dataclass
class OCRResult:
    text: str
    lines: list[str] = field(default_factory=list)
    mode: str = ''
    line_height: float = 0.0
    skew: float = 0.0
    warning: str | None = None
    ms: float = 0.0
    rectify_summary: str = ''
    markdown: str = ''
    bill: object = None  # BillReport from lao_ocr.bill_checker


def _find_rec_dir(root: Path) -> Path:
    if (root / 'models' / 'rec' / 'rec.onnx').is_file():          # install package layout
        return root / 'models' / 'rec'
    try:                                                           # training folder: the web app's choice
        s = json.loads((root / 'app_settings.json').read_text(encoding='utf-8'))
        if s.get('model_dir') and (root / s['model_dir'] / 'rec.onnx').is_file():
            return root / s['model_dir']
    except Exception:
        pass
    runs = sorted(root.glob('runs/*/onnx/rec.onnx'), key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        return runs[0].parent
    raise FileNotFoundError('no recognizer found (models/rec/rec.onnx)')


def _find_lm(root: Path) -> Path | None:
    try:
        s = json.loads((root / 'app_settings.json').read_text(encoding='utf-8'))
        if s.get('lm_file') and (root / s['lm_file']).is_file():
            return root / s['lm_file']
    except Exception:
        pass
    for d in (root / 'models' / 'lm', root / 'models'):
        for name in ('lao_charlm_v6.npz', 'lao_charlm_v5.npz', 'lao_charlm_v4.npz', 'lao_charlm_v3.npz', 'lao_charlm_v2.npz', 'lao_charlm.npz'):
            f = d / name
            if f.is_file():
                return f
        files = sorted(d.glob('lao_charlm*.npz'))
        if files:
            return files[-1]
    return None


class LaoOCR:
    def __init__(self, root: str | os.PathLike | None = None, rec_dir=None, lm_file=None,
                 use_detector: bool = True, use_gpu: bool = True, threads: int = 4):
        self.root = Path(root) if root else PKG_ROOT
        self.rec_dir = Path(rec_dir) if rec_dir else _find_rec_dir(self.root)
        self.codec = Codec((self.rec_dir / 'rec_dict.txt').read_text(encoding='utf-8').splitlines())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        providers = []
        avail = ort.get_available_providers()
        if use_gpu:
            for p in ('CUDAExecutionProvider', 'DmlExecutionProvider', 'CoreMLExecutionProvider'):
                if p in avail:
                    providers.append(p)
        providers.append('CPUExecutionProvider')
        try:
            self.session = ort.InferenceSession(str(self.rec_dir / 'rec.onnx'), opts, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(self.rec_dir / 'rec.onnx'), opts, providers=['CPUExecutionProvider'])
        p0 = self.session.get_providers()[0]
        self.device = {
            'CUDAExecutionProvider': 'GPU (CUDA)',
            'DmlExecutionProvider': 'GPU (DirectML)',
            'CoreMLExecutionProvider': 'GPU (CoreML)',
        }.get(p0, 'CPU')
        self.input_name = self.session.get_inputs()[0].name

        chars = self.codec.chars
        self._lao = np.array([('຀' <= c <= '໿') for c in chars])
        self._thai = np.array([('฀' <= c <= '๿') for c in chars])
        self._latin = np.array([c.isascii() and c.isalpha() for c in chars])
        self._block = {'auto': None, 'lao+en': self._thai, 'th+en': self._lao, 'lao': self._thai | self._latin,
                       'th': self._lao | self._latin, 'en': self._lao | self._thai}

        self.lm, self.lm_file = None, None
        try:
            from .lm_decode import CharLM, beam_decode
            f = Path(lm_file) if lm_file else _find_lm(self.root)
            if f and f.is_file():
                self.lm, self.lm_file, self._beam = CharLM(f), f, beam_decode
        except Exception as exc:
            print('Lao word helper not loaded:', exc)

        self.detector = None
        if use_detector:
            try:
                from . import detlayout
                dets = sorted(self.root.glob('models/OpenThai2.*-OCR-Text-*/det.onnx'))
                if dets:
                    detlayout.DET_DIR = dets[-1].parent
                if detlayout.available():
                    self.detector = detlayout
                else:
                    print('text-line detector not used:', detlayout._ERR)
            except Exception as exc:
                print('text-line detector not used:', exc)

    # ------------------------------------------------------------------ one line
    def _logits(self, gray: np.ndarray) -> np.ndarray:
        y = self.session.run(None, {self.input_name: to_tensor(gray)})[0][0]
        if not (y.min() >= 0 and abs(float(y.sum(-1).mean()) - 1) < 0.01):
            e = np.exp(y - y.max(-1, keepdims=True))
            y = e / e.sum(-1, keepdims=True)
        return y

    def read_line(self, gray: np.ndarray, lang: str = 'auto', use_lm: bool = True) -> str:
        p = self._logits(gray)
        block = self._block.get(lang)
        if block is not None:
            p = p.copy()
            p[:, block] = 0.0
        text = self.codec.decode(p.argmax(-1).tolist())
        n_lao = sum('຀' <= c <= '໿' for c in text)
        n_thai = sum('฀' <= c <= '๿' for c in text)
        if use_lm and self.lm is not None and n_lao and n_lao >= n_thai:
            q = p.copy()
            q[:, self._thai] = 0.0
            return self._beam(q, self.codec.chars, self.lm, normalize=lambda t: t.replace('ໍາ', 'ຳ').strip())
        if block is None and n_lao and n_thai:   # mixed Lao/Thai: re-read in the majority script
            q = p.copy()
            q[:, self._thai if n_lao >= n_thai else self._lao] = 0.0
            text = self.codec.decode(q.argmax(-1).tolist())
        return text.replace('ໍາ', 'ຳ').strip()

    # ------------------------------------------------------------------ page
    @staticmethod
    def load_image(image) -> np.ndarray:
        """path / PIL image / numpy (RGB or grey) -> RGB numpy"""
        if isinstance(image, (str, os.PathLike)):
            img = cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f'cannot open image: {image}')
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not isinstance(image, np.ndarray):
            image = np.asarray(image.convert('RGB'))
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return np.ascontiguousarray(image)

    def read(self, image, lang: str = 'auto', use_lm: bool = True, rectify: bool = True) -> OCRResult:
        from .bill_checker import check_bill
        from .markdown_fmt import format_markdown

        t0 = time.time()
        img = self.load_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        res = None
        if self.detector is not None and not is_single_line(gray if gray.mean() >= 127 else 255 - gray):
            try:
                rows_b, info = self.detector.read_rows_with_boxes(img, rectify=rectify)
            except Exception as exc:
                print('detector failed, using projection cutting:', exc)
                rows_b = None
            if rows_b:
                lines, rows_cells = [], []
                for items in rows_b:
                    texts, cell_pairs = [], []
                    for c, box in items:
                        t = self.read_line(c, lang, use_lm)
                        if t and not is_junk_line(t):
                            texts.append(t)
                            cell_pairs.append((t, box))
                    if texts:
                        lines.append('   '.join(texts))
                        rows_cells.append(cell_pairs)
                lh = info['line_height']
                rect_img = info.get('rectified_img')
                pw = rect_img.shape[1] if rect_img is not None else img.shape[1]
                plain_text = '\n'.join(lines)
                md_text = format_markdown(rows_cells, page_w=pw, line_height=lh) if rows_cells else plain_text
                res = OCRResult(plain_text, lines, 'detector', lh, info['skew'],
                                'low resolution: lines are only ~%.0f px tall' % lh if lh and lh < 14 else None,
                                rectify_summary=info.get('rectify_summary', ''),
                                markdown=md_text,
                                bill=check_bill(plain_text))
        if res is None:
            page = segment_page(img, rectify=rectify)
            lines = []
            for line in page.lines:
                if line.image.shape[1] * 48 / line.image.shape[0] <= 4096:
                    t = self.read_line(line.image, lang, use_lm)
                else:
                    t = '   '.join(self.read_line(s, lang, use_lm) for s in line.segments)
                if t and not is_junk_line(t):
                    lines.append(t)
            if not page.lines:
                lines = [self.read_line(gray, lang, use_lm)]
            plain_text = '\n'.join(lines)
            res = OCRResult(plain_text, lines, 'single line' if len(page.lines) <= 1 else 'projection',
                            page.char_height, page.skew, page.warning,
                            rectify_summary=page.rectify_summary,
                            markdown=plain_text,
                            bill=check_bill(plain_text))
        res.ms = (time.time() - t0) * 1000
        return res

    def read_file(self, path, lang: str = 'auto', use_lm: bool = True, dpi: int = 200, rectify: bool = True):
        """yields (label, OCRResult) for an image file or every page of a PDF (needs pymupdf)"""
        path = Path(path)
        if path.suffix.lower() == '.pdf':
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz  # older PyMuPDF
            doc = fitz.open(str(path))
            for i, page in enumerate(doc, 1):
                pix = page.get_pixmap(dpi=dpi)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                yield f'{path.name} p{i}', self.read(np.ascontiguousarray(arr), lang, use_lm, rectify=rectify)
        else:
            yield path.name, self.read(path, lang, use_lm, rectify=rectify)

    def info(self) -> dict:
        return {'recognizer': str(self.rec_dir), 'device': self.device,
                'language_model': str(self.lm_file) if self.lm_file else None,
                'line_detector': str(self.detector.DET_DIR) if self.detector else None,
                'characters': len(self.codec.dictionary)}
