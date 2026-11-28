from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import gradio as gr

from lao_ocr.core import Codec
from lao_ocr.layout import is_junk_line, segment_page, to_tensor
from lao_ocr.bill_checker import check_bill
try:
    from lao_ocr.thai_fusion import OpenThai22Rec, fuse_thai_reading
    OT22_THAI = OpenThai22Rec()
    print(f"Thai Dual-Fusion: OpenThai 2.2 loaded ({OT22_THAI.model_dir.name})")
except Exception as _ot_exc:
    OT22_THAI = None
    fuse_thai_reading = None
    print("Thai Dual-Fusion not loaded:", _ot_exc)

# Dedicated Thai Recognizer (Thai R3 All-1256-Fonts 99.5% -> R2 fallback)
THAI_MODEL_CANDIDATES = [
    ROOT / "runs" / "gpu-thai-r3-allfonts-skew-receipt" / "onnx",
    ROOT / "runs" / "gpu-thai-print-r2-realworld-tipitaka" / "onnx",
    ROOT / "runs" / "gpu-thai-print-01" / "onnx",
]
THAI_SESSION = None
THAI_CODEC = None
THAI_MODEL_NAME = None
for _td in THAI_MODEL_CANDIDATES:
    if (_td / "rec.onnx").is_file() and (_td / "rec_dict.txt").is_file():
        try:
            _topt = ort.SessionOptions()
            _topt.intra_op_num_threads = 4
            _topt.log_severity_level = 3
            THAI_SESSION = ort.InferenceSession(str(_td / "rec.onnx"), _topt, providers=["CPUExecutionProvider"])
            THAI_CODEC = Codec((_td / "rec_dict.txt").read_text(encoding="utf-8").splitlines())
            THAI_MODEL_NAME = _td.parent.name
            print(f"Dedicated Thai Model: {_td} ({THAI_MODEL_NAME})")
            break
        except Exception as _texc:
            print("Dedicated Thai model failed:", _texc)

ROOT = Path(__file__).resolve().parent
# Prefer newly trained Tri-lingual model (Lao + Thai + English)
# First existing model wins: newest training first.
MODEL_CANDIDATES = [
    ROOT / "runs" / "gpu-lao-r18-allfonts-5000" / "onnx",       # Round 18: ALL 343 Lao Fonts Equal-Quota (5,000+ / Font)
    ROOT / "runs" / "gpu-lao-r17-skew-screen" / "onnx",         # Round 17: Extreme Skew Camera + Simulated Screen-Photo
    ROOT / "runs" / "gpu-lao-r16-overnight-subvowel" / "onnx",  # Round 16: Extreme Sub-Vowel (ຸ ູ ຼ) & Tall-Mark Mastery
    ROOT / "runs" / "gpu-lao-r15-warped-a4" / "onnx",        # Round 15: All-Conditions Real-World + Warped/Wavy + Tight/Wide Spacing
    ROOT / "runs" / "gpu-lao-r14-tallmarks-khoso" / "onnx",  # Round 14: Tall Stacked Marks + ຂ/ຊ + Worn Fonts
    ROOT / "runs" / "gpu-lao-r13-phetsarath-subvowel" / "onnx", # Round 13: Phetsarath OT + Sub-Vowel Precision
    ROOT / "runs" / "gpu-lao-r12-receipts-01" / "onnx",      # Round 12: Degraded Lao Receipts & Stamped Docs
    ROOT / "runs" / "gpu-lao-r11-handwriting-01" / "onnx",   # Round 11: + user strokes + Kaggle handwriting
    ROOT / "runs" / "gpu-lao-r10-books-01" / "onnx",  # Round 10: lao-online Lao books + MoES + Wiki
    ROOT / "runs" / "gpu-lao-r9-vocab-01" / "onnx",   # Round 9: MoES textbooks + clean Wiki + rich Lao vocab
    ROOT / "runs" / "gpu-lao-photo-01" / "onnx",      # Round 8: phone photos, low-res, tilted/curved pages
    ROOT / "runs" / "gpu-lao-small-pdf-01" / "onnx",  # + real lines from Lao PDFs (laws, education, health)
    ROOT / "runs" / "gpu-lao-small-01" / "onnx",   # PP-OCRv6 small (5x bigger, transformer neck) - big Lao run
    ROOT / "runs" / "gpu-lao-focus-01" / "onnx",   # Lao-focused: domain text, place names, terms
    ROOT / "runs" / "gpu-styles-ft-01" / "onnx",   # + dirty/aged paper, worn ink, book bend (fine-tune)
    ROOT / "runs" / "gpu-full-a4-01" / "onnx",     # full run with simulated A4 pages + font catalog
    ROOT / "runs" / "gpu-lao-en-th-01" / "onnx",
    ROOT / "runs" / "gpu-lao-en-01" / "onnx",
]
# fine-tunes on your own corrected lines (finetune_feedback.bat) come first, newest first
MODEL_CANDIDATES = sorted(list(ROOT.glob("runs/feedback-*/onnx")) + list(ROOT.glob("runs/handwriting-*/onnx")),
                          key=lambda d: d.parent.stat().st_mtime, reverse=True) + MODEL_CANDIDATES
# choices made in the control panel (control_panel.py -> app_settings.json) win
import json as _json
SETTINGS = {}
try:
    SETTINGS = _json.loads((ROOT / "app_settings.json").read_text(encoding="utf-8"))
except Exception:
    pass
if SETTINGS.get("model_dir"):
    MODEL_CANDIDATES.insert(0, ROOT / SETTINGS["model_dir"])
MODEL_DIR = next((d for d in MODEL_CANDIDATES if (d / "rec.onnx").is_file()), MODEL_CANDIDATES[-1])
print(f"Model: {MODEL_DIR}")
SAMPLES_DIR = ROOT / "demo_samples"

# Initialize ONNX session
if not (MODEL_DIR / "rec.onnx").is_file():
    raise FileNotFoundError(f"Model not found at {MODEL_DIR / 'rec.onnx'}")

codec = Codec((MODEL_DIR / "rec_dict.txt").read_text(encoding="utf-8").splitlines())
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.log_severity_level = 3

# Auto-detect GPU for ONNX Runtime: CUDA (NVIDIA) -> DirectML (Windows NVIDIA/AMD/Intel) -> CoreML (Mac) -> CPU
_avail = ort.get_available_providers()
_ort_providers = []
if os.environ.get("OCR_CPU") != "1":
    for _p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"):
        if _p in _avail:
            _ort_providers.append(_p)
_ort_providers.append("CPUExecutionProvider")
try:
    session = ort.InferenceSession(str(MODEL_DIR / "rec.onnx"), opts, providers=_ort_providers)
except Exception:
    session = ort.InferenceSession(str(MODEL_DIR / "rec.onnx"), opts, providers=["CPUExecutionProvider"])

_active_ort_provider = session.get_providers()[0]

# Same model on the GPU through PyTorch when available (fastest fp16 on NVIDIA GPUs);
# ONNX Runtime (CUDA / DirectML / CoreML / CPU) stays as the automatic fallback.
TORCH_MODEL = None
ACTIVE_DEVICE = {
    "CUDAExecutionProvider": "GPU (ONNX CUDA)",
    "DmlExecutionProvider": "GPU (ONNX DirectML)",
    "CoreMLExecutionProvider": "GPU (Apple CoreML)",
}.get(_active_ort_provider, "CPU (ONNX)")

if os.environ.get("OCR_CPU") != "1":
    try:
        import torch
        from lao_ocr.core import load_checkpoint
        _ckpt = MODEL_DIR.parent / "best.pt"
        if torch.cuda.is_available() and _ckpt.is_file():
            _m, _codec, _ = load_checkpoint(_ckpt)
            if _codec.dictionary != codec.dictionary:
                raise ValueError("best.pt and rec_dict.txt disagree")
            TORCH_MODEL = _m.cuda().eval()
            ACTIVE_DEVICE = f"GPU ({torch.cuda.get_device_name(0)})"
    except Exception as exc:
        print("PyTorch GPU not used, falling back to ONNX:", exc)
        TORCH_MODEL = None
print(f"Recognizer device: {ACTIVE_DEVICE}")


def _infer(x: np.ndarray) -> np.ndarray:
    """(1, 3, 48, W) float32 -> (1, T, classes) probabilities."""
    if TORCH_MODEL is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            y = TORCH_MODEL(torch.from_numpy(x).cuda())
        return y.float().softmax(-1).cpu().numpy()
    return session.run(None, {"x": x})[0]


def preprocess_line_crop(crop: np.ndarray, max_width: int = 1600) -> np.ndarray:
    """Preprocess single line crop to [1, 3, 48, W]."""
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)

    h, w = crop.shape[:2]
    new_w = max(int(round(w * (48.0 / h))), 16)
    if new_w > max_width:
        new_w = max_width
    new_w = max(int(round(new_w / 8.0)) * 8, 8)

    resized = cv2.resize(crop, (new_w, 48), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 127.5 - 1.0
    tensor = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(tensor, 0)


def detect_and_split_lines(img: np.ndarray) -> list[np.ndarray]:
    """Segment a multi-line document or paragraph into individual line strips."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape

    # Adaptive / Otsu thresholding to find dark text on light background
    # Check if inverted (white text on dark bg)
    if np.mean(gray) < 127:
        gray = 255 - gray

    # Binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection
    proj = np.sum(thresh, axis=1) / 255.0
    min_text_px = max(2, w * 0.01)

    in_line = False
    raw_bands = []
    start = 0

    for y, val in enumerate(proj):
        if val > min_text_px and not in_line:
            in_line = True
            start = max(0, y - 4)
        elif val <= min_text_px and in_line:
            in_line = False
            end = min(h, y + 4)
            if (end - start) >= 8:
                raw_bands.append((start, end))

    if in_line:
        end = min(h, len(proj))
        if (end - start) >= 8:
            raw_bands.append((start, end))

    # Merge bands that are too close (e.g. tone marks or lower vowels separated by small gap)
    merged_bands = []
    for band in raw_bands:
        if not merged_bands:
            merged_bands.append(band)
        else:
            prev_start, prev_end = merged_bands[-1]
            if band[0] - prev_end <= 4:
                merged_bands[-1] = (prev_start, band[1])
            else:
                merged_bands.append(band)

    # Extract crop images
    line_crops = []
    for y1, y2 in merged_bands:
        # Trim left/right margins using horizontal projection of the crop
        crop = img[y1:y2, :]
        line_crops.append(crop)

    return line_crops


# ---------------------------------------------------------------- language mode
# One model reads Lao + Thai + English. Lao and Thai letters look alike, so the user can
# lock the language: characters of the other scripts are then switched off at decoding
# (digits, punctuation and spaces always stay allowed).
_IS_LAO = np.array([("\u0e80" <= c <= "\u0eff") for c in codec.chars])
_IS_THAI = np.array([("\u0e00" <= c <= "\u0e7f") for c in codec.chars])
_IS_LATIN = np.array([c.isascii() and c.isalpha() for c in codec.chars])

LANG_AUTO = "อัตโนมัติ (ລາວ / ไทย / English)"
LANG_MODES = {
    LANG_AUTO: None,
    "ລາວ + English": _IS_THAI,
    "ไทย + English": _IS_LAO,
    "ລາວ ເທົ່ານັ້ນ (Lao only)": _IS_THAI | _IS_LATIN,
    "ไทยอย่างเดียว (Thai only)": _IS_LAO | _IS_LATIN,
    "English only": _IS_LAO | _IS_THAI,
}


# Dictionary-assisted decoding (Lao character language model, built from datasets/lao-text):
# look-alike letters and lost tone marks are resolved by which spelling forms real Lao words.
try:
    from lao_ocr.lm_decode import LM_FILE, CharLM, beam_decode
    if SETTINGS.get("lm_file") and (ROOT / SETTINGS["lm_file"]).is_file():
        LM_FILE = ROOT / SETTINGS["lm_file"]
    print(f"Lao language model: {LM_FILE}")
    LAO_LM = CharLM(LM_FILE) if LM_FILE.is_file() else None
except Exception as exc:  # the app still works without it
    print("Lao language model not loaded:", exc)
    LAO_LM = None


def _probs(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0 and abs(float(logits.sum(-1).mean()) - 1) < 0.01:
        return logits
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _recognize(gray_crop: np.ndarray, lang: str = LANG_AUTO, use_lm: bool = True) -> str:
    # Use dedicated Thai R3 model when user selects Thai mode (ไทย + English / ไทยอย่างเดียว)
    if THAI_SESSION is not None and THAI_CODEC is not None and lang in ("ไทย + English", "ไทยอย่างเดียว (Thai only)"):
        t_logits = THAI_SESSION.run(None, {THAI_SESSION.get_inputs()[0].name: to_tensor(gray_crop)})[0][0]
        t_probs = _probs(t_logits)
        if lang == "ไทยอย่างเดียว (Thai only)":
            t_probs = t_probs.copy()
            _t_block = np.array([("\u0e80" <= c <= "\u0eff") or (c.isascii() and c.isalpha()) for c in THAI_CODEC.chars])
            t_probs[:, _t_block] = 0.0
        else:
            t_probs = t_probs.copy()
            _t_block = np.array([("\u0e80" <= c <= "\u0eff") for c in THAI_CODEC.chars])
            t_probs[:, _t_block] = 0.0
        idxs = t_probs.argmax(-1).tolist()
        text = THAI_CODEC.decode(idxs).strip()
        if OT22_THAI is not None and fuse_thai_reading is not None and text:
            confs, prev = [], None
            for ti, idx in enumerate(idxs):
                if idx and idx != prev:
                    confs.append(float(t_probs[ti, idx]))
                prev = idx
            mean_conf = float(np.mean(confs)) if confs else 0.0
            text, _, _ = fuse_thai_reading(text, mean_conf, gray_crop, OT22_THAI, fallback_below=0.92)
        return text.strip()

    logits = _infer(to_tensor(gray_crop))[0]
    block = LANG_MODES.get(lang)
    if block is not None:
        logits = logits.copy()
        logits[:, block] = -1e9
    text = codec.decode(logits.argmax(-1).tolist())
    if use_lm and LAO_LM is not None:
        n_lao = sum("\u0e80" <= c <= "\u0eff" for c in text)
        n_thai = sum("\u0e00" <= c <= "\u0e7f" for c in text)
        if n_lao and n_lao >= n_thai:  # the model was trained on Lao text only
            p = _probs(logits)
            if block is None:
                p = p.copy()
                p[:, _IS_THAI] = 0.0
            return beam_decode(p, codec.chars, LAO_LM, normalize=lambda t: t.replace("ໍ\u0eb2", "ຳ").strip())
    if block is None:  # auto: a line that comes out with BOTH Lao and Thai is re-read in its majority script
        n_lao = sum("\u0e80" <= c <= "\u0eff" for c in text)
        n_thai = sum("\u0e00" <= c <= "\u0e7f" for c in text)
        if n_lao and n_thai:
            masked = logits.copy()
            masked[:, _IS_THAI if n_lao >= n_thai else _IS_LAO] = -1e9
            text = codec.decode(masked.argmax(-1).tolist())
    # Dual-Recognizer Thai Fusion: if line is Thai and confidence < 0.92, let OpenThai 2.2 verify!
    if OT22_THAI is not None and fuse_thai_reading is not None:
        n_lao = sum("\u0e80" <= c <= "\u0eff" for c in text)
        n_thai = sum("\u0e00" <= c <= "\u0e7f" for c in text)
        if n_thai > 0 and n_thai >= n_lao:
            p = _probs(logits)
            idxs = p.argmax(-1).tolist()
            confs, prev = [], None
            for t_i, idx in enumerate(idxs):
                if idx and idx != prev:
                    confs.append(float(p[t_i, idx]))
                prev = idx
            mean_conf = float(np.mean(confs)) if confs else 0.0
            text, _, _ = fuse_thai_reading(text, mean_conf, gray_crop, OT22_THAI, fallback_below=0.92)
    return text.replace("ໍ\u0eb2", "ຳ").strip()


# ---------------------------------------------------------------- teaching data from real documents
FB_DIR = ROOT / "datasets" / "user-feedback"   # pending/ -> (checked) images/ + train.jsonl, skipped/


def _queue_lines(lines, texts):
    """Keep the line images of a real document so they can be checked and used for training."""
    pend = FB_DIR / "pending"
    pend.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    for k, (img, text) in enumerate(zip(lines, texts)):
        if img is None or img.size == 0 or not text:
            continue
        name = f"{stamp}_{k:03d}"
        cv2.imwrite(str(pend / f"{name}.png"), img)
        (pend / f"{name}.txt").write_text(text, encoding="utf-8")


_KEPT_PAGES: set = set()


def _new_page(img) -> bool:
    """True the first time a page is seen (re-reading the same page must not queue its lines twice)."""
    import hashlib
    key = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
    if key in _KEPT_PAGES:
        return False
    _KEPT_PAGES.add(key)
    return True


# ---------------------------------------------------------------- text-line detector
LAYOUT = SETTINGS.get("layout", "detector")  # "detector" (learned, default) or "projection" (lao_ocr/layout.py)
try:
    from lao_ocr import detlayout
    DETECTOR_OK = LAYOUT == "detector" and detlayout.available()
except Exception as _exc:
    detlayout, DETECTOR_OK = None, False
print("Line finder:", "text detector (PP-OCRv6 DB)" if DETECTOR_OK else
      f"projection cutting ({getattr(detlayout, '_ERR', None) or LAYOUT})")


def _detector_rows(img, rectify: bool = True, clean_stamps: bool = True):
    """rows of (crop, quad_box) from the learned detector, or None to use projection cutting"""
    if not DETECTOR_OK:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    from lao_ocr.layout import is_single_line
    if is_single_line(gray if gray.mean() >= 127 else 255 - gray):
        return None  # a single cropped line: read it as it is
    try:
        rows, info = detlayout.read_rows_with_boxes(img, rectify=rectify, clean_stamps=clean_stamps)
    except Exception as exc:
        print("detector failed, using projection cutting:", exc)
        return None
    return (rows, info) if rows else None


def run_ocr(input_image, lang=LANG_AUTO, use_lm=True, keep_lines=False, as_markdown=False, rectify=True,
            clean_stamps=True, read_watermark=False):
    """Single-pass unified OCR: returns (plain_text, md_text, bill_html, lat_str, char_count, res_str, rectified_img)."""
    if input_image is None:
        return (
            "กรุณาอัปโหลดหรือเลือกรูปภาพตัวอย่างก่อนครับ",
            "",
            check_bill("").to_html(),
            "0.0 ms",
            "0 ตัวอักษร",
            "-",
            None,
        )

    t0 = time.time()
    h, w = input_image.shape[:2]
    res_str = f"{w} × {h} px"

    # Works for both a single line crop and a whole page: the page is rectified,
    # filtered (stamps/watermarks/icons), split into multi-page columns if needed,
    # and each line goes to the recognizer ONCE.
    det_rows = _detector_rows(input_image, rectify=rectify, clean_stamps=clean_stamps)
    if det_rows is not None:  # learned text detector: photos, tilt, columns, tables
        from lao_ocr.doc_filters import format_blocks_text, read_watermark_layer
        rows, info = det_rows
        recognized_lines, rows_cells, kept_blocks, keep = [], [], [], []
        raw_blocks = info.get("row_blocks") or [0] * len(rows)
        for items, bi in zip(rows, raw_blocks):
            texts, cell_pairs = [], []
            for c, box in items:
                t = _recognize(c, lang, use_lm)
                keep.append((c, t))
                if t and not is_junk_line(t):
                    texts.append(t)
                    cell_pairs.append((t, box))
            if texts:
                recognized_lines.append("   ".join(texts))
                rows_cells.append(cell_pairs)
                kept_blocks.append(bi)
        if keep_lines and keep and _new_page(input_image):
            try:
                _queue_lines([k[0] for k in keep], [k[1] for k in keep])
            except Exception as exc:
                print("could not keep lines:", exc)
        lh = info["line_height"]
        plain_text = format_blocks_text(rows_cells, kept_blocks, line_height=lh) if rows_cells else "\n".join(recognized_lines)
        if read_watermark and info.get("watermark_mask") is not None:
            wm_lines = read_watermark_layer(
                info["watermark_mask"],
                lambda c: _recognize(c, lang, use_lm),
                detect_fn=detlayout._engine().detect,
            )
            if wm_lines:
                plain_text += "\n\n[ลายน้ำ / ตราประทับที่กรองออก]\n" + "\n".join(wm_lines)
        from lao_ocr.markdown_fmt import format_markdown
        rect_img = info.get("rectified_img")
        rw = rect_img.shape[1] if rect_img is not None else w
        md_text = format_markdown(rows_cells, page_w=rw, line_height=lh) if rows_cells else plain_text
        bill_html = check_bill(plain_text).to_html()
        display_text = md_text if as_markdown else plain_text
        rect_tag = f" | 📐 {info.get('rectify_summary', '')}" if rectify and info.get("rectify_summary") else ""
        n_blk = info.get("num_blocks", 1)
        blk_tag = f" | 📄 แยก {n_blk} คอลัมน์/หน้า" if n_blk > 1 else ""
        mode_str = f"เอกสาร ({len(rows)} แถว, {info['boxes']} กล่องข้อความ){blk_tag} | ตัวตรวจจับข้อความ | สูงบรรทัด ~{lh:.0f}px{rect_tag}"
        char_count = f"{len(plain_text)} ตัวอักษร ({mode_str})"
        if lh and lh < 14:
            char_count = "⚠️ ภาพความละเอียดต่ำ: บรรทัดสูงเพียง ~%.0f px — ถ้าอ่านผิดมาก ให้ใช้ภาพที่ละเอียดกว่า\n" % lh + char_count
        return display_text, md_text, bill_html, f"{(time.time() - t0) * 1000:.1f} ms", char_count, res_str, info.get("rectified_img", input_image)

    page = segment_page(input_image, rectify=rectify)
    recognized_lines, keep = [], []
    for line in page.lines:
        if line.image.shape[1] * 48 / line.image.shape[0] <= 4096:
            text_line = _recognize(line.image, lang, use_lm)
            keep.append((line.image, text_line))
        else:  # extremely long line: read each column/segment separately
            seg_texts = [_recognize(seg, lang, use_lm) for seg in line.segments]
            keep.extend(zip(line.segments, seg_texts))
            text_line = "   ".join(seg_texts)
        if text_line and not is_junk_line(text_line):  # '-------', '.......': dotted leaders / ruled lines
            recognized_lines.append(text_line)
    if keep_lines and keep and _new_page(input_image):
        try:
            _queue_lines([k[0] for k in keep], [k[1] for k in keep])
        except Exception as exc:
            print("could not keep lines:", exc)
    if not page.lines:
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) if input_image.ndim == 3 else input_image
        recognized_lines = [_recognize(gray, lang, use_lm)]

    text = "\n".join(recognized_lines)
    bill_html = check_bill(text).to_html()
    n = len(page.lines)
    mode_str = "บรรทัดเดี่ยว (Single Line)" if n <= 1 else f"เอกสาร ({n} บรรทัด)"
    mode_str += f" | ตัวอักษรสูง ~{page.char_height:.0f}px"
    if rectify and page.rectify_summary:
        mode_str += f" | 📐 {page.rectify_summary}"

    latency_ms = (time.time() - t0) * 1000
    char_count = f"{len(text)} ตัวอักษร ({mode_str})"
    if page.warning:
        char_count = "⚠️ " + page.warning + "\n" + char_count
    lat_str = f"{latency_ms:.1f} ms"

    return text, text, bill_html, lat_str, char_count, res_str, (page.rectified_img if page.rectified_img is not None else input_image)


# ---------------------------------------------------------------- many pages / PDF
def _parse_page_range(raw: str, total: int) -> list[int] | None:
    s = (raw or "").strip()
    if not s:
        return None
    pages = set()
    for part in _re.split(r"[,;\s]+", s):
        if not part:
            continue
        m = _re.match(r"^(\d+)(?:-(\d+))?$", part)
        if not m:
            continue
        a = max(1, min(total, int(m.group(1))))
        b = max(1, min(total, int(m.group(2)))) if m.group(2) else a
        if a > b:
            a, b = b, a
        for p in range(a, b + 1):
            pages.add(p)
    return sorted(pages) if pages else None


def _pages_from_files(paths, page_range: str = ""):
    """Yield (label, RGB image) for every image file and selected pages of every PDF."""
    for path in paths:
        path = str(getattr(path, "name", path))
        name = Path(path).name
        if path.lower().endswith(".pdf"):
            try:
                import pymupdf as fitz  # PyMuPDF >= 1.24
            except ImportError:
                try:
                    import fitz  # older PyMuPDF
                except ImportError:
                    fitz = None
            if fitz is None:
                yield name, "PDF ต้องติดตั้ง PyMuPDF ก่อน: ดับเบิลคลิก install_pdf_support.bat"
                continue
            doc = fitz.open(path)
            sel = _parse_page_range(page_range, len(doc)) or list(range(1, len(doc) + 1))
            for i in sel:
                page = doc[i - 1]
                pix = page.get_pixmap(dpi=200)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                yield f"{name} หน้า {i}/{len(doc)}", np.ascontiguousarray(arr[:, :, :3])
        else:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                yield name, "เปิดไฟล์ภาพไม่ได้"
                continue
            yield name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_batch(files, lang=LANG_AUTO, use_lm=True, keep_lines=False, page_range="", rectify=True, progress=gr.Progress()):
    if not files:
        return "กรุณาเลือกไฟล์ภาพหรือ PDF ก่อน (เลือกได้หลายไฟล์พร้อมกัน)", None, "-"
    t0 = time.time()
    parts, n_pages, warnings = [], 0, []
    pages = list(_pages_from_files(files, page_range=page_range))
    for k, (label, img) in enumerate(pages, 1):
        progress((k - 1) / max(1, len(pages)), desc=f"กำลังอ่าน {label} ({k}/{len(pages)})")
        if isinstance(img, str):
            parts.append(f"===== {label} =====\n[{img}]")
            continue
        text, _, _, _, info, _, _ = run_ocr(img, lang, use_lm, keep_lines, rectify=rectify)
        if info.startswith("⚠️"):
            warnings.append(label)
        parts.append(f"===== {label} =====\n{text}")
        n_pages += 1
    full = "\n\n".join(parts)
    out_dir = ROOT / "ocr_results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"ocr_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    out_file.write_text(full, encoding="utf-8")
    dt = time.time() - t0
    info = f"{n_pages} หน้า ใน {dt:.1f} วินาที ({dt / max(1, n_pages):.1f} วินาที/หน้า)"
    if warnings:
        info += f" | ⚠️ ความละเอียดต่ำ: {', '.join(warnings[:5])}"
    return full, str(out_file), info


def _fb_count():
    f = FB_DIR / "train.jsonl"
    return sum(1 for _ in open(f, encoding="utf-8")) if f.is_file() else 0


def fb_next():
    pend = sorted((FB_DIR / "pending").glob("*.png")) if (FB_DIR / "pending").is_dir() else []
    if not pend:
        return None, "", "", (f"ไม่มีบรรทัดรอตรวจ | ตรวจแล้วทั้งหมด {_fb_count()} บรรทัด  "
                              "(ติ๊ก 💾 แล้วอ่านเอกสารจริง เพื่อเก็บบรรทัดมาตรวจ)")
    p = pend[0]
    t = p.with_suffix(".txt")
    text = t.read_text(encoding="utf-8") if t.is_file() else ""
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img, text, p.stem, f"รอตรวจ {len(pend)} บรรทัด | ตรวจแล้ว {_fb_count()} บรรทัด"


def fb_save(item, text):
    import json
    if item:
        text = " ".join((text or "").replace("ໍ\u0eb2", "ຳ").split())
        bad = sorted(set(text) - set(codec.ids) - {" "})
        if not text:
            return fb_skip(item)
        if bad:
            img, _, _, _ = fb_next()
            return img, text, item, "⚠️ มีตัวอักษรที่โมเดลไม่รู้จัก: " + " ".join(bad) + " — แก้แล้วกดบันทึกอีกครั้ง"
        src = FB_DIR / "pending" / f"{item}.png"
        if src.is_file():
            (FB_DIR / "images").mkdir(parents=True, exist_ok=True)
            src.replace(FB_DIR / "images" / src.name)
            src.with_suffix(".txt").unlink(missing_ok=True)
            with open(FB_DIR / "train.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"image": f"images/{src.name}", "text": text}, ensure_ascii=False) + "\n")
    return fb_next()


def fb_skip(item):
    src = FB_DIR / "pending" / f"{item}.png" if item else None
    if src is not None and src.is_file():
        (FB_DIR / "skipped").mkdir(parents=True, exist_ok=True)
        src.replace(FB_DIR / "skipped" / src.name)
        t = src.with_suffix(".txt")
        if t.is_file():
            t.replace(FB_DIR / "skipped" / t.name)
    return fb_next()


import html as _html
import re as _re

_WEB_FONTS = {
    "Phetsarath OT (ลาวมาตรฐาน)": "'Phetsarath OT', 'Saysettha OT', 'Leelawadee UI', sans-serif",
    "Saysettha OT": "'Saysettha OT', 'Phetsarath OT', sans-serif",
    "TH Sarabun New (ไทยมาตรฐาน)": "'TH Sarabun New', 'Leelawadee UI', sans-serif",
    "Leelawadee UI": "'Leelawadee UI', 'Noto Sans Thai', 'Noto Sans Lao', sans-serif",
    "Noto Sans Lao / Thai": "'Noto Sans Lao', 'Noto Sans Thai', sans-serif",
    "Tahoma": "'Tahoma', sans-serif",
}
_TOK_RE = _re.compile(r"([0-9໐-໙๐-๙]+(?:[.,:/\-][0-9໐-໙๐-๙]+)*)|([A-Za-z]+(?:['’\-][A-Za-z]+)*)")


def render_proofread_html(text: str, font_label: str, font_sz: int,
                          main_b: bool, main_i: bool,
                          num_b: bool, num_i: bool,
                          en_b: bool, en_i: bool) -> str:
    fam = _WEB_FONTS.get(font_label, _WEB_FONTS["Phetsarath OT (ลาวมาตรฐาน)"])
    if not text:
        return (
            f'<div style="font-family:{fam};font-size:{font_sz}px;color:#94a3b8;'
            f'border:1px solid #cbd5e1;border-radius:8px;padding:12px;min-height:110px;background:#fff;">'
            f'ผลลัพธ์ตรวจสี (ลาว/ไทย = ดำ • ตัวเลข = ส้ม • อังกฤษ = น้ำเงิน) จะแสดงที่นี่...</div>'
        )
    st_main = f"color:#0f172a;font-weight:{'700' if main_b else '400'};font-style:{'italic' if main_i else 'normal'};"
    st_num = f"color:#c2410c;font-weight:{'700' if num_b else '400'};font-style:{'italic' if num_i else 'normal'};"
    st_en = f"color:#1d4ed8;font-weight:{'700' if en_b else '400'};font-style:{'italic' if en_i else 'normal'};"
    parts = []
    last = 0
    for m in _TOK_RE.finditer(text):
        if m.start() > last:
            parts.append(f'<span style="{st_main}">{_html.escape(text[last:m.start()])}</span>')
        if m.group(1):
            parts.append(f'<span style="{st_num}">{_html.escape(m.group(1))}</span>')
        else:
            parts.append(f'<span style="{st_en}">{_html.escape(m.group(2))}</span>')
        last = m.end()
    if last < len(text):
        parts.append(f'<span style="{st_main}">{_html.escape(text[last:])}</span>')
    body = "".join(parts)
    return (
        f'<div style="font-family:{fam};font-size:{font_sz}px;line-height:1.75;'
        f'white-space:pre-wrap;word-break:break-word;border:1px solid #93c5fd;'
        f'border-radius:8px;padding:14px;min-height:140px;max-height:420px;overflow-y:auto;background:#fff;">'
        f'{body}</div>'
    )


def _export_txt_file(text: str):
    if not (text or "").strip():
        return None
    out_dir = ROOT / "ocr_results"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"MaeBounme_OCR_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    p.write_text("\ufeff" + text, encoding="utf-8")
    return str(p)


def _export_md_file(md_text: str):
    if not (md_text or "").strip():
        return None
    out_dir = ROOT / "ocr_results"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"MaeBounme_OCR_{time.strftime('%Y%m%d_%H%M%S')}.md"
    p.write_text(md_text, encoding="utf-8")
    return str(p)


def _export_word_file(text: str, font_label: str, font_sz: int,
                      main_b: bool, main_i: bool,
                      num_b: bool, num_i: bool,
                      en_b: bool, en_i: bool):
    if not (text or "").strip():
        return None
    fam = _WEB_FONTS.get(font_label, _WEB_FONTS["Phetsarath OT (ลาวมาตรฐาน)"])
    body_div = render_proofread_html(text, font_label, font_sz, main_b, main_i, num_b, num_i, en_b, en_i)
    doc_html = (
        '\ufeff<html xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:w="urn:schemas-microsoft-com:office:word" '
        'xmlns="http://www.w3.org/TR/REC-html40">'
        f'<head><meta charset="utf-8"><style>body {{ font-family: {fam}; font-size: {max(12, int(font_sz * 0.85))}pt; line-height: 1.5; }}</style></head>'
        f'<body>{body_div}</body></html>'
    )
    out_dir = ROOT / "ocr_results"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"MaeBounme_OCR_{time.strftime('%Y%m%d_%H%M%S')}.doc"
    p.write_text(doc_html, encoding="utf-8")
    return str(p)


# Collect sample images for gallery
sample_images = []
if SAMPLES_DIR.is_dir():
    for ext in ("*.png", "*.jpg"):
        sample_images.extend(sorted(str(p) for p in SAMPLES_DIR.glob(ext)))

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.header-box {
    text-align: center;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.result-box textarea {
    font-size: 19px !important;
    font-family: 'Phetsarath OT', 'Saysettha OT', 'Noto Sans Lao', sans-serif !important;
    line-height: 1.8 !important;
    background-color: #f8fafc !important;
}
"""

# Handwriting stroke pad (lao_ocr/strokes.py): served by this app on its own port, shown at the bottom of the page
STROKE_PORT = 7862
try:
    from lao_ocr import strokes as _strokes
    if _strokes.port_free(STROKE_PORT):
        _strokes.serve(STROKE_PORT, background=True)
    print(f"Stroke pad: http://127.0.0.1:{STROKE_PORT}")
except Exception as _exc:
    print("stroke pad not started:", _exc)

with gr.Blocks(title="Lao + Thai + English OCR Studio") as demo:
    gr.HTML(
        f"""
        <div class="header-box">
            <h1 style="margin:0; font-size:26px;">🇱🇦🇹🇭 MaeBounme Lao + Thai + English OCR Studio</h1>
            <p style="margin:6px 0 0; font-size:14px; opacity:0.95;">
                🕊️ ฟรี 100% Open-Source (Apache-2.0 • อุทิศส่วนกุศลแด่ <b>แม่บุญมี</b>) &nbsp;|&nbsp;
                โมเดล: <b>{MODEL_DIR.parent.name}</b> &nbsp;|&nbsp;
                <b>⚡ {ACTIVE_DEVICE}</b> &nbsp;|&nbsp;
                ตัวช่วยเดาคำ: <b>{'LM v6' if LAO_LM else 'Off'}</b>
            </p>
            <p style="margin:5px 0 0; font-size:12px; opacity:0.88;">
                ☕ สนับสนุนค่ากาแฟตามศรัทธา: <b>ໂກ້ (MaeBounme)</b> • FB: <b>Chert Victor Sign</b> • WhatsApp: <a href="https://wa.me/8562059592939" target="_blank" style="color:#fde68a;font-weight:700;">020 5959 2939</a>
            </p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("📄 อ่านภาพ (ง่าย / เร็ว)"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📥 วางภาพ (Ctrl+V) หรืออัปโหลดภาพเอกสาร/บรรทัด",
                        type="numpy",
                    )
                    lang_input = gr.Dropdown(
                        choices=list(LANG_MODES),
                        value=(SETTINGS.get("default_lang") if SETTINGS.get("default_lang") in LANG_MODES else LANG_AUTO),
                        label="🌐 ภาษาของเอกสาร / ພາສາຂອງເອກະສານ",
                    )
                    with gr.Row():
                        lm_input = gr.Checkbox(value=LAO_LM is not None, interactive=LAO_LM is not None,
                                               label="🧠 ตัวช่วยเดาคำ (LM v6)")
                        rectify_input = gr.Checkbox(value=True, label="📐 ปรับรูป & จัดแถวตรง")
                        clean_stamps_input = gr.Checkbox(value=True, label="🧹 กรองตราประทับ/ลายน้ำ/ไอคอน")
                    with gr.Row():
                        read_wm_input = gr.Checkbox(value=False, label="🔍 อ่านลายน้ำ 45° / ตราประทับ")
                        md_input = gr.Checkbox(value=False, label="📑 Markdown / ตาราง")
                        keep_input = gr.Checkbox(value=False, label="💾 เก็บไว้สอนโมเดล")
                    with gr.Row():
                        submit_btn = gr.Button("🚀 เริ่มสแกน OCR (กดเมื่อพร้อม)", variant="primary", size="lg")
                        clear_btn = gr.Button("🧹 ล้าง", size="lg")
                    with gr.Accordion("🖼️ ภาพหลังปรับตรง (Rectified Preview)", open=False):
                        rectified_preview = gr.Image(label="ภาพหลังแก้เอียง/แก้ภาพบิดก่อนส่งอ่าน", type="numpy", interactive=False)

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("📝 ข้อความ (Text)"):
                            text_output = gr.Textbox(
                                label="📝 ข้อความที่อ่านได้ (แก้ไขได้โดยตรง — แก้แล้วคำนวณบิลใหม่ให้อัตโนมัติ)",
                                elem_classes=["result-box"],
                                lines=14,
                                buttons=["copy"],
                                placeholder="อัปโหลดหรือวางภาพทางซ้าย เลือกภาษา แล้วกดปุ่ม '🚀 เริ่มสแกน OCR'...",
                            )
                        with gr.Tab("📑 Markdown & ตาราง"):
                            md_preview = gr.Markdown(value="_กดสแกนเพื่อดูตารางและหัวข้อแบบ Markdown_")
                            md_raw_output = gr.Textbox(
                                label="📋 โค้ด Markdown (คัดลอกได้)",
                                lines=8,
                                buttons=["copy"],
                            )
                        with gr.Tab("🧾 ตรวจบิล & คำนวณราคา"):
                            bill_check_html = gr.HTML(value=check_bill("").to_html())

                    with gr.Row():
                        export_txt_btn = gr.Button("📄 ดาวน์โหลด .txt", size="sm")
                        export_md_btn = gr.Button("📑 ดาวน์โหลด .md", size="sm")
                        export_word_btn = gr.Button("📘 ดาวน์โหลด Word (.doc)", variant="secondary", size="sm")
                    export_file_out = gr.File(label="📥 ไฟล์พร้อมดาวน์โหลด", visible=True)
                    with gr.Row():
                        speed_badge = gr.Textbox(label="⚡ ความเร็ว", value="-", interactive=False)
                        count_badge = gr.Textbox(label="📊 สรุป", value="-", interactive=False)
                        res_badge = gr.Textbox(label="📐 ขนาดภาพ", value="-", interactive=False)

                    with gr.Accordion("🎨 ตรวจทานสี & เลือกฟอนต์มาตรฐาน (แยกสีตัวเลข / อังกฤษ • หนา / เอียง)", open=False):
                        with gr.Row():
                            font_picker = gr.Dropdown(
                                choices=list(_WEB_FONTS.keys()),
                                value="Phetsarath OT (ลาวมาตรฐาน)",
                                label="🔤 ฟอนต์มาตรฐาน",
                            )
                            font_size_slider = gr.Slider(14, 30, value=19, step=1, label="ขนาดฟอนต์ (px)")
                        with gr.Row():
                            main_bold = gr.Checkbox(value=False, label="ลาว/ไทย หนา (B)")
                            main_italic = gr.Checkbox(value=False, label="ลาว/ไทย เอียง (I)")
                            num_bold = gr.Checkbox(value=True, label="ตัวเลข (ส้ม) หนา")
                            num_italic = gr.Checkbox(value=False, label="ตัวเลข เอียง")
                            en_bold = gr.Checkbox(value=False, label="อังกฤษ (น้ำเงิน) หนา")
                            en_italic = gr.Checkbox(value=True, label="อังกฤษ เอียง")
                        proofread_html = gr.HTML(value=render_proofread_html("", "Phetsarath OT (ลาวมาตรฐาน)", 19, False, False, True, False, False, True))

            if sample_images:
                with gr.Accordion("🖼️ ภาพตัวอย่างสำหรับทดสอบ (คลิกเพื่อลองทันที)", open=False):
                    gr.Examples(
                        examples=[[img] for img in sample_images],
                        inputs=[image_input],
                        outputs=[text_output, md_raw_output, bill_check_html, speed_badge, count_badge, res_badge, rectified_preview],
                        fn=lambda img: run_ocr(img, LANG_AUTO, True, False, False, True),
                        cache_examples=False,
                        examples_per_page=8,
                    )

        with gr.Tab("📚 อ่าน PDF / หลายไฟล์พร้อมกัน"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(label="เลือกไฟล์ PDF หรือภาพหลายไฟล์ (jpg, png, pdf)", file_count="multiple",
                                          file_types=["image", ".pdf"], type="filepath")
                    batch_page_range = gr.Textbox(
                        label="📑 ระบุหน้า PDF ที่ต้องการสแกน (เว้นว่าง = อ่านทุกหน้า)",
                        placeholder="เช่น 1-3, 5, 8-10",
                        value="",
                    )
                    batch_btn = gr.Button("📚 อ่านหน้าที่เลือก", variant="primary", size="lg")
                with gr.Column(scale=1):
                    batch_text = gr.Textbox(label="ข้อความทุกหน้า", lines=16, buttons=["copy"], elem_classes=["result-box"])
                    batch_info = gr.Textbox(label="สรุป", value="-", interactive=False)
                    batch_download = gr.File(label="ดาวน์โหลดผลเป็นไฟล์ .txt")
            batch_btn.click(fn=run_batch, inputs=[batch_files, lang_input, lm_input, keep_input, batch_page_range, rectify_input],
                            outputs=[batch_text, batch_download, batch_info])

        with gr.Tab("✍️ สอนโมเดล & กระดานลายมือ"):
            gr.Markdown("#### ✍️ ตรวจแก้บรรทัดจากเอกสารจริง (ข้อมูลสอนโมเดลรอบถัดไป)")
            with gr.Row():
                with gr.Column(scale=1):
                    fb_img = gr.Image(label="ภาพบรรทัด", type="numpy", interactive=False, height=140)
                with gr.Column(scale=1):
                    fb_text = gr.Textbox(label="ข้อความที่ถูกต้อง", lines=2, elem_classes=["result-box"])
                    fb_item = gr.State("")
                    with gr.Row():
                        fb_ok = gr.Button("✅ บันทึก + ถัดไป", variant="primary")
                        fb_skip_btn = gr.Button("⏭ ข้าม")
                        fb_load = gr.Button("🔄 โหลดบรรทัดรอตรวจ")
                    fb_info = gr.Textbox(label="สถานะ", value="-", interactive=False)
            fb_out = [fb_img, fb_text, fb_item, fb_info]
            fb_load.click(fn=fb_next, inputs=None, outputs=fb_out)
            fb_ok.click(fn=fb_save, inputs=[fb_item, fb_text], outputs=fb_out)
            fb_text.submit(fn=fb_save, inputs=[fb_item, fb_text], outputs=fb_out)
            fb_skip_btn.click(fn=fb_skip, inputs=[fb_item], outputs=fb_out)

            with gr.Accordion("🖊️ ກະດານຂຽນລາຍມື · กระดานวาดลายมือ", open=False):
                gr.HTML(
                    f'<div style="font-size:13px;margin-bottom:6px">เลือกตัวอักษรทางซ้าย วาดด้วยเมาส์/ปากกา/นิ้ว แล้วกด Enter — '
                    f'<a href="/pad/" target="_blank">เปิดเต็มจอในแท็บใหม่ ↗</a> · '
                    f'ใช้จากมือถือ/แท็บเล็ต: <code>stroke_pad_lan.bat</code></div>'
                    f'<iframe src="/pad/" style="width:100%;height:960px;border:1px solid #d7dce5;border-radius:10px"></iframe>'
                )

    # Manual Scan ONLY when user clicks '🚀 เริ่มสแกน OCR' — ONE pass fills Text + Markdown Table + Bill Calc!
    submit_btn.click(
        fn=run_ocr,
        inputs=[image_input, lang_input, lm_input, keep_input, md_input, rectify_input, clean_stamps_input, read_wm_input],
        outputs=[text_output, md_raw_output, bill_check_html, speed_badge, count_badge, res_badge, rectified_preview],
    )
    md_raw_output.change(fn=lambda m: m or "_ไม่มีข้อมูล Markdown_", inputs=[md_raw_output], outputs=[md_preview])
    export_txt_btn.click(fn=_export_txt_file, inputs=[text_output], outputs=[export_file_out])
    export_md_btn.click(fn=_export_md_file, inputs=[md_raw_output], outputs=[export_file_out])
    export_word_btn.click(
        fn=_export_word_file,
        inputs=[text_output, font_picker, font_size_slider, main_bold, main_italic, num_bold, num_italic, en_bold, en_italic],
        outputs=[export_file_out],
    )
    clear_btn.click(
        fn=lambda: (None, "", "", "_กดสแกนเพื่อดูตารางและหัวข้อแบบ Markdown_", check_bill("").to_html(), "-", "-", "-", None, None),
        inputs=None,
        outputs=[image_input, text_output, md_raw_output, md_preview, bill_check_html, speed_badge, count_badge, res_badge, export_file_out, rectified_preview],
    )

    _proof_inputs = [text_output, font_picker, font_size_slider, main_bold, main_italic, num_bold, num_italic, en_bold, en_italic]
    for _ctrl in _proof_inputs:
        _ctrl.change(fn=render_proofread_html, inputs=_proof_inputs, outputs=[proofread_html])
    text_output.change(fn=lambda txt: check_bill(txt or "").to_html(), inputs=[text_output], outputs=[bill_check_html])

if __name__ == "__main__":
    port = 7860
    print(f"Starting Lao + Thai + English OCR Studio on http://127.0.0.1:{port} ...")
    try:  # the stroke pad (:7862) is also served at /pad/ on this address, so it can be shown inside this page
        import asyncio
        import urllib.error
        import urllib.request
        import uvicorn
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import RedirectResponse
    except ImportError:
        demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False, css=CUSTOM_CSS)
    else:
        server = FastAPI()
        _direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        @server.get("/pad")
        def _pad_root():
            return RedirectResponse("/pad/")

        @server.api_route("/pad/{path:path}", methods=["GET", "POST"])
        async def _pad_proxy(path: str, request: Request):
            url = f"http://127.0.0.1:{STROKE_PORT}/{path}" + (("?" + request.url.query) if request.url.query else "")
            body = await request.body() if request.method == "POST" else None

            def fetch():
                req = urllib.request.Request(url, data=body, method=request.method,
                                             headers={"Content-Type": request.headers.get("content-type", "application/json")})
                try:
                    with _direct.open(req, timeout=120) as r:
                        return r.status, r.read(), r.headers.get("Content-Type", "application/octet-stream")
                except urllib.error.HTTPError as e:
                    return e.code, e.read(), e.headers.get("Content-Type", "application/json")
                except Exception as e:
                    return 502, f'{{"error": "stroke pad not running: {e}"}}'.encode(), "application/json"

            st, data, ct = await asyncio.to_thread(fetch)
            return Response(content=data, status_code=st, media_type=ct, headers={"Cache-Control": "no-store"})

        app = gr.mount_gradio_app(server, demo, path="/", css=CUSTOM_CSS)
        print(f"* Running on local URL:  http://127.0.0.1:{port}   (stroke pad: http://127.0.0.1:{port}/pad/)")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
