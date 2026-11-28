"""Dual-Recognizer Thai Fusion: combines our PP-OCRv6-small Thai model with OpenThai 2.2 OCR Text.

When a Thai line's primary CTC confidence is below `fallback_below` (default 0.92) or
contains suspicious mixed Latin-in-Thai artifacts, OpenThai 2.2's `rec.onnx` re-reads
the exact same straightened line crop. The reading with higher mean CTC confidence
(penalizing invalid Thai mark stacks) wins!
"""
from __future__ import annotations

from pathlib import Path
import re
import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
_RE_THAI = re.compile(r"[\u0E00-\u0E7F]")
_RE_BAD_THAI = re.compile(r"[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]{2,}")


class OpenThai22Rec:
    """Lightweight ONNX wrapper around OpenThai2.2-OCR-Text `rec.onnx`."""

    def __init__(self, model_dir: Path | None = None, threads: int = 4):
        if model_dir is None:
            cands = sorted(ROOT.glob("models/OpenThai2.2-OCR-Text-*/rec.onnx"))
            if not cands:
                cands = sorted(ROOT.glob("models/OpenThai2.*-OCR-Text-*/rec.onnx"))
            if not cands:
                raise FileNotFoundError("OpenThai 2.2 rec.onnx not found in models/")
            model_dir = cands[-1].parent
        self.model_dir = Path(model_dir)
        self.chars = [""] + (self.model_dir / "rec_dict.txt").read_text(encoding="utf-8").splitlines() + [" "]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.model_dir / "rec.onnx"), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def recognize_with_conf(self, gray_crop: np.ndarray, max_width: int = 3200) -> tuple[str, float]:
        if gray_crop.ndim == 3:
            gray_crop = cv2.cvtColor(gray_crop, cv2.COLOR_RGB2GRAY)
        h, w = gray_crop.shape[:2]
        if h < 4 or w < 4:
            return "", 0.0
        width = min(max_width, max(8, int(np.ceil((48 * w / h) / 8)) * 8))
        g = cv2.resize(gray_crop, (width, 48), interpolation=cv2.INTER_AREA if h > 48 else cv2.INTER_LINEAR)
        x = np.repeat((g.astype(np.float32) / 127.5 - 1.0)[None, None], 3, axis=1)
        y = self.sess.run(None, {self.input_name: x})[0][0]
        # Softmax if logits
        if not (y.min() >= 0 and abs(float(y.sum(-1).mean()) - 1.0) < 0.02):
            e = np.exp(y - y.max(-1, keepdims=True))
            y = e / e.sum(-1, keepdims=True)
        idxs = y.argmax(-1).tolist()
        out, confs, prev = [], [], None
        for t, i in enumerate(idxs):
            if i and i != prev and i < len(self.chars):
                out.append(self.chars[i])
                confs.append(float(y[t, i]))
            prev = i
        text = "".join(out).strip()
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf


def fuse_thai_reading(primary_text: str, primary_conf: float, gray_crop: np.ndarray,
                      ot22: OpenThai22Rec | None, fallback_below: float = 0.92) -> tuple[str, float, str]:
    """Return (best_text, best_conf, source_tag) fusing primary model with OpenThai 2.2."""
    if ot22 is None or not primary_text:
        return primary_text, primary_conf, "primary"
    has_bad = bool(_RE_BAD_THAI.search(primary_text))
    if primary_conf >= fallback_below and not has_bad:
        return primary_text, primary_conf, "primary"
    t2, c2 = ot22.recognize_with_conf(gray_crop)
    if not t2 or not _RE_THAI.search(t2):
        return primary_text, primary_conf, "primary"
    s1 = primary_conf - (0.12 if has_bad else 0.0)
    s2 = c2 - (0.12 if _RE_BAD_THAI.search(t2) else 0.0)
    if s2 > s1 + 0.02:
        return t2, c2, "opentha2.2"
    return primary_text, primary_conf, "primary"
