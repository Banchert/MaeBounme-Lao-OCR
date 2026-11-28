"""Page -> text lines with a *learned* text detector.

Uses the PP-OCRv6-tiny DB text detector, reading order and perspective line crops of
OpenThai2.1-OCR-Text (models/OpenThai2.1-OCR-Text-20261118, Apache-2.0).  The detector finds
text regions, not Thai letters, so it works for Lao too, and it handles what projection cutting
(layout.py) cannot: phone photos with perspective, tilted pages, several columns or several
documents side by side, mixed letter sizes, table cells.  Every detected quad is warped to a
straight line image before recognition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_DET_CANDIDATES = sorted(ROOT.glob('models/OpenThai2.*-OCR-Text-*/det.onnx'))
DET_DIR = _DET_CANDIDATES[-1].parent if _DET_CANDIDATES else (ROOT / 'models' / 'OpenThai2.2-OCR-Text-20261127')
_ENGINE = None
_ERR = None


def available() -> bool:
    global _ERR
    if not (DET_DIR / 'det.onnx').is_file():
        _ERR = 'det.onnx not found'
        return False
    try:
        import pyclipper  # noqa: F401
    except ImportError:
        _ERR = 'pyclipper not installed (run install_detector.bat)'
        return False
    return True


def _engine():
    global _ENGINE
    if _ENGINE is None:
        if str(DET_DIR) not in sys.path:
            sys.path.insert(0, str(DET_DIR))
        from runtime.line_engine import LineOCR
        _ENGINE = LineOCR(DET_DIR, threads=4)
    return _ENGINE


def _pad_quad(b: np.ndarray, v: float = 0.15, h: float = 0.25) -> np.ndarray:
    """grow a detected quad: v x line height above and below, h x line height at both ends
    (the recognizer was trained on lines with a little margin)"""
    b = b.astype(np.float32).copy()
    height = max(1.0, float(np.linalg.norm(b[0] - b[3])))
    up = (b[0] - b[3]) / height
    width = max(1.0, float(np.linalg.norm(b[1] - b[0])))
    right = (b[1] - b[0]) / width
    b[0] += up * v * height - right * h * height
    b[1] += up * v * height + right * h * height
    b[2] += -up * v * height + right * h * height
    b[3] += -up * v * height - right * h * height
    return b


def read_rows_with_boxes(img: np.ndarray, rgb: bool = True, rectify: bool = True):
    """Same as read_rows, but each row contains (crop, quad_box) tuples for Markdown/table layout.

    When rectify=True:
      1. Page-level perspective dewarp + 90-deg fix + wide-angle deskew (lao_ocr.rectify)
      2. Detector-guided fine deskew: if detected lines still lean >= 1.5 deg, rotate full page
         and re-detect so quads are axis-aligned and thin Lao marks aren't lost on angled crops
      3. Per-row baseline straightening (removes residual tilt / curved book-spine wave)
    """
    eng = _engine()
    from runtime.line_engine import crop_line, line_height, page_skew, reading_order
    from .rectify import RectifyInfo, rectify_page, rotate_bound, straighten_line_crop

    work = img
    rinfo = RectifyInfo()
    if rectify:
        work, rinfo = rectify_page(work)

    if work.ndim == 2:
        bgr = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(work[:, :, :3], cv2.COLOR_RGB2BGR) if rgb else work[:, :, :3]

    boxes = eng.detect(bgr)
    if not boxes:
        return [], {'boxes': 0, 'line_height': 0.0, 'skew': rinfo.skew_deg,
                    'rectify_summary': rinfo.summary_th(), 'rectified_img': work}

    # Detector-guided fine deskew: if boxes still lean >= 1.5 deg, straighten full page & re-detect
    det_skew = float(page_skew(boxes))
    if rectify and 1.5 <= abs(det_skew) <= 35.0:
        work = rotate_bound(work, det_skew)
        rinfo.skew_deg += det_skew
        if work.ndim == 2:
            bgr = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(work[:, :, :3], cv2.COLOR_RGB2BGR) if rgb else work[:, :, :3]
        boxes2 = eng.detect(bgr)
        if boxes2:
            boxes = boxes2
            det_skew = float(page_skew(boxes))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    rows = []
    n_straight = 0
    for row in reading_order(boxes):
        items = []
        for k in row:
            q = _pad_quad(boxes[k])
            q[:, 0] = q[:, 0].clip(0, W - 1)
            q[:, 1] = q[:, 1].clip(0, H - 1)
            c = crop_line(gray, q)
            if c.shape[0] >= 4 and c.shape[1] >= 4:
                if rectify:
                    c, mod = straighten_line_crop(c)
                    if mod:
                        n_straight += 1
                items.append((c, boxes[k]))
        if items:
            rows.append(items)
    rinfo.lines_straightened = n_straight
    return rows, {
        'boxes': len(boxes),
        'line_height': float(line_height(boxes)),
        'skew': float(rinfo.skew_deg if abs(rinfo.skew_deg) >= 0.25 else det_skew),
        'rectify_summary': rinfo.summary_th(),
        'rectified_img': work,
    }


def read_rows(img: np.ndarray, rgb: bool = True, rectify: bool = True):
    """-> (rows, info): rows = list of rows top-to-bottom, each a list of grey line crops left-to-right
    (cells of one table row / parts of one line); info = {'boxes', 'line_height', 'skew', ...}"""
    rows_b, info = read_rows_with_boxes(img, rgb=rgb, rectify=rectify)
    return [[c for c, _ in r] for r in rows_b], info
