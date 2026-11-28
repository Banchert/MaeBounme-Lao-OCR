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


def _expand_quad_ink_aware(
    gray: np.ndarray,
    boxes: list[np.ndarray],
    idx: int,
    max_top: float = 0.45,
    max_bot: float = 0.42,
    h_pad: float = 0.34,
) -> np.ndarray:
    """Smart Ink-Aware Box Expansion for Lao & Thai:
      1. Clamps upward/downward expansion at the midpoint to neighboring lines above/below
         so dense lines NEVER swallow adjacent tone marks or sub-vowels.
      2. Inspects actual ink pixels inside the safe band above/below the DB box:
         - Captures floating tone marks (່ ້ ໊ ໋), top vowels (ິ ີ ຶ ື), tall stems (ປ ຝ ຟ ໂ ໃ ໄ),
           and bottom vowels (ຸ ູ ຼ) completely.
         - Avoids bloating plain lines with excess whitespace so letters stay tall at H=48px!
      3. Adds breathing margin (~10% of height) matching the HarfBuzz training distribution.
    """
    H, W = gray.shape[:2]
    b = boxes[idx].astype(np.float32).copy()
    height = max(4.0, float(np.linalg.norm(b[0] - b[3])))
    up = (b[0] - b[3]) / height
    width = max(4.0, float(np.linalg.norm(b[1] - b[0])))
    right = (b[1] - b[0]) / width

    x1_cur, x2_cur = float(b[:, 0].min()), float(b[:, 0].max())
    y1_cur, y2_cur = float(b[:, 1].min()), float(b[:, 1].max())

    # 1) Find nearest horizontally-overlapping neighbor above and below to clamp expansion
    ceil_y = max(0.0, y1_cur - max_top * height)
    floor_y = min(float(H - 1), y2_cur + max_bot * height)
    for j, ob in enumerate(boxes):
        if j == idx:
            continue
        ox1, ox2 = float(ob[:, 0].min()), float(ob[:, 0].max())
        # Must overlap horizontally by at least 20% of min width
        ov_x = min(x2_cur, ox2) - max(x1_cur, ox1)
        if ov_x < 0.20 * min(x2_cur - x1_cur, ox2 - ox1):
            continue
        oy1, oy2 = float(ob[:, 1].min()), float(ob[:, 1].max())
        if oy2 <= y1_cur + 0.15 * height:
            mid = 0.5 * (oy2 + y1_cur)
            if mid > ceil_y:
                ceil_y = mid
        elif oy1 >= y2_cur - 0.15 * height:
            mid = 0.5 * (y2_cur + oy1)
            if mid < floor_y:
                floor_y = mid

    # 2) Inspect actual ink pixels in the clamped region to snap tightly to top marks & bottom vowels
    rx1 = max(0, int(np.floor(x1_cur - h_pad * height)))
    rx2 = min(W, int(np.ceil(x2_cur + h_pad * height)))
    ry1 = max(0, int(np.floor(ceil_y)))
    ry2 = min(H, int(np.ceil(floor_y)))

    v_top_px = min(0.24 * height, max(2.0, y1_cur - ceil_y))
    v_bot_px = min(0.22 * height, max(2.0, floor_y - y2_cur))

    if ry2 - ry1 >= 6 and rx2 - rx1 >= 6:
        patch = gray[ry1:ry2, rx1:rx2]
        bg = float(np.percentile(patch, 85)) if patch.mean() >= 127 else float(np.percentile(patch, 15))
        dark = np.abs(patch.astype(np.float32) - bg) > max(26.0, 0.18 * abs(bg - float(patch.min() if patch.mean() >= 127 else patch.max())))
        row_has_ink = dark.sum(axis=1) >= max(2, int((rx2 - rx1) * 0.008))
        ink_ys = np.flatnonzero(row_has_ink)
        if len(ink_ys) >= 3:
            # Find ink extent connected within gap <= 0.30*height of the core box (ignores stray neighbor specks)
            core_top_rel = max(0, int(y1_cur - ry1))
            core_bot_rel = min(ry2 - ry1 - 1, int(y2_cur - ry1))
            max_gap = max(3, int(round(0.32 * height)))

            # Walk upward from core_top_rel
            snap_top = core_top_rel
            gap = 0
            for r in range(core_top_rel, -1, -1):
                if row_has_ink[r]:
                    snap_top = r
                    gap = 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break

            # Walk downward from core_bot_rel
            snap_bot = core_bot_rel
            gap = 0
            for r in range(core_bot_rel, ry2 - ry1):
                if row_has_ink[r]:
                    snap_bot = r
                    gap = 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break

            breath = max(2.0, 0.10 * height)
            target_y1 = max(ceil_y, float(ry1 + snap_top) - breath)
            target_y2 = min(floor_y, float(ry1 + snap_bot) + breath)
            v_top_px = max(2.0, y1_cur - target_y1)
            v_bot_px = max(2.0, target_y2 - y2_cur)

    b[0] += up * v_top_px - right * h_pad * height
    b[1] += up * v_top_px + right * h_pad * height
    b[2] += -up * v_bot_px + right * h_pad * height
    b[3] += -up * v_bot_px - right * h_pad * height
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
            q = _expand_quad_ink_aware(gray, boxes, k)
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
