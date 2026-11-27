"""Page -> text-line segmentation for the line recognizer.

The recognizer was trained on single text lines (48 px high after resize,
~4-35% vertical padding).  A whole page must therefore be cut into one crop per
line first.  This module does that with resolution-independent rules (all
thresholds are derived from the measured character height), so it works for
both phone photos and 300-DPI scans.

Lao-specific detail: tone marks and above/below vowels (່ ້ ິ ີ ຸ ູ ...) often form
their own thin horizontal bands; they are merged into the nearest text line
instead of being emitted as separate "lines".
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Below this measured x-height (px) the text is simply too small to read.
MIN_READABLE_CHAR_PX = 9
# Pages are rescaled so a typical line body is about this tall before cutting.
TARGET_CHAR_PX = 32


@dataclass
class Line:
    image: np.ndarray            # grayscale crop, ready for the recognizer
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in the *input* image
    segments: list[np.ndarray]   # sub-crops split at very wide horizontal gaps


@dataclass
class PageResult:
    lines: list[Line]
    char_height: float           # measured in the input image (px)
    scale: float                 # resize factor applied before cutting
    warning: str | None
    skew: float = 0.0               # degrees the page was rotated to straighten it
    rectify_summary: str = ""       # human-readable summary of rectification applied
    rectified_img: np.ndarray | None = None


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255.  Flattens uneven lighting (photos) before Otsu."""
    if gray.mean() < 127:
        gray = 255 - gray
    k = max(15, (min(gray.shape) // 20) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Ink must be clearly darker than the local paper; stops paper texture / noise on
    # faint pages being taken as text when Otsu's threshold lands inside the noise.
    t = min(t, 225)
    return np.where(norm < t, 255, 0).astype(np.uint8)


def estimate_char_height(bw: np.ndarray) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return 0.0
    h = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    w = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = (h >= 3) & (area >= 6) & (w < 0.5 * bw.shape[1]) & (h < max(0.2 * bw.shape[0], 0.9 * min(bw.shape[0], 200)))
    if keep.sum() < 5:
        return float(np.median(h)) if len(h) else 0.0
    # Use the upper half of component heights = main letter bodies, not dots/marks.
    hk = np.sort(h[keep])
    return float(np.median(hk[len(hk) // 2:]))


def _remove_non_text(bw: np.ndarray, ch: float) -> np.ndarray:
    """Drop logos/photos/table rules so they do not glue lines together."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()
    W = bw.shape[1]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        too_tall = h > 4.0 * ch                    # logo, stamp, photo, vertical rule
        rule = w > max(8 * ch, 0.25 * W) and h < 0.5 * ch and area > 0.6 * w * h  # solid horizontal line
        if too_tall or rule:
            out[labels == i] = 0
    return out


def estimate_skew(bw: np.ndarray, max_deg: float = 7.0) -> float:
    """Page tilt in degrees: the angle whose row profile has the sharpest peaks (text lines vs gaps)."""
    h, w = bw.shape
    f = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(bw, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    c = (sw / 2, sh / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = r.sum(1).astype(np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    angles = np.arange(-max_deg, max_deg + 1e-6, 0.25)
    best = max(angles, key=score)
    fine = np.arange(best - 0.25, best + 0.2501, 0.05)
    return float(max(fine, key=score))


def _rotate(img, angle, border):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def _bands(profile: np.ndarray, thr: float) -> list[list[int]]:
    on = profile > thr
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, len(profile)])
    return bands


def _split_tall(profile: np.ndarray, band: list[int], line_h: float) -> list[list[int]]:
    """Recursively cut a band that contains several touching lines at profile valleys."""
    y1, y2 = band
    if (y2 - y1) < 1.6 * line_h:
        return [band]
    seg = profile[y1:y2].astype(np.float32)
    k = max(3, int(line_h * 0.25) | 1)
    seg = cv2.GaussianBlur(seg.reshape(-1, 1), (1, k), 0).ravel()
    lo, hi = int(0.55 * line_h), (y2 - y1) - int(0.55 * line_h)
    if hi <= lo:
        return [band]
    cut = y1 + lo + int(np.argmin(seg[lo:hi]))
    return _split_tall(profile, [y1, cut], line_h) + _split_tall(profile, [cut, y2], line_h)


def _x_segments(bw_line: np.ndarray, ch: float) -> list[tuple[int, int]]:
    cols = bw_line.sum(0) > 0
    xs = np.flatnonzero(cols)
    if len(xs) == 0:
        return []
    gap_split = 4.0 * ch        # wide gap = separate column (e.g. signature row)
    segs, s, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - prev > gap_split:
            segs.append((s, prev + 1))
            s = x
        prev = x
    segs.append((s, prev + 1))
    return segs


def _crop(gray: np.ndarray, x1, y1, x2, y2, ch) -> np.ndarray:
    """Crop with padding similar to the training renders (~15-25% of char height)."""
    H, W = gray.shape
    pv, ph = int(round(0.2 * ch)), int(round(0.5 * ch))
    c = gray[max(0, y1 - pv):min(H, y2 + pv), max(0, x1 - ph):min(W, x2 + ph)]
    return c


def is_single_line(img: np.ndarray) -> bool:
    """A typical line crop: short and wide. Fed to the recognizer as-is."""
    h, w = img.shape[:2]
    return not (h > 64 and h / w > 0.15)


def _segment_single(img: np.ndarray) -> PageResult:
    gray0 = _to_gray(img)
    if gray0.mean() < 127:
        gray0 = 255 - gray0
    if is_single_line(gray0):
        h, w = gray0.shape
        return PageResult([Line(gray0, (0, 0, w, h), [gray0])], float(h), 1.0, None)
    bw0 = _binarize(gray0)
    ch0 = estimate_char_height(bw0)
    if ch0 <= 0:
        return PageResult([], 0.0, 1.0, "ไม่พบตัวอักษรในภาพ")

    warning = None
    if ch0 < MIN_READABLE_CHAR_PX:
        warning = (f"ภาพความละเอียดต่ำเกินไป: ตัวอักษรสูงเพียง ~{ch0:.0f} px "
                   f"(ควรอย่างน้อย {MIN_READABLE_CHAR_PX * 2} px). ผลลัพธ์จะผิดพลาดมาก — "
                   "กรุณาใช้ไฟล์สแกน/ถ่ายรูปต้นฉบับ (≥200 DPI) แทนภาพหน้าจอย่อ")

    # Normalize scale so every rule below sees ~TARGET_CHAR_PX tall characters.
    scale = float(np.clip(TARGET_CHAR_PX / ch0, 0.35, 4.0))
    if abs(scale - 1) > 0.15:
        gray = cv2.resize(gray0, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        bw = _binarize(gray)
        ch = estimate_char_height(bw) or ch0 * scale
    else:
        scale, gray, bw, ch = 1.0, gray0, bw0, ch0

    text_bw = _remove_non_text(bw, ch)
    angle = estimate_skew(text_bw)
    if abs(angle) >= 0.15:  # straighten tilted scans / photos, otherwise long lines run into each other
        gray = _rotate(gray, angle, int(np.median(gray)))
        bw = _binarize(gray)
        text_bw = _remove_non_text(bw, ch)
    profile = text_bw.sum(1) / 255.0
    thr = max(1.0, 0.02 * ch)   # ignore single stray pixels only
    raw = _bands(profile, thr)
    if not raw:
        return PageResult([], ch0, scale, warning or "ไม่พบบรรทัดข้อความ")

    mass = np.array([profile[a:b].sum() for a, b in raw])
    heights = np.array([b - a for a, b in raw])
    big = mass >= 0.25 * np.median(mass[mass >= np.percentile(mass, 50)])
    line_h = float(np.median(heights[big])) if big.any() else float(np.median(heights))
    # When most bands are merged lines the median is itself too tall: a Lao line (with tone marks
    # and lower vowels) is about 2x the letter-body height, so cap the estimate there.
    line_h = float(np.clip(line_h, ch, 2.4 * ch))

    # 1) split bands that contain several touching lines
    bands = []
    for b in raw:
        bands.extend(_split_tall(profile, b, line_h))

    # 2) merge thin "satellite" bands (tone marks, upper/lower vowels, dots) into the nearest line
    def is_main(b):
        a, z = b
        return (z - a) >= 0.45 * line_h and profile[a:z].sum() >= 0.15 * np.median(mass)

    mains = [list(b) for b in bands if is_main(b)]
    if not mains:
        mains = [list(b) for b in bands]
    for b in bands:
        if is_main(b):
            continue
        a, z = b
        best, dist = None, None
        for m in mains:
            d = max(m[0] - z, a - m[1], 0)
            if dist is None or d < dist:
                best, dist = m, d
        if best is not None and dist <= 0.6 * line_h:
            best[0], best[1] = min(best[0], a), max(best[1], z)
        # else: isolated noise -> dropped
    mains.sort()

    lines: list[Line] = []
    for y1, y2 in mains:
        segs = _x_segments(text_bw[y1:y2], ch)
        if not segs:
            continue
        x1, x2 = segs[0][0], segs[-1][1]
        if (x2 - x1) < 0.6 * ch and (y2 - y1) < 0.8 * ch:
            continue   # a lone speck
        full = _crop(gray, x1, y1, x2, y2, ch)
        parts = [_crop(gray, s, y1, e, y2, ch) for s, e in segs]
        inv = 1.0 / scale
        box = (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
        lines.append(Line(full, box, parts))
    return PageResult(lines, ch0, scale, warning, angle)


# ---------------------------------------------------------------- columns / side-by-side documents
def _gaps(occ: np.ndarray, min_w: float):
    """runs of empty positions (occ == False) at least min_w long, excluding the borders"""
    out, i, n = [], 0, len(occ)
    while i < n:
        if not occ[i]:
            j = i
            while j < n and not occ[j]:
                j += 1
            if i > 0 and j < n and j - i >= min_w:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _xy_cut(text: np.ndarray, ch: float, x0=0, y0=0, depth=0):
    """Recursive XY-cut. A full-height vertical gutter (two columns / two documents side by side)
    is cut first, otherwise wide horizontal gaps. Returns (blocks in reading order, used_vertical)."""
    ys = np.where(text.any(1))[0]
    xs = np.where(text.any(0))[0]
    if not len(ys) or not len(xs):
        return [], False
    t = text[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    bx, by = x0 + xs[0], y0 + ys[0]
    h, w = t.shape
    if depth > 14 or h < 2.5 * ch:
        return [(bx, by, bx + w, by + h)], False
    col = (t > 0).sum(0)
    # a real gutter: wide, through a tall region, with a substantial text block on both sides
    # (a form with big gaps between a label and its value must NOT be cut into columns)
    vg = [g for g in _gaps(col > max(1, 0.01 * h), max(2.2 * ch, 0.025 * w))
          if 0.25 * w < g[0] and g[1] < 0.75 * w and h > 10 * ch and depth <= 1]
    def bands(part):
        occ = (part > 0).sum(1) > max(1, 0.01 * part.shape[1])
        return int(np.sum(occ[1:] & ~occ[:-1]) + occ[0])
    vg = [g for g in vg if bands(t[:, :g[0]]) >= 5 and bands(t[:, g[1]:]) >= 5]  # columns of real text on both sides
    if vg:
        a, b = max(vg, key=lambda g: g[1] - g[0])
        L, Lv = _xy_cut(t[:, :a], ch, bx, by, depth + 1)
        R, Rv = _xy_cut(t[:, b:], ch, bx + b, by, depth + 1)
        return L + R, True
    row = (t > 0).sum(1)
    hg = _gaps(row > max(1, 0.004 * w), 1.3 * ch)
    if hg:
        a, b = max(hg, key=lambda g: g[1] - g[0])
        T, Tv = _xy_cut(t[:a], ch, bx, by, depth + 1)
        B, Bv = _xy_cut(t[b:], ch, bx, by + b, depth + 1)
        return T + B, Tv or Bv
    return [(bx, by, bx + w, by + h)], False


def segment_page(img: np.ndarray, rectify: bool = True) -> PageResult:
    """Cut a page into text lines. Pages with several columns or several documents side by side
    (two forms photographed together, a screenshot of a page next to other text) are first split
    into blocks; every block is then cut on its own, so each gets its own letter size."""
    from .rectify import RectifyInfo, rectify_page, straighten_line_crop

    rinfo = RectifyInfo()
    work = img
    if rectify:
        work, rinfo = rectify_page(work)

    gray0 = _to_gray(work)
    if gray0.mean() < 127:
        gray0 = 255 - gray0

    def _finalize(res: PageResult) -> PageResult:
        if rectify and res.lines:
            n_st = 0
            for ln in res.lines:
                ln.image, m = straighten_line_crop(ln.image)
                if m:
                    n_st += 1
                new_segs = []
                for s in ln.segments:
                    s2, _ = straighten_line_crop(s)
                    new_segs.append(s2)
                ln.segments = new_segs
            rinfo.lines_straightened = n_st
        rinfo.skew_deg = rinfo.skew_deg if abs(rinfo.skew_deg) >= 0.25 else res.skew
        res.skew = rinfo.skew_deg
        res.rectify_summary = rinfo.summary_th()
        res.rectified_img = work
        return res

    if is_single_line(gray0) or min(gray0.shape) < 200:
        return _finalize(_segment_single(work))
    try:
        f = min(1.0, 1400.0 / max(gray0.shape))
        small = cv2.resize(gray0, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray0
        bw = _binarize(small)
        ch = estimate_char_height(bw)
        if ch <= 0:
            return _finalize(_segment_single(work))
        angle = estimate_skew(_remove_non_text(bw, ch))
        if abs(angle) >= 0.15:
            small = _rotate(small, angle, int(np.median(small)))
            bw = _binarize(small)
        k = max(10, int(4 * ch))
        # box frames / table rules (also slightly tilted ones: thicken first) must not block a gutter
        vbw = cv2.dilate(bw, np.ones((1, 5), np.uint8))
        hbw = cv2.dilate(bw, np.ones((5, 1), np.uint8))
        rules = cv2.bitwise_or(cv2.morphologyEx(vbw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))),
                               cv2.morphologyEx(hbw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))))
        text = _remove_non_text(cv2.subtract(bw, rules), ch)
        blocks, vertical = _xy_cut(text, ch)
    except Exception:
        return _finalize(_segment_single(work))
    if not vertical or len(blocks) < 2:
        return _finalize(_segment_single(work))  # ordinary single-column page: unchanged behaviour
    # cut every block from the straightened full-resolution page
    full = _rotate(gray0, angle, int(np.median(gray0))) if abs(angle) >= 0.15 else gray0
    inv = 1.0 / f
    pad = int(0.6 * ch * inv)
    lines, chs, warns = [], [], []
    H, W = full.shape
    for x1, y1, x2, y2 in blocks:
        X1, Y1 = max(0, int(x1 * inv) - pad), max(0, int(y1 * inv) - pad)
        X2, Y2 = min(W, int(x2 * inv) + pad), min(H, int(y2 * inv) + pad)
        crop = full[Y1:Y2, X1:X2]
        if crop.size == 0:
            continue
        res = _segment_single(crop)
        for ln in res.lines:
            bx1, by1, bx2, by2 = ln.box
            lines.append(Line(ln.image, (bx1 + X1, by1 + Y1, bx2 + X1, by2 + Y1), ln.segments))
        if res.char_height:
            chs.append(res.char_height)
        if res.warning:
            warns.append(res.warning)
    warn = warns[0] if len(warns) > len(blocks) / 2 else None  # a tiny block (page number, stamp text) is not "the page"
    return _finalize(PageResult(lines, float(np.median(chs)) if chs else ch * inv, 1.0, warn, angle))


_JUNK = set('-_=.…·•*~+|/\\:;,\'"`()[]{}<>!?^°#oO0 ')


def is_junk_line(text: str) -> bool:
    """dotted leaders / ruled lines / separators read as '------' or '.......' with no real text"""
    t = text.strip()
    if not t:
        return True
    core = [c for c in t if c not in _JUNK]
    return len(core) <= 1 and len(t) >= 3


def to_tensor(gray_crop: np.ndarray, max_width: int = 4096) -> np.ndarray:
    """Same preprocessing as lao_ocr.core.preprocess (training), numpy only."""
    h, w = gray_crop.shape[:2]
    width = max(8, int(np.ceil((48 * w / h) / 8)) * 8)
    width = min(width, max_width)
    g = cv2.resize(gray_crop, (width, 48), interpolation=cv2.INTER_AREA if h > 48 else cv2.INTER_LINEAR)
    v = g.astype(np.float32) / 127.5 - 1.0
    return np.repeat(v[None, None], 3, axis=1)
