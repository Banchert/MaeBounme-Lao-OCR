"""Page-level and line-level geometric rectification before OCR.

Handles real-world smartphone photos and screen captures:
  1. Perspective / keystone dewarp (4-corner document or screen quad)
  2. Coarse 90-degree orientation detection (vertical -> horizontal text)
  3. Wide-angle page deskew (up to +-25 deg, coarse-to-fine projection profile)
  4. Per-row / per-line baseline straightening (removes residual tilt & book-spine curve)

All stages include strict safety guards so clean scans and flat screenshots pass
through untouched without unnecessary resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class RectifyInfo:
    perspective_applied: bool = False
    rotated_90: int = 0          # 0, 90, 180, 270 (clockwise degrees applied)
    skew_deg: float = 0.0        # fine deskew angle applied (degrees)
    lines_straightened: int = 0  # count of individual rows straightened

    def summary_th(self) -> str:
        parts = []
        if self.perspective_applied:
            parts.append("แก้ภาพบิด")
        if self.rotated_90:
            parts.append(f"หมุน {self.rotated_90}°")
        if abs(self.skew_deg) >= 0.25:
            parts.append(f"ปรับเอียง {self.skew_deg:+.1f}°")
        if self.lines_straightened > 0:
            parts.append(f"จัดแถวตรง {self.lines_straightened} แถว")
        return " + ".join(parts) if parts else "ภาพตรงอยู่แล้ว"


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return 4 corners ordered as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_angles_ok(q: np.ndarray, min_deg: float = 48.0, max_deg: float = 132.0) -> bool:
    for i in range(4):
        p0 = q[(i - 1) % 4]
        p1 = q[i]
        p2 = q[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return False
        cos_a = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
        deg = math.degrees(math.acos(cos_a))
        if not (min_deg <= deg <= max_deg):
            return False
    return True


def _needs_perspective_warp(q: np.ndarray, w: int, h: int) -> bool:
    """Return True only when the detected quad has real perspective/keystone distortion."""
    tl, tr, br, bl = q
    top_w = float(np.linalg.norm(tr - tl))
    bot_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    if min(top_w, bot_w, left_h, right_h) < 40:
        return False

    # Check trapezoid ratio (opposite edges differ by >= 4.5%)
    w_ratio = abs(top_w - bot_w) / max(top_w, bot_w)
    h_ratio = abs(left_h - right_h) / max(left_h, right_h)

    # Check edge tilt angles relative to horizontal/vertical
    top_ang = abs(math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])))
    bot_ang = abs(math.degrees(math.atan2(br[1] - bl[1], br[0] - bl[0])))
    left_ang = abs(math.degrees(math.atan2(bl[0] - tl[0], bl[1] - tl[1])))
    right_ang = abs(math.degrees(math.atan2(br[0] - tr[0], br[1] - tr[1])))
    max_tilt = max(top_ang, bot_ang, left_ang, right_ang)

    # Also skip if quad is essentially the entire image border with no tilt
    area_frac = cv2.contourArea(q) / float(w * h)
    if area_frac > 0.96 and max_tilt < 2.5 and max(w_ratio, h_ratio) < 0.04:
        return False
    return (max(w_ratio, h_ratio) >= 0.045) or (max_tilt >= 2.5)


def find_document_quad(img: np.ndarray) -> np.ndarray | None:
    """Find a clear 4-corner document or screen boundary in a camera photo, or None."""
    H, W = img.shape[:2]
    if min(H, W) < 220:
        return None
    scale = min(1.0, 900.0 / max(H, W))
    small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    sh, sw = small.shape[:2]
    gray = _to_gray(small)

    # Try two complementary edge maps: Canny on smoothed image + Otsu foreground mask
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    lo = int(max(10, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(blur, lo, hi)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_edges = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

    img_area = float(sh * sw)
    best_quad = None
    best_area = 0.0

    for mask in (edges, otsu_edges):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            area = cv2.contourArea(c)
            if not (0.20 * img_area <= area <= 0.96 * img_area):
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    q = _order_quad(approx)
                    if _quad_angles_ok(q) and area > best_area:
                        best_area = area
                        best_quad = q
                    break

    if best_quad is None:
        return None
    quad_full = best_quad / scale
    if not _needs_perspective_warp(quad_full, W, H):
        return None
    return quad_full


def rectify_perspective(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Warp a 4-corner photographed page/screen to a flat rectangle if clearly distorted."""
    quad = find_document_quad(img)
    if quad is None:
        return img, False
    tl, tr, br, bl = quad
    dst_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    dst_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if dst_w < 100 or dst_h < 100 or dst_w > 10000 or dst_h > 10000:
        return img, False
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    border = tuple(int(v) for v in np.median(img.reshape(-1, img.shape[2]), axis=0)) if img.ndim == 3 else int(np.median(img))
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return warped, True


def _binarize_for_skew(gray: np.ndarray) -> np.ndarray:
    """Fast local-normalized binary mask (ink = 255) suitable for skew estimation."""
    g = 255 - gray if gray.mean() < 127 else gray
    k = max(15, (min(g.shape) // 24) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    norm = cv2.divide(g, np.maximum(bg, 1), scale=255)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = min(t, 225)
    bw = np.where(norm <= t, 255, 0).astype(np.uint8)
    # Remove huge non-text blocks/borders
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n > 1:
        H, W = bw.shape
        for i in range(1, n):
            w_i, h_i = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if w_i > 0.45 * W or h_i > 0.25 * H or (w_i < 2 and h_i < 2):
                bw[labels == i] = 0
    return bw


def detect_vertical_90(gray: np.ndarray) -> int:
    """Detect if text lines run vertically (needs 90 deg rotation) in a tilt-invariant way.

    Compares the best horizontal-line projection score across [-25, +25] deg against
    the best vertical-line projection score across [65, 115] deg, so a tilted horizontal
    page (e.g. 10-20 deg skew) never false-triggers a 90 deg rotation.
    """
    H, W = gray.shape[:2]
    if min(H, W) < 180:
        return 0
    f = min(1.0, 600.0 / max(H, W))
    small = cv2.resize(gray, (max(16, int(W * f)), max(16, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else gray
    bw = _binarize_for_skew(small)
    if (bw > 0).sum() < 150:
        return 0

    sh, sw = bw.shape
    c = (sw / 2.0, sh / 2.0)

    def _peak_score(bw_img: np.ndarray) -> float:
        hh, ww = bw_img.shape
        cc = (ww / 2.0, hh / 2.0)
        best = 0.0
        for a in range(-24, 25, 4):
            M = cv2.getRotationMatrix2D(cc, float(a), 1.0)
            r = cv2.warpAffine(bw_img, M, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0)
            prof = r.sum(axis=1).astype(np.float64)
            sc = float(np.sum(np.diff(prof) ** 2))
            if sc > best:
                best = sc
        return best

    horiz_score = _peak_score(bw)
    vert_score = _peak_score(cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE))
    # Require decisive dominance (> 2.5x) to rotate 90 deg
    if vert_score > 2.5 * max(horiz_score, 1.0):
        return 90
    return 0


def estimate_skew_wide(gray: np.ndarray, max_deg: float = 25.0) -> float:
    """Estimate page tilt in degrees within [-max_deg, +max_deg] using coarse-to-fine projection."""
    H, W = gray.shape[:2]
    if min(H, W) < 120:
        return 0.0
    f = min(1.0, 800.0 / max(H, W))
    small = cv2.resize(gray, (max(16, int(W * f)), max(16, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else gray
    bw = _binarize_for_skew(small)
    if (bw > 0).sum() < 120:
        return 0.0

    sh, sw = bw.shape
    c = (sw / 2.0, sh / 2.0)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(bw, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = r.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    # Coarse scan (-max_deg .. +max_deg, step 1.0 deg)
    coarse = np.arange(-max_deg, max_deg + 1e-6, 1.0)
    best_c = float(max(coarse, key=score))
    # Medium scan around best_c (+-1.2 deg, step 0.25 deg)
    med = np.arange(max(-max_deg, best_c - 1.25), min(max_deg, best_c + 1.25) + 1e-6, 0.25)
    best_m = float(max(med, key=score))
    # Fine scan (+-0.25 deg, step 0.05 deg)
    fine = np.arange(best_m - 0.25, best_m + 0.2501, 0.05)
    best_f = float(max(fine, key=score))

    # Sanity check: require actual improvement over 0 deg so flat pages stay 0.0
    if abs(best_f) < 0.35 or score(best_f) < score(0.0) * 1.03:
        return 0.0
    return best_f


def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg (counter-clockwise, OpenCV convention) expanding canvas so no text is clipped."""
    if abs(angle_deg) < 0.05:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(math.ceil(h * sin + w * cos))
    nh = int(math.ceil(h * cos + w * sin))
    M[0, 2] += (nw / 2.0) - cx
    M[1, 2] += (nh / 2.0) - cy
    if img.ndim == 3:
        edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], axis=0)
        border = tuple(int(v) for v in np.median(edge, axis=0))
    else:
        edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        border = int(np.median(edge))
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def rectify_page(img: np.ndarray, enable_perspective: bool = True,
                 enable_rot90: bool = True, enable_deskew: bool = True,
                 max_skew_deg: float = 25.0) -> tuple[np.ndarray, RectifyInfo]:
    """Full page rectification: perspective dewarp -> 90 deg fix -> wide-angle deskew."""
    info = RectifyInfo()
    h, w = img.shape[:2]
    # Skip single-line crops
    if h <= 72 and w > h * 3.0:
        return img, info

    out = img
    if enable_perspective:
        out, p_ok = rectify_perspective(out)
        info.perspective_applied = p_ok

    gray = _to_gray(out)
    if enable_rot90:
        r90 = detect_vertical_90(gray)
        if r90 == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            gray = _to_gray(out)
            info.rotated_90 = 90

    if enable_deskew:
        ang = estimate_skew_wide(gray, max_deg=max_skew_deg)
        if abs(ang) >= 0.35:
            out = rotate_bound(out, ang)
            info.skew_deg = ang

    return out, info


def straighten_line_crop(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Straighten a single cropped text line (removes residual tilt and curved book-spine wave).

    Returns (straightened_gray, was_modified).
    """
    if gray is None or gray.size == 0:
        return gray, False
    if gray.ndim == 3:
        gray = _to_gray(gray)
    h, w = gray.shape
    # Only worth straightening reasonably wide lines
    if h < 16 or w < max(80, h * 2.5):
        return gray, False

    border = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    g = 255 - gray if border < 127 else gray

    # Local binarize (ink = 255)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = min(t, 225)
    bw = np.where(g <= t, 255, 0).astype(np.uint8)
    if (bw > 0).sum() < max(40, int(h * 1.5)):
        return gray, False

    # Measure vertical ink centroid in horizontal slices across the line
    n_slices = int(np.clip(w // max(28, int(h * 1.1)), 5, 14))
    slice_w = w / n_slices
    xs_pts, ys_pts = [], []
    for i in range(n_slices):
        x0 = int(round(i * slice_w))
        x1 = int(round((i + 1) * slice_w))
        patch = bw[:, x0:x1]
        ys, _ = np.nonzero(patch)
        if len(ys) < max(8, int(h * 0.25)):
            continue
        # Trimmed mean (20th..80th percentile) tracks the main consonant body, ignoring lone tone marks
        lo_y, hi_y = np.percentile(ys, [20, 80])
        core = ys[(ys >= lo_y) & (ys <= hi_y)]
        if len(core) >= 5:
            xs_pts.append(0.5 * (x0 + x1))
            ys_pts.append(float(np.mean(core)))

    if len(xs_pts) < 4:
        return gray, False

    xs_arr = np.array(xs_pts, dtype=np.float64)
    ys_arr = np.array(ys_pts, dtype=np.float64)
    # Span check: points should cover at least 55% of the line width
    if (xs_arr[-1] - xs_arr[0]) < 0.55 * w:
        return gray, False

    # Fit linear and quadratic baseline curves
    x_norm = (xs_arr - w * 0.5) / max(w * 0.5, 1.0)
    deg = 2 if len(xs_pts) >= 6 else 1
    coeffs = np.polyfit(x_norm, ys_arr, deg)
    pred = np.polyval(coeffs, x_norm)

    # Goodness-of-fit guard: only apply if curve genuinely explains vertical drift
    resid = float(np.std(ys_arr - pred))
    drift = float(pred.max() - pred.min())
    # Must have meaningful drift (>= 2.2 px and >= 6% of height), not huge crazy jump, and low residual
    if drift < max(2.2, 0.06 * h) or drift > 0.35 * h or resid > max(3.5, 0.14 * h):
        return gray, False

    # Build vertical shift map to flatten the baseline to its mean y
    grid_x = np.arange(w, dtype=np.float32)
    xn_full = (grid_x - w * 0.5) / max(w * 0.5, 1.0)
    curve_y = np.polyval(coeffs, xn_full).astype(np.float32)
    dy = curve_y - float(np.mean(curve_y))
    # Cap shift per column to avoid clipping tone marks or sub-vowels
    max_shift = float(h * 0.22)
    dy = np.clip(dy, -max_shift, max_shift)

    pad = int(math.ceil(np.max(np.abs(dy)))) + 2
    padded = cv2.copyMakeBorder(gray, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=border)
    H_pad = padded.shape[0]
    map_x, map_y = np.meshgrid(grid_x, np.arange(H_pad, dtype=np.float32))
    map_y = map_y + dy[None, :]
    straight = cv2.remap(padded, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    # Trim excess empty padding while keeping safe margin for Lao tone marks & sub-vowels
    g_s = 255 - straight if border < 127 else straight
    ink_rows = np.where((g_s <= t).any(axis=1))[0]
    if len(ink_rows) >= 4:
        margin_t = max(4, int(round(h * 0.12)))
        margin_b = max(4, int(round(h * 0.14)))
        y0 = max(0, int(ink_rows[0]) - margin_t)
        y1 = min(H_pad, int(ink_rows[-1]) + 1 + margin_b)
        if (y1 - y0) >= max(14, int(h * 0.7)):
            straight = straight[y0:y1, :]

    return straight, True
