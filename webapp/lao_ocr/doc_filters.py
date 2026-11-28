"""Document Preprocessing Filters & Multi-Page/Multi-Column Reading Order for Lao & Thai OCR.

Provides:
  1. remove_stamps_and_watermarks(rgb):
     - Removes red/blue rubber stamps without damaging overlapping black text
     - Suppresses faint gray diagonal (45 deg) watermarks while preserving thin dark Lao tone marks
     - Optionally returns the isolated watermark mask so it can be rotated +-45 deg and read separately
  2. order_boxes_multipage(boxes, page_w, page_h):
     - Recursive XY-Cut supporting single-column, 2-page book spreads (left page -> right page),
       and 4-page 2x2 grids (TL -> TR -> BL -> BR), plus paragraph indentation markers!
"""
from __future__ import annotations

import cv2
import numpy as np


def remove_stamps_and_watermarks(
    rgb: np.ndarray,
    remove_red_blue_stamps: bool = True,
    remove_faint_watermark: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Clean colored stamps & faint gray watermarks from an RGB page.

    Returns (cleaned_rgb, watermark_gray_mask_or_None).
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb, None

    out = rgb[:, :, :3].copy()
    H, W = out.shape[:2]
    r = out[:, :, 0].astype(np.int16)
    g = out[:, :, 1].astype(np.int16)
    b = out[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

    # Estimate paper white
    paper = int(np.percentile(gray, 92))
    if paper < 150:
        return out, None  # dark-mode screen: skip paper-white stamp/watermark cleaning

    wm_mask = np.zeros((H, W), dtype=np.uint8)

    if remove_red_blue_stamps:
        # Red stamp pixels: R clearly higher than G & B, and not dark black ink
        red_mask = ((r - np.maximum(g, b)) > 36) & (gray > 75)
        # Blue stamp pixels: B clearly higher than R & G, and not dark black ink
        blue_mask = ((b - np.maximum(r, g)) > 44) & (gray > 75)
        stamp_mask = red_mask | blue_mask
        if stamp_mask.any():
            out[stamp_mask] = (paper, paper, paper)
            wm_mask[stamp_mask] = 255

    if remove_faint_watermark:
        # Faint gray watermarks: large low-contrast gray blobs far lighter than real text ink
        g2 = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        dark_ink = g2[g2 < paper - 50]
        if dark_ink.size > 200:
            ink_med = float(np.percentile(dark_ink, 35))
            # Faint watermark band: clearly lighter than dark text ink, darker than pure paper
            # Safety margin ensures thin anti-aliased Lao tone marks (core < ink_med + 55) are untouched!
            wm_lo = int(min(paper - 18, max(168, ink_med + 95)))
            wm_hi = int(paper - 6)
            if wm_hi > wm_lo + 8:
                cand = ((g2 >= wm_lo) & (g2 <= wm_hi)).astype(np.uint8) * 255
                # Only remove LARGE connected watermark strokes (area >= 220 px), never tiny tone marks!
                n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
                for i in range(1, n):
                    area = stats[i, cv2.CC_STAT_AREA]
                    bw_i = stats[i, cv2.CC_STAT_WIDTH]
                    bh_i = stats[i, cv2.CC_STAT_HEIGHT]
                    if area >= 220 and (bw_i >= 24 or bh_i >= 24):
                        m = labels == i
                        out[m] = (paper, paper, paper)
                        wm_mask[m] = 255

    has_wm = (wm_mask > 0).sum() > 400
    return out, (255 - wm_mask if has_wm else None)


def _xy_cut_indices(rects: list[tuple[float, float, float, float]], indices: list[int],
                    lh: float, depth: int = 0) -> list[list[int]]:
    """Recursively partition box indices into page/column blocks (handles 2-page spreads & 4-page 2x2 grids)."""
    if len(indices) < 4 or depth >= 4:
        return [indices]

    xs1 = [rects[i][0] for i in indices]
    ys1 = [rects[i][1] for i in indices]
    xs2 = [rects[i][2] for i in indices]
    ys2 = [rects[i][3] for i in indices]
    bx1, by1, bx2, by2 = min(xs1), min(ys1), max(xs2), max(ys2)
    bw, bh = bx2 - bx1, by2 - by1

    # 1) Try a clean horizontal split first if this looks like a 2-row (4-page) grid
    if bh > 12 * lh:
        ys_sorted = sorted(indices, key=lambda i: 0.5 * (rects[i][1] + rects[i][3]))
        best_h_gap, best_h_split = 0.0, None
        max_y2 = rects[ys_sorted[0]][3]
        for k in range(1, len(ys_sorted)):
            cur_y1 = rects[ys_sorted[k]][1]
            gap = cur_y1 - max_y2
            if gap > best_h_gap and k >= 3 and (len(ys_sorted) - k) >= 3:
                # Gap must sit roughly in middle 25%..75% of block height
                mid_y = 0.5 * (max_y2 + cur_y1)
                if by1 + 0.25 * bh < mid_y < by1 + 0.75 * bh:
                    best_h_gap = gap
                    best_h_split = k
            if rects[ys_sorted[k]][3] > max_y2:
                max_y2 = rects[ys_sorted[k]][3]
        if best_h_split is not None and best_h_gap >= max(3.2 * lh, 0.06 * bh):
            top_ids = ys_sorted[:best_h_split]
            bot_ids = ys_sorted[best_h_split:]
            return _xy_cut_indices(rects, top_ids, lh, depth + 1) + _xy_cut_indices(rects, bot_ids, lh, depth + 1)

    # 2) Try a clean full-height vertical gutter (2-page book spread / 2-column page)
    # Allow up to 1 stray box (e.g. center book spine or watermark fragment) in the gutter when >= 10 boxes
    if bw > 10 * lh:
        bins = max(16, int(bw / 2))
        occ = np.zeros(bins, dtype=np.int32)
        for i in indices:
            c0 = max(0, int((rects[i][0] - bx1) / 2))
            c1 = min(bins, int(np.ceil((rects[i][2] - bx1) / 2)))
            occ[c0:c1] += 1
        lo_b, hi_b = int(bins * 0.25), int(bins * 0.75)
        max_cross = 1 if len(indices) >= 10 else 0
        best_g, best_mid_x = 0, None
        for thr in (0, max_cross):
            s = None
            for b_i in range(lo_b, hi_b):
                if occ[b_i] <= thr:
                    if s is None:
                        s = b_i
                else:
                    if s is not None:
                        gw = (b_i - s) * 2
                        if gw > best_g:
                            best_g = gw
                            best_mid_x = bx1 + (s + b_i)
                        s = None
            if s is not None:
                gw = (hi_b - s) * 2
                if gw > best_g:
                    best_g = gw
                    best_mid_x = bx1 + (s + hi_b)
            if best_mid_x is not None and best_g >= max(1.6 * lh, 0.025 * bw):
                break

        if best_mid_x is not None and best_g >= max(1.6 * lh, 0.025 * bw):
            left_ids = [i for i in indices if 0.5 * (rects[i][0] + rects[i][2]) < best_mid_x]
            right_ids = [i for i in indices if 0.5 * (rects[i][0] + rects[i][2]) >= best_mid_x]
            if len(left_ids) >= 3 and len(right_ids) >= 3:
                return _xy_cut_indices(rects, left_ids, lh, depth + 1) + _xy_cut_indices(rects, right_ids, lh, depth + 1)

    return [indices]


def filter_non_text_boxes(boxes: list[np.ndarray]) -> list[np.ndarray]:
    """Drop giant square icons, ornaments, logos, or stamp blobs (> 3x median line height)."""
    if len(boxes) < 3:
        return boxes
    hs = []
    for b in boxes:
        ys = [float(p[1]) for p in b]
        hs.append(max(ys) - min(ys))
    med_lh = max(10.0, float(np.median(hs)))
    kept = []
    for b in boxes:
        xs = [float(p[0]) for p in b]
        ys = [float(p[1]) for p in b]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        is_giant_icon = bh > med_lh * 3.0 and (bw / max(1.0, bh)) < 2.2
        if not is_giant_icon:
            kept.append(b)
    return kept if kept else boxes


def group_rows_multipage_blocks(boxes: list[np.ndarray]) -> list[list[list[int]]]:
    """Return list of page/column blocks, each block containing reading-order rows of box indices.
    Automatically splits 2-page book spreads and 4-page 2x2 grids!"""
    if not boxes:
        return []
    rects = []
    hs = []
    for b in boxes:
        xs = [float(p[0]) for p in b]
        ys = [float(p[1]) for p in b]
        rects.append((min(xs), min(ys), max(xs), max(ys)))
        hs.append(max(ys) - min(ys))
    lh = max(10.0, float(np.median(hs)))

    blocks = _xy_cut_indices(rects, list(range(len(boxes))), lh)
    out_blocks: list[list[list[int]]] = []
    for blk in blocks:
        blk_sorted = sorted(blk, key=lambda i: 0.5 * (rects[i][1] + rects[i][3]))
        rows: list[list[int]] = []
        for idx in blk_sorted:
            _, y1, _, y2 = rects[idx]
            placed = False
            for r in rows:
                _, ry1, _, ry2 = rects[r[-1]]
                ov = min(ry2, y2) - max(ry1, y1)
                if ov > 0.45 * min(ry2 - ry1, y2 - y1):
                    r.append(idx)
                    placed = True
                    break
            if not placed:
                rows.append([idx])
        for r in rows:
            r.sort(key=lambda i: rects[i][0])
        if rows:
            out_blocks.append(rows)
    return out_blocks


def group_rows_multipage(boxes: list[np.ndarray]) -> list[list[int]]:
    """Flat reading-order rows across all detected page/column blocks."""
    out: list[list[int]] = []
    for blk_rows in group_rows_multipage_blocks(boxes):
        out.extend(blk_rows)
    return out


def format_blocks_text(
    rows_cells: list[list[tuple[str, np.ndarray]]],
    row_blocks: list[int] | None = None,
    line_height: float = 20.0,
) -> str:
    """Format recognized rows with automatic multi-page block separation & paragraph indentation."""
    if not rows_cells:
        return ""
    lh = max(10.0, float(line_height or 20.0))
    if not row_blocks or len(row_blocks) != len(rows_cells):
        row_blocks = [0] * len(rows_cells)

    # Group rows by block index
    blocks: dict[int, list[list[tuple[str, np.ndarray]]]] = {}
    for rc, bi in zip(rows_cells, row_blocks):
        blocks.setdefault(bi, []).append(rc)

    rendered_blocks: list[str] = []
    for bi in sorted(blocks.keys()):
        b_rows = blocks[bi]
        lefts = sorted(float(r[0][1][:, 0].min()) for r in b_rows)
        body_left = lefts[int(len(lefts) * 0.25)] if lefts else 0.0
        out: list[str] = []
        prev_y2: float | None = None
        for r in b_rows:
            y1 = min(float(b[:, 1].min()) for _, b in r)
            y2 = max(float(b[:, 1].max()) for _, b in r)
            x1 = float(r[0][1][:, 0].min())
            txt = "   ".join(t for t, _ in r)
            is_indented = (x1 - body_left > lh * 1.15) and (len(r) == 1)
            big_gap = prev_y2 is not None and (y1 - prev_y2 > lh * 1.15)
            if (big_gap or is_indented) and out and out[-1] != "":
                out.append("")
            out.append(("    " if is_indented else "") + txt)
            prev_y2 = y2
        txt_blk = "\n".join(out).strip()
        if txt_blk:
            rendered_blocks.append(txt_blk)

    return "\n\n".join(rendered_blocks)


def read_watermark_layer(
    wm_gray: np.ndarray | None,
    recognize_fn,
    detect_fn=None,
) -> list[str]:
    """Optionally read isolated watermark/stamp mask at -45 deg, +45 deg, and 0 deg."""
    if wm_gray is None or wm_gray.size == 0:
        return []
    from .rectify import rotate_bound

    best_lines: list[str] = []
    best_score = 0
    for angle in (-45.0, 45.0, 0.0):
        rot = rotate_bound(wm_gray, angle) if angle != 0.0 else wm_gray
        lines_here: list[str] = []
        if detect_fn is not None:
            bgr = cv2.cvtColor(rot, cv2.COLOR_GRAY2BGR)
            boxes = detect_fn(bgr)
            for b in boxes[:8]:
                xs = [int(p[0]) for p in b]
                ys = [int(p[1]) for p in b]
                x1, x2 = max(0, min(xs) - 4), min(rot.shape[1], max(xs) + 4)
                y1, y2 = max(0, min(ys) - 4), min(rot.shape[0], max(ys) + 4)
                if x2 - x1 >= 16 and y2 - y1 >= 10:
                    t = recognize_fn(rot[y1:y2, x1:x2]).strip()
                    if len(t) >= 2:
                        lines_here.append(t)
        score = sum(len(s) for s in lines_here)
        if score > best_score:
            best_score = score
            best_lines = lines_here
    return best_lines
