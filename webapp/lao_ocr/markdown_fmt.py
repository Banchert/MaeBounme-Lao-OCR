"""Structured Markdown & Table formatter for Lao/Thai/English OCR pages.

Converts detected row boxes + recognized cell texts into clean GitHub-flavored Markdown:
  - Headings: larger lines (>= 1.35x median line height) -> # / ##
  - Tables: consecutive multi-cell rows (>= 2 columns) with consistent column alignment -> | col | col |
  - Bullet / numbered lists & Lao legal articles (ມາດຕາ / ຂໍ້) -> clean Markdown structure
  - Paragraph spacing: vertical gaps > 1.6x median line height -> blank line
"""
from __future__ import annotations

import re
from typing import Sequence


def _esc_cell(text: str) -> str:
    return text.replace("|", "\\|").strip()


def _box_rect(box) -> tuple[float, float, float, float]:
    """4-point quad -> (x1, y1, x2, y2)."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _is_table_row(cells: list[tuple[str, tuple[float, float, float, float]]], page_w: float) -> bool:
    if len(cells) < 2:
        return False
    # Check horizontal separation between adjacent cells
    gaps = 0
    for i in range(len(cells) - 1):
        _, (_, _, x2a, _) = cells[i]
        _, (x1b, _, _, _) = cells[i + 1]
        if x1b - x2a >= max(10.0, page_w * 0.012):
            gaps += 1
    return gaps >= 1


def _render_table(table_rows: list[list[tuple[str, tuple[float, float, float, float]]]]) -> list[str]:
    """Align cells across rows into a clean Markdown table using x-center clustering."""
    # Determine number of columns from the most common / max cell count (capped at 10)
    max_cols = min(10, max(len(r) for r in table_rows))
    if max_cols < 2:
        return ["   ".join(t for t, _ in r) for r in table_rows]

    # Collect x1 positions of rows with max_cols to build column anchors
    anchors_list = [sorted(rect[0] for _, rect in r) for r in table_rows if len(r) == max_cols]
    if anchors_list:
        col_x = [sum(a[c] for a in anchors_list) / len(anchors_list) for c in range(max_cols)]
    else:
        all_x = sorted(rect[0] for r in table_rows for _, rect in r)
        step = max(1, len(all_x) // max_cols)
        col_x = [all_x[min(len(all_x) - 1, c * step)] for c in range(max_cols)]

    grid: list[list[str]] = []
    for r in table_rows:
        row_cells = [""] * max_cols
        for text, rect in r:
            # Assign to nearest column anchor
            c_idx = min(range(max_cols), key=lambda c: abs(rect[0] - col_x[c]))
            if row_cells[c_idx]:
                row_cells[c_idx] += " " + _esc_cell(text)
            else:
                row_cells[c_idx] = _esc_cell(text)
        grid.append(row_cells)

    md = []
    md.append("| " + " | ".join(grid[0]) + " |")
    md.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in grid[1:]:
        md.append("| " + " | ".join(row) + " |")
    return md


def format_markdown(
    rows_cells: Sequence[Sequence[tuple[str, Sequence[Sequence[float]]]]],
    page_w: float,
    line_height: float,
) -> str:
    """rows_cells: list of rows, each row is list of (text, quad_box)."""
    lh = max(float(line_height or 20.0), 8.0)
    norm_rows: list[list[tuple[str, tuple[float, float, float, float]]]] = []
    for row in rows_cells:
        items = []
        for text, box in row:
            t = text.strip()
            if t:
                items.append((t, _box_rect(box)))
        if items:
            items.sort(key=lambda x: x[1][0])
            norm_rows.append(items)

    if not norm_rows:
        return ""

    out: list[str] = []
    i = 0
    n = len(norm_rows)
    prev_y2: float | None = None

    while i < n:
        row = norm_rows[i]
        row_y1 = min(r[1][1] for r in row)
        row_y2 = max(r[1][3] for r in row)
        row_h = max(r[1][3] - r[1][1] for r in row)

        # Paragraph break on large vertical gap
        if prev_y2 is not None and (row_y1 - prev_y2) > lh * 1.45 and out and out[-1] != "":
            out.append("")

        # Check for a multi-row table block (>= 2 consecutive multi-cell rows)
        if _is_table_row(row, page_w):
            j = i + 1
            while j < n and _is_table_row(norm_rows[j], page_w):
                j += 1
            if j - i >= 2:
                if out and out[-1] != "":
                    out.append("")
                out.extend(_render_table(norm_rows[i:j]))
                out.append("")
                prev_y2 = max(r[1][3] for r in norm_rows[j - 1])
                i = j
                continue

        joined = "   ".join(t for t, _ in row)
        # Heading detection: single short line significantly taller than body text
        if len(row) == 1 and len(joined) <= 90 and row_h >= lh * 1.45:
            out.append(f"# {joined}")
        elif len(row) == 1 and len(joined) <= 90 and row_h >= lh * 1.25:
            out.append(f"## {joined}")
        elif re.match(r"^(?:ມາດຕາ|ໝວດທີ|ພາກທີ|ບົດທີ)\s*\d+", joined):
            out.append(f"**{joined}**")
        elif re.match(r"^[•·▪–—]\s*", joined):
            out.append("- " + re.sub(r"^[•·▪–—]\s*", "", joined))
        else:
            out.append(joined)

        prev_y2 = row_y2
        i += 1

    # Collapse duplicate blank lines
    cleaned: list[str] = []
    for line in out:
        if line == "" and (not cleaned or cleaned[-1] == ""):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()
