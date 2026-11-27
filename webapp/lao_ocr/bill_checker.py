"""Automatic Bill / Receipt Checker & Calculator for Lao, Thai, and English OCR text.

Parses OCR-extracted lines from a receipt/invoice:
  1. Detects item rows (name, qty, unit price, printed line total)
  2. Verifies each row's math: qty * unit_price == printed_line_total
  3. Sums all item totals (+ VAT - discount if present)
  4. Compares calculated grand total against the printed Total (ລວມ / ລວມທັງໝົດ / รวมทั้งหมด / Total)
  5. Renders a clean Markdown + color-coded HTML audit report for the Gradio UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import html
import re

_LAO_THAI_DIGITS = str.maketrans("໐໑໒໓໔໕໖໗໘໙๐๑๒๓๔๕๖๗๘๙", "01234567890123456789")

# Keywords
_RE_TOTAL = re.compile(
    r"(ລວມທັງໝົດ|ລວມຍອດ|ຍອດລວມ|ລວມເງິນ|ລວມສຸດທິ|รวมทั้งหมด|รวมทั้งสิ้น|ยอดรวม|รวมสุทธิ|รวมเงิน|grand\s*total|net\s*total|total\s*amount|amount\s*due|\btotal\b|^ລວມ\s*[:\d]|^รวม\s*[:\d])",
    re.IGNORECASE,
)
_RE_SUBTOTAL = re.compile(r"(ລວມຍ່ອຍ|รวมย่อย|sub\s*-?\s*total)", re.IGNORECASE)
_RE_VAT = re.compile(r"(ອາກອນ|ພາສີ|ภาษี|vat\b|tax\b)", re.IGNORECASE)
_RE_DISCOUNT = re.compile(r"(ສ່ວນຫຼຸດ|ຫຼຸດລາຄາ|ส่วนลด|discount)", re.IGNORECASE)
_RE_CASH = re.compile(r"(ຮັບເງິນ|ເງິນສົດ|ຊຳລະ|รับเงิน|เงินสด|cash\b|paid\b|tender)", re.IGNORECASE)
_RE_CHANGE = re.compile(r"(ເງິນທອນ|เงินทอน|change\b)", re.IGNORECASE)
_RE_HEADER_SKIP = re.compile(
    r"(ລາຍການສິນຄ້າ|ລາຍການ\s+ຈຳນວນ|ໃບເກັບເງິນ|ໃບບິນ|ເລກທີບິນ|ວັນທີ\s*:|วันที่\s*:|ເບີໂທ|โทร|invoice|receipt|tax\s*id|ຂໍຂອບໃຈ|ขอบคุณ|thank\s*you)",
    re.IGNORECASE,
)
_RE_DATE = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
_RE_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_RE_PHONE = re.compile(r"(?:\+?856|020|030|0\d{2})[\s\-]?\d{3,4}[\s\-]?\d{3,4}")
_RE_NUM = re.compile(r"(?<![A-Za-z0-9/.\-])\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|(?<![A-Za-z0-9/.\-])\d+(?:\.\d{1,2})?(?![A-Za-z0-9/\-])")


@dataclass
class BillItem:
    raw_line: str
    name: str
    qty: float | None
    unit_price: float | None
    printed_total: float
    calc_total: float
    ok: bool = True
    note: str = ""


@dataclass
class BillReport:
    items: list[BillItem] = field(default_factory=list)
    sum_items: float = 0.0
    subtotal_printed: float | None = None
    vat: float = 0.0
    discount: float = 0.0
    expected_total: float = 0.0
    printed_total: float | None = None
    cash_paid: float | None = None
    change_printed: float | None = None
    currency: str = "ກີບ"
    diff: float | None = None
    row_errors: int = 0

    @property
    def is_match(self) -> bool:
        if not self.items:
            return False
        if self.row_errors > 0:
            return False
        if self.printed_total is not None and self.diff is not None:
            return abs(self.diff) <= 1.0
        return True

    def fmt(self, v: float | None) -> str:
        if v is None:
            return "-"
        if abs(v - round(v)) < 1e-6:
            return f"{int(round(v)):,}"
        return f"{v:,.2f}"

    def to_markdown(self) -> str:
        if not self.items and self.printed_total is None:
            return "_ไม่พบรายการสินค้าหรือยอดเงินในภาพนี้_"
        lines = []
        if self.printed_total is not None:
            if self.is_match:
                lines.append(f"**✅ ยอดบิลถูกต้องตรงกัน:** `{self.fmt(self.expected_total)} {self.currency}`")
            else:
                lines.append(
                    f"**⚠️ ยอดไม่ตรง:** คำนวณจริง `{self.fmt(self.expected_total)} {self.currency}` "
                    f"vs บนบิล `{self.fmt(self.printed_total)} {self.currency}` (ต่าง `{self.fmt(self.diff)} {self.currency}`)"
                )
        else:
            lines.append(f"**🧮 รวมยอดจาก {len(self.items)} รายการ:** `{self.fmt(self.expected_total)} {self.currency}`")
        lines.append("")
        lines.append("| # | รายการสินค้า | จำนวน | ราคา/หน่วย | ยอดบนบิล | คำนวณจริง | สถานะ |")
        lines.append("|---:|---|---:|---:|---:|---:|:---:|")
        for idx, it in enumerate(self.items, 1):
            st = "✅" if it.ok else f"⚠️ {it.note}"
            lines.append(
                f"| {idx} | {it.name.replace('|', '/')} | {self.fmt(it.qty)} | {self.fmt(it.unit_price)} | "
                f"{self.fmt(it.printed_total)} | {self.fmt(it.calc_total)} | {st} |"
            )
        lines.append(
            f"| | **รวมทั้งหมด ({self.currency})** | | | **{self.fmt(self.printed_total)}** | **{self.fmt(self.expected_total)}** | |"
        )
        return "\n".join(lines)

    def to_html(self) -> str:
        if not self.items and self.printed_total is None:
            return (
                '<div style="padding:14px;border-radius:10px;background:#f8fafc;border:1px solid #cbd5e1;color:#475569;">'
                '🧾 ยังไม่พบรายการสินค้าหรือยอดเงินในข้อความที่อ่าน — ลองอัปโหลดภาพใบเสร็จ/บิลที่มีรายการและราคาแล้วกดสแกนครับ'
                '</div>'
            )

        if self.printed_total is not None:
            if self.is_match:
                banner_bg, banner_bd, banner_fg = "#ecfdf5", "#10b981", "#065f46"
                banner_txt = f"✅ ยอดบิลถูกต้องตรงกัน! คำนวณได้ {self.fmt(self.expected_total)} {self.currency} (ตรงกับยอดบนบิล {self.fmt(self.printed_total)} {self.currency})"
            else:
                banner_bg, banner_bd, banner_fg = "#fef2f2", "#ef4444", "#991b1b"
                diff_str = f"{self.diff:+,.0f}" if self.diff is not None and abs(self.diff - round(self.diff)) < 1e-6 else f"{(self.diff or 0):+,.2f}"
                banner_txt = (
                    f"⚠️ พบความคลาดเคลื่อน! คำนวณจริงได้ {self.fmt(self.expected_total)} {self.currency} "
                    f"แต่บนบิลระบุ {self.fmt(self.printed_total)} {self.currency} (ต่างกัน {diff_str} {self.currency})"
                )
                if self.row_errors:
                    banner_txt += f" • มี {self.row_errors} แถวที่คูณไม่ตรง"
        else:
            banner_bg, banner_bd, banner_fg = "#eff6ff", "#3b82f6", "#1e40af"
            banner_txt = f"🧮 รวมยอดจาก {len(self.items)} รายการได้ {self.fmt(self.expected_total)} {self.currency} (ไม่พบบรรทัดยอดรวมบนบิล)"

        rows_html = []
        for idx, it in enumerate(self.items, 1):
            st_badge = (
                '<span style="color:#059669;font-weight:700;">✅ ตรง</span>'
                if it.ok else
                f'<span style="color:#dc2626;font-weight:700;">⚠️ {html.escape(it.note)}</span>'
            )
            bg_row = "#fff5f5" if not it.ok else ("#ffffff" if idx % 2 else "#f8fafc")
            rows_html.append(
                f'<tr style="background:{bg_row};">'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;text-align:center;">{idx}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;">{html.escape(it.name)}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;text-align:right;">{self.fmt(it.qty)}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;text-align:right;">{self.fmt(it.unit_price)}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;text-align:right;font-weight:600;">{self.fmt(it.printed_total)}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;text-align:right;">{self.fmt(it.calc_total)}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #e2e8f0;text-align:center;">{st_badge}</td>'
                f'</tr>'
            )

        extras = []
        if self.discount > 0:
            extras.append(f"ส่วนลด: -{self.fmt(self.discount)} {self.currency}")
        if self.vat > 0:
            extras.append(f"ภาษี/VAT: +{self.fmt(self.vat)} {self.currency}")
        if self.cash_paid is not None:
            exp_change = self.cash_paid - self.expected_total
            extras.append(f"รับเงิน: {self.fmt(self.cash_paid)} | เงินทอนที่ควรได้: {self.fmt(exp_change)}")
        extra_html = (
            f'<div style="margin-top:8px;font-size:13px;color:#334155;">{" &nbsp;|&nbsp; ".join(extras)}</div>'
            if extras else ""
        )

        return (
            f'<div style="font-family:Phetsarath OT,Sarabun,sans-serif;">'
            f'<div style="padding:12px 14px;border-radius:10px;background:{banner_bg};border:1.5px solid {banner_bd};color:{banner_fg};font-weight:700;font-size:15px;margin-bottom:10px;">'
            f'{banner_txt}</div>'
            f'<div style="overflow-x:auto;border:1px solid #cbd5e1;border-radius:8px;">'
            f'<table style="width:100%;border-collapse:collapse;font-size:14px;">'
            f'<thead><tr style="background:#1e3a8a;color:#ffffff;">'
            f'<th style="padding:7px 8px;">#</th>'
            f'<th style="padding:7px 8px;text-align:left;">รายการสินค้า</th>'
            f'<th style="padding:7px 8px;text-align:right;">จำนวน</th>'
            f'<th style="padding:7px 8px;text-align:right;">ราคา/หน่วย</th>'
            f'<th style="padding:7px 8px;text-align:right;">ยอดบนบิล</th>'
            f'<th style="padding:7px 8px;text-align:right;">คำนวณจริง</th>'
            f'<th style="padding:7px 8px;text-align:center;">สถานะ</th>'
            f'</tr></thead><tbody>{"".join(rows_html)}</tbody>'
            f'<tfoot><tr style="background:#eff6ff;font-weight:700;">'
            f'<td colspan="4" style="padding:8px;text-align:right;">รวมคำนวณ ({len(self.items)} รายการ):</td>'
            f'<td style="padding:8px;text-align:right;">{self.fmt(self.printed_total)}</td>'
            f'<td style="padding:8px;text-align:right;color:#1d4ed8;">{self.fmt(self.expected_total)}</td>'
            f'<td style="padding:8px;text-align:center;">{html.escape(self.currency)}</td>'
            f'</tr></tfoot></table></div>{extra_html}</div>'
        )


def _normalize_line(line: str) -> str:
    return line.translate(_LAO_THAI_DIGITS).strip()


def _detect_currency(text: str) -> str:
    if any(k in text for k in ("ກີບ", "₭", "KIP", "Kip", "kip")):
        return "ກີບ"
    if any(k in text for k in ("บาท", "฿", "THB")):
        return "บาท"
    if any(k in text for k in ("USD", "$")):
        return "USD"
    return "ກີບ"


def _extract_numbers(line: str) -> list[tuple[int, int, float]]:
    """Return list of (start, end, float_value) ignoring dates, times, and phone numbers."""
    masked = _RE_DATE.sub(lambda m: " " * len(m.group(0)), line)
    masked = _RE_TIME.sub(lambda m: " " * len(m.group(0)), masked)
    masked = _RE_PHONE.sub(lambda m: " " * len(m.group(0)), masked)
    out = []
    for m in _RE_NUM.finditer(masked):
        raw = m.group(0).replace(",", "")
        try:
            val = float(raw)
            out.append((m.start(), m.end(), val))
        except ValueError:
            continue
    return out


def check_bill(ocr_text: str) -> BillReport:
    """Parse OCR text from a bill/receipt and verify all row multiplications and grand total."""
    rep = BillReport(currency=_detect_currency(ocr_text or ""))
    if not (ocr_text or "").strip():
        return rep

    for raw in ocr_text.splitlines():
        line = _normalize_line(raw)
        if not line or len(line) < 2:
            continue
        nums = _extract_numbers(line)
        if not nums:
            continue

        # Classify summary / metadata lines first
        if _RE_SUBTOTAL.search(line):
            rep.subtotal_printed = nums[-1][2]
            continue
        if _RE_DISCOUNT.search(line):
            rep.discount += abs(nums[-1][2])
            continue
        if _RE_VAT.search(line):
            rep.vat += abs(nums[-1][2])
            continue
        if _RE_CASH.search(line):
            rep.cash_paid = nums[-1][2]
            continue
        if _RE_CHANGE.search(line):
            rep.change_printed = nums[-1][2]
            continue
        if _RE_TOTAL.search(line):
            rep.printed_total = nums[-1][2]
            continue
        if _RE_HEADER_SKIP.search(line):
            continue

        # Parse item row from trailing numeric columns
        # Strip leading row index like "1." or "2)" if present
        line_no_idx = re.sub(r"^\s*\d{1,2}[.)]\s+", "", line)
        nums = _extract_numbers(line_no_idx)
        if not nums:
            continue

        # Case 1: >= 3 numbers -> check if last 3 are (qty, unit_price, line_total)
        if len(nums) >= 3:
            s_q, _, q = nums[-3]
            _, _, p = nums[-2]
            _, _, tot = nums[-1]
            # Plausible qty (0 < q <= 10000) and price > 0
            if 0 < q <= 10000 and p > 0 and tot > 0:
                calc = q * p
                name = line_no_idx[:s_q].strip(" -|:\t") or line_no_idx
                # Guard against false-positive unit number in name like "200 ໜ້າ  18,000  108,000"
                # If q * p matches tot (within 1%), definite 3-col match!
                ok = abs(calc - tot) <= max(1.0, 0.005 * max(calc, tot))
                note = "" if ok else f"ควรเป็น {rep.fmt(calc)}"
                rep.items.append(BillItem(raw, name, q, p, tot, calc, ok=ok, note=note))
                if not ok:
                    rep.row_errors += 1
                continue

        # Case 2: 2 numbers -> could be (qty, total) or (name_size, total)
        if len(nums) == 2:
            s1, _, v1 = nums[0]
            s2, _, v2 = nums[1]
            # If first number is embedded inside item name (followed by Lao/Thai unit text before second num)
            between = line_no_idx[nums[0][1]:s2].strip()
            has_unit_text = bool(re.search(r"[ກ-໿ก-๛A-Za-z]", between))
            if has_unit_text and v2 >= 100:
                name = line_no_idx[:s2].strip(" -|:\t") or line_no_idx
                rep.items.append(BillItem(raw, name, 1.0, v2, v2, v2, ok=True))
            elif 1 <= v1 <= 999 and v2 >= 100:
                # Could be qty and line total (no unit price shown)
                name = line_no_idx[:s1].strip(" -|:\t") or line_no_idx
                rep.items.append(BillItem(raw, name, v1, v2 / v1, v2, v2, ok=True))
            continue

        # Case 3: 1 trailing money number
        s0, _, val = nums[-1]
        name = line_no_idx[:s0].strip(" -|:\t")
        if name and val >= 100:
            rep.items.append(BillItem(raw, name, 1.0, val, val, val, ok=True))

    rep.sum_items = sum(it.calc_total for it in rep.items)
    rep.expected_total = rep.sum_items - rep.discount + rep.vat
    if rep.printed_total is not None:
        rep.diff = rep.printed_total - rep.expected_total
    return rep
