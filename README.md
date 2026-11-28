<p align="center">
  <img src="assets/logo.png" alt="MaeBounme Lao-OCR" width="220" />
</p>

<h1 align="center">MaeBounme (Lao • Thai • EN OCR)</h1>

<p align="center">
  <b>100% Free, Lightweight &amp; Truly Offline AI OCR for Lao, Thai, and English</b><br/>
  <i>Apache-2.0 Open-Source — Dedicated in loving memory of Mae Bounme (ອຸທິດແດ່ ແມ່ບຸນມີ)</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License: Apache-2.0" />
  <img src="https://img.shields.io/badge/Offline-100%25%20Private-green.svg" alt="100% Offline" />
  <img src="https://img.shields.io/badge/Lao%20%7C%20Thai%20%7C%20EN-ONNX%20FP32-orange.svg" alt="ONNX FP32" />
</p>

---

## ✨ Key Features

- **🔒 100% Offline & Private** — Runs completely on your device (Android / Chrome / Desktop). Your photos and confidential documents are **never** uploaded to any cloud server.
- **⚡ Lightweight & Fast** — Compact ONNX models (`~11.8 MB` per language) with high accuracy across Lao, Thai, and English text.
- **📄 Photos & Multi-Page PDFs** — Scan single or multi-page images and PDFs with page-range selection and per-page 90° rotation.
- **📐 Smart Page & Row Rectification** — Auto perspective dewarp (4-corner), 90° orientation fix, wide-angle deskew (±25°), and per-row curved baseline straightening before OCR.
- **🧾 Single-Pass Unified OCR (Text + Markdown Tables + Bill Auto-Checker)** — One model outputs Plain Text, structured Markdown tables (`.md`), and automatic receipt/bill math verification (`qty × price` & Grand Total check).
- **🎯 Per-Word Confidence Highlights & Mobile UX** — Low-confidence words (`<85%`) are automatically underlined in orange with error-risk %, and mobile results open in a smooth read-only scroll view (keyboard only opens when tapping Edit).
- **💡 Built-in Lao Lexicon & Word Suggestions** — Offline Lao vocabulary post-correction and tap-to-fix word suggestions.
- **📑 Side-by-Side Comparison & Export** — Compare original document alongside OCR output, color-coded proofreading, 1-tap copy, and free `.txt` / `.md` / Word (`.docx`) export.
- **🕊️ Free Forever** — Zero ads, zero subscriptions, zero page limits.

---

## 📦 Downloads (Releases)

Check the **[Releases](https://github.com/Banchert/MaeBounme-Lao-OCR/releases)** page to download ready-to-use packages:
- **Android App (`.apk`)**: `MaeBounme-Lao-OCR-v1.8.0.apk` (Lao R18 All-343-Fonts + Thai R3 All-1,256-Fonts 99.5% + OpenThai 2.2 Detector + Confidence Underlines + Keyboard Lock)
- **Chrome Side Panel Extension (`.zip`)**: `MaeBounme-Lao-OCR-v1.3.0.zip` (Lao R18 + Thai R3 + OpenThai 2.2 Detector)

---

## 🧩 Install Chrome Extension (Offline)

1. Download and unzip `MaeBounme-Lao-OCR-v1.0.0.zip` (or use the `chrome-extension/` folder in this repo).
2. Open Chrome and navigate to `chrome://extensions/`.
3. Enable **Developer mode** (top right).
4. Click **Load unpacked** and select the unzipped folder.
5. Click the **MaeBounme** icon in your Chrome toolbar to open the Side Panel.

---

## 🐍 Python ONNX Inference

Ready-to-use ONNX models are located in `models/`:
- `models/det.onnx` — Text line detector (`1.70 MB`)
- `models/rec.onnx` — Lao + English recognizer (`11.83 MB`)
- `models/thai_rec.onnx` — Thai + English recognizer (`11.83 MB`)
- `models/dict.txt`, `models/thai_dict.txt` — Character dictionaries
- `models/lao_lexicon.json` — Offline Lao word lexicon

See [`examples/ocr_demo.py`](examples/ocr_demo.py) for a minimal inference example:

```bash
pip install onnxruntime opencv-python numpy
python examples/ocr_demo.py path/to/document.jpg
```

---

## ☕ Voluntary Support & Contact

This project is **100% Free & Open-Source (Apache-2.0)**, dedicated in memory of **แม่บุญมี (Mae Bounme)**.  
If it helps your work or studies, voluntary coffee support or feedback is warmly welcome:

- **Developer**: ໂກ້ (MaeBounme)
- **Facebook**: Chert Victor Sign
- **WhatsApp**: [+856 20 5959 2939](https://wa.me/8562059592939)

---

## 📄 License

Licensed under the [Apache License 2.0](LICENSE).
