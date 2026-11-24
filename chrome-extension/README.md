# Lao + Thai + English OCR — Chrome Side Panel Extension (Offline AI)

ส่วนขยาย Google Chrome (Manifest V3) สำหรับอ่านข้อความ **ภาษาลาว ไทย และอังกฤษ** แบบ **Offline 100%** ด้วยโมเดล **PP-OCRv6 Small SVTR (Round 10)** + **PP-OCRv6 DB Detector** รันตรงในเบราว์เซอร์ด้วย ONNX Runtime Web (ไม่ต้องเปิดเซิร์ฟเวอร์ ไม่ส่งข้อมูลออกนอกเครื่อง)

## ความสามารถหลัก
- **Chrome Side Panel:** เปิดแถบข้างทำงานคู่กับทุกหน้าเว็บและ PDF Viewer
- **✂️ ครอบหน้าจอ (Snip):** ลากครอบข้อความบนหน้าเว็บ ภาพ หรือวิดีโอ แล้วอ่านทันที
- **📋 วางภาพ (`Ctrl+V`) / ลากวางไฟล์:** วางภาพจาก Clipboard ได้ทันที
- **🖱️ คลิกขวาที่รูปภาพ:** เมนูคลิกขวา `🇱🇦 อ่านข้อความในภาพด้วย Lao OCR`
- **🌐 เลือกโหมดภาษา:** อัตโนมัติ / ລາວ+English / ไทย+English / ลาวล้วน / ไทยล้วน / อังกฤษล้วน

## วิธีทดสอบติดตั้งบน Chrome ทันที (Developer Mode)
1. เปิด Chrome ไปที่ `chrome://extensions/`
2. เปิดสวิตช์ **Developer mode** (มุมขวาบน)
3. กด **Load unpacked** แล้วเลือกโฟลเดอร์นี้ (`D:\THAI OCR\lao-ocr-chrome-extension`)
4. กดปักหมุดไอคอน **Lao OCR** บนแถบเครื่องมือ Chrome แล้วคลิกเปิด Side Panel ใช้งานได้เลย!

## วิธีแพ็กเพื่ออัปโหลดขึ้น Chrome Web Store
1. บีบอัดไฟล์ทั้งหมดในโฟลเดอร์ `lao-ocr-chrome-extension/` (ให้ `manifest.json` อยู่ที่ระดับบนสุดของไฟล์ `.zip`)
2. ขนาดรวมหลังบีบอัด ~18–22 MB (ต่ำกว่าเกณฑ์สูงสุดของ Chrome Web Store มาก และไม่มี Remote Code จึงผ่านเงื่อนไข Manifest V3)
3. อัปโหลดขึ้น [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole)
