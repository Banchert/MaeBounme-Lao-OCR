# 📱 Mae Bounmee (Scan) — Offline Lao OCR App (Android & iOS)

แอปพลิเคชันสแกนและแปลงภาพเอกสารภาษาลาว ไทย และอังกฤษ เป็นข้อความดิจิทัลแบบ **Offline 100%** ด้วยโมเดล **Round 10 PP-OCRv6 Small SVTR (`rec.onnx` 12.4 MB)** + **DB Detector (`det.onnx` 1.8 MB)** พัฒนาด้วย **React Native (Expo TypeScript)** สร้างได้ทั้ง **Android (.apk/.aab)** และ **iOS (.ipa)** จากโค้ดชุดเดียว

## สถาปัตยกรรมที่วางไว้ครบแล้ว
- **นำเข้าภาพ:** ถ่ายภาพ (Camera), เลือกหลายภาพจากคลัง (Gallery), นำเข้า PDF
- **ปรับแต่งภาพ (Pre-processing):** หมุนภาพ 90°, ย่อ/ปรับขนาดเหมาะสำหรับ A4 (`expo-image-manipulator`)
- **ระบบสร้างรายได้ Hybrid ([src/services/monetization.ts](src/services/monetization.ts)):**
  - **FREE:** แสดง Dialog แจ้งเตือนสุภาพ → เล่น **AdMob Interstitial Ad** ควบคู่กับการรัน OCR เบื้องหลังพร้อมกัน
  - **PRO (ซื้อขาดถาวร):** สแกนทันทีไม่มีโฆษณา + ปลดล็อกส่งออกไฟล์ **Word (`.docx`)**
- **หน้าจอผลลัพธ์ ([src/screens/ResultScreen.tsx](src/screens/ResultScreen.tsx)):** แก้ไขข้อความด้วยฟอนต์ **Phetsarath OT**, คัดลอกคลิกเดียว (One-Tap Copy), ส่งออก `.txt` (ฟรี) และ `.docx` (PRO)
- **รองรับ UI 3 ภาษา:** ລາວ / ไทย / English ([src/i18n/translations.ts](src/i18n/translations.ts))

## วิธีติดตั้งและ Build เป็น Android APK / iOS IPA
```bash
cd "D:\THAI OCR\mae-bounmee-scan-app"
npm install
npm install -g eas-cli
eas login

# Build Android APK สำหรับติดตั้งทดสอบบนมือถือทันที (ไม่ต้องลง Android Studio!)
eas build --platform android --profile preview

# Build iOS IPA สำหรับขึ้น App Store (Build บนคลาวด์ ไม่ต้องมีเครื่อง Mac!)
eas build --platform ios --profile production
```

## ก่อนปล่อยขึ้น Google Play & Apple App Store
1. เปลี่ยน `androidAppId` / `iosAppId` ใน `app.json` และ `ADMOB_INTERSTITIAL_ID` ใน `src/services/monetization.ts` เป็นรหัส Google AdMob จริงของคุณ
2. เชื่อมรหัสสินค้าซื้อขาด (Lifetime PRO) ใน `src/services/monetization.ts` ผ่าน RevenueCat (`react-native-purchases`)
