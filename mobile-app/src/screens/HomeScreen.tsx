import React, { useState } from "react";
import { Alert, Image, Linking, Modal, ScrollView, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { TRANSLATIONS, UiLang } from "../i18n/translations";
import { mobileOcr, OcrModelMode, OutputFormatMode } from "../services/ocrEngine";
import { DONATE_INFO } from "../services/monetization";

interface Props {
  uiLang: UiLang;
  setUiLang: (l: UiLang) => void;
  onResult: (text: string, ms: number, avgConf: number, lineCount: number, images: string[], rotations: number[]) => void;
}

export default function HomeScreen({ uiLang, setUiLang, onResult }: Props) {
  const t = TRANSLATIONS[uiLang];
  const [images, setImages] = useState<string[]>([]);
  const [rotations, setRotations] = useState<number[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [modelMode, setModelMode] = useState<OcrModelMode>("lao");
  const [outputFormat, setOutputFormat] = useState<OutputFormatMode>("plain");
  const [busy, setBusy] = useState(false);
  const [donateOpen, setDonateOpen] = useState(false);
  const [activeQr, setActiveQr] = useState<"lak" | "usd" | "thb" | null>(null);

  const QR_IMAGES = {
    lak: require("../../assets/images/qr_lak.jpg"),
    usd: require("../../assets/images/qr_usd.jpg"),
    thb: require("../../assets/images/qr_thb.jpg")
  };

  const pickCamera = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) return;
    const res = await ImagePicker.launchCameraAsync({ quality: 0.95, allowsEditing: true });
    if (!res.canceled && res.assets[0]) {
      setImages([res.assets[0].uri]);
      setRotations([0]);
      setActiveIdx(0);
    }
  };

  const pickGallery = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: true,
      quality: 0.95
    });
    if (!res.canceled && res.assets.length) {
      setImages(res.assets.map((a) => a.uri));
      setRotations(res.assets.map(() => 0));
      setActiveIdx(0);
    }
  };

  const rotateCurrent = () => {
    setRotations((prev) => prev.map((r, idx) => (idx === activeIdx ? (r + 90) % 360 : r)));
  };

  const rotateAll = () => {
    setRotations((prev) => prev.map((r) => (r + 90) % 360));
  };

  const removeCurrentPage = () => {
    setImages((prev) => prev.filter((_, idx) => idx !== activeIdx));
    setRotations((prev) => prev.filter((_, idx) => idx !== activeIdx));
    setActiveIdx((idx) => Math.max(0, idx - 1));
  };

  const onScanPress = async () => {
    if (!images.length || busy) return;
    setBusy(true);
    try {
      const res = await mobileOcr.scanImages(images, rotations, modelMode, outputFormat);
      onResult(res.text, res.ms, res.avgConf, res.lineCount, images, rotations);
    } catch (e: any) {
      Alert.alert("OCR Error", e?.message || "Scan failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.wrap}>
      <View style={styles.header}>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>{t.appName}</Text>
          <Text style={styles.sub}>{t.tagline}</Text>
        </View>
        <View style={styles.langRow}>
          {(["lo", "th", "en"] as UiLang[]).map((l) => (
            <TouchableOpacity key={l} style={[styles.pill, uiLang === l && styles.pillActive]} onPress={() => setUiLang(l)}>
              <Text style={[styles.pillText, uiLang === l && styles.pillTextActive]}>
                {l === "lo" ? "ລາວ" : l === "th" ? "ไทย" : "EN"}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.tierBar}>
        <Text style={styles.tierText} numberOfLines={1}>{t.freeBanner}</Text>
        <TouchableOpacity style={styles.donateMiniBtn} onPress={() => setDonateOpen(true)}>
          <Text style={styles.donateMiniText}>{t.donateBtn}</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.promoBox}>
        <Text style={styles.promoText}>{t.promoLine}</Text>
      </View>

      <View style={styles.modelRow}>
        {([
          ["lao", "🇱🇦 ລາວ (MaeBounme)"],
          ["lao+en", "🇱🇦 ລາວ + EN"],
          ["thai", "🇹🇭 ไทย Free"]
        ] as [OcrModelMode, string][]).map(([val, lbl]) => (
          <TouchableOpacity
            key={val}
            style={[styles.modelBtn, modelMode === val && styles.modelBtnActive]}
            onPress={() => setModelMode(val)}
          >
            <Text style={[styles.modelBtnText, modelMode === val && styles.modelBtnTextActive]}>{lbl}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <View style={styles.modelRow}>
        {([
          ["plain", "📄 ຂໍ້ຄວາມປົກກະຕິ (Text)"],
          ["markdown", "🧾 Markdown (.md + ตาราง/หัวข้อ)"]
        ] as [OutputFormatMode, string][]).map(([val, lbl]) => (
          <TouchableOpacity
            key={val}
            style={[styles.modelBtn, outputFormat === val && styles.modelBtnActive]}
            onPress={() => setOutputFormat(val)}
          >
            <Text style={[styles.modelBtnText, outputFormat === val && styles.modelBtnTextActive]}>{lbl}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <View style={styles.card}>
        <TouchableOpacity style={[styles.actionBtn, styles.primary]} onPress={pickCamera}>
          <Text style={styles.btnText}>{t.cameraBtn}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionBtn, styles.secondary]} onPress={pickGallery}>
          <Text style={styles.btnTextDark}>{t.galleryBtn}</Text>
        </TouchableOpacity>
      </View>

      {images.length > 0 && (
        <View style={styles.previewCard}>
          <View style={styles.previewHeader}>
            <Text style={styles.pageBadge}>
              📄 {t.pageLabel} {activeIdx + 1} / {images.length} • {rotations[activeIdx] || 0}°
            </Text>
            <TouchableOpacity style={styles.removeMiniBtn} onPress={removeCurrentPage}>
              <Text style={styles.removeMiniText}>{t.removePageBtn}</Text>
            </TouchableOpacity>
          </View>

          <Image
            source={{ uri: images[activeIdx] || images[0] }}
            style={[styles.preview, { transform: [{ rotate: `${rotations[activeIdx] || 0}deg` }] }]}
            resizeMode="contain"
          />

          {images.length > 1 && (
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.thumbStrip}>
              {images.map((uri, idx) => (
                <TouchableOpacity
                  key={`${uri}-${idx}`}
                  style={[styles.thumbWrap, idx === activeIdx && styles.thumbActive]}
                  onPress={() => setActiveIdx(idx)}
                >
                  <Image
                    source={{ uri }}
                    style={[styles.thumbImg, { transform: [{ rotate: `${rotations[idx] || 0}deg` }] }]}
                    resizeMode="cover"
                  />
                  <Text style={[styles.thumbNum, idx === activeIdx && styles.thumbNumActive]}>
                    {idx + 1}{rotations[idx] ? ` (${rotations[idx]}°)` : ""}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          )}

          <View style={styles.previewTools}>
            <TouchableOpacity style={styles.toolBtn} onPress={rotateCurrent}>
              <Text style={styles.toolBtnText}>{t.rotateBtn}</Text>
            </TouchableOpacity>
            {images.length > 1 && (
              <TouchableOpacity style={styles.toolBtn} onPress={rotateAll}>
                <Text style={styles.toolBtnText}>{t.rotateAllBtn}</Text>
              </TouchableOpacity>
            )}
          </View>

          <TouchableOpacity style={[styles.toolBtn, styles.primary]} onPress={onScanPress} disabled={busy}>
            <Text style={styles.btnText}>{busy ? t.scanning : `${t.scanNowBtn} (${images.length})`}</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Subtle Voluntary Donate Coffee Modal */}
      <Modal visible={donateOpen} transparent animationType="fade">
        <View style={styles.modalBg}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>{t.donateTitle}</Text>
            <View style={styles.dedicationBox}>
              <Text style={styles.dedicationText}>{t.dedication}</Text>
            </View>
            <Text style={styles.modalBody}>{t.donateDesc}</Text>
            <View style={styles.contactBox}>
              <Text style={styles.contactLine}>👤 ຜູ້ພັດທະນາ: {DONATE_INFO.DEVELOPER_NAME}</Text>
              <Text style={styles.contactLine}>📘 Facebook: {DONATE_INFO.FACEBOOK_NAME}</Text>
              <Text style={styles.contactLine}>📱 WhatsApp: {DONATE_INFO.WHATSAPP_DISPLAY}</Text>
            </View>
            <View style={styles.qrBtnRow}>
              {(
                [
                  ["lak", "📱 QR ກີບ (LAK)"],
                  ["usd", "💵 QR USD"],
                  ["thb", "🇹🇭 QR ບາດ"]
                ] as ["lak" | "usd" | "thb", string][]
              ).map(([k, lbl]) => (
                <TouchableOpacity
                  key={k}
                  style={[styles.qrBtn, activeQr === k && styles.qrBtnActive]}
                  onPress={() => setActiveQr((prev) => (prev === k ? null : k))}
                >
                  <Text style={[styles.qrBtnTxt, activeQr === k && styles.qrBtnTxtActive]}>{lbl}</Text>
                </TouchableOpacity>
              ))}
            </View>
            {activeQr && (
              <View style={styles.qrPreviewBox}>
                <Image source={QR_IMAGES[activeQr]} style={styles.qrImg} resizeMode="contain" />
              </View>
            )}
            <TouchableOpacity style={[styles.actionBtn, styles.whatsapp]} onPress={() => Linking.openURL(DONATE_INFO.WHATSAPP_URL)}>
              <Text style={styles.btnText}>💬 WhatsApp: {DONATE_INFO.WHATSAPP_DISPLAY}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, styles.secondary]} onPress={() => { setActiveQr(null); setDonateOpen(false); }}>
              <Text style={styles.btnTextDark}>{t.closeBtn}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flexGrow: 1, backgroundColor: "#f8fafc", paddingBottom: 32 },
  header: { backgroundColor: "#1e3a8a", padding: 16, paddingTop: 48, flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 },
  title: { color: "#fff", fontSize: 19, fontWeight: "700" },
  sub: { color: "#bfdbfe", fontSize: 11.5, marginTop: 2 },
  langRow: { flexDirection: "row", backgroundColor: "rgba(255,255,255,0.2)", borderRadius: 8, padding: 2 },
  pill: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  pillActive: { backgroundColor: "#fff" },
  pillText: { color: "#fff", fontSize: 11, fontWeight: "700" },
  pillTextActive: { color: "#1e3a8a" },
  tierBar: { backgroundColor: "#f0fdf4", paddingHorizontal: 12, paddingVertical: 8, flexDirection: "row", justifyContent: "space-between", alignItems: "center", borderBottomWidth: 1, borderColor: "#dcfce7", gap: 8 },
  tierText: { flex: 1, fontSize: 11.5, fontWeight: "700", color: "#15803d" },
  donateMiniBtn: { backgroundColor: "#fffbeb", borderWidth: 1, borderColor: "#fcd34d", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  donateMiniText: { color: "#92400e", fontSize: 11, fontWeight: "700" },
  promoBox: { marginHorizontal: 16, marginTop: 12, backgroundColor: "#eff6ff", borderWidth: 1, borderColor: "#bfdbfe", borderRadius: 8, padding: 8, alignItems: "center" },
  promoText: { fontSize: 11.5, fontWeight: "700", color: "#1e3a8a", textAlign: "center" },
  modelRow: { flexDirection: "row", gap: 6, marginHorizontal: 16, marginTop: 10 },
  modelBtn: { flex: 1, paddingVertical: 8, borderRadius: 8, backgroundColor: "#fff", borderWidth: 1, borderColor: "#cbd5e1", alignItems: "center" },
  modelBtnActive: { backgroundColor: "#1d4ed8", borderColor: "#1d4ed8" },
  modelBtnText: { fontSize: 11.5, fontWeight: "700", color: "#334155" },
  modelBtnTextActive: { color: "#fff" },
  card: { padding: 16, gap: 10 },
  actionBtn: { padding: 14, borderRadius: 10, alignItems: "center" },
  primary: { backgroundColor: "#2563eb" },
  secondary: { backgroundColor: "#e2e8f0" },
  whatsapp: { backgroundColor: "#16a34a" },
  btnText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  btnTextDark: { color: "#0f172a", fontWeight: "700", fontSize: 15 },
  previewCard: { marginHorizontal: 16, backgroundColor: "#fff", borderRadius: 12, padding: 12, borderWidth: 1, borderColor: "#cbd5e1", gap: 10 },
  previewHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  pageBadge: { fontSize: 12.5, fontWeight: "700", color: "#1e3a8a" },
  removeMiniBtn: { backgroundColor: "#fee2e2", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  removeMiniText: { fontSize: 11.5, fontWeight: "700", color: "#b91c1c" },
  preview: { width: "100%", height: 240, borderRadius: 8, backgroundColor: "#f1f5f9" },
  thumbStrip: { flexDirection: "row", gap: 8, paddingVertical: 4 },
  thumbWrap: { width: 64, alignItems: "center", borderRadius: 8, borderWidth: 2, borderColor: "#e2e8f0", padding: 3, backgroundColor: "#f8fafc" },
  thumbActive: { borderColor: "#2563eb", backgroundColor: "#eff6ff" },
  thumbImg: { width: 54, height: 54, borderRadius: 5, backgroundColor: "#e2e8f0" },
  thumbNum: { fontSize: 10, fontWeight: "700", color: "#475569", marginTop: 2 },
  thumbNumActive: { color: "#1d4ed8" },
  previewTools: { flexDirection: "row", gap: 8 },
  toolBtn: { flex: 1, padding: 11, borderRadius: 8, backgroundColor: "#e2e8f0", alignItems: "center" },
  toolBtnText: { color: "#0f172a", fontWeight: "700", fontSize: 13.5 },
  modalBg: { flex: 1, backgroundColor: "rgba(15,23,42,0.55)", justifyContent: "center", padding: 20 },
  modal: { backgroundColor: "#fff", borderRadius: 14, padding: 18, gap: 10 },
  modalTitle: { fontSize: 17, fontWeight: "700", color: "#1e3a8a" },
  dedicationBox: { backgroundColor: "#fffbeb", borderWidth: 1, borderColor: "#fde68a", borderRadius: 8, padding: 10 },
  dedicationText: { fontSize: 12.5, color: "#78350f", lineHeight: 18 },
  modalBody: { fontSize: 13, color: "#475569", lineHeight: 19 },
  contactBox: { backgroundColor: "#f1f5f9", borderRadius: 8, padding: 10, gap: 4 },
  contactLine: { fontSize: 12.5, color: "#0f172a", fontWeight: "600" },
  qrBtnRow: { flexDirection: "row", gap: 6 },
  qrBtn: { flex: 1, paddingVertical: 8, paddingHorizontal: 4, borderRadius: 8, backgroundColor: "#e2e8f0", alignItems: "center" },
  qrBtnActive: { backgroundColor: "#1d4ed8" },
  qrBtnTxt: { fontSize: 11, fontWeight: "700", color: "#0f172a" },
  qrBtnTxtActive: { color: "#fff" },
  qrPreviewBox: { backgroundColor: "#fff", borderRadius: 10, borderWidth: 1, borderColor: "#cbd5e1", padding: 8, alignItems: "center" },
  qrImg: { width: 220, height: 260 }
});
