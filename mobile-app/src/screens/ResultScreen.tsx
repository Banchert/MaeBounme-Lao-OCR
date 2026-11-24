import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Image,
  Linking,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Clipboard from "expo-clipboard";
import * as FileSystem from "expo-file-system";
import * as Sharing from "expo-sharing";
import { TRANSLATIONS, UiLang } from "../i18n/translations";
import { exportDocx, exportMarkdown, exportTxt } from "../services/exporter";
import { findSuspectLaoWords, suggestLaoWords } from "../services/ocrEngine";
import { DONATE_INFO } from "../services/monetization";

interface Props {
  uiLang: UiLang;
  initialText: string;
  ms: number;
  avgConf: number;
  lineCount: number;
  images: string[];
  rotations: number[];
  onBack: () => void;
}

type CatKey = "main" | "num" | "en";
type ViewTab = "split" | "orig" | "ocr";
interface CatStyle {
  bold: boolean;
  italic: boolean;
}

const FEEDBACK_KEY = "@maebounmee_feedback_v1";
const ISSUE_TAGS = [
  "สระบน/วรรณยุกต์หาย",
  "สระล่าง (ຸ/ູ/ຼ) หาย",
  "ຂ / ຊ สลับกัน",
  "กระดาษเอียง/พับ/คลื่น",
  "คำศัพท์/การสะกดผิด",
  "อื่นๆ"
];

const FONT_OPTIONS: { label: string; value: string | undefined }[] = [
  { label: "Phetsarath OT", value: "PhetsarathOT" },
  { label: "System", value: undefined },
  { label: "Serif", value: "serif" },
  { label: "Mono", value: "monospace" }
];

export default function ResultScreen({
  uiLang,
  initialText,
  ms,
  avgConf,
  lineCount,
  images,
  rotations,
  onBack
}: Props) {
  const t = TRANSLATIONS[uiLang];
  const [text, setText] = useState(initialText);
  const [viewTab, setViewTab] = useState<ViewTab>(images.length > 0 ? "split" : "ocr");
  const [pageIdx, setPageIdx] = useState(0);
  const [imgZoom, setImgZoom] = useState(1);

  const [fontSize, setFontSize] = useState(17);
  const [fontIdx, setFontIdx] = useState(0);
  const [colorMode, setColorMode] = useState(false);
  const [catStyles, setCatStyles] = useState<Record<CatKey, CatStyle>>({
    main: { bold: false, italic: false },
    num: { bold: true, italic: false },
    en: { bold: false, italic: true }
  });
  const [copied, setCopied] = useState(false);

  // Keyboard-style word suggestion state
  const [cursorWord, setCursorWord] = useState<{ word: string; start: number; end: number } | null>(null);
  const [selectedSuspect, setSelectedSuspect] = useState<string | null>(null);

  // Feedback modal state
  const [fbOpen, setFbOpen] = useState(false);
  const [fbTags, setFbTags] = useState<string[]>([]);
  const [fbNote, setFbNote] = useState("");
  const [fbCount, setFbCount] = useState(0);

  useEffect(() => {
    AsyncStorage.getItem(FEEDBACK_KEY)
      .then((raw) => {
        if (raw) setFbCount((JSON.parse(raw) || []).length);
      })
      .catch(() => {});
  }, []);

  const toggleCat = (cat: CatKey, prop: "bold" | "italic") => {
    setCatStyles((prev) => ({
      ...prev,
      [cat]: { ...prev[cat], [prop]: !prev[cat][prop] }
    }));
  };

  const tokens = useMemo(() => {
    const out: { text: string; cat: CatKey }[] = [];
    const re = /([0-9໐-໙๐-๙]+(?:[.,:/\-][0-9໐-໙๐-๙]+)*)|([A-Za-z]+(?:['’\-][A-Za-z]+)*)/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push({ text: text.slice(last, m.index), cat: "main" });
      if (m[1]) out.push({ text: m[1], cat: "num" });
      else if (m[2]) out.push({ text: m[2], cat: "en" });
      last = re.lastIndex;
    }
    if (last < text.length) out.push({ text: text.slice(last), cat: "main" });
    return out;
  }, [text]);

  const suspects = useMemo(() => findSuspectLaoWords(text, 8), [text]);

  const activeWord = cursorWord?.word || selectedSuspect || (suspects[0]?.word ?? "");
  const activeSuggestions = useMemo(() => {
    if (!activeWord) return [];
    return suggestLaoWords(activeWord, 6);
  }, [activeWord]);

  const handleSelectionChange = (e: any) => {
    const pos: number = e?.nativeEvent?.selection?.start ?? 0;
    let s = pos;
    let en = pos;
    while (s > 0 && /[\u0E80-\u0EFF]/.test(text[s - 1])) s--;
    while (en < text.length && /[\u0E80-\u0EFF]/.test(text[en])) en++;
    const w = text.slice(s, en);
    if (w.length >= 2 && w.length <= 14) {
      setCursorWord({ word: w, start: s, end: en });
      setSelectedSuspect(null);
    } else {
      setCursorWord(null);
    }
  };

  const applySuggestion = (replacement: string) => {
    if (cursorWord && text.slice(cursorWord.start, cursorWord.end) === cursorWord.word) {
      const next = text.slice(0, cursorWord.start) + replacement + text.slice(cursorWord.end);
      setText(next);
      setCursorWord({ word: replacement, start: cursorWord.start, end: cursorWord.start + replacement.length });
    } else if (activeWord) {
      const idx = text.indexOf(activeWord);
      if (idx >= 0) {
        setText(text.slice(0, idx) + replacement + text.slice(idx + activeWord.length));
        setSelectedSuspect(null);
      }
    }
  };

  const toggleFbTag = (tag: string) => {
    setFbTags((prev) => (prev.includes(tag) ? prev.filter((x) => x !== tag) : [...prev, tag]));
  };

  const saveFeedbackOffline = async (): Promise<any> => {
    const entry = {
      ts: new Date().toISOString(),
      issues: fbTags.length ? fbTags : ["อื่นๆ"],
      note: fbNote.trim(),
      avgConf,
      lineCount,
      pages: images.length,
      ocrSnippet: text.slice(0, 600)
    };
    const raw = await AsyncStorage.getItem(FEEDBACK_KEY);
    const list = raw ? JSON.parse(raw) : [];
    list.push(entry);
    await AsyncStorage.setItem(FEEDBACK_KEY, JSON.stringify(list));
    setFbCount(list.length);
    Alert.alert(t.feedbackSavedOk, `รวมบันทึกไว้ในเครื่อง ${list.length} รายการ`);
    return entry;
  };

  const sendFeedbackWhatsApp = async () => {
    const entry = await saveFeedbackOffline();
    const msg =
      `[MaeBounmee OCR Feedback]\n` +
      `ปัญหา: ${entry.issues.join(", ")}\n` +
      `AI Conf: ${avgConf}% | ${lineCount} แถว\n` +
      (entry.note ? `หมายเหตุ: ${entry.note}\n` : "") +
      `ข้อความบางส่วน:\n${text.slice(0, 260)}`;
    Linking.openURL(`${DONATE_INFO.WHATSAPP_URL}?text=${encodeURIComponent(msg)}`);
  };

  const exportFeedbackJson = async () => {
    const raw = (await AsyncStorage.getItem(FEEDBACK_KEY)) || "[]";
    const uri = `${FileSystem.cacheDirectory}maebounmee_feedback_${Date.now()}.json`;
    await FileSystem.writeAsStringAsync(uri, raw, { encoding: FileSystem.EncodingType.UTF8 });
    await Sharing.shareAsync(uri, { mimeType: "application/json" });
  };

  const onCopy = async () => {
    await Clipboard.setStringAsync(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const sec = Math.max(0.01, ms / 1000);
  const cps = Math.round(text.length / sec);
  const activeFont = FONT_OPTIONS[fontIdx].value;

  const renderOriginalPane = (fullHeight: boolean) => {
    if (!images.length) return null;
    const curUri = images[pageIdx] || images[0];
    const curRot = rotations[pageIdx] || 0;
    return (
      <View style={[styles.origPane, fullHeight ? { flex: 1 } : { height: 220 }]}>
        <View style={styles.origBar}>
          <Text style={styles.origTitle}>
            🖼️ {t.pageLabel || "หน้า"} {pageIdx + 1}/{images.length} ({curRot}°)
          </Text>
          <View style={{ flexDirection: "row", gap: 4 }}>
            {images.length > 1 && (
              <>
                <TouchableOpacity
                  style={styles.miniChip}
                  onPress={() => setPageIdx((p) => (p - 1 + images.length) % images.length)}
                >
                  <Text style={styles.miniChipTxt}>◀</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.miniChip}
                  onPress={() => setPageIdx((p) => (p + 1) % images.length)}
                >
                  <Text style={styles.miniChipTxt}>▶</Text>
                </TouchableOpacity>
              </>
            )}
            <TouchableOpacity
              style={styles.miniChip}
              onPress={() => setImgZoom((z) => (z >= 2 ? 1 : +(z + 0.5).toFixed(1)))}
            >
              <Text style={styles.miniChipTxt}>🔍 {imgZoom}x</Text>
            </TouchableOpacity>
          </View>
        </View>
        <ScrollView horizontal contentContainerStyle={{ flexGrow: 1, justifyContent: "center" }}>
          <ScrollView contentContainerStyle={{ flexGrow: 1, justifyContent: "center", alignItems: "center" }}>
            <Image
              source={{ uri: curUri }}
              style={{
                width: Math.round(330 * imgZoom),
                height: Math.round((fullHeight ? 420 : 180) * imgZoom),
                transform: [{ rotate: `${curRot}deg` }]
              }}
              resizeMode="contain"
            />
          </ScrollView>
        </ScrollView>
      </View>
    );
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack}>
          <Text style={styles.back}>{t.backBtn}</Text>
        </TouchableOpacity>
        <Text style={styles.title} numberOfLines={1}>{t.resultTitle}</Text>
        <View style={styles.fontRow}>
          <TouchableOpacity style={styles.fontBtn} onPress={() => setFontSize((s) => Math.max(12, s - 2))}>
            <Text style={styles.fontBtnTxt}>A-</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.fontBtn} onPress={() => setFontSize((s) => Math.min(28, s + 2))}>
            <Text style={styles.fontBtnTxt}>A+</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.fontBtn, { backgroundColor: "#f59e0b" }]} onPress={() => setFbOpen(true)}>
            <Text style={styles.fontBtnTxt}>🐛{fbCount > 0 ? ` ${fbCount}` : ""}</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* 3-Mode Workspace Switcher (Matches Chrome Extension!) */}
      {images.length > 0 && (
        <View style={styles.tabRow}>
          {(
            [
              ["split", t.tabSplit],
              ["orig", t.tabOrig],
              ["ocr", t.tabOcr]
            ] as [ViewTab, string][]
          ).map(([tab, lbl]) => (
            <TouchableOpacity
              key={tab}
              style={[styles.tabBtn, viewTab === tab && styles.tabBtnActive]}
              onPress={() => setViewTab(tab)}
            >
              <Text style={[styles.tabBtnTxt, viewTab === tab && styles.tabBtnTxtActive]} numberOfLines={1}>
                {lbl}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      <View style={styles.perfRow}>
        <Text style={styles.perfPill}>⚡ {sec.toFixed(2)}s</Text>
        <Text style={styles.perfPill}>🚀 {cps} อักษร/วิ</Text>
        <Text style={styles.perfPill}>🎯 AI {avgConf}%</Text>
        <Text style={styles.perfPill}>📄 {lineCount} แถว</Text>
      </View>

      {(viewTab === "split" || viewTab === "orig") && renderOriginalPane(viewTab === "orig")}

      {viewTab !== "orig" && (
        <>
          {/* Font Picker + Color Proofread / Edit Toggle + Category Bold/Italic */}
          <View style={styles.styleBar}>
            <View style={styles.styleTopRow}>
              <TouchableOpacity
                style={[styles.modePill, colorMode && styles.modePillActive]}
                onPress={() => setColorMode((v) => !v)}
              >
                <Text style={[styles.modePillTxt, colorMode && styles.modePillTxtActive]}>
                  {colorMode ? "🎨 ตรวจสี" : "✏️ แก้ไข"}
                </Text>
              </TouchableOpacity>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 5 }}>
                {FONT_OPTIONS.map((f, idx) => (
                  <TouchableOpacity
                    key={f.label}
                    style={[styles.fontChip, fontIdx === idx && styles.fontChipActive]}
                    onPress={() => setFontIdx(idx)}
                  >
                    <Text style={[styles.fontChipTxt, fontIdx === idx && styles.fontChipTxtActive]}>{f.label}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>
            <View style={styles.catRow}>
              {(
                [
                  ["main", "ลาว/ไทย", "#0f172a"],
                  ["num", "123", "#c2410c"],
                  ["en", "ABC", "#1d4ed8"]
                ] as [CatKey, string, string][]
              ).map(([cat, lbl, col]) => (
                <View key={cat} style={styles.catGroup}>
                  <Text style={[styles.catLabel, { color: col }]}>{lbl}</Text>
                  <TouchableOpacity
                    style={[styles.fmtBtn, catStyles[cat].bold && styles.fmtBtnActive]}
                    onPress={() => toggleCat(cat, "bold")}
                  >
                    <Text style={[styles.fmtBtnTxt, catStyles[cat].bold && styles.fmtBtnTxtActive]}>B</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.fmtBtn, catStyles[cat].italic && styles.fmtBtnActive]}
                    onPress={() => toggleCat(cat, "italic")}
                  >
                    <Text style={[styles.fmtBtnTxt, { fontStyle: "italic" }, catStyles[cat].italic && styles.fmtBtnTxtActive]}>I</Text>
                  </TouchableOpacity>
                </View>
              ))}
            </View>
          </View>

          {/* Mobile Keyboard-Style Predictive Word Suggestion Bar */}
          {(activeSuggestions.length > 0 || suspects.length > 0) && (
            <View style={styles.suggestBar}>
              <Text style={styles.suggestTitle}>
                {t.suggestBar} {activeWord ? `「${activeWord}」` : ""}
              </Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 6 }}>
                {activeSuggestions.map((s) => (
                  <TouchableOpacity key={s} style={styles.sugChip} onPress={() => applySuggestion(s)}>
                    <Text style={[styles.sugChipTxt, { fontFamily: activeFont }]}>{s}</Text>
                  </TouchableOpacity>
                ))}
                {suspects.length > 1 &&
                  suspects.slice(0, 5).map((item) => (
                    <TouchableOpacity
                      key={item.word}
                      style={[styles.suspectChip, activeWord === item.word && styles.suspectChipActive]}
                      onPress={() => {
                        setCursorWord(null);
                        setSelectedSuspect(item.word);
                      }}
                    >
                      <Text style={styles.suspectChipTxt}>? {item.word}</Text>
                    </TouchableOpacity>
                  ))}
              </ScrollView>
            </View>
          )}

          {colorMode ? (
            <ScrollView style={styles.richBox} contentContainerStyle={{ padding: 14 }}>
              <Text selectable style={{ fontFamily: activeFont, fontSize, lineHeight: Math.round(fontSize * 1.65) }}>
                {tokens.map((tk, i) => {
                  const st = catStyles[tk.cat];
                  const color = tk.cat === "num" ? "#c2410c" : tk.cat === "en" ? "#1d4ed8" : "#0f172a";
                  return (
                    <Text
                      key={i}
                      style={{
                        color,
                        fontWeight: st.bold ? "700" : "400",
                        fontStyle: st.italic ? "italic" : "normal"
                      }}
                    >
                      {tk.text}
                    </Text>
                  );
                })}
              </Text>
            </ScrollView>
          ) : (
            <TextInput
              style={[
                styles.editor,
                {
                  fontFamily: activeFont,
                  fontSize,
                  lineHeight: Math.round(fontSize * 1.65),
                  fontWeight: catStyles.main.bold ? "700" : "400",
                  fontStyle: catStyles.main.italic ? "italic" : "normal"
                }
              ]}
              multiline
              value={text}
              onChangeText={setText}
              onSelectionChange={handleSelectionChange}
              textAlignVertical="top"
            />
          )}
        </>
      )}

      <View style={styles.actions}>
        <View style={styles.row}>
          <TouchableOpacity style={[styles.btn, styles.green, { flex: 2 }]} onPress={onCopy}>
            <Text style={styles.btnText}>{copied ? t.copied : t.copyBtn}</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, styles.amber, { flex: 1 }]} onPress={() => setFbOpen(true)}>
            <Text style={styles.btnTextDark}>{t.feedbackBtn}</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.row}>
          <TouchableOpacity style={[styles.btn, styles.outline, { flex: 1 }]} onPress={() => exportTxt(text)}>
            <Text style={styles.btnTextDark}>💾 .txt</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, styles.outline, { flex: 1 }]} onPress={() => exportMarkdown(text)}>
            <Text style={styles.btnTextDark}>🧾 .md</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, styles.blue, { flex: 1.2 }]} onPress={() => exportDocx(text)}>
            <Text style={styles.btnText}>📘 .docx</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Offline-First User Feedback / Issue Report Modal */}
      <Modal visible={fbOpen} transparent animationType="fade">
        <View style={styles.modalBg}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>{t.feedbackTitle}</Text>
            <Text style={styles.modalSub}>{t.feedbackDesc}</Text>
            <View style={styles.tagGrid}>
              {ISSUE_TAGS.map((tag) => {
                const on = fbTags.includes(tag);
                return (
                  <TouchableOpacity
                    key={tag}
                    style={[styles.issueTag, on && styles.issueTagOn]}
                    onPress={() => toggleFbTag(tag)}
                  >
                    <Text style={[styles.issueTagTxt, on && styles.issueTagTxtOn]}>
                      {on ? "✓ " : ""}{tag}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
            <TextInput
              style={styles.fbInput}
              placeholder="พิมพ์คำที่อ่านผิด หรือรายละเอียดเพิ่มเติม..."
              multiline
              value={fbNote}
              onChangeText={setFbNote}
            />
            <View style={styles.row}>
              <TouchableOpacity style={[styles.btn, styles.blue, { flex: 1 }]} onPress={saveFeedbackOffline}>
                <Text style={styles.btnText}>{t.feedbackSave}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.btn, styles.green, { flex: 1 }]} onPress={sendFeedbackWhatsApp}>
                <Text style={styles.btnText}>{t.feedbackSend}</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.row}>
              <TouchableOpacity style={[styles.btn, styles.outline, { flex: 1 }]} onPress={exportFeedbackJson}>
                <Text style={styles.btnTextDark}>{t.feedbackExport} ({fbCount})</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.btn, styles.outline, { flex: 1 }]} onPress={() => setFbOpen(false)}>
                <Text style={styles.btnTextDark}>{t.closeBtn}</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: "#f8fafc" },
  header: { backgroundColor: "#1e3a8a", padding: 12, paddingTop: 46, flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 6 },
  back: { color: "#bfdbfe", fontSize: 13.5, fontWeight: "700" },
  title: { flex: 1, color: "#fff", fontSize: 13.5, fontWeight: "700", textAlign: "center" },
  fontRow: { flexDirection: "row", gap: 4 },
  fontBtn: { backgroundColor: "rgba(255,255,255,0.22)", paddingHorizontal: 7, paddingVertical: 4, borderRadius: 6 },
  fontBtnTxt: { color: "#fff", fontSize: 11.5, fontWeight: "700" },
  tabRow: { flexDirection: "row", backgroundColor: "#e2e8f0", padding: 4, gap: 4 },
  tabBtn: { flex: 1, paddingVertical: 6, borderRadius: 6, alignItems: "center" },
  tabBtnActive: { backgroundColor: "#fff" },
  tabBtnTxt: { fontSize: 11.5, fontWeight: "700", color: "#475569" },
  tabBtnTxtActive: { color: "#1d4ed8" },
  perfRow: { flexDirection: "row", justifyContent: "space-around", backgroundColor: "#f0fdf4", borderBottomWidth: 1, borderColor: "#dcfce7", paddingVertical: 5 },
  perfPill: { fontSize: 11, fontWeight: "700", color: "#15803d" },
  origPane: { backgroundColor: "#0f172a", borderBottomWidth: 1, borderColor: "#334155" },
  origBar: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 10, paddingVertical: 5, backgroundColor: "#1e293b" },
  origTitle: { color: "#e2e8f0", fontSize: 11.5, fontWeight: "700" },
  miniChip: { backgroundColor: "#334155", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 5 },
  miniChipTxt: { color: "#fff", fontSize: 11, fontWeight: "700" },
  styleBar: { backgroundColor: "#fff", borderBottomWidth: 1, borderColor: "#e2e8f0", paddingHorizontal: 10, paddingVertical: 6, gap: 5 },
  styleTopRow: { flexDirection: "row", alignItems: "center", gap: 6 },
  modePill: { borderWidth: 1, borderColor: "#cbd5e1", backgroundColor: "#f8fafc", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  modePillActive: { backgroundColor: "#eff6ff", borderColor: "#2563eb" },
  modePillTxt: { fontSize: 11, fontWeight: "700", color: "#334155" },
  modePillTxtActive: { color: "#1d4ed8" },
  fontChip: { borderWidth: 1, borderColor: "#cbd5e1", backgroundColor: "#f8fafc", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  fontChipActive: { backgroundColor: "#1e3a8a", borderColor: "#1e3a8a" },
  fontChipTxt: { fontSize: 11, fontWeight: "600", color: "#334155" },
  fontChipTxtActive: { color: "#fff" },
  catRow: { flexDirection: "row", justifyContent: "space-between", gap: 6 },
  catGroup: { flexDirection: "row", alignItems: "center", gap: 3, backgroundColor: "#f8fafc", borderWidth: 1, borderColor: "#e2e8f0", borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
  catLabel: { fontSize: 10.5, fontWeight: "700", marginRight: 2 },
  fmtBtn: { width: 22, height: 22, borderRadius: 4, borderWidth: 1, borderColor: "#cbd5e1", backgroundColor: "#fff", alignItems: "center", justifyContent: "center" },
  fmtBtnActive: { backgroundColor: "#1d4ed8", borderColor: "#1d4ed8" },
  fmtBtnTxt: { fontSize: 11, fontWeight: "800", color: "#334155" },
  fmtBtnTxtActive: { color: "#fff" },
  suggestBar: { backgroundColor: "#eff6ff", borderBottomWidth: 1, borderColor: "#bfdbfe", paddingHorizontal: 10, paddingVertical: 6, gap: 4 },
  suggestTitle: { fontSize: 11, fontWeight: "700", color: "#1e40af" },
  sugChip: { backgroundColor: "#2563eb", paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999 },
  sugChipTxt: { color: "#fff", fontSize: 13, fontWeight: "700" },
  suspectChip: { backgroundColor: "#fef3c7", borderWidth: 1, borderColor: "#f59e0b", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999 },
  suspectChipActive: { backgroundColor: "#fde68a", borderColor: "#b45309" },
  suspectChipTxt: { color: "#92400e", fontSize: 11.5, fontWeight: "700" },
  richBox: { flex: 1, margin: 10, backgroundColor: "#fff", borderRadius: 10, borderWidth: 1, borderColor: "#93c5fd" },
  editor: { flex: 1, margin: 10, padding: 12, backgroundColor: "#fff", borderRadius: 10, borderWidth: 1, borderColor: "#cbd5e1", color: "#0f172a" },
  actions: { padding: 10, paddingBottom: 24, gap: 8 },
  row: { flexDirection: "row", gap: 8 },
  btn: { padding: 11, borderRadius: 9, alignItems: "center" },
  green: { backgroundColor: "#059669" },
  blue: { backgroundColor: "#2563eb" },
  amber: { backgroundColor: "#fef3c7", borderWidth: 1, borderColor: "#f59e0b" },
  outline: { backgroundColor: "#fff", borderWidth: 1, borderColor: "#cbd5e1" },
  btnText: { color: "#fff", fontWeight: "700", fontSize: 13 },
  btnTextDark: { color: "#0f172a", fontWeight: "700", fontSize: 13 },
  modalBg: { flex: 1, backgroundColor: "rgba(15,23,42,0.55)", justifyContent: "center", padding: 18 },
  modal: { backgroundColor: "#fff", borderRadius: 14, padding: 16, gap: 10 },
  modalTitle: { fontSize: 16, fontWeight: "700", color: "#1e3a8a" },
  modalSub: { fontSize: 12.5, color: "#475569" },
  tagGrid: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  issueTag: { borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 999, paddingHorizontal: 10, paddingVertical: 6, backgroundColor: "#f8fafc" },
  issueTagOn: { backgroundColor: "#dbeafe", borderColor: "#2563eb" },
  issueTagTxt: { fontSize: 12, color: "#334155", fontWeight: "600" },
  issueTagTxtOn: { color: "#1d4ed8", fontWeight: "700" },
  fbInput: { borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 8, padding: 10, minHeight: 72, textAlignVertical: "top", fontSize: 13 }
});
