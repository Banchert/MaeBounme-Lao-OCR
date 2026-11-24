const ocr = new OfflineLaoOCR();
const $ = (id) => document.getElementById(id);

const statusBadge = $("statusBadge");
const quotaBadge = $("quotaBadge");
const openProBtn = $("openProBtn");
const snipBtn = $("snipBtn");
const fileInput = $("fileInput");
const dropZone = $("dropZone");
const dropHint = $("dropHint");
const previewImg = $("previewImg");
const togglePreviewBtn = $("togglePreviewBtn");
const zoomPreviewBtn = $("zoomPreviewBtn");
const zoomModal = $("zoomModal");
const zoomImg = $("zoomImg");
const zoomVal = $("zoomVal");
const zoomInBtn = $("zoomInBtn");
const zoomOutBtn = $("zoomOutBtn");
const zoomFitBtn = $("zoomFitBtn");
const closeZoomBtn = $("closeZoomBtn");
const langSelect = $("langSelect");
const layoutSelect = $("layoutSelect");
const pageBar = $("pageBar");
const pageSelect = $("pageSelect");
const pageRangeInput = $("pageRangeInput");
const scanNowBtn = $("scanNowBtn");
const a4Toggle = $("a4Toggle");
const fontDecBtn = $("fontDecBtn");
const fontIncBtn = $("fontIncBtn");
const fontSizeVal = $("fontSizeVal");
const progressWrap = $("progressWrap");
const progressFill = $("progressFill");
const progressText = $("progressText");
const cancelScanBtn = $("cancelScanBtn");
let cancelRequested = false;
const outputText = $("outputText");
const richPreview = $("richPreview");
const workspaceGrid = $("workspaceGrid");
const origGallery = $("origGallery");
const origZoomOut = $("origZoomOut");
const origZoomIn = $("origZoomIn");
const origZoomFit = $("origZoomFit");
const origZoomVal = $("origZoomVal");
const origFullBtn = $("origFullBtn");
const viewModeBtn = $("viewModeBtn");
const fontSelect = $("fontSelect");
const metaInfo = $("metaInfo");
const livePerf = $("livePerf");
const perfTime = $("perfTime");
const perfSpeed = $("perfSpeed");
const perfConf = $("perfConf");
const promoTags = $("promoTags");
const copyBtn = $("copyBtn");
const downloadBtn = $("downloadBtn");
const downloadWordBtn = $("downloadWordBtn");
const clearBtn = $("clearBtn");
const toastEl = $("toast");

let toastTimer = null;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toastEl.hidden = true), 1800);
}

const proModal = $("proModal");
const creditsModal = $("creditsModal");
const openCreditsBtn = $("openCreditsBtn");
const closeCreditsBtn = $("closeCreditsBtn");
const supportLinkBtn = $("supportLinkBtn");
const closeProBtn = $("closeProBtn");

let uiLang = "lo";
let lastCanvases = [];
let lastPageResults = []; // [{ items, pageW }] for instant 1ms layout re-formatting!
let outputFontSize = 16;
let colorProofreadMode = true;
let previewCollapsed = false;
let zoomPct = 100;
const catStyles = {
  main: { bold: false, italic: false },
  num: { bold: true, italic: false },
  en: { bold: false, italic: true }
};

function styleSpanCss(cat) {
  const s = catStyles[cat];
  const color = cat === "num" ? "#c2410c" : cat === "en" ? "#1d4ed8" : "#0f172a";
  return `color:${color};font-weight:${s.bold ? "700" : "400"};font-style:${s.italic ? "italic" : "normal"};`;
}

function colorizeTextToHtml(raw) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // Split into Numbers (Arabic + Lao + Thai digits), English words, and Main (Lao/Thai/other)
  const re = /([0-9໐-໙๐-๙]+(?:[.,:/\-][0-9໐-໙๐-๙]+)*)|([A-Za-z]+(?:['’\-][A-Za-z]+)*)/g;
  let html = "";
  let last = 0;
  let m;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > last) {
      html += `<span style="${styleSpanCss("main")}">${esc(raw.slice(last, m.index))}</span>`;
    }
    if (m[1]) {
      html += `<span class="tok-num" style="${styleSpanCss("num")}">${esc(m[1])}</span>`;
    } else if (m[2]) {
      html += `<span class="tok-en" style="${styleSpanCss("en")}">${esc(m[2])}</span>`;
    }
    last = re.lastIndex;
  }
  if (last < raw.length) {
    html += `<span style="${styleSpanCss("main")}">${esc(raw.slice(last))}</span>`;
  }
  return html || `<span style="color:#94a3b8">${esc(outputText.placeholder)}</span>`;
}

function applyTypographyAndPreview() {
  const fam = fontSelect.value;
  outputText.style.fontFamily = fam;
  outputText.style.fontSize = outputFontSize + "px";
  outputText.style.fontWeight = catStyles.main.bold ? "700" : "400";
  outputText.style.fontStyle = catStyles.main.italic ? "italic" : "normal";
  richPreview.style.fontFamily = fam;
  richPreview.style.fontSize = outputFontSize + "px";
  richPreview.innerHTML = colorizeTextToHtml(outputText.value);

  const showRich = colorProofreadMode && outputText.value.length > 0;
  richPreview.hidden = !showRich;
  outputText.hidden = showRich;
  viewModeBtn.classList.toggle("active", colorProofreadMode);
  const vm = {
    lo: colorProofreadMode ? "🎨 ກວດສີ" : "✏️ ແກ້ໄຂ",
    th: colorProofreadMode ? "🎨 ตรวจสี" : "✏️ แก้ไข",
    en: colorProofreadMode ? "🎨 Color" : "✏️ Edit"
  };
  viewModeBtn.textContent = vm[uiLang] || vm.lo;
}

function setOutputFontSize(sz) {
  outputFontSize = Math.max(12, Math.min(28, sz));
  fontSizeVal.textContent = outputFontSize + "px";
  applyTypographyAndPreview();
  chrome.storage.local.set({ mbm_font_sz: outputFontSize }).catch(() => {});
}

function renderFormattedOutput() {
  if (!lastPageResults.length) return;
  const mode = layoutSelect.value;
  const totalPool = lastCanvases.length || lastPageResults.length;
  const multi = lastPageResults.length > 1 || totalPool > 1;
  const parts = lastPageResults.map((r, i) => {
    const txt = ocr.formatLayout(r.items, r.pageW, mode) || t("noText");
    return multi ? `===== Page ${r.pageNum || (i + 1)} / ${totalPool} =====\n${txt}` : txt;
  });
  outputText.value = parts.join("\n\n");
  applyTypographyAndPreview();
}

function t(key, ...args) {
  const pack = I18N[uiLang] || I18N.lo;
  const val = pack[key];
  return typeof val === "function" ? val(...args) : val;
}

function applyUiLang(lang) {
  uiLang = I18N[lang] ? lang : "lo";
  document.querySelectorAll(".lang-pill").forEach((b) => {
    b.classList.toggle("active", b.dataset.uilang === uiLang);
  });
  $("uiSubtitle").textContent = t("subtitle");
  snipBtn.textContent = "✂️";
  snipBtn.title = t("snipBtn");
  const uploadMap = {
    lo: "📁 ອັບໂຫຼດ / ລາກວາງ ຮູບພາບ & PDF",
    th: "📁 อัปโหลด / ลากวาง รูปภาพ & PDF",
    en: "📁 Upload / Drag & Drop Images & PDF"
  };
  $("uploadLabel").textContent = uploadMap[uiLang] || uploadMap.lo;
  $("dropTitle").textContent = uiLang === "en"
    ? "Click or drag Images / PDF here, or press Ctrl+V"
    : uiLang === "th"
      ? "คลิกหรือลากวาง รูปภาพ (ทุกสกุล) / ไฟล์ PDF หรือกด Ctrl+V"
      : "ຄລິກ ຫຼື ລາກວາງ ຮູບພາບ (ທຸກສະກຸນ) / ໄຟລ໌ PDF ຫຼື ກົດ Ctrl+V";
  $("dropSub").textContent = t("dropSub");
  $("perfTimeLbl").textContent = uiLang === "en" ? "⚡ Speed" : uiLang === "th" ? "⚡ ความเร็ว" : "⚡ ຄວາມໄວ";
  $("perfSpeedLbl").textContent = uiLang === "en" ? "🚀 Rate" : uiLang === "th" ? "🚀 อัตราอ่าน" : "🚀 ອັດຕາອ່ານ";
  $("perfConfLbl").textContent = uiLang === "en" ? "🎯 AI Conf" : uiLang === "th" ? "🎯 ความมั่นใจ AI" : "🎯 ຄວາມໝັ້ນໃຈ AI";
  $("catMainLbl").textContent = uiLang === "en" ? "Lao/Thai" : uiLang === "th" ? "ลาว/ไทย" : "ລາວ/ໄທ";
  $("wsSplitBtn").textContent = uiLang === "en" ? "📑 Original + OCR" : uiLang === "th" ? "📑 ต้นฉบับ + OCR" : "📑 ຕົ້ນສະບັບ + OCR";
  $("wsOrigBtn").textContent = uiLang === "en" ? "🖼️ Original" : uiLang === "th" ? "🖼️ เอกสารเดิม" : "🖼️ ເອກະສານເດີມ";
  $("wsOcrBtn").textContent = uiLang === "en" ? "📝 OCR Text" : uiLang === "th" ? "📝 เอกสารใหม่" : "📝 ເອກະສານໃໝ່";
  $("origPaneHeader").textContent = uiLang === "en" ? "🖼️ Original Document" : uiLang === "th" ? "🖼️ รูปต้นแบบ" : "🖼️ ຮູບຕົ້ນແບບ";
  $("previewBarTitle").textContent = uiLang === "en"
    ? "🖼️ Preview (drag bottom-right corner to resize)"
    : uiLang === "th"
      ? "🖼️ ตัวอย่างต้นฉบับ (ลากมุมขวาล่างเพื่อปรับขนาด)"
      : "🖼️ ຕົວຢ່າງຕົ້ນສະບັບ (ລາກມຸມຂວາລຸ່ມເພື່ອປັບຂະໜາດ)";
  zoomPreviewBtn.textContent = uiLang === "en" ? "🔍 Zoom Original" : uiLang === "th" ? "🔍 ขยายต้นฉบับ" : "🔍 ຂະຫຍາຍຕົ້ນສະບັບ";
  togglePreviewBtn.textContent = previewCollapsed
    ? (uiLang === "en" ? "🔽 Expand" : uiLang === "th" ? "🔽 ขยาย" : "🔽 ຂະຫຍາຍ")
    : (uiLang === "en" ? "🔼 Collapse" : uiLang === "th" ? "🔼 ย่อ" : "🔼 ຫຍໍ້");
  openCreditsBtn.textContent = uiLang === "en" ? "ℹ️ Model Info" : uiLang === "th" ? "ℹ️ รายละเอียดโมเดล" : "ℹ️ ລາຍລະອຽດໂມເດລ";
  updateScanBtnLabel();
  applyTypographyAndPreview();
  $("outputTitle").textContent = t("outputTitle");
  outputText.placeholder = t("outputPlaceholder");
  copyBtn.textContent = t("copyBtn");
  downloadBtn.textContent = "📄 .txt";
  downloadWordBtn.textContent = "📘 Word";
  clearBtn.textContent = "🧹";
  openProBtn.textContent = "☕ Donate";
  const bannerMap = {
    lo: "🕊️ ຟຣີ 100% ບໍ່ຈຳກັດ • Open-Source (ອຸທິດແດ່ ແມ່ບຸນມີ)",
    th: "🕊️ ฟรี 100% ไม่จำกัด • Open-Source (อุทิศแด่ แม่บุญมี)",
    en: "🕊️ 100% Free & Unlimited • Open-Source (In Memory of Mae Bounme)"
  };
  const dedMap = {
    lo: "🙏 ໂປຣແກຣມ ແລະ ໂມເດລ AI ນີ້ເປີດໃຫ້ໃຊ້ຟຣີ 100% ບໍ່ຈຳກັດ (Apache-2.0) ເພື່ອເປັນວິທະຍາທານ ແລະ ອຸທິດສ່ວນກຸສົນແດ່ <b>ແມ່ບຸນມີ</b> (ມານດາຜູ້ລ່ວງລັບ)",
    th: "🙏 โปรแกรมและโมเดล AI นี้เปิดให้ใช้ฟรี 100% ไม่จำกัด (Apache-2.0) เพื่อเป็นวิทยาทานและอุทิศส่วนกุศลแด่ <b>แม่บุญมี</b> (มารดาผู้ล่วงลับ)",
    en: "🙏 This tool and AI model are 100% Free & Open-Source (Apache-2.0), dedicated as a public gift in loving memory of <b>Mae Bounme</b>."
  };
  const donateDescMap = {
    lo: "ຫາກໂປຣແກຣມນີ້ເປັນປະໂຫຍດ ທ່ານສາມາດຮ່ວມສະໜັບສະໜູນຄ່າກາແຟ ຫຼື ຄ່າພັດທະນາໄດ້ຕາມສັດທາ:",
    th: "หากโปรแกรมนี้เป็นประโยชน์ สามารถร่วมสนับสนุนค่ากาแฟหรือค่าพัฒนาได้ตามศรัทธา:",
    en: "If this tool helps you, voluntary coffee donations are warmly appreciated:"
  };
  const promoMap = {
    lo: ["⚡ ໄວ Offline 100%", "🎯 ແມ່ນຍຳ 99.9%", "📚 ບໍ່ຈຳກັດເອກະສານ (A4/ໜັງສື/ຕາຕະລາງ/ລາຍມື)"],
    th: ["⚡ เร็ว Offline 100%", "🎯 แม่นยำสูงสุด 99.9%", "📚 ไม่จำกัดเอกสาร (A4/หนังสือ/ตาราง/ภาพถ่าย/ลายมือ)"],
    en: ["⚡ Fast 100% Offline", "🎯 Up to 99.9% Accuracy", "📚 Any Document (A4 / Books / Tables / Handwriting)"]
  };
  promoTags.innerHTML = (promoMap[uiLang] || promoMap.lo)
    .map((s) => `<span class="promo-pill">${s}</span>`)
    .join("");
  quotaBadge.textContent = bannerMap[uiLang] || bannerMap.lo;
  quotaBadge.className = "quota-text pro";
  openProBtn.hidden = false;
  $("proTitle").textContent = uiLang === "en" ? "☕ Buy Me a Coffee (Donate)" : uiLang === "th" ? "☕ สนับสนุนค่ากาแฟ (Donate)" : "☕ ສະໜັບສະໜູນຄ່າກາແຟ (Donate)";
  $("dedicationBox").innerHTML = dedMap[uiLang] || dedMap.lo;
  $("proDesc").textContent = donateDescMap[uiLang] || donateDescMap.lo;
  $("proContact").innerHTML =
    `<b>👤 ຜູ້ພັດທະນາ:</b> ${LICENSING.DEVELOPER_NAME}<br/>` +
    `<b>📘 Facebook:</b> ${LICENSING.FACEBOOK_NAME}<br/>` +
    `<b>📱 WhatsApp:</b> ${LICENSING.WHATSAPP_DISPLAY}`;
  supportLinkBtn.textContent = `💬 WhatsApp: ${LICENSING.WHATSAPP_DISPLAY}`;
  closeProBtn.textContent = t("proCloseBtn");
  if (ocr.ready) setStatus(t("ready"), "ready");
}

function setStatus(text, cls) {
  statusBadge.textContent = text;
  statusBadge.className = "badge " + cls;
}

function openProModal() {
  proModal.hidden = false;
}

// Init UI language + AI models
(async () => {
  const st = await LICENSING.getState();
  applyUiLang(st.uiLang || "lo");
  setStatus(t("loading"), "loading");
  try {
    await ocr.init((msg) => setStatus(msg, "loading"));
    setStatus(t("ready"), "ready");
    snipBtn.disabled = false;
  } catch (err) {
    setStatus(t("error"), "error");
    outputText.value = "Error loading AI models: " + err.message;
  }
})();

document.querySelectorAll(".lang-pill").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const lang = btn.dataset.uilang;
    await LICENSING.setUiLang(lang);
    applyUiLang(lang);
  });
});

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Cannot load image"));
    img.src = src;
  });
}

async function blobToCanvas(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const img = await loadImageElement(url);
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext("2d").drawImage(img, 0, 0);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function populatePageSelector(canvases) {
  pageSelect.innerHTML = "";
  if (canvases.length > 1) {
    const optAll = document.createElement("option");
    optAll.value = "all";
    optAll.textContent = `ทุกหน้า / ທຸກໜ້າ (${canvases.length})`;
    pageSelect.appendChild(optAll);
  }
  canvases.forEach((_, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = t("pageLabel", idx + 1, canvases.length);
    pageSelect.appendChild(opt);
  });
  pageBar.hidden = canvases.length === 0;
}

function updateScanBtnLabel() {
  const n = getSelectedCanvases().length;
  scanNowBtn.disabled = n === 0;
  const base = uiLang === "en" ? "🚀 Start Scan OCR" : uiLang === "th" ? "🚀 เริ่มสแกน OCR" : "🚀 ເລີ່ມສະແກນ OCR";
  scanNowBtn.textContent = n > 0 ? `${base} (${n})` : base;
}

let origZoomPct = 100;

function applyOrigZoom(pct) {
  origZoomPct = Math.max(50, Math.min(400, pct));
  origZoomVal.textContent = origZoomPct + "%";
  origGallery.querySelectorAll(".orig-card").forEach((card) => {
    card.style.width = origZoomPct + "%";
  });
}

function renderOriginalGallery(entries) {
  origGallery.innerHTML = "";
  origZoomPct = 100;
  origZoomVal.textContent = "100%";
  if (!entries.length) {
    origGallery.innerHTML = `<div class="orig-empty">📄 ${uiLang === "en" ? "No document loaded" : uiLang === "th" ? "ยังไม่มีเอกสาร" : "ຍັງບໍ່ມີເອກະສານ"}</div>`;
    return;
  }
  const totalPool = lastCanvases.length || entries.length;
  const maxShow = 15;
  entries.slice(0, maxShow).forEach((ent) => {
    const cv = ent.canvas || ent;
    const pNum = ent.pageNum || 1;
    const card = document.createElement("div");
    card.className = "orig-card";
    const src = cv.toDataURL("image/jpeg", 0.72);
    card.innerHTML = `<div class="orig-card-num">Page ${pNum} / ${totalPool}</div><img src="${src}" alt="Page ${pNum}" draggable="false" />`;
    card.addEventListener("dblclick", () => {
      zoomImg.src = src;
      setZoom(100);
      zoomModal.hidden = false;
    });
    origGallery.appendChild(card);
  });
  if (entries.length > maxShow) {
    const more = document.createElement("div");
    more.className = "orig-empty";
    more.style.padding = "12px";
    more.textContent = `... +${entries.length - maxShow} pages (Page ${entries[maxShow].pageNum}–${entries[entries.length - 1].pageNum})`;
    origGallery.appendChild(more);
  }
}

// Inline Zoom (+ / - / Fit / Fullscreen / Ctrl+Wheel) & Click-Drag Panning in Original Pane
origZoomIn.addEventListener("click", () => applyOrigZoom(origZoomPct + 25));
origZoomOut.addEventListener("click", () => applyOrigZoom(origZoomPct - 25));
origZoomFit.addEventListener("click", () => {
  applyOrigZoom(100);
  origGallery.scrollTo({ top: 0, left: 0, behavior: "smooth" });
});
origFullBtn.addEventListener("click", openZoomLightbox);
origGallery.addEventListener("wheel", (e) => {
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault();
    applyOrigZoom(origZoomPct + (e.deltaY < 0 ? 20 : -20));
  }
}, { passive: false });

let isPanningOrig = false;
let panStartX = 0, panStartY = 0, panScrollL = 0, panScrollT = 0;
origGallery.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  isPanningOrig = true;
  origGallery.classList.add("grabbing");
  panStartX = e.clientX;
  panStartY = e.clientY;
  panScrollL = origGallery.scrollLeft;
  panScrollT = origGallery.scrollTop;
});
window.addEventListener("mousemove", (e) => {
  if (!isPanningOrig) return;
  origGallery.scrollLeft = panScrollL - (e.clientX - panStartX);
  origGallery.scrollTop = panScrollT - (e.clientY - panStartY);
});
window.addEventListener("mouseup", () => {
  if (!isPanningOrig) return;
  isPanningOrig = false;
  origGallery.classList.remove("grabbing");
});

function stageCanvasesForManualScan(canvases) {
  if (!canvases.length) return;
  lastCanvases = canvases;
  pageRangeInput.value = "";
  populatePageSelector(lastCanvases);
  renderOriginalGallery(getSelectedEntries());
  previewImg.src = lastCanvases[0].toDataURL("image/jpeg", 0.75);
  previewImg.hidden = false;
  zoomPreviewBtn.hidden = false;
  dropHint.hidden = true;
  progressWrap.hidden = true;
  updateScanBtnLabel();
  showToast(
    uiLang === "en"
      ? `📄 Loaded ${canvases.length} page(s) — Click "Start Scan OCR"`
      : uiLang === "th"
        ? `📄 โหลดแล้ว ${canvases.length} หน้า — กดปุ่ม "เริ่มสแกน OCR"`
        : `📄 ໂຫຼດແລ້ວ ${canvases.length} ໜ້າ — ກົດປຸ່ມ "ເລີ່ມສະແກນ OCR"`
  );
}

function parsePageRange(raw, total) {
  const s = String(raw || "").trim();
  if (!s) return null;
  const set = new Set();
  for (const part of s.split(/[,;\s]+/)) {
    if (!part) continue;
    const m = part.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) continue;
    let a = Math.max(1, Math.min(total, Number(m[1])));
    let b = m[2] ? Math.max(1, Math.min(total, Number(m[2]))) : a;
    if (a > b) [a, b] = [b, a];
    for (let p = a; p <= b; p++) set.add(p - 1);
  }
  const arr = Array.from(set).sort((x, y) => x - y);
  return arr.length ? arr : null;
}

function getSelectedEntries() {
  if (!lastCanvases.length) return [];
  const rangeIdxs = parsePageRange(pageRangeInput.value, lastCanvases.length);
  if (rangeIdxs) return rangeIdxs.map((i) => ({ canvas: lastCanvases[i], pageNum: i + 1 }));
  if (pageSelect.value === "all" || pageSelect.value === "") {
    return lastCanvases.map((c, i) => ({ canvas: c, pageNum: i + 1 }));
  }
  const idx = Number(pageSelect.value);
  return Number.isInteger(idx) && lastCanvases[idx]
    ? [{ canvas: lastCanvases[idx], pageNum: idx + 1 }]
    : lastCanvases.map((c, i) => ({ canvas: c, pageNum: i + 1 }));
}

function getSelectedCanvases() {
  return getSelectedEntries().map((e) => e.canvas);
}

async function runOnEntries(entries) {
  if (!entries.length) return;
  if (!ocr.ready) await ocr.init();

  const modelMode = langSelect.value;
  const toProcess = entries;
  const totalPool = lastCanvases.length || toProcess.length;

  setStatus(t("reading"), "busy");
  const multi = toProcess.length > 1;
  cancelRequested = false;
  progressWrap.hidden = false;
  progressFill.style.width = "3%";
  progressText.textContent = `Page ${toProcess[0].pageNum} (1/${toProcess.length})`;

  const outputs = [];
  lastPageResults = [];
  let totalLines = 0;
  let totalMs = 0;
  let confWeightedSum = 0;
  let confCount = 0;
  let lastDetected = "";
  const a4Mode = a4Toggle.checked;
  const layoutMode = layoutSelect.value;
  scanNowBtn.disabled = true;

  for (let i = 0; i < toProcess.length; i++) {
    if (cancelRequested) break;
    const ent = toProcess[i];
    const onLineProg = (done, tot) => {
      const overall = Math.round(((i + done / Math.max(1, tot)) / toProcess.length) * 100);
      progressFill.style.width = `${Math.max(3, Math.min(100, overall))}%`;
      progressText.textContent = multi
        ? `Page ${ent.pageNum} (${i + 1}/${toProcess.length}) • ${done}/${tot} (${overall}%)`
        : `Page ${ent.pageNum} • ${done}/${tot} (${overall}%)`;
    };
    const res = await ocr.readCanvas(
      ent.canvas,
      modelMode,
      a4Mode,
      (m) => setStatus(m, "busy"),
      layoutMode,
      onLineProg,
      () => cancelRequested
    );
    lastPageResults.push({ items: res.items, pageW: res.pageW, pageNum: ent.pageNum });
    totalLines += res.lineCount;
    totalMs += res.ms;
    if (res.detectedLang) lastDetected = res.detectedLang;
    if (res.avgConf > 0) {
      confWeightedSum += res.avgConf;
      confCount++;
    }
    if (multi || totalPool > 1) {
      outputs.push(`===== Page ${ent.pageNum} / ${totalPool} =====\n${res.text || t("noText")}`);
    } else {
      outputs.push(res.text || t("noText"));
    }
    outputText.value = outputs.join("\n\n");
    applyTypographyAndPreview();
    await new Promise((r) => setTimeout(r, 0));
  }

  progressFill.style.width = "100%";
  setTimeout(() => (progressWrap.hidden = true), 700);

  const finalText = outputs.join("\n\n");
  outputText.value = finalText;
  applyTypographyAndPreview();
  const autoTag = modelMode === "auto" && lastDetected
    ? ` • 🤖 ${lastDetected === "thai" ? "🇹🇭 ไทย" : "🇱🇦 ລາວ"}`
    : "";
  metaInfo.textContent = t("doneMeta", toProcess.length, totalLines, finalText.length, totalMs) + autoTag;
  updateScanBtnLabel();

  // Update live Speed & AI Confidence metrics bar
  const sec = Math.max(0.01, totalMs / 1000);
  const cps = Math.round(finalText.length / sec);
  const msPerLine = totalLines > 0 ? Math.round(totalMs / totalLines) : totalMs;
  const avgConf = confCount > 0 ? (confWeightedSum / confCount).toFixed(1) + "%" : "-";
  perfTime.textContent = `${sec.toFixed(2)}s (${msPerLine}ms/แถว)`;
  perfSpeed.textContent = `${cps.toLocaleString()} อักษร/วิ`;
  perfConf.textContent = avgConf;
  livePerf.hidden = false;

  setStatus(t("ready"), "ready");
}

async function pdfFileToCanvases(file) {
  if (typeof pdfjsLib === "undefined") {
    throw new Error("PDF.js not loaded");
  }
  pdfjsLib.GlobalWorkerOptions.workerSrc = chrome.runtime.getURL("lib/pdf.worker.min.js");
  const buf = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  const canvases = [];
  progressWrap.hidden = false;
  for (let p = 1; p <= pdf.numPages; p++) {
    setStatus(`PDF ${p}/${pdf.numPages}`, "busy");
    progressText.textContent = `PDF ${p}/${pdf.numPages}`;
    progressFill.style.width = `${Math.round((p / pdf.numPages) * 100)}%`;
    const page = await pdf.getPage(p);
    const baseVp = page.getViewport({ scale: 1.0 });
    const scale = Math.min(2.0, 1450 / Math.max(1, baseVp.width));
    const viewport = page.getViewport({ scale });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    await page.render({ canvasContext: ctx, viewport }).promise;
    canvases.push(canvas);
    await new Promise((r) => setTimeout(r, 0));
  }
  return canvases;
}

async function processFileList(files) {
  const list = Array.from(files || []);
  if (!list.length) return;
  const canvases = [];
  for (const f of list) {
    const lower = (f.name || "").toLowerCase();
    if (f.type === "application/pdf" || lower.endsWith(".pdf")) {
      try {
        const pdfCanvases = await pdfFileToCanvases(f);
        canvases.push(...pdfCanvases);
      } catch (err) {
        metaInfo.textContent = "PDF error: " + err.message;
      }
    } else if (f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif|avif|tiff?)$/.test(lower)) {
      canvases.push(await blobToCanvas(f));
    }
  }
  if (canvases.length) {
    stageCanvasesForManualScan(canvases);
    setStatus(t("ready"), "ready");
  } else {
    showToast("⚠️ กรุณาเลือกไฟล์รูปภาพหรือ PDF");
  }
}

// 1. Snip Screen
snipBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "START_SNIP" }, (resp) => {
    if (resp && !resp.ok) metaInfo.textContent = resp.error;
  });
});

// 2. Receive captured snip or right-clicked image
chrome.runtime.onMessage.addListener(async (msg) => {
  if (msg.type === "SNIP_CAPTURED") {
    try {
      const img = await loadImageElement(msg.dataUrl);
      const { x, y, w, h } = msg.rect;
      const dpr = msg.dpr || 1;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      canvas.getContext("2d").drawImage(
        img,
        Math.round(x * dpr), Math.round(y * dpr), canvas.width, canvas.height,
        0, 0, canvas.width, canvas.height
      );
      stageCanvasesForManualScan([canvas]);
    } catch (err) {
      metaInfo.textContent = err.message;
    }
  } else if (msg.type === "OCR_IMAGE_URL" && msg.url) {
    try {
      const blob = await (await fetch(msg.url)).blob();
      stageCanvasesForManualScan([await blobToCanvas(blob)]);
    } catch (err) {
      metaInfo.textContent = err.message;
    }
  }
});

// 3. Unified file upload (Images + PDF), click drop-zone, drag-and-drop & paste
fileInput.addEventListener("change", (e) => {
  if (e.target.files?.length) processFileList(e.target.files);
  fileInput.value = "";
});
dropZone.addEventListener("click", (e) => {
  if (e.target === previewImg && !previewImg.hidden) {
    openZoomLightbox();
    return;
  }
  fileInput.click();
});

// Collapsible Preview & Full-Screen Original Zoom Lightbox
function setZoom(pct) {
  zoomPct = Math.max(40, Math.min(350, pct));
  zoomVal.textContent = zoomPct + "%";
  zoomImg.style.width = zoomPct + "%";
}
function openZoomLightbox() {
  if (!previewImg.src || previewImg.hidden) return;
  zoomImg.src = previewImg.src;
  setZoom(100);
  zoomModal.hidden = false;
}
togglePreviewBtn.addEventListener("click", () => {
  previewCollapsed = !previewCollapsed;
  dropZone.hidden = previewCollapsed;
  togglePreviewBtn.textContent = previewCollapsed
    ? (uiLang === "en" ? "🔽 Expand" : uiLang === "th" ? "🔽 ขยาย" : "🔽 ຂະຫຍາຍ")
    : (uiLang === "en" ? "🔼 Collapse" : uiLang === "th" ? "🔼 ย่อ" : "🔼 ຫຍໍ້");
});
zoomPreviewBtn.addEventListener("click", openZoomLightbox);
zoomInBtn.addEventListener("click", () => setZoom(zoomPct + 25));
zoomOutBtn.addEventListener("click", () => setZoom(zoomPct - 25));
zoomFitBtn.addEventListener("click", () => setZoom(100));
closeZoomBtn.addEventListener("click", () => (zoomModal.hidden = true));
zoomModal.addEventListener("click", (e) => {
  if (e.target === zoomModal || e.target === $("zoomStage")) zoomModal.hidden = true;
});

const uploadDropBtn = $("uploadDropBtn");
[dropZone, uploadDropBtn, window].forEach((target) => {
  target.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
    uploadDropBtn.classList.add("dragover");
  });
});
[dropZone, uploadDropBtn].forEach((target) => {
  target.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
    uploadDropBtn.classList.remove("dragover");
  });
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  uploadDropBtn.classList.remove("dragover");
  if (e.dataTransfer?.files?.length) processFileList(e.dataTransfer.files);
});

window.addEventListener("paste", async (e) => {
  const blobs = [];
  for (const it of e.clipboardData?.items || []) {
    if (it.type.startsWith("image/")) {
      const b = it.getAsFile();
      if (b) blobs.push(b);
    }
  }
  if (blobs.length) {
    const canvases = [];
    for (const b of blobs) canvases.push(await blobToCanvas(b));
    stageCanvasesForManualScan(canvases);
  }
});

// Compact Icon Pill Controls (Language, Layout, 3-Mode Workspace)
document.querySelectorAll("#langPills .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#langPills .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    langSelect.value = btn.dataset.lang;
  });
});
document.querySelectorAll("#layoutPills .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#layoutPills .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    layoutSelect.value = btn.dataset.layout;
    renderFormattedOutput();
  });
});
document.querySelectorAll("#wsModeTabs .ws-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#wsModeTabs .ws-tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    workspaceGrid.className = "workspace-grid mode-" + btn.dataset.ws;
  });
});

// Instant 1ms layout switch & Typography / Color Proofread controls
layoutSelect.addEventListener("change", () => renderFormattedOutput());
fontDecBtn.addEventListener("click", () => setOutputFontSize(outputFontSize - 2));
fontIncBtn.addEventListener("click", () => setOutputFontSize(outputFontSize + 2));
fontSelect.addEventListener("change", () => {
  applyTypographyAndPreview();
  chrome.storage.local.set({ mbm_font_fam: fontSelect.value }).catch(() => {});
});
viewModeBtn.addEventListener("click", () => {
  colorProofreadMode = !colorProofreadMode;
  applyTypographyAndPreview();
  if (!colorProofreadMode) outputText.focus();
});
richPreview.addEventListener("dblclick", () => {
  colorProofreadMode = false;
  applyTypographyAndPreview();
  outputText.focus();
  showToast("✏️ โหมดแก้ไขข้อความ");
});
outputText.addEventListener("input", () => {
  richPreview.innerHTML = colorizeTextToHtml(outputText.value);
});
document.querySelectorAll(".btn-fmt").forEach((btn) => {
  btn.addEventListener("click", () => {
    const cat = btn.dataset.cat;
    const prop = btn.dataset.prop;
    catStyles[cat][prop] = !catStyles[cat][prop];
    btn.classList.toggle("active", catStyles[cat][prop]);
    applyTypographyAndPreview();
  });
});
chrome.storage.local.get(["mbm_font_sz", "mbm_font_fam"]).then((d) => {
  if (d.mbm_font_fam) fontSelect.value = d.mbm_font_fam;
  if (d.mbm_font_sz) setOutputFontSize(Number(d.mbm_font_sz));
  else applyTypographyAndPreview();
}).catch(() => {});

// Manual Scan ONLY when user clicks the Scan button (no auto-scan on change!)
pageSelect.addEventListener("change", () => {
  pageRangeInput.value = "";
  const sel = getSelectedEntries();
  if (sel.length) {
    previewImg.src = sel[0].canvas.toDataURL("image/jpeg", 0.75);
    renderOriginalGallery(sel);
  }
  updateScanBtnLabel();
});
let rangeTimer = null;
pageRangeInput.addEventListener("input", () => {
  updateScanBtnLabel();
  clearTimeout(rangeTimer);
  rangeTimer = setTimeout(() => {
    const sel = getSelectedEntries();
    if (sel.length) {
      previewImg.src = sel[0].canvas.toDataURL("image/jpeg", 0.75);
      renderOriginalGallery(sel);
    }
  }, 220);
});
scanNowBtn.addEventListener("click", () => {
  const sel = getSelectedEntries();
  if (sel.length) runOnEntries(sel);
});
cancelScanBtn.addEventListener("click", () => {
  cancelRequested = true;
  showToast("⏹️ กำลังหยุดสแกน...");
});

// Output actions
copyBtn.addEventListener("click", async () => {
  if (!outputText.value) return;
  await navigator.clipboard.writeText(outputText.value);
  showToast(t("copied"));
});

downloadBtn.addEventListener("click", () => {
  if (!outputText.value) return;
  const blob = new Blob(["\uFEFF" + outputText.value], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `MaeBounme_OCR_${Date.now()}.txt`;
  a.click();
  showToast("📄 บันทึก .txt แล้ว!");
});

downloadWordBtn.addEventListener("click", () => {
  if (!outputText.value) return;
  const lines = outputText.value.split(/\r?\n/);
  const bodyHtml = lines
    .map((ln) => {
      if (!ln.trim()) return "<p>&nbsp;</p>";
      const styled = colorProofreadMode ? colorizeTextToHtml(ln) : ln.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      if (ln.startsWith("## ")) return `<h2>${styled.replace(/^##\s*/, "")}</h2>`;
      if (ln.startsWith("# ")) return `<h1>${styled.replace(/^#\s*/, "")}</h1>`;
      return `<p>${styled.replace(/ {2}/g, "&nbsp;&nbsp;")}</p>`;
    })
    .join("\n");
  const docHtml = `\uFEFF<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
<head><meta charset="utf-8"><style>
body { font-family: ${fontSelect.value}; font-size: ${Math.max(12, Math.round(outputFontSize * 0.85))}pt; line-height: 1.5; }
h1, h2 { color: #1e3a8a; }
p { margin: 0 0 6pt 0; }
</style></head><body>${bodyHtml}</body></html>`;
  const blob = new Blob([docHtml], { type: "application/msword;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `MaeBounme_OCR_${Date.now()}.doc`;
  a.click();
  showToast("📘 บันทึกไฟล์ Word แล้ว!");
});

clearBtn.addEventListener("click", () => {
  lastCanvases = [];
  renderOriginalGallery([]);
  pageBar.hidden = true;
  pageSelect.innerHTML = "";
  previewImg.hidden = true;
  zoomPreviewBtn.hidden = true;
  dropHint.hidden = false;
  progressWrap.hidden = true;
  outputText.value = "";
  applyTypographyAndPreview();
  updateScanBtnLabel();
  metaInfo.textContent = "";
});

// Model Credits Modal events
openCreditsBtn.addEventListener("click", () => (creditsModal.hidden = false));
closeCreditsBtn.addEventListener("click", () => (creditsModal.hidden = true));
creditsModal.addEventListener("click", (e) => {
  if (e.target === creditsModal) creditsModal.hidden = true;
});

// Donate Modal events + collapsible QR toggle (LAK / USD / THB)
const donateQrBox = $("donateQrBox");
const donateQrImg = $("donateQrImg");
const qrToggleBtns = Array.from(document.querySelectorAll(".qr-toggle-btn"));
function hideDonateQr() {
  if (donateQrBox) donateQrBox.hidden = true;
  if (donateQrImg) donateQrImg.src = "";
  qrToggleBtns.forEach((b) => { b.classList.remove("primary"); b.classList.add("secondary"); });
}
qrToggleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const src = chrome.runtime.getURL(btn.dataset.qr);
    const isSame = !donateQrBox.hidden && donateQrImg.src === src;
    hideDonateQr();
    if (!isSame) {
      donateQrImg.src = src;
      donateQrBox.hidden = false;
      btn.classList.remove("secondary");
      btn.classList.add("primary");
    }
  });
});
openProBtn.addEventListener("click", () => { hideDonateQr(); openProModal(); });
closeProBtn.addEventListener("click", () => { hideDonateQr(); proModal.hidden = true; });
proModal.addEventListener("click", (e) => {
  if (e.target === proModal) { hideDonateQr(); proModal.hidden = true; }
});
supportLinkBtn.addEventListener("click", () => {
  chrome.tabs.create({ url: LICENSING.SUPPORT_URL });
});
