/**
 * MaeBounme Lao-OCR Dual-Model Offline Engine (ONNX Runtime Web / WASM)
 * - Shared Detector: PP-OCRv6 DB Text-Line Detector (models/det.onnx)
 * - Lao Model:       MaeBounme Round 10 PP-OCRv6 Small SVTR (models/rec.onnx, 595 classes)
 * - Thai Model:      OpenThai2.1-OCR-Text (models/thai_rec.onnx, 530 classes, 100% Free)
 */
class OfflineLaoOCR {
  constructor() {
    this.detSession = null;
    this.laoSession = null;
    this.laoChars = [];
    this.laoIsThai = [];
    this.laoIsLatin = [];
    this.thaiSession = null;
    this.thaiChars = [];
    this.laoLex = new Map();
    this.ready = false;
  }

  async init(onStatus) {
    if (this.ready) return;
    const status = onStatus || (() => {});
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");
    ort.env.wasm.numThreads = 1;

    status("1/3 Dictionary & Lexicon...");
    const dictText = await (await fetch(chrome.runtime.getURL("models/rec_dict.txt"))).text();
    const rawChars = dictText.split(/\r?\n/).filter((c) => c.length > 0);
    this.laoChars = ["", ...rawChars, " "];
    this.laoIsThai = this.laoChars.map((c) => c >= "\u0e00" && c <= "\u0e7f");
    this.laoIsLatin = this.laoChars.map((c) => /^[A-Za-z]$/.test(c));

    try {
      const lexObj = await (await fetch(chrome.runtime.getURL("models/lao_lexicon.json"))).json();
      const words = lexObj?.w || [];
      this.laoLex = new Map();
      for (let i = 0; i < words.length; i++) this.laoLex.set(words[i], i);
    } catch (_) {
      this.laoLex = new Map();
    }

    status("2/3 Detector...");
    const detBuf = await (await fetch(chrome.runtime.getURL("models/det.onnx"))).arrayBuffer();
    this.detSession = await ort.InferenceSession.create(detBuf, { executionProviders: ["wasm"] });

    status("3/3 MaeBounme Lao-OCR...");
    const recBuf = await (await fetch(chrome.runtime.getURL("models/rec.onnx"))).arrayBuffer();
    this.laoSession = await ort.InferenceSession.create(recBuf, { executionProviders: ["wasm"] });

    this.ready = true;
  }

  /** Lazy-load the dedicated OpenThai2.1-OCR Thai model on first Thai scan */
  async ensureThaiModel(onStatus) {
    if (this.thaiSession) return;
    if (onStatus) onStatus("Loading OpenThai2.1...");
    const dictText = await (await fetch(chrome.runtime.getURL("models/thai_rec_dict.txt"))).text();
    const rawChars = dictText.split(/\r?\n/).filter((c) => c.length > 0);
    this.thaiChars = ["", ...rawChars, " "];
    const buf = await (await fetch(chrome.runtime.getURL("models/thai_rec.onnx"))).arrayBuffer();
    this.thaiSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  }

  enhanceA4Canvas(srcCanvas) {
    const W = srcCanvas.width;
    const H = srcCanvas.height;
    const out = document.createElement("canvas");
    out.width = W;
    out.height = H;
    const ctx = out.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(srcCanvas, 0, 0);
    const imgData = ctx.getImageData(0, 0, W, H);
    const d = imgData.data;

    const hist = new Uint32Array(256);
    const n = W * H;
    for (let i = 0; i < d.length; i += 4) {
      const y = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      hist[y]++;
    }
    let lo = 0, hi = 255, acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= n * 0.02) { lo = i; break; }
    }
    acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= n * 0.90) { hi = i; break; }
    }
    const span = Math.max(32, hi - lo);
    const scale = 255.0 / span;
    for (let i = 0; i < d.length; i += 4) {
      const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      const v = Math.max(0, Math.min(255, Math.round((y - lo) * scale)));
      d[i] = d[i + 1] = d[i + 2] = v;
    }
    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  async detectRows(canvas, a4Mode = true) {
    const W = canvas.width;
    const H = canvas.height;
    if (!a4Mode && H <= 140 && W >= 1.8 * H) {
      return [[[0, 0, W, H]]];
    }

    const limit = a4Mode ? 1280 : 1024;
    const scale = Math.min(1.0, limit / Math.max(W, H));
    const dw = Math.max(32, Math.round((W * scale) / 32) * 32);
    const dh = Math.max(32, Math.round((H * scale) / 32) * 32);

    const tmp = document.createElement("canvas");
    tmp.width = dw;
    tmp.height = dh;
    const ctx = tmp.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(canvas, 0, 0, dw, dh);
    const { data } = ctx.getImageData(0, 0, dw, dh);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const hw = dw * dh;
    const input = new Float32Array(3 * hw);
    for (let i = 0; i < hw; i++) {
      const r = data[i * 4] / 255.0;
      const g = data[i * 4 + 1] / 255.0;
      const b = data[i * 4 + 2] / 255.0;
      input[i] = (b - mean[0]) / std[0];
      input[hw + i] = (g - mean[1]) / std[1];
      input[2 * hw + i] = (r - mean[2]) / std[2];
    }

    const tensor = new ort.Tensor("float32", input, [1, 3, dh, dw]);
    const outMap = await this.detSession.run({ x: tensor });
    const prob = outMap[this.detSession.outputNames[0]].data;

    const mask = new Uint8Array(hw);
    for (let i = 0; i < hw; i++) mask[i] = prob[i] > 0.30 ? 1 : 0;

    const visited = new Uint8Array(hw);
    const boxes = [];
    const queue = new Int32Array(hw);
    const compPts = new Int32Array(hw);
    const sx = W / dw;
    const sy = H / dh;

    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        const idx = y * dw + x;
        if (!mask[idx] || visited[idx]) continue;
        let head = 0, tail = 0;
        queue[tail++] = idx;
        visited[idx] = 1;
        let minX = x, maxX = x, minY = y, maxY = y;
        let sumProb = 0, count = 0;

        while (head < tail) {
          const cur = queue[head++];
          const cx = cur % dw;
          const cy = (cur - cx) / dw;
          compPts[count++] = cur;
          sumProb += prob[cur];
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          if (cx > 0 && mask[cur - 1] && !visited[cur - 1]) { visited[cur - 1] = 1; queue[tail++] = cur - 1; }
          if (cx + 1 < dw && mask[cur + 1] && !visited[cur + 1]) { visited[cur + 1] = 1; queue[tail++] = cur + 1; }
          if (cy > 0 && mask[cur - dw] && !visited[cur - dw]) { visited[cur - dw] = 1; queue[tail++] = cur - dw; }
          if (cy + 1 < dh && mask[cur + dw] && !visited[cur + dw]) { visited[cur + dw] = 1; queue[tail++] = cur + dw; }
        }

        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;
        if (count < 10 || bw < 4 || bh < 3) continue;
        if (sumProb / count < 0.50) continue;

        const padTop = Math.max(3, bh * 0.56);
        const padBottom = Math.max(2, bh * 0.48);
        const padX = Math.max(3, bh * 0.55);
        const x1 = Math.max(0, Math.floor((minX - padX) * sx));
        const y1 = Math.max(0, Math.floor((minY - padTop) * sy));
        const x2 = Math.min(W, Math.ceil((maxX + padX) * sx));
        const y2 = Math.min(H, Math.ceil((maxY + padBottom) * sy));
        if (x2 - x1 >= 6 && y2 - y1 >= 6) {
          const bObj = [x1, y1, x2, y2];
          // Attach smoothed column centerline profile for wavy/tilted line dewarping
          if (bw >= 10) {
            const cMin = new Float32Array(bw).fill(1e9);
            const cMax = new Float32Array(bw).fill(-1e9);
            for (let k = 0; k < count; k++) {
              const p = compPts[k];
              const px = (p % dw) - minX;
              const py = (p - (p % dw)) / dw;
              if (py < cMin[px]) cMin[px] = py;
              if (py > cMax[px]) cMax[px] = py;
            }
            let lastMin = minY, lastMax = maxY;
            const hs = [];
            for (let c = 0; c < bw; c++) {
              if (cMin[c] <= cMax[c]) {
                lastMin = cMin[c]; lastMax = cMax[c];
                hs.push((cMax[c] - cMin[c] + 1) * sy);
              } else { cMin[c] = lastMin; cMax[c] = lastMax; }
            }
            hs.sort((a, b) => a - b);
            const medH = hs.length ? hs[Math.floor(hs.length * 0.65)] : bh * sy;
            const win = Math.max(2, Math.min(12, Math.floor(bw / 8)));
            const colTop = new Float32Array(bw);
            const colBot = new Float32Array(bw);
            for (let c = 0; c < bw; c++) {
              let sMid = 0, n = 0;
              for (let j = Math.max(0, c - win); j <= Math.min(bw - 1, c + win); j++) {
                sMid += ((cMin[j] + cMax[j] + 1) * 0.5) * sy;
                n++;
              }
              const mid = sMid / Math.max(1, n);
              colTop[c] = Math.max(0, mid - medH * 0.5 - Math.max(4, medH * 0.52));
              colBot[c] = Math.min(H, mid + medH * 0.5 + Math.max(3, medH * 0.44));
            }
            bObj._colTop = colTop;
            bObj._colBot = colBot;
            bObj._medH = medH * 1.96;
          }
          boxes.push(bObj);
        }
      }
    }

    if (boxes.length === 0) return [[[0, 0, W, H]]];

    boxes.sort((a, b) => (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2);
    const rows = [];
    for (const b of boxes) {
      let placed = false;
      for (const row of rows) {
        const ref = row[0];
        const ov = Math.min(ref[3], b[3]) - Math.max(ref[1], b[1]);
        const minH = Math.min(ref[3] - ref[1], b[3] - b[1]);
        if (ov > 0.45 * minH) {
          row.push(b);
          placed = true;
          break;
        }
      }
      if (!placed) rows.push([b]);
    }
    for (const row of rows) row.sort((a, b) => a[0] - b[0]);
    return rows;
  }

  async recognizeCrop(canvas, box, modelMode = "lao") {
    const [x1, y1, x2, y2] = box;
    const cw = Math.max(1, x2 - x1);
    const ch = box._medH ? Math.max(8, box._medH) : Math.max(1, y2 - y1);
    const H = 48;
    let newW = Math.max(16, Math.round(cw * (H / ch)));
    if (newW > 1600) newW = 1600;
    newW = Math.max(8, Math.round(newW / 8) * 8);

    const tmp = document.createElement("canvas");
    tmp.width = newW;
    tmp.height = H;
    const ctx = tmp.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, newW, H);

    // Dewarp wavy / tilted text lines slice-by-slice along smoothed column centerline
    if (box._colTop && box._colBot && box._colTop.length >= 10) {
      const nCols = box._colTop.length;
      const slices = Math.min(48, Math.max(12, Math.floor(newW / 8)));
      for (let s = 0; s < slices; s++) {
        const u0 = s / slices;
        const u1 = (s + 1) / slices;
        const uMid = (u0 + u1) * 0.5;
        const ci = Math.min(nCols - 1, Math.floor(uMid * nCols));
        const sy1 = box._colTop[ci];
        const sy2 = box._colBot[ci];
        const sx1 = x1 + u0 * cw;
        const sW = Math.max(1, (u1 - u0) * cw);
        const dx1 = Math.floor(u0 * newW);
        const dW = Math.max(1, Math.ceil(u1 * newW) - dx1);
        ctx.drawImage(canvas, sx1, sy1, sW, Math.max(4, sy2 - sy1), dx1, 0, dW, H);
      }
    } else {
      ctx.drawImage(canvas, x1, y1, cw, Math.max(1, y2 - y1), 0, 0, newW, H);
    }
    const { data } = ctx.getImageData(0, 0, newW, H);

    const hw = newW * H;
    const input = new Float32Array(3 * hw);
    for (let i = 0; i < hw; i++) {
      const gray = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
      const v = gray / 127.5 - 1.0;
      input[i] = v;
      input[hw + i] = v;
      input[2 * hw + i] = v;
    }

    const useThai = modelMode === "thai";
    const sess = useThai ? this.thaiSession : this.laoSession;
    const chars = useThai ? this.thaiChars : this.laoChars;
    const tensor = new ort.Tensor("float32", input, [1, 3, H, newW]);
    const res = await sess.run({ x: tensor });
    const out = res[sess.outputNames[0]];
    const T = out.dims[1];
    const C = out.dims[2];
    const probs = out.data;

    let blockFn = null;
    if (modelMode === "lao") {
      blockFn = (c) => this.laoIsThai[c] || this.laoIsLatin[c];
    } else if (modelMode === "lao+en") {
      blockFn = (c) => this.laoIsThai[c];
    }

    let text = "";
    let prev = -1;
    let confSum = 0;
    let confCnt = 0;
    for (let t = 0; t < T; t++) {
      let bestIdx = 0;
      let bestVal = -1e9;
      const base = t * C;
      for (let c = 0; c < C; c++) {
        if (blockFn && blockFn(c)) continue;
        const v = probs[base + c];
        if (v > bestVal) {
          bestVal = v;
          bestIdx = c;
        }
      }
      if (bestIdx !== 0 && bestIdx !== prev) {
        text += chars[bestIdx];
        // Handle both already-softmaxed probabilities [0..1] and raw logits
        let rowSum = 0;
        for (let c = 0; c < C; c++) rowSum += probs[base + c];
        if (bestVal >= 0 && bestVal <= 1.001 && rowSum > 0.95 && rowSum < 1.05) {
          confSum += bestVal;
        } else {
          let sumExp = 0;
          for (let c = 0; c < C; c++) {
            const d = probs[base + c] - bestVal;
            if (d > -20) sumExp += Math.exp(d);
          }
          confSum += sumExp > 0 ? 1 / sumExp : 1;
        }
        confCnt++;
      }
      prev = bestIdx;
    }
    this._lastConf = confCnt > 0 ? confSum / confCnt : 0;
    if (useThai) return text.trim();
    const cleaned = text.replace(/ໍ\u0eb2/g, "ຳ").trim();
    return this.correctLaoLine(cleaned);
  }

  repairLaoToken(tok) {
    const lex = this.laoLex;
    if (!lex || !lex.size || tok.length < 2 || tok.length > 14 || lex.has(tok)) return tok;
    const CONF = {
      "ຂ": ["ຊ"], "ຊ": ["ຂ"], "ບ": ["ປ"], "ປ": ["ບ"], "ຜ": ["ຝ"], "ຝ": ["ຜ"],
      "ພ": ["ຟ"], "ຟ": ["ພ"], "ຄ": ["ດ"], "ດ": ["ຄ", "ຕ"], "ຕ": ["ດ"],
      "ທ": ["ຫ"], "ຫ": ["ທ"], "ນ": ["ມ"], "ມ": ["ນ"],
      "່": ["້"], "້": ["່"], "ຸ": ["ູ"], "ູ": ["ຸ"]
    };
    const MARKS = ["ຸ", "ູ", "່", "້", "ັ", "ົ", "ິ", "ີ", "ຶ", "ື"];
    let bestWord = tok;
    let bestScore = Infinity;

    for (let i = 0; i < tok.length; i++) {
      const alts = CONF[tok[i]];
      if (!alts) continue;
      for (const alt of alts) {
        const cand = tok.slice(0, i) + alt + tok.slice(i + 1);
        const rank = lex.get(cand);
        if (rank !== undefined && rank < 22000) {
          const score = rank + (tok[i] === "ຂ" || tok[i] === "ຊ" ? 0 : 1200);
          if (score < bestScore) { bestScore = score; bestWord = cand; }
        }
      }
    }
    for (let i = 0; i < tok.length; i++) {
      const ch = tok[i];
      const next = tok[i + 1] || "";
      const isCons = /^[\u0E81-\u0EAE]$/.test(ch);
      const isTopV = /^[\u0EB1\u0EB4-\u0EB7\u0EBB]$/.test(ch);
      if (!isCons && !isTopV) continue;
      const marks = isTopV ? ["່", "້"] : MARKS;
      for (const m of marks) {
        if (next === m) continue;
        const cand = tok.slice(0, i + 1) + m + tok.slice(i + 1);
        const rank = lex.get(cand);
        if (rank !== undefined && rank < 10000) {
          const score = rank + 2500;
          if (score < bestScore) { bestScore = score; bestWord = cand; }
        }
      }
    }
    return bestWord;
  }

  correctLaoLine(line) {
    const lex = this.laoLex;
    if (!lex || !lex.size || !/[\u0E80-\u0EFF]/.test(line)) return line;
    return line.replace(/[\u0E80-\u0EFF]{2,}/g, (span) => {
      if (lex.has(span)) return span;
      if (span.length <= 12) return this.repairLaoToken(span);
      let out = "";
      let i = 0;
      while (i < span.length) {
        let matched = "";
        for (let len = Math.min(14, span.length - i); len >= 2; len--) {
          const sub = span.slice(i, i + len);
          if (lex.has(sub)) { matched = sub; break; }
        }
        if (matched) {
          out += matched;
          i += matched.length;
        } else {
          let j = i + 1;
          while (j < span.length && j - i < 10) {
            let foundNext = false;
            for (let len = Math.min(10, span.length - j); len >= 3; len--) {
              if (lex.has(span.slice(j, j + len))) { foundNext = true; break; }
            }
            if (foundNext) break;
            j++;
          }
          out += this.repairLaoToken(span.slice(i, j));
          i = j;
        }
      }
      return out;
    });
  }

  /** Format structured row boxes into 4 layout modes: 'paragraph' | 'exact' | 'markdown' | 'two_col' */
  formatLayout(pageItems, pageW, mode = "paragraph") {
    // pageItems: array of { box: [x1,y1,x2,y2], text: string }
    if (!pageItems.length) return "";
    const heights = pageItems.map((it) => it.box[3] - it.box[1]).sort((a, b) => a - b);
    const lh = Math.max(10, heights[Math.floor(heights.length / 2)] || 20);

    const groupRows = (items) => {
      const sorted = [...items].sort((a, b) => (a.box[1] + a.box[3]) / 2 - (b.box[1] + b.box[3]) / 2);
      const rows = [];
      for (const it of sorted) {
        let placed = false;
        for (const r of rows) {
          const ref = r[0].box;
          const ov = Math.min(ref[3], it.box[3]) - Math.max(ref[1], it.box[1]);
          const minH = Math.min(ref[3] - ref[1], it.box[3] - it.box[1]);
          if (ov > 0.45 * minH) {
            r.push(it);
            placed = true;
            break;
          }
        }
        if (!placed) rows.push([it]);
      }
      for (const r of rows) r.sort((a, b) => a.box[0] - b.box[0]);
      return rows;
    };

    if (mode === "two_col") {
      const mid = pageW * 0.49;
      const fullW = [], left = [], right = [];
      for (const it of pageItems) {
        const w = it.box[2] - it.box[0];
        if (w > pageW * 0.55) fullW.push(it);
        else if ((it.box[0] + it.box[2]) / 2 < mid) left.push(it);
        else right.push(it);
      }
      const renderCol = (arr) => groupRows(arr).map((r) => r.map((x) => x.text).join("   ")).join("\n");
      return [renderCol(fullW), renderCol(left), renderCol(right)].filter(Boolean).join("\n\n");
    }

    const rows = groupRows(pageItems);

    if (mode === "exact") {
      const leftMin = Math.min(...rows.map((r) => r[0].box[0]));
      const out = [];
      let prevY2 = null;
      for (const r of rows) {
        const y1 = Math.min(...r.map((x) => x.box[1]));
        const y2 = Math.max(...r.map((x) => x.box[3]));
        if (prevY2 !== null && y1 - prevY2 > lh * 1.15) out.push("");
        const indentSpaces = Math.max(0, Math.min(12, Math.round((r[0].box[0] - leftMin) / (lh * 0.6))));
        let line = " ".repeat(indentSpaces);
        for (let i = 0; i < r.length; i++) {
          if (i > 0) {
            const gap = r[i].box[0] - r[i - 1].box[2];
            line += gap > lh * 2.5 ? "      " : "   ";
          }
          line += r[i].text;
        }
        out.push(line);
        prevY2 = y2;
      }
      return out.join("\n").trim();
    }

    if (mode === "markdown") {
      const out = [];
      let i = 0, prevY2 = null;
      while (i < rows.length) {
        const r = rows[i];
        const y1 = Math.min(...r.map((x) => x.box[1]));
        const y2 = Math.max(...r.map((x) => x.box[3]));
        const rh = y2 - y1;
        if (prevY2 !== null && y1 - prevY2 > lh * 1.35 && out.length && out[out.length - 1] !== "") out.push("");
        // Table detection: >= 2 consecutive multi-cell rows
        if (r.length >= 2) {
          let j = i + 1;
          while (j < rows.length && rows[j].length >= 2) j++;
          if (j - i >= 2) {
            const cols = Math.min(8, Math.max(...rows.slice(i, j).map((x) => x.length)));
            const mkRow = (row) => "| " + Array.from({ length: cols }, (_, c) => (row[c]?.text || "").replace(/\|/g, "\\|")).join(" | ") + " |";
            out.push(mkRow(rows[i]));
            out.push("| " + Array(cols).fill("---").join(" | ") + " |");
            for (let k = i + 1; k < j; k++) out.push(mkRow(rows[k]));
            out.push("");
            prevY2 = Math.max(...rows[j - 1].map((x) => x.box[3]));
            i = j;
            continue;
          }
        }
        const txt = r.map((x) => x.text).join("   ");
        if (r.length === 1 && txt.length <= 80 && rh >= lh * 1.35) out.push(`## ${txt}`);
        else if (/^(?:ມາດຕາ|มาตรา|ข้อ|ຂໍ້)\s*\d+/.test(txt)) out.push(`**${txt}**`);
        else out.push(txt);
        prevY2 = y2;
        i++;
      }
      return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
    }

    // Default: 'paragraph' — smart paragraph grouping by vertical gaps & first-line indentation
    const leftPositions = rows.map((r) => r[0].box[0]).sort((a, b) => a - b);
    const bodyLeft = leftPositions[Math.floor(leftPositions.length * 0.25)] || 0;
    const out = [];
    let prevY2 = null;
    for (const r of rows) {
      const y1 = Math.min(...r.map((x) => x.box[1]));
      const y2 = Math.max(...r.map((x) => x.box[3]));
      const x1 = r[0].box[0];
      const txt = r.map((x) => x.text).join("   ");
      const isIndented = x1 - bodyLeft > lh * 1.35 && r.length === 1;
      const bigGap = prevY2 !== null && y1 - prevY2 > lh * 1.25;
      if ((bigGap || isIndented) && out.length && out[out.length - 1] !== "") {
        out.push("");
      }
      out.push((isIndented ? "    " : "") + txt);
      prevY2 = y2;
    }
    return out.join("\n").trim();
  }

  async readCanvas(canvas, modelMode = "auto", a4Mode = true, onStatus = null, layoutMode = "paragraph", onProgress = null, shouldCancel = null) {
    const t0 = performance.now();
    const workCanvas = a4Mode ? this.enhanceA4Canvas(canvas) : canvas;
    const rows = await this.detectRows(workCanvas, a4Mode);
    const flatBoxes = rows.flat();
    const totalBoxes = Math.max(1, flatBoxes.length);

    let effectiveMode = modelMode;
    let detectedLang = modelMode;
    if (modelMode === "auto") {
      await this.ensureThaiModel(onStatus);
      // Sample up to 4 representative boxes to auto-detect Lao vs Thai accurately
      const step = Math.max(1, Math.floor(flatBoxes.length / 4));
      const probes = flatBoxes.filter((_, idx) => idx % step === 0).slice(0, 4);
      let laoScore = 0;
      let thaiScore = 0;
      for (const box of probes) {
        if (shouldCancel && shouldCancel()) break;
        const lTxt = await this.recognizeCrop(workCanvas, box, "lao+en");
        const lConf = this._lastConf || 0;
        const lChars = (lTxt.match(/[\u0E80-\u0EFF]/g) || []).length;
        const tTxt = await this.recognizeCrop(workCanvas, box, "thai");
        const tConf = this._lastConf || 0;
        const tChars = (tTxt.match(/[\u0E00-\u0E7F]/g) || []).length;
        laoScore += lConf * (1 + lChars * 0.25);
        thaiScore += tConf * (1 + tChars * 0.25);
      }
      effectiveMode = thaiScore > laoScore ? "thai" : "lao+en";
      detectedLang = effectiveMode === "thai" ? "thai" : "lao";
    } else if (modelMode === "thai") {
      await this.ensureThaiModel(onStatus);
    }

    const items = [];
    let confSum = 0;
    let doneBoxes = 0;
    for (const row of rows) {
      for (const box of row) {
        if (shouldCancel && shouldCancel()) break;
        const txt = await this.recognizeCrop(workCanvas, box, effectiveMode);
        doneBoxes++;
        if (txt && !/^[\.\-_=~*·•\s]+$/.test(txt)) {
          items.push({ box, text: txt, conf: this._lastConf || 0 });
          confSum += this._lastConf || 0;
        }
        if (onProgress && (doneBoxes % 2 === 0 || doneBoxes === totalBoxes)) {
          onProgress(doneBoxes, totalBoxes);
          // Yield to UI paint loop so Chrome never freezes and progress bar animates smoothly
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      if (shouldCancel && shouldCancel()) break;
    }

    const text = this.formatLayout(items, workCanvas.width, layoutMode);
    return {
      text,
      items,
      detectedLang,
      pageW: workCanvas.width,
      lineCount: text ? text.split("\n").filter((l) => l.trim()).length : 0,
      avgConf: items.length ? Math.round((confSum / items.length) * 1000) / 10 : 0,
      ms: Math.round(performance.now() - t0)
    };
  }
}

window.OfflineLaoOCR = OfflineLaoOCR;
