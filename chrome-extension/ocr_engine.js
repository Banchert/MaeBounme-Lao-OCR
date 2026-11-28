/**
 * Compact 5-gram Lao Character Language Model + CTC Prefix Beam Search (100% Offline)
 * Loads models/lao_charlm_mobile.bin (~1.7 MB) to resolve context-dependent Lao tone marks
 * (ຜ່ານ vs ຜານ, ຕ່າງໆ vs ຕາງໆ, ໄດ້ແຕ່ງຕັ້ງ) and look-alike letters (ງ↔າ, ຂ↔ຊ).
 */
class CompactCharLM {
  constructor(buf) {
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    if (dv.getUint32(0, true) !== 0x4d4c434c) throw new Error("Bad LCLM header");
    const count = dv.getUint32(4, true);
    this.order = dv.getUint32(8, true);
    this.unk = dv.getFloat32(12, true);
    this.hashes = new Uint32Array(count);
    for (let i = 0; i < count; i++) this.hashes[i] = dv.getUint32(16 + i * 4, true);
    this.qlogp = u8.slice(16 + count * 4, 16 + count * 5);
    this.cache = new Map();
    this._backoffPen = Math.log10(0.4);
  }

  static fnv1a32(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      const cp = str.charCodeAt(i);
      if (cp < 0x80) {
        h ^= cp; h = Math.imul(h, 0x01000193) >>> 0;
      } else if (cp < 0x800) {
        h ^= 0xc0 | (cp >> 6); h = Math.imul(h, 0x01000193) >>> 0;
        h ^= 0x80 | (cp & 0x3f); h = Math.imul(h, 0x01000193) >>> 0;
      } else {
        h ^= 0xe0 | (cp >> 12); h = Math.imul(h, 0x01000193) >>> 0;
        h ^= 0x80 | ((cp >> 6) & 0x3f); h = Math.imul(h, 0x01000193) >>> 0;
        h ^= 0x80 | (cp & 0x3f); h = Math.imul(h, 0x01000193) >>> 0;
      }
    }
    return h >>> 0;
  }

  _lookup(gram) {
    const h = CompactCharLM.fnv1a32(gram);
    let lo = 0, hi = this.hashes.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const v = this.hashes[mid];
      if (v === h) return -(this.qlogp[mid] / 31.875);
      if (v < h) lo = mid + 1; else hi = mid - 1;
    }
    return null;
  }

  score(ctx, ch) {
    const key = ("\x02".repeat(this.order) + ctx).slice(-(this.order - 1)) + ch;
    const cached = this.cache.get(key);
    if (cached !== undefined) return cached;
    let pen = 0, res = this.unk + this.order * this._backoffPen;
    for (let n = this.order; n >= 1; n--) {
      const v = this._lookup(key.slice(-n));
      if (v !== null) { res = v + pen; break; }
      pen += this._backoffPen;
    }
    if (this.cache.size > 60000) this.cache.clear();
    this.cache.set(key, res);
    return res;
  }
}

function _lse10(a, b) {
  if (a < b) { const t = a; a = b; b = t; }
  return b > -1e8 ? a + Math.log10(1 + Math.pow(10, b - a)) : a;
}

function beamDecodeCTC(probs, T, C, chars, lm, blockFn = null, alpha = 0.36, beta = 0.72, beamW = 8) {
  let beams = new Map([["", [0.0, -1e9]]]);
  const lmScore = new Map([["", 0.0]]);
  for (let t = 0; t < T; t++) {
    const base = t * C;
    const p0 = probs[base];
    const cands = [];
    if (p0 >= 0.98) {
      cands.push(0);
    } else {
      let bestC = 0, bestP = -1;
      for (let c = 0; c < C; c++) {
        if (c > 0 && blockFn && blockFn(c)) continue;
        const p = probs[base + c];
        if (p > bestP) { bestP = p; bestC = c; }
        if (p > 0.025) cands.push(c);
      }
      if (!cands.length) cands.push(bestC);
    }
    if (cands.length === 1 && cands[0] === 0) {
      const lp0 = Math.log10(Math.max(p0, 1e-12));
      const next = new Map();
      for (const [pref, [pb, pnb]] of beams) next.set(pref, [_lse10(pb, pnb) + lp0, -1e9]);
      beams = next;
      continue;
    }
    const nxt = new Map();
    const get = (k) => { let v = nxt.get(k); if (!v) { v = [-1e9, -1e9]; nxt.set(k, v); } return v; };
    for (const [pref, [pb, pnb]] of beams) {
      const tot = _lse10(pb, pnb);
      const last = pref.length ? pref[pref.length - 1] : "";
      for (let ci = 0; ci < cands.length; ci++) {
        const c = cands[ci];
        const lp = Math.log10(Math.max(probs[base + c], 1e-12));
        if (c === 0) { const e = get(pref); e[0] = _lse10(e[0], tot + lp); continue; }
        if (c >= chars.length) continue;
        const ch = chars[c];
        const np = pref + ch;
        if (!lmScore.has(np)) lmScore.set(np, (lmScore.get(pref) || 0) + (lm ? alpha * lm.score(pref, ch) + beta : 0));
        const e = get(np);
        if (ch === last) {
          e[1] = _lse10(e[1], pb + lp);
          const s = get(pref); s[1] = _lse10(s[1], pnb + lp);
        } else {
          e[1] = _lse10(e[1], tot + lp);
        }
      }
    }
    const ranked = Array.from(nxt.entries()).sort((a, b) =>
      (_lse10(b[1][0], b[1][1]) + (lmScore.get(b[0]) || 0)) - (_lse10(a[1][0], a[1][1]) + (lmScore.get(a[0]) || 0))
    );
    beams = new Map(ranked.slice(0, beamW));
  }
  let best = "", bestS = -1e18;
  for (const [pref, [pb, pnb]] of beams) {
    const sc = _lse10(pb, pnb) + (lmScore.get(pref) || 0);
    if (sc > bestS) { bestS = sc; best = pref; }
  }
  return best;
}

/**
 * MaeBounme Lao-OCR Dual-Model Offline Engine (ONNX Runtime Web / WASM)
 * - Shared Detector: PP-OCRv6 DB Text-Line Detector (models/det.onnx — OpenThai 2.2)
 * - Lao Model:       MaeBounme Round 18 PP-OCRv6 Small SVTR (models/rec.onnx, 595 classes) + 5-gram CharLM Beam Search
 * - Thai Model:      Thai Round 3 PP-OCRv6 Small SVTR (models/thai_rec.onnx, 99.5% Thai)
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
    this.charLM = null;
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
    try {
      const lmBuf = await (await fetch(chrome.runtime.getURL("models/lao_charlm_mobile.bin"))).arrayBuffer();
      this.charLM = new CompactCharLM(lmBuf);
    } catch (_) {
      this.charLM = null;
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

  /**
   * Remove red/blue rubber stamps & faint gray diagonal (45 deg) watermarks without
   * damaging real dark text or thin Lao tone marks (່ ້).
   */
  cleanStampsAndWatermarks(srcCanvas) {
    const W = srcCanvas.width;
    const H = srcCanvas.height;
    const out = document.createElement("canvas");
    out.width = W;
    out.height = H;
    const ctx = out.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(srcCanvas, 0, 0);
    const imgData = ctx.getImageData(0, 0, W, H);
    const d = imgData.data;

    // Estimate paper brightness (92nd percentile)
    const hist = new Uint32Array(256);
    const n = W * H;
    for (let i = 0; i < d.length; i += 4) {
      hist[(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0]++;
    }
    let paper = 250, acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= n * 0.92) { paper = i; break; }
    }
    if (paper < 155) return srcCanvas; // dark-mode screen: leave untouched

    let changed = false;
    // Pass 1: remove red & blue rubber stamps without touching dark black ink
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      const isRed = (r - Math.max(g, b)) > 36 && y > 75;
      const isBlue = (b - Math.max(r, g)) > 44 && y > 75;
      if (isRed || isBlue) {
        d[i] = d[i + 1] = d[i + 2] = paper;
        changed = true;
      }
    }

    // Pass 2: suppress LARGE faint-gray diagonal watermarks while NEVER touching dark text or tiny 2-4px tone marks (່ ້)
    let darkCount = 0;
    const darkHist = new Uint32Array(256);
    const darkLim = Math.max(40, paper - 50);
    for (let i = 0; i < d.length; i += 4) {
      const y = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      if (y < darkLim) { darkHist[y]++; darkCount++; }
    }
    if (darkCount > 200) {
      let inkMed = 40, dAcc = 0;
      for (let i = 0; i < darkLim; i++) {
        dAcc += darkHist[i];
        if (dAcc >= darkCount * 0.35) { inkMed = i; break; }
      }
      const wmLo = Math.min(paper - 18, Math.max(168, inkMed + 95));
      const wmHi = paper - 6;
      if (wmHi > wmLo + 8) {
        // 2x-downsampled connected-component filter so it runs in <10ms in browser WASM/JS
        const sw = Math.ceil(W / 2);
        const sh = Math.ceil(H / 2);
        const mask = new Uint8Array(sw * sh);
        for (let sy = 0; sy < sh; sy++) {
          const py = Math.min(H - 1, sy * 2);
          const rowOff = py * W;
          const mOff = sy * sw;
          for (let sx = 0; sx < sw; sx++) {
            const p = (rowOff + Math.min(W - 1, sx * 2)) * 4;
            const y = (0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2]) | 0;
            if (y >= wmLo && y <= wmHi) mask[mOff + sx] = 1;
          }
        }
        const vis = new Uint8Array(sw * sh);
        const erase = new Uint8Array(sw * sh);
        const q = new Int32Array(sw * sh);
        const pts = new Int32Array(sw * sh);
        for (let idx = 0; idx < sw * sh; idx++) {
          if (!mask[idx] || vis[idx]) continue;
          let head = 0, tail = 0, cnt = 0;
          q[tail++] = idx;
          vis[idx] = 1;
          let minX = sw, maxX = 0, minY = sh, maxY = 0;
          while (head < tail) {
            const cur = q[head++];
            pts[cnt++] = cur;
            const cx = cur % sw, cy = (cur - cx) / sw;
            if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
            if (cx > 0 && mask[cur - 1] && !vis[cur - 1]) { vis[cur - 1] = 1; q[tail++] = cur - 1; }
            if (cx + 1 < sw && mask[cur + 1] && !vis[cur + 1]) { vis[cur + 1] = 1; q[tail++] = cur + 1; }
            if (cy > 0 && mask[cur - sw] && !vis[cur - sw]) { vis[cur - sw] = 1; q[tail++] = cur - sw; }
            if (cy + 1 < sh && mask[cur + sw] && !vis[cur + sw]) { vis[cur + sw] = 1; q[tail++] = cur + sw; }
          }
          // Only erase large faint-gray blobs (>= 220 full-res px & >= 24px span)
          if (cnt >= 55 && ((maxX - minX + 1) >= 12 || (maxY - minY + 1) >= 12)) {
            for (let k = 0; k < cnt; k++) erase[pts[k]] = 1;
          }
        }
        for (let py = 0; py < H; py++) {
          const mOff = (py >> 1) * sw;
          const rowOff = py * W;
          for (let px = 0; px < W; px++) {
            if (erase[mOff + (px >> 1)]) {
              const p = (rowOff + px) * 4;
              const y = (0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2]) | 0;
              if (y >= wmLo - 8) {
                d[p] = d[p + 1] = d[p + 2] = paper;
                changed = true;
              }
            }
          }
        }
      }
    }

    if (!changed) return srcCanvas;
    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  /**
   * Recursive XY-Cut to split 2-page book spreads (left page -> right page) and
   * 4-page 2x2 grids into reading-order blocks so columns NEVER merge left-to-right!
   */
  splitMultiPageBlocks(boxes, lh, depth = 0) {
    if (boxes.length < 4 || depth >= 4) return [boxes];
    let bx1 = 1e9, by1 = 1e9, bx2 = -1e9, by2 = -1e9;
    for (const b of boxes) {
      if (b[0] < bx1) bx1 = b[0];
      if (b[1] < by1) by1 = b[1];
      if (b[2] > bx2) bx2 = b[2];
      if (b[3] > by2) by2 = b[3];
    }
    const bw = bx2 - bx1;
    const bh = by2 - by1;

    // 1) Horizontal split for 4-page (2x2) grids
    if (bh > 12 * lh) {
      const ys = [...boxes].sort((a, b) => (a[1] + a[3]) * 0.5 - (b[1] + b[3]) * 0.5);
      let bestGap = 0, splitIdx = -1, maxY2 = ys[0][3];
      for (let k = 1; k < ys.length; k++) {
        const gap = ys[k][1] - maxY2;
        const midY = 0.5 * (maxY2 + ys[k][1]);
        if (gap > bestGap && k >= 3 && (ys.length - k) >= 3 && midY > by1 + 0.25 * bh && midY < by1 + 0.75 * bh) {
          bestGap = gap;
          splitIdx = k;
        }
        if (ys[k][3] > maxY2) maxY2 = ys[k][3];
      }
      if (splitIdx > 0 && bestGap >= Math.max(3.2 * lh, 0.06 * bh)) {
        return [
          ...this.splitMultiPageBlocks(ys.slice(0, splitIdx), lh, depth + 1),
          ...this.splitMultiPageBlocks(ys.slice(splitIdx), lh, depth + 1)
        ];
      }
    }

    // 2) Full-height vertical gutter for 2-page book spreads / 2-column pages
    if (bw > 10 * lh) {
      const bins = Math.max(16, Math.floor(bw / 2));
      const occ = new Uint16Array(bins);
      for (const b of boxes) {
        const c0 = Math.max(0, Math.floor((b[0] - bx1) / 2));
        const c1 = Math.min(bins, Math.ceil((b[2] - bx1) / 2));
        for (let c = c0; c < c1; c++) occ[c]++;
      }
      const loB = Math.floor(bins * 0.25);
      const hiB = Math.floor(bins * 0.75);
      const maxCross = boxes.length >= 10 ? 1 : 0;
      let bestG = 0, bestMidX = -1;
      for (const thr of (maxCross > 0 ? [0, maxCross] : [0])) {
        let s = -1;
        for (let bi = loB; bi < hiB; bi++) {
          if (occ[bi] <= thr) {
            if (s === -1) s = bi;
          } else if (s !== -1) {
            const gw = (bi - s) * 2;
            if (gw > bestG) { bestG = gw; bestMidX = bx1 + (s + bi); }
            s = -1;
          }
        }
        if (s !== -1) {
          const gw = (hiB - s) * 2;
          if (gw > bestG) { bestG = gw; bestMidX = bx1 + (s + hiB); }
        }
        if (bestMidX > 0 && bestG >= Math.max(1.5 * lh, 0.025 * bw)) break;
      }
      if (bestMidX > 0 && bestG >= Math.max(1.5 * lh, 0.025 * bw)) {
        const left = boxes.filter((b) => (b[0] + b[2]) * 0.5 < bestMidX);
        const right = boxes.filter((b) => (b[0] + b[2]) * 0.5 >= bestMidX);
        if (left.length >= 3 && right.length >= 3) {
          return [
            ...this.splitMultiPageBlocks(left, lh, depth + 1),
            ...this.splitMultiPageBlocks(right, lh, depth + 1)
          ];
        }
      }
    }
    return [boxes];
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
      if (acc >= n * 0.01) { lo = i; break; }
    }
    acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= n * 0.985) { hi = i; break; }
    }
    const span = hi - lo;
    // Do NOT bleach clean scans / high-contrast pages (preserves faint thin Lao tone marks ່ ້!)
    if (span >= 155 || span < 20) return srcCanvas;
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
          bObj._coreY1 = minY * sy;
          bObj._coreY2 = (maxY + 1) * sy;
          bObj._coreH = Math.max(4, bh * sy);
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

    // Smart Ink-Aware + Neighbor-Clamped Box Refinement:
    // prevents clipping Lao tall stems / top marks / bottom vowels (ຸ ູ ຼ) while NEVER swallowing adjacent lines
    this.refineBoxesInkAware(canvas, boxes);

    // Filter out giant icon/watermark/ornament boxes (square blobs > 2.8x median line height)
    const hsAll = boxes.map((b) => b[3] - b[1]).sort((a, b) => a - b);
    const medLineH = Math.max(10, hsAll[Math.floor(hsAll.length / 2)] || 22);
    const filtered = boxes.filter((b) => {
      const bwB = b[2] - b[0];
      const bhB = b[3] - b[1];
      const isGiantIcon = bhB > medLineH * 3.0 && bwB / Math.max(1, bhB) < 2.2;
      return !isGiantIcon;
    });
    const useBoxes = filtered.length ? filtered : boxes;

    // Automatic 2-page book spread / 4-page 2x2 grid / multi-column XY-Cut ordering!
    const blocks = this.splitMultiPageBlocks(useBoxes, medLineH);
    const rows = [];
    for (let bi = 0; bi < blocks.length; bi++) {
      const blk = [...blocks[bi]].sort((a, b) => (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2);
      const blkRows = [];
      for (const b of blk) {
        let placed = false;
        for (const row of blkRows) {
          const ref = row[0];
          const ov = Math.min(ref[3], b[3]) - Math.max(ref[1], b[1]);
          const minH = Math.min(ref[3] - ref[1], b[3] - b[1]);
          if (ov > 0.45 * minH) {
            row.push(b);
            placed = true;
            break;
          }
        }
        if (!placed) blkRows.push([b]);
      }
      for (const row of blkRows) {
        row.sort((a, b) => a[0] - b[0]);
        row._blockIdx = bi;
      }
      rows.push(...blkRows);
    }
    return rows;
  }

  /**
   * Smart Ink-Aware + Neighbor-Clamped Box Refinement for Lao & Thai:
   *  1) Clamps each box's vertical expansion at the midpoint to neighboring lines above/below
   *  2) Scans actual ink pixels inside the safe band to wrap tall stems (ປ ຝ ຟ ໂ ໃ ໄ),
   *     stacked top marks (ິ້ ີ່ ັ້), and bottom vowels (ຸ ູ ຼ) without bloating plain lines.
   */
  refineBoxesInkAware(canvas, boxes) {
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const fullData = ctx.getImageData(0, 0, W, H).data;

    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const coreY1 = b._coreY1 !== undefined ? b._coreY1 : b[1];
      const coreY2 = b._coreY2 !== undefined ? b._coreY2 : b[3];
      const coreH = Math.max(6, b._coreH || (coreY2 - coreY1));
      const x1 = Math.max(0, Math.floor(b[0]));
      const x2 = Math.min(W, Math.ceil(b[2]));
      if (x2 - x1 < 6) continue;

      // 1) Clamp ceiling/floor at midpoint to nearest horizontally-overlapping neighbor above/below
      let ceilY = Math.max(0, coreY1 - coreH * 0.95);
      let floorY = Math.min(H, coreY2 + coreH * 0.80);
      for (let j = 0; j < boxes.length; j++) {
        if (j === i) continue;
        const ob = boxes[j];
        const ovX = Math.min(b[2], ob[2]) - Math.max(b[0], ob[0]);
        if (ovX < 0.20 * Math.min(b[2] - b[0], ob[2] - ob[0])) continue;
        const oy1 = ob._coreY1 !== undefined ? ob._coreY1 : ob[1];
        const oy2 = ob._coreY2 !== undefined ? ob._coreY2 : ob[3];
        if (oy2 <= coreY1 + 0.15 * coreH) {
          const mid = 0.5 * (oy2 + coreY1);
          if (mid > ceilY) ceilY = mid;
        } else if (oy1 >= coreY2 - 0.15 * coreH) {
          const mid = 0.5 * (coreY2 + oy1);
          if (mid < floorY) floorY = mid;
        }
      }

      const ry1 = Math.max(0, Math.floor(ceilY));
      const ry2 = Math.min(H, Math.ceil(floorY));
      const rh = ry2 - ry1;
      const rw = x2 - x1;
      if (rh < 6) continue;

      // 2) Scan EVERY pixel horizontally (stride=1, minCols=1) so tiny 2-3px Lao tone marks (່ ້)
      //    are NEVER skipped!
      let sumLum = 0, minLum = 255, maxLum = 0, nLum = 0;
      const bgStride = Math.max(1, Math.floor(rw / 120));
      for (let y = ry1; y < ry2; y += Math.max(1, Math.floor(rh / 30))) {
        const rowOff = y * W;
        for (let x = x1; x < x2; x += bgStride) {
          const p = (rowOff + x) * 4;
          const lum = 0.299 * fullData[p] + 0.587 * fullData[p + 1] + 0.114 * fullData[p + 2];
          sumLum += lum;
          if (lum < minLum) minLum = lum;
          if (lum > maxLum) maxLum = lum;
          nLum++;
        }
      }
      const meanLum = nLum > 0 ? sumLum / nLum : 240;
      const isLightBg = meanLum >= 127;
      const bg = isLightBg ? Math.min(255, maxLum - 4) : Math.max(0, minLum + 4);
      const thr = Math.max(20, 0.16 * Math.abs(bg - (isLightBg ? minLum : maxLum)));

      const rowInk = new Uint8Array(rh);
      for (let r = 0; r < rh; r++) {
        const rowOff = (ry1 + r) * W;
        for (let x = x1; x < x2; x++) {
          const p = (rowOff + x) * 4;
          const lum = 0.299 * fullData[p] + 0.587 * fullData[p + 1] + 0.114 * fullData[p + 2];
          if (Math.abs(lum - bg) > thr) {
            rowInk[r] = 1;
            break;
          }
        }
      }

      // Walk upward from coreY1 and downward from coreY2 allowing floating tone-mark gap (<= 0.55 * coreH)
      const coreTopRel = Math.max(0, Math.min(rh - 1, Math.floor(coreY1 - ry1)));
      const coreBotRel = Math.max(0, Math.min(rh - 1, Math.ceil(coreY2 - ry1)));
      const maxGap = Math.max(5, Math.round(0.55 * coreH));

      let snapTop = coreTopRel;
      let gap = 0;
      for (let r = coreTopRel; r >= 0; r--) {
        if (rowInk[r]) { snapTop = r; gap = 0; }
        else if (++gap > maxGap) break;
      }
      let snapBot = coreBotRel;
      gap = 0;
      for (let r = coreBotRel; r < rh; r++) {
        if (rowInk[r]) { snapBot = r; gap = 0; }
        else if (++gap > maxGap) break;
      }

      // Guarantee generous minimum top/bottom margin (NEVER shrink below 0.62*coreH top / 0.45*coreH bottom,
      // clamped only by neighbor midpoint ceilY / floorY) so stacked Lao tone marks (່ ້) never clip!
      const breath = Math.max(3.5, 0.14 * coreH);
      const safeMinY1 = Math.max(ceilY, coreY1 - coreH * 0.62);
      const safeMaxY2 = Math.min(floorY, coreY2 + coreH * 0.45);
      const newY1 = Math.max(ceilY, Math.min(safeMinY1, ry1 + snapTop - breath));
      const newY2 = Math.min(floorY, Math.max(safeMaxY2, ry1 + snapBot + 1 + breath));
      if (newY2 - newY1 >= 6) {
        const dTop = newY1 - b[1];
        const dBot = newY2 - b[3];
        b[1] = Math.floor(newY1);
        b[3] = Math.ceil(newY2);
        if (b._colTop && b._colBot) {
          for (let c = 0; c < b._colTop.length; c++) {
            b._colTop[c] = Math.max(ceilY, Math.min(H - 2, b._colTop[c] + dTop));
            b._colBot[c] = Math.min(floorY, Math.max(b._colTop[c] + 4, b._colBot[c] + dBot));
          }
          b._medH = Math.max(8, newY2 - newY1);
        }
      }
    }
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
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, newW, H);

    // Only slice-dewarp if the line is genuinely tilted/curved (variation >= max(5px, 14% of height));
    // on straight book/PDF lines, a single smooth drawImage preserves thin 2px Lao tone marks (່ ້)!
    let needsDewarp = false;
    if (box._colTop && box._colBot && box._colTop.length >= 10) {
      let minT = 1e9, maxT = -1e9;
      for (let c = 0; c < box._colTop.length; c++) {
        if (box._colTop[c] < minT) minT = box._colTop[c];
        if (box._colTop[c] > maxT) maxT = box._colTop[c];
      }
      needsDewarp = (maxT - minT) >= Math.max(5, ch * 0.14);
    }

    if (needsDewarp) {
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
    const rawOut = out.data;

    const toSoftmax = (src) => {
      const sm = new Float32Array(T * C);
      for (let t = 0; t < T; t++) {
        const base = t * C;
        let maxV = -1e9, sumV = 0;
        for (let c = 0; c < C; c++) { const v = src[base + c]; if (v > maxV) maxV = v; sumV += v; }
        if (maxV <= 1.001 && sumV > 0.95 && sumV < 1.05) {
          for (let c = 0; c < C; c++) sm[base + c] = Math.max(0, src[base + c]);
        } else {
          let sumE = 0;
          for (let c = 0; c < C; c++) { const e = Math.exp(Math.max(-30, src[base + c] - maxV)); sm[base + c] = e; sumE += e; }
          const inv = sumE > 0 ? 1 / sumE : 1 / C;
          for (let c = 0; c < C; c++) sm[base + c] *= inv;
        }
      }
      return sm;
    };

    let probs = toSoftmax(rawOut);

    let blockFn = null;
    if (modelMode === "lao") {
      blockFn = (c) => this.laoIsThai[c] || this.laoIsLatin[c];
    } else if (modelMode === "lao+en") {
      blockFn = (c) => this.laoIsThai[c];
    }

    const evalConf = (pArr) => {
      let cSum = 0, cCnt = 0, prevC = -1, gTxt = "";
      for (let t = 0; t < T; t++) {
        const base = t * C;
        let bI = 0, bV = -1;
        for (let c = 0; c < C; c++) {
          if (blockFn && c > 0 && blockFn(c)) continue;
          const v = pArr[base + c];
          if (v > bV) { bV = v; bI = c; }
        }
        if (bI !== 0 && bI !== prevC && bI < chars.length) {
          gTxt += chars[bI];
          cSum += bV;
          cCnt++;
        }
        prevC = bI;
      }
      return { conf: cCnt > 0 ? cSum / cCnt : 0, greedy: gTxt };
    };

    let { conf, greedy } = evalConf(probs);
    // Low-Confidence TTA: if mean conf < 0.88, re-read with slightly wider top/bottom margin and average probs
    if (conf < 0.88 && !needsDewarp && (y2 - y1) >= 12) {
      const ty1 = Math.max(0, y1 - Math.max(3, Math.round((y2 - y1) * 0.10)));
      const ty2 = Math.min(canvas.height, y2 + Math.max(2, Math.round((y2 - y1) * 0.08)));
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, newW, H);
      ctx.drawImage(canvas, x1, ty1, cw, Math.max(1, ty2 - ty1), 0, 0, newW, H);
      const d2 = ctx.getImageData(0, 0, newW, H).data;
      const in2 = new Float32Array(3 * hw);
      for (let i = 0; i < hw; i++) {
        const v = (0.299 * d2[i * 4] + 0.587 * d2[i * 4 + 1] + 0.114 * d2[i * 4 + 2]) / 127.5 - 1.0;
        in2[i] = v; in2[hw + i] = v; in2[2 * hw + i] = v;
      }
      const res2 = await sess.run({ x: new ort.Tensor("float32", in2, [1, 3, H, newW]) });
      const out2 = res2[sess.outputNames[0]];
      if (out2.dims[1] === T && out2.dims[2] === C) {
        const p2 = toSoftmax(out2.data);
        for (let i = 0; i < probs.length; i++) probs[i] = 0.5 * (probs[i] + p2[i]);
        const ev2 = evalConf(probs);
        conf = ev2.conf;
        greedy = ev2.greedy;
      }
    }

    this._lastConf = conf;
    if (useThai) return greedy.trim();

    // 5-gram Lao CharLM CTC Prefix Beam Search (resolves ຜ່ານ/ຜານ, ຕ່າງໆ/ຕາງໆ, ໄດ້ແຕ່ງຕັ້ງ, ງ/າ!)
    let text = greedy;
    if (/[\u0E80-\u0EFF]/.test(greedy) && this.charLM) {
      const beamBlock = blockFn || ((c) => this.laoIsThai[c]);
      text = beamDecodeCTC(probs, T, C, chars, this.charLM, beamBlock, 0.36, 0.72, 8);
    }
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
      "ງ": ["າ"], "າ": ["ງ"], "ໂ": ["ໃ", "ໄ"], "ໃ": ["ໄ", "ໂ"], "ໄ": ["ໃ", "ໂ"],
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
          const score = rank + (tok[i] === "ຂ" || tok[i] === "ຊ" || tok[i] === "ງ" || tok[i] === "າ" ? 0 : 1200);
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

  /** Format structured row boxes into 4 layout modes: 'paragraph' | 'exact' | 'markdown' | 'two_col'
   *  Respects multi-page / multi-column block indices (blockIdx) from splitMultiPageBlocks()
   *  so 2-page book spreads and 4-page grids NEVER merge left-to-right!
   */
  formatLayout(pageItems, pageW, mode = "paragraph") {
    // pageItems: array of { box: [x1,y1,x2,y2], text: string, blockIdx?: number }
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

    // Partition items by detected page/column blockIdx (preserving reading order 0, 1, 2, ...)
    const blockMap = new Map();
    for (const it of pageItems) {
      const bi = it.blockIdx || 0;
      if (!blockMap.has(bi)) blockMap.set(bi, []);
      blockMap.get(bi).push(it);
    }
    const blockLists = [...blockMap.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1]);

    const formatBlock = (blkItems) => {
      const rows = groupRows(blkItems);
      if (!rows.length) return "";

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
        const isIndented = x1 - bodyLeft > lh * 1.15 && r.length === 1;
        const bigGap = prevY2 !== null && y1 - prevY2 > lh * 1.15;
        if ((bigGap || isIndented) && out.length && out[out.length - 1] !== "") {
          out.push("");
        }
        out.push((isIndented ? "    " : "") + txt);
        prevY2 = y2;
      }
      return out.join("\n").trim();
    };

    const sep = mode === "markdown" && blockLists.length > 1 ? "\n\n---\n\n" : "\n\n";
    return blockLists.map(formatBlock).filter(Boolean).join(sep);
  }

  async readCanvas(canvas, modelMode = "auto", a4Mode = true, onStatus = null, layoutMode = "paragraph", onProgress = null, shouldCancel = null) {
    const t0 = performance.now();
    const cleanCanvas = a4Mode ? this.cleanStampsAndWatermarks(canvas) : canvas;
    const workCanvas = a4Mode ? this.enhanceA4Canvas(cleanCanvas) : cleanCanvas;
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
          items.push({ box, text: txt, conf: this._lastConf || 0, blockIdx: row._blockIdx || 0 });
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
