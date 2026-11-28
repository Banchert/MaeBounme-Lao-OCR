/**
 * Real 100% Offline Lao + Thai + English OCR Engine for Android & iOS
 * Powered by onnxruntime-react-native + pure-JS jpeg-js pixel decoder:
 *  - Detector : PP-OCRv6 DB Detector (det.onnx)
 *  - Lao Model: MaeBounme Lao-OCR Round 15 All-Conditions Master (rec.onnx, Apache-2.0)
 *  - Thai Model: MaeBounme Thai Print R2 + Siam Rath Tripitaka (thai_rec.onnx, Apache-2.0)
 */
import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system";
import * as ImageManipulator from "expo-image-manipulator";
import { Buffer } from "buffer";
import { beamDecodeCTC, CompactCharLM } from "./charLm";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const jpeg = require("jpeg-js");
// eslint-disable-next-line @typescript-eslint/no-var-requires
const laoLexiconData: { w?: string[] } = require("../../assets/models/lao_lexicon.json");

const LAO_CONFUSABLES: Record<string, string[]> = {
  "ຂ": ["ຊ"],
  "ຊ": ["ຂ"],
  "ບ": ["ປ"],
  "ປ": ["ບ"],
  "ຜ": ["ຝ"],
  "ຝ": ["ຜ"],
  "ພ": ["ຟ"],
  "ຟ": ["ພ"],
  "ຄ": ["ດ"],
  "ດ": ["ຄ", "ຕ"],
  "ຕ": ["ດ"],
  "ທ": ["ຫ"],
  "ຫ": ["ທ"],
  "ນ": ["ມ"],
  "ມ": ["ນ"],
  "ງ": ["າ"],
  "າ": ["ງ"],
  "ໂ": ["ໃ", "ໄ"],
  "ໃ": ["ໄ", "ໂ"],
  "ໄ": ["ໃ", "ໂ"],
  "່": ["້"],
  "້": ["່"],
  "ຸ": ["ູ"],
  "ູ": ["ຸ"]
};
const LAO_INSERT_MARKS = ["ຸ", "ູ", "່", "້", "ັ", "ົ", "ິ", "ີ", "ຶ", "ື"];
const RE_LAO_CONS = /^[\u0E81-\u0EAE]$/;
const RE_LAO_TOP_VOWEL = /^[\u0EB1\u0EB4-\u0EB7\u0EBB]$/;

let laoLexMap: Map<string, number> | null = null;
function getLaoLexicon(): Map<string, number> {
  if (!laoLexMap) {
    laoLexMap = new Map<string, number>();
    const words = laoLexiconData?.w || [];
    for (let i = 0; i < words.length; i++) {
      laoLexMap.set(words[i], i);
    }
  }
  return laoLexMap;
}

function repairLaoToken(tok: string, lex: Map<string, number>): string {
  if (tok.length < 2 || tok.length > 14 || lex.has(tok)) return tok;
  let bestWord = tok;
  let bestScore = Infinity;

  // 1) Visual confusable 1-char substitutions (especially ຂ↔ຊ, ບ↔ປ, ຜ↔ຝ, ພ↔ຟ, ່↔້, ຸ↔ູ)
  for (let i = 0; i < tok.length; i++) {
    const alts = LAO_CONFUSABLES[tok[i]];
    if (!alts) continue;
    for (const alt of alts) {
      const cand = tok.slice(0, i) + alt + tok.slice(i + 1);
      const rank = lex.get(cand);
      if (rank !== undefined && rank < 22000) {
        const penalty = (tok[i] === "ຂ" || tok[i] === "ຊ") ? 0 : 1200;
        const score = rank + penalty;
        if (score < bestScore) {
          bestScore = score;
          bestWord = cand;
        }
      }
    }
  }

  // 2) Missing bottom vowel (ຸ ູ) or tall tone/top mark (່ ້ ັ ົ ິ ີ ຶ ື)
  for (let i = 0; i < tok.length; i++) {
    const ch = tok[i];
    const next = tok[i + 1] || "";
    if (!RE_LAO_CONS.test(ch) && !RE_LAO_TOP_VOWEL.test(ch)) continue;
    const marks = RE_LAO_TOP_VOWEL.test(ch) ? ["່", "້"] : LAO_INSERT_MARKS;
    for (const m of marks) {
      if (next === m) continue;
      const cand = tok.slice(0, i + 1) + m + tok.slice(i + 1);
      const rank = lex.get(cand);
      if (rank !== undefined && rank < 10000) {
        const score = rank + 2500;
        if (score < bestScore) {
          bestScore = score;
          bestWord = cand;
        }
      }
    }
  }

  return bestWord;
}

/** Keyboard-style predictive Lao word suggestions from the 33,604-word Master Lexicon */
export function suggestLaoWords(rawWord: string, limit = 6): string[] {
  const tok = (rawWord || "").trim();
  const lex = getLaoLexicon();
  if (!lex.size || tok.length < 1) return [];

  const cands = new Map<string, number>();
  const add = (w: string, score: number) => {
    if (w === tok) return;
    const prev = cands.get(w);
    if (prev === undefined || score < prev) cands.set(w, score);
  };

  // 1) Confusable consonant/mark swaps (ຂ↔ຊ, ບ↔ປ, ່↔້, ຸ↔ູ...)
  for (let i = 0; i < tok.length; i++) {
    const alts = LAO_CONFUSABLES[tok[i]];
    if (!alts) continue;
    for (const alt of alts) {
      const w = tok.slice(0, i) + alt + tok.slice(i + 1);
      const r = lex.get(w);
      if (r !== undefined) add(w, r + (tok[i] === "ຂ" || tok[i] === "ຊ" ? 0 : 800));
    }
  }

  // 2) Missing bottom vowel (ຸ ູ) or top/tone mark (່ ້ ັ ົ ິ ີ ຶ ື)
  for (let i = 0; i < tok.length; i++) {
    const ch = tok[i];
    if (!RE_LAO_CONS.test(ch) && !RE_LAO_TOP_VOWEL.test(ch)) continue;
    const marks = RE_LAO_TOP_VOWEL.test(ch) ? ["່", "້"] : LAO_INSERT_MARKS;
    for (const m of marks) {
      const w = tok.slice(0, i + 1) + m + tok.slice(i + 1);
      const r = lex.get(w);
      if (r !== undefined) add(w, r + 1500);
    }
  }

  // 3) Extra mark deletion (1-char drop)
  if (tok.length >= 3) {
    for (let i = 0; i < tok.length; i++) {
      const w = tok.slice(0, i) + tok.slice(i + 1);
      const r = lex.get(w);
      if (r !== undefined) add(w, r + 2200);
    }
  }

  // 4) Keyboard prefix autocomplete (if fewer than limit candidates found)
  if (cands.size < limit && tok.length >= 2) {
    const words = laoLexiconData?.w || [];
    for (let i = 0; i < words.length && cands.size < limit * 3; i++) {
      const w = words[i];
      if (w.length > tok.length && w.startsWith(tok)) {
        add(w, i + 4000);
      }
    }
  }

  return Array.from(cands.entries())
    .sort((a, b) => a[1] - b[1])
    .slice(0, limit)
    .map(([w]) => w);
}

/** Find non-dictionary Lao words in OCR output that have strong keyboard fix suggestions */
export function findSuspectLaoWords(text: string, maxItems = 10): { word: string; suggestions: string[] }[] {
  const lex = getLaoLexicon();
  if (!lex.size) return [];
  const seen = new Set<string>();
  const out: { word: string; suggestions: string[] }[] = [];
  const matches = text.match(/[\u0E80-\u0EFF]{2,14}/g) || [];
  for (const w of matches) {
    if (seen.has(w) || lex.has(w)) continue;
    seen.add(w);
    const sug = suggestLaoWords(w, 4);
    if (sug.length > 0) {
      out.push({ word: w, suggestions: sug });
      if (out.length >= maxItems) break;
    }
  }
  return out;
}

function correctLaoLine(line: string): string {
  const lex = getLaoLexicon();
  if (!lex.size || !/[\u0E80-\u0EFF]/.test(line)) return line;
  return line.replace(/[\u0E80-\u0EFF]{2,}/g, (span) => {
    if (lex.has(span)) return span;
    if (span.length <= 12) return repairLaoToken(span, lex);
    // Greedy longest-match segmentation for unspaced Lao text; repair short unknown chunks
    let out = "";
    let i = 0;
    while (i < span.length) {
      let matched = "";
      const maxL = Math.min(14, span.length - i);
      for (let len = maxL; len >= 2; len--) {
        const sub = span.slice(i, i + len);
        if (lex.has(sub)) {
          matched = sub;
          break;
        }
      }
      if (matched) {
        out += matched;
        i += matched.length;
      } else {
        // Collect unknown residual until next known word starts
        let j = i + 1;
        while (j < span.length && j - i < 10) {
          let foundNext = false;
          for (let len = Math.min(10, span.length - j); len >= 3; len--) {
            if (lex.has(span.slice(j, j + len))) {
              foundNext = true;
              break;
            }
          }
          if (foundNext) break;
          j++;
        }
        const chunk = span.slice(i, j);
        out += repairLaoToken(chunk, lex);
        i = j;
      }
    }
    return out;
  });
}

export type OcrModelMode = "lao" | "lao+en" | "thai";
export type OutputFormatMode = "plain" | "markdown";

export interface OcrResult {
  text: string;
  lineCount: number;
  ms: number;
  avgConf: number;
  /** Map of low-confidence words (< 85% CTC confidence) -> lowest confidence percentage (0-100) */
  lowConfWords: Record<string, number>;
}

interface RawRGBA {
  width: number;
  height: number;
  data: Uint8Array;
}

interface DetectedLine {
  box: [number, number, number, number];
  /** Smoothed per-column top/bottom Y in image coordinates for wavy/tilted line dewarping */
  colTop?: Float32Array;
  colBot?: Float32Array;
}

class MobileLaoOcrEngine {
  private detSession: any = null;
  private recSession: any = null;
  private thaiRecSession: any = null;
  private chars: string[] = [];
  private thaiChars: string[] = [];
  private ready = false;
  private thaiReady = false;
  private lastConf = 0;
  private lastLowConfWords: Record<string, number> = {};
  private charLM: CompactCharLM | null = null;

  async init(): Promise<void> {
    if (this.ready) return;
    const ort = require("onnxruntime-react-native");

    const dictAsset = Asset.fromModule(require("../../assets/models/rec_dict.txt"));
    const detAsset = Asset.fromModule(require("../../assets/models/det.onnx"));
    const recAsset = Asset.fromModule(require("../../assets/models/rec.onnx"));
    const lmAsset = Asset.fromModule(require("../../assets/models/lao_charlm_mobile.bin"));
    await Promise.all([dictAsset.downloadAsync(), detAsset.downloadAsync(), recAsset.downloadAsync(), lmAsset.downloadAsync()]);

    const dictText = await FileSystem.readAsStringAsync(dictAsset.localUri!);
    this.chars = ["", ...dictText.split(/\r?\n/).filter(Boolean), " "];

    try {
      const lmB64 = await FileSystem.readAsStringAsync(lmAsset.localUri!, { encoding: FileSystem.EncodingType.Base64 });
      this.charLM = new CompactCharLM(Buffer.from(lmB64, "base64"));
    } catch (_) {
      this.charLM = null;
    }

    this.detSession = await ort.InferenceSession.create(detAsset.localUri!);
    this.recSession = await ort.InferenceSession.create(recAsset.localUri!);
    this.ready = true;
  }

  async ensureThaiModel(): Promise<void> {
    if (this.thaiReady) return;
    const ort = require("onnxruntime-react-native");
    const dictAsset = Asset.fromModule(require("../../assets/models/thai_rec_dict.txt"));
    const recAsset = Asset.fromModule(require("../../assets/models/thai_rec.onnx"));
    await Promise.all([dictAsset.downloadAsync(), recAsset.downloadAsync()]);
    const dictText = await FileSystem.readAsStringAsync(dictAsset.localUri!);
    this.thaiChars = ["", ...dictText.split(/\r?\n/).filter(Boolean), " "];
    this.thaiRecSession = await ort.InferenceSession.create(recAsset.localUri!);
    this.thaiReady = true;
  }

  /** Rotate & resize image to high-res JPEG (1600px, 96%), decode RGBA, and apply gentle shape-preserving clarity boost */
  async decodeUriToRGBA(uri: string, rotateDeg = 0): Promise<RawRGBA> {
    const actions: ImageManipulator.Action[] = [];
    if (rotateDeg % 360 !== 0) actions.push({ rotate: rotateDeg });
    actions.push({ resize: { width: 1600 } });
    const res = await ImageManipulator.manipulateAsync(uri, actions, {
      compress: 0.96,
      format: ImageManipulator.SaveFormat.JPEG,
      base64: true
    });
    const rawBuf = Buffer.from(res.base64 || "", "base64");
    const decoded = jpeg.decode(rawBuf, { useTArray: true, formatAsRGBA: true });
    const raw: RawRGBA = { width: decoded.width, height: decoded.height, data: decoded.data };
    this.gentleEnhanceForOcr(raw);
    return raw;
  }

  /**
   * Gentle, shape-preserving clarity enhancement for blurry / low-contrast phone photos:
   *  1) Mild percentile contrast stretch (only if image is low-contrast)
   *  2) Auto blur check + thresholded 3x3 unsharp mask (sharpens blurry edges WITHOUT distorting glyph shapes)
   */
  private gentleEnhanceForOcr(img: RawRGBA): void {
    const { width: W, height: H, data } = img;
    const n = W * H;
    if (n < 64) return;

    const lum = new Uint8Array(n);
    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      const y = (0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2] + 0.5) | 0;
      lum[i] = y;
      hist[y]++;
    }

    // 1) Mild contrast stretch only if washed out / dim (preserves smooth anti-aliasing)
    let lo = 0, hi = 255, acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * 0.015) { lo = v; break; } }
    acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * 0.94) { hi = v; break; } }
    const span = hi - lo;
    const doStretch = span > 24 && span < 210;
    const scale = doStretch ? 255 / span : 1;

    // 2) Estimate sharpness (sampled Laplacian energy) to adapt unsharp strength gently
    let lapSum = 0, lapCnt = 0;
    const step = Math.max(2, Math.floor(Math.min(W, H) / 160));
    for (let y = 1; y < H - 1; y += step) {
      const row = y * W;
      for (let x = 1; x < W - 1; x += step) {
        const idx = row + x;
        const lap = Math.abs(4 * lum[idx] - lum[idx - 1] - lum[idx + 1] - lum[idx - W] - lum[idx + W]);
        lapSum += lap;
        lapCnt++;
      }
    }
    const avgLap = lapCnt > 0 ? lapSum / lapCnt : 10;
    // Blurry photo (low avgLap) -> slightly firmer unsharp (0.70); sharp photo -> very gentle (0.30)
    const sharpAmt = avgLap < 7.5 ? 0.70 : avgLap < 13 ? 0.48 : 0.28;

    // Apply thresholded 3x3 unsharp mask to luminance delta (keeps natural glyph thickness & shape!)
    for (let y = 1; y < H - 1; y++) {
      const row = y * W;
      for (let x = 1; x < W - 1; x++) {
        const idx = row + x;
        const c = lum[idx];
        const blur = (
          lum[idx - W - 1] + lum[idx - W] + lum[idx - W + 1] +
          lum[idx - 1]     + c            + lum[idx + 1] +
          lum[idx + W - 1] + lum[idx + W] + lum[idx + W + 1]
        ) / 9;
        const diff = c - blur;
        let outY = c;
        if (Math.abs(diff) > 4) {
          outY = c + diff * sharpAmt;
        }
        if (doStretch) {
          outY = (outY - lo) * scale;
        }
        const delta = Math.max(0, Math.min(255, Math.round(outY))) - c;
        if (delta !== 0) {
          const p = idx * 4;
          data[p]     = Math.max(0, Math.min(255, data[p] + delta));
          data[p + 1] = Math.max(0, Math.min(255, data[p + 1] + delta));
          data[p + 2] = Math.max(0, Math.min(255, data[p + 2] + delta));
        }
      }
    }
  }

  /** Bilinear sample RGB from a sub-rectangle into target (dstW, dstH) NCHW Float32Array */
  private cropResizeToNCHW(
    img: RawRGBA,
    sx: number,
    sy: number,
    sw: number,
    sh: number,
    dstW: number,
    dstH: number,
    mode: "det" | "rec"
  ): Float32Array {
    const out = new Float32Array(3 * dstH * dstW);
    const plane = dstH * dstW;
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const { width: W, height: H, data } = img;

    for (let y = 0; y < dstH; y++) {
      const fy = sy + ((y + 0.5) * sh) / dstH - 0.5;
      const y0 = Math.max(0, Math.min(H - 1, Math.floor(fy)));
      const y1 = Math.min(H - 1, y0 + 1);
      const wy = Math.max(0, Math.min(1, fy - y0));

      for (let x = 0; x < dstW; x++) {
        const fx = sx + ((x + 0.5) * sw) / dstW - 0.5;
        const x0 = Math.max(0, Math.min(W - 1, Math.floor(fx)));
        const x1 = Math.min(W - 1, x0 + 1);
        const wx = Math.max(0, Math.min(1, fx - x0));

        const i00 = (y0 * W + x0) * 4;
        const i10 = (y0 * W + x1) * 4;
        const i01 = (y1 * W + x0) * 4;
        const i11 = (y1 * W + x1) * 4;

        const w00 = (1 - wx) * (1 - wy);
        const w10 = wx * (1 - wy);
        const w01 = (1 - wx) * wy;
        const w11 = wx * wy;

        const r = (data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11) / 255;
        const g = (data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11) / 255;
        const b = (data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11) / 255;

        const idx = y * dstW + x;
        if (mode === "det") {
          out[idx] = (r - mean[0]) / std[0];
          out[plane + idx] = (g - mean[1]) / std[1];
          out[2 * plane + idx] = (b - mean[2]) / std[2];
        } else {
          out[idx] = (r - 0.5) / 0.5;
          out[plane + idx] = (g - 0.5) / 0.5;
          out[2 * plane + idx] = (b - 0.5) / 0.5;
        }
      }
    }
    return out;
  }

  /** Dewarp & crop a tilted or wavy text line along its smoothed column centerline into NCHW Float32Array */
  private cropDewarpToNCHW(img: RawRGBA, line: DetectedLine, dstW: number, dstH: number): Float32Array {
    const [x1, y1, x2, y2] = line.box;
    const sw = Math.max(1, x2 - x1);
    const sh = Math.max(1, y2 - y1);
    if (!line.colTop || !line.colBot || line.colTop.length < 2) {
      return this.cropResizeToNCHW(img, x1, y1, sw, sh, dstW, dstH, "rec");
    }

    const out = new Float32Array(3 * dstH * dstW);
    const plane = dstH * dstW;
    const { width: W, height: H, data } = img;
    const nCols = line.colTop.length;

    for (let x = 0; x < dstW; x++) {
      const u = (x + 0.5) / dstW;
      const fx = x1 + u * sw - 0.5;
      const x0 = Math.max(0, Math.min(W - 1, Math.floor(fx)));
      const x1p = Math.min(W - 1, x0 + 1);
      const wx = Math.max(0, Math.min(1, fx - x0));

      // Interpolate smoothed top & bottom contour at this column
      const cf = u * (nCols - 1);
      const c0 = Math.max(0, Math.min(nCols - 1, Math.floor(cf)));
      const c1 = Math.min(nCols - 1, c0 + 1);
      const wc = cf - c0;
      const topY = line.colTop[c0] * (1 - wc) + line.colTop[c1] * wc;
      const botY = line.colBot[c0] * (1 - wc) + line.colBot[c1] * wc;
      const colH = Math.max(6, botY - topY);

      for (let y = 0; y < dstH; y++) {
        const fy = topY + ((y + 0.5) * colH) / dstH - 0.5;
        const y0 = Math.max(0, Math.min(H - 1, Math.floor(fy)));
        const y1p = Math.min(H - 1, y0 + 1);
        const wy = Math.max(0, Math.min(1, fy - y0));

        const i00 = (y0 * W + x0) * 4;
        const i10 = (y0 * W + x1p) * 4;
        const i01 = (y1p * W + x0) * 4;
        const i11 = (y1p * W + x1p) * 4;

        const w00 = (1 - wx) * (1 - wy);
        const w10 = wx * (1 - wy);
        const w01 = (1 - wx) * wy;
        const w11 = wx * wy;

        const r = (data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11) / 255;
        const g = (data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11) / 255;
        const b = (data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11) / 255;

        const idx = y * dstW + x;
        out[idx] = (r - 0.5) / 0.5;
        out[plane + idx] = (g - 0.5) / 0.5;
        out[2 * plane + idx] = (b - 0.5) / 0.5;
      }
    }
    return out;
  }

  /** Run PP-OCRv6 DB Detector and return sorted text lines with wavy/tilted contour profiles */
  private async detectLines(img: RawRGBA): Promise<DetectedLine[]> {
    const ort = require("onnxruntime-react-native");
    const maxSide = 1280;
    const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
    const dw = Math.max(32, Math.round((img.width * scale) / 32) * 32);
    const dh = Math.max(32, Math.round((img.height * scale) / 32) * 32);

    const nchw = this.cropResizeToNCHW(img, 0, 0, img.width, img.height, dw, dh, "det");
    const inputName = this.detSession.inputNames[0];
    const outMap = await this.detSession.run({ [inputName]: new ort.Tensor("float32", nchw, [1, 3, dh, dw]) });
    const probTensor = outMap[this.detSession.outputNames[0]];
    const prob = probTensor.data as Float32Array;
    const mapH = probTensor.dims[2];
    const mapW = probTensor.dims[3];

    const visited = new Uint8Array(mapH * mapW);
    const thresh = 0.25;
    const lines: DetectedLine[] = [];
    const stack = new Int32Array(mapH * mapW);
    const compPts = new Int32Array(mapH * mapW);

    for (let y = 0; y < mapH; y++) {
      for (let x = 0; x < mapW; x++) {
        const start = y * mapW + x;
        if (visited[start] || prob[start] < thresh) continue;
        let top = 0;
        stack[top++] = start;
        visited[start] = 1;
        let minX = x, maxX = x, minY = y, maxY = y, cnt = 0;

        while (top > 0) {
          const p = stack[--top];
          const py = (p / mapW) | 0;
          const px = p - py * mapW;
          compPts[cnt++] = p;
          if (px < minX) minX = px;
          if (px > maxX) maxX = px;
          if (py < minY) minY = py;
          if (py > maxY) maxY = py;

          if (px > 0 && !visited[p - 1] && prob[p - 1] >= thresh) { visited[p - 1] = 1; stack[top++] = p - 1; }
          if (px + 1 < mapW && !visited[p + 1] && prob[p + 1] >= thresh) { visited[p + 1] = 1; stack[top++] = p + 1; }
          if (py > 0 && !visited[p - mapW] && prob[p - mapW] >= thresh) { visited[p - mapW] = 1; stack[top++] = p - mapW; }
          if (py + 1 < mapH && !visited[p + mapW] && prob[p + mapW] >= thresh) { visited[p + mapW] = 1; stack[top++] = p + mapW; }
        }

        if (cnt < 10) continue;
        const sx = img.width / mapW;
        const sy = img.height / mapH;
        const spanW = maxX - minX + 1;
        const bw = spanW * sx;
        const bh = (maxY - minY + 1) * sy;
        if (bw < 8 || bh < 6) continue;

        // Build per-column Y-mid & local height profile to dewarp tilted / curved / wavy lines
        let colTop: Float32Array | undefined;
        let colBot: Float32Array | undefined;
        if (spanW >= 8) {
          const cMin = new Float32Array(spanW).fill(1e9);
          const cMax = new Float32Array(spanW).fill(-1e9);
          for (let k = 0; k < cnt; k++) {
            const p = compPts[k];
            const py = (p / mapW) | 0;
            const px = p - py * mapW - minX;
            if (py < cMin[px]) cMin[px] = py;
            if (py > cMax[px]) cMax[px] = py;
          }
          // Fill any empty columns from nearest valid neighbor
          let lastMin = minY, lastMax = maxY;
          const heights: number[] = [];
          for (let c = 0; c < spanW; c++) {
            if (cMin[c] <= cMax[c]) {
              lastMin = cMin[c];
              lastMax = cMax[c];
              heights.push((cMax[c] - cMin[c] + 1) * sy);
            } else {
              cMin[c] = lastMin;
              cMax[c] = lastMax;
            }
          }
          heights.sort((a, b) => a - b);
          const medH = heights.length ? heights[Math.floor(heights.length * 0.65)] : bh;
          // Generous vertical padding so Lao tall stems (ປ ຝ ຟ ໂ ໃ ໄ), stacked top marks, and bottom ຸ ູ ຼ are NEVER clipped
          const padTop = Math.max(8, medH * 0.56);
          const padBot = Math.max(7, medH * 0.48);

          // Moving-average smooth centerline so curve follows wavy/tilted text smoothly
          const win = Math.max(2, Math.min(12, Math.floor(spanW / 8)));
          colTop = new Float32Array(spanW);
          colBot = new Float32Array(spanW);
          for (let c = 0; c < spanW; c++) {
            let sumMid = 0, n = 0;
            const lo = Math.max(0, c - win);
            const hi = Math.min(spanW - 1, c + win);
            for (let j = lo; j <= hi; j++) {
              sumMid += ((cMin[j] + cMax[j] + 1) * 0.5) * sy;
              n++;
            }
            const mid = sumMid / Math.max(1, n);
            colTop[c] = Math.max(0, mid - medH * 0.5 - padTop);
            colBot[c] = Math.min(img.height, mid + medH * 0.5 + padBot);
          }
        }

        const padX = Math.max(6, bh * 0.38);
        const padTopBox = Math.max(8, bh * 0.56);
        const padBotBox = Math.max(7, bh * 0.48);
        lines.push({
          box: [
            Math.max(0, Math.floor(minX * sx - padX)),
            Math.max(0, Math.floor(minY * sy - padTopBox)),
            Math.min(img.width, Math.ceil((maxX + 1) * sx + padX)),
            Math.min(img.height, Math.ceil((maxY + 1) * sy + padBotBox))
          ],
          colTop,
          colBot
        });
      }
    }

    if (!lines.length) return [{ box: [0, 0, img.width, img.height] }];
    // Sort top-to-bottom, tilt-aware row grouping (compare with rightmost box in row)
    lines.sort((a, b) => (a.box[1] + a.box[3]) / 2 - (b.box[1] + b.box[3]) / 2);
    const rows: DetectedLine[][] = [];
    for (const ln of lines) {
      let placed = false;
      for (const r of rows) {
        const ref = r[r.length - 1].box;
        const ov = Math.min(ref[3], ln.box[3]) - Math.max(ref[1], ln.box[1]);
        const minH = Math.min(ref[3] - ref[1], ln.box[3] - ln.box[1]);
        if (ov > 0.42 * minH) {
          r.push(ln);
          r.sort((a, b) => a.box[0] - b.box[0]);
          placed = true;
          break;
        }
      }
      if (!placed) rows.push([ln]);
    }
    const ordered: DetectedLine[] = [];
    for (const r of rows) {
      r.sort((a, b) => a.box[0] - b.box[0]);
      ordered.push(...r);
    }
    return ordered;
  }

  /** Recognize a single (possibly tilted/wavy) text line with Lao or Thai SVTR ONNX model */
  private async recognizeLine(img: RawRGBA, line: DetectedLine, mode: OcrModelMode): Promise<string> {
    const ort = require("onnxruntime-react-native");
    const [x1, y1, x2, y2] = line.box;
    const sw = Math.max(1, x2 - x1);
    const sh = line.colTop && line.colBot
      ? Math.max(8, line.colBot[Math.floor(line.colBot.length / 2)] - line.colTop[Math.floor(line.colTop.length / 2)])
      : Math.max(1, y2 - y1);
    const dstH = 48;
    const dstW = Math.max(16, Math.min(1280, Math.round((sw * dstH) / sh)));

    const nchw = this.cropDewarpToNCHW(img, line, dstW, dstH);
    const useThai = mode === "thai";
    const sess = useThai ? this.thaiRecSession : this.recSession;
    const chars = useThai ? this.thaiChars : this.chars;

    const inputName = sess.inputNames[0];
    const outMap = await sess.run({ [inputName]: new ort.Tensor("float32", nchw, [1, 3, dstH, dstW]) });
    const logitsTensor = outMap[sess.outputNames[0]];
    const probs = logitsTensor.data as Float32Array;
    const T = logitsTensor.dims[1];
    const C = logitsTensor.dims[2];

    let text = "";
    let prev = -1;
    let confSum = 0;
    let confCnt = 0;
    const charConfs: number[] = [];

    for (let t = 0; t < T; t++) {
      let bestIdx = 0;
      let bestVal = -1e9;
      const base = t * C;
      for (let c = 0; c < C; c++) {
        const v = probs[base + c];
        if (v > bestVal) {
          bestVal = v;
          bestIdx = c;
        }
      }
      let charConf = 1;
      if (bestIdx !== 0 && bestIdx !== prev && bestIdx < chars.length) {
        const ch = chars[bestIdx];
        text += ch;
        let rowSum = 0;
        for (let c = 0; c < C; c++) rowSum += probs[base + c];
        if (bestVal >= 0 && bestVal <= 1.001 && rowSum > 0.95 && rowSum < 1.05) {
          charConf = bestVal;
        } else {
          let sumExp = 0;
          for (let c = 0; c < C; c++) {
            const d = probs[base + c] - bestVal;
            if (d > -20) sumExp += Math.exp(d);
          }
          charConf = sumExp > 0 ? 1 / sumExp : 1;
        }
        confSum += charConf;
        confCnt++;
        charConfs.push(charConf);
      }
      prev = bestIdx;
    }

    this.lastConf = confCnt > 0 ? confSum / confCnt : 0;
    this.lastLowConfWords = {};
    // Compute per-word CTC confidence (flag words with avg < 85% or min < 68%)
    let wStart = -1;
    for (let i = 0; i <= text.length; i++) {
      const ch = i < text.length ? text[i] : " ";
      const isSep = /[\s,.;:!?()[\]{}"'\/\\|]/.test(ch);
      if (!isSep && wStart === -1) {
        wStart = i;
      } else if (isSep && wStart !== -1) {
        const word = text.slice(wStart, i).replace(/ໍ\u0eb2/g, "ຳ").trim();
        if (word.length >= 1) {
          let wSum = 0;
          let wMin = 1;
          let wCnt = 0;
          for (let k = wStart; k < i && k < charConfs.length; k++) {
            const p = charConfs[k];
            wSum += p;
            if (p < wMin) wMin = p;
            wCnt++;
          }
          if (wCnt > 0) {
            const wAvg = wSum / wCnt;
            const scorePct = Math.round(Math.min(wAvg, (wAvg + wMin) * 0.5) * 100);
            if (scorePct < 85) {
              const prevPct = this.lastLowConfWords[word];
              if (prevPct === undefined || scorePct < prevPct) {
                this.lastLowConfWords[word] = scorePct;
              }
            }
          }
        }
        wStart = -1;
      }
    }

    if (useThai) return text.trim();
    // Apply 5-gram Lao CharLM CTC Prefix Beam Search for context-accurate tone marks & look-alike letters!
    if (/[\u0E80-\u0EFF]/.test(text) && this.charLM) {
      const sm = new Float32Array(T * C);
      for (let t = 0; t < T; t++) {
        const base = t * C;
        let maxV = -1e9, sumV = 0;
        for (let c = 0; c < C; c++) { const v = probs[base + c]; if (v > maxV) maxV = v; sumV += v; }
        if (maxV <= 1.001 && sumV > 0.95 && sumV < 1.05) {
          for (let c = 0; c < C; c++) sm[base + c] = Math.max(0, probs[base + c]);
        } else {
          let sumE = 0;
          for (let c = 0; c < C; c++) { const e = Math.exp(Math.max(-30, probs[base + c] - maxV)); sm[base + c] = e; sumE += e; }
          const inv = sumE > 0 ? 1 / sumE : 1 / C;
          for (let c = 0; c < C; c++) sm[base + c] *= inv;
        }
      }
      const blockThai = (c: number) => {
        const ch = chars[c] || "";
        if (mode === "lao") return (ch >= "\u0e00" && ch <= "\u0e7f") || /^[A-Za-z]$/.test(ch);
        return ch >= "\u0e00" && ch <= "\u0e7f";
      };
      text = beamDecodeCTC(sm, T, C, chars, this.charLM, blockThai, 0.36, 0.72, 8);
    }
    const cleaned = text.replace(/ໍ\u0eb2/g, "ຳ").trim();
    return correctLaoLine(cleaned);
  }

  /** Format detected boxes + text into structured Markdown (headings, tables, bullets, paragraphs) */
  private formatPageMarkdown(items: { box: [number, number, number, number]; text: string }[]): string {
    if (!items.length) return "";
    const heights = items.map((it) => it.box[3] - it.box[1]).sort((a, b) => a - b);
    const lh = Math.max(10, heights[Math.floor(heights.length / 2)] || 20);

    // Group into visual rows
    const sorted = [...items].sort((a, b) => (a.box[1] + a.box[3]) / 2 - (b.box[1] + b.box[3]) / 2);
    const rows: { box: [number, number, number, number]; text: string }[][] = [];
    for (const it of sorted) {
      let placed = false;
      for (const r of rows) {
        const ref = r[r.length - 1].box;
        const ov = Math.min(ref[3], it.box[3]) - Math.max(ref[1], it.box[1]);
        const minH = Math.min(ref[3] - ref[1], it.box[3] - it.box[1]);
        if (ov > 0.42 * minH) {
          r.push(it);
          r.sort((a, b) => a.box[0] - b.box[0]);
          placed = true;
          break;
        }
      }
      if (!placed) rows.push([it]);
    }

    const out: string[] = [];
    let i = 0;
    let prevY2: number | null = null;
    while (i < rows.length) {
      const r = rows[i];
      const y1 = Math.min(...r.map((x) => x.box[1]));
      const y2 = Math.max(...r.map((x) => x.box[3]));
      const rh = y2 - y1;
      if (prevY2 !== null && y1 - prevY2 > lh * 1.3 && out.length && out[out.length - 1] !== "") {
        out.push("");
      }
      // Table detection: >= 2 consecutive multi-column rows
      if (r.length >= 2) {
        let j = i + 1;
        while (j < rows.length && rows[j].length >= 2) j++;
        if (j - i >= 2) {
          const cols = Math.min(8, Math.max(...rows.slice(i, j).map((x) => x.length)));
          const mkRow = (row: typeof r) =>
            "| " + Array.from({ length: cols }, (_, c) => (row[c]?.text || "").replace(/\|/g, "\\|")).join(" | ") + " |";
          out.push(mkRow(rows[i]));
          out.push("| " + Array(cols).fill("---").join(" | ") + " |");
          for (let k = i + 1; k < j; k++) out.push(mkRow(rows[k]));
          out.push("");
          prevY2 = Math.max(...rows[j - 1].map((x) => x.box[3]));
          i = j;
          continue;
        }
      }
      const txt = r.map((x) => x.text).join("   ").trim();
      if (r.length === 1 && txt.length <= 80 && rh >= lh * 1.32) {
        out.push(`## ${txt}`);
      } else if (/^(?:ມາດຕາ|มาตรา|ໝວດ|หมวด|ພາກ|ภาค|ຂໍ້|ข้อ)\s*[\d໐-໙๐-๙]+/.test(txt)) {
        out.push(`**${txt}**`);
      } else if (/^[•·▪►\-–]\s*/.test(txt)) {
        out.push(`- ${txt.replace(/^[•·▪►\-–]\s*/, "")}`);
      } else {
        out.push(txt);
      }
      prevY2 = y2;
      i++;
    }
    return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  /** Run complete 100% offline OCR on one or multiple image URIs with per-page rotations */
  async scanImages(
    uris: string[],
    rotations: number | number[] = 0,
    mode: OcrModelMode = "lao",
    outputFormat: OutputFormatMode = "plain"
  ): Promise<OcrResult> {
    const t0 = Date.now();
    await this.init();
    if (mode === "thai") await this.ensureThaiModel();

    const pages: string[] = [];
    let totalLines = 0;
    let confSum = 0;
    let confCnt = 0;
    const lowConfWords: Record<string, number> = {};

    for (let i = 0; i < uris.length; i++) {
      const deg = Array.isArray(rotations) ? (rotations[i] || 0) : rotations;
      const img = await this.decodeUriToRGBA(uris[i], deg);
      const detLines = await this.detectLines(img);
      const items: { box: [number, number, number, number]; text: string }[] = [];

      for (const ln of detLines) {
        const txt = await this.recognizeLine(img, ln, mode);
        if (txt && !/^[\.\-_=~*·•\s]+$/.test(txt)) {
          items.push({ box: ln.box, text: txt });
          confSum += this.lastConf;
          confCnt++;
          for (const [w, pct] of Object.entries(this.lastLowConfWords)) {
            if (lowConfWords[w] === undefined || pct < lowConfWords[w]) {
              lowConfWords[w] = pct;
            }
          }
        }
      }
      totalLines += items.length;
      const pageText =
        outputFormat === "markdown"
          ? this.formatPageMarkdown(items)
          : items.map((x) => x.text).join("\n");
      const pageHeader =
        outputFormat === "markdown"
          ? `# ໜ້າ ${i + 1}/${uris.length}\n\n${pageText}`
          : `===== ໜ້າ ${i + 1}/${uris.length} =====\n${pageText}`;
      pages.push(uris.length > 1 ? pageHeader : pageText);
    }

    return {
      text: pages.join("\n\n"),
      lineCount: totalLines,
      ms: Date.now() - t0,
      avgConf: confCnt > 0 ? Math.round((confSum / confCnt) * 1000) / 10 : 0,
      lowConfWords
    };
  }
}

export const mobileOcr = new MobileLaoOcrEngine();
