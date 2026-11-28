/**
 * Compact 5-gram Lao Character Language Model + CTC Prefix Beam Search (100% Offline)
 * Reads lao_charlm_mobile.bin (~1.7 MB, 353k n-grams) and resolves context-dependent
 * Lao tone marks (ຜ່ານ vs ຜານ, ຕ່າງໆ vs ຕາງໆ, ໄດ້ແຕ່ງຕັ້ງ) & look-alike letters (ງ↔າ, ຂ↔ຊ).
 */

const BOS = "\x02";
const BACKOFF_PEN = Math.log10(0.4);

function fnv1a32Utf8(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    let cp = str.charCodeAt(i);
    if (cp < 0x80) {
      h ^= cp;
      h = Math.imul(h, 0x01000193) >>> 0;
    } else if (cp < 0x800) {
      h ^= 0xc0 | (cp >> 6);
      h = Math.imul(h, 0x01000193) >>> 0;
      h ^= 0x80 | (cp & 0x3f);
      h = Math.imul(h, 0x01000193) >>> 0;
    } else {
      h ^= 0xe0 | (cp >> 12);
      h = Math.imul(h, 0x01000193) >>> 0;
      h ^= 0x80 | ((cp >> 6) & 0x3f);
      h = Math.imul(h, 0x01000193) >>> 0;
      h ^= 0x80 | (cp & 0x3f);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
  }
  return h >>> 0;
}

export class CompactCharLM {
  private hashes: Uint32Array;
  private qlogp: Uint8Array;
  readonly order: number;
  readonly unk: number;
  private cache = new Map<string, number>();

  constructor(buf: ArrayBuffer | Uint8Array) {
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const magic = dv.getUint32(0, true);
    if (magic !== 0x4d4c434c) {
      throw new Error("Invalid lao_charlm_mobile.bin header");
    }
    const count = dv.getUint32(4, true);
    this.order = dv.getUint32(8, true);
    this.unk = dv.getFloat32(12, true);
    const hashBytes = count * 4;
    // Copy into aligned typed arrays
    this.hashes = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      this.hashes[i] = dv.getUint32(16 + i * 4, true);
    }
    this.qlogp = u8.slice(16 + hashBytes, 16 + hashBytes + count);
  }

  private lookup(gram: string): number | null {
    const h = fnv1a32Utf8(gram);
    let lo = 0;
    let hi = this.hashes.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const v = this.hashes[mid];
      if (v === h) return -(this.qlogp[mid] / 31.875);
      if (v < h) lo = mid + 1;
      else hi = mid - 1;
    }
    return null;
  }

  score(ctx: string, ch: string): number {
    const padded = BOS.repeat(this.order) + ctx;
    const key = padded.slice(-(this.order - 1)) + ch;
    const cached = this.cache.get(key);
    if (cached !== undefined) return cached;

    let pen = 0;
    let res = this.unk + this.order * BACKOFF_PEN;
    for (let n = this.order; n >= 1; n--) {
      const v = this.lookup(key.slice(-n));
      if (v !== null) {
        res = v + pen;
        break;
      }
      pen += BACKOFF_PEN;
    }
    if (this.cache.size > 60000) this.cache.clear();
    this.cache.set(key, res);
    return res;
  }
}

function lse10(a: number, b: number): number {
  if (a < b) {
    const tmp = a;
    a = b;
    b = tmp;
  }
  return b > -1e8 ? a + Math.log10(1 + Math.pow(10, b - a)) : a;
}

/**
 * CTC Prefix Beam Search guided by CompactCharLM.
 * probs: flat Float32Array [T * C] softmax probabilities (index 0 = blank).
 */
export function beamDecodeCTC(
  probs: Float32Array,
  T: number,
  C: number,
  chars: string[],
  lm: CompactCharLM | null,
  blockFn?: ((c: number) => boolean) | null,
  alpha = 0.36,
  beta = 0.72,
  beamWidth = 8,
  prune = 0.025,
  blankSkip = 0.98
): string {
  let beams = new Map<string, [number, number]>([["", [0.0, -1e9]]]);
  const lmScore = new Map<string, number>([["", 0.0]]);

  for (let t = 0; t < T; t++) {
    const base = t * C;
    const pBlank = probs[base];
    const cands: number[] = [];

    if (pBlank >= blankSkip) {
      cands.push(0);
    } else {
      let bestC = 0;
      let bestP = -1;
      for (let c = 0; c < C; c++) {
        if (c > 0 && blockFn && blockFn(c)) continue;
        const p = probs[base + c];
        if (p > bestP) {
          bestP = p;
          bestC = c;
        }
        if (p > prune) cands.push(c);
      }
      if (cands.length === 0) cands.push(bestC);
    }

    if (cands.length === 1 && cands[0] === 0) {
      const lp0 = Math.log10(Math.max(pBlank, 1e-12));
      const next = new Map<string, [number, number]>();
      for (const [pref, [pb, pnb]] of beams) {
        next.set(pref, [lse10(pb, pnb) + lp0, -1e9]);
      }
      beams = next;
      continue;
    }

    const nxt = new Map<string, [number, number]>();
    const getOrInit = (k: string): [number, number] => {
      let v = nxt.get(k);
      if (!v) {
        v = [-1e9, -1e9];
        nxt.set(k, v);
      }
      return v;
    };

    for (const [pref, [pb, pnb]] of beams) {
      const tot = lse10(pb, pnb);
      const last = pref.length > 0 ? pref[pref.length - 1] : "";
      for (let ci = 0; ci < cands.length; ci++) {
        const c = cands[ci];
        const lp = Math.log10(Math.max(probs[base + c], 1e-12));
        if (c === 0) {
          const e = getOrInit(pref);
          e[0] = lse10(e[0], tot + lp);
          continue;
        }
        if (c >= chars.length) continue;
        const ch = chars[c];
        const np = pref + ch;
        if (!lmScore.has(np)) {
          const prevLm = lmScore.get(pref) || 0;
          lmScore.set(np, prevLm + (lm ? alpha * lm.score(pref, ch) + beta : 0));
        }
        const e = getOrInit(np);
        if (ch === last) {
          e[1] = lse10(e[1], pb + lp);
          const same = getOrInit(pref);
          same[1] = lse10(same[1], pnb + lp);
        } else {
          e[1] = lse10(e[1], tot + lp);
        }
      }
    }

    const ranked = Array.from(nxt.entries()).sort((a, b) => {
      const sa = lse10(a[1][0], a[1][1]) + (lmScore.get(a[0]) || 0);
      const sb = lse10(b[1][0], b[1][1]) + (lmScore.get(b[0]) || 0);
      return sb - sa;
    });
    beams = new Map(ranked.slice(0, beamWidth));
  }

  let bestPref = "";
  let bestScore = -1e18;
  for (const [pref, [pb, pnb]] of beams) {
    const sc = lse10(pb, pnb) + (lmScore.get(pref) || 0);
    if (sc > bestScore) {
      bestScore = sc;
      bestPref = pref;
    }
  }
  return bestPref;
}
