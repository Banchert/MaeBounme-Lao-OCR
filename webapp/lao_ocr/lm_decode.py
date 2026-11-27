"""Dictionary-assisted decoding: CTC prefix beam search + Lao character n-gram language model.

Greedy decoding reads every character on its own. With the language model the decoder
prefers character sequences that form real Lao words (learned from datasets/lao-text),
so look-alike letters (ບ/ປ, ພ/ຟ, ຍ/ຢ, missing tone marks ...) are resolved by context.

Build:  python -m lao_ocr.lm_decode build        -> models/lao_charlm.npz
"""
from __future__ import annotations

import collections
import hashlib
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = 5
LM_FILE_V1 = ROOT / 'models' / 'lao_charlm.npz'
LM_FILE_V2 = ROOT / 'models' / 'lao_charlm_v2.npz'   # + lessons, colloquial speech, GitHub corpora
LM_FILE_V3 = ROOT / 'models' / 'lao_charlm_v3.npz'   # + weblao, VOA Lao news, LaoNews (1.3 M more lines)
LM_FILE_V4 = ROOT / 'models' / 'lao_charlm_v4.npz'   # + FineWeb-2 news/articles, PDF text, Wikipedia
LM_FILE_V5 = ROOT / 'models' / 'lao_charlm_v5.npz'   # + MoES textbooks, LaoTradePortal, clean Wiki
LM_FILE_V6 = ROOT / 'models' / 'lao_charlm_v6.npz'   # + lao-online Lao books (folktales, literature, prayers, history)
LM_FILE = next((f for f in (LM_FILE_V6, LM_FILE_V5, LM_FILE_V4, LM_FILE_V3, LM_FILE_V2, LM_FILE_V1) if f.is_file()), LM_FILE_V1)
BOS = '\x02'


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'little', signed=True)


class CharLM:
    """Stupid-backoff character n-gram model stored as sorted hash arrays (small, fast to load)."""

    def __init__(self, path=LM_FILE):
        z = np.load(path)
        self.keys, self.logp, self.order = z['keys'], z['logp'], int(z['order'])
        self.unk = float(z['unk'])
        self._cache = {}

    def _lookup(self, s):
        k = _h(s)
        i = np.searchsorted(self.keys, k)
        if i < len(self.keys) and self.keys[i] == k:
            return float(self.logp[i])
        return None

    def score(self, ctx: str, ch: str) -> float:
        """log10 P(ch | ctx) with stupid backoff (penalty 0.4 per backoff step)."""
        ctx = (BOS * self.order + ctx)[-(self.order - 1):]
        key = ctx + ch
        r = self._cache.get(key)
        if r is not None:
            return r
        pen = 0.0
        for n in range(self.order, 0, -1):
            v = self._lookup(key[-n:])
            if v is not None:
                r = v + pen
                break
            pen += math.log10(0.4)
        else:
            r = self.unk + pen
        if len(self._cache) > 500000:
            self._cache.clear()
        self._cache[key] = r
        return r


def build(texts, out=LM_FILE, order=ORDER, min_count=2):
    counts = [collections.Counter() for _ in range(order + 1)]
    for t in texts:
        t = BOS * (order - 1) + t
        for n in range(1, order + 1):
            c = counts[n]
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                if n == 1 and g == BOS:
                    continue
                c[g] += 1
    total1 = sum(counts[1].values())
    keys, logp = [], []
    for n in range(1, order + 1):
        for g, c in counts[n].items():
            if c < min_count and n > 1:
                continue
            den = total1 if n == 1 else counts[n - 1].get(g[:-1], 0)
            if den <= 0:
                continue
            keys.append(_h(g))
            logp.append(math.log10(c / den))
    keys = np.array(keys, np.int64)
    logp = np.array(logp, np.float32)
    o = np.argsort(keys)
    np.savez_compressed(out, keys=keys[o], logp=logp[o], order=order, unk=math.log10(0.5 / total1))
    return len(keys)


def _default_beam_params() -> tuple[float, float]:
    p = ROOT / 'models' / 'lm_decode_params.json'
    if p.is_file():
        try:
            import json as _j
            d = _j.loads(p.read_text(encoding='utf-8'))
            return float(d.get('alpha', 0.4)), float(d.get('beta', 0.8))
        except Exception:
            pass
    return 0.4, 0.8


_DEF_ALPHA, _DEF_BETA = _default_beam_params()


def beam_decode(probs: np.ndarray, chars: list[str], lm: CharLM | None, alpha=None, beta=None, beam=10,
                prune=0.02, blank_skip=0.98, normalize=None):
    """CTC prefix beam search. probs: [T, C] softmax (index 0 = blank). Returns text."""
    if alpha is None:
        alpha = _DEF_ALPHA
    if beta is None:
        beta = _DEF_BETA
    T, C = probs.shape
    # beams: prefix -> (p_blank, p_nonblank) in log10, plus LM score
    beams = {'': (0.0, -1e9)}
    lm_score = {'': 0.0}

    def lse(a, b):
        if a < b:
            a, b = b, a
        return a + math.log10(1 + 10 ** (b - a)) if b > -1e8 else a

    logp = np.log10(np.maximum(probs, 1e-12))
    for t in range(T):
        # only characters the image really supports are considered; near-certain blanks stay blank
        cand = np.nonzero(probs[t] > prune)[0] if probs[t, 0] < blank_skip else np.array([0])
        if len(cand) == 0:
            cand = np.array([int(probs[t].argmax())])
        if len(cand) == 1 and cand[0] == 0:  # pure blank frame: only the blank path survives
            beams = {p: (lse(b, nb) + logp[t, 0], -1e9) for p, (b, nb) in beams.items()}
            continue
        nxt = collections.defaultdict(lambda: [-1e9, -1e9])
        for prefix, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in cand:
                lp = logp[t, c]
                if c == 0:
                    e = nxt[prefix]
                    e[0] = lse(e[0], tot + lp)
                    continue
                ch = chars[c]
                last = prefix[-1] if prefix else None
                np_ = prefix + ch
                if np_ not in lm_score:
                    lm_score[np_] = lm_score[prefix] + (alpha * lm.score(prefix, ch) + beta if lm else 0.0)
                e = nxt[np_]
                if ch == last:  # repeated char needs a blank in between
                    e[1] = lse(e[1], pb + lp)
                    s = nxt[prefix]
                    s[1] = lse(s[1], pnb + lp)
                else:
                    e[1] = lse(e[1], tot + lp)
        ranked = sorted(nxt.items(), key=lambda kv: lse(*kv[1]) + lm_score[kv[0]], reverse=True)[:beam]
        beams = {p: tuple(v) for p, v in ranked}
    best = max(beams.items(), key=lambda kv: lse(*kv[1]) + lm_score[kv[0]])[0]
    return normalize(best) if normalize else best


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'build':
    sys.path.insert(0, str(ROOT))
    from lao_ocr.synth import normalize_label
    rng = random.Random(5)
    texts = []
    d = ROOT / 'datasets' / 'lao-text'
    per_file = int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000
    if len(sys.argv) > 3:
        d = ROOT / sys.argv[3]
    out_file = ROOT / sys.argv[4] if len(sys.argv) > 4 else LM_FILE_V2
    for f in sorted(d.glob('dom_*.txt')) + [d / 'addresses.txt', d / 'terms.txt']:
        lines = f.read_text(encoding='utf-8').splitlines()
        rng.shuffle(lines)
        n = 0
        for l in lines:
            t = normalize_label(l)
            texts.append(t)
            n += len(t)
            if n > per_file:
                break
    texts += [normalize_label(w) for w in (d / 'lexicon.txt').read_text(encoding='utf-8').splitlines()] * 2
    print(f'[lm] {len(texts)} lines, {sum(map(len, texts)) / 1e6:.1f} M chars')
    n = build(texts, out=out_file)
    print(f'[lm] {n} n-grams -> {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)')
