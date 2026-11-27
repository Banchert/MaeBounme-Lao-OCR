from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from transformers import PPOCRV6TinyRecConfig, PPOCRV6TinyRecForTextRecognition

SMALL_KEYS = ('model_type', 'hidden_act', 'backbone_config', 'hidden_size', 'mlp_ratio', 'depth',
              'head_out_channels', 'conv_kernel_size', 'qkv_bias', 'num_attention_heads',
              'attention_dropout', 'layer_norm_eps')

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'models/OpenThai2.1-OCR-Text-20261118'
ARCHITECTURE = ROOT / 'models/ppocrv6-tiny-architecture/config.json'
# Explicit modern Lao inventory, not unassigned Unicode slots. A corpus may
# require extra historical/Pali characters; supply them with prepare --extra-chars.
LAO = 'ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮຽໜໝຯະັາຳິີຶືຸູົຼເແໂໃໄໆ່້໊໋໌ໍ໐໑໒໓໔໕໖໗໘໙'


class Codec:
    def __init__(self, dictionary: list[str]):
        if not dictionary or any(len(c) != 1 or c.isspace() for c in dictionary):
            raise ValueError('Dictionary must contain one non-whitespace code point per entry')
        if len(set(dictionary)) != len(dictionary):
            raise ValueError('Duplicate dictionary entries')
        self.dictionary = list(dictionary)
        self.chars = [''] + self.dictionary + [' ']
        self.ids = {c: i for i, c in enumerate(self.chars) if i}

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty or non-string line label')
        unknown = sorted(set(text) - self.ids.keys())
        if unknown:
            raise ValueError('Unsupported label characters: ' + ', '.join(f'{c!r} U+{ord(c):04X}' for c in unknown))
        return [self.ids[c] for c in text]

    def decode(self, sequence) -> str:
        out, previous = [], None
        for value in sequence:
            index = int(value)
            if index and index != previous:
                out.append(self.chars[index])
            previous = index
        return ''.join(out)


class Recognizer(nn.Module):
    """Expose pre-softmax logits for numerically stable CTC training.

    Supports PP-OCRv6 tiny (conv head, ~0.6M params) and PP-OCRv6 small
    (SVTR transformer neck, ~3M params + head) recognizers."""
    def __init__(self, config: dict):
        super().__init__()
        self.config_dict = config
        self.kind = config.get('model_type', 'pp_ocrv6_tiny_rec')
        if self.kind == 'pp_ocrv6_small_rec':
            from transformers import PPOCRV6SmallRecConfig, PPOCRV6SmallRecForTextRecognition
            self.m = PPOCRV6SmallRecForTextRecognition(
                PPOCRV6SmallRecConfig(**{k: v for k, v in config.items() if k in SMALL_KEYS}))
        else:
            self.m = PPOCRV6TinyRecForTextRecognition(PPOCRV6TinyRecConfig(**config))

    def set_eager_attention(self):
        """Plain matmul attention (needed for ONNX export)."""
        for module in self.modules():
            cfg = getattr(module, 'config', None)
            if cfg is not None and hasattr(cfg, '_attn_implementation'):
                try:
                    cfg._attn_implementation = 'eager'
                except Exception:
                    pass

    def forward(self, images):
        if self.kind == 'pp_ocrv6_small_rec':
            features = self.m.model(images).last_hidden_state          # B, C, 1, W/8
            features = self.m.head.encoder(features).last_hidden_state  # B, W/8, hidden
            return self.m.head.head(features)
        features = self.m.model(images).last_hidden_state.squeeze(2)
        head = self.m.head
        features = head.act_fn(head.norm1(head.conv1(features)))
        features = head.act_fn(head.norm2(head.conv2(features)))
        return head.fc2(head.fc1(features.transpose(1, 2)))


def load_original():
    chars = [c for c in (ORIGINAL / 'rec_dict.txt').read_text(encoding='utf-8').split('\n') if c]
    codec = Codec(chars)
    config = json.loads(ARCHITECTURE.read_text(encoding='utf-8'))
    config['head_out_channels'] = len(codec.chars)
    model = Recognizer(config)
    checkpoint = torch.load(ORIGINAL / 'pytorch/rec_weights.pt', weights_only=True, map_location='cpu')
    state = dict(checkpoint['net'])
    # The source checkpoint includes duplicate aliases outside the HF model.
    for suffix in ('weight', 'bias'):
        alias = state.pop('classifier.' + suffix)
        if not torch.equal(alias, state['m.head.fc2.' + suffix]):
            raise ValueError('Source classifier alias disagrees with the model head')
    model.load_state_dict(state, strict=True)
    return model, codec


def extend_for_lao(model, old_codec, extra_chars=''):
    additions = sorted(set(LAO + extra_chars) - set(old_codec.dictionary) - {' '})
    codec = Codec(old_codec.dictionary + additions)
    old = model.m.head.fc2
    new = nn.Linear(old.in_features, len(codec.chars)).to(old.weight.device)
    nn.init.normal_(new.weight, std=0.01)
    nn.init.constant_(new.bias, -4.0)
    with torch.no_grad():
        # Blank remains zero; space moves to the end of the expanded dictionary.
        for old_id, character in enumerate(old_codec.chars):
            new_id = 0 if old_id == 0 else codec.ids[character]
            new.weight[new_id].copy_(old.weight[old_id])
            new.bias[new_id].copy_(old.bias[old_id])
    model.m.head.fc2 = new
    model.config_dict['head_out_channels'] = len(codec.chars)
    model.m.config.head_out_channels = len(codec.chars)
    return codec, additions


def save_checkpoint(path, model, codec, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format_version': 1, 'config': model.config_dict,
               'dictionary': codec.dictionary, 'model': model.state_dict(), **metadata}
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if payload['format_version'] != 1:
        raise ValueError('Unsupported checkpoint format')
    codec = Codec(payload['dictionary'])
    if payload['config']['head_out_channels'] != len(codec.chars):
        raise ValueError('Classifier and dictionary size disagree')
    model = Recognizer(payload['config'])
    model.load_state_dict(payload['model'], strict=True)
    return model, codec, payload


def preprocess(image, max_width=4096):
    """Grayscale, aspect-preserving 48px height, width rounded to 8px.

    Shared by training and line inference. No Thai normalization or rec2.
    Overlong lines are rejected, never silently cropped or squashed.
    """
    if isinstance(image, (str, Path)):
        # imdecode supports Unicode Windows paths.
        image = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError('Cannot read input image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    width = max(8, math.ceil((48 * gray.shape[1] / gray.shape[0]) / 8) * 8)
    if width > max_width:
        raise ValueError(f'Line width {width} exceeds {max_width}; split the source line')
    gray = cv2.resize(gray, (width, 48), interpolation=cv2.INTER_LINEAR)
    values = gray.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.repeat(values[None], 3, axis=0))


def collate_lines(items):
    images, labels = zip(*items)
    widths = [x.shape[-1] for x in images]
    batch = torch.zeros(len(images), 3, 48, max(widths))
    for i, image in enumerate(images):
        batch[i, :, :, :image.shape[-1]] = image
    lengths = torch.tensor([w // 8 for w in widths], dtype=torch.long)
    for ids, frames in zip(labels, lengths):
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if minimum > int(frames):
            raise ValueError(f'CTC needs {minimum} frames but line has {int(frames)}')
    return (batch, torch.tensor([i for ids in labels for i in ids], dtype=torch.long),
            lengths, torch.tensor([len(ids) for ids in labels], dtype=torch.long))


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]
