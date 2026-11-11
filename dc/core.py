"""Shared pieces for the v2 decimal-comparison study: pairs, prompts, the letter metric, hooks.

v1 (scripts/) measured `tok.encode(" A")[0]`, which for phi-3's sentencepiece tokenizer is the
bare-space token '▁' for both letters, so every logit difference was exactly 0 and every
probability exactly 0.5. Everything here measures the letter tokens themselves and asserts they
differ.
"""
import difflib
import random

import torch

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = 'microsoft/Phi-3-mini-4k-instruct'

TEMPLATE = '{prefix}Which number is larger? Answer with just one letter: A or B.\nA) {a}\nB) {b}\nAnswer:'
PREFIX = {'ambiguous': '', 'decimal': 'Treat these as decimal numbers. ', 'padded': 'Treat these as decimal numbers. '}


def make_pairs(n_each=60, seed=0):
    """Same integer part, one number with two decimal digits and one with one.

    trap:    the two-digit number is smaller (7.11 vs 7.8), so 'more digits = larger' fails
    control: the two-digit number is larger  (7.81 vs 7.2), so the same heuristic succeeds
    """
    rng = random.Random(seed)
    pairs, seen = [], set()
    while len(pairs) < 2 * n_each:
        i, d1, d2, c = rng.randint(1, 9), rng.randint(0, 9), rng.randint(1, 9), rng.randint(1, 9)
        longer, shorter = f'{i}.{d1}{d2}', f'{i}.{c}'
        kind = 'trap' if float(longer) < float(shorter) else 'control'
        if (longer, shorter) in seen or sum(p['kind'] == kind for p in pairs) >= n_each:
            continue
        seen.add((longer, shorter))
        long_first = len(pairs) % 2 == 0  # balance which letter holds the correct answer
        pairs.append({'kind': kind, 'longer': longer, 'shorter': shorter, 'long_first': long_first})
    return pairs


def pad(x):
    whole, frac = x.split('.')
    return f'{whole}.{frac}0' if len(frac) == 1 else x


def prompt(pair, condition):
    shorter = pad(pair['shorter']) if condition == 'padded' else pair['shorter']
    a, b = (pair['longer'], shorter) if pair['long_first'] else (shorter, pair['longer'])
    gold = 'A' if float(a) > float(b) else 'B'
    return TEMPLATE.format(prefix=PREFIX[condition], a=a, b=b), gold, a, b


class Letters:
    def __init__(self, tok):
        self.A = tok.encode('Answer: A', add_special_tokens=False)[-1]
        self.B = tok.encode('Answer: B', add_special_tokens=False)[-1]
        assert self.A != self.B, 'letter tokens collide'
        assert tok.decode([self.A]).strip() == 'A' and tok.decode([self.B]).strip() == 'B'

    def diff(self, logits, gold):
        """logit(correct letter) - logit(other letter) at the answer position."""
        right, wrong = (self.A, self.B) if gold == 'A' else (self.B, self.A)
        return (logits[..., right] - logits[..., wrong]).float()


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEVICE).eval()
    return model, tok


def encode(tok, text):
    enc = tok(text, return_tensors='pt', return_offsets_mapping=True)
    return enc['input_ids'].to(DEVICE), enc['offset_mapping'][0].tolist()


def align(src_ids, dst_ids):
    """Token positions that match between two prompts (dst -> src), e.g. 7.8 inside 7.80."""
    sm = difflib.SequenceMatcher(a=dst_ids, b=src_ids, autojunk=False)
    pairs = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs[blk.a + k] = blk.b + k
    return pairs


def span_positions(offsets, text, needle, occurrence):
    """Token positions overlapping the `occurrence`-th appearance of `needle` in text."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(needle, start + 1)
    end = start + len(needle)
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]


def residual_hook(layer, positions_map, source):
    """Overwrite the residual stream after `layer` at dst positions with source activations."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        for d, s in positions_map.items():
            hidden[:, d] = source[:, s]
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    return layer.register_forward_hook(hook)
