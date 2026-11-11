"""v2 experiments on phi-3-mini-4k-instruct.

    python -m dc.run baseline   # accuracy and logit diff by condition, plus a logit lens
    python -m dc.run patch      # padded (clean) -> unpadded (corrupt) residual patching on trap pairs
    python -m dc.run report

Outputs go to outputs/v2/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dc.core import (Letters, align, encode, load_model, make_pairs, prompt, residual_hook, span_positions)

OUT = Path('outputs/v2')
CONDITIONS = ['ambiguous', 'decimal', 'padded']
N_PATCH = 24


@torch.no_grad()
def baseline():
    model, tok = load_model()
    L = Letters(tok)
    norm, head = model.model.norm, model.lm_head
    rows, lens = [], []
    for i, pair in enumerate(make_pairs()):
        for cond in CONDITIONS:
            text, gold, a, b = prompt(pair, cond)
            ids, _ = encode(tok, text)
            out = model(ids, output_hidden_states=True)
            ld = L.diff(out.logits[0, -1], gold).item()
            pa = torch.softmax(out.logits[0, -1].float(), -1)
            rows.append({'pair': i, 'kind': pair['kind'], 'condition': cond, 'a': a, 'b': b, 'gold': gold,
                         'logit_diff': ld, 'correct': int(ld > 0),
                         'p_letters': (pa[L.A] + pa[L.B]).item()})
            # logit lens: each layer's final-position residual through the final norm and unembedding
            per_layer = [L.diff(head(norm(h[0, -1])), gold).item() for h in out.hidden_states[1:]]
            lens.append({'pair': i, 'kind': pair['kind'], 'condition': cond, 'lens': per_layer})
        if i % 20 == 0:
            print(f'baseline {i}/120', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'baseline.csv', index=False)
    (OUT / 'logit_lens.json').write_text(json.dumps(lens))
    df = pd.DataFrame(rows)
    print(df.groupby(['kind', 'condition'])['correct'].mean().unstack())


@torch.no_grad()
def patch():
    """Denoising: run the corrupt (unpadded trap) prompt, restore clean (padded) activations.

    Effect is normalised per pair: 0 = corrupt behaviour, 1 = clean behaviour. Segments are the
    two numbers as they appear in the prompt, the final token, and every aligned position at once.
    """
    model, tok = load_model()
    L = Letters(tok)
    layers = model.model.layers
    base = pd.read_csv(OUT / 'baseline.csv')
    pairs = make_pairs()
    piv = base.pivot_table(index='pair', columns='condition', values='logit_diff')
    # trap pairs the model gets wrong unpadded and right padded, i.e. where padding flips the answer
    flips = [i for i in piv.index if pairs[i]['kind'] == 'trap' and piv.loc[i, 'decimal'] < 0 < piv.loc[i, 'padded']]
    use = flips[:N_PATCH]
    print(f'{len(flips)} trap pairs flip with padding; patching {len(use)}', flush=True)
    rows = []
    for n, i in enumerate(use):
        pair = pairs[i]
        c_text, gold, ca, cb = prompt(pair, 'padded')
        x_text, gold_x, xa, xb = prompt(pair, 'decimal')
        assert gold == gold_x
        c_ids, c_off = encode(tok, c_text)
        x_ids, x_off = encode(tok, x_text)
        clean = model(c_ids, output_hidden_states=True)
        ld_clean = L.diff(clean.logits[0, -1], gold).item()
        ld_corrupt = L.diff(model(x_ids).logits[0, -1], gold).item()
        amap = align(c_ids[0].tolist(), x_ids[0].tolist())
        shorter_x = xb if pair['long_first'] else xa
        longer_x = xa if pair['long_first'] else xb
        seg = {
            'longer number': span_positions(x_off, x_text, longer_x, 0),
            'shorter number': span_positions(x_off, x_text, shorter_x, 0),
            'final token': [x_ids.shape[1] - 1],
            'all positions': list(amap),
        }
        seg = {k: {d: amap[d] for d in v if d in amap} for k, v in seg.items()}
        for li, layer in enumerate(layers):
            src = clean.hidden_states[li + 1]
            for name, m in seg.items():
                h = residual_hook(layer, m, src)
                try:
                    ld = L.diff(model(x_ids).logits[0, -1], gold).item()
                finally:
                    h.remove()
                rows.append({'pair': i, 'layer': li, 'segment': name, 'logit_diff': ld,
                             'ld_clean': ld_clean, 'ld_corrupt': ld_corrupt,
                             'effect': (ld - ld_corrupt) / (ld_clean - ld_corrupt)})
        print(f'patched {n + 1}/{len(use)}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'patching.csv', index=False)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == 'report':
        from dc.report import report
        report()
    else:
        {'baseline': baseline, 'patch': patch}[cmd]()
