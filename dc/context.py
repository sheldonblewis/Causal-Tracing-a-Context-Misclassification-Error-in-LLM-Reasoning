"""Does the model read "7.11 vs 7.8" as decimals or as versions, and where is that decided?

7.11 is smaller than 7.8 as a decimal and later than 7.8 as a version. The prompts below differ in
one word (decimal / version), so the two readings can be compared and patched position by
position.

    python -m dc.context baseline   # which number wins under each reading
    python -m dc.context patch      # restore the decimal run's activations into the version run
    python -m dc.context figure     # figures/context_patching.png
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from dc.core import Letters, encode, load_model, make_pairs, residual_hook, span_positions

OUT = Path("outputs/v2")
TEMPLATE = "These are {kind} numbers. Which is greater? Answer with just one letter: A or B.\nA) {a}\nB) {b}\nAnswer:"
KINDS = ["decimal", "version"]
NEUTRAL = "Which is greater? Answer with just one letter: A or B.\nA) {a}\nB) {b}\nAnswer:"
N_PATCH = 24


def build(pair, kind):
    a, b = (pair["longer"], pair["shorter"]) if pair["long_first"] else (pair["shorter"], pair["longer"])
    template = NEUTRAL if kind == "neutral" else TEMPLATE
    return template.format(kind=kind, a=a, b=b), ("A" if pair["long_first"] else "B"), a, b


@torch.no_grad()
def baseline():
    """logit(letter of the longer number) - logit(other): positive means 'the longer one is greater'."""
    model, tok = load_model()
    letters = Letters(tok)
    rows = []
    for i, pair in enumerate(make_pairs()):
        for kind in [*KINDS, "neutral"]:
            text, longer_letter, a, b = build(pair, kind)
            ids, _ = encode(tok, text)
            diff = letters.diff(model(ids).logits[0, -1], longer_letter).item()
            rows.append({"pair": i, "kind": pair["kind"], "context": kind, "a": a, "b": b, "prefers_longer": diff})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "context_baseline.csv", index=False)
    summary = df.assign(longer=df.prefers_longer > 0).groupby(["kind", "context"])["longer"].mean().unstack()
    print(summary)
    print(df.groupby(["kind", "context"])["prefers_longer"].mean().unstack().round(2))


@torch.no_grad()
def patch():
    """Run the version prompt, restore decimal-run residuals at one segment and layer at a time.

    Effect: 0 = version behaviour, 1 = decimal behaviour, on trap pairs where the two readings
    disagree.
    """
    model, tok = load_model()
    letters = Letters(tok)
    layers = model.model.layers
    base = pd.read_csv(OUT / "context_baseline.csv")
    piv = base.pivot_table(index="pair", columns="context", values="prefers_longer")
    pairs = make_pairs()
    gap = (piv["version"] - piv["decimal"]).sort_values(ascending=False)
    use = [i for i in gap.index if pairs[i]["kind"] == "trap" and gap[i] > 1.0][:N_PATCH]
    print(f"{len(use)} trap pairs where the version reading favours the longer number by > 1 logit", flush=True)
    rows = []
    for n, i in enumerate(use):
        pair = pairs[i]
        clean_text, longer_letter, a, b = build(pair, "decimal")
        corrupt_text, _, _, _ = build(pair, "version")
        clean_ids, clean_off = encode(tok, clean_text)
        corrupt_ids, corrupt_off = encode(tok, corrupt_text)
        assert clean_ids.shape == corrupt_ids.shape, "prompts must tokenize to the same length"
        clean = model(clean_ids, output_hidden_states=True)
        ld_clean = letters.diff(clean.logits[0, -1], longer_letter).item()
        ld_corrupt = letters.diff(model(corrupt_ids).logits[0, -1], longer_letter).item()
        word = span_positions(corrupt_off, corrupt_text, "version", 0)
        numbers = span_positions(corrupt_off, corrupt_text, a, 0) + span_positions(corrupt_off, corrupt_text, b, 0)
        last = corrupt_ids.shape[1] - 1
        segments = {
            "context word": word,
            "the two numbers": numbers,
            "final token": [last],
            "everything else": [p for p in range(last) if p not in word and p not in numbers],
        }
        for li, layer in enumerate(layers):
            source = clean.hidden_states[li + 1]
            for name, positions in segments.items():
                hook = residual_hook(layer, {p: p for p in positions}, source)
                try:
                    ld = letters.diff(model(corrupt_ids).logits[0, -1], longer_letter).item()
                finally:
                    hook.remove()
                rows.append(
                    {
                        "pair": i,
                        "layer": li,
                        "segment": name,
                        "effect": (ld - ld_corrupt) / (ld_clean - ld_corrupt),
                        "ld_clean": ld_clean,
                        "ld_corrupt": ld_corrupt,
                    }
                )
        print(f"patched {n + 1}/{len(use)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "context_patching.csv", index=False)
    mean = df.groupby(["segment", "layer"])["effect"].mean().unstack(0)
    (OUT / "context_patching_mean.json").write_text(json.dumps({c: mean[c].round(3).tolist() for c in mean}))
    print(mean.round(2).iloc[::4])


def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT / "context_patching.csv")
    g = df.groupby(["segment", "layer"])["effect"]
    mean, sem = g.mean().unstack(0), g.sem().unstack(0)
    colors = {"context word": "#c9a46c", "the two numbers": "#7d93b8", "final token": "#5b8f6e", "everything else": "#bbb"}
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=140)
    for name, color in colors.items():
        ax.plot(mean.index, mean[name], color=color, lw=1.8, label=name)
        ax.fill_between(mean.index, mean[name] - sem[name], mean[name] + sem[name], color=color, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xlabel("layer (residual stream after it)")
    ax.set_ylabel("share of the decimal reading restored")
    ax.set_title(f"patching 'decimal' into 'version' prompts, phi-3-mini, {df.pair.nunique()} trap pairs")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/context_patching.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"baseline": baseline, "patch": patch, "figure": figure}[sys.argv[1]]()
