"""
NOTE:
Steering experiments were considered but intentionally not run.

Rationale:
Causal patching failed to identify any layer-localized or low-dimensional mechanism controlling this behavior (see outputs/patching.csv).
In the absence of such a signal, steering would not be justified and would likely amount to fishing for effects.

This file is retained for completeness and reproducibility.
"""


import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

A = "9.8"
B = "9.11"
LAYER = 10
ALPHAS = np.linspace(-5, 5, 21)

AMBIG = f"Which is larger, {A} or {B}? Answer with just one: {A} or {B}."
DECIM = f"Treat these as decimal numbers. Which is larger, {A} or {B}? Answer with just one: {A} or {B}."

def forced_choice(logits, tok):
    ida = tok.encode(" " + A, add_special_tokens=False)[0]
    idb = tok.encode(" " + B, add_special_tokens=False)[0]
    pa = logits[ida].item()
    pb = logits[idb].item()
    total = torch.logsumexp(torch.tensor([pa, pb]), dim=0)
    return torch.exp(torch.tensor(pa) - total).item()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        output_hidden_states=True
    ).to(DEVICE)
    model.eval()

    amb = tok(AMBIG, return_tensors="pt").to(DEVICE)
    dec = tok(DECIM, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        h_amb = model(**amb).hidden_states[LAYER + 1]
        h_dec = model(**dec).hidden_states[LAYER + 1]

    d = (h_dec - h_amb).mean(dim=1)
    d = d / (d.norm() + 1e-6)

    rows = []

    for a in ALPHAS:
        def hook(_, __, output):
            out = output.clone()
            out[:, -1, :] += a * d.to(out.device)
            return out

        h = model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**amb)
        h.remove()

        p = forced_choice(out.logits[0, -1], tok)
        rows.append({"alpha": float(a), "p_correct": p})

    df = pd.DataFrame(rows)
    df.to_csv("outputs/steering.csv", index=False)

if __name__ == "__main__":
    main()
