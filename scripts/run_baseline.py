import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AMBIG = "Which is larger, {a} or {b}? Answer with just one: {a} or {b}."
DECIM = "Treat these as decimal numbers. Which is larger, {a} or {b}? Answer with just one: {a} or {b}."
CLEAN = "Treat these as decimal numbers. Which is larger, {a2} or {b2}? Answer with just one: {a2} or {b2}."

def clean(x):
    l, r = x.split(".")
    return f"{l}.{r}0" if len(r) == 1 else x

def load_pairs():
    with open("data/prompts.json") as f:
        return json.load(f)["pairs"]

import torch
import torch.nn.functional as F

def logprob_of_string(model, tok, prompt, answer):
    """
    Computes sum log-probability of `answer` conditioned on `prompt`.
    """
    device = next(model.parameters()).device

    prompt_ids = tok(prompt, return_tensors="pt").to(device)
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out = model(**prompt_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits

    logp = 0.0

    for i, tid in enumerate(answer_ids):
        if i == 0:
            step_logits = logits[0, -1]
        else:
            step_out = model(
                input_ids=torch.tensor([[answer_ids[i-1]]], device=device),
                past_key_values=past,
                use_cache=True,
            )
            step_logits = step_out.logits[0, -1]
            past = step_out.past_key_values

        logp += F.log_softmax(step_logits, dim=-1)[tid].item()

    return logp

def score(prompt, a, b, tok, model):
    la = logprob_of_string(model, tok, prompt, " " + a)
    lb = logprob_of_string(model, tok, prompt, " " + b)

    if la > lb:
        return a, la - lb
    else:
        return b, lb - la

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    ).to(DEVICE)
    model.eval()

    rows = []
    pairs = load_pairs()

    for i, (a, b) in enumerate(tqdm(pairs)):
        gold = a if float(a) > float(b) else b

        for cond in ["ambiguous", "decimal", "clean"]:
            if cond == "ambiguous":
                prompt = AMBIG.format(a=a, b=b)
                a_eval, b_eval = a, b
            elif cond == "decimal":
                prompt = DECIM.format(a=a, b=b)
                a_eval, b_eval = a, b
            else:
                a2, b2 = clean(a), clean(b)
                prompt = CLEAN.format(a2=a2, b2=b2)
                a_eval, b_eval = a2, b2
                gold = a_eval if float(a_eval) > float(b_eval) else b_eval

            pred, delta = score(prompt, a_eval, b_eval, tok, model)
            rows.append({
                "pair_id": i,
                "condition": cond,
                "a": a_eval,
                "b": b_eval,
                "gold": gold,
                "pred": pred,
                "correct": int(pred == gold),
                "delta_logprob": delta
            })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/baseline.csv", index=False)
    print(df.groupby("condition")["correct"].mean())

if __name__ == "__main__":
    main()
