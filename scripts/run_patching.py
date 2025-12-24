import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16

A = "7.11"
B = "7.8"
GOLD_LETTER = "B"

PROMPT_AMB = (
    f"Which is larger, {A} or {B}? "
    f"Answer with just one letter: A or B.\n"
    f"A) {A}\nB) {B}\nAnswer:"
)

PROMPT_DEC = (
    f"Treat these as decimal numbers. Which is larger? "
    f"Answer with just one letter: A or B.\n"
    f"A) {A}\nB) {B}\nAnswer:"
)

# -----------------------------
# UTILS
# -----------------------------
def token_positions_for_substring(tok, input_ids, substring, occurrence=-1):
    """
    Return token indices for the specified substring by matching token IDs.
    """
    ids = input_ids[0].tolist()
    target = tok(substring, add_special_tokens=False)["input_ids"]
    if not target:
        return []

    matches = []
    span = len(target)
    for i in range(len(ids) - span + 1):
        if ids[i:i + span] == target:
            matches.append(list(range(i, i + span)))

    if not matches:
        return []
    return matches[occurrence]

def p_correct_letter(model, tok, inputs, correct_letter="B"):
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1]
    idA = tok.encode(" A", add_special_tokens=False)[0]
    idB = tok.encode(" B", add_special_tokens=False)[0]
    logA = F.log_softmax(logits, dim=-1)[idA].item()
    logB = F.log_softmax(logits, dim=-1)[idB].item()
    lg, lo = (logB, logA) if correct_letter == "B" else (logA, logB)
    p = float(torch.softmax(torch.tensor([lg, lo]), dim=0)[0].item())
    delta = lg - lo
    return p, delta

# -----------------------------
# MAIN
# -----------------------------
def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        output_hidden_states=True,
    )
    model.to(DEVICE).eval()

    amb = tok(PROMPT_AMB, return_tensors="pt").to(DEVICE)
    dec = tok(PROMPT_DEC, return_tensors="pt").to(DEVICE)

    pos_amb = token_positions_for_substring(tok, amb["input_ids"], B)
    pos_dec = token_positions_for_substring(tok, dec["input_ids"], B)

    with torch.no_grad():
        out_amb = model(**amb, use_cache=False)
        out_dec = model(**dec, use_cache=False)

    hs_amb = out_amb.hidden_states
    hs_dec = out_dec.hidden_states

    p_amb_base, d_amb_base = p_correct_letter(model, tok, amb, GOLD_LETTER)
    p_dec_base, d_dec_base = p_correct_letter(model, tok, dec, GOLD_LETTER)

    layers = model.model.layers
    num_layers = len(layers)
    rows = []

    def run_patched(target_inputs, source_states, target_positions, source_positions, layer_idx):
        def hook(_module, _inputs, output):
            out = output.clone()
            rep = source_states[layer_idx + 1].to(out.device)
            for tp, sp in zip(target_positions, source_positions):
                out[:, tp, :] = rep[:, sp, :]
            return out

        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            p_corr, delta = p_correct_letter(model, tok, target_inputs, GOLD_LETTER)
        finally:
            handle.remove()
        return p_corr, delta

    for L in tqdm(range(0, num_layers, 2)):
        p1, d1 = run_patched(amb, hs_dec, pos_amb, pos_dec, L)
        rows.append({
            "direction": "dec_to_amb",
            "layer": L,
            "p_correct": p1,
            "delta_logprob": d1,
            "p_base": p_amb_base,
            "delta_base": d_amb_base,
        })

        p2, d2 = run_patched(dec, hs_amb, pos_dec, pos_amb, L)
        rows.append({
            "direction": "amb_to_dec",
            "layer": L,
            "p_correct": p2,
            "delta_logprob": d2,
            "p_base": p_dec_base,
            "delta_base": d_dec_base,
        })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/patching.csv", index=False)

if __name__ == "__main__":
    main()
