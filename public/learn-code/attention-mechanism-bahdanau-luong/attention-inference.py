"""NumPy recurrent attention: cached source memory and complete decoder inference.

Place calculated-inputs.json alongside this file. The saved experiment supplies
parameters; this program implements the forward mechanism without a model API.
"""
from pathlib import Path
import json
import string
import numpy as np

TOKENS = ["<pad>", "<bos>", "<eos>", "<past>", "<participle>", "<third_person>"] + list(string.ascii_lowercase)
INDEX = {token: index for index, token in enumerate(TOKENS)}


def softmax(scores, valid=None):
    valid = np.ones(scores.shape, dtype=bool) if valid is None else valid
    if not np.any(valid):
        raise ValueError("The read needs at least one valid memory.")
    masked = np.where(valid, scores, -np.inf)
    numerators = np.exp(masked - np.max(masked))
    return numerators / numerators.sum()


def gru(x, state, weights, prefix, encoder=False):
    suffix = "_l0" if encoder else ""
    incoming = weights[f"{prefix}.weight_ih{suffix}"] @ x + weights[f"{prefix}.bias_ih{suffix}"]
    previous = weights[f"{prefix}.weight_hh{suffix}"] @ state + weights[f"{prefix}.bias_hh{suffix}"]
    ir, iz, inn = np.split(incoming, 3)
    hr, hz, hn = np.split(previous, 3)
    reset = 1 / (1 + np.exp(-ir - hr))
    update = 1 / (1 + np.exp(-iz - hz))
    candidate = np.tanh(inn + reset * hn)
    return (1 - update) * candidate + update * state


def encode(lemma, feature, weights):
    if not 1 <= len(lemma) <= 32 or any(c not in string.ascii_lowercase for c in lemma):
        raise ValueError("Use 1 to 32 lowercase letters.")
    if feature not in ("past", "participle", "third_person"):
        raise ValueError("Unknown inflection request.")
    ids = [INDEX[f"<{feature}>"]] + [INDEX[c] for c in lemma] + [INDEX["<eos>"]]
    state = np.zeros(64)
    memory = []
    for token in ids:
        state = gru(weights["embedding.weight"][token], state, weights, "encoder", True)
        memory.append(state)
    memory = np.stack(memory)
    keys = memory @ weights["key_projection.weight"].T
    return memory, keys, state


def decode(lemma, feature, weights, kind, prefix="", cap=16):
    if kind not in ("additive", "general"):
        raise ValueError("Choose additive or general scoring.")
    if not 1 <= cap <= 16 or any(c not in string.ascii_lowercase for c in prefix):
        raise ValueError("Use a 1–16 token cap and lowercase forced prefix.")
    memory, keys, state = encode(lemma, feature, weights)
    previous = INDEX["<bos>"]
    output, trace = [], []
    for step in range(cap):
        embedding = weights["embedding.weight"][previous]
        if kind == "general":
            state = gru(embedding, state, weights, "decoder")
        query = state
        if kind == "additive":
            features = np.tanh(keys + weights["query_projection.weight"] @ query)
            scores = (features @ weights["score_projection.weight"].T).ravel()
        else:
            scores = keys @ query
        attention = softmax(scores)
        context = attention @ memory
        if kind == "additive":
            state = gru(np.r_[embedding, context], state, weights, "decoder")
        combined = np.tanh(weights["combine.weight"] @ np.r_[state, context] + weights["combine.bias"])
        logits = weights["readout.weight"] @ combined + weights["readout.bias"]
        probabilities = softmax(logits, ~weights["invalid_output"].astype(bool))
        argmax = int(probabilities.argmax())
        emitted = INDEX[prefix[step]] if step < len(prefix) else argmax
        trace.append({"query": query, "scores": scores, "attention": attention,
                      "context": context, "state": state, "probabilities": probabilities,
                      "argmax": argmax, "emitted": emitted})
        output.append(emitted)
        if emitted == INDEX["<eos>"]:
            break
        previous = emitted
    word = "".join(TOKENS[token] for token in output if token != INDEX["<eos>"])
    return word, output[-1] == INDEX["<eos>"], trace


if __name__ == "__main__":
    report = json.loads((Path(__file__).parent / "calculated-inputs.json").read_text(encoding="utf-8"))
    run = next(row for row in report["runs"] if row["kind"] == "additive" and row["seed"] == 1)
    parameters = {name: np.array(value) for name, value in run["weights"].items()}
    word, ended, trace = decode("lactate", "past", parameters, run["kind"])
    print(word, ended)
