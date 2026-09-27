"""Saved-model traces, fresh interventions and independent NumPy correspondence."""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True
import json
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("attentive_inflection", ROOT/"attentive-inflection.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gru(x, state, weights, prefix, layer=False):
    ending = "_l0" if layer else ""
    a = weights[prefix+".weight_ih"+ending]@x + weights[prefix+".bias_ih"+ending]
    b = weights[prefix+".weight_hh"+ending]@state + weights[prefix+".bias_hh"+ending]
    ar, az, an = np.split(a, 3); br, bz, bn = np.split(b, 3)
    reset, update = sigmoid(ar+br), sigmoid(az+bz)
    candidate = np.tanh(an+reset*bn)
    return (1-update)*candidate+update*state

def normalized(scores):
    out = np.exp(scores-scores.max())
    return out/out.sum()

def manual_trace(run, row, force_first=None, pad=0, admit_padding=False):
    weights = {name: np.array(value) for name, value in run["weights"].items()}
    source, _ = module.source_batch([row])
    ids = source[0].tolist()
    state, memory = np.zeros(64), []
    for token in ids:
        state = gru(weights["embedding.weight"][token], state, weights, "encoder", True)
        memory.append(state.copy())
    memory = np.array(memory)
    if pad:
        memory = np.vstack([memory, np.zeros((pad, 64))])
    keys = memory@weights["key_projection.weight"].T
    valid = np.arange(len(memory)) < len(ids)
    if admit_padding:
        valid[:] = True
    rows, output, previous = [], [], module.BOS
    for t in range(16):
        embedding = weights["embedding.weight"][previous]
        if run["kind"] == "general":
            state = gru(embedding, state, weights, "decoder")
            scores = keys@state
        else:
            scores = (np.tanh(keys+weights["query_projection.weight"]@state)@weights["score_projection.weight"].T).ravel()
        query = state.copy()
        attention = normalized(np.where(valid, scores, -np.inf))
        context = attention@memory
        if run["kind"] == "additive":
            state = gru(np.r_[embedding, context], state, weights, "decoder")
        attentional = np.tanh(weights["combine.weight"]@np.r_[state, context]+weights["combine.bias"])
        logits = weights["readout.weight"]@attentional+weights["readout.bias"]
        logits[weights["invalid_output"].astype(bool)] = -np.inf
        probabilities = normalized(logits)
        chosen = int(probabilities.argmax())
        consumed = module.INDEX[force_first] if t == 0 and force_first else chosen
        rows.append({"step": t, "previous_token": previous, "query": query.tolist(), "state": state.tolist(),
                     "scores": [float(value) if ok else None for value, ok in zip(scores, valid)],
                     "attention": attention.tolist(), "context": context.tolist(),
                     "probabilities": probabilities.tolist(), "argmax_token": chosen, "emitted_token": consumed})
        output.append(consumed)
        if consumed == module.EOS:
            break
        previous = consumed
    return {"input": {"lemma": row["lemma"], "feature": row["feature"]}, "source_ids": ids,
            "memory": memory.tolist(), "projected_keys": keys.tolist(), "valid": valid.tolist(),
            "prediction": "".join(module.TOKENS[token] for token in output if token != module.EOS),
            "ended_with_eos": output[-1] == module.EOS, "rows": rows}

@torch.no_grad()
def native_trace(model, row, force_first=None, pad=0, admit_padding=False):
    source, lengths = module.source_batch([row])
    if pad:
        source = torch.nn.functional.pad(source, (0, pad), value=module.PAD)
    cache, state, fed = model.encode(source, lengths)
    if admit_padding:
        cache = (cache[0], cache[1], torch.ones_like(cache[2]))
    previous, result = torch.tensor([module.BOS]), []
    for t in range(16):
        logits, state, fed, attention, context, _ = model.step(previous, state, fed, cache)
        probabilities = logits.softmax(-1)
        chosen = int(probabilities.argmax(-1)[0])
        emitted = module.INDEX[force_first] if t == 0 and force_first else chosen
        result.append({"attention": attention[0].tolist(), "probabilities": probabilities[0].tolist(),
                       "context": context[0].tolist(), "emitted": emitted})
        if emitted == module.EOS:
            break
        previous = torch.tensor([emitted])
    return result

def model_from(run):
    model = module.AttentiveInflector(run["kind"])
    expected = model.state_dict()
    model.load_state_dict({name: torch.tensor(value, dtype=expected[name].dtype) for name, value in run["weights"].items()})
    return model.eval()

def main():
    report = json.loads((ROOT/"calculated-inputs.json").read_text())
    train, development = module.load_records()
    def record(lemma, feature):
        return next(row for row in train+development if row["lemma"] == lemma and row["feature"] == feature)
    fixtures = [
        ("worked", record("lactate", "past"), {}),
        ("fresh", record("cash", "past"), {}),
        ("fresh_source_edit", {"lemma": "cask", "feature": "past"}, {}),
        ("fresh_request_edit", record("cash", "participle"), {}),
        ("fresh_forced_prefix", record("cash", "past"), {"force_first": "b"}),
        ("fresh_extra_padding", record("cash", "past"), {"pad": 2}),
        ("fresh_broken_mask", record("cash", "past"), {"pad": 2, "admit_padding": True}),
        ("fresh_display_label_null", {**record("cash", "past"), "display_label": "Another display name"}, {}),
    ]
    output = {"versions": report["versions"], "source_data_sha256": hashlib.sha256((ROOT/"english-inflections.csv").read_bytes()).hexdigest(),
              "runs": [], "traces": {}, "max_native_numpy_probability_error": 0.}
    for run in report["runs"]:
        model = model_from(run)
        actual = module.greedy(model, [{"lemma": row["lemma"], "feature": row["feature"]} for row in development])
        assert [item["tokens"] for item in actual] == [item["tokens"] for item in run["development_predictions"]]
        assert module.summarize(development, actual)["exact"] == run["development"]["exact"]
        slices = {}
        for name, keep in (("length3to5", lambda row: len(row["lemma"]) <= 5), ("length6to8", lambda row: len(row["lemma"]) >= 6)):
            chosen = [i for i, row in enumerate(development) if keep(row)]
            slices[name] = module.summarize([development[i] for i in chosen], [actual[i] for i in chosen])
        output["runs"].append({"kind": run["kind"], "seed": run["seed"], "slices": slices})
        if run["seed"] != 1:
            continue
        traces = {}
        for name, row, kwargs in fixtures:
            trace = manual_trace(run, row, **kwargs)
            native = native_trace(model, row, **kwargs)
            assert len(trace["rows"]) == len(native)
            for a, b in zip(trace["rows"], native):
                assert a["emitted_token"] == b["emitted"]
                error = max(abs(np.array(a["probabilities"])-b["probabilities"]))
                output["max_native_numpy_probability_error"] = max(output["max_native_numpy_probability_error"], float(error))
                assert np.allclose(a["probabilities"], b["probabilities"], atol=3e-5)
                assert np.allclose(a["attention"], b["attention"], atol=3e-5)
                assert np.allclose(a["context"], b["context"], atol=3e-5)
            traces[name] = trace
        for name in ("fresh_extra_padding", "fresh_display_label_null"):
            assert np.allclose(traces[name]["rows"][0]["probabilities"], traces["fresh"]["rows"][0]["probabilities"], atol=1e-12)
        assert np.allclose(traces["fresh_forced_prefix"]["rows"][0]["probabilities"], traces["fresh"]["rows"][0]["probabilities"])
        mask_difference = max(float(np.max(abs(np.array(a["probabilities"])-b["probabilities"])))
                              for a, b in zip(traces["fresh_broken_mask"]["rows"], traces["fresh"]["rows"]))
        assert mask_difference > 1e-5
        assert max(sum(row["attention"][-2:]) for row in traces["fresh_broken_mask"]["rows"]) > .1
        assert not np.allclose(traces["fresh_source_edit"]["rows"][0]["probabilities"], traces["fresh"]["rows"][0]["probabilities"])
        source, lengths, previous, target = module.batch([record("cash", "past"), record("lactate", "past")])
        original, alpha = model(source, lengths, previous)
        padded, padded_alpha = model(torch.nn.functional.pad(source, (0, 3)), lengths, previous)
        assert torch.allclose(original[..., module.ALLOWED], padded[..., module.ALLOWED], atol=1e-6)
        assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-6)
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(original.flatten(0, 1), target.flatten(), ignore_index=module.PAD)
        loss.backward()
        gradient = model.key_projection.weight.grad.norm().item()
        assert gradient > 0 and torch.isfinite(model.key_projection.weight.grad).all()
        traces["key_projection_gradient_norm"] = gradient
        output["traces"][run["kind"]] = traces
    # Input feeding is an optional architecture path, checked on an initialized model, not a trained result.
    torch.manual_seed(19)
    feeding = module.AttentiveInflector("general", input_feeding=True)
    source, lengths = module.source_batch([record("cash", "past")])
    cache, state, fed = feeding.encode(source, lengths)
    first = feeding.step(torch.tensor([module.BOS]), state, fed, cache)
    second = feeding.step(torch.tensor([module.INDEX["c"]]), first[1], first[2], cache)
    zero_fed = feeding.step(torch.tensor([module.INDEX["c"]]), first[1], torch.zeros_like(first[2]), cache)
    feeding_difference = float((second[0].softmax(-1)-zero_fed[0].softmax(-1)).abs().max().detach())
    assert feeding_difference > 1e-6
    output["initialized_input_feeding_check"] = {"seed": 19, "parameters": sum(p.numel() for p in feeding.parameters()),
                                               "second_output_probability_max_difference": feeding_difference,
                                               "trained": False}
    baseline_path = ROOT.parent/"sequence-to-sequence-encoder-decoder"/"calculated-inputs.json"
    baseline = json.loads(baseline_path.read_text())
    output["prior_baseline"] = {"origin": "../sequence-to-sequence-encoder-decoder/calculated-inputs.json",
            "source_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "baselines": baseline["baselines"],
            "runs": [{key: run[key] for key in ("seed", "parameters", "train", "development", "checkpoints")} for run in baseline["runs"]]}
    (ROOT/"mechanics-results.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"error": output["max_native_numpy_probability_error"],
                     "fixtures": {kind: {name: trace["prediction"] for name, trace in traces.items() if isinstance(trace, dict)}
                                  for kind, traces in output["traces"].items()},
                     "feeding": output["initialized_input_feeding_check"]}, indent=2))

if __name__ == "__main__":
    main()
