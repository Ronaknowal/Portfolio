"""Bounded author math and fixed-weight mechanism traces, separate from fitting."""
from pathlib import Path
import importlib.util
import json
import math
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("inflection_seq2seq", ROOT/"inflection-seq2seq.py")
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)
torch.set_num_threads(1)


def scalar_joint(inputs, target=(0, 1), learning_rate=.1):
    # Two-class output A/EOS; fixed decoder parameters, one trainable encoder input weight.
    weight = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    state = torch.tensor(0., dtype=torch.float64)
    encoder = []
    for value in inputs:
        state = torch.tanh(weight*value+.4*state+.1)
        encoder.append(state)
    context = state
    probabilities, decoder = [], []
    for previous_embedding in (.1, .4):
        state = torch.tanh(.6*previous_embedding+.5*state+.05)
        decoder.append(state)
        probabilities.append(torch.stack([state, -state]).softmax(0))
    losses = [-probability[index].log() for probability, index in zip(probabilities, target)]
    loss = sum(losses)/2
    gradient = torch.autograd.grad(loss, weight)[0]
    after_weight = weight.detach()-learning_rate*gradient
    def evaluate(w):
        s = torch.tensor(0., dtype=torch.float64)
        for value in inputs:
            s = torch.tanh(w*value+.4*s+.1)
        costs = []
        for previous_embedding, correct in zip((.1, .4), target):
            s = torch.tanh(.6*previous_embedding+.5*s+.05)
            costs.append(-torch.stack([s, -s]).log_softmax(0)[correct])
        return float(sum(costs)/2)
    finite_difference = (evaluate(.70001)-evaluate(.69999))/.00002
    assert abs(float(gradient)-finite_difference) < 1e-9
    return {"inputs": inputs, "target_ids_A_EOS": target, "encoder_states": [float(x.detach()) for x in encoder],
            "context": float(context.detach()), "decoder_states": [float(x.detach()) for x in decoder],
            "probabilities_A_EOS": [x.detach().tolist() for x in probabilities],
            "per_token_loss": [float(x.detach()) for x in losses], "mean_loss": float(loss.detach()),
            "encoder_input_weight_gradient": float(gradient), "central_difference": finite_difference,
            "learning_rate": learning_rate, "updated_weight": float(after_weight), "updated_mean_loss": evaluate(after_weight)}


def gru_step(vector, hidden, weights, prefix):
    incoming = weights[prefix+".weight_ih_l0"]@vector+weights[prefix+".bias_ih_l0"]
    recurrent = weights[prefix+".weight_hh_l0"]@hidden+weights[prefix+".bias_hh_l0"]
    ir, iz, inn = np.split(incoming, 3)
    hr, hz, hn = np.split(recurrent, 3)
    reset, retain = 1/(1+np.exp(-(ir+hr))), 1/(1+np.exp(-(iz+hz)))
    candidate = np.tanh(inn+reset*hn)
    return retain*hidden+(1-retain)*candidate


def manual_trace(record, weights, prefix_override=None, context_override=None, max_output=16):
    weights = {key: np.asarray(value) for key, value in weights.items()}
    ids = lesson.source_ids(record["lemma"], record["feature"])
    state, encoder = np.zeros(64), []
    for token in ids:
        state = gru_step(weights["embedding.weight"][token], state, weights, "encoder")
        encoder.append(state.tolist())
    context = state.copy()
    state = context if context_override is None else np.asarray(context_override)
    previous, rows, output = lesson.BOS, [], []
    for position in range(max_output):
        state = gru_step(weights["embedding.weight"][previous], state, weights, "decoder")
        logits = weights["readout.weight"]@state+weights["readout.bias"]
        logits[weights["invalid_output"].astype(bool)] = -np.inf
        probability = np.exp(logits-logits.max()); probability /= probability.sum()
        selected = int(probability.argmax())
        if prefix_override is not None and position < len(prefix_override):
            selected = lesson.INDEX[prefix_override[position]]
        rows.append({"previous_token": lesson.TOKENS[previous], "hidden": state.tolist(),
                     "probabilities": probability.tolist(), "selected_token": lesson.TOKENS[selected]})
        output.append(selected)
        if selected == lesson.EOS:
            break
        previous = selected
    return {"lemma": record["lemma"], "feature": record["feature"], "source_ids": ids,
            "encoder_states": encoder, "context": context.tolist(), "decoder": rows,
            "generated": "".join(lesson.TOKENS[token] for token in output if token != lesson.EOS),
            "output_ids": output, "ended_with_eos": output[-1] == lesson.EOS}


def tree_search(first_a=.6, after_a_end=.51, after_b_end=.9, width=1):
    # Constructed finite distribution: first A/B, then EOS/C, then mandatory EOS.
    states = [("", 1.)]
    history = []
    for step in range(3):
        expanded = []
        for path, probability in states:
            if path.endswith("!"):
                expanded.append((path, probability))
            elif path == "":
                expanded += [("A", probability*first_a), ("B", probability*(1-first_a))]
            elif path in ("A", "B"):
                end = after_a_end if path == "A" else after_b_end
                expanded += [(path+"!", probability*end), (path+"C", probability*(1-end))]
            else:
                expanded.append((path+"!", probability))
        states = sorted(expanded, key=lambda item: (-item[1], item[0]))[:width]
        history.append(states)
    return {"parameters": [first_a, after_a_end, after_b_end], "beam_width": width, "history": history, "best": states[0]}


def main():
    trained = json.loads((ROOT/"calculated-inputs.json").read_text())
    assert len(trained["runs"]) == 3
    train, development = lesson.load_records()
    model = lesson.Inflector().eval()
    weights = trained["runs"][0]["weights"]
    model.load_state_dict({key: torch.tensor(value, dtype=torch.bool if key == "invalid_output" else torch.float32) for key, value in weights.items()})
    worked = train[0]
    fresh = development[0]
    traces = {}
    for label, record in (("worked_lactate_past", worked), ("fresh_emmove_past", fresh),
                          ("fresh_emmove_participle", {**fresh, "feature": "participle"}),
                          ("fresh_source_edit_emmode", {**fresh, "lemma": "emmode"})):
        trace = manual_trace(record, weights)
        native = lesson.greedy(model, [record])[0]
        assert trace["output_ids"] == native["tokens"]
        x, lengths, dec_input, targets = lesson.batch([record])
        with torch.no_grad():
            embedded = model.embedding(x)
            sequence, final = model.encoder(embedded)
        error = float(np.max(np.abs(sequence[0].numpy()-trace["encoder_states"])))
        assert error < 3e-6
        trace["native_encoder_max_error"] = error
        with torch.no_grad():
            native_state = model.encode(x, lengths)
            parity = []
            previous = torch.tensor([lesson.BOS])
            for token, manual_step in zip(trace["output_ids"], trace["decoder"]):
                native_logits, native_state = model.step(previous, native_state)
                parity.append(float(np.max(np.abs(native_logits.softmax(-1)[0].numpy()-manual_step["probabilities"]))))
                previous = torch.tensor([token])
        trace["native_decoder_probability_max_error"] = max(parity)
        assert max(parity) < 3e-6
        traces[label] = trace
    fresh_trace = traces["fresh_emmove_past"]
    traces["fresh_forced_first_a"] = manual_trace(fresh, weights, prefix_override="a")
    traces["fresh_zero_context"] = manual_trace(fresh, weights, context_override=np.zeros(64))
    traces["fresh_context_from_lactate"] = manual_trace(fresh, weights, context_override=traces["worked_lactate_past"]["context"])
    replay = manual_trace(fresh, weights, prefix_override=fresh_trace["generated"][:2])
    assert replay["output_ids"] == fresh_trace["output_ids"]
    original_first = fresh_trace["decoder"][0]["probabilities"]
    forced_first = traces["fresh_forced_first_a"]["decoder"][0]["probabilities"]
    assert original_first == forced_first
    assert traces["fresh_context_from_lactate"]["output_ids"] == traces["worked_lactate_past"]["output_ids"]
    for trace in traces.values():
        for row in trace["decoder"]:
            assert abs(sum(row["probabilities"])-1) < 1e-12
    # Test valid-position loss and sequence construction, including a target edit.
    small = [{"lemma": "care", "feature": "past", "form": "cared"}, {"lemma": "try", "feature": "past", "form": "tried"}]
    x, lengths, decoder_input, targets = lesson.batch(small)
    small_decoder_input = decoder_input.tolist()
    with torch.no_grad():
        logits = model(x, lengths, decoder_input)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=lesson.PAD)
    expanded_logits = torch.cat([logits, torch.zeros(len(small), 2, len(lesson.TOKENS))], dim=1)
    expanded_targets = torch.cat([targets, torch.full((len(small), 2), lesson.PAD)], dim=1)
    masked = F.cross_entropy(expanded_logits.flatten(0, 1), expanded_targets.flatten(), ignore_index=lesson.PAD)
    assert torch.allclose(loss, masked)
    # Teacher-forced past targets do not affect the encoder; a changed future token cannot affect earlier logits.
    changed_target = {**small[0], "form": "carts"}
    x2, lengths2, changed_input, _ = lesson.batch([changed_target, small[1]])
    with torch.no_grad():
        changed_logits = model(x2, lengths2, changed_input)
    assert torch.equal(logits[0, :4, lesson.ALLOWED], changed_logits[0, :4, lesson.ALLOWED])
    source, lengths, decoder_input, targets = lesson.batch(train[:4])
    model.zero_grad(set_to_none=True)
    loss_joint = F.cross_entropy(model(source, lengths, decoder_input).flatten(0, 1), targets.flatten(), ignore_index=lesson.PAD)
    loss_joint.backward()
    full_gradient = float(model.encoder.weight_ih_l0.grad.norm())
    model.zero_grad(set_to_none=True)
    fixed_state = model.encode(source, lengths).detach()
    hidden, _ = model.decoder(model.embedding(decoder_input), fixed_state)
    loss_detached = F.cross_entropy(model.output_logits(hidden).flatten(0, 1), targets.flatten(), ignore_index=lesson.PAD)
    loss_detached.backward()
    assert model.encoder.weight_ih_l0.grad is None
    score_examples = [{"length": length, "log_probability": logp,
                       "alpha_0": logp, "alpha_1": logp/((5+length)/6)} for length, logp in ((3, -1.5), (6, -1.8))]
    report = {"scalar_worked": scalar_joint([.2, .8]), "scalar_fresh": scalar_joint([-.3, .6]),
              "scalar_fresh_input_edit": scalar_joint([-.3, .2]), "scalar_zero_rate": scalar_joint([-.3, .6], learning_rate=0),
              "traces": traces, "repeat_prefix_null": True, "first_distribution_before_forced_edit_null": True,
              "joint_encoder_gradient_norm": full_gradient, "detached_encoder_gradient_absent": True,
              "loss_mask": {"inputs": small, "decoder_input_ids": small_decoder_input,
                            "valid_loss": float(loss), "added_padding_valid_loss": float(masked),
                            "future_target_edit_first4_logits_unchanged": True},
              "tree_worked_greedy": tree_search(), "tree_worked_beam2": tree_search(width=2),
              "tree_fresh_greedy": tree_search(.55, .6, .85), "tree_fresh_beam2": tree_search(.55, .6, .85, width=2),
              "tree_fresh_branch_edit": tree_search(.55, .9, .85, width=2), "length_score": score_examples,
              "slices": {}, "fresh_cap_3": lesson.greedy(model, [fresh], max_output=3)[0]}
    for run in trained["runs"]:
        groups = {}
        for feature in ("past", "participle", "third_person"):
            selected = [(row, output) for row, output in zip(development, run["development_predictions"]) if row["feature"] == feature]
            groups[feature] = lesson.summarize([row for row, _ in selected], [output for _, output in selected])
        for label, predicate in (("length3to5", lambda length: length <= 5), ("length6to8", lambda length: length > 5)):
            selected = [(row, output) for row, output in zip(development, run["development_predictions"]) if predicate(len(row["lemma"]))]
            groups[label] = lesson.summarize([row for row, _ in selected], [output for _, output in selected])
        report["slices"][str(run["seed"])] = groups
    (ROOT/"mechanics-results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"scalar_worked": report["scalar_worked"], "scalar_fresh": report["scalar_fresh"],
                     "trace_outputs": {label: trace["generated"] for label, trace in traces.items()},
                     "joint_encoder_gradient_norm": full_gradient, "slices": report["slices"],
                     "fresh_cap_3": report["fresh_cap_3"]}, indent=2))


if __name__ == "__main__":
    main()
