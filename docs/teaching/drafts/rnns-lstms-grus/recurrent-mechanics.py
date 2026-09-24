"""Independent arithmetic, native parity and changed-input author calculations."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def manual_sequence(kind, inputs, weights, hidden=None, cell=None):
    hidden_size = len(weights["recurrent.weight_hh_l0"][0])
    hidden = np.zeros(hidden_size) if hidden is None else np.array(hidden, dtype=float)
    cell = np.zeros(hidden_size) if cell is None else np.array(cell, dtype=float)
    weights = {name: np.array(value) for name, value in weights.items()}
    states = []
    for point in inputs:
        incoming = weights["recurrent.weight_ih_l0"] @ point + weights["recurrent.bias_ih_l0"]
        recurrent = weights["recurrent.weight_hh_l0"] @ hidden + weights["recurrent.bias_hh_l0"]
        previous_hidden, previous_cell = hidden.copy(), cell.copy()
        if kind == "rnn":
            hidden = np.tanh(incoming + recurrent)
            gates = {}
        elif kind == "lstm":
            input_pre, forget_pre, candidate_pre, output_pre = np.split(incoming + recurrent, 4)
            input_gate, forget_gate, output_gate = map(sigmoid, (input_pre, forget_pre, output_pre))
            candidate = np.tanh(candidate_pre)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)
            gates = {"input": input_gate.tolist(), "forget": forget_gate.tolist(),
                     "candidate": candidate.tolist(), "output": output_gate.tolist(),
                     "retained": (forget_gate * previous_cell).tolist(),
                     "written": (input_gate * candidate).tolist(), "cell": cell.tolist()}
        else:
            input_reset, input_update, input_new = np.split(incoming, 3)
            hidden_reset, hidden_update, hidden_new = np.split(recurrent, 3)
            reset = sigmoid(input_reset + hidden_reset)
            update = sigmoid(input_update + hidden_update)
            candidate = np.tanh(input_new + reset * hidden_new)
            hidden = update * hidden + (1 - update) * candidate
            gates = {"reset": reset.tolist(), "retain_update": update.tolist(),
                     "candidate": candidate.tolist()}
        scores = weights["readout.weight"] @ hidden + weights["readout.bias"]
        probability = np.exp(scores - scores.max())
        probability /= probability.sum()
        states.append({"point": np.asarray(point).tolist(), "previous_hidden": previous_hidden.tolist(),
                       "hidden": hidden.tolist(), "gates": gates, "probabilities": probability.tolist()})
    return states


def scalar_credit():
    inputs = np.array([0.4, -0.2, 0.7])
    input_weight, recurrent_weight, bias, target = 0.8, 0.6, 0.1, 0.3
    states = [0.0]
    for value in inputs:
        states.append(math.tanh(input_weight * value + recurrent_weight * states[-1] + bias))
    credit = states[-1] - target
    contributions = []
    for time in range(3, 0, -1):
        pre_credit = credit * (1 - states[time] ** 2)
        contributions.append({"step": time, "hidden_credit": credit, "preactivation_credit": pre_credit,
                              "input_weight": pre_credit * inputs[time - 1],
                              "recurrent_weight": pre_credit * states[time - 1], "bias": pre_credit})
        credit = recurrent_weight * pre_credit
    gradient = {name: sum(row[name] for row in contributions) for name in ("input_weight", "recurrent_weight", "bias")}
    theta = torch.tensor([input_weight, recurrent_weight, bias], dtype=torch.float64, requires_grad=True)
    state = torch.tensor(0., dtype=torch.float64)
    for value in inputs:
        state = torch.tanh(theta[0] * value + theta[1] * state + theta[2])
    loss = (state - target).square() / 2
    loss.backward()
    assert np.max(np.abs(theta.grad.numpy() - list(gradient.values()))) < 1e-12
    changed = theta.detach() - 0.1 * theta.grad
    new_state = torch.tensor(0., dtype=torch.float64)
    for value in inputs:
        new_state = torch.tanh(changed[0] * value + changed[1] * new_state + changed[2])
    return {"inputs": inputs.tolist(), "states": states, "contributions_reverse_order": contributions,
            "gradient": gradient, "loss_before": float(loss.detach()),
            "parameters_after": changed.tolist(), "loss_after": float((new_state-target).square()/2)}


def state_and_padding():
    torch.manual_seed(19)
    sequence = torch.tensor([[[.2, -.1], [.8, .3], [-.4, .9], [.7, -.2], [.1, .5]]], dtype=torch.float64)
    module = nn.GRU(2, 3, batch_first=True).double().eval()
    whole, whole_state = module(sequence)
    first, first_state = module(sequence[:, :2])
    remaining, remaining_state = module(sequence[:, 2:], first_state)
    detached, _ = module(sequence[:, 2:], first_state.detach())
    reset, _ = module(sequence[:, 2:])
    assert torch.allclose(torch.cat([first, remaining], 1), whole, atol=1e-12)
    assert torch.equal(detached, remaining)
    source = sequence.clone().requires_grad_()
    _, boundary = module(source[:, :2])
    end, _ = module(source[:, 2:], boundary)
    full_grad = torch.autograd.grad(end.square().sum(), source)[0]
    source2 = sequence.clone().requires_grad_()
    _, boundary2 = module(source2[:, :2])
    end2, _ = module(source2[:, 2:], boundary2.detach())
    truncated_grad = torch.autograd.grad(end2.square().sum(), source2)[0]
    assert truncated_grad[:, :2].abs().max() == 0
    streams = torch.cat([sequence, sequence.flip(1)], dim=0)
    _, stream_states = module(streams[:, :2])
    correct_streams, _ = module(streams[:, 2:], stream_states)
    wrong_streams, _ = module(streams[:, 2:], stream_states.flip(1))
    reordered_streams, _ = module(streams.flip(0)[:, 2:], stream_states.flip(1))
    assert torch.allclose(reordered_streams, correct_streams.flip(0), atol=1e-12)
    short = [sequence[0], sequence[0, :3], sequence[0, :1]]
    lengths = torch.tensor([5, 3, 1])
    padded = nn.utils.rnn.pad_sequence(short, batch_first=True)
    changed = padded.clone()
    for row, length in enumerate(lengths):
        changed[row, length:] = 4.0
    padding_report = {}
    for bidirectional in (False, True):
        torch.manual_seed(8)
        cell = nn.GRU(2, 3, batch_first=True, bidirectional=bidirectional).double().eval()
        raw, raw_final = cell(padded)
        changed_raw, _ = cell(changed)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, packed_final = cell(packed)
        unpacked, restored_lengths = pad_packed_sequence(packed_output, batch_first=True)
        individual = [cell(item[None])[1][:, 0] for item in short]
        expected = torch.stack(individual, 1)
        assert torch.allclose(expected, packed_final, atol=1e-12)
        assert torch.equal(restored_lengths, lengths)
        valid = torch.arange(5)[None] < lengths[:, None]
        padding_report[str(bidirectional)] = {
            "weights": {name: value.detach().tolist() for name, value in cell.state_dict().items()},
            "packed_final_shape": list(packed_final.shape),
            "packed_individual_error": float((expected-packed_final).abs().max().detach()),
            "edited_padding_valid_forward_error": float((raw[..., :3][valid]-changed_raw[..., :3][valid]).abs().max().detach()),
            "edited_padding_valid_all_error": float((raw[valid]-changed_raw[valid]).abs().max().detach()),
            "padded_vs_true_final_error": float((raw_final-packed_final).abs().max().detach()),
            "last_index_backward_vs_backward_final_error": (
                float((unpacked[torch.arange(3), lengths-1, 3:]-packed_final[1]).abs().max().detach())
                if bidirectional else None)}
    return {"sequence": sequence.tolist(),
            "lengths": lengths.tolist(),
            "weights": {name: value.detach().tolist() for name, value in module.state_dict().items()},
            "two_streams": {"inputs": streams.tolist(), "prefix_states": stream_states.detach().tolist(),
                "correct_final": correct_streams[:, -1].detach().tolist(),
                "wrong_state_final": wrong_streams[:, -1].detach().tolist(),
                "wrong_assignment_max_difference": float((wrong_streams-correct_streams).abs().max().detach()),
                "reordered_owner_matched_max_difference": float((reordered_streams-correct_streams.flip(0)).abs().max().detach())},
            "whole_chunk_error": float((remaining_state-whole_state).abs().max().detach()),
            "reset_vs_carried_last_error": float((reset[:, -1]-remaining[:, -1]).abs().max().detach()),
            "detach_forward_error": float((detached-remaining).abs().max().detach()),
            "full_input_gradient": full_grad.tolist(), "truncated_input_gradient": truncated_grad.tolist(),
            "padding": padding_report}


def main():
    report = json.loads((ROOT / "calculated-inputs.json").read_text(encoding="utf-8"))
    calculations = {"scalar_credit": scalar_credit(), "state_and_padding": state_and_padding(),
                    "memory": [], "saved_model_traces": {}}
    for forget in (0.5, float(sigmoid(1.0)), 0.99, 0.999):
        calculations["memory"].append({"forget": forget, "after_10": forget**10, "after_100": forget**100,
            "half_life_steps": math.log(.5)/math.log(forget),
            "curve": [forget**step for step in range(201)]})
    half_retention = .5**.01
    calculations["half_after_100"] = {"forget": half_retention, "bias": math.log(half_retention/(1-half_retention))}
    calculations["saturated_rnn_jacobian"] = 2*(1-math.tanh(5)**2)
    a, b = np.array([[0., 2.], [0., 0.]]), np.array([[0., 0.], [2., 0.]])
    calculations["matrix_product"] = {"a": a.tolist(), "b": b.tolist(), "a_eigenvalues": np.linalg.eigvals(a).tolist(),
        "b_eigenvalues": np.linalg.eigvals(b).tolist(), "product": (b@a).tolist(), "product_norm": float(np.linalg.norm(b@a, 2))}
    # Complete (h,c) state; f is only one entry of this Jacobian.
    def lstm_state(state):
        hidden, cell = state
        input_gate = torch.sigmoid(.2 + .3*hidden)
        forget = torch.sigmoid(1. + .4*hidden)
        candidate = torch.tanh(-.1 + .7*hidden)
        output = torch.sigmoid(.5 - .2*hidden)
        new_cell = forget*cell + input_gate*candidate
        return torch.stack([output*torch.tanh(new_cell), new_cell])
    point = torch.tensor([.2, .8], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(lstm_state, point).numpy()
    calculations["lstm_full_state"] = {"input_h_c": point.tolist(), "output_h_c": lstm_state(point).tolist(),
        "jacobian_rows_h_c_columns_h_c": jacobian.tolist(), "forget": float(torch.sigmoid(1+.4*point[0]))}
    reset = np.array([.2, .8]); previous = np.array([1., 2.]); matrix = np.array([[1., 2.], [3., 4.]])
    hidden_bias = np.array([.5, -.5])
    calculations["gru_reset_placement"] = {"reset": reset.tolist(), "previous": previous.tolist(),
        "matrix": matrix.tolist(), "hidden_bias": hidden_bias.tolist(),
        "before": (matrix@(reset*previous)+hidden_bias).tolist(),
        "after": (reset*(matrix@previous+hidden_bias)).tolist()}
    calculations["clip"] = {"gradient": [3., 4.], "threshold": 2.,
        "global_norm_clipped": [1.2, 1.6], "component_clipped": [2., 2.]}
    parity_errors = []
    for kind, weights in report["saved_models"].items():
        run = next(row for row in report["runs"] if row["kind"] == kind and row["seed"] == 1)
        examples = run["first_two_examples"]
        traces = []
        for index, sequence in enumerate(examples["coordinates"]):
            actual = manual_sequence(kind, sequence, weights)
            error = float(np.max(np.abs(np.array([row["hidden"] for row in actual])-examples["hidden_states"][index])))
            prob_error = float(np.max(np.abs(np.array([row["probabilities"] for row in actual])-examples["prefix_readout_probabilities"][index])))
            assert error < 3e-6 and prob_error < 3e-6
            parity_errors.append({"kind": kind, "source_id": examples["source_ids"][index], "hidden_error": error,
                                  "probability_error": prob_error})
            changed = np.array(sequence)
            changed[2, 0] = np.clip(changed[2, 0] + .2, -1, 1)
            changed_trace = manual_sequence(kind, changed, weights)
            null_trace = manual_sequence(kind, sequence, weights)
            assert actual == null_trace
            traces.append({"source_id": examples["source_ids"][index], "actual_digit": examples["labels"][index],
                "trace": actual, "point_3_x_plus_0_2": changed_trace,
                "changed_final_probability_max_difference": float(np.max(np.abs(np.array(actual[-1]["probabilities"])-changed_trace[-1]["probabilities"])))})
        calculations["saved_model_traces"][kind] = traces
    calculations["native_parity"] = parity_errors
    # Nonzero initial states and both bias vectors, independent of the fitted zeros.
    torch.manual_seed(41)
    nonzero_parity = {}
    for kind, constructor in (("rnn", nn.RNN), ("lstm", nn.LSTM), ("gru", nn.GRU)):
        native = constructor(2, 3, batch_first=True).double()
        sequence = torch.randn(1, 4, 2, dtype=torch.float64)
        hidden = torch.randn(1, 1, 3, dtype=torch.float64)
        cell = torch.randn(1, 1, 3, dtype=torch.float64)
        output, _ = native(sequence, (hidden, cell) if kind == "lstm" else hidden)
        weights = {"recurrent."+name: value.detach().tolist() for name, value in native.state_dict().items()}
        weights.update({"readout.weight": np.eye(3).tolist(), "readout.bias": [0., 0., 0.]})
        manual = manual_sequence(kind, sequence[0].numpy(), weights, hidden[0, 0].numpy(), cell[0, 0].numpy())
        error = float(np.max(np.abs(np.array([row["hidden"] for row in manual])-output[0].detach().numpy())))
        assert error < 1e-12
        nonzero_parity[kind] = error
    calculations["nonzero_state_bias_parity"] = nonzero_parity
    (ROOT / "mechanics-results.json").write_text(json.dumps(calculations, separators=(",", ":"))+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in calculations.items() if key not in ("saved_model_traces", "memory")}, indent=2))


if __name__ == "__main__":
    main()
