"""Fresh, unsolved investigation defaults; fixed-model calculations, no fits."""
from pathlib import Path
import csv
import importlib.util
import json
import math
import sys
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("recurrent_mechanics", ROOT / "recurrent-mechanics.py")
mechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanics)


def scalar(inputs, target=-.1, rate=.1):
    theta = torch.tensor([.7, .4, -.05], dtype=torch.float64, requires_grad=True)
    states = [torch.tensor(.1, dtype=torch.float64)]
    for value in inputs:
        states.append(torch.tanh(theta[0]*value + theta[1]*states[-1] + theta[2]))
    loss = (states[-1]-target).square()/2
    loss.backward()
    updated = theta.detach()-rate*theta.grad
    after = torch.tensor(.1, dtype=torch.float64)
    for value in inputs:
        after = torch.tanh(updated[0]*value+updated[1]*after+updated[2])
    return {"inputs": inputs, "h0": .1, "parameters": theta.detach().tolist(), "target": target,
            "rate": rate, "states": [float(x.detach()) for x in states],
            "gradient_wx_wh_b": theta.grad.tolist(), "loss": float(loss.detach()),
            "updated_parameters": updated.tolist(), "updated_final": float(after),
            "updated_loss": float((after-target).square()/2)}


def gate(candidate=.6, input_gate=.3, output=.7):
    retained, written = .85*(-.4), input_gate*candidate
    cell = retained+written
    return {"retained": retained, "written": written, "cell": cell,
            "hidden": output*math.tanh(cell)}


def reset_placement(matrix, reset, bias):
    hidden = np.array([-.5, 1.5])
    matrix, reset, bias = map(np.asarray, (matrix, reset, bias))
    return {"matrix": matrix.tolist(), "reset": reset.tolist(), "bias": bias.tolist(),
            "before": (matrix@(reset*hidden)+bias).tolist(),
            "after": (reset*(matrix@hidden+bias)).tolist()}


def main():
    report = {"B": {"default": scalar([.3, -.5, .2]), "middle_edit": scalar([.3, .1, .2]),
                     "target_only": scalar([.3, -.5, .2], target=.4),
                     "zero_rate": scalar([.3, -.5, .2], rate=0)}}
    assert report["B"]["default"]["states"] == report["B"]["target_only"]["states"]
    assert report["B"]["default"]["states"][1] == report["B"]["middle_edit"]["states"][1]
    assert report["B"]["zero_rate"]["updated_parameters"] == [.7, .4, -.05]
    forget = .65**(1/80)
    memory_curve = [forget**step for step in range(81)]
    write_curve = [1.]
    for step in range(1, 81):
        write_curve.append(forget*write_curve[-1]+(.15 if step == 25 else 0))
    report["C"] = {"inputs": {"old_cell": -.4, "forget": .85, "input": .3, "candidate": .6, "output": .7},
                   "default": gate(), "candidate_edit_to_minus_0_2": gate(candidate=-.2),
                   "output_edit_to_0_2": gate(output=.2),
                   "input_zero": gate(input_gate=0), "input_zero_candidate_edit": gate(candidate=-.2, input_gate=0),
                   "retention": {"desired_fraction": .65, "horizon": 80, "forget": forget,
                                 "half_life": math.log(.5)/math.log(forget),
                                 "curve": memory_curve, "step25_write_0_15_curve": write_curve,
                                 "zero_write_curve": [forget**step for step in range(81)]}}
    assert gate(input_gate=0) == gate(candidate=-.2, input_gate=0)
    assert gate()["cell"] == gate(output=.2)["cell"]
    assert abs(memory_curve[-1]-.65) < 1e-12
    matrix, reset, bias = [[.4, -1.2], [1.1, .5]], [.7, .3], [.2, -.1]
    report["D"] = {"hidden": [-.5, 1.5], "default": reset_placement(matrix, reset, bias),
                   "matrix_0_1_edit_to_minus_0_4": reset_placement([[.4, -.4], [1.1, .5]], reset, bias),
                   "reset_ones": reset_placement(matrix, [1., 1.], bias),
                   "diagonal_zero_bias": reset_placement([[.4, 0], [0, .5]], reset, [0., 0.]),
                   "diagonal_nonzero_bias": reset_placement([[.4, 0], [0, .5]], reset, bias)}
    for name in ("reset_ones", "diagonal_zero_bias"):
        assert np.allclose(report["D"][name]["before"], report["D"][name]["after"], atol=1e-12)
    model_data = json.loads((ROOT/"calculated-inputs.json").read_text())
    with (ROOT/"pen-trajectories.csv").open(newline="", encoding="utf-8") as stream:
        row = next(row for row in csv.DictReader(stream) if row["source_id"] == "pendigits.tes:6")
    sequence = np.array([[float(row[f"{axis}{time+1}"]) for axis in ("x", "y")] for time in range(8)])/50-1
    edited = sequence.copy()
    edited[2, 0] = np.clip(edited[2, 0]+.2, -1, 1)
    report["E"] = {"source_id": row["source_id"], "actual_digit": int(row["digit"]),
                   "default_model": "gru", "selected_probability_class": 1,
                   "input": sequence.tolist(), "edited_input": edited.tolist(), "models": {}}
    for kind, weights in model_data["saved_models"].items():
        original = mechanics.manual_sequence(kind, sequence, weights)
        changed = mechanics.manual_sequence(kind, edited, weights)
        repeated = mechanics.manual_sequence(kind, sequence[::-1][::-1], weights)
        original_p, changed_p = np.array(original[-1]["probabilities"]), np.array(changed[-1]["probabilities"])
        assert original == repeated and original[:2] == changed[:2]
        report["E"]["models"][kind] = {"original_trace": original, "edited_trace": changed,
            "original_argmax": int(original_p.argmax()), "edited_argmax": int(changed_p.argmax()),
            "selected_class_probability_change": float(changed_p[1]-original_p[1]),
            "max_probability_change": float(np.abs(original_p-changed_p).max()),
            "unchanged_prefix_error": 0., "reverse_twice_error": 0., "repeated_input_error": 0.}
    previous = json.loads((ROOT/"mechanics-results.json").read_text())["state_and_padding"]
    cell = nn.GRU(2, 3, batch_first=True).double().eval()
    cell.load_state_dict({key: torch.tensor(value, dtype=torch.float64) for key, value in previous["weights"].items()})
    inputs = torch.tensor(previous["sequence"], dtype=torch.float64)
    full, _ = cell(inputs)
    _, prefix = cell(inputs[:, :3])
    carried, _ = cell(inputs[:, 3:], prefix)
    detached, _ = cell(inputs[:, 3:], prefix.detach())
    reset_values, _ = cell(inputs[:, 3:])
    gradients = {}
    for detach in (False, True):
        tracked = inputs.clone().requires_grad_()
        _, state = cell(tracked[:, :3])
        tail, _ = cell(tracked[:, 3:], state.detach() if detach else state)
        gradients[str(detach)] = torch.autograd.grad(tail.square().sum(), tracked)[0].tolist()
    streams = torch.cat([inputs, inputs.flip(1)])
    _, prefix_states = cell(streams[:, :3])
    correct, _ = cell(streams[:, 3:], prefix_states)
    wrong, _ = cell(streams[:, 3:], prefix_states.flip(1))
    reorder, _ = cell(streams.flip(0)[:, 3:], prefix_states.flip(1))
    def maximum(value):
        return float(value.detach().abs().max())
    report["F"] = {"boundary": 3, "sequence": inputs.tolist(), "weights": previous["weights"],
                   "carry_vs_full_error": maximum(carried-full[:, 3:]),
                   "detach_vs_carry_error": maximum(detached-carried),
                   "reset_last_error": maximum(reset_values[:, -1]-carried[:, -1]),
                   "input_gradients_detach_false_true": gradients,
                   "streams": streams.tolist(), "prefix_states": prefix_states.detach().tolist(),
                   "correct_final": correct[:, -1].detach().tolist(), "wrong_final": wrong[:, -1].detach().tolist(),
                   "wrong_owner_error": maximum(wrong-correct), "reordered_owned_state_error": maximum(reorder-correct.flip(0))}
    assert report["F"]["carry_vs_full_error"] == report["F"]["detach_vs_carry_error"] == 0
    assert np.max(np.abs(gradients["True"][0][:3])) == 0
    short = [inputs[0], inputs[0, :4], inputs[0, :2]]
    lengths = torch.tensor([5, 4, 2])
    padded = pad_sequence(short, batch_first=True)
    changed_padding = padded.clone()
    valid = torch.arange(5)[None] < lengths[:, None]
    changed_padding[~valid] = -3.5
    padding = {}
    for bidirectional in (False, True):
        module = nn.GRU(2, 3, batch_first=True, bidirectional=bidirectional).double().eval()
        weights = previous["padding"][str(bidirectional)]["weights"]
        module.load_state_dict({key: torch.tensor(value, dtype=torch.float64) for key, value in weights.items()})
        raw, _ = module(padded)
        altered, _ = module(changed_padding)
        _, packed_state = module(pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False))
        _, changed_packed = module(pack_padded_sequence(changed_padding, lengths, batch_first=True, enforce_sorted=False))
        expected = torch.stack([module(item[None])[1][:, 0] for item in short], 1)
        padding[str(bidirectional)] = {"weights": weights,
            "valid_forward_error": maximum(raw[..., :3][valid]-altered[..., :3][valid]),
            "valid_all_error": maximum(raw[valid]-altered[valid]),
            "packed_individual_error": maximum(packed_state-expected),
            "packed_padding_edit_error": maximum(packed_state-changed_packed)}
        assert padding[str(bidirectional)]["packed_padding_edit_error"] == 0
    report["F"]["padding"] = {"lengths": lengths.tolist(), "edit_value": -3.5,
                              "inputs": padded.tolist(), "edited_inputs": changed_padding.tolist(), "models": padding}
    (ROOT/"fresh-investigation-fixtures.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"B": report["B"], "C_default": report["C"]["default"],
                      "C_retention": {key: value for key, value in report["C"]["retention"].items() if not isinstance(value, list)},
                      "D": report["D"], "E": {kind: {key: value for key, value in details.items() if not key.endswith("trace")} for kind, details in report["E"]["models"].items()},
                      "F": {key: value for key, value in report["F"].items() if key.endswith("error")},
                      "F_padding": {key: {name: value for name, value in model.items() if name != "weights"} for key, model in padding.items()}}, indent=2))


if __name__ == "__main__":
    main()
