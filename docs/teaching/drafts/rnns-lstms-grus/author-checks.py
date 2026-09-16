"""Bounded author checks for changed questions and retained calculation consistency."""
from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
lesson = (ROOT / "lesson.md").read_text(encoding="utf-8")
fence = chr(96) * 3
programs = re.findall(fence + r"python\n(.*?)\n" + fence, lesson, flags=re.S)
assert len(programs) == 2
assert programs[0].strip() == (ROOT / "pen-sequence-learning.py").read_text(encoding="utf-8").strip()
# Execute the actual small displayed packing program; do not repeat nine fits.
exec(compile(programs[1], "displayed-packing-program", "exec"), {})
assert lesson.count("<details>") == lesson.count("</details>") == 16
assert "<details open" not in lesson
source = Path("src/learn/data/topics/rnns-lstms-grus.jsx")
assert hashlib.sha256(source.read_bytes()).hexdigest() == "7fd7f02af2f2eca7aceb5b34422793e571c186f8d8525138bfdf2c913ce0f2cf"
assert hashlib.sha256((ROOT / "pen-trajectories.csv").read_bytes()).hexdigest() == "e9f7d82554a7704edd3d63b07f62b575572675bfa6927e8fea3c52ec879be82f"
data = json.loads((ROOT / "calculated-inputs.json").read_text())
assert len(data["runs"]) == 9
for run in data["runs"]:
    assert len(run["trajectory"]) == 5
    for mode in ("final", "reversed", "swapped_points_3_4"):
        probabilities = np.array(run[mode]["probabilities"])
        assert probabilities.shape == (300, 10)
        assert np.isfinite(probabilities).all()
        assert np.max(np.abs(probabilities.sum(1)-1)) < 2e-6
        expected = int((probabilities.argmax(1) == data["development_labels"]).sum())
        assert expected == run[mode]["correct"]
h1 = math.tanh(.42)
h2 = math.tanh(.16 + .6*h1 + .1)
h3 = math.tanh(.56 + .6*h2 + .1)
assert abs(h3 - .7335640857622394) < 1e-12
cell = .8*(-.6)+.25*.4
assert .9*.8 + 0*(-.5) == .9*.8 + 0*(.7)
assert .6*math.tanh(.62) != .2*math.tanh(.62)
reset, state = np.array([.2, .8]), np.array([1., 2.])
matrix, bias = np.array([[1., 2.], [3., 4.]]), np.array([.5, -.5])
assert np.allclose(matrix@(np.ones(2)*state)+bias, np.ones(2)*(matrix@state+bias))
diagonal = np.diag([2., 3.])
assert np.allclose(diagonal@(reset*state), reset*(diagonal@state))
assert not np.allclose(diagonal@(reset*state)+bias, reset*(diagonal@state+bias))
# Independently finite-difference the retained complete LSTM state Jacobian.
def local_lstm(point):
    hidden, cell = point
    sigmoid = lambda x: 1/(1+np.exp(-x))
    new_cell = sigmoid(1+.4*hidden)*cell + sigmoid(.2+.3*hidden)*np.tanh(-.1+.7*hidden)
    new_hidden = sigmoid(.5-.2*hidden)*np.tanh(new_cell)
    return np.array([new_hidden, new_cell])
point = np.array([.2, .8])
numeric = np.column_stack([(local_lstm(point+np.eye(2)[j]*1e-5)-local_lstm(point-np.eye(2)[j]*1e-5))/2e-5 for j in range(2)])
mechanics = json.loads((ROOT / "mechanics-results.json").read_text())
exact = np.array(mechanics["lstm_full_state"]["jacobian_rows_h_c_columns_h_c"])
assert np.max(np.abs(exact-numeric)) < 1e-9
report = {
    "practice_middle_edit": {"first": h1, "second": h2, "final": h3},
    "practice_negative_cell": {"retained": -.48, "written": .1, "cell": cell, "hidden": .5*math.tanh(cell)},
    "practice_gru": .9*.8+.1*(-.2),
    "practice_forget_80_percent_after_50": .8**(1/50),
    "lstm_full_jacobian_finite_difference_max_error": float(np.max(np.abs(exact-numeric))),
    "gru_all_one_reset_null": True, "gru_diagonal_zero_bias_null": True,
    "gru_diagonal_nonzero_bias_contrast": True,
    "displayed_packing_program_executed": True,
    "displayed_training_program_matches_executed_source": True,
    "saved_run_probability_and_count_checks": 27,
    "closed_hint_solution_blocks": 16,
    "original_published_source_unchanged": True,
    "data_hash_matches": True,
    "scope": "Author arithmetic and input consistency only; no phase-two implementation or formal review."
}
(ROOT / "author-check-results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))
