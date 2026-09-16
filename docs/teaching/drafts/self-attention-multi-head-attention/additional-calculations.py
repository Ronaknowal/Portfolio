"""Reproduce changed practice fixtures using the already fitted small model.

Run after author-calculations.py. No training or network access.
"""
import importlib.util
import json
import math
from pathlib import Path

import torch


PACKET = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("attention_author", PACKET / "author-calculations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
saved = json.loads((PACKET / "attention-model.json").read_text(encoding="utf-8"))
model = module.TrajectoryClassifier("attention-2").eval()
model.load_state_dict({name: torch.tensor(values) for name, values in saved["state_dict"].items()})
points = torch.tensor(saved["probes"]["points"])[None]
with torch.no_grad():
    initial = model(points)
    doubled = model(points.repeat_interleave(2, dim=1))
    first_repeated = model(torch.cat([points, points[:, :1]], dim=1))
first_weight = math.exp(math.sqrt(2)) / (1 + math.exp(math.sqrt(2)))
query = torch.tensor([[.5, 1.5]], dtype=torch.float64)
keys = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
values = torch.tensor([[2., 0.], [0., 2.], [1., 1.]], dtype=torch.float64)
base_output, base_weights = module.attention(query, keys, values)
edited_values = values.clone(); edited_values[2, 1] = 2.
value_output, value_weights = module.attention(query, keys, edited_values)
edited_keys = keys.clone(); edited_keys[0, 1] = 1.
key_output, key_weights = module.attention(query, edited_keys, values)
mask_query = torch.tensor([[0., 1.]], dtype=torch.float64)
mask_keys = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [.5, -1.]], dtype=torch.float64)
mask_values = torch.tensor([[1., 0.], [0., 2.], [3., 1.], [-2., 3.]], dtype=torch.float64)
mask_edit = mask_values.clone(); mask_edit[3, 0] = 4.
legal = torch.tensor([[True, True, False, False]])
legal_before, _ = module.attention(mask_query, mask_keys, mask_values, legal)
legal_after, _ = module.attention(mask_query, mask_keys, mask_edit, legal)
leaky_before, _ = module.attention(mask_query, mask_keys, mask_values)
leaky_after, _ = module.attention(mask_query, mask_keys, mask_edit)
result = {
    "message_lab": {"query": query.tolist(), "weights": base_weights.tolist(), "output": base_output.tolist(),
                    "value_edit_output": value_output.tolist(), "value_edit_weight_error": float((base_weights-value_weights).abs().max()),
                    "key_edit_weights": key_weights.tolist(), "key_edit_output": key_output.tolist()},
    "mask_lab": {"causal_before": legal_before.tolist(), "causal_after": legal_after.tolist(),
                 "leaky_before": leaky_before.tolist(), "leaky_after": leaky_after.tolist()},
    "practice1": {
        "weights": [first_weight, 1 - first_weight],
        "output": [first_weight + 5 * (1 - first_weight), 3 * first_weight + (1 - first_weight)],
        "edited_second_coordinate": 3 * first_weight + 5 * (1 - first_weight),
    },
    "repeat_all_points_max_logit_error": float((initial - doubled).abs().max()),
    "repeat_first_point_max_logit_change": float((initial - first_repeated).abs().max()),
    "practice5": {"causal_mean_entropy": math.log(24) / 4, "two_donor_entropy": -.8 * math.log(.8) - .2 * math.log(.2)},
    "practice7": {"weights": [1 / 7, 2 / 7, 4 / 7], "output": 31 / 7, "equal_block_average": 25 / 6},
    "practice8": {"entropy_floor": 3 / 8 * math.log(10)},
    "two_heads_output": [math.exp(1) / (math.exp(1) + 1), .5],
}
(PACKET / "additional-fixtures.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
