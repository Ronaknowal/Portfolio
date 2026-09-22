from pathlib import Path
import json
import runpy
import torch

folder = Path(__file__).resolve().parent
api = runpy.run_path(str(folder / "inflection-seq2seq.py"))
saved = json.loads((folder / "calculated-inputs.json").read_text())
model = api["Inflector"]().eval()
weights = saved["runs"][0]["weights"]
model.load_state_dict({
    key: torch.tensor(value, dtype=torch.bool if key == "invalid_output" else torch.float32)
    for key, value in weights.items()
})
query = {"lemma": "lactate", "feature": "past"}
for limit in (3, 16):
    result = api["greedy"](model, [query], max_output=limit)[0]
    print(limit, result["prediction"], result["ended_with_eos"])
