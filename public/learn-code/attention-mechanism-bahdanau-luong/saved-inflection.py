from pathlib import Path
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
import torch

root = Path.cwd()
spec = importlib.util.spec_from_file_location("inflection", root/"attentive-inflection.py")
inflection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inflection)
torch.set_num_threads(1)
report = json.loads((root/"calculated-inputs.json").read_text(encoding="utf-8"))
run = next(item for item in report["runs"] if item["kind"] == "additive" and item["seed"] == 1)
model = inflection.AttentiveInflector("additive")
shapes = model.state_dict()
model.load_state_dict({
    name: torch.tensor(value, dtype=shapes[name].dtype)
    for name, value in run["weights"].items()
})
result = inflection.greedy(model, [{"lemma": "lactate", "feature": "past"}])[0]
print(result["prediction"], result["ended_with_eos"])
