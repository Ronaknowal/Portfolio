"""Run shipped attention programs and compare independent inference paths."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'docs/teaching/evidence/recurrent-attention-native.json').write_text(
    json.dumps({'passed': False, 'reason': 'Verification started; all checks must pass before success is recorded.'}) + '\n'
)
ASSETS = ROOT / "public/learn-code/attention-mechanism-bahdanau-luong"
DRAFT = ROOT / "docs/teaching/drafts/attention-mechanism-bahdanau-luong"
torch.set_num_threads(1)
sys.dont_write_bytecode = True


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load("attention_native", ASSETS / "attentive-inflection.py")
scratch = load("attention_scratch", ASSETS / "attention-inference.py")
mechanics = load("attention_mechanics", ASSETS / "attention-mechanics.py")
calculations = load("attention_calculations", ASSETS / "attention-calculations.py")
report = json.loads((ASSETS / "calculated-inputs.json").read_text())
saved_traces = json.loads((ASSETS / "mechanics-results.json").read_text())
analytic = json.loads((ASSETS / "analytic-results.json").read_text())
checks, source_files = [], []
for name in ["attentive-inflection.py", "attention-calculations.py", "attention-mechanics.py", "english-inflections.csv", "calculated-inputs.json", "mechanics-results.json", "analytic-results.json"]:
    assert (ASSETS / name).read_bytes() == (DRAFT / name).read_bytes()
    source_files.append(ASSETS / name)
checks.append("Canonical downloads preserve exact prepared source/data/measurement bytes")

train, development = native.load_records()
assert len(train) == 1353 and len(development) == 447
maximum_probability_error = 0.0
for run in report["runs"]:
    model = mechanics.model_from(run)
    assert sum(parameter.numel() for parameter in model.parameters()) == run["parameters"]
    outputs = native.greedy(model, development)
    assert [row["tokens"] for row in outputs] == [row["tokens"] for row in run["development_predictions"]]
    assert native.summarize(development, outputs) == {key: run["development"][key] for key in native.summarize(development, outputs)}
    if run["seed"] != 1:
        continue
    weights = {key: np.array(value) for key, value in run["weights"].items()}
    for lemma, feature, prefix, cap in [("lactate", "past", "", 16), ("cash", "past", "", 16), ("cask", "past", "", 16), ("cash", "participle", "b", 16), ("prone", "third_person", "ab", 16), ("aaa", "past", "", 1), ("zzzzzzzz", "participle", "", 16)]:
        word, ended, rows = scratch.decode(lemma, feature, weights, run["kind"], prefix, cap)
        source, lengths = native.source_batch([{"lemma": lemma, "feature": feature}])
        cache, state, fed = model.encode(source, lengths)
        previous = torch.tensor([native.BOS])
        for step, row in enumerate(rows):
            with torch.no_grad():
                logits, state, fed, attention, context, scores = model.step(previous, state, fed, cache)
                probabilities = logits.softmax(-1)[0].numpy()
            error = float(np.max(abs(probabilities - row["probabilities"])))
            maximum_probability_error = max(maximum_probability_error, error)
            np.testing.assert_allclose(row["probabilities"], probabilities, atol=3e-6)
            np.testing.assert_allclose(row["attention"], attention[0].detach().numpy(), atol=3e-6)
            np.testing.assert_allclose(row["context"], context[0].detach().numpy(), atol=3e-6)
            previous = torch.tensor([row["emitted"]])
        assert word == "".join(native.TOKENS[row["emitted"]] for row in rows if row["emitted"] != native.EOS)
        assert ended == (rows[-1]["emitted"] == native.EOS)
checks.append("All six saved models reproduce 447 development sequences each; fourteen new NumPy/native state, attention and output comparisons")

for fixture in analytic["fixtures"].values():
    result = calculations.calculate(fixture["query"], np.array(fixture["keys"]), np.array(fixture["values"]), fixture["valid"], fixture["rate"])
    np.testing.assert_allclose(result["query_gradient"], fixture["query_gradient"], atol=1e-12)
    np.testing.assert_allclose(result["next_loss"], fixture["next_loss"], atol=1e-12)
checks.append("Every constructed query derivative matches native autograd and independent finite differences")
for name in ["attention-inference.py", "saved-inflection.py"]:
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ASSETS / name)], cwd=ASSETS, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "lactated True", result.stdout
    source_files.append(ASSETS / name)
checks.append("Both complete learner entry programs execute and print lactated True")

source_files.append(Path(__file__))
evidence = {"passed": True, "date": "2026-09-26", "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}, "checks": checks, "maximum_probability_error": maximum_probability_error, "training": "Unchanged six prepared fits reused; no new training or performance benchmark claimed", "reviewedFiles": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}}
(ROOT / "docs/teaching/evidence/recurrent-attention-native.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"checks": len(checks), "maximum_probability_error": maximum_probability_error}))
