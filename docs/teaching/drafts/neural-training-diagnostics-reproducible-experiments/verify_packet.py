"""Bounded author checks for the written packet; no website/browser checks."""

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine

PACKET = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((PACKET / name).read_text(encoding="utf-8"))


def main():
    checks = []
    rows = np.loadtxt(PACKET / "wine.csv", delimiter=",", skiprows=1)
    source = load_wine()
    assert np.array_equal(rows[:, 0], np.arange(178))
    assert np.array_equal(rows[:, 1], source.target)
    assert np.array_equal(rows[:, 2:], source.data)
    split = read_json("split.json")
    assert sorted(split["train"] + split["validation"] + split["test"]) == list(range(178))
    assert set(split["tiny"]) <= set(split["train"])
    counts = {role: np.bincount(rows[ids, 1].astype(int), minlength=3).tolist()
              for role, ids in split.items()}
    assert counts == {"train": [24, 24, 24], "validation": [12, 12, 12],
                      "test": [23, 35, 12], "tiny": [4, 4, 4]}
    checks.append("CSV matches bundled source exactly; all178 rows partitioned once; target/row-ID excluded; tiny subset and class counts correct")

    wine = read_json("wine-results.json")
    assert np.allclose(np.mean(rows[split["train"], 2:], axis=0), wine["training_mean"], atol=1e-12)
    assert np.allclose(np.std(rows[split["train"], 2:], axis=0), wine["training_scale"], atol=1e-12)
    assert len(wine["runs"]) == 8
    for run in wine["runs"]:
        expected_length = 121 if run["treatment"] in {"tiny", "omitted_update"} else 401
        assert len(run["trace"]) == expected_length
        assert [t["update"] for t in run["trace"]] == list(range(expected_length))
        for role, predictions in run["final_predictions"].items():
            expected_ids = run["training_row_ids"] if role == "training" else split["validation"]
            assert [item["row_id"] for item in predictions] == expected_ids
            assert not (set(expected_ids) & set(split["test"]))
            probabilities = np.array([item["probabilities"] for item in predictions])
            targets = np.array([item["supplied_target"] for item in predictions])
            assert np.allclose(probabilities.sum(axis=1), 1)
            assert np.array_equal(probabilities.argmax(axis=1), [item["predicted_class"] for item in predictions])
            score = run["trace"][-1]["train_supplied" if role == "training" else "validation"]
            recomputed_loss = -np.log(probabilities[np.arange(len(targets)), targets]).mean()
            assert math.isclose(recomputed_loss, score["loss"], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(np.mean(probabilities.argmax(axis=1) == targets), score["accuracy"], abs_tol=1e-14)
    clean = {r["seed"]: r for r in wine["runs"] if r["treatment"] == "clean"}
    shuffled = {r["seed"]: r for r in wine["runs"] if r["treatment"] == "shuffled_labels"}
    assert list(clean) == [3, 7, 19]
    assert [round(shuffled[s]["trace"][-1]["validation"]["accuracy"] * 36) for s in clean] == [11, 15, 14]
    for seed in clean:
        assert clean[seed]["trace"][0]["validation"] == shuffled[seed]["trace"][0]["validation"]
        assert clean[seed]["trace"][-1]["train_supplied"]["accuracy"] == 1
        assert shuffled[seed]["trace"][-1]["train_supplied"]["accuracy"] == 1
        assert clean[seed]["trace"][-1]["validation"]["accuracy"] == 1
    omitted = next(r for r in wine["runs"] if r["treatment"] == "omitted_update")
    assert all(t["train_supplied"] == omitted["trace"][0]["train_supplied"] for t in omitted["trace"])
    assert all(t["next_update_gradient_norm"] > 0 and t["next_update_parameter_change_norm"] == 0 for t in omitted["trace"][:-1])
    assert round(shuffled[7]["trace"][-1]["train_original"]["accuracy"] * 72) == 22
    accuracies = [shuffled[s]["trace"][-1]["validation"]["accuracy"] for s in clean]
    assert round(float(np.mean(accuracies)), 6) == 0.370370
    assert round(float(np.std(accuracies, ddof=1)), 6) == 0.057824
    checks.append("All8 measured runs retained; final CE/accuracy independently recomputed from probabilities; fixed-update/same-initial/null/omitted-step contrasts and reported repetition statistics checked; no test evaluation")

    calculation = read_json("calculation-results.json")
    assert calculation["finite_difference"]["loss"] == 0.3125
    assert calculation["finite_difference"]["autograd"] == 0.75
    for key, inputs, old_mean, old_variance in [
        ("mode_probe", [1, 3], 0, 1),
        ("mode_fresh_input", [-2, 6], 1, 4),
        ("mode_matching_buffers_null", [0, 1], 0.5, 0.5),
    ]:
        for result in calculation[key]:
            mean = np.mean(inputs) if result["training"] else old_mean
            variance = np.var(inputs) if result["training"] else old_variance
            expected = (np.array(inputs) - mean) / math.sqrt(variance + 1e-5)
            assert np.allclose(result["output"], expected, rtol=0, atol=1e-14)
            assert result["output_requires_grad"] == result["grad_enabled"]
    checks.append("Independent NumPy formulas agree with actual Torch four-mode BatchNorm probes on default/fresh/matching-statistic fixtures; Fraction scalar and paired-practice results retained")

    replay = read_json("checkpoint-results.json")
    comparisons = {r["omission"]: r for r in replay["comparisons"]}
    reference = replay["uninterrupted_continuation"]
    assert replay["checkpoint_after_update"] == 5 and replay["checkpoint_cursor"] == 60
    assert comparisons["none"]["trace"] == reference
    assert comparisons["none"]["exact_parameters_equal"] and comparisons["none"]["maximum_parameter_difference"] == 0
    for name in ["optimizer_buffers", "torch_rng", "active_order_cursor", "scheduler"]:
        assert comparisons[name]["maximum_parameter_difference"] > 0
        assert not comparisons[name]["exact_parameters_equal"]
    assert comparisons["optimizer_buffers"]["trace"][0] == reference[0]
    assert comparisons["optimizer_buffers"]["trace"][1]["pre_update_loss"] != reference[1]["pre_update_loss"]
    assert comparisons["torch_rng"]["trace"][0]["pre_update_loss"] != reference[0]["pre_update_loss"]
    assert comparisons["active_order_cursor"]["trace"][0]["training_positions"] != reference[0]["training_positions"]
    assert comparisons["scheduler"]["trace"][0] == reference[0]
    assert comparisons["scheduler"]["trace"][1]["pre_update_loss"] == reference[1]["pre_update_loss"]
    assert comparisons["scheduler"]["trace"][1]["learning_rate_used"] == 0.015
    assert reference[1]["learning_rate_used"] == 0.0075
    checks.append("Actual replay matches all saved state exactly; all4 omission outcomes and first-observable-divergence statements match retained traces")

    manuscript = (PACKET / "lesson.md").read_text(encoding="utf-8")
    program = re.findall(r"```python\n(.*?)\n```", manuscript, flags=re.S)[0]
    executed = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True, check=True)
    displayed_output = re.findall(r"```text\n(.*?)\n```", manuscript, flags=re.S)[0] + "\n"
    assert executed.stdout == displayed_output
    for script in PACKET.glob("*.py"):
        ast.parse(script.read_text(encoding="utf-8"))
    checks.append("Exact complete finite-difference program extracted from manuscript executed and matched displayed output; all packet Python sources parse")

    missing_links = []
    for document in PACKET.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("https:", "http:", "#", "/")):
                continue
            if not (document.parent / target.split("#")[0]).exists():
                missing_links.append({"file": document.name, "target": target})
    assert not missing_links, missing_links
    checks.append("All local Markdown links resolve within the available packet and current prepared-topic/scope files")
    sources = sorted(p for p in PACKET.iterdir() if p.is_file() and p.name not in {"author-verification.json", "design.md"})
    result = {"status": "passed", "date": "2026-09-13", "command": "scratch/lesson-tools/Scripts/python.exe -B docs/teaching/drafts/neural-training-diagnostics-reproducible-experiments/verify_packet.py",
              "checks": checks, "displayed_program_output": executed.stdout,
              "class_counts": counts,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "not_performed": ["website implementation", "formal independent review", "browser/figure/accessibility review", "application build", "publication", "GPU/distributed reproduction"]}
    (PACKET / "author-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "check_groups": len(checks), "hash_bound_files": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
