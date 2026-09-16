"""Execute the imbalanced-learning lesson's displayed Python programs.

Every program is extracted **verbatim from the frozen manuscript's own code
fences**, each pinned by the SHA-256 of its fence body, so the page cannot drift
from the manuscript and a transcription cannot silently introduce a difference.
The extracted files are written into a scratch directory beside this lesson's
served copy of `yeast.data` and run as real separate processes, because the
manuscript tells a learner to save three files that import one another.

The third program uses `imbalanced-learn`, which the content phase deliberately
did not install or execute. Phase two installs it (0.14.2) and runs it, so the
page shows real output rather than none.

`--write` regenerates src/learn/data/imbalance-examples.js from the executed
programs; without it the recorded output must match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-imbalance-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-imbalance-examples.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs/teaching/drafts/imbalanced-learning-smote-cost-sensitive-learning/lesson.md"
PACKET_INPUTS = ROOT / "docs/teaching/drafts/imbalanced-learning-smote-cost-sensitive-learning/calculated-inputs.json"
DATASET = ROOT / "public/learn-assets/imbalanced-learning/yeast.data"
MODULE = ROOT / "src/learn/data/imbalance-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/imbalance-native.json"
WORKSPACE = ROOT / "scratch/imbalance-programs"

# Fence order in the manuscript, with the filename each block is told to be
# saved as and the SHA-256 of the fence body exactly as it stands there. A
# changed manuscript must be re-pinned deliberately, never absorbed silently.
PROGRAMS = [
    {
        "key": "models",
        "file": "imbalance_models.py",
        "sha256": "770e9d29627f94f717690d626a319b27f23dc60694950e64e134eff3445a2199",
        "title": "The teaching functions: a weighted logistic fit and one interpolator",
        "question": "a normalised weighted objective, an explicit minority distance matrix and one scalar fraction per generated vector. What does the printed midpoint have to be, and do the three seeded samples land on segments between the three given points?",
        "runs": True,
    },
    {
        "key": "study",
        "file": "yeast_imbalance.py",
        "sha256": "fd0bd8436c15b0bca8d6d15bf06efe4acd70eb0e4a1dd2063a5fd6c9cabdaa90",
        "title": "The complete five-procedure study on the observed proteins",
        "question": "five declared procedures, one protected inspection partition and a threshold chosen on a separate tuning partition. Which procedure reaches the lowest realised cost, and is it the one with the highest average precision?",
        "runs": True,
    },
    {
        "key": "pipeline",
        "file": "yeast_smote_cv.py",
        "sha256": "675b409c4316b4d798a6d8b31bb85fcc87ba68ec62aa4846e6593b1b8297a7fa",
        "title": "Optional: the same training-only resampling through imbalanced-learn",
        "question": "a three-fold development cross-validation in which the scaler and the sampler are refitted inside every fold. Do its fold scores reproduce the five-model table, and should they?",
        "runs": True,
    },
]

write = "--write" in sys.argv
oracle_count = 0
failures: list[str] = []


def oracle(condition, label):
    global oracle_count
    oracle_count += 1
    if not condition:
        failures.append(label)


NUMBER = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def floats(text):
    """Every number in a line of NumPy output, including `1.` and `0.`."""
    return [float(token) for token in re.findall(NUMBER, text)]


def extract():
    text = MANUSCRIPT.read_text(encoding="utf-8")
    fences = re.findall(r"^```python\n(.*?)^```$", text, re.S | re.M)
    assert len(fences) == len(PROGRAMS), f"expected {len(PROGRAMS)} python fences, found {len(fences)}"
    for program, body in zip(PROGRAMS, fences):
        found = digest(body)
        assert found == program["sha256"], (
            f"{program['file']}: the manuscript fence now hashes {found}, not the pinned "
            f"{program['sha256']}. Re-pin deliberately after reading the change.")
        program["code"] = body.rstrip("\n")
    return fences


def run():
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copyfile(DATASET, WORKSPACE / "yeast.data")
    for program in PROGRAMS:
        (WORKSPACE / program["file"]).write_text(program["code"] + "\n", encoding="utf-8", newline="\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    for program in PROGRAMS:
        if not program["runs"]:
            continue
        finished = subprocess.run(
            [sys.executable, program["file"]], cwd=WORKSPACE, env=environment,
            capture_output=True, text=True, timeout=1800)
        assert finished.returncode == 0, f"{program['file']} failed:\n{finished.stderr}"
        program["expected"] = finished.stdout.replace("\r\n", "\n").strip()
        print(f"Executed {program['file']}: {len(program['code'].splitlines())} lines, "
              f"{len(program['expected'].splitlines())} output lines")


extract()
run()
recorded = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))

# ------------------------------------------------------------------- oracles

models_out = next(p for p in PROGRAMS if p["key"] == "models")["expected"].splitlines()
oracle(len(models_out) >= 2, "the helper program prints a midpoint and its seeded samples")
midpoint = floats(models_out[0])
oracle(midpoint == [1.0, 0.0], "the declared midpoint of (0,0) and (2,0) prints as (1, 0)")
oracle(midpoint == recorded["constructed"]["synthetic"], "and it equals the packet's recorded synthetic point")
sample_text = "\n".join(models_out[1:])
sample = floats(sample_text)
oracle(len(sample) == 6, "three seeded synthetic vectors of two coordinates each")
corners = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]


def on_segment(point):
    """Every generated vector must lie on a segment between two of the inputs."""
    for index, start in enumerate(corners):
        for end in corners[index + 1:]:
            span = (end[0] - start[0], end[1] - start[1])
            length = span[0] ** 2 + span[1] ** 2
            u = ((point[0] - start[0]) * span[0] + (point[1] - start[1]) * span[1]) / length
            if -1e-9 <= u <= 1 + 1e-9:
                projected = (start[0] + u * span[0], start[1] + u * span[1])
                if math.dist(point, projected) < 1e-9:
                    return True
    return False


generated = [(sample[index], sample[index + 1]) for index in range(0, 6, 2)]
for point in generated:
    oracle(on_segment(point), f"the generated vector {point} lies on a segment between two minority points")
hull = all(0 <= x <= 2 and 0 <= y <= 2 and x + y <= 2 + 1e-9 for x, y in generated)
oracle(hull, "every generated vector lies inside the triangle its three inputs span")
oracle(len({point for point in generated}) == 3, "the three seeded draws are distinct")

study_out = next(p for p in PROGRAMS if p["key"] == "study")["expected"].splitlines()
oracle(len(study_out) == 20, "the study prints four lines for each of the five procedures")
blocks = {}
for index in range(0, 20, 4):
    header = study_out[index].split()
    name = header[0]
    blocks[name] = {
        "rows": int(header[3]),
        "threshold": float(header[5]),
        "default": [int(token) for token in re.findall(r"-?\d+", study_out[index + 1].split("(")[1])[:4]],
        "defaultCost": int(re.findall(r"-?\d+", study_out[index + 1].split("(")[1])[4]),
        "tuned": [int(token) for token in re.findall(r"-?\d+", study_out[index + 2].split("(")[1])[:4]],
        "tunedCost": int(re.findall(r"-?\d+", study_out[index + 2].split("(")[1])[4]),
        "metrics": floats(study_out[index + 3].split("AP")[1]),
    }
oracle(list(blocks) == ["original", "balanced_weight", "random_over", "random_under", "smote"],
       "the five procedures print in their declared order")
oracle([blocks[name]["rows"] for name in blocks] == [600, 600, 1158, 42, 1158],
       "the printed training-row counts are 600, 600, 1158, 42 and 1158")
for record in recorded["methods"]:
    block = blocks[record["name"]]
    oracle(abs(block["threshold"] - record["chosen_threshold"]) < 1e-12,
           f"{record['name']} prints its recorded selected threshold")
    oracle(block["default"] == [record["inspection_default"][key] for key in ("tp", "fp", "fn", "tn")],
           f"{record['name']} prints its recorded inspection counts at 0.5")
    oracle(block["tuned"] == [record["inspection_tuned"][key] for key in ("tp", "fp", "fn", "tn")],
           f"{record['name']} prints its recorded inspection counts at the selected threshold")
    oracle(block["defaultCost"] == record["inspection_default"]["cost"]
           and block["tunedCost"] == record["inspection_tuned"]["cost"],
           f"{record['name']} prints both recorded costs")
    oracle(sum(block["tuned"]) == 200 and block["tuned"][0] + block["tuned"][2] == 7,
           f"{record['name']} inspection counts total 200 with seven positives")
    oracle(block["defaultCost"] == record["inspection_default"]["fp"] + 12 * record["inspection_default"]["fn"],
           f"{record['name']} cost is FP + 12 FN, recomputed from the printed counts")
    oracle(abs(block["metrics"][0] - record["average_precision"]) < 1e-12
           and abs(block["metrics"][1] - record["roc_auc"]) < 1e-12
           and abs(block["metrics"][2] - record["brier_score"]) < 1e-12,
           f"{record['name']} prints its recorded AP, ROC-AUC and Brier score")
# The manuscript's §7 table, read straight off this run rather than off the packet.
oracle(min(blocks, key=lambda name: blocks[name]["tunedCost"]) == "smote"
       and blocks["smote"]["tunedCost"] == 49,
       "SMOTE reaches the lowest realised tuned cost, 49")
oracle(max(blocks, key=lambda name: blocks[name]["metrics"][0]) == "random_over",
       "random oversampling reaches the highest average precision")
oracle(blocks["original"]["defaultCost"] == 85 and blocks["original"]["tunedCost"] == 72,
       "the original model's cost falls from 85 at 0.5 to 72 at its tuned threshold")
oracle(blocks["smote"]["defaultCost"] == 65 and blocks["smote"]["tunedCost"] == 49,
       "SMOTE's cost falls from 65 at 0.5 to 49 at its tuned threshold")
oracle(all(blocks[name]["tunedCost"] < 84 for name in blocks),
       "every tuned procedure beats the always-negative baseline's cost of 84")
oracle(blocks["original"]["metrics"][2] < blocks["balanced_weight"]["metrics"][2]
       and blocks["original"]["metrics"][2] < blocks["smote"]["metrics"][2],
       "the original scores have the smallest Brier value of the three the manuscript names")
oracle(abs(blocks["original"]["metrics"][2] - 0.033475) < 5e-7
       and abs(blocks["balanced_weight"]["metrics"][2] - 0.131518) < 5e-7
       and abs(blocks["smote"]["metrics"][2] - 0.121856) < 5e-7,
       "the three Brier values the manuscript quotes are the ones printed")

pipeline_out = next(p for p in PROGRAMS if p["key"] == "pipeline")["expected"].splitlines()
fold_scores = floats(pipeline_out[0].split("[")[1])
mean_fold = floats(pipeline_out[-1].split("AP")[1])[0]
oracle(len(fold_scores) == 3, "the optional program prints three fitting-partition fold scores")
oracle(all(0 <= value <= 1 for value in fold_scores), "every fold average precision lies in [0, 1]")
oracle(abs(sum(fold_scores) / 3 - mean_fold) < 1e-9, "the printed mean is the mean of the three printed folds")
oracle(not any(abs(value - record["average_precision"]) < 1e-9
               for value in fold_scores + [mean_fold]
               for record in recorded["methods"]),
       "no fold score coincides with a five-model inspection AP: this is a different protocol, "
       "not a reproduction of the table")

if failures:
    for label in failures:
        print(f"FAIL {label}")
    raise SystemExit(f"{len(failures)} of {oracle_count} example oracles failed")

records = {
    program["key"]: {
        "title": program["title"],
        "question": program["question"],
        "code": program["code"],
        "expected": program["expected"],
        "language": "python",
        "file": program["file"],
    } for program in PROGRAMS
}

if not write:
    existing = MODULE.read_text(encoding="utf-8")
    prefix = "export const imbalanceExamples = "
    stored = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert stored[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert stored[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{stored[key]['expected']}\n---\n{record['expected']}")
else:
    MODULE.write_text(
        "// Complete displayed programs for the imbalanced-learning lesson.\n"
        "// Extracted verbatim from the frozen manuscript's code fences, each pinned by\n"
        "// the SHA-256 of its fence body, then executed as real separate files beside\n"
        "// this lesson's served copy of yeast.data by\n"
        "// scripts/verify-imbalance-examples.py. `file` is the name the lesson asks a\n"
        "// learner to save the block as. Do not edit by hand.\n"
        "export const imbalanceExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n")

shutil.rmtree(WORKSPACE, ignore_errors=True)
EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE.write_text(json.dumps({
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native execution of the displayed programs; browser, independent and integration review are separate",
    "extraction": "verbatim from the manuscript's python code fences, pinned by SHA-256 of each fence body",
    "manuscript": str(MANUSCRIPT.relative_to(ROOT)).replace("\\", "/"),
    "manuscriptSha256": hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest(),
    "source": str(MODULE.relative_to(ROOT)).replace("\\", "/"),
    "sourceHash": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-imbalance-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name)
                 for name in ["numpy", "scipy", "scikit-learn", "imbalanced-learn"]},
    "programs": {program["key"]: {
        "file": program["file"],
        "fenceSha256": program["sha256"],
        "codeHash": digest(program["code"]),
        "stdoutHash": digest(program["expected"]),
        "stdout": program["expected"],
        "executedAsSeparateProcess": True,
    } for program in PROGRAMS},
    "oracles": oracle_count,
    "notes": [
        "The three files were written into a scratch directory beside this lesson's served copy of "
        "yeast.data and run as separate processes, because the manuscript tells a learner to save three "
        "files that import one another. No network access was used.",
        "The content phase deliberately left the optional imbalanced-learn program unexecuted; phase two "
        "installed imbalanced-learn 0.14.2 and ran it, so the page shows real output.",
        "The optional program is a separate three-fold development protocol with scikit-learn's C = 1 "
        "convention; its fold scores are asserted NOT to coincide with the five-model table.",
    ],
    "limits": [
        "One row-level split of one historical collection; the scores are model-development results.",
        "No reserved protein was predicted or scored by any program here.",
        "Numerical fitting and RNG stream conventions can differ on other library versions.",
    ],
}, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(PROGRAMS)} displayed programs extracted verbatim and executed, {oracle_count} oracle assertions.")
