"""Execute the feature-selection lesson's displayed Python programs.

The programs are not transcribed here. They are extracted verbatim from the
frozen manuscript's python code fences and pinned by SHA-256, so the code the
lesson displays, the code this script executes and the code the packet froze are
provably the same text.

`--write` regenerates src/learn/data/selection-examples.js from the executed
programs; without it, the recorded output must match a fresh execution.

The Wine program reads the copy of `wine.data` that the lesson serves, from
public/learn-assets/feature-selection, which is the same bytes a learner
receives. Nothing is downloaded.

The fourth program is the manuscript's optional `shap.TreeExplainer` comparison.
The content phase explicitly did not execute it. It is executed here when a
compatible `shap` is importable, and its version is recorded; when it is not
importable the script says so and skips it rather than inventing output.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-selection-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-selection-examples.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/feature-selection-importance-shap-permutation-mutual-info"
LESSON = PACKET / "lesson.md"
ASSETS = ROOT / "public/learn-assets/feature-selection"
WORKSPACE = ROOT / "scratch/selection-programs"
MODULE = ROOT / "src/learn/data/selection-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/selection-native.json"
NL = chr(10)

# The four manuscript programs, in the order their code fences appear, each
# pinned to the exact block the content phase froze.
PROGRAMS = [
    {
        "key": "informationFromCounts",
        "file": "information_from_counts.py",
        "title": "Mutual information from a table of counts",
        "question": "three count tables, one noisy copy, one perfect copy and one independent pair. Which of them reveals a whole bit, and which reveals nothing?",
        "sha256": "d1d399c9d0f33afcca46d42d5bb12e043f778c71a5a0dfddd84e1e9364d6627b",
        "executed": True,
    },
    {
        "key": "coalitionAttribution",
        "file": "coalition_attribution.py",
        "title": "Every coalition, cached once, then allocated",
        "question": "the same model and the same instance, explained against two different references. Will the prediction change, and will the attributions?",
        "sha256": "2cab1585bf4c3a6ce5cc7cbf18d8e923191b21d6766bf16f1be1d0c13815919e",
        "executed": True,
    },
    {
        "key": "wineStudy",
        "file": "wine_feature_study.py",
        "title": "The complete offline Wine study",
        "question": "three retained sizes are compared inside the fitting rows, then a separately declared four-field model is inspected. Will the largest size win, and will the two models differ on the inspection rows?",
        "sha256": "3cf2ae8bc89feb4bd18818bf44bc860c42c62ea70e03ff0574609da481a386cb",
        "executed": True,
    },
    {
        "key": "treeExplainerCheck",
        "file": "check_tree_explanation.py",
        "title": "Optional: compare a production tree explainer with the exhaustive game",
        "question": "an installed TreeExplainer, in interventional mode on the probability scale, against the exhaustive sixteen-coalition oracle. Will they agree to 1e-6?",
        "sha256": "aaaa8fcc3a68d0da93b9138135f09097cbf15a0902b77ee51bb07763417fe25f",
        "executed": True,
        "optional": True,
        "requires": "shap",
    },
]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_programs():
    text = LESSON.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) == len(PROGRAMS), (
        f"the manuscript has {len(blocks)} python blocks, this verifier knows {len(PROGRAMS)}")
    for program, block in zip(PROGRAMS, blocks):
        assert digest(block) == program["sha256"], (
            f"{program['file']}: the manuscript block changed (now {digest(block)}). "
            "The manuscript is frozen input; investigate before updating this pin.")
        program["code"] = block.rstrip(NL)
    return PROGRAMS


def main():
    write = "--write" in sys.argv
    programs = extract_programs()

    have_shap = importlib.util.find_spec("shap") is not None
    shap_version = importlib.metadata.version("shap") if have_shap else None

    # Run the programs exactly as the lesson asks a learner to: four files saved
    # beside the supplied wine.data, imported from one another by name.
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    for program in programs:
        (WORKSPACE / program["file"]).write_text(program["code"] + NL, encoding="utf-8", newline="\n")
    served = ASSETS / "wine.data"
    assert served.exists(), "the served dataset is missing; run verify-selection-data.py --write first"
    assert served.read_bytes() == (PACKET / "wine.data").read_bytes(), "the served copy is not the packet file"
    shutil.copyfile(served, WORKSPACE / "wine.data")

    records, namespaces, caught = {}, {}, []
    sys.path.insert(0, str(WORKSPACE))
    try:
        for program in programs:
            if program.get("requires") == "shap" and not have_shap:
                print(f"SKIPPED {program['key']}: {program['requires']} is not installed in this environment.")
                program["executed"] = False
                continue
            namespace = {"__file__": str(WORKSPACE / program["file"]), "__name__": "__main__"}
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with contextlib.chdir(WORKSPACE), threadpool_limits(limits=1), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(compile(program["code"] + NL, f"selection-example:{program['key']}", "exec"), namespace)
            caught.extend(f"{program['key']}: {entry.message}" for entry in observed)
            namespaces[program["key"]] = namespace
            records[program["key"]] = {
                "title": program["title"],
                "question": program["question"],
                "file": program["file"],
                "code": program["code"],
                "expected": output.getvalue().strip(),
                "language": "python",
            }
            print(f"Executed {program['key']}: {len(program['code'].splitlines())} lines, "
                  f"{len(records[program['key']]['expected'].splitlines())} output lines")
    finally:
        sys.path.remove(str(WORKSPACE))

    # ------------------------------------------------------------- oracles
    oracle_count = 0

    def oracle(condition, label):
        nonlocal oracle_count
        assert condition, label
        oracle_count += 1

    counts = namespaces["informationFromCounts"]
    lines = records["informationFromCounts"]["expected"].splitlines()
    oracle(lines == ["MI = 0.188722 bits", "MI = 1.000000 bits", "MI = 0.000000 bits"],
           "the three printed mutual-information totals")
    cells, total = counts["information_bits"]([[3, 1], [1, 3]])
    oracle(abs(total - 0.18872187554086717) < 1e-15, "the 3/1/1/3 table gives 0.188722 bits")
    oracle(abs(counts["information_bits"]([[4, 0], [0, 4]])[1] - 1.0) < 1e-15, "the 4/0/0/4 table gives one bit")
    oracle(counts["information_bits"]([[2, 2], [2, 2]])[1] == 0.0, "the 2/2/2/2 table gives exactly zero")
    oracle(abs(counts["information_bits"]([[2, 0], [0, 6]])[1] - 0.8112781244591328) < 1e-15,
           "practice 1's table gives 0.811278 bits")
    # Doubling every count leaves the empirical value untouched.
    oracle(counts["information_bits"]([[6, 2], [2, 6]])[1] == total, "doubling every count leaves MI unchanged")
    # A zero cell contributes zero rather than a logarithm of zero.
    oracle(cells.shape == (2, 2) and abs(cells.sum() - total) < 1e-15, "the cell contributions sum to the total")
    oracle(counts["information_bits"]([[4, 0], [4, 0]])[1] == 0.0, "a target column with no observations is not an error")
    # A negative count, an empty total, a one-dimensional list and a non-finite
    # entry are each refused rather than silently repaired.
    for bad in ([[-1, 1], [1, 1]], [[0, 0], [0, 0]], [1, 2], [[float("nan"), 1], [1, 1]]):
        try:
            counts["information_bits"](bad)
            raise AssertionError(f"invalid counts must be refused: {bad}")
        except ValueError:
            oracle(True, f"invalid counts refused: {bad}")
    # A legitimate non-square table is accepted: the guard checks shape, not squareness.
    oracle(abs(counts["information_bits"]([[2, 1, 1], [0, 1, 3]])[1] - 0.3443609377704336) < 1e-12,
           "a two-by-three table is computed rather than refused")

    coalition = namespaces["coalitionAttribution"]
    coalition_lines = records["coalitionAttribution"]["expected"].splitlines()
    oracle(coalition_lines == [
        "coalitions [ 0.  2.  3. 11.]",
        "baseline 0.0 contributions [5. 6.]",
        "reconstructed prediction 11.0",
        "coalitions [ 1.5  3.5  5.  11. ]",
        "baseline 1.5 contributions [4.  5.5]",
        "reconstructed prediction 11.0",
    ], "the printed coalition arrays, attributions and reconstructions")
    explain = coalition["explain_background"]

    def polynomial(rows):
        return rows[:, 0] + rows[:, 1] + rows[:, 0] * rows[:, 1]

    baseline, phi, values = explain(polynomial, [2, 3], [[0, 0]])
    oracle(np.allclose(values, [0, 2, 3, 11]) and np.allclose(phi, [5, 6]) and baseline == 0,
           "the zero reference gives coalitions 0, 2, 3, 11 and attributions 5 and 6")
    baseline, phi, values = explain(polynomial, [2, 3], [[0, 0], [1, 1]])
    oracle(np.allclose(values, [1.5, 3.5, 5, 11]) and np.allclose(phi, [4, 5.5]) and baseline == 1.5,
           "the two-row reference gives coalitions 1.5, 3.5, 5, 11 and attributions 4 and 5.5")
    oracle(abs(baseline + phi.sum() - values[-1]) < 1e-12, "and reconstructs the same prediction of eleven")
    # The mean prediction is not the prediction at the mean input.
    oracle(abs(polynomial(np.array([[0.5, 0.5]]))[0] - 1.25) < 1e-15, "f at the mean background input is 1.25, not 1.5")
    # Practice 4.
    _, practice_phi, practice_values = explain(
        lambda rows: rows[:, 0] + rows[:, 1] + 2 * rows[:, 0] * rows[:, 1], [1, 2], [[0, 0]])
    oracle(np.allclose(practice_values, [0, 1, 2, 7]) and np.allclose(practice_phi, [3, 4]),
           "practice 4 gives coalitions 0, 1, 2, 7 and attributions 3 and 4")
    # The dependent-feature comparison's replacement game.
    _, duplicate_phi, duplicate_values = explain(lambda rows: rows[:, 0], [1, 1], [[0, 0], [1, 1]])
    oracle(np.allclose(duplicate_values, [0.5, 1, 0.5, 1]) and np.allclose(duplicate_phi, [0.5, 0]),
           "the duplicate-input replacement game gives 0.5 and exactly zero")
    for bad in ((polynomial, [2, 3], []), (polynomial, [[2, 3]], [[0, 0]]), (polynomial, [2, 3], [[0, 0, 0]]),
                (polynomial, [2, float("nan")], [[0, 0]])):
        try:
            explain(*bad)
            raise AssertionError("invalid explanation input must be refused")
        except (ValueError, IndexError):
            oracle(True, "invalid explanation input refused")

    study = namespaces["wineStudy"]
    printed = records["wineStudy"]["expected"].splitlines()
    for k, score in [(3, "0.809269"), (6, "0.849673"), (13, "0.839572")]:
        oracle(f"retained size {k} mean CV accuracy {score}" in printed, f"size {k} prints mean CV accuracy {score}")
    oracle("selected names ['alcohol', 'total_phenols', 'flavanoids', 'color_intensity', 'od280_od315', 'proline']" in printed,
           "the selected six fields are printed by name")
    oracle("selected-model inspection accuracy 0.9473684210526315" in printed, "the selected model is correct on 36 of 38")
    oracle("four-field inspection accuracy 0.9473684210526315" in printed, "the four-field model is correct on 36 of 38")
    oracle("majority inspection accuracy 0.39473684210526316" in printed, "the training-majority baseline is 15 of 38")
    for name, mean, sd in [("alcohol", "0.221053", "0.048809"), ("malic_acid", "0.0", "0.0"),
                           ("flavanoids", "0.359211", "0.082076"), ("proline", "0.189474", "0.029539")]:
        oracle(f"{name} accuracy drop {mean} permutation SD {sd}" in printed,
               f"{name} prints accuracy drop {mean} and SD {sd}")
    oracle("explained source row 104 class-1 probability 0.0" in printed, "source row 104 has class-1 probability zero")
    oracle("background prediction 0.33 contributions [-0.45  0.    0.07  0.05]" in printed,
           "the background baseline 0.33 and the four contributions")
    oracle(printed[-1].startswith("reconstruction error"), "the reconstruction error is printed")
    oracle(abs(float(printed[-1].split()[-1])) < 1e-15, "and it is floating-point dust, about 1.1e-16")
    with contextlib.chdir(WORKSPACE), contextlib.redirect_stdout(io.StringIO()):
        model, features, fitting, inspection, names = study["run_study"]()
    oracle(len(fitting) == 100 and len(inspection) == 38, "100 fitting rows and 38 inspection rows")
    oracle(len(set(fitting.tolist()) & set(inspection.tolist())) == 0, "and they are disjoint")
    oracle(names == ["alcohol", "malic_acid", "flavanoids", "proline"], "the four predeclared field names")
    oracle(model.tree_.node_count == 9, "the predeclared tree has nine nodes")
    oracle(int(model.tree_.feature[0]) == 0 and abs(model.tree_.threshold[0] - 12.78000020980835) < 1e-15,
           "its root splits alcohol at 12.78000020980835")
    oracle(1 not in set(model.tree_.feature[model.tree_.children_left != -1].tolist()),
           "no split in this tree uses malic acid")
    oracle(int(inspection[0]) == 104, "the first inspection row is source row 104")

    if "treeExplainerCheck" in records:
        check = namespaces["treeExplainerCheck"]
        oracle(check["class_one"].values.shape == (12, 4), "the class-1 explanation is twelve rows by four features")
        oracle(np.allclose(check["class_one"].base_values + check["class_one"].values.sum(axis=1),
                           check["probabilities"], rtol=0, atol=1e-6),
               "every explained row reconstructs its own predicted probability")
        oracle(np.allclose(check["class_one"].values[0], check["exact_phi"], rtol=0, atol=1e-6),
               "the library's attributions match the exhaustive oracle for source row 104")
        oracle(np.allclose(check["class_one"].base_values[0], check["baseline"], rtol=0, atol=1e-6),
               "and so does its baseline")
        oracle(abs(float(check["baseline"]) - 0.33) < 1e-12, "that baseline is the recorded 0.33")

    # ------------------------------------------------------------- module
    if not write:
        assert MODULE.exists(), "the examples module has not been generated; run with --write"
        existing = MODULE.read_text(encoding="utf-8")
        prefix = "export const selectionExamples = "
        recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";").strip())
        for key, record in records.items():
            assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
            assert recorded[key]["expected"] == record["expected"], (
                f"{key}: output changed:{NL}{recorded[key]['expected']}{NL}---{NL}{record['expected']}")
        for key in recorded:
            assert key in records, f"{key}: the module records a program this run did not execute"
    else:
        MODULE.write_text(
            "// Complete displayed programs for the feature-selection lesson.\n"
            "//\n"
            "// Extracted verbatim from the frozen manuscript's python code fences and\n"
            "// executed by scripts/verify-selection-examples.py, which pins each block by\n"
            "// SHA-256. `file` is the filename the lesson asks the learner to save it as.\n"
            "// `expected` is the actual captured stdout of that execution.\n"
            "export const selectionExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
            encoding="utf-8", newline="\n")

    evidence = {
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native execution of the displayed programs; browser, model and data checks are separate",
        "mode": "write" if write else "read-only comparison",
        "source": str(MODULE.relative_to(ROOT)).replace("\\", "/"),
        "sourceHash": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "verifier": "scripts/verify-selection-examples.py",
        "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manuscript": str(LESSON.relative_to(ROOT)).replace("\\", "/"),
        "manuscriptHash": hashlib.sha256(LESSON.read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ["numpy", "scikit-learn"] + (["shap"] if have_shap else [])},
        "programs": {key: {"file": record["file"], "codeHash": digest(record["code"]),
                           "stdoutHash": digest(record["expected"]), "stdout": record["expected"]}
                     for key, record in records.items()},
        "optionalShapComparison": {
            "executed": "treeExplainerCheck" in records,
            "package": "shap",
            "version": shap_version,
            "mode": "feature_perturbation='interventional', model_output='probability'",
            "result": ("TreeExplainer agreed with the exhaustive sixteen-coalition oracle for source row 104 "
                       "to within 1e-6 in attributions and baseline, and every one of the twelve explained rows "
                       "reconstructed its own predicted class-1 probability to within 1e-6.")
            if "treeExplainerCheck" in records else "not executed: shap is not importable in this environment",
        },
        "warnings": caught,
        "oracles": oracle_count,
        "limits": [
            "The Wine program ran against the served copy of the supplied file; no network access was used.",
            "One row-level stratified split of one small collection; the inner scores are selection records.",
            "No reserved row was predicted or scored by any program here.",
            "Numerical estimates can differ on other library versions.",
            "A matching library explainer is numerical agreement, not evidence that the explanatory question is the right one.",
        ],
        "passed": True,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + NL, encoding="utf-8", newline="\n")
    print(f"PASS: {len(records)} displayed programs executed verbatim from the manuscript, "
          f"{oracle_count} oracle assertions"
          f"{'' if 'treeExplainerCheck' in records else ', shap comparison skipped'}.")


main()
