"""Execute the AutoML & NAS lesson's displayed Python programs.

Every displayed program is extracted **verbatim from the frozen manuscript's own
fenced code blocks**, pinned by the SHA-256 of the manuscript and of each block,
rather than transcribed. A changed manuscript therefore fails here instead of
silently diverging from the page.

`--write` regenerates src/learn/data/automl-examples.js from what was extracted
and executed; without it, the recorded module must match a fresh run.

Three programs are displayed:

  * `banknote_search.py` is the complete section 5 study. It is executed against
    the comma-separated file this lesson serves, in a temporary directory, with
    one BLAS thread. 35 estimator fits; nothing is downloaded.
  * `banknote_flaml.py` and `banknote_keras_search.py` are the optional section 8
    translations. FLAML and TensorFlow/KerasTuner are deliberately **not**
    installed into the shared lesson runtime, because their dependency
    resolution can move NumPy or scikit-learn and invalidate other topics'
    recorded outputs. They are therefore parsed, structurally checked, and their
    one substantive data claim - that each uses only the first development fold
    and never touches an inspection or reserved row - is executed for real, by
    importing the extracted `banknote_search.py` as a module and resolving the
    fold they index. Their status is recorded as not executed, and the lesson
    displays no output for them.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-automl-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-automl-examples.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/automl-neural-architecture-search-nas"
MANUSCRIPT = PACKET / "lesson.md"
ASSET = ROOT / "public/learn-assets/automl-nas/banknote-data.csv"
MODULE = ROOT / "src/learn/data/automl-examples.js"
DATA_MODULE = ROOT / "src/learn/data/automl-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/automl-native.json"

MANUSCRIPT_SHA = "7604d8ca4947c0271a11358df4272715b8d9bc07bc299b324a88e86ab32ab0d4"
NL = chr(10)
FENCE = re.compile(r"^```([a-z]*)" + NL + r"(.*?)^```$", re.S | re.M)

#: Manuscript fence index -> what that block is. Every fence is accounted for, so
#: a block added to or removed from the manuscript fails this mapping.
BLOCKS = {
    0: ("searchSpaceGrammar", "text"),
    1: ("banknoteSearch.setup", "bash"),
    2: ("banknoteSearch.code", "python"),
    3: ("banknoteFlaml.setup", "bash"),
    4: ("banknoteFlaml.code", "python"),
    5: ("banknoteKerasSearch.setup", "bash"),
    6: ("banknoteKerasSearch.code", "python"),
}

TITLES = {
    "banknoteSearch": (
        "The complete bounded study: eleven configurations, three folds, one declared comparison",
        "eleven declared configurations, three group-respecting folds and two final refits. Which "
        "configuration wins the declared criterion, and does the deeper network beat the wider one?",
    ),
    "banknoteFlaml": (
        "FLAML: make the validation method explicit",
        "one library's search over two estimator families, told exactly which rows are its holdout.",
    ),
    "banknoteKerasSearch": (
        "KerasTuner: a conditional neural search space",
        "a second width that exists only when the depth is two, searched for four trials.",
    ),
}

failures: list[str] = []
oracles = 0


def expect(condition: bool, label: str) -> None:
    global oracles
    oracles += 1
    if not condition:
        failures.append(label)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def js_constant(name: str, source: str):
    """Read one exported JSON-shaped constant out of a generated module."""
    match = re.search(r"export const " + name + r" = (.*?);" + NL + NL, source, re.S)
    return json.loads(match.group(1))


def load_blocks() -> dict[str, str]:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    expect(digest(text) == MANUSCRIPT_SHA,
           f"the manuscript is the pinned revision ({digest(text)})")
    found = list(FENCE.finditer(text))
    expect(len(found) == len(BLOCKS), f"the manuscript still has {len(BLOCKS)} fenced blocks, not {len(found)}")
    blocks = {}
    for index, match in enumerate(found):
        name, language = BLOCKS[index]
        expect(match.group(1) == language, f"fence {index} is still {language}, not {match.group(1)!r}")
        blocks[name] = match.group(2)
    return blocks


def structural_checks(blocks: dict[str, str]) -> None:
    """Claims the manuscript makes about its own programs, checked on the source."""
    search = blocks["banknoteSearch.code"]
    tree = ast.parse(search)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    expect(functions == {"load_roles", "candidate_models", "main", "rows"},
           f"the study defines exactly its four named functions, not {sorted(functions)}")
    expect(search.count("clone(estimator)") == 2
           and "clone(estimator).fit(x[fitting], y[fitting])" in search
           and "clone(estimator).fit(x[development], y[development])" in search,
           "every fold fit and every refit starts from a fresh clone of the declared estimator")
    expect("make_pipeline(StandardScaler()" in search,
           "and every scaled candidate carries its scaler inside a pipeline")
    expect("random_state=71" in search and "random_state=72" in search and "random_state=73" in search,
           "the three split seeds are declared in the displayed source")
    expect("threadpool_limits(limits=1)" in search, "and the run is pinned to one BLAS thread")
    # The manuscript's claim that only the two declared models reach inspection.
    inspection_uses = re.findall(r"x\[inspection\]|y\[inspection\]", search)
    expect(inspection_uses == ["x[inspection]", "y[inspection]", "y[inspection]"],
           f"inspection rows are read once to predict and twice to score, not {inspection_uses}")
    expect(len(re.findall(r"model\.predict\(x\[inspection\]\)", search)) == 1,
           "and exactly one prediction call reaches them, inside the two-model loop")
    expect(not re.search(r"\[reserve\]", search), "and no reserved row is ever indexed")
    expect(search.count("for role, index in") == 1 and '("declared_baseline", 3)' in search,
           "the predeclared baseline is registry entry 3, fixed before the search runs")

    # The two optional programs: no inspection or reserved row is ever loaded.
    for key in ("banknoteFlaml", "banknoteKerasSearch"):
        code = blocks[f"{key}.code"]
        optional = ast.parse(code)
        loaded = {node.id for node in ast.walk(optional)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        expect("inspection" not in loaded and "reserve" not in loaded,
               f"{key} never reads the inspection or reserved rows it unpacked")
        expect("folds[0]" in code, f"{key} uses the first development fold")
        expect("from banknote_search import load_roles" in code,
               f"{key} reuses the study's declared roles rather than resplitting")
    expect("eval_method=\"holdout\"" in blocks["banknoteFlaml.code"]
           and "seed=76" in blocks["banknoteFlaml.code"],
           "the FLAML example fixes both its resampling method and its seed")
    expect("set_random_seed(77)" in blocks["banknoteKerasSearch.code"]
           and "seed=77" in blocks["banknoteKerasSearch.code"],
           "and the KerasTuner example fixes both of its seeds")
    expect("conditional_scope(\"depth\", [2])" in blocks["banknoteKerasSearch.code"]
           and "if depth == 2:" in blocks["banknoteKerasSearch.code"],
           "whose second width is registered conditionally and added only at depth two")
    # The setup commands install exactly what their programs import.
    expect("numpy scikit-learn threadpoolctl" in blocks["banknoteSearch.setup"],
           "the study's setup command installs what it imports")
    expect('"flaml[automl]"' in blocks["banknoteFlaml.setup"], "the FLAML extras expression stays quoted")
    expect("tensorflow keras keras-tuner" in blocks["banknoteKerasSearch.setup"],
           "and the Keras setup names all three packages")


def run_study(code: str) -> tuple[str, dict]:
    """Execute the displayed study beside a copy of the served dataset."""
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        shutil.copy(ASSET, workspace / "banknote-data.csv")
        script = workspace / "banknote_search.py"
        script.write_text(code, encoding="utf-8", newline=NL)
        sys.path.insert(0, str(workspace))
        try:
            module = importlib.import_module("banknote_search")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                module.main()
            # The optional programs' one executable claim, resolved for real.
            x, y, development, inspection, reserve, folds = module.load_roles()
            fitting, validation = folds[0]
            protected = set(inspection.tolist()) | set(reserve.tolist())
            pool = set(development.tolist())
            expect(not (set(fitting.tolist()) & protected) and not (set(validation.tolist()) & protected),
                   "the first fold the optional programs use touches no inspection or reserved row")
            expect(set(fitting.tolist()) <= pool and set(validation.tolist()) <= pool,
                   "and lies entirely inside the development pool")
            boundary = {
                "firstFoldFittingRows": int(len(fitting)),
                "firstFoldValidationRows": int(len(validation)),
                "protectedRowsTouched": 0,
            }
        finally:
            sys.path.remove(str(workspace))
            sys.modules.pop("banknote_search", None)
    return captured.getvalue(), boundary


def check_output(printed: str) -> None:
    """Every printed number, against the separately generated data module."""
    source = DATA_MODULE.read_text(encoding="utf-8")
    candidates = js_constant("candidates", source)
    inspection = js_constant("inspection", source)
    roles = js_constant("roles", source)

    lines = printed.strip().split(NL)
    expect(lines[0] == "Role rows: " + " ".join(str(role["rows"]) for role in roles),
           f"the printed role sizes, {lines[0]!r}")
    for position, candidate in enumerate(candidates):
        line = lines[1 + position]
        expect(line.startswith(candidate["id"] + " folds "), f"candidate line {position} names {candidate['id']}")
        numbers = [float(value) for value in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", line.split("folds")[1])]
        expect(len(numbers) == 4, f"candidate line {position} prints three folds and a mean")
        for fold, value in enumerate(numbers[:3]):
            expect(abs(value - round(candidate["foldAccuracy"][fold], 6)) < 5e-7,
                   f"{candidate['id']} fold {fold + 1} prints {value}")
        expect(abs(numbers[3] - round(candidate["meanFoldAccuracy"], 6)) < 5e-7,
               f"{candidate['id']} mean prints {numbers[3]}")
    tail = NL.join(lines[1 + len(candidates):])
    for entry in inspection:
        expect(f"{entry['role']} {entry['id']} {entry['accuracy']}" in tail,
               f"the {entry['role']} line prints {entry['id']} at {entry['accuracy']}")
        matrix = entry["confusion"]
        width = max(len(str(value)) for row in matrix for value in row)
        rendered = NL.join(
            ("[[" if index == 0 else " [")
            + " ".join(str(value).rjust(width) for value in row)
            + ("]]" if index == len(matrix) - 1 else "]")
            for index, row in enumerate(matrix))
        expect(rendered in tail, f"and its confusion matrix {matrix} is printed as NumPy renders it")
    expect("nan" not in printed.lower() and "Warning" not in printed,
           "the run printed no warning and no missing value")


def main() -> None:
    write = "--write" in sys.argv
    blocks = load_blocks()
    structural_checks(blocks)
    printed, boundary = run_study(blocks["banknoteSearch.code"])
    check_output(printed)

    records = {
        "banknoteSearch": {
            "title": TITLES["banknoteSearch"][0],
            "question": TITLES["banknoteSearch"][1],
            "setup": blocks["banknoteSearch.setup"].rstrip(NL),
            "code": blocks["banknoteSearch.code"].rstrip(NL),
            "expected": printed.rstrip(NL),
            "language": "python",
            "file": "banknote_search.py",
            "executed": True,
        },
    }
    for key in ("banknoteFlaml", "banknoteKerasSearch"):
        records[key] = {
            "title": TITLES[key][0],
            "question": TITLES[key][1],
            "setup": blocks[f"{key}.setup"].rstrip(NL),
            "code": blocks[f"{key}.code"].rstrip(NL),
            "language": "python",
            "file": {"banknoteFlaml": "banknote_flaml.py",
                     "banknoteKerasSearch": "banknote_keras_search.py"}[key],
            "executed": False,
            "note": ("Optional. Parsed and structurally checked, and its use of the first development fold "
                     "was resolved against the study's own roles; its own library was not installed in this "
                     "project's lesson runtime, so no output is recorded for it."),
        }
    records["searchSpaceGrammar"] = {
        "title": "The declared search space, as a grammar",
        "code": blocks["searchSpaceGrammar"].rstrip(NL),
        "language": "text",
        "executed": False,
    }

    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        raise SystemExit(f"{len(failures)} of {oracles} oracle assertions failed.")

    text = (
        "// Complete displayed programs for the AutoML & NAS lesson, extracted verbatim" + NL
        + "// from the frozen manuscript's fenced code blocks and executed by" + NL
        + "// scripts/verify-automl-examples.py. `file` is the filename the lesson asks the" + NL
        + "// learner to save the block as; `setup` is its displayed install command." + NL
        + "// `executed: false` marks a block this project's lesson runtime cannot run, and" + NL
        + "// such a block carries no recorded output. Do not edit by hand." + NL
        + "export const automlExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";" + NL
    )
    if write:
        MODULE.write_text(text, encoding="utf-8", newline=NL)
    elif MODULE.read_text(encoding="utf-8") != text:
        raise SystemExit("src/learn/data/automl-examples.js is stale; rerun with --write.")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
        "source": "src/learn/data/automl-examples.js",
        "sourceHash": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "verifier": "scripts/verify-automl-examples.py",
        "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manuscript": "docs/teaching/drafts/automl-neural-architecture-search-nas/lesson.md",
        "manuscriptHash": MANUSCRIPT_SHA,
        "extraction": "verbatim from the manuscript's fenced code blocks, each pinned by SHA-256",
        "versions": {name: importlib.metadata.version(name)
                     for name in ["numpy", "scikit-learn", "scipy", "threadpoolctl"]},
        "blocks": {name: {"fence": index, "language": language,
                          "sha256": digest(blocks[name])}
                   for index, (name, language) in BLOCKS.items()},
        "programs": {key: {"file": record.get("file"), "executed": record["executed"],
                           "codeHash": digest(record["code"]),
                           "stdoutHash": digest(record["expected"]) if record.get("expected") else None}
                     for key, record in records.items()},
        "estimatorFits": 35,
        "firstFoldBoundary": boundary,
        "oracles": oracles,
        "notInstalled": ["flaml", "tensorflow", "keras", "keras-tuner"],
        "notInstalledReason": (
            "Installing them into the shared lesson runtime can move NumPy or scikit-learn and invalidate "
            "other completed topics' recorded outputs. The optional programs are therefore displayed without "
            "output, exactly as the manuscript presents them."
        ),
        "limits": [
            "The study ran against this lesson's served copy of the CSV; no network access was used.",
            "One fixed row-level protocol on one small public collection; the scores are development results.",
            "No reserved row was predicted or scored by any program here.",
            "Numerical fitting can differ on other library versions.",
        ],
        "passed": True,
    }, indent=2) + NL, encoding="utf-8", newline=NL)
    executed = sum(1 for record in records.values() if record["executed"])
    print(f"PASS: {executed} of {len(records)} displayed blocks executed, {oracles} oracle assertions, "
          f"35 estimator fits; module {'written' if write else 'current'}.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
