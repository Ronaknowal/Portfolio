"""Execute the exact downloadable active-learning program and compare every output leaf."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import numpy
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/active-learning"
DOWNLOAD = ROOT / "public/learn/downloads/active-learning"
expected = json.loads((PACKET / "checked-results.json").read_text(encoding="utf-8"))
numeric_leaves = 0


def compare(actual, reference, path="root"):
    global numeric_leaves
    if isinstance(reference, dict):
        assert actual.keys() == reference.keys(), path
        for key in reference:
            compare(actual[key], reference[key], f"{path}.{key}")
    elif isinstance(reference, list):
        assert len(actual) == len(reference), path
        for index, (left, right) in enumerate(zip(actual, reference)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(reference, (int, float)):
        numeric_leaves += 1
        assert abs(actual - reference) <= 1e-10 * max(1, abs(reference)), (path, actual, reference)
    else:
        assert actual == reference, (path, actual, reference)


for name in ("banknote-subset.csv", "banknote-active-learning.py", "data-provenance.md"):
    assert (DOWNLOAD / name).read_bytes() == (PACKET / name).read_bytes(), name
scratch = ROOT / "scratch/active-learning-native"
scratch.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=scratch) as temporary:
    work = Path(temporary)
    for name in ("banknote-subset.csv", "banknote-active-learning.py"):
        shutil.copyfile(DOWNLOAD / name, work / name)
    program = work / "banknote-active-learning.py"
    executed = subprocess.run([sys.executable, str(program)], text=True, encoding="utf-8", capture_output=True, check=True)
    actual = json.loads((work / "active-learning-results.json").read_text(encoding="utf-8"))
    compare(actual, expected["banknotes"])
    practice = subprocess.run([sys.executable, str(program), "--budget", "15", "--strategies", "random", "entropy", "--development-only"], text=True, encoding="utf-8", capture_output=True, check=True)
    practice_result = json.loads((work / "active-learning-results.json").read_text(encoding="utf-8"))
    assert practice_result["final_test"] == []
    for traces in practice_result["traces"].values():
        for trace in traces:
            assert len(trace["development_correct"]) == 16
            assert len({query["pool_row"] for query in trace["queries"]}) == 15
    compare(practice_result, expected["practice_development_only"])
    spec = importlib.util.spec_from_file_location("active_author", PACKET / "author-calculations.py")
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    derived = author.small_examples()
    for key, value in derived.items():
        compare(value, expected["examples"][key], "examples." + key)

record = {
    "status": "passed",
    "python": platform.python_version(),
    "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
    "numericLeavesCompared": numeric_leaves,
    "protocol": "Exact downloadable default program plus changed-budget development-only CLI; entire nested outputs compared, with independent small author arithmetic reexecuted.",
    "defaultStdout": executed.stdout,
    "practiceStdout": practice.stdout,
    "sourceHashes": {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in [DOWNLOAD / "banknote-active-learning.py", DOWNLOAD / "banknote-subset.csv", ROOT / "src/learn/data/active-learning-experiment.json"]},
    "limitations": "Reproduction checks numerical identity under this recorded environment; it is not an independent population performance study or a universal acquisition ranking.",
}
target = ROOT / "docs/teaching/evidence/active-learning-native.json"
target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(f"PASS: exact native default + changed-budget programs, {numeric_leaves} numeric leaves compared; evidence {target.relative_to(ROOT)}")
