"""Execute the hidden Markov models lesson's displayed Python programs.

Nothing displayed on the page is transcribed. Every line comes out of a frozen
packet file or the manuscript's own fenced block, pinned by SHA-256, so a
changed source fails here instead of silently diverging from the page.

Two programs are displayed, and BOTH are executed:

  * `hmm-experiments.py` is the complete offline NumPy program. It is run in a
    temporary directory beside a copy of the extract this lesson serves, with
    one BLAS thread. Its oracle is unusually strong: the file it writes is
    compared leaf by leaf against the content packet's own
    `calculated-inputs.json`, so every one of the 13,243 numbers the packet
    shares with a fresh run must agree exactly. The page displays four verbatim
    excerpts of it - taken as exact source segments through Python's own AST,
    never retyped - plus the command and the output of the complete run, and
    serves the whole file for download.

  * `hmmlearn-examples.py` is the optional library program. The content packet
    marks it unexecuted because hmmlearn was unavailable during authoring, and
    the manuscript requires phase two to run it before presenting native
    output. It is NOT installed into the shared lesson runtime, because
    resolving it moves NumPy and other completed lessons depend on the exact
    versions there. Instead it runs in an isolated environment at
    `scratch/hmm-optional`, whose resolved versions are recorded here, and its
    printed numbers are checked against oracles derived from the model rather
    than from its own output.

The single most interesting oracle is the one the manuscript warns about: in
hmmlearn 0.3.3 the MAP decoder returns the SUM of the selected marginal
probabilities, not a log probability. That is now executed rather than asserted -
the returned value is matched against the sum of the four smoothed marginal
maxima, and separately shown to be positive, which no log probability of a
probability can be.

`--write` regenerates src/learn/data/hmm-examples.js from what was extracted and
executed; without it, the recorded module must match a fresh run.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-hmm-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-hmm-examples.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ast
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/hidden-markov-models-hmm"
MANUSCRIPT = PACKET / "lesson.md"
EXPERIMENTS = PACKET / "hmm-experiments.py"
OPTIONAL = PACKET / "hmmlearn-examples.py"
PACKET_INPUTS = PACKET / "calculated-inputs.json"
SEQUENCES = ROOT / "public/learn-assets/hmm/ewt-sequences.json"
MODULE = ROOT / "src/learn/data/hmm-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/hmm-native.json"
ISOLATED = ROOT / "scratch/hmm-optional/Scripts/python.exe"

MANUSCRIPT_SHA = "f811fd57331134727c91bef4386b13bcb8b4097a11024619d177415611cdd295"
EXPERIMENTS_SHA = "25e7e4016c5420b7aae33a21e869275564d6951bc22865fd173f566c1d7b2f77"
OPTIONAL_SHA = "991ccca13385eb983d1364fb736bc4a5ac2640a05ef375bab5dae257fb8f033e"

NL = chr(10)
#: The manuscript uses tilde fences, so a backtick pattern would silently find
#: nothing and this mapping would pass over an empty set.
FENCE = re.compile("^~~~([a-z]*)" + NL + "(.*?)^~~~$", re.S | re.M)
FENCES = {0: ("runCommand", "bash")}

#: Which functions of the complete program each displayed excerpt shows, in
#: reading order. Every function of the file is accounted for, so a function
#: added to or removed from the frozen program fails this mapping.
EXCERPTS = [
    ("inference", ["log_values", "infer"],
     "The trellis itself: one sum, one backward sum, one maximum",
     "Follow one forward sum, one backward sum, the posterior normalisation, and then the "
     "separate maximum-and-backpointer recurrence that does not share them."),
    ("scaling", ["forward_scaled"],
     "Filtering that keeps its scale factors instead of its logarithms",
     "Each row is normalised and its factor kept, so the evidence is the product of the factors "
     "and its logarithm is their sum. This is the exact alternative to log space."),
    ("learning", ["expected_counts", "normalize_counts", "em_step"],
     "Fractional events, one recording at a time",
     "Add fractional events within each recording and reset the initial event at every boundary. "
     "An unvisited state's row is unidentified, so it is retained rather than divided by zero."),
    ("tagging", ["real_tagging"],
     "The supervised fit on real sentences, and two decision rules from it",
     "Fit category counts on the training sentences, then compare the declared decoders on the "
     "development sentences. The same fitted counts serve both rules."),
]
OTHER_FUNCTIONS = ["sample_sequences", "fit_em", "mechanism_examples", "serializable", "main"]

failures: list[str] = []
oracles = 0


def expect(condition: bool, label: str) -> None:
    global oracles
    oracles += 1
    if not condition:
        failures.append(label)


def close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    expect(abs(actual - expected) <= tolerance * max(1.0, abs(expected)),
           f"{label} - {actual!r} versus {expected!r}")


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaf_paths(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


def load_manuscript_fences() -> dict[str, str]:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    found = list(FENCE.finditer(text))
    expect(len(found) == len(FENCES),
           f"the manuscript still has {len(FENCES)} fenced block(s), not {len(found)}")
    blocks = {}
    for index, match in enumerate(found):
        name, language = FENCES[index]
        expect(match.group(1) == language, f"fence {index} is still {language}, not {match.group(1)!r}")
        blocks[name] = match.group(2)
    return blocks


def extract_excerpts(source: str) -> dict[str, dict]:
    """Exact source segments of named functions, taken through Python's own AST.

    `ast.get_source_segment` returns the bytes the parser saw, so an excerpt is a
    slice of the frozen file rather than a retyping of it. Its start and end line
    numbers are recorded so a reader can find it in the served file.
    """
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    named = [name for _, names, _, _ in EXCERPTS for name in names]
    expect(set(functions) == set(named) | set(OTHER_FUNCTIONS),
           f"the complete program still defines exactly its known functions, not {sorted(functions)}")
    expect(len(named) == len(set(named)), "no function is displayed in two excerpts")
    lines = source.split(NL)
    blocks = {}
    for key, names, title, guidance in EXCERPTS:
        first = functions[names[0]].lineno
        last = functions[names[-1]].end_lineno
        # A CONTIGUOUS region of the file, not a join of separate segments.
        # Joining two functions with a blank line produces text that appears
        # nowhere in the source, and "verbatim" then means only "each piece
        # was". This way the excerpt is one slice a reader can find by line
        # number in the copy the lesson serves.
        code = NL.join(lines[first - 1:last])
        expect(code in source, f"the {key} excerpt is one contiguous slice of the frozen program")
        covered = [name for name, node in functions.items()
                   if node.lineno >= first and node.end_lineno <= last]
        expect(sorted(covered) == sorted(names),
               f"the {key} excerpt covers exactly {names} and no other function, not {sorted(covered)}")
        blocks[key] = {
            "title": title,
            "guidance": guidance,
            "functions": names,
            "code": code,
            "lines": [first, last],
            "sha256": digest_text(code),
        }
    return blocks


def run_experiments(code: str) -> tuple[str, dict]:
    """Execute the complete program and compare what it writes with the packet."""
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        shutil.copy(SEQUENCES, workspace / "ewt-sequences.json")
        script = workspace / "hmm-experiments.py"
        script.write_bytes(EXPERIMENTS.read_bytes())
        expect(digest_file(script) == EXPERIMENTS_SHA, "the executed copy is the frozen program, byte for byte")
        completed = subprocess.run([sys.executable, str(script)], cwd=workspace,
                                   capture_output=True, text=True, timeout=1800, check=False)
        expect(completed.returncode == 0, f"the complete program exits cleanly ({completed.stderr[-400:]})")
        expect(completed.stderr.strip() == "", f"and prints nothing to stderr ({completed.stderr[:200]!r})")
        produced = json.loads((workspace / "calculated-inputs.json").read_text(encoding="utf-8"))
        recorded = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))
        fresh = dict(leaf_paths(produced))
        stored = dict(leaf_paths(recorded))
        shared = set(fresh) & set(stored)
        expect(len(shared) == len(fresh) >= 13243,
               f"a fresh run reproduces every leaf the packet shares with it ({len(shared)} of {len(fresh)})")
        differing = [path for path in shared if fresh[path] != stored[path]]
        expect(not differing, f"and every shared value is identical ({len(differing)} differ: {differing[:5]})")
        only_packet = sorted(set(stored) - set(fresh))
        expect(only_packet == ["additional_author_checks.duplicated_dataset_counts_double",
                               "additional_author_checks.duplicated_dataset_parameter_null",
                               "additional_author_checks.length_one_transition_rows_retained",
                               "additional_author_checks.native_example_ast_parsed_not_executed"],
               f"the only keys the program does not write are the author's four recorded probes ({only_packet})")
        oracle = {"leavesReproduced": len(shared), "leavesDiffering": len(differing),
                  "packetOnlyLeaves": only_packet}
    return completed.stdout, oracle


def run_optional(smoothed, evidence: float, best_joint: float) -> dict:
    """Execute the optional hmmlearn program in its isolated environment."""
    expect(ISOLATED.exists(),
           "an isolated environment exists at scratch/hmm-optional, so hmmlearn is never installed "
           "into the shared lesson runtime")
    if not ISOLATED.exists():
        return {"executed": False, "stdout": "", "stderr": "", "versions": {}}
    versions = json.loads(subprocess.run(
        [str(ISOLATED), "-c",
         "import json,sys,numpy,scipy,sklearn,hmmlearn;"
         "print(json.dumps({'python':sys.version.split()[0],'hmmlearn':hmmlearn.__version__,"
         "'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__}))"],
        capture_output=True, text=True, timeout=600, check=True).stdout)
    expect(versions["hmmlearn"] == "0.3.3", f"the resolved hmmlearn is 0.3.3, not {versions['hmmlearn']}")

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        script = workspace / "hmmlearn_examples.py"
        script.write_bytes(OPTIONAL.read_bytes())
        expect(digest_file(script) == OPTIONAL_SHA, "the executed optional copy is the frozen program")
        first = subprocess.run([str(ISOLATED), str(script)], cwd=workspace,
                               capture_output=True, text=True, timeout=1800, check=False)
        second = subprocess.run([str(ISOLATED), str(script)], cwd=workspace,
                                capture_output=True, text=True, timeout=1800, check=False)
    expect(first.returncode == 0, f"the optional program exits cleanly ({first.stderr[-400:]})")
    expect(first.stdout == second.stdout, "and prints the same output twice, so its seeds really fix it")
    printed = first.stdout.replace(chr(13), "")
    lines = printed.strip().split(NL)
    expect(len(lines) == 19, f"it prints 19 lines, not {len(lines)}")

    # Its own two assertions ran inside the process; a failure would have been a
    # nonzero exit. These oracles are computed here, from the model.
    score = float(lines[0].split(": ")[1])
    close(score, math.log(evidence), "the library's observation log probability is log P(o)", 1e-12)
    close(score, math.log(0.00933936), "which is log .00933936", 1e-12)
    joint_line = lines[1].split(": ")[1]
    joint_score = float(joint_line.split(" [")[0])
    close(joint_score, math.log(best_joint), "its Viterbi score is the log of the best path's joint mass", 1e-12)
    expect(joint_line.endswith("[1 1 1 0]"), f"and its path is Sunny Sunny Sunny Rainy ({joint_line})")
    expect(joint_score < score, "a single path's joint mass is below the evidence that sums over all of them")

    marginal_text = printed.split("Smoothed marginals: ")[1].split("Pointwise")[0]
    numbers = [float(value) for value in re.findall(r"-?[0-9]+[.][0-9e+-]*[0-9]", marginal_text)]
    expect(len(numbers) == 8, f"the marginal block prints eight numbers, not {len(numbers)}")
    for index, value in enumerate(numbers):
        close(value, smoothed[index // 2][index % 2],
              f"smoothed marginal {index} agrees with the lesson's own posterior", 5e-8)

    map_line = next(line for line in lines if line.startswith("Pointwise"))
    map_value = float(map_line.split(": ")[1].split(" [")[0])
    # The manuscript's version-specific trap, executed rather than asserted.
    close(map_value, sum(max(row) for row in smoothed),
          "the MAP decoder returns the SUM of the selected marginal probabilities", 5e-9)
    expect(map_value > 0, "which is positive, so it cannot be a log probability of a probability")
    expect(1 < map_value <= len(smoothed),
           "and lies between one and the number of time steps, as an expected correct count must")

    fitted = float(next(line for line in lines if line.startswith("Fitted independent-sequence score")).split(": ")[1])
    close(fitted, 2 * math.log(0.5),
          "fitting two two-step recordings reaches log .25: each recording is explained with "
          "probability one half", 1e-9)
    history_text = next(line for line in lines if line.startswith("Training history"))
    history = [float(value) for value in re.findall(r"-?[0-9]+[.][0-9e+-]*[0-9]", history_text)]
    expect(len(history) >= 2, "the monitor records more than one iteration")
    expect(all(later >= earlier - 1e-9 for earlier, later in zip(history, history[1:])),
           "whose objective never decreased")
    close(history[-1], fitted, "and whose last entry is the score reported afterwards", 1e-9)
    expect(len(history) < 30, "it stopped on its tolerance rather than exhausting its 30-iteration budget")

    means = [float(value) for value in re.findall(
        r"-?[0-9]+[.][0-9e+-]*[0-9]",
        next(line for line in lines if line.startswith("Means")))]
    expect(len(means) == 2, "the Gaussian fit prints two means")
    expect(min(means) < 0 < max(means), "one negative and one positive, as the construction intends")
    for value in means:
        close(abs(value), 1.5, "each within a little of the declared generating mean of 1.5", 0.15)
    variance_text = printed.split("Variances: ")[1].split("First recording")[0]
    variances = [float(value) for value in re.findall(r"-?[0-9]+[.][0-9e+-]*[0-9]", variance_text)]
    expect(len(variances) == 2, "and two variances")
    for value in variances:
        close(value, 0.4 ** 2, "each near the declared generating variance of .16", 0.45)
    density = float(next(line for line in lines if line.startswith("Sensor log density")).split(": ")[1])
    expect(density < 0, "the sensor log density is negative for these data")
    expect(density > -400, "and is a finite log density rather than an underflow")

    # The warning stderr carries is itself checkable: its free-parameter count is
    # exactly the manuscript's p = (N-1) + N(N-1) + N(M-1) for two states and
    # three symbols, and its four data points are the two two-step recordings.
    stderr = first.stderr.replace(chr(13), "").strip()
    expect("7 free scalar parameters" in stderr,
           f"the library counts 7 free parameters, matching the lesson's formula ({stderr[:120]!r})")
    expect((2 - 1) + 2 * (2 - 1) + 2 * (3 - 1) == 7, "which the lesson's own formula also gives")
    expect("only 4 data points" in stderr, "over the four observations of the two recordings")
    return {"executed": True, "stdout": printed.rstrip(NL), "stderr": stderr, "versions": versions}


def main() -> None:
    write = "--write" in sys.argv
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    manuscript_sha = digest_text(manuscript_text)
    expect(manuscript_sha == MANUSCRIPT_SHA,
           f"the manuscript is the pinned revision ({manuscript_sha})")
    expect(digest_file(EXPERIMENTS) == EXPERIMENTS_SHA,
           f"the complete program is the frozen revision ({digest_file(EXPERIMENTS)})")
    expect(digest_file(OPTIONAL) == OPTIONAL_SHA,
           f"the optional program is the frozen revision ({digest_file(OPTIONAL)})")
    for name in ("hmm-experiments.py", "hmmlearn-examples.py"):
        served = ROOT / "public/learn-assets/hmm" / name
        expect(served.read_bytes() == (PACKET / name).read_bytes(),
               f"the served copy of {name} is the packet's file byte for byte")

    fences = load_manuscript_fences()
    expect(fences["runCommand"].strip() == "python hmm-experiments.py",
           f"the manuscript's run command is unchanged ({fences['runCommand'].strip()!r})")

    source = EXPERIMENTS.read_text(encoding="utf-8")
    excerpts = extract_excerpts(source)
    printed, oracle = run_experiments(source)
    expect(len(source.splitlines()) == source.count(NL),
           "the frozen program ends with a newline, so its line count is its newline count")
    expect(printed.strip() == "Recorded exact mechanisms, four EM fits, and four supervised tagging configurations.",
           f"the complete program's printed line ({printed.strip()!r})")

    recorded = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))
    smoothed = recorded["mechanisms"]["main"]["smoothed"]
    evidence = math.exp(recorded["mechanisms"]["main"]["log_evidence"])
    optional = run_optional(smoothed, evidence, recorded["mechanisms"]["main"]["path_joint"])

    records = {
        "experiments": {
            "title": "The complete offline program: inference, learning and the real tagging comparison",
            "question": "Which rows does one changed report move, and which does it leave exactly alone?",
            "file": "hmm-experiments.py",
            "setup": fences["runCommand"].strip(),
            "language": "python",
            "download": "/learn-assets/hmm/hmm-experiments.py",
            "dataFile": "/learn-assets/hmm/ewt-sequences.json",
            # `count(NL) + 1` counts a phantom final line, because the file ends
            # with a newline. The page uses this number to justify showing four
            # excerpts instead of the whole program, so it has to be the number
            # `wc -l` gives.
            "lineCount": len(source.splitlines()),
            "sha256": EXPERIMENTS_SHA,
            "expected": printed.rstrip(NL),
            "executed": True,
            "excerpts": [dict(key=key, **excerpts[key]) for key, _, _, _ in EXCERPTS],
            "writes": "calculated-inputs.json",
        },
        "hmmlearn": {
            "title": "The same four reports through hmmlearn 0.3.3",
            "question": "What exactly does each returned number mean, and which one is not a log probability?",
            "file": "hmmlearn-examples.py",
            # Authored in phase two: the frozen program carries its instruction in
            # prose rather than as a command, and this is the command that was run.
            "setup": 'python -m venv hmm-optional' + NL
                     + 'hmm-optional/Scripts/python -m pip install "hmmlearn==0.3.3"' + NL
                     + 'hmm-optional/Scripts/python hmmlearn-examples.py',
            "language": "python",
            "download": "/learn-assets/hmm/hmmlearn-examples.py",
            "code": OPTIONAL.read_text(encoding="utf-8").rstrip(NL),
            "sha256": OPTIONAL_SHA,
            "expected": optional["stdout"],
            "warning": optional["stderr"],
            "executed": optional["executed"],
            "environment": optional["versions"],
            "environmentNote": (
                "Run in an isolated environment, not in this project's shared lesson runtime: "
                "resolving hmmlearn moves NumPy, and other completed lessons' recorded outputs "
                "depend on the exact versions there."),
        },
    }

    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        raise SystemExit(f"{len(failures)} of {oracles} oracle assertions failed.")

    text = (
        "// Displayed programs for the hidden Markov models lesson." + NL
        + "//" + NL
        + "// Extracted verbatim from the frozen packet files and the manuscript's own fenced" + NL
        + "// block, each pinned by SHA-256, and executed by scripts/verify-hmm-examples.py." + NL
        + "// The complete program's excerpts are exact source segments taken through Python's" + NL
        + "// AST, with the line numbers they occupy in the served file. The optional hmmlearn" + NL
        + "// program was executed in an isolated environment whose resolved versions are" + NL
        + "// recorded beside its output. Do not edit by hand." + NL
        + "export const hmmExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";" + NL
    )
    if write:
        MODULE.write_text(text, encoding="utf-8", newline=NL)
    elif not MODULE.exists() or MODULE.read_text(encoding="utf-8") != text:
        raise SystemExit("src/learn/data/hmm-examples.js is stale or missing; rerun with --write.")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native author verification of displayed programs; browser and independent review are separate",
        "source": "src/learn/data/hmm-examples.js",
        "sourceHash": digest_file(MODULE),
        "verifier": "scripts/verify-hmm-examples.py",
        "verifierHash": digest_file(Path(__file__)),
        "manuscript": "docs/teaching/drafts/hidden-markov-models-hmm/lesson.md",
        "manuscriptHash": manuscript_sha,
        "extraction": (
            "The manuscript's single fenced block is extracted by tilde-fence match and pinned by "
            "the manuscript hash. The Python programs are the packet's own frozen files, pinned by "
            "SHA-256; displayed excerpts are exact source segments taken through ast.get_source_segment, "
            "never transcribed."),
        "programs": {
            "hmm-experiments.py": {
                "sha256": EXPERIMENTS_SHA,
                "executed": True,
                "runtime": "scratch/lesson-tools (the shared lesson runtime)",
                "stdoutHash": digest_text(printed),
                "oracle": oracle,
                "excerpts": {key: {"functions": value["functions"], "lines": value["lines"],
                                   "sha256": value["sha256"]}
                             for key, value in excerpts.items()},
            },
            "hmmlearn-examples.py": {
                "sha256": OPTIONAL_SHA,
                "executed": optional["executed"],
                "runtime": "scratch/hmm-optional (isolated; hmmlearn is NOT in the shared runtime)",
                "resolvedVersions": optional["versions"],
                "stdoutHash": digest_text(optional["stdout"]),
                "stderr": optional["stderr"],
                "deterministic": True,
                "supersedes": (
                    "The packet records additional_author_checks.native_example_ast_parsed_not_executed "
                    "as true, meaning hmmlearn was absent while the content was written. That is now "
                    "superseded: the program was executed for real here and the lesson displays its "
                    "actual output."),
            },
        },
        "sharedRuntimeVersions": {name: importlib.metadata.version(name) for name in
                                  ["numpy", "scipy", "scikit-learn", "pandas"]},
        "oracles": oracles,
        "limits": [
            "No network access; the program ran beside this lesson's served copy of the extract.",
            "The optional program's Gaussian numbers are checked against the declared generating "
            "mean and variance and against qualitative properties, not pinned to exact digits, "
            "because a different hmmlearn or NumPy build may land elsewhere inside those bounds.",
            "Rendering of this code and output on the page is verified separately.",
        ],
        "passed": True,
    }, indent=2) + NL, encoding="utf-8", newline=NL)
    print(f"PASS: 2 of 2 displayed programs executed, {oracles} oracle assertions, "
          f"{oracle['leavesReproduced']:,} recorded leaves reproduced by a fresh run; "
          f"module {'written' if write else 'current'}.")


if __name__ == "__main__":
    main()
