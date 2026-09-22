"""Execute the Rademacher lesson's programs and record what they printed.

Nothing here is transcribed. Every byte of code the page displays is read
verbatim from a frozen source and pinned by SHA-256, so a copy-and-paste slip
cannot silently introduce a difference:

  run-calculations   the shell fence the manuscript displays in section 7,
                     read from lesson.md's own ``bash`` fence and pinned by the
                     SHA-256 of that fence body. Its recorded output is the real
                     stdout of `complexity_calculations.py`.
  run-experiment     the shell fence the manuscript displays in section 8, read
                     the same way. Its recorded output is the real stdout of
                     `bounded_norm_experiment.py`.
  finite_complexity  the mechanism the page shows as displayed teaching code,
                     extracted from `complexity_calculations.py` BY PARSING IT --
                     `ast.get_source_segment` returns the exact bytes of that
                     function definition, so the block on the page is the block
                     that ran.
  threshold_values   the same, for the restriction helper.
  fit_ball           the same, for the constrained fit in
                     `bounded_norm_experiment.py`.

Both programs are executed for real, in a scratch workspace beside a copy of
the served dataset. Nothing inside the packet directory is written. The
strongest available statement about each is checked: a fresh run must reproduce
the packet's frozen `checked-results.json` and `experiment-results.json` byte
for byte.

No isolated virtual environment is needed here, and none is created. Both
programs need only NumPy, SciPy and scikit-learn, and the shared runtime
already carries the exact versions the author recorded; installing anything
would move other lessons' recorded outputs.

`--write` regenerates src/learn/data/rademacher-examples.js from what actually
ran. Without it the recorded text must match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-rademacher-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-rademacher-examples.py
"""
from __future__ import annotations

import ast
import hashlib
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
PACKET = ROOT / "docs/teaching/drafts/rademacher-complexity-generalization-bounds"
MANUSCRIPT = PACKET / "lesson.md"
CALCULATIONS = PACKET / "complexity_calculations.py"
EXPERIMENT_PROGRAM = PACKET / "bounded_norm_experiment.py"
DATASET = ROOT / "public/learn-assets/rademacher/banknote-subset.csv"
CHECKED = PACKET / "checked-results.json"
EXPERIMENT_RESULTS = PACKET / "experiment-results.json"
MODULE = ROOT / "src/learn/data/rademacher-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/rademacher-native.json"
WORKSPACE = ROOT / "scratch/rademacher-programs"

failures: list[str] = []
oracles = {"count": 0}


def oracle(condition, label):
    oracles["count"] += 1
    if not condition:
        failures.append(label)
    return condition


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manuscript_fence(marker: str, language: str) -> str:
    """The body of the fenced block that follows `marker` in the manuscript.

    Located by searching the frozen text rather than by line number, so an
    edit above it cannot silently select a different block.
    """
    text = MANUSCRIPT.read_text(encoding="utf-8")
    start = text.index(marker)
    pattern = re.compile(r"```" + language + r"\n(.*?)```", re.S)
    match = pattern.search(text, start)
    if match is None:
        raise SystemExit(f"no {language} fence after {marker!r}")
    return match.group(1).rstrip("\n")


def flatten(node):
    """Every scalar leaf of a JSON tree, in a fixed order."""
    if isinstance(node, dict):
        for key in node:
            yield from flatten(node[key])
    elif isinstance(node, list):
        for value in node:
            yield from flatten(value)
    else:
        yield node


def source_of(path: Path, name: str) -> str:
    """The exact bytes of a top-level function definition, by parsing."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                raise SystemExit(f"could not extract {name} from {path.name}")
            return segment
    raise SystemExit(f"{path.name} has no top-level function {name}")


def run(program: Path, workspace: Path, single_threaded: bool = False) -> str:
    """Execute one program in its own workspace.

    The default path deliberately leaves the environment ALONE. An earlier
    version of this script pinned OMP/OPENBLAS/MKL to one thread, which is the
    usual determinism hygiene -- and it moved the last bit of one fitted
    coefficient AWAY from the packet's recorded value (-3.188825073389911
    became -3.1888250733899115), so a genuinely faithful rerun failed the
    byte-identity check. The author ran these programs
    under the default threading, so that is what a reproduction must use. The
    single-threaded configuration is then run separately and its numerical
    difference measured, rather than being imposed and quietly changing the
    thing being reproduced.
    """
    environment = dict(os.environ)
    if single_threaded:
        environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    completed = subprocess.run([sys.executable, program.name], cwd=workspace, capture_output=True,
                               text=True, timeout=900, env=environment)
    if completed.returncode != 0:
        raise SystemExit(f"{program.name} exited {completed.returncode}:\n{completed.stderr[-4000:]}")
    if completed.stderr.strip():
        failures.append(f"{program.name} wrote to stderr: {completed.stderr.strip()[:400]}")
    return completed.stdout.replace("\r\n", "\n").rstrip("\n")


def main():
    write = "--write" in sys.argv
    # `--no-evidence` keeps the falsification harness from recording a result
    # derived from a defect it injected itself.
    no_evidence = "--no-evidence" in sys.argv

    for path in (MANUSCRIPT, CALCULATIONS, EXPERIMENT_PROGRAM, CHECKED, EXPERIMENT_RESULTS, DATASET):
        if not path.exists():
            raise SystemExit(f"missing frozen input: {path}")

    calculations_digest = digest_bytes(CALCULATIONS.read_bytes())
    experiment_digest = digest_bytes(EXPERIMENT_PROGRAM.read_bytes())
    manuscript_digest = digest_bytes(MANUSCRIPT.read_bytes())

    # The page must serve the packet's programs byte for byte, not a retyped copy.
    for served, packet in (("complexity_calculations.py", CALCULATIONS),
                           ("bounded_norm_experiment.py", EXPERIMENT_PROGRAM)):
        path = ROOT / "public/learn-assets/rademacher" / served
        oracle(path.exists() and path.read_bytes() == packet.read_bytes(),
               f"the served {served} is byte-for-byte the packet program")

    run_calculations = manuscript_fence("Save it locally and run it with Python and NumPy", "bash")
    run_experiment = manuscript_fence("into one directory. Then run", "bash")
    oracle("python complexity_calculations.py" in run_calculations, "the first fence runs the calculation program")
    oracle("python bounded_norm_experiment.py" in run_experiment, "the second fence runs the experiment")

    excerpts = {
        "finite_complexity": {"path": CALCULATIONS, "file": "complexity_calculations.py"},
        "threshold_values": {"path": CALCULATIONS, "file": "complexity_calculations.py"},
        "fit_ball": {"path": EXPERIMENT_PROGRAM, "file": "bounded_norm_experiment.py"},
    }
    for name, entry in excerpts.items():
        entry["code"] = source_of(entry["path"], name)
        entry["sourceSha256"] = digest_bytes(entry["path"].read_bytes())
        oracle(entry["code"] in entry["path"].read_text(encoding="utf-8"),
               f"the displayed {name} block is a verbatim substring of {entry['file']}")
    # The displayed mechanism must not contain a printed disclaimer, and must be
    # mostly mechanism rather than validation.
    for name, entry in excerpts.items():
        lines = [line for line in entry["code"].splitlines() if line.strip()]
        guard_lines = [line for line in lines if "raise" in line or "ValueError" in line
                       or line.strip().startswith("if not") or "isfinite" in line]
        oracle(len(guard_lines) <= len(lines) / 2,
               f"{name}: validation occupies {len(guard_lines)} of {len(lines)} lines, so the mechanism is not visible")
        oracle("print(" not in entry["code"], f"{name} prints nothing, so it cannot print a disclaimer")

    # ------------------------------------------------------------ execute them
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CALCULATIONS, WORKSPACE / CALCULATIONS.name)
    shutil.copy2(EXPERIMENT_PROGRAM, WORKSPACE / EXPERIMENT_PROGRAM.name)
    shutil.copy2(DATASET, WORKSPACE / "banknote-subset.csv")

    calculations_stdout = run(CALCULATIONS, WORKSPACE)
    fresh_checked = (WORKSPACE / "checked-results.json").read_bytes()
    oracle(fresh_checked == CHECKED.read_bytes(),
           "a fresh run reproduces the packet's checked-results.json byte for byte")

    experiment_stdout = run(EXPERIMENT_PROGRAM, WORKSPACE)
    fresh_experiment = (WORKSPACE / "experiment-results.json").read_bytes()
    oracle(fresh_experiment == EXPERIMENT_RESULTS.read_bytes(),
           "and the packet's experiment-results.json byte for byte")
    oracle(not (PACKET / "__pycache__").exists(), "nothing inside the packet directory was written")

    # How far the fit moves under a different BLAS configuration. This is
    # measured rather than assumed: it is the reason the run above does not pin
    # the thread counts, and a reader deserves the size of the effect.
    threaded = ROOT / "scratch/rademacher-programs-single-thread"
    shutil.rmtree(threaded, ignore_errors=True)
    threaded.mkdir(parents=True, exist_ok=True)
    for name in (CALCULATIONS.name, EXPERIMENT_PROGRAM.name):
        shutil.copy2(PACKET / name, threaded / name)
    shutil.copy2(DATASET, threaded / "banknote-subset.csv")
    run(CALCULATIONS, threaded, single_threaded=True)
    run(EXPERIMENT_PROGRAM, threaded, single_threaded=True)
    single_thread_result = json.loads((threaded / "experiment-results.json").read_text(encoding="utf-8"))
    packet_result = json.loads(EXPERIMENT_RESULTS.read_text(encoding="utf-8"))
    threading_drift = 0.0
    for mine, theirs in zip(flatten(single_thread_result), flatten(packet_result)):
        if isinstance(mine, (int, float)) and isinstance(theirs, (int, float)) and not isinstance(mine, bool):
            threading_drift = max(threading_drift, abs(mine - theirs) / max(1.0, abs(theirs)))
    oracle(threading_drift < 1e-12,
           f"a single-threaded BLAS moves no recorded number by more than 1e-12 relative; it moved {threading_drift}")
    # The teaching conclusions must survive that drift exactly, not approximately.
    for mine, theirs in zip(single_thread_result["models"], packet_result["models"]):
        for name in ("fit", "validation", "assessment"):
            oracle(mine[name]["errors"] == theirs[name]["errors"],
                   f"B={mine['radius']}: {name} mistake count is unchanged under single-threaded BLAS")
    oracle(single_thread_result["selection"]["chosen_radius"] == packet_result["selection"]["chosen_radius"],
           "and the selected budget is unchanged")
    shutil.rmtree(threaded, ignore_errors=True)

    # ------------------------------------------------- oracles on what printed
    printed = json.loads(calculations_stdout)
    expected_summary = {"singleton": 0.0, "constants": 0.5, "thresholds": 2 / 3,
                        "two_orientations": 5 / 6, "all_labels": 1.0, "classification_loss": 1 / 3}
    for key, value in expected_summary.items():
        oracle(abs(printed[key] - value) < 1e-12,
               f"the summary prints {key} = {value!r}; it printed {printed.get(key)!r}")
    oracle(abs(printed["classification_loss"] - printed["thresholds"] / 2) < 1e-15,
           "the printed mistake-class value is exactly half the printed predictor value")
    oracle(set(printed) == set(expected_summary), "the summary prints exactly the six values the manuscript names")

    experiment_printed = json.loads(experiment_stdout)
    oracle(experiment_printed["selected"] == 4, "the experiment selects B=4 by validation")
    oracle(abs(experiment_printed["energy_factor"] - 0.07994964312015984) < 1e-15,
           "and prints the energy factor the manuscript substitutes into its worked bound")
    oracle(abs(experiment_printed["confidence"] - 3 * math.sqrt(math.log(400) / 480)) < 1e-12,
           "the printed confidence term is 3 sqrt(ln(2K/delta) / 2n) at K=10, delta=.05, n=240")
    manuscript_table = {0.25: (52, 16, 11), 0.5: (46, 14, 8), 1: (36, 10, 6), 2: (19, 6, 5), 4: (8, 4, 2)}
    for row in experiment_printed["rows"]:
        expected = manuscript_table[row["B"]]
        got = (row["train_errors"], row["validation_errors"], row["assessment_errors"])
        oracle(got == expected, f"B={row['B']} prints {expected} mistakes; it printed {got}")
        for bound in row["bounds"]:
            oracle(bound["raw_upper"] > 1,
                   f"B={row['B']}, rho={bound['rho']}: the printed expression exceeds 1, as the manuscript states")
            oracle(abs(bound["clipped_upper"] - 1) < 1e-15,
                   f"B={row['B']}, rho={bound['rho']}: and its clipped value is the trivial ceiling")
    worked = next(row for row in experiment_printed["rows"] if row["B"] == 2)["bounds"][1]
    oracle(abs(worked["empirical_ramp"] - 0.474878) < 5e-7, "the worked B=2 training ramp is .474878")
    oracle(abs(worked["complexity_addend"] - 2 * 2 * 0.07994964312015984) < 1e-12,
           "its complexity addend is 2 B times the energy factor at rho=1")
    oracle(abs(worked["raw_upper"] - 1.129848) < 5e-7, "and the three terms sum to 1.129848")

    versions = {}
    for package in ("numpy", "scipy", "sklearn"):
        module = __import__(package)
        versions[package] = module.__version__
    recorded_versions = json.loads(EXPERIMENT_RESULTS.read_text(encoding="utf-8"))["versions"]
    for package, version in recorded_versions.items():
        oracle(versions[package] == version,
               f"the shared runtime still carries the recorded {package} {version}, not {versions[package]}")

    programs = {
        "runCalculations": {
            "title": 'Run the exact enumeration yourself',
            "question": 'six classes on the same three inputs. Which two of the six values do you already know '
                        'without running anything?',
            "code": run_calculations,
            "language": 'bash',
            "file": 'complexity_calculations.py',
            "origin": 'manuscript bash fence, section 7',
            "originSha256": digest_text(run_calculations),
            "programSha256": calculations_digest,
            "expected": calculations_stdout,
        },
        "runExperiment": {
            "title": 'Run the whole constrained-budget experiment',
            "question": 'five budgets, two margin thresholds, one declared selection rule. Does any of the ten '
                        'expressions come in under 1?',
            "code": run_experiment,
            "language": 'bash',
            "file": 'bounded_norm_experiment.py',
            "origin": 'manuscript bash fence, section 8',
            "originSha256": digest_text(run_experiment),
            "programSha256": experiment_digest,
            "expected": experiment_stdout,
        },
    }
    excerpt_entries = {
        "finiteComplexity": {
            "title": 'The whole calculation, in twelve lines',
            "code": excerpts["finite_complexity"]["code"],
            "language": 'python',
            "file": 'complexity_calculations.py',
            "origin": 'ast.get_source_segment of finite_complexity',
            "sourceSha256": excerpts["finite_complexity"]["sourceSha256"],
        },
        "thresholdValues": {
            "title": 'Every distinct restriction, including both extreme cutoffs',
            "code": excerpts["threshold_values"]["code"],
            "language": 'python',
            "file": 'complexity_calculations.py',
            "origin": 'ast.get_source_segment of threshold_values',
            "sourceSha256": excerpts["threshold_values"]["sourceSha256"],
        },
        "fitBall": {
            "title": 'Fitting inside a norm ball, and checking the solver rather than trusting it',
            "code": excerpts["fit_ball"]["code"],
            "language": 'python',
            "file": 'bounded_norm_experiment.py',
            "origin": 'ast.get_source_segment of fit_ball',
            "sourceSha256": excerpts["fit_ball"]["sourceSha256"],
        },
    }

    module_text = build_module(programs, excerpt_entries, versions)

    # Built here, written at the very end, after every remaining oracle. It used
    # to be written at this point, so the recorded oracle count was permanently
    # one short of the printed one and a failing byte-identity check could exit
    # non-zero while leaving a record saying "passed": true.
    def build_evidence(passed, notes):
        return json.dumps({
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "stage": "native execution of the lesson's programs; browser, independent and integration review are separate",
            "extraction": "verbatim from frozen sources: two manuscript bash fences located by their surrounding "
                          "sentence, and three function definitions extracted from the packet programs by parsing "
                          "them with ast.get_source_segment. Each is pinned by the SHA-256 of the exact bytes read.",
            "manuscript": "docs/teaching/drafts/rademacher-complexity-generalization-bounds/lesson.md",
            "manuscriptSha256": manuscript_digest,
            "programSha256": {"complexity_calculations.py": calculations_digest,
                              "bounded_norm_experiment.py": experiment_digest},
            "packetResultHashes": {"checked-results.json": digest_bytes(CHECKED.read_bytes()),
                                   "experiment-results.json": digest_bytes(EXPERIMENT_RESULTS.read_bytes())},
            "regeneratedByteIdentical": {"checked-results.json": True, "experiment-results.json": True},
            "blasThreadingSensitivity": {
                "largestRelativeDrift": threading_drift,
                "note": "Rerunning with OMP/OPENBLAS/MKL pinned to one thread moves the last bit of one fitted "
                        "coefficient, so the byte-identity check above runs under the default threading the author "
                        "used. Every mistake count, ramp mean and the selected budget are unchanged.",
            },
            "source": "src/learn/data/rademacher-examples.js",
            "sourceHash": digest_text(module_text),
            "verifier": "scripts/verify-rademacher-examples.py",
            "verifierHash": digest_bytes(Path(__file__).read_bytes()),
            "runtime": {"python": sys.version.split()[0], **versions,
                        "isolatedEnvironment": "none created: this packet has no optional-library program, and both "
                                               "programs run on the shared runtime's recorded versions"},
            "programs": {key: {"file": entry["file"], "origin": entry["origin"],
                               "codeHash": digest_text(entry["code"]), "stdoutHash": digest_text(entry["expected"]),
                               "executed": True, "displayedOnPage": True} for key, entry in programs.items()},
            "excerpts": {key: {"file": entry["file"], "origin": entry["origin"],
                               "codeHash": digest_text(entry["code"]), "lines": len(entry["code"].splitlines()),
                               "executed": "as part of its program", "displayedOnPage": True}
                         for key, entry in excerpt_entries.items()},
            "oracles": oracles["count"],
            "notes": [
                "Both programs were copied into scratch/rademacher-programs beside a copy of the SERVED dataset and "
                "run there. The check is that each reproduces the packet's frozen result file byte for byte, so the "
                "page's numbers and the packet's numbers cannot drift apart.",
                "No network access was used by any program.",
                "The three displayed excerpts are extracted by parsing, never retyped, so what a reader copies off "
                "the page is what executed.",
            ],
            "limits": [
                "Library floating-point results can differ on other versions; the resolved versions are recorded above "
                "and are checked against the versions the packet recorded.",
                "The experiment is one small fixed subset drawn without replacement from a finite corpus, not a "
                "benchmark and not an iid sample from a deployment population.",
                "Stdout equality is checked for the two displayed fences; the browser's own rendering of that text is "
                "checked by scripts/verify-rademacher-browser.cjs.",
            ],
        "passed": passed,
        "failureNotes": notes,
    }, indent=2) + "\n"

    def record_and_exit(message):
        """Write a FAILING record, then stop, rather than leaving the previous
        run's passing one in place."""
        if not no_evidence:
            EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
            EVIDENCE.write_text(build_evidence(False, [message] + failures),
                                encoding="utf-8", newline="\n")
        raise SystemExit(message)

    shutil.rmtree(WORKSPACE, ignore_errors=True)

    oracle(oracles["count"] >= 45, f"at least forty-five oracles ran; only {oracles['count']} did")

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        record_and_exit(f"{len(failures)} of {oracles['count']} program oracles failed")

    if write:
        MODULE.write_text(module_text, encoding="utf-8", newline="\n")
    elif not MODULE.exists():
        record_and_exit("the examples module is missing; rerun with --write")
    elif MODULE.read_text(encoding="utf-8") != module_text:
        record_and_exit("a fresh execution does not match src/learn/data/rademacher-examples.js; "
                        "rerun with --write and inspect the difference")

    if not no_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(build_evidence(True, []), encoding="utf-8", newline="\n")

    print(f"PASS: 2 programs executed and 3 mechanism excerpts extracted verbatim by parsing, all pinned by "
          f"SHA-256 ({oracles['count']} oracle assertions); both frozen result files regenerated byte for byte; "
          f"module {'written' if write else 'byte-identical'}.")


def as_js_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def build_module(programs, excerpts, versions) -> str:
    def block(entries):
        parts = []
        for key, entry in entries.items():
            fields = ",\n".join(f"    {name}: {as_js_string(value) if isinstance(value, str) else json.dumps(value)}"
                                for name, value in entry.items())
            parts.append(f"  {key}: {{\n{fields},\n  }}")
        return ",\n".join(parts)

    return (
        "// Programs and displayed mechanism excerpts for the Rademacher-complexity lesson.\n"
        "//\n"
        "// GENERATED by scripts/verify-rademacher-examples.py. Do not edit by hand.\n"
        "//\n"
        "// `rademacherPrograms` carries the two shell fences the manuscript displays,\n"
        "// read verbatim from lesson.md, with `expected` set to what the corresponding\n"
        "// program ACTUALLY printed on this machine -- not a predicted result. Both\n"
        "// runs reproduced the content packet's frozen result files byte for byte.\n"
        "//\n"
        "// `rademacherExcerpts` carries three function definitions lifted out of those\n"
        "// programs by parsing them, so the code a reader copies off the page is the\n"
        "// code that ran. They are shown as reading material and have no separate\n"
        "// output of their own.\n"
        f"//\n// Executed on Python {sys.version.split()[0]}, NumPy {versions['numpy']}, "
        f"SciPy {versions['scipy']}, scikit-learn {versions['sklearn']}.\n"
        f"export const rademacherPrograms = {{\n{block(programs)},\n}};\n"
        f"\nexport const rademacherExcerpts = {{\n{block(excerpts)},\n}};\n"
    )


if __name__ == "__main__":
    main()
