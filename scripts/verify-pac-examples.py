"""Execute the PAC/VC lesson's programs and record what they actually printed.

Nothing here is transcribed. Every line of algorithm shown on the page is a
byte-exact slice of a frozen packet program, located by its `def` line and
pinned by the SHA-256 of the extracted bytes. If the packet changes, the pin
fails and the change has to be read rather than absorbed.

Five programs, in two groups.

  DISPLAYED (three): each is assembled as
      [import lines composed here] + [verbatim extracted functions] + [driver
      composed here]
  and then run. The evidence file records which bytes were extracted and which
  were composed, so "verbatim" is a checkable claim rather than a word. The
  driver exists because the frozen file's own `__main__` block writes a 60 KB
  JSON file; a learner reading the page needs four printed lines instead.

  DOWNLOADED (two): `pac-calculations.py` and `banknote-learning-curves.py` are
  offered whole, as the manuscript offers them. They are not displayed in full,
  so the strongest available statement about them is checked instead: copied
  into a scratch directory beside the served dataset and run there, each must
  reproduce the packet's frozen results file BYTE FOR BYTE. Nothing inside the
  packet directory is written, and that is checked by fingerprinting the whole
  directory before and after.

`--write` regenerates src/learn/data/pac-examples.js from what actually ran.
Without it the recorded text must match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-pac-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-pac-examples.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/pac-learning-vc-dimension"
CALCULATIONS = PACKET / "pac-calculations.py"
CURVES = PACKET / "banknote-learning-curves.py"
PACKET_RESULTS = PACKET / "checked-results.json"
PACKET_CURVE_RESULTS = PACKET / "banknote-learning-curve-results.json"
DATASET = ROOT / "public/learn-assets/pac-learning/banknote-subset.csv"
MODULE = ROOT / "src/learn/data/pac-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/pac-native.json"
WORKSPACE = ROOT / "scratch/pac-programs"

CALCULATIONS_SHA = "33bad8e470ef16a5e5052620d0b747d1b880e734c9e440eff9a39d3db26ea51e"
CURVES_SHA = "ab4f7106bd5c33ea07f5f199c530b269997328c2c96a3f9df67b296c5f9a29d9"

DISPLAYED = [
    {
        "key": "patterns",
        "file": "interval_patterns.py",
        "title": "Which label patterns can a threshold or an interval actually make?",
        "question": "three points and eight possible labelings. One of the eight has no interval at all. "
                    "Which one, and is that a limit of the search or a limit of the class?",
        "imports": [],
        "functions": ["interval_patterns", "threshold_patterns"],
        "extractedSha256": "20b4f035bd99094d82c6008d7df5c44c9a8476646e1b9e3999278fcbd3fdbe93",
        "driver": [
            "for n in range(1, 6):",
            "    print(n, len(threshold_patterns(n)), len(interval_patterns(n)), 2**n)",
            "print(sorted(set(map(''.join, (map(str, p) for p in interval_patterns(3))))))",
        ],
    },
    {
        "key": "finite-world",
        "file": "finite_world.py",
        "title": "The exact failure probability in a four-input world",
        "question": "sixteen candidate rules, four equally likely inputs and a learner that predicts 0 where it "
                    "has seen nothing. How fast does the exact failure probability fall, and what does the "
                    "generic finite-class bound say at the same sizes?",
        "imports": ["from fractions import Fraction", "from math import exp"],
        "functions": ["finite_world"],
        "extractedSha256": "9994314494a7958cf8a9a1182b0561009d90b01ae2348f56915b5cc94b4c31b9",
        "driver": [
            "for n in (1, 2, 4, 8, 16, 24):",
            "    run = finite_world(n)",
            "    print(n, run['failure_fraction_exact'], round(run['bound_raw'], 6), "
            "round(run['bound_clipped'], 6))",
            "print('all-zero target at n=4:', finite_world(4, target=(0, 0, 0, 0))"
            "['failure_fraction_exact'])",
        ],
    },
    {
        "key": "interval-risk",
        "file": "interval_risk.py",
        "title": "A fitted interval whose true risk can be calculated, not estimated",
        "question": "the target is [.3, .7] and the inputs are uniform on [0, 1], so a length is a probability. "
                    "Which of these four samples moves the fit, and which moves only the picture?",
        "imports": ["import numpy as np"],
        "functions": ["interval_witness", "interval_risk", "fit_realizable_interval"],
        "extractedSha256": "5cb5119425f032a1be1ba98f4d79f6dbbd179b7b6abd55f8c2feabf974f0f89e",
        "driver": [
            "for sample in ([.1, .2, .35, .55, .65, .9], [.1, .2, .31, .35, .55, .65, .69, .9],",
            "               [.01, .1, .2, .35, .55, .65, .9, .99], [.1, .2, .8, .9]):",
            "    fit = fit_realizable_interval(sample)",
            "    print(fit['interval'], fit['empirical_error'], round(fit['true_error'], 6))",
            "print(interval_witness([.2, .5, .8], [1, 0, 1]))",
        ],
    },
]

DOWNLOADS = [
    {
        "key": "calculations",
        "file": "pac-calculations.py",
        "source": CALCULATIONS,
        "sha256": CALCULATIONS_SHA,
        "produces": "checked-results.json",
        "frozen": PACKET_RESULTS,
        "needsDataset": False,
        "title": "The complete constructions behind every number on this page",
        "question": "pattern enumeration, checked separator margins, exact four-input probabilities, the bound "
                    "expressions, the sine witnesses and the repeated interval experiment, in one run. Does it "
                    "still produce exactly the recorded results?",
    },
    {
        "key": "curves",
        "file": "banknote-learning-curves.py",
        "source": CURVES,
        "sha256": CURVES_SHA,
        "produces": "banknote-learning-curve-results.json",
        "frozen": PACKET_CURVE_RESULTS,
        "needsDataset": True,
        "title": "The measured development curves",
        "question": "three predeclared procedures at five nested training sizes against one fixed development "
                    "set. No test row is touched. Does a fresh run reproduce the recorded counts?",
    },
]

write = "--write" in sys.argv
# S9: an independent reviewer is the person most likely to re-run this and the
# person who must not disturb the record. `--no-evidence` skips the write.
keep_evidence = "--no-evidence" not in sys.argv
oracle_count = 0
failures: list[str] = []


def oracle(condition, label):
    global oracle_count
    oracle_count += 1
    if not condition:
        failures.append(label)


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def extract_function(source, name):
    """The exact bytes of one top-level function, from its `def` line to the
    line before the next top-level statement."""
    lines = source.split("\n")
    start = next((index for index, line in enumerate(lines) if line.startswith(f"def {name}(")), None)
    if start is None:
        raise SystemExit(f"the frozen program has no top-level function {name}; the extraction pattern is wrong")
    end = start + 1
    while end < len(lines) and (lines[end] == "" or lines[end].startswith((" ", "\t", ")"))):
        end += 1
    while end > start and lines[end - 1] == "":
        end -= 1
    return "\n".join(lines[start:end])


def assemble(program, source):
    parts = list(program["imports"])
    extracted = [extract_function(source, name) for name in program["functions"]]
    # O2: `oracle(piece in source)` could not fail -- a contiguous run of lines
    # rejoined by newline is a substring of the source by construction. The
    # property is carried by the SHA pin below; what is worth asserting here is
    # that the span is a whole function and nothing was silently truncated.
    for piece in extracted:
        first = piece.split(chr(10))[0]
        oracle(first.startswith("def ") and first.rstrip().endswith(":"),
               f"{program['file']}: the extracted span does not begin at a def line: {first[:60]}")
        oracle(len(piece.split(chr(10))) >= 2,
               f"{program['file']}: the extracted span for {first[:40]} has no body")
    body = "\n\n\n".join(extracted)
    program["extracted"] = body
    program["extractedDigest"] = digest_text(body)
    # A pin on the extracted bytes, not only on the whole frozen file: it fails
    # if the extraction pattern ever picks up a different span.
    # O2: an empty pin used to disable this silently. A pin is now required.
    oracle(program["extractedSha256"] != "", f"{program['file']}: the extracted algorithm carries no SHA pin")
    oracle(program["extractedSha256"] == program["extractedDigest"],
           f"{program['file']}: the extracted algorithm now hashes {program['extractedDigest']}, "
           f"not the pinned {program['extractedSha256']}")
    header = ("\n".join(parts) + "\n\n\n") if parts else ""
    program["code"] = header + body + "\n\n\n" + "\n".join(program["driver"])
    program["composedLines"] = len(parts) + len(program["driver"])
    program["extractedLines"] = len(body.split("\n"))


def run_file(path, cwd, timeout=900):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONHASHSEED="0", PYTHONWARNINGS="ignore")
    return subprocess.run([sys.executable, str(path)], cwd=str(cwd), env=environment,
                          capture_output=True, text=True, timeout=timeout)


def packet_fingerprint():
    """Every file in the frozen packet, by digest. The claim is that running
    these programs writes nothing inside the packet; comparing one file against
    a copy of itself taken after the run would assert nothing."""
    return {path.relative_to(PACKET).as_posix(): digest_bytes(path.read_bytes())
            for path in sorted(PACKET.rglob("*")) if path.is_file()}


NUMBER = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"


def floats(text):
    return [float(token) for token in re.findall(NUMBER, text)]


def main():
    before = packet_fingerprint()
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copyfile(DATASET, WORKSPACE / "banknote-subset.csv")

    calculations_source = CALCULATIONS.read_bytes()
    oracle(digest_bytes(calculations_source) == CALCULATIONS_SHA,
           f"pac-calculations.py hashes {digest_bytes(calculations_source)}, not the pinned {CALCULATIONS_SHA}")
    oracle(digest_bytes(CURVES.read_bytes()) == CURVES_SHA,
           f"banknote-learning-curves.py hashes {digest_bytes(CURVES.read_bytes())}, not the pinned {CURVES_SHA}")
    source_text = calculations_source.decode("utf-8").replace("\r\n", "\n")

    for program in DISPLAYED:
        assemble(program, source_text)
        target = WORKSPACE / program["file"]
        target.write_text(program["code"] + "\n", encoding="utf-8", newline="\n")
        completed = run_file(target, WORKSPACE)
        oracle(completed.returncode == 0,
               f"{program['file']} exited {completed.returncode}: {completed.stderr.strip()[:400]}")
        program["executed"] = completed.returncode == 0
        program["expected"] = completed.stdout.replace("\r\n", "\n").strip("\n")
        program["stderr"] = completed.stderr.replace("\r\n", "\n").strip("\n")

    by_key = {program["key"]: program for program in DISPLAYED}

    # ---- the pattern program prints the growth table and names the missing 101
    patterns_output = by_key["patterns"].get("expected", "")
    rows = [line.split() for line in patterns_output.split("\n") if re.fullmatch(r"\d+( \d+){3}", line)]
    oracle(len(rows) == 5, f"the pattern program prints five rows, got {len(rows)}")
    expected_rows = [["1", "2", "2", "2"], ["2", "3", "4", "4"], ["3", "4", "7", "8"],
                     ["4", "5", "11", "16"], ["5", "6", "16", "32"]]
    oracle(rows == expected_rows, f"the printed growth rows are {rows}, not {expected_rows}")
    oracle("'101'" not in patterns_output,
           "the printed three-point interval patterns must not contain 101")
    oracle("'111'" in patterns_output and "'000'" in patterns_output,
           "but they do contain 111 and 000")
    oracle(patterns_output.count("'") == 14, "seven three-point patterns are printed, each quoted twice")

    # ---- the finite world prints exact fractions that halve each step
    world_output = by_key["finite-world"].get("expected", "")
    fractions = re.findall(r"^(\d+) (\S+) (\S+) (\S+)$", world_output, re.M)
    oracle(len(fractions) == 6, f"the finite-world program prints six sizes, got {len(fractions)}")
    exact = {int(row[0]): row[1] for row in fractions}
    oracle(exact.get(1) == "1/2" and exact.get(4) == "1/16" and exact.get(24) == "1/16777216",
           f"the exact failure fractions are wrong: {exact}")
    for row in fractions:
        n, _fraction, raw, clipped = int(row[0]), row[1], float(row[2]), float(row[3])
        oracle(abs(min(1.0, raw) - clipped) < 1e-9, f"the clipped bound at n={n} is not min(1, raw)")
    oracle("all-zero target at n=4: 0" in world_output,
           "the all-zero target is an exact null: the same class, no failure at all")

    # ---- the interval program prints four fits, two of which are the same
    risk_output = by_key["interval-risk"].get("expected", "")
    fit_lines = [line for line in risk_output.split("\n") if line.startswith("[") or line.startswith("None")]
    oracle(len(fit_lines) == 4, f"the interval program prints four fits, got {len(fit_lines)}")
    if len(fit_lines) == 4:
        oracle(fit_lines[0] == "[0.35, 0.65] 0.0 0.1", f"the base fit prints {fit_lines[0]!r}")
        oracle(fit_lines[1] == "[0.31, 0.69] 0.0 0.02", f"the near-edge fit prints {fit_lines[1]!r}")
        oracle(fit_lines[2] == fit_lines[0],
               f"exterior negatives are an exact null: {fit_lines[2]!r} versus {fit_lines[0]!r}")
        oracle(fit_lines[3] == "None 0.0 0.4", f"the no-positive fit prints {fit_lines[3]!r}")
    oracle("'feasible': False" in risk_output,
           "a positive-negative-positive request has no consistent interval, and the program says so")

    # ---- the two whole programs reproduce the frozen results byte for byte
    for download in DOWNLOADS:
        target = WORKSPACE / download["file"]
        shutil.copyfile(download["source"], target)
        completed = run_file(target, WORKSPACE)
        oracle(completed.returncode == 0,
               f"{download['file']} exited {completed.returncode}: {completed.stderr.strip()[:400]}")
        download["executed"] = completed.returncode == 0
        download["stdout"] = completed.stdout.replace("\r\n", "\n").strip("\n")
        produced = WORKSPACE / download["produces"]
        oracle(produced.exists(), f"{download['file']} wrote {download['produces']} beside itself")
        if produced.exists():
            fresh = produced.read_bytes()
            frozen = download["frozen"].read_bytes()
            download["producedSha256"] = digest_bytes(fresh)
            oracle(fresh == frozen,
                   f"a fresh run of {download['file']} reproduces {download['produces']} exactly "
                   f"({digest_bytes(fresh)} versus {digest_bytes(frozen)})")

    after = packet_fingerprint()
    oracle(bool(before), "the packet directory held files to fingerprint before the run")
    oracle(after == before,
           "nothing inside the frozen packet directory was written: "
           + (", ".join(sorted(set(before) ^ set(after))
                        or [name for name in before if before[name] != after.get(name)]) or "unchanged"))
    oracle(not any(path.name.endswith(".py") and path.parent == PACKET and path.stat().st_size == 0
                   for path in PACKET.glob("*.py")), "no packet program was truncated")

    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    environment = {"python": sys.version.split()[0], **versions}

    entries = {}
    for program in DISPLAYED:
        entries[program["key"]] = {
            "title": program["title"],
            "question": program["question"],
            "code": program["code"],
            "language": "python",
            "file": program["file"],
            "executed": bool(program.get("executed")),
            "expected": program.get("expected", ""),
            "environment": environment,
            "extraction": {
                "origin": "docs/teaching/drafts/pac-learning-vc-dimension/pac-calculations.py",
                "functions": program["functions"],
                "extractedSha256": program["extractedDigest"],
                "extractedLines": program["extractedLines"],
                "composedLines": program["composedLines"],
            },
        }
    for download in DOWNLOADS:
        entries[download["key"]] = {
            "title": download["title"],
            "question": download["question"],
            "file": download["file"],
            "language": "python",
            "downloadOnly": True,
            "executed": bool(download.get("executed")),
            "expected": download.get("stdout", ""),
            "environment": environment,
            "reproduces": download["produces"],
            "sourceSha256": download["sha256"],
        }

    header = (
        "// Programs for the PAC learning and VC dimension lesson.\n"
        "//\n"
        "// Generated by scripts/verify-pac-examples.py. The algorithm in each displayed\n"
        "// program is a byte-exact slice of the content packet's pac-calculations.py,\n"
        "// located by its def line and pinned by SHA-256; only the import lines and the\n"
        "// few printing lines under them are composed, and `extraction` records exactly\n"
        "// how many lines are which. `expected` is what the program printed on this\n"
        "// machine, not a predicted result.\n"
        "//\n"
        "// The two whole programs are offered as downloads rather than displayed. Each\n"
        "// was run in scratch/pac-programs beside the served dataset and had to\n"
        "// reproduce the packet's frozen results file byte for byte.\n"
        "//\n"
        "// Do not edit by hand.\n"
    )
    module = header + "export const pacExamples = " + json.dumps(entries, indent=2) + ";\n"

    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        if not MODULE.exists():
            raise SystemExit("the examples module is missing; rerun with --write")
        if MODULE.read_text(encoding="utf-8") != module:
            raise SystemExit("a fresh execution does not reproduce src/learn/data/pac-examples.js; "
                             "rerun with --write and inspect the difference")

    # S5/S8: the coverage floor runs BEFORE the evidence is written. Written
    # after, a tripped floor would exit non-zero and still leave `passed: true`
    # on disk -- and the evidence file, not the console, is the durable record.
    oracle(oracle_count >= 40, f"at least forty oracles ran; only {oracle_count} did")

    evidence = json.dumps({
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native execution of the lesson's programs; browser, independent and integration review are separate",
        "extraction": "Each displayed algorithm is a byte-exact slice of a frozen packet program, located by its "
                      "def line and pinned by the SHA-256 of the extracted bytes. Import lines and driver prints "
                      "are composed by this verifier and counted separately below.",
        "verifier": "scripts/verify-pac-examples.py",
        "verifierSha256": digest_bytes(Path(__file__).read_bytes()),
        "frozenSources": {
            "pac-calculations.py": digest_bytes(calculations_source),
            "banknote-learning-curves.py": digest_bytes(CURVES.read_bytes()),
            "checked-results.json": digest_bytes(PACKET_RESULTS.read_bytes()),
            "banknote-learning-curve-results.json": digest_bytes(PACKET_CURVE_RESULTS.read_bytes()),
        },
        "servedDataset": {"path": "/learn-assets/pac-learning/banknote-subset.csv",
                          "sha256": digest_bytes(DATASET.read_bytes())},
        "module": {"path": "src/learn/data/pac-examples.js",
                   "sha256": digest_bytes(MODULE.read_bytes()) if MODULE.exists() else None,
                   "regeneration": "written" if write else "byte-identical"},
        "runtime": environment,
        "displayedPrograms": {program["key"]: {
            "file": program["file"],
            "functions": program["functions"],
            "extractedSha256": program["extractedDigest"],
            "extractedLines": program["extractedLines"],
            "composedLines": program["composedLines"],
            "executed": bool(program.get("executed")),
            "stdoutSha256": digest_text(program.get("expected", "")),
            "stdout": program.get("expected", ""),
        } for program in DISPLAYED},
        "downloadedPrograms": {download["key"]: {
            "file": download["file"],
            "sourceSha256": download["sha256"],
            "executed": bool(download.get("executed")),
            "produces": download["produces"],
            "producedSha256": download.get("producedSha256"),
            "reproducedFrozenFileExactly": download.get("producedSha256")
                                           == digest_bytes(download["frozen"].read_bytes()),
        } for download in DOWNLOADS},
        "oracles": oracle_count,
        "notes": [
            "No isolated virtual environment was needed: both packet programs resolve against the shared "
            "scratch/lesson-tools runtime, whose versions are recorded above. Nothing was installed anywhere.",
            "The displayed drivers exist because the frozen file's own __main__ block writes a 60 KB JSON "
            "file. The algorithm above each driver is extracted, not retyped.",
            "No network access was used by any program.",
        ],
        "limits": [
            "Floating-point results can differ on other library versions; the resolved versions are recorded.",
            "Byte-identical regeneration of the two frozen results files depends on those exact versions, and "
            "on NumPy's default_rng streams for seeds 41, 44 and 23.",
            "The banknote counts are one small nested sequence against one development set, not a benchmark.",
        ],
        "passed": not failures,
    }, indent=2) + "\n"
    if keep_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")

    shutil.rmtree(WORKSPACE, ignore_errors=True)

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        raise SystemExit(f"{len(failures)} of {oracle_count} program oracles failed")

    print(f"PASS: {len(DISPLAYED)} displayed programs assembled from byte-exact slices of the frozen packet and "
          f"executed, {len(DOWNLOADS)} whole programs run and each reproducing its frozen results file byte for "
          f"byte ({oracle_count} oracle assertions); module "
          f"{'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
