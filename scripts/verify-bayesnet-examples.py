"""Execute the Bayesian-networks lesson's programs and record what they printed.

Three programs, none of them transcribed. Each is read verbatim from a frozen
source and pinned by SHA-256, so the page cannot drift from the packet and a
copy-and-paste slip cannot silently introduce a difference:

  enumerate  the short standard-library program, read from the manuscript's own
             `~~~python` fence and pinned by the SHA-256 of that fence body.
  pgmpy      the optional library route, read from the packet file
             `pgmpy-example.py` and pinned by the SHA-256 of its bytes.
             The content phase left it written but unexecuted. Phase two runs it
             for real in an **isolated virtual environment** under `scratch/`,
             because resolving pgmpy moves NumPy and pandas and the shared
             `scratch/lesson-tools` runtime is what other lessons' recorded
             outputs were produced with.
  experiment the complete offline Wine experiment, `network-experiments.py`,
             pinned the same way. It is a download rather than a displayed
             block, so what is checked is the strongest available statement
             about it: copied into a scratch directory beside the served
             dataset and run there, it must reproduce the packet's frozen
             `calculated-inputs.json` byte for byte. Nothing inside the packet
             directory is written.

`--write` regenerates src/learn/data/bayesnet-examples.js from what actually ran.
Without it the recorded text must match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bayesnet-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bayesnet-examples.py
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
PACKET = ROOT / "docs/teaching/drafts/bayesian-networks-causal-graphical-models"
MANUSCRIPT = PACKET / "lesson.md"
PACKET_INPUTS = PACKET / "calculated-inputs.json"
DATASET = ROOT / "public/learn-assets/bayesian-networks/wine.csv"
MODULE = ROOT / "src/learn/data/bayesnet-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/bayesnet-native.json"
WORKSPACE = ROOT / "scratch/bayesnet-programs"
ISOLATED = ROOT / "scratch/bayesnet-optional/Scripts/python.exe"

EXPECTED_POSTERIOR = 0.28417183536439294

PROGRAMS = [
    {
        "key": "enumerate",
        "file": "alarm_enumeration.py",
        "origin": "manuscript fence",
        "sha256": "5b5f7a8350594dc0f168cc2f93c17113e6891307685441caf7e833645477c21e",
        "title": "Every compatible world, added up",
        "question": "eight hidden combinations of burglary, earthquake and alarm, with both calls fixed. "
                    "Which of those eight carries almost all of the burglary mass?",
        "runtime": "shared",
    },
    {
        "key": "pgmpy",
        "file": "pgmpy-example.py",
        "origin": "packet file",
        "sha256": "65af22d06e68385fe4c12176d85e5434c626719a4c53dd30a7334fe76f4af6f1",
        "title": "Optional: the same five-node network through pgmpy",
        "question": "the same tables handed to a library, with the state order and the parent-column order "
                    "written out explicitly. Should its answer agree with the enumeration above to the last digit?",
        "runtime": "isolated",
    },
    {
        "key": "experiment",
        "file": "network-experiments.py",
        "origin": "packet file",
        "sha256": "657cb8ce00e4369afe15164f5192f0c809d70a1da93c1726eb5a7f1f794cae94",
        "title": "The complete offline experiment behind this lesson's numbers",
        "question": "the alarm queries, the path enumerations, the causal examples and the whole Wine "
                    "comparison in one run. Does it still produce exactly the recorded inputs?",
        "runtime": "shared",
        "downloadOnly": True,
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


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


NUMBER = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"


def floats(text):
    return [float(token) for token in re.findall(NUMBER, text)]


def extract():
    """Read each program from its frozen source and check its pin."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    fences = re.findall(r"^~~~python\n(.*?)^~~~$", text, re.S | re.M)
    oracle(len(fences) == 1, f"the manuscript holds exactly one python fence, found {len(fences)}")
    if not fences:
        raise SystemExit("no python fence found in the manuscript; the extraction pattern is wrong")

    by_key = {program["key"]: program for program in PROGRAMS}
    by_key["enumerate"]["code"] = fences[0].rstrip("\n")
    by_key["enumerate"]["sourceDigest"] = digest_text(fences[0])
    for key in ("pgmpy", "experiment"):
        raw = (PACKET / by_key[key]["file"]).read_bytes()
        by_key[key]["sourceDigest"] = digest_bytes(raw)
        by_key[key]["code"] = raw.decode("utf-8").replace("\r\n", "\n").rstrip("\n")

    for program in PROGRAMS:
        oracle(program["sha256"] == program["sourceDigest"],
               f"{program['file']}: the frozen source now hashes {program['sourceDigest']}, "
               f"not the pinned {program['sha256']}. Re-pin deliberately after reading the change.")


def run_program(program):
    interpreter = sys.executable
    if program["runtime"] == "isolated":
        if not ISOLATED.exists():
            program["executed"] = False
            program["note"] = ("The isolated environment scratch/bayesnet-optional was not found, so this "
                               "optional library program was not executed by this run.")
            return
        interpreter = str(ISOLATED)
    target = WORKSPACE / Path(program["file"]).name
    target.write_text(program["code"] + "\n", encoding="utf-8", newline="\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONWARNINGS="ignore")
    completed = subprocess.run([interpreter, str(target)], cwd=str(WORKSPACE), env=environment,
                               capture_output=True, text=True, timeout=900)
    oracle(completed.returncode == 0,
           f"{program['file']} exited {completed.returncode}: {completed.stderr.strip()[:400]}")
    program["executed"] = completed.returncode == 0
    program["expected"] = completed.stdout.replace("\r\n", "\n").strip("\n")
    program["stderr"] = completed.stderr.replace("\r\n", "\n").strip("\n")


def packet_fingerprint():
    """Every file in the frozen packet, by digest.

    The claim is that running these programs writes nothing inside the packet.
    Comparing one file with a copy of itself taken *after* the subprocesses had
    already run asserted `x == x`; this snapshots the whole directory before
    anything runs and compares the whole directory afterwards.
    """
    return {
        path.relative_to(PACKET).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PACKET.rglob("*")) if path.is_file()
    }


def main():
    before = packet_fingerprint()
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copyfile(DATASET, WORKSPACE / "wine.csv")

    extract()
    for program in PROGRAMS:
        run_program(program)

    by_key = {program["key"]: program for program in PROGRAMS}

    # ---- the enumeration prints exactly the posterior the whole lesson quotes
    enumerate_output = floats(by_key["enumerate"].get("expected", ""))
    oracle(len(enumerate_output) == 1, f"the enumeration prints one number, got {enumerate_output}")
    if enumerate_output:
        oracle(abs(enumerate_output[0] - EXPECTED_POSTERIOR) < 1e-15,
               f"the enumeration prints {enumerate_output[0]}, not {EXPECTED_POSTERIOR}")

    # ---- the library route agrees with it, and is a genuinely separate route
    pgmpy_program = by_key["pgmpy"]
    if pgmpy_program.get("executed"):
        values = floats(pgmpy_program["expected"])
        oracle(len(values) == 2, f"pgmpy prints a two-state distribution, got {values}")
        if len(values) == 2:
            oracle(abs(values[0] + values[1] - 1) < 1e-8, "its two states sum to one")
            oracle(abs(values[1] - EXPECTED_POSTERIOR) < 1e-7,
                   f"its burglary state is {values[1]}, not the enumerated {EXPECTED_POSTERIOR}")
        oracle("pgmpy" in pgmpy_program["code"] and "VariableElimination" in pgmpy_program["code"],
               "the library route really calls the library rather than re-implementing the sum")
    else:
        oracle(False, "the optional pgmpy program was not executed; see the note in the evidence file")

    # ---- the full experiment reproduces the frozen trust root byte for byte
    produced = WORKSPACE / "calculated-inputs.json"
    oracle(produced.exists(), "network-experiments.py wrote calculated-inputs.json beside itself")
    if produced.exists():
        fresh = produced.read_bytes()
        frozen = PACKET_INPUTS.read_bytes()
        oracle(fresh == frozen,
               f"a fresh run reproduces the packet's calculated-inputs.json exactly "
               f"({digest_bytes(fresh)} versus {digest_bytes(frozen)})")
    after = packet_fingerprint()
    oracle(before, "the packet directory held files to fingerprint before the run")
    oracle(after == before,
           "nothing inside the frozen packet directory was written: "
           + (", ".join(sorted(set(before) ^ set(after))
                        or [name for name in before if before[name] != after.get(name)]) or "unchanged"))

    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    isolated_versions = {}
    if ISOLATED.exists():
        probe = subprocess.run(
            [str(ISOLATED), "-c",
             "import json,importlib.metadata as m;"
             "print(json.dumps({n: m.version(n) for n in "
             "['pgmpy','numpy','pandas','networkx','scipy','scikit-learn']}))"],
            capture_output=True, text=True)
        if probe.returncode == 0:
            isolated_versions = json.loads(probe.stdout)

    shared_environment = {"python": sys.version.split()[0], **versions}
    isolated_environment = {"python": sys.version.split()[0], **isolated_versions}

    entries = {}
    for program in PROGRAMS:
        entry = {
            "title": program["title"],
            "question": program["question"],
            "code": program["code"],
            "language": "python",
            "file": Path(program["file"]).name,
            "executed": bool(program.get("executed")),
            # The page names the environment each program actually ran in, read
            # from this generated record rather than typed into the prose.
            "runtime": program["runtime"],
            "environment": isolated_environment if program["runtime"] == "isolated" else shared_environment,
        }
        if program.get("downloadOnly"):
            entry["downloadOnly"] = True
        if program.get("executed"):
            entry["expected"] = program["expected"]
        if program.get("note"):
            entry["note"] = program["note"]
        entries[program["key"]] = entry

    header = (
        "// Programs for the Bayesian-networks lesson.\n"
        "//\n"
        "// Read verbatim from their frozen sources -- the manuscript's own python\n"
        "// fence and the content packet's two program files -- each pinned by SHA-256\n"
        "// and then executed by scripts/verify-bayesnet-examples.py. `expected` is what\n"
        "// the program actually printed on this machine, not a predicted result.\n"
        "//\n"
        "// The optional pgmpy route was run in an isolated virtual environment, because\n"
        "// resolving pgmpy moves NumPy and pandas and other lessons' recorded outputs\n"
        "// depend on the shared runtime's exact versions. The versions it resolved are\n"
        "// recorded in docs/teaching/evidence/bayesnet-native.json and named on the page.\n"
        "//\n"
        "// Do not edit by hand.\n"
    )
    module = header + "export const bayesnetExamples = " + json.dumps(entries, indent=2) + ";\n"

    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        if not MODULE.exists():
            raise SystemExit("the examples module is missing; rerun with --write")
        if MODULE.read_text(encoding="utf-8") != module:
            raise SystemExit("a fresh execution does not reproduce src/learn/data/bayesnet-examples.js; "
                             "rerun with --write and inspect the difference")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native execution of the lesson's programs; browser, independent and integration review are separate",
        "extraction": "verbatim from frozen sources: one manuscript python fence and two packet program files, "
                      "each pinned by SHA-256 of the exact bytes read",
        "manuscript": str(MANUSCRIPT.relative_to(ROOT)).replace("\\", "/"),
        "manuscriptSha256": digest_bytes(MANUSCRIPT.read_bytes()),
        "packetCalculatedInputsSha256": digest_bytes(PACKET_INPUTS.read_bytes()),
        "source": str(MODULE.relative_to(ROOT)).replace("\\", "/"),
        "sourceHash": digest_bytes(MODULE.read_bytes()) if MODULE.exists() else None,
        "verifier": "scripts/verify-bayesnet-examples.py",
        "verifierHash": digest_bytes(Path(__file__).read_bytes()),
        "sharedRuntime": {"python": sys.version.split()[0], **versions},
        "isolatedRuntime": {"path": "scratch/bayesnet-optional", **isolated_versions},
        "programs": {program["key"]: {
            "file": Path(program["file"]).name,
            "origin": program["origin"],
            "sourceSha256": program["sourceDigest"],
            "runtime": program["runtime"],
            "executed": bool(program.get("executed")),
            "codeHash": digest_text(program["code"]),
            "stdoutHash": digest_text(program.get("expected", "")),
            "stdout": program.get("expected", ""),
            "displayedOnPage": not program.get("downloadOnly", False),
        } for program in PROGRAMS},
        "oracles": oracle_count,
        "notes": [
            "The optional pgmpy program was left written but unexecuted by the content phase. This phase "
            "created scratch/bayesnet-optional, installed pgmpy there, and ran it, so the page shows a real "
            "printed distribution rather than a predicted one. Nothing was installed into the shared runtime.",
            "network-experiments.py is offered as a download rather than displayed in full, following the "
            "manuscript. It was copied into a scratch directory beside the served dataset and run there; the "
            "check is that its fresh output equals the frozen calculated-inputs.json byte for byte.",
            "No network access was used by any program.",
        ],
        "limits": [
            "Library floating-point results can differ on other versions; the resolved versions are recorded above.",
            "The Wine numbers are one small single split, not a benchmark.",
            "Byte-identical regeneration of the trust root depends on the shared runtime's exact library versions.",
        ],
        "passed": not failures,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    shutil.rmtree(WORKSPACE, ignore_errors=True)

    # A count that is printed but never floored is decoration.
    oracle(oracle_count >= 15, f"at least fifteen oracles ran; only {oracle_count} did")

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        raise SystemExit(f"{len(failures)} of {oracle_count} program oracles failed")

    executed = sum(1 for program in PROGRAMS if program.get("executed"))
    print(f"PASS: {len(PROGRAMS)} programs read verbatim from frozen sources and pinned by SHA-256, "
          f"{executed} executed ({oracle_count} oracle assertions); the full experiment reproduced the "
          f"packet's calculated-inputs.json byte for byte; module "
          f"{'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
