"""Execute the problem-formulation lesson's programs and record what they printed.

Nothing here is transcribed. Both displayed programs are lifted byte for byte
out of the FROZEN MANUSCRIPT's own fenced code blocks, located by block index
and pinned by the SHA-256 of the extracted bytes and of the manuscript itself.
If the packet changes, the pin fails and the change has to be read rather than
absorbed.

Three programs, in two groups.

  DISPLAYED (two): the as-known calibration lookup of section 3 and the complete
  fixed experiment of section 5. Each is a WHOLE fenced block -- no imports are
  composed here and no driver is added -- so "verbatim" needs no qualification:
  the bytes executed are the bytes the page shows, and the evidence file records
  the digest of both. The second reads the dataset this lesson serves, so it is
  run in a scratch workspace with the SERVED copy beside it rather than the
  packet's, which makes a served-asset regression fail here as well.

  DOWNLOADED (one): formulation-calculations.py is offered whole. It is not
  displayed, so the strongest available statement is checked instead: copied
  into a scratch workspace beside the served dataset and run there, it must
  reproduce the packet's calculated-inputs.json BYTE FOR BYTE. Nothing inside
  the packet directory is written, and that is checked by fingerprinting the
  whole directory before and after.

`--write` regenerates src/learn/data/formulation-examples.js from what actually
ran. Without it the recorded text must match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-formulation-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-formulation-examples.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/ml-problem-formulation-baselines-data-leakage"
MANUSCRIPT = PACKET / "lesson.md"
PACKET_RESULTS = PACKET / "calculated-inputs.json"
ASSET_DIR = ROOT / "public/learn-assets/problem-formulation"
ASSET_DATASET = ASSET_DIR / "bank-additional.csv"
ASSET_PROGRAM = ASSET_DIR / "formulation-calculations.py"
MODULE = ROOT / "src/learn/data/formulation-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/formulation-native.json"
WORKSPACE = ROOT / "scratch/formulation/programs"

MANUSCRIPT_SHA = "1ff63549d7055024cdc8a4a982d1af05b6c575e305b824f88f90d8f0642b3139"
CALCULATIONS_SHA = "7a5f34931369f274e2de608c92c269fa40f0f215012a00409aadc0dace85dda0"
PACKET_RESULTS_SHA = "d694b6546ba00e7bc6d4592d21e973a4f7faa3d81551b252762b5b7d26b405a7"

DISPLAYED = [
    {
        "key": "latest-known",
        "block": 0,
        "file": "latest_known.py",
        "sha256": "fb9be4a437ef9d61386ae35543ae5509632d15bc8f452fb5968f9bef6bece4f9",
        "needsDataset": False,
        "title": "Which calibration was knowable at the cutoff?",
        "question": "one sensor, three of its own records and one belonging to another sensor. The newest event "
                    "has not arrived yet and one earlier event has been revised. Before reading the output, "
                    "decide which value a prediction at time 5 is entitled to, and which at 7 and 9.",
        "reads": "nothing; the history is four literal records",
    },
    {
        "key": "experiment",
        "block": 1,
        "file": "pre_call_experiment.py",
        "sha256": "dc239183325293cfe1c6b1712b55afb861a88783f67effe4c7d787827bd49950",
        "needsDataset": True,
        "title": "The same rows, scored three ways",
        "question": "a constant baseline, a recorded-feature pipeline, and that same pipeline with the final "
                    "call duration added. Which of the four printed numbers will separate the last one from "
                    "the middle one, and which will barely move?",
        "reads": "bank-additional.csv, the file served beside this lesson",
    },
]

write = "--write" in sys.argv
keep_evidence = "--no-evidence" not in sys.argv

oracle_count = 0
failures: list[str] = []


def oracle(condition, label):
    global oracle_count
    oracle_count += 1
    if not condition:
        failures.append(label)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fenced_blocks(text: str) -> list[str]:
    """Every ~~~python block in the manuscript, in document order."""
    return re.findall(r"~~~python\n(.*?)~~~", text, re.S)


def fingerprint_directory(directory: Path) -> dict[str, str]:
    return {entry.name: digest_bytes(entry.read_bytes())
            for entry in sorted(directory.iterdir()) if entry.is_file()}


def run_program(path: Path, cwd: Path) -> tuple[str, int]:
    completed = subprocess.run(
        [sys.executable, str(path)], cwd=str(cwd), capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=900)
    if completed.returncode != 0:
        raise SystemExit(f"{path.name} exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout.replace("\r\n", "\n").rstrip("\n"), completed.returncode


def js_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_module(programs: dict, environment: dict, downloads: dict) -> str:
    out = []
    add = out.append
    add("/* GENERATED by scripts/verify-formulation-examples.py. Do not edit by hand.")
    add(" *")
    add(" * The programs this lesson displays, and the output they actually printed.")
    add(" *")
    add(" * Each `code` field is a whole fenced block lifted byte for byte out of the")
    add(" * frozen manuscript at")
    add(" * docs/teaching/drafts/ml-problem-formulation-baselines-data-leakage/lesson.md,")
    add(" * pinned by the SHA-256 recorded in `source`. No import line and no driver was")
    add(" * composed here, so the bytes a reader sees are the bytes that ran.")
    add(" *")
    add(" * `expected` is captured standard output, not a transcription.")
    add(" */")
    add("export const formulationExamples = {")
    for key, program in programs.items():
        add(f"  {js_string(key)}: {{")
        add(f"    title: {js_string(program['title'])},")
        add(f"    question: {js_string(program['question'])},")
        add(f"    file: {js_string(program['file'])},")
        add("    language: \"python\",")
        add(f"    reads: {js_string(program['reads'])},")
        add(f"    code: {js_string(program['code'])},")
        add(f"    expected: {js_string(program['expected'])},")
        add("    source: {")
        add("      manuscript: \"docs/teaching/drafts/ml-problem-formulation-baselines-data-leakage/lesson.md\",")
        add(f"      manuscriptSha256: {js_string(MANUSCRIPT_SHA)},")
        add(f"      blockIndex: {program['block']},")
        add(f"      sha256: {js_string(program['sha256'])},")
        add(f"      codeLines: {program['codeLines']},")
        add(f"      outputLines: {program['outputLines']},")
        add("    },")
        add("    environment: {")
        for name, version in environment.items():
            add(f"      {js_string(name)}: {js_string(version)},")
        add("    },")
        add("  },")
    for key, download in downloads.items():
        add(f"  {js_string(key)}: {{")
        add("    downloadOnly: true,")
        add(f"    title: {js_string(download['title'])},")
        add(f"    question: {js_string(download['question'])},")
        add(f"    file: {js_string(download['file'])},")
        add(f"    path: {js_string(download['path'])},")
        add(f"    sha256: {js_string(download['sha256'])},")
        add(f"    produces: {js_string(download['produces'])},")
        add(f"    producesSha256: {js_string(download['producesSha256'])},")
        add(f"    producesBytes: {download['producesBytes']},")
        add("    environment: {")
        for name, version in environment.items():
            add(f"      {js_string(name)}: {js_string(version)},")
        add("    },")
        add("  },")
    add("};")
    add("")
    add("export default formulationExamples;")
    add("")
    return "\n".join(out)


def write_provisional():
    """A record stamped `passed: false`, written before the first program runs.

    Writing evidence only at the end looks safe and is not: a run that fails
    leaves the PREVIOUS file on disk, still saying `passed: true`, describing
    an execution that did not happen this time. Anyone reading the directory
    then sees a green record for a red tree.
    """
    if not keep_evidence:
        return
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-formulation-examples.py",
        "status": "running",
        "note": "Provisional record written before the first program ran. If this is what is on disk, the run "
                "did not reach its end: it raised, or it was killed.",
        "passed": False,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    write_provisional()
    manuscript_bytes = MANUSCRIPT.read_bytes()
    oracle(digest_bytes(manuscript_bytes) == MANUSCRIPT_SHA,
           f"the frozen manuscript now hashes {digest_bytes(manuscript_bytes)}, not the pinned {MANUSCRIPT_SHA}; "
           "the extraction below would take different bytes")
    blocks = fenced_blocks(manuscript_bytes.decode("utf-8"))
    oracle(len(blocks) == 2, f"the manuscript displays {len(blocks)} python programs, expected 2")

    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    environment = {"python": sys.version.split()[0]}
    for name, module in (("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn")):
        import importlib
        environment[name] = importlib.import_module(module).__version__

    packet_before = fingerprint_directory(PACKET)

    programs = {}
    for program in DISPLAYED:
        code = blocks[program["block"]]
        extracted = digest_text(code)
        oracle(extracted == program["sha256"],
               f"the extracted algorithm now hashes {extracted}, not the pinned {program['sha256']} "
               f"for block {program['block']}")
        directory = WORKSPACE / program["key"]
        directory.mkdir()
        target = directory / program["file"]
        target.write_text(code, encoding="utf-8", newline="\n")
        if program["needsDataset"]:
            shutil.copyfile(ASSET_DATASET, directory / "bank-additional.csv")
        printed, _ = run_program(target, directory)
        programs[program["key"]] = {
            **program,
            "code": code,
            "expected": printed,
            "codeLines": len(code.rstrip("\n").split("\n")),
            "outputLines": len(printed.split("\n")),
        }

    # ------------------------------------------ what the two programs printed
    lookup = programs["latest-known"]["expected"].split("\n")
    oracle(len(lookup) == 4, f"the calibration program printed {len(lookup)} lines, expected 4")
    oracle(lookup[0] == "5 10",
           f"at cutoff 5 the program prints the admissible value 10, got {lookup[0]!r}")
    oracle(lookup[1] == "7 12", f"at cutoff 7 it prints the known revision 12, got {lookup[1]!r}")
    oracle(lookup[2] == "9 20", f"at cutoff 9 the delayed value has arrived, got {lookup[2]!r}")
    oracle(lookup[3] == "None",
           f"and the age-limited query prints None rather than substituting a value, got {lookup[3]!r}")
    oracle("10" in lookup[0] and "20" not in lookup[0],
           "the cutoff-5 line does not contain the value that had not arrived")

    experiment = programs["experiment"]["expected"].split("\n")
    oracle(len(experiment) == 3, f"the experiment printed {len(experiment)} lines, expected 3")
    recorded = json.loads(PACKET_RESULTS.read_text(encoding="utf-8"))
    expected_rows = [
        ("training prior", "training_prior"),
        ("candidate pre-call", "candidate_pre_call"),
        ("unavailable duration", "unavailable_duration"),
    ]
    # Parsed by FIELD NAME, not by scanning for numbers: "top-50" contains a
    # hyphen and a number, and a bare number sweep read it as the value -50.
    printed = re.compile(r"^(?P<name>.+?) AP (?P<ap>[0-9.]+) log loss (?P<loss>[0-9.]+) "
                         r"correct (?P<correct>[0-9]+) top-50 positives (?P<found>[0-9]+)$")
    parsed = {}
    for index, (label, key) in enumerate(expected_rows):
        line = experiment[index]
        result = recorded["results"][key]
        match = printed.match(line)
        oracle(match is not None, f"line {index} has the four labelled fields the program prints: got {line!r}")
        if match is None:
            continue
        parsed[key] = match
        oracle(match.group("name") == label, f"line {index} names the procedure {label}: got {match.group('name')!r}")
        oracle(abs(float(match.group("ap")) - round(result["averagePrecision"], 6)) < 5e-13,
               f"{label}: printed AP {match.group('ap')} against the packet's {result['averagePrecision']}")
        oracle(abs(float(match.group("loss")) - round(result["logLoss"], 6)) < 5e-13,
               f"{label}: printed log loss {match.group('loss')} against the packet's {result['logLoss']}")
        oracle(int(match.group("correct")) == result["correct"],
               f"{label}: printed correct {match.group('correct')} against the packet's {result['correct']}")
        oracle(int(match.group("found")) == result["top50Positives"],
               f"{label}: printed top-50 positives {match.group('found')} against the packet's "
               f"{result['top50Positives']}")

    # The comparison the section is FOR: a better score with a worse contract.
    oracle(len(parsed) == 3, "all three printed rows parsed, so the comparisons below have something to compare")
    if len(parsed) == 3:
        ap = {key: float(match.group("ap")) for key, match in parsed.items()}
        correct = {key: int(match.group("correct")) for key, match in parsed.items()}
        found = {key: int(match.group("found")) for key, match in parsed.items()}
        oracle(ap["unavailable_duration"] > ap["candidate_pre_call"] > ap["training_prior"],
               "the printed average precision rises from baseline to candidate to the unavailable-duration model")
        oracle(correct["candidate_pre_call"] < correct["training_prior"],
               "while the candidate's printed correct count is BELOW the constant baseline's, which is the "
               "contrast the section turns on")
        oracle(found["candidate_pre_call"] > found["training_prior"],
               "and its top-50 count is well above it, so the two measures disagree in the printed output "
               "itself rather than only in the prose")

    # ------------------------------------------------- the downloaded program
    oracle(digest_bytes(ASSET_PROGRAM.read_bytes()) == CALCULATIONS_SHA,
           "the served calculation program is the packet's, byte for byte")
    download_directory = WORKSPACE / "calculations"
    download_directory.mkdir()
    shutil.copyfile(ASSET_PROGRAM, download_directory / "formulation-calculations.py")
    shutil.copyfile(ASSET_DATASET, download_directory / "bank-additional.csv")
    run_program(download_directory / "formulation-calculations.py", download_directory)
    produced = (download_directory / "calculated-inputs.json").read_bytes()
    frozen = PACKET_RESULTS.read_bytes()
    oracle(digest_bytes(frozen) == PACKET_RESULTS_SHA,
           f"the packet's results file now hashes {digest_bytes(frozen)}, not the pinned {PACKET_RESULTS_SHA}")
    oracle(produced == frozen,
           f"a fresh run reproduces the packet's calculated-inputs.json byte for byte; got "
           f"{len(produced)} bytes hashing {digest_bytes(produced)} against {len(frozen)} bytes hashing "
           f"{digest_bytes(frozen)}")

    packet_after = fingerprint_directory(PACKET)
    oracle(packet_before == packet_after,
           "nothing inside the content packet was written; every program ran in a scratch workspace")

    downloads = {
        "calculations": {
            "title": "The complete calculation behind every measured number on this page",
            "question": "the two fixed splits, both logistic fits, the constant baseline, the six calibration "
                        "fixtures and the action-set fixtures, in one run. Does it still produce exactly the "
                        "recorded results?",
            "file": "formulation-calculations.py",
            "path": "/learn-assets/problem-formulation/formulation-calculations.py",
            "sha256": CALCULATIONS_SHA,
            "produces": "calculated-inputs.json",
            "producesSha256": digest_bytes(produced),
            "producesBytes": len(produced),
        },
    }

    module = build_module(programs, environment, downloads)
    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        oracle(MODULE.exists(), "the examples module exists; rerun with --write if not")
        if MODULE.exists():
            oracle(MODULE.read_text(encoding="utf-8") == module,
                   "the recorded module matches a fresh execution byte for byte")

    oracle(oracle_count >= 30, f"only {oracle_count} oracles ran; the suite has lost coverage")

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-formulation-examples.py",
        "verifierSha256": digest_bytes(Path(__file__).read_bytes()),
        "mode": "write" if write else "read-only",
        "environment": environment,
        "manuscriptSha256": digest_bytes(manuscript_bytes),
        "fencedBlocksFound": len(blocks),
        "displayed": [
            {
                "key": program["key"],
                "file": program["file"],
                "blockIndex": program["block"],
                "extractedSha256": program["sha256"],
                "extractedLines": programs[program["key"]]["codeLines"],
                "composedLines": 0,
                "outputLines": programs[program["key"]]["outputLines"],
                "readsServedDataset": program["needsDataset"],
            } for program in DISPLAYED
        ],
        "downloads": [
            {
                "file": "formulation-calculations.py",
                "sha256": CALCULATIONS_SHA,
                "reproduces": "docs/teaching/drafts/ml-problem-formulation-baselines-data-leakage/"
                              "calculated-inputs.json",
                "reproducedSha256": digest_bytes(produced),
                "byteIdentical": produced == frozen,
            },
        ],
        "packetUnchanged": packet_before == packet_after,
        "oracles": oracle_count,
        "scope": "Both displayed programs are whole fenced blocks extracted from the frozen manuscript, pinned "
                 "by SHA-256, written to a scratch workspace and executed with this repository's shared "
                 "interpreter. Nothing is composed and nothing is transcribed: the recorded output is captured "
                 "standard output. The experiment reads the dataset SERVED by this lesson rather than the "
                 "packet's copy, so a served-asset regression fails here too. The downloadable calculation "
                 "program is run the same way and must reproduce the packet's calculated-inputs.json byte for "
                 "byte. The content packet directory is fingerprinted before and after and must be unchanged.",
        "limitations": [
            "One interpreter, one platform, one set of library versions, all recorded above. Reproducing these "
            "numbers on a different BLAS or a different scikit-learn is not claimed.",
            "This verifier executes programs and compares their output with the packet. Whether the page "
            "DISPLAYS that code and output is checked by scripts/verify-formulation-browser.cjs.",
        ],
        "passed": not failures,
    }, indent=2) + "\n"
    if keep_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(f"{len(failures)} of {oracle_count} program oracles failed")

    print(f"PASS: {oracle_count} oracles; 2 displayed programs extracted verbatim from the frozen manuscript "
          f"({sum(programs[key]['codeLines'] for key in programs)} lines, 0 composed) and executed, "
          f"1 downloadable program reproducing the packet's {len(produced):,}-byte results file exactly, "
          f"module {'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
