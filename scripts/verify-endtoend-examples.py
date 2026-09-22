"""Execute the end-to-end lesson's displayed program and record what it actually printed.

Nothing here is transcribed. The program the page shows is a byte-exact slice of
the frozen manuscript, located by its fenced code block and pinned by the
SHA-256 of the extracted bytes. If the manuscript changes, the pin fails and the
change has to be read rather than absorbed.

What this verifier establishes.

  * The displayed program is the manuscript's program, byte for byte, and the
    file served for download beside the dataset is the same bytes again.
  * Running it beside the served copy of wine.csv reproduces the manuscript's
    recorded output block byte for byte.
  * Its printed output is split at the line where the study opens the test set.
    The development half is what the page shows beside the program; the held-out
    half is what the page shows only after the learner has committed the
    decision it reports on. The two halves concatenate back to the exact stdout,
    which is checked rather than assumed, so the split cannot quietly drop a
    line.
  * The development half contains no held-out quantity and the held-out half
    contains every one of them. This is the property the whole lesson turns on,
    so it is asserted on the actual bytes rather than trusted to the split
    index.
  * Every number the program printed agrees with the independently regenerated
    browser data module, so the page and the program cannot drift apart.

`--write` regenerates src/learn/data/endtoend-examples.js and the served
wine_study.py from what actually ran. Without it both must already match a fresh
execution.

`--isolated` additionally builds a throwaway virtual environment under
scratch/endtoend-venv, installs exactly the two pinned versions the lesson's own
setup line names, and runs the program there as well. That checks the displayed
instruction rather than assuming it. It is off by default because it downloads
packages; when it is off the evidence file records that it did not run rather
than implying that it did. The environment is isolated because a sibling agent's
install once resolved a different NumPy against the shared runtime.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-endtoend-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-endtoend-examples.py
  scratch/lesson-tools/Scripts/python.exe scripts/verify-endtoend-examples.py --isolated
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/end-to-end-supervised-learning-error-analysis"
MANUSCRIPT = PACKET / "lesson.md"
PACKET_DATASET = PACKET / "wine.csv"
ASSET_DIRECTORY = ROOT / "public/learn-assets/end-to-end"
ASSET_DATASET = ASSET_DIRECTORY / "wine.csv"
ASSET_PROGRAM = ASSET_DIRECTORY / "wine_study.py"
DATA_MODULE = ROOT / "src/learn/data/endtoend-data.js"
MODULE = ROOT / "src/learn/data/endtoend-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/endtoend-native.json"
WORKSPACE = ROOT / "scratch/endtoend-programs"
VENV = ROOT / "scratch/endtoend-venv"

# The bytes of the manuscript's one displayed Python program, and of the frozen
# manuscript itself. A changed packet fails here instead of being absorbed
# silently: the packet is an input to this lesson, not a file it may follow.
PROGRAM_SHA = "e0dcd56d83d241228d0c92f5308a102fb11ba90fbc8adfec9c4bd201714beecc"
MANUSCRIPT_SHA = "0b25eaad5fe45232c97f756222805e9222bb91659aeec08835c7a88e5e5338e0"

# The line at which the study stops being development work and opens the test
# set. Everything from here on is a held-out quantity.
HELD_OUT_MARKER = "test "

write = "--write" in sys.argv
isolated = "--isolated" in sys.argv
keep_evidence = "--no-evidence" not in sys.argv

failures: list[str] = []
oracles = 0


def oracle(condition, label):
    global oracles
    oracles += 1
    if not condition:
        failures.append(label)
    return bool(condition)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fenced(text, language):
    """Every fenced block of one language, in document order."""
    return re.findall(rf"```{language}\r?\n(.*?)```", text, re.S)


def main():
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    oracle(digest(MANUSCRIPT.read_bytes()) == MANUSCRIPT_SHA,
           f"the frozen manuscript now hashes to {digest(MANUSCRIPT.read_bytes())}, not the pinned "
           f"{MANUSCRIPT_SHA}; read the change rather than updating the pin")
    programs = fenced(manuscript, "python")
    oracle(len(programs) == 1, f"the manuscript displays {len(programs)} python programs, not one")
    if not programs:
        raise SystemExit("no displayed program to check")
    program = programs[0]
    program_bytes = program.encode("utf-8")
    extracted_digest = digest(program_bytes)
    oracle(extracted_digest == PROGRAM_SHA,
           f"the extracted program now hashes to {extracted_digest}, not the pinned {PROGRAM_SHA}")

    setups = fenced(manuscript, "sh")
    oracle(len(setups) == 1, f"the manuscript displays {len(setups)} shell setup blocks, not one")
    setup = setups[0] if setups else ""
    oracle("numpy==2.3.5" in setup and "scikit-learn==1.9.1" in setup,
           "the displayed setup names the two pinned versions the packet recorded")

    recorded_blocks = fenced(manuscript, "text")
    oracle(len(recorded_blocks) == 1, f"the manuscript records {len(recorded_blocks)} output blocks, not one")
    recorded = recorded_blocks[0].replace("\r\n", "\n").strip("\n") if recorded_blocks else ""

    # ---------------------------------------------------- run it, beside the asset
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "wine_study.py").write_bytes(program_bytes)
    oracle(ASSET_DATASET.exists(), "the lesson serves its own copy of wine.csv")
    (WORKSPACE / "wine.csv").write_bytes(ASSET_DATASET.read_bytes())
    oracle(ASSET_DATASET.read_bytes() == PACKET_DATASET.read_bytes(),
           "the served copy of wine.csv is the packet's frozen bytes")

    before = {path.name: digest(path.read_bytes()) for path in sorted(PACKET.iterdir()) if path.is_file()}
    run = subprocess.run([sys.executable, "wine_study.py"], cwd=WORKSPACE,
                         capture_output=True, text=True, timeout=600)
    after = {path.name: digest(path.read_bytes()) for path in sorted(PACKET.iterdir()) if path.is_file()}
    oracle(before == after, "running the program wrote nothing into the frozen packet directory")
    oracle(run.returncode == 0, f"the displayed program exited {run.returncode}: {run.stderr[-600:]}")
    stdout = run.stdout.replace("\r\n", "\n").strip("\n")
    oracle(stdout == recorded,
           "the displayed program's output differs from the manuscript's recorded block")

    # ------------------------------------------------- split at the held-out line
    lines = stdout.split("\n")
    marker_positions = [index for index, line in enumerate(lines) if line.startswith(HELD_OUT_MARKER)]
    oracle(len(marker_positions) == 1,
           f"the output has {len(marker_positions)} lines starting with {HELD_OUT_MARKER!r}, not one")
    cut = marker_positions[0] if marker_positions else len(lines)
    development = "\n".join(lines[:cut])
    held_out = "\n".join(lines[cut:])
    oracle(f"{development}\n{held_out}" == stdout,
           "the two halves of the recorded output do not concatenate back to what the program printed")

    # The property the lesson turns on, asserted on the bytes.
    held_out_numbers = ["0.966667", "0.972222", "0.128708", "[[12", "14", " 9]]"]
    for token in held_out_numbers:
        oracle(token in held_out, f"the held-out half is missing {token!r}")
    for token in ("0.966667", "0.972222", "0.128708"):
        oracle(token not in development,
               f"the development half leaks the held-out quantity {token!r}")
    oracle("[[" not in development, "the development half leaks the held-out confusion matrix")
    oracle("selected linear_three" in development,
           "the selection, which is development evidence, stays in the development half")
    for token in ("0.792857", "0.888889", "0.820635"):
        oracle(token in development, f"the development half is missing the validation score {token!r}")

    # --------------------------------- the program and the browser data agree
    data_text = DATA_MODULE.read_text(encoding="utf-8")
    data = json.loads(data_text.split("export const endToEndData = ", 1)[1].rsplit(";", 1)[0])
    by_key = {candidate["key"]: candidate for candidate in data["candidates"]}
    for line in development.split("\n"):
        parts = line.split()
        if len(parts) == 4 and parts[0] in by_key:
            candidate = by_key[parts[0]]
            oracle(f"{candidate['validationBalancedAccuracy']['value']:.6f}" == parts[1],
                   f"{parts[0]}: the program printed balanced accuracy {parts[1]}, the data module holds "
                   f"{candidate['validationBalancedAccuracy']['value']:.6f}")
            oracle(f"{candidate['validationAccuracy']['value']:.6f}" == parts[2],
                   f"{parts[0]}: the program printed accuracy {parts[2]}")
            oracle(f"{candidate['validationLogLoss']['value']:.6f}" == parts[3],
                   f"{parts[0]}: the program printed log loss {parts[3]}")
        if parts[:1] == ["paired"]:
            paired = data["paired"][parts[1]]
            oracle(len(paired["repairedIds"]) == int(parts[2]) and len(paired["brokenIds"]) == int(parts[3]),
                   f"paired {parts[1]}: the program printed {parts[2]}/{parts[3]}, the data module holds "
                   f"{len(paired['repairedIds'])}/{len(paired['brokenIds'])}")
        if parts[:1] == ["slice"]:
            recorded_slice = data["colourSliceReference"]["slices"][parts[1]][parts[2]]
            oracle(recorded_slice["n"] == int(parts[3]) and recorded_slice["errors"] == int(parts[4]),
                   f"slice {parts[1]} {parts[2]}: the program printed {parts[3]}/{parts[4]}")
        if parts[:1] == ["selected"]:
            oracle(parts[1] == data["heldOut"]["selected"],
                   f"the program selected {parts[1]}, the data module records {data['heldOut']['selected']}")
        if parts[:1] == ["split"]:
            oracle([int(value) for value in parts[2:]]
                   == [data["contract"]["trainRows"], data["contract"]["validationRows"],
                       data["contract"]["testRows"]],
                   "the program's split sizes differ from the contract in the data module")
    held_parts = held_out.split("\n")[0].split()
    oracle(f"{data['heldOut']['balancedAccuracy']['value']:.6f}" == held_parts[1],
           "the program's held-out balanced accuracy differs from the data module")
    oracle(f"{data['heldOut']['accuracy']['value']:.6f}" == held_parts[2],
           "the program's held-out accuracy differs from the data module")
    oracle(f"{data['heldOut']['logLoss']['value']:.6f}" == held_parts[3],
           "the program's held-out log loss differs from the data module")

    # ------------------------------------------------- the optional isolated run
    isolated_record = None
    if isolated:
        shutil.rmtree(VENV, ignore_errors=True)
        venv.EnvBuilder(with_pip=True, clear=True).create(VENV)
        interpreter = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = subprocess.run(
            [str(interpreter), "-m", "pip", "install", "--quiet", "numpy==2.3.5", "scikit-learn==1.9.1"],
            capture_output=True, text=True, timeout=1800)
        oracle(install.returncode == 0,
               f"the lesson's displayed setup line failed in a clean environment: {install.stderr[-600:]}")
        resolved = subprocess.run(
            [str(interpreter), "-c",
             "import json,sys,numpy,sklearn,scipy;"
             "print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,"
             "'sklearn':sklearn.__version__,'scipy':scipy.__version__}))"],
            capture_output=True, text=True, timeout=600)
        versions = json.loads(resolved.stdout) if resolved.returncode == 0 else {}
        oracle(versions.get("numpy") == "2.3.5" and versions.get("sklearn") == "1.9.1",
               f"the isolated environment resolved {versions}, not the two pinned versions")
        isolated_run = subprocess.run([str(interpreter), "wine_study.py"], cwd=WORKSPACE,
                                      capture_output=True, text=True, timeout=1800)
        isolated_stdout = isolated_run.stdout.replace("\r\n", "\n").strip("\n")
        oracle(isolated_run.returncode == 0,
               f"the program exited {isolated_run.returncode} in the isolated environment")
        oracle(isolated_stdout == recorded,
               "the program prints something different in an environment built from its own displayed setup line")
        isolated_record = {
            "environment": "scratch/endtoend-venv, built by this run and installed from the lesson's own "
                           "displayed setup line",
            "resolvedVersions": versions,
            "reproducedRecordedOutput": isolated_stdout == recorded,
            "stdoutSha256": digest(isolated_stdout.encode("utf-8")),
        }

    # --------------------------------------------------- the served program file
    if write:
        ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
        ASSET_PROGRAM.write_bytes(program_bytes)
    else:
        oracle(ASSET_PROGRAM.exists() and ASSET_PROGRAM.read_bytes() == program_bytes,
               "the served wine_study.py is not the manuscript's program byte for byte")

    # ------------------------------------------------------------- the module
    payload = {
        "study": {
            "key": "study",
            "title": "The complete offline study",
            "file": "wine_study.py",
            "language": "python",
            "question": "Which candidate does the declared metric choose, and what did the additional "
                        "measurement actually change?",
            "setup": setup.replace("\r\n", "\n").strip("\n"),
            "code": program.replace("\r\n", "\n").strip("\n"),
            "developmentOutput": development,
            "heldOutOutput": held_out,
            "download": "/learn-assets/end-to-end/wine_study.py",
            "dataset": "/learn-assets/end-to-end/wine.csv",
            "extractedSha256": extracted_digest,
            "stdoutSha256": digest(stdout.encode("utf-8")),
        },
    }
    header = (
        "// The program the end-to-end lesson displays, and what it actually printed.\n"
        "//\n"
        "// Generated by scripts/verify-endtoend-examples.py. `code` is a byte-exact slice\n"
        "// of the frozen manuscript's one fenced python block; `developmentOutput` and\n"
        "// `heldOutOutput` are the two halves of a real execution's stdout, split at the\n"
        "// line where the study opens the test set, and they concatenate back to it.\n"
        "//\n"
        "// The page shows `developmentOutput` beside the program and `heldOutOutput` only\n"
        "// after the learner has committed the decision it reports on.\n"
        "//\n"
        "// Do not edit by hand.\n"
    )
    module_text = header + "export const endToEndExamples = " + json.dumps(payload, indent=2) + ";\n"
    if write:
        MODULE.write_text(module_text, encoding="utf-8", newline="\n")
    else:
        existing = MODULE.read_text(encoding="utf-8") if MODULE.exists() else None
        oracle(existing == module_text,
               "src/learn/data/endtoend-examples.js is not byte-identical to a fresh execution; "
               "re-run with --write and read the difference")

    environment = subprocess.run(
        [sys.executable, "-c",
         "import json,sys,numpy,sklearn,scipy;"
         "print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,"
         "'sklearn':sklearn.__version__,'scipy':scipy.__version__}))"],
        capture_output=True, text=True, timeout=600)

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-endtoend-examples.py",
        "verifierSha256": digest(Path(__file__).read_bytes()),
        "manuscriptSha256": digest(MANUSCRIPT.read_bytes()),
        "displayedProgram": {
            "source": "the manuscript's single fenced python block, extracted verbatim",
            "extractedSha256": extracted_digest,
            "extractedLines": len(program.strip("\n").split("\n")),
            "composedLines": 0,
            "servedAt": "/learn-assets/end-to-end/wine_study.py",
            "executed": run.returncode == 0,
            "stdoutSha256": digest(stdout.encode("utf-8")),
            "reproducedRecordedBlock": stdout == recorded,
            "developmentLines": len(development.split("\n")),
            "heldOutLines": len(held_out.split("\n")),
        },
        "heldOutSeparation": {
            "marker": HELD_OUT_MARKER,
            "developmentHalfContainsNoHeldOutQuantity": True,
            "note": "Checked on the actual output bytes: the three held-out scores and the held-out confusion "
                    "matrix appear in the held-out half and in no part of the half the page shows beside the "
                    "program.",
        },
        "isolatedRun": isolated_record,
        "servedDataset": {"path": "/learn-assets/end-to-end/wine.csv",
                          "sha256": digest(ASSET_DATASET.read_bytes())},
        "module": {"path": "src/learn/data/endtoend-examples.js",
                   "sha256": digest(module_text.encode("utf-8")),
                   "regeneration": "written" if write else "byte-identical"},
        "runtime": json.loads(environment.stdout) if environment.returncode == 0 else None,
        "oracles": oracles,
        "notes": [
            "The program is displayed whole, exactly as the manuscript displays it. Its printed output is "
            "split for presentation only; the split is checked to reconstruct the stdout exactly.",
            "No network access was used by the program itself: the dataset it reads is the file served beside "
            "it.",
            "Nothing inside the frozen packet directory was written; the directory was fingerprinted before "
            "and after the run.",
        ],
        "limits": [
            "Floating-point results can differ on other library versions; the resolved versions are recorded.",
            "Without --isolated no clean-environment installation was attempted, and isolatedRun is null "
            "rather than a claim.",
            "This verifier executes a program and compares text. What the page renders is checked by "
            "scripts/verify-endtoend-browser.cjs.",
        ],
        "passed": not failures,
    }, indent=2) + "\n"
    if keep_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")

    shutil.rmtree(WORKSPACE, ignore_errors=True)

    # A floor under the counter. Five of the oracle groups pattern-match scraped
    # program text; if the print format changed they would all match zero lines,
    # run zero oracles and still print PASS. The floor sits just under today's
    # count so that outcome fails loudly instead.
    oracle(oracles >= 50, f"only {oracles} program oracles ran; the suite has lost coverage")

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        raise SystemExit(f"{len(failures)} of {oracles} program oracles failed")

    print(f"PASS: the manuscript's displayed program extracted verbatim ({extracted_digest[:12]}...), executed, "
          f"and reproducing its recorded output block byte for byte; its output split into "
          f"{len(development.split(chr(10)))} development and {len(held_out.split(chr(10)))} held-out lines with "
          f"no held-out quantity in the development half; {oracles} oracle assertions; module "
          f"{'written' if write else 'byte-identical'}"
          + (f"; isolated environment resolved {isolated_record['resolvedVersions']}" if isolated_record else
             "; isolated environment not built (pass --isolated)") + ".")


if __name__ == "__main__":
    main()
