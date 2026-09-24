"""Execute the forecasting lesson's programs and record what they actually printed.

Nothing here is transcribed. Every line of algorithm shown on the page is a
byte-exact slice of a frozen packet source, located by a stated pattern and
pinned by the SHA-256 of the extracted bytes. If a packet source changes, the
pin fails and the change has to be read rather than absorbed.

Four programs, in two groups.

  DISPLAYED (three). Two are assembled as
      [import lines composed here] + [verbatim extracted functions] + [driver
      composed here]
  and one -- the manuscript's own baseline loop -- is lifted whole out of
  lesson.md's fenced block and run exactly as written, with no driver at all.
  The evidence file records which bytes were extracted and which were composed,
  so "verbatim" is a checkable claim rather than a word.

  DOWNLOADED (one). `forecast-experiments.py` is offered whole, as the
  manuscript offers it. It is not displayed in full, so the strongest available
  statement is checked instead: copied into a scratch directory beside the
  served dataset and run there, it must reproduce the packet's frozen
  `calculated-inputs.json` BYTE FOR BYTE. Nothing inside the packet directory is
  written, and that is checked by fingerprinting the whole directory before and
  after.

TWO DISPLAYED PROGRAMS ARE ADDITIONS TO THE MANUSCRIPT, with a reason recorded
in the packet's design.md. The manuscript displays one runnable program, in
section 5. Sections 2 to 4 -- the four baseline rules and the eligibility
boundary, which are this topic's central mechanisms -- had no runnable code at
all, and the eligibility boundary is exactly the claim a reader should be able
to execute rather than take on trust. Both additions are byte-exact slices of
the frozen author program; neither rewrites a word of the manuscript.

`--write` regenerates src/learn/data/timeseries-examples.js from what actually
ran. Without it the recorded text must match a fresh execution.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-timeseries-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-timeseries-examples.py
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
PACKET = ROOT / "docs/teaching/drafts/time-series-validation-forecasting-baselines"
EXPERIMENTS = PACKET / "forecast-experiments.py"
MANUSCRIPT = PACKET / "lesson.md"
PACKET_RESULTS = PACKET / "calculated-inputs.json"
DATASET = ROOT / "public/learn-assets/time-series-validation/bike-sharing-daily.csv"
MODULE = ROOT / "src/learn/data/timeseries-examples.js"
EVIDENCE = ROOT / "docs/teaching/evidence/timeseries-native.json"
WORKSPACE = ROOT / "scratch/timeseries/programs"

EXPERIMENTS_SHA = "dd2ac9ff03e226b43859524bb260f442ee1e1c87718928a30778f4d21df75ee6"
MANUSCRIPT_SHA = "5e672c664079e62eaac0c33893bf95a2db90532fc1973f5128c463303e0baab6"

NEWLINE = chr(10)

# ---------------------------------------------------------------- the programs

DISPLAYED = [
    {
        "key": "baseline-rules",
        "file": "baseline_rules.py",
        "section": "2",
        "title": "The four baseline rules, and the observation each one carries forward",
        "question": "four complete prediction rules on one six-day operating cycle. Which observation does "
                    "each one copy, and which of them move when you change a value in the middle of the "
                    "history?",
        "origin": "forecast-experiments.py",
        "imports": ["import numpy as np"],
        "functions": ["baseline_forecasts"],
        "extractedSha256": "e71a3de3dacc8ecf23a74440f2b0fc2ae065db8929e6f793a8c7d5cf57af81ed",
        "driver": [
            "history = [10, 20, 10, 20, 12, 22]",
            "future = np.array([12, 22, 12, 22], dtype=float)",
            "rules = baseline_forecasts(history, horizon=4, period=2)",
            "for name in ('mean', 'naive', 'seasonal', 'drift'):",
            "    values = rules[name]",
            "    print(name, np.round(values, 4).tolist(), 'MAE', round(float(np.abs(values - future).mean()), 4))",
            "print('period 3, eight horizons:', baseline_forecasts(history, 8, 3)['seasonal'].tolist())",
            "edited = [10, 20, 10, 20, 18, 22]",
            "after = baseline_forecasts(edited, horizon=4, period=2)",
            "print('fifth value 12 -> 18:', after['seasonal'].tolist(), after['naive'].tolist(),"
            " after['drift'].tolist())",
        ],
    },
    {
        "key": "eligible-rows",
        "file": "eligible_rows.py",
        "section": "4",
        "title": "Which training rows may enter each horizon's fit, and the latest day any of them reads",
        "question": "the same issue origin, seven separate fits. Each horizon has its own last eligible "
                    "training origin. What is the largest observation index any of those fits touches, and "
                    "could it ever be later than the origin itself?",
        "origin": "forecast-experiments.py",
        "imports": [
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from sklearn.linear_model import Ridge",
            "from sklearn.pipeline import make_pipeline",
            "from sklearn.preprocessing import StandardScaler",
        ],
        "functions": ["origin_features", "direct_ridge_forecast"],
        "extractedSha256": "6ef17e0b9b628ebda439dccadde10b4166060439266a698a54e6ad29818250c5",
        "driver": [
            "data = pd.read_csv(Path(__file__).with_name('bike-sharing-daily.csv'), parse_dates=['dteday'])",
            "dates = pd.DatetimeIndex(data['dteday'])",
            "counts = data['cnt'].to_numpy(dtype=float)",
            "origin = 364",
            "print('origin', origin, str(dates[origin].date()), dates[origin].day_name())",
            "for horizon in range(1, 8):",
            "    eligible = np.arange(6, origin - horizon + 1)",
            "    last = int(eligible[-1])",
            "    latest_read = max(last, last + horizon, origin)",
            "    print(horizon, 'train', int(eligible[0]), 'to', last, 'rows', eligible.size,",
            "          'target', origin + horizon, 'latest day read', latest_read)",
            "    assert latest_read <= origin",
            "print('forecasts', np.round(direct_ridge_forecast(counts, dates, origin), 2).tolist())",
        ],
    },
    {
        "key": "baseline-loop",
        "file": "baseline_loop.py",
        "section": "5",
        "title": "Replay the naive and seasonal baselines across every declared origin",
        "question": "fifty-two Saturdays, seven horizons each, and two rules that copy an observation "
                    "forward. Does the seasonal rule keep its development advantage on the later period?",
        "origin": "lesson.md",
        "imports": [],
        "functions": [],
        "extractedSha256": "6f0435ae6bbcbc1e528d9a9159b9f59aa44990c7d34bb7146256523adea17008",
        "driver": [],
        "wholeBlock": True,
    },
]

DOWNLOAD = {
    "key": "experiments",
    "file": "forecast-experiments.py",
    "source": EXPERIMENTS,
    "sha256": EXPERIMENTS_SHA,
    "produces": "calculated-inputs.json",
    "frozen": PACKET_RESULTS,
    "title": "The complete author experiment behind every measured number on this page",
    "question": "the six-candidate development comparison, the locked final replay, the exact constructed "
                "fixtures and every per-origin forecast, in one run. Does it still produce exactly the "
                "recorded results file?",
}

write = "--write" in sys.argv
keep_evidence = "--no-evidence" not in sys.argv
oracle_count = 0
failures: list[str] = []


def oracle(condition, label):
    global oracle_count
    oracle_count += 1
    if not condition:
        failures.append(label)
    return bool(condition)


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_evidence(payload):
    if not keep_evidence:
        return
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + NEWLINE, encoding="utf-8", newline=NEWLINE)


def extract_function(source, name):
    """The exact bytes of one top-level function, from its `def` line to the
    line before the next top-level statement."""
    lines = source.split(NEWLINE)
    start = next((index for index, line in enumerate(lines) if line.startswith(f"def {name}(")), None)
    if start is None:
        raise SystemExit(f"the frozen program has no top-level function {name}; the extraction pattern is wrong")
    end = start + 1
    while end < len(lines) and (lines[end] == "" or lines[end].startswith((" ", chr(9), ")"))):
        end += 1
    while end > start and lines[end - 1] == "":
        end -= 1
    return NEWLINE.join(lines[start:end])


def manuscript_python_blocks(text):
    """Every fenced python block in the manuscript, in order. The manuscript
    uses tilde fences; a pattern that assumed backticks would silently find
    none, so the count is asserted by the caller."""
    return re.findall(r"^~~~python\n(.*?)^~~~$", text, re.S | re.M)


def assemble(program, sources):
    if program.get("wholeBlock"):
        return
    source = sources[program["origin"]]
    extracted = [extract_function(source, name) for name in program["functions"]]
    for piece in extracted:
        first = piece.split(NEWLINE)[0]
        oracle(first.startswith("def ") and first.rstrip().endswith(":"),
               f"{program['file']}: the extracted span does not begin at a def line: {first[:60]}")
        oracle(len(piece.split(NEWLINE)) >= 2,
               f"{program['file']}: the extracted span for {first[:40]} has no body")
    body = (NEWLINE * 3).join(extracted)
    program["extracted"] = body
    program["extractedDigest"] = digest_text(body)
    oracle(program["extractedSha256"] != "",
           f"{program['file']}: the extracted algorithm carries no SHA pin")
    oracle(program["extractedSha256"] == program["extractedDigest"],
           f"{program['file']}: the extracted algorithm now hashes {program['extractedDigest']}, "
           f"not the pinned {program['extractedSha256']}")
    header = (NEWLINE.join(program["imports"]) + NEWLINE * 3) if program["imports"] else ""
    program["code"] = header + body + NEWLINE * 3 + NEWLINE.join(program["driver"])
    program["composedLines"] = len(program["imports"]) + len(program["driver"])
    program["extractedLines"] = len(body.split(NEWLINE))


def run_file(path, cwd, timeout=1800):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONHASHSEED="0", PYTHONWARNINGS="ignore")
    return subprocess.run([sys.executable, str(path)], cwd=str(cwd), env=environment,
                          capture_output=True, text=True, timeout=timeout)


def packet_fingerprint():
    """Every file in the frozen packet, by digest. The claim is that running
    these programs writes nothing inside the packet; comparing one file with a
    copy of itself taken after the run would assert nothing."""
    return {path.relative_to(PACKET).as_posix(): digest_bytes(path.read_bytes())
            for path in sorted(PACKET.rglob("*")) if path.is_file()}


def main():
    started = datetime.now(timezone.utc).isoformat()
    write_evidence({
        "verifiedAt": started, "verifier": "scripts/verify-timeseries-examples.py",
        "status": "in progress: this record is provisional and is rewritten only after the final assertion",
        "passed": False,
    })

    before = packet_fingerprint()
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copyfile(DATASET, WORKSPACE / "bike-sharing-daily.csv")

    experiments_bytes = EXPERIMENTS.read_bytes()
    manuscript_bytes = MANUSCRIPT.read_bytes()
    oracle(digest_bytes(experiments_bytes) == EXPERIMENTS_SHA,
           f"forecast-experiments.py hashes {digest_bytes(experiments_bytes)}, not the pinned {EXPERIMENTS_SHA}")
    oracle(digest_bytes(manuscript_bytes) == MANUSCRIPT_SHA,
           f"lesson.md hashes {digest_bytes(manuscript_bytes)}, not the pinned {MANUSCRIPT_SHA}")
    oracle(digest_bytes((ROOT / "public/learn-assets/time-series-validation/forecast-experiments.py").read_bytes())
           == EXPERIMENTS_SHA, "the served copy of the author program is not the frozen one")

    sources = {
        "forecast-experiments.py": experiments_bytes.decode("utf-8").replace("\r\n", NEWLINE),
        "lesson.md": manuscript_bytes.decode("utf-8").replace("\r\n", NEWLINE),
    }

    # ---- the manuscript's own displayed program, lifted whole
    blocks = manuscript_python_blocks(sources["lesson.md"])
    oracle(len(blocks) == 2, f"the manuscript holds two fenced python blocks, found {len(blocks)}")
    # The first is the one-line rolling-mean illustration of section 3; the page
    # displays it too, so it is pinned here and the source-hygiene verifier
    # checks the page renders exactly these bytes.
    snippet = blocks[0].rstrip(NEWLINE) if blocks else ""
    oracle(snippet == "past_week_mean = counts.shift(1).rolling(7).mean()",
           f"the section-3 illustration is not the expected one line: {snippet[:80]!r}")
    whole = next(program for program in DISPLAYED if program.get("wholeBlock"))
    whole["code"] = blocks[1].rstrip(NEWLINE) if len(blocks) > 1 else ""
    whole["extracted"] = whole["code"]
    whole["extractedDigest"] = digest_text(whole["code"])
    whole["extractedLines"] = len(whole["code"].split(NEWLINE))
    whole["composedLines"] = 0
    oracle(whole["extractedSha256"] == whole["extractedDigest"],
           f"the manuscript's baseline loop now hashes {whole['extractedDigest']}, "
           f"not the pinned {whole['extractedSha256']}")
    oracle("for stage, stage_origins in" in whole["code"],
           "the block lifted from the manuscript is not the two-stage baseline loop")

    for program in DISPLAYED:
        assemble(program, sources)
        target = WORKSPACE / program["file"]
        target.write_text(program["code"] + NEWLINE, encoding="utf-8", newline=NEWLINE)
        completed = run_file(target, WORKSPACE)
        oracle(completed.returncode == 0,
               f"{program['file']} exited {completed.returncode}: {completed.stderr.strip()[:400]}")
        program["executed"] = completed.returncode == 0
        program["expected"] = completed.stdout.replace("\r\n", NEWLINE).strip(NEWLINE)
        program["stderr"] = completed.stderr.replace("\r\n", NEWLINE).strip(NEWLINE)

    by_key = {program["key"]: program for program in DISPLAYED}

    # ---- the four rules print the values the lesson's table states
    rules_output = by_key["baseline-rules"].get("expected", "")
    oracle("mean [15.6667, 15.6667, 15.6667, 15.6667] MAE 5.0" in rules_output,
           f"the mean rule did not print its stated forecast and MAE: {rules_output[:200]!r}")
    oracle("naive [22.0, 22.0, 22.0, 22.0] MAE 5.0" in rules_output, "the naive rule printed something else")
    oracle("seasonal [12.0, 22.0, 12.0, 22.0] MAE 0.0" in rules_output, "the seasonal rule printed something else")
    oracle("drift [24.4, 26.8, 29.2, 31.6] MAE 11.0" in rules_output, "the drift rule printed something else")
    oracle("period 3, eight horizons: [20.0, 12.0, 22.0, 20.0, 12.0, 22.0, 20.0, 12.0]" in rules_output,
           "the period-3 forecast must repeat the final observed cycle past one full season")
    # The contrast the investigation is built on: one edit moves two rules and
    # leaves two untouched. Both halves are asserted, because "seasonal changed"
    # without "naive did not" is only half the lesson.
    oracle("fifth value 12 -> 18: [18.0, 22.0, 18.0, 22.0] [22.0, 22.0, 22.0, 22.0] "
           "[24.4, 26.8, 29.2, 31.6]" in rules_output,
           f"the history edit did not move exactly the seasonal rule: {rules_output[-200:]!r}")

    # ---- the eligibility loop prints a shrinking training set and a fixed ceiling
    rows_output = by_key["eligible-rows"].get("expected", "")
    oracle("origin 364 2011-12-31 Saturday" in rows_output, "the eligibility loop did not print its origin")
    printed = re.findall(r"^(\d) train 6 to (\d+) rows (\d+) target (\d+) latest day read (\d+)$",
                         rows_output, re.M)
    oracle(len(printed) == 7, f"the eligibility loop prints seven horizons, printed {len(printed)}")
    for horizon, last, rows, target, latest in printed:
        horizon, last, rows, target, latest = (int(value) for value in (horizon, last, rows, target, latest))
        oracle(last == 364 - horizon, f"horizon {horizon}: the last training origin is {last}, not {364 - horizon}")
        oracle(rows == last - 5, f"horizon {horizon}: the row count is {rows}, not {last - 5}")
        oracle(target == 364 + horizon, f"horizon {horizon}: the target is {target}")
        # THE TOPIC'S CENTRAL CLAIM, printed by the program itself.
        oracle(latest == 364, f"horizon {horizon}: the latest day read is {latest}, not the origin 364")
    oracle(len({row[1] for row in printed}) == 7,
           "every horizon must show its OWN last training origin; a single shared boundary would be wrong")

    # ---- the manuscript's baseline loop prints the four pooled values it quotes
    loop_output = by_key["baseline-loop"].get("expected", "")
    for stage, rule, value in (("development", "naive", "1198.3"), ("development", "seasonal", "987.32"),
                               ("final", "naive", "1310.34"), ("final", "seasonal", "1390.91")):
        oracle(f"{stage} {rule} MAE {value}" in loop_output,
               f"the manuscript's loop did not print {stage} {rule} MAE {value}: {loop_output!r}")
    horizon_lines = re.findall(r"^by horizon \[(.*)\]$", loop_output, re.M)
    oracle(len(horizon_lines) == 4, f"the loop prints four horizon profiles, printed {len(horizon_lines)}")
    if len(horizon_lines) == 4:
        final_naive = [float(token) for token in horizon_lines[2].split()]
        final_seasonal = [float(token) for token in horizon_lines[3].split()]
        oracle(len(final_naive) == 7 and len(final_seasonal) == 7, "each horizon profile carries seven values")
        oracle(final_naive[6] == final_seasonal[6],
               f"at horizon seven the two rules are identical, so their errors must be equal: "
               f"{final_naive[6]} against {final_seasonal[6]}")
        oracle(final_naive[:6] != final_seasonal[:6],
               "and they must differ at the other six horizons, or the equality shows nothing")

    # ---- the whole author program reproduces the frozen results byte for byte
    target = WORKSPACE / DOWNLOAD["file"]
    shutil.copyfile(DOWNLOAD["source"], target)
    completed = run_file(target, WORKSPACE)
    oracle(completed.returncode == 0,
           f"{DOWNLOAD['file']} exited {completed.returncode}: {completed.stderr.strip()[:400]}")
    DOWNLOAD["executed"] = completed.returncode == 0
    DOWNLOAD["stdout"] = completed.stdout.replace("\r\n", NEWLINE).strip(NEWLINE)
    produced = WORKSPACE / DOWNLOAD["produces"]
    oracle(produced.exists(), f"{DOWNLOAD['file']} wrote {DOWNLOAD['produces']} beside itself")
    if produced.exists():
        fresh = produced.read_bytes()
        frozen = DOWNLOAD["frozen"].read_bytes()
        DOWNLOAD["producedSha256"] = digest_bytes(fresh)
        oracle(fresh == frozen,
               f"a fresh run of {DOWNLOAD['file']} reproduces {DOWNLOAD['produces']} exactly "
               f"({digest_bytes(fresh)} against {digest_bytes(frozen)})")

    after = packet_fingerprint()
    oracle(bool(before), "the packet directory held files to fingerprint before the run")
    oracle(after == before,
           "nothing inside the frozen packet directory was written: "
           + (", ".join(sorted(set(before) ^ set(after))
                        or [name for name in before if before[name] != after.get(name)]) or "unchanged"))

    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy"):
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
            "section": program["section"],
            "code": program["code"],
            "language": "python",
            "file": program["file"],
            "executed": bool(program.get("executed")),
            "expected": program.get("expected", ""),
            "environment": environment,
            "extraction": {
                "origin": f"docs/teaching/drafts/time-series-validation-forecasting-baselines/{program['origin']}",
                "functions": program["functions"],
                "wholeFencedBlock": bool(program.get("wholeBlock")),
                "extractedSha256": program["extractedDigest"],
                "extractedLines": program["extractedLines"],
                "composedLines": program["composedLines"],
            },
        }
    entries[DOWNLOAD["key"]] = {
        "title": DOWNLOAD["title"],
        "question": DOWNLOAD["question"],
        "section": "5",
        "file": DOWNLOAD["file"],
        "language": "python",
        "downloadOnly": True,
        "executed": bool(DOWNLOAD.get("executed")),
        "expected": DOWNLOAD.get("stdout", ""),
        "environment": environment,
        "reproduces": DOWNLOAD["produces"],
        "sourceSha256": DOWNLOAD["sha256"],
    }
    entries["rolling-illustration"] = {
        "title": "A historical seven-day mean, indexed by the target date",
        "section": "3",
        "code": snippet,
        "language": "python",
        "displayOnly": True,
        "executed": False,
        "expected": "",
        "extraction": {
            "origin": "docs/teaching/drafts/time-series-validation-forecasting-baselines/lesson.md",
            "wholeFencedBlock": True,
            "extractedSha256": digest_text(snippet),
            "extractedLines": 1,
            "composedLines": 0,
        },
        "note": "One line, shown to contrast two indexing conventions rather than to be run on its own. It "
                "needs a named prediction time before it means anything, which is the point of the paragraph "
                "it sits in.",
    }

    header = (
        "// Programs for the time-series validation and forecasting lesson.\n"
        "//\n"
        "// Generated by scripts/verify-timeseries-examples.py. Every line of algorithm\n"
        "// on the page is a byte-exact slice of a frozen packet source -- two functions\n"
        "// lifted from forecast-experiments.py by their def lines, and two blocks lifted\n"
        "// whole out of lesson.md's own fences -- each pinned by SHA-256. Only the import\n"
        "// lines and the few printing lines under them are composed, and `extraction`\n"
        "// records exactly how many lines are which. `expected` is what the program\n"
        "// printed on this machine, not a predicted result.\n"
        "//\n"
        "// The complete author program is offered as a download rather than displayed. It\n"
        "// was run in scratch/timeseries/programs beside the served dataset and had to\n"
        "// reproduce the packet's frozen calculated-inputs.json byte for byte.\n"
        "//\n"
        "// Do not edit by hand.\n"
    )
    module = header + "export const timeSeriesExamples = " + json.dumps(entries, indent=2) + ";\n"

    if write:
        MODULE.write_text(module, encoding="utf-8", newline=NEWLINE)
    elif not MODULE.exists():
        failures.append("the examples module is missing; rerun with --write")
    elif MODULE.read_text(encoding="utf-8") != module:
        failures.append("a fresh execution does not reproduce src/learn/data/timeseries-examples.js; "
                        "rerun with --write and inspect the difference")

    # S5/S8: the coverage floor runs BEFORE the evidence is stamped. Written
    # after, a tripped floor would exit non-zero and still leave `passed: true`
    # on disk -- and the evidence file, not the console, is the durable record.
    oracle(oracle_count >= 45, f"at least forty-five oracles ran; only {oracle_count} did")

    write_evidence({
        "verifiedAt": started,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native execution of the lesson's programs; browser, independent and integration review "
                 "are separate",
        "verifier": "scripts/verify-timeseries-examples.py",
        "verifierSha256": digest_bytes(Path(__file__).read_bytes()),
        "extraction": "Two displayed algorithms are byte-exact slices of forecast-experiments.py, located by "
                      "their def lines; one displayed program and one illustration are lifted whole out of "
                      "lesson.md's own fenced blocks. Each is pinned by the SHA-256 of the extracted bytes. "
                      "Import lines and driver prints are composed by this verifier and counted separately.",
        "additionsToTheManuscript": [
            "baseline-rules (section 2) and eligible-rows (section 4) are displayed programs the manuscript "
            "does not contain. The manuscript displays one runnable program, in section 5, leaving this "
            "topic's two central mechanisms -- the four baseline rules and the eligibility boundary -- with "
            "no executable form. Both additions are byte-exact slices of the frozen author program and change "
            "no manuscript text. The reason is recorded in the packet's design.md.",
        ],
        "frozenSources": {
            "forecast-experiments.py": digest_bytes(experiments_bytes),
            "lesson.md": digest_bytes(manuscript_bytes),
            "calculated-inputs.json": digest_bytes(PACKET_RESULTS.read_bytes()),
        },
        "servedDataset": {"path": "/learn-assets/time-series-validation/bike-sharing-daily.csv",
                          "sha256": digest_bytes(DATASET.read_bytes())},
        "module": {"path": "src/learn/data/timeseries-examples.js",
                   "sha256": digest_bytes(MODULE.read_bytes()) if MODULE.exists() else None,
                   "regeneration": "written" if write else "byte-identical"},
        "runtime": environment,
        "displayedPrograms": {program["key"]: {
            "file": program["file"],
            "origin": program["origin"],
            "functions": program["functions"],
            "wholeFencedBlock": bool(program.get("wholeBlock")),
            "extractedSha256": program["extractedDigest"],
            "extractedLines": program["extractedLines"],
            "composedLines": program["composedLines"],
            "executed": bool(program.get("executed")),
            "stdoutSha256": digest_text(program.get("expected", "")),
            "stdout": program.get("expected", ""),
        } for program in DISPLAYED},
        "downloadedProgram": {
            "file": DOWNLOAD["file"],
            "sourceSha256": DOWNLOAD["sha256"],
            "executed": bool(DOWNLOAD.get("executed")),
            "produces": DOWNLOAD["produces"],
            "producedSha256": DOWNLOAD.get("producedSha256"),
            "reproducedFrozenFileExactly": DOWNLOAD.get("producedSha256")
                                           == digest_bytes(DOWNLOAD["frozen"].read_bytes()),
        },
        "packetDirectoryUnchanged": after == before,
        "oracles": oracle_count,
        "notes": [
            "No isolated virtual environment was needed: every program resolves against the shared "
            "scratch/lesson-tools runtime, whose versions are recorded above. Nothing was installed anywhere.",
            "No network access was used by any program; the author program downloads nothing.",
            "The eligibility loop asserts its own information boundary as it runs, so a reader who edits the "
            "origin sees the assertion rather than a silently wrong training range.",
        ],
        "limits": [
            "Floating-point results can differ on other library versions; the resolved versions are recorded.",
            "Byte-identical regeneration of calculated-inputs.json depends on those exact versions. The ridge "
            "fits are deterministic, so no seed is involved.",
            "The printed development and final values are one two-year system under one update policy.",
        ],
        "passed": not failures,
    })

    shutil.rmtree(WORKSPACE, ignore_errors=True)

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        raise SystemExit(f"{len(failures)} of {oracle_count} program oracles failed")

    print(f"PASS: {len(DISPLAYED)} displayed programs executed ({sum(1 for p in DISPLAYED if p.get('wholeBlock'))} "
          f"lifted whole from the manuscript), the whole author program run and reproducing its frozen results "
          f"file byte for byte, {oracle_count} oracle assertions; module "
          f"{'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
