"""Bind the completed, explicitly reviewed Survival sources and evidence.

This records completed checks; it does not rerun them or manufacture a review.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/teaching/evidence"
ARCHIVE = ROOT / "docs/teaching/archive/survival-author-freeze"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


reading = read("scratch/survival/browser/final-reading-results.json")
behavior = read("scratch/survival/browser/results.json")
native = read("scratch/survival/native-verification/results.json")
execution = read("scratch/survival/program-execution.json")
figures = read("scratch/survival-figures/preflight.json")
paint = read("scratch/survival/browser/cox-paint-proof.json")
assert reading["passed"] and not reading["errors"]
assert not behavior["errors"] and not behavior["failedRequests"]
assert native["totalFiniteFixtures"] == 121 and len(native["invalidContracts"]) == 13
assert execution["programs"] == 17
source_hashes = {file: digest(ROOT / file) for file in reading["sourceHashes"]}
assert len(source_hashes) == 8
amendments = []
for file, tested_hash in reading["sourceHashes"].items():
    if source_hashes[file] != tested_hash:
        assert file.endswith("/survival-analysis-cox-regression-kaplan-meier-hazard-models.jsx")
        amendment = read("docs/teaching/evidence/survival-integration-amendment.json")
        assert amendment["beforeSha256"] == tested_hash
        assert amendment["afterSha256"] == source_hashes[file]
        assert digest(ROOT / amendment["archive"]) == tested_hash
        amendments.append({"file": file, "before": tested_hash, "after": source_hashes[file], "owner": "root", "record": "docs/teaching/evidence/survival-integration-amendment.json", "changes": amendment["changes"], "verification": amendment["conservation"], "previousSource": amendment["archive"]})
for file, value in native["sourceHashes"].items():
    assert source_hashes[file] == value
assert execution["sha256"] == source_hashes["src/learn/data/survival-examples.js"]
for entry in figures["sources"]:
    assert source_hashes[entry["path"]] == entry["sha256"]
for row in behavior["records"]:
    assert len(row["states"]) == 26 and len(row["keyboard"]) == 69
    assert len(row["anchors"]) == 16 and len(row["programs"]) == 17
    assert not row["geometryOverflow"]

for file in source_hashes:
    destination = ARCHIVE / file
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / file, destination)
    assert digest(destination) == source_hashes[file]

records = {
    "programExecution": ("survival-program-execution.json", execution),
    "nativeVerification": ("survival-native-verification.json", native),
    "browserBehavior": ("survival-browser-behavior.json", behavior),
    "browserFinalReading": ("survival-browser-final-reading.json", reading),
    "figurePreflight": ("survival-figures-preflight.json", figures),
    "paintProof": ("survival-cox-paint-proof.json", paint),
}
for name, data in records.values():
    save("docs/teaching/evidence/" + name, data)

final_images = [
    "cox-gold-bars-verified-320.png", "final-cox-weight-paint-390.png",
    "final-logrank-equation-320.png", "final-score-equation-320.png",
    "final-discrete-equation-320.png", "final-hazard-equation-390.png",
    "final-censor-equation-390.png", "final-cox-reading-1440.png",
    "final-competing-reading-1440.png",
]
earlier_images = [
    "risk-lanes-default-390.png", "hazard-clock-390.png",
    "restricted-area-320.png", "pair-board-320.png",
    "changed-report-output-320.png",
]
opened = []
for stage, images in [("final reading", final_images), ("earlier behavior, unchanged numerical geometry/output; surrounding prose subsequently spaced", earlier_images)]:
    for image in images:
        file = "scratch/survival/browser/" + image
        opened.append({"path": file, "sha256": digest(ROOT / file), "stage": stage, "openedBy": "workflow_visual_improvements"})
for entry in figures["openedImages"]:
    assert digest(ROOT / entry["path"]) == entry["sha256"]
    opened.append({**entry, "stage": "isolated figure preflight; actual lesson placement also included in full route geometry checks", "openedBy": "scientific_visual_improvements"})

packet = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "topicId": "survival-analysis-cox-regression-kaplan-meier-hazard-models",
    "status": "author verified and source frozen; root independent closure and production integration remain separately attributed",
    "sourceCount": 8, "sourceHashes": source_hashes,
    "sourceArchive": "docs/teaching/archive/survival-author-freeze",
    "sourceAmendments": amendments,
    "design": "docs/teaching/SURVIVAL-ANALYSIS-LESSON-DESIGN.md",
    "originalArchive": "docs/teaching/evidence/survival-original.json",
    "originalExecution": "docs/teaching/evidence/survival-original-execution.json",
    "records": {key: "docs/teaching/evidence/" + name for key, (name, _) in records.items()},
    "versions": {**native["versions"], "scikit-learn": "1.9.1", "Python": "3.12.14"},
    "programs": {"count": 17, "wholeRunAt": execution["unchangedProgramExecution"]["executedAt"], "changedRossiSentenceRunAt": execution["executedAt"], "reuse": "Sixteen unchanged actual program executions retained; only corrected Rossi sentence rerun. Numerical verifier reused exact current module output evidence."},
    "nativeSummary": {"finiteFixtures": 121, "rejectionContracts": 13, "changedPractice": "B–L", "checkedAt": native["verifiedAt"], "owner": "testing_documentation_completion, scoped author-verification contribution"},
    "browserSummary": {"widths": [1440, 390, 320], "statesPerWidth": 26, "keyboardActionsPerWidth": 69, "anchorsPerWidth": 16, "programPairsPerWidth": 17, "displayEquationsPerWidth": 16, "behaviorAt": behavior["checkedAt"], "finalReadingAt": reading["checkedAt"], "originalFonts": True, "route": "http://127.0.0.1:5173/learn/path/full-curriculum/survival-analysis-cox-regression-kaplan-meier-hazard-models?module=classical-ml", "attribution": "Behavior pass retains its actual source hashes and initial mobile-formula failures. Final source-matched focused reading closes formula wrapping, prose spacing and prerequisite links. Pure models, examples, CSS and figures unchanged between these passes; root's metadata/export and AST-conserved JSX entity amendment is separately bound above."},
    "resolvedFindings": [
        "Rossi financial-aid assignment was randomized; fitted example and printed output now distinguish that fact from a complete causal analysis or held-out assessment.",
        "Cox range step=any preserves the exact log(2) preset in actual DOM and calculations.",
        "RMST identifies censor-only risk-table rows correctly; competing-risk control uses observed days rather than array indices.",
        "Prerequisite link uses actual classical-ml module; absolute numerical risk-tie tolerance and positive-lifetime left-censor lower bound are explicit.",
        "Local intermediate symbols and equivalent multi-line equations close narrow display overflow, without changing the numerical models.",
        "A misleading old 320px image preview suggested absent event fills; final PNG pixel evidence and actual 18px DOM bars establish correct gold fills. No production painting fix was needed.",
    ],
    "openedImages": opened,
    "limits": [
        "Source/native/browser checks establish the stated finite fixtures and implementations, not a universal statistical or causal guarantee.",
        "Reliability comparisons are two generated splits under declared laws, not industrial or clinical benchmarks.",
        "Censor independence/support, first-event definition, prediction timing and asymptotic uncertainty are assumptions, not certified by these demonstrations.",
        "Optional Fine–Gray/deep/discrete/interval branches are scoped mechanisms; no unavailable library implementation or unexecuted neural benchmark is claimed.",
        "Video titles/chapter descriptions were inspected; full playback is not claimed. Primary written-source inspected scope is in the design.",
        "Root's independent 23 invariance checks and final independent image/source review are not recounted as author tests; their record remains root-owned.",
    ],
    "retention": {"selectedMainImages": len(final_images) + len(earlier_images), "figureImages": len(figures["openedImages"]), "supersededImageMapping": {"scratch/survival/browser/final-cox-weight-paint-320.png": "scratch/survival/browser/cox-gold-bars-verified-320.png"}, "note": "Unselected and superseded generated captures plus installed duplicate draft are retired by a bounded literal-path cleanup after this packet is saved. Full test records preserve actual checks, not a promise that every intermediate capture remains."},
}
save("docs/teaching/evidence/survival-author-review.json", packet)
print(json.dumps({"timestamp": packet["timestamp"], "sourceCount": 8, "sourceHashes": source_hashes, "openedImages": len(opened)}, indent=2))
