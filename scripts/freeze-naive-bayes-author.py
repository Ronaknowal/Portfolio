from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json

packet_path = Path("docs/teaching/evidence/naive-bayes-author-review.json")
if packet_path.exists():
    raise RuntimeError("Preserve an existing author freeze; use an explicit amendment.")
now = datetime.now(timezone.utc).isoformat()
design_path = Path("docs/teaching/NAIVE-BAYES-LESSON-DESIGN.md")
design = design_path.read_text(encoding="utf-8").replace(
    "Status: design and evaluated fixtures; production authoring, final numerical review and browser review pending.",
    "Status: complete author implementation and numerical/browser review; exact freeze is in evidence/naive-bayes-author-review.json. Independent review and production integration are separate."
)
design += "\n\n## Implemented assessment — 11 September 2026\n\nThe complete body retains the scoped coverage and supplies six distinct investigations, eleven executed complete programs, two early checkpoints and twelve independent practice tasks. Counts reflect the actual teaching mechanisms, not targets. Numerical/final-font browser and opened-reading evidence is in [NAIVE-BAYES-VERIFICATION.md](NAIVE-BAYES-VERIFICATION.md) and the exact author packet. Current installed calibration source inspection confirms the sigmoid receives GaussianNB probabilities; the separate temperature branch is not conflated with it. The changed report includes actual outputs and the empty highest-probability bin/action limitation. The original body and six original program responses remain archived; no original snapshot is overwritten. Destination findings were saved to canonical Calibration and Ensemble notes; those destinations remain open until their own scoped assessment.\n"
design_path.write_text(design, encoding="utf-8")
note_path = Path("docs/teaching/topic-notes/calibration-conformal-prediction.md")
note = note_path.read_text(encoding="utf-8").replace(
    "# Naive Bayes score-interface and fit-boundary discovery — 11 September 2026",
    "## Naive Bayes score-interface and fit-boundary discovery — 11 September 2026"
).replace(
    "Origin is the in-progress [Naive Bayes design](../NAIVE-BAYES-LESSON-DESIGN.md), not yet a reviewed implementation. Retain the earlier conditioning/axis and Decision Theory findings below.",
    "Origin is now author-reviewed: [Naive Bayes verification](../NAIVE-BAYES-VERIFICATION.md) and [exact source/evidence packet](../evidence/naive-bayes-author-review.json). Independent review/integration remain separate. Retain the earlier conditioning/axis and Decision Theory findings above."
)
note_path.write_text(note, encoding="utf-8")
ensemble_path = Path("docs/teaching/topic-notes/ensemble-methods-stacking.md")
note = ensemble_path.read_text(encoding="utf-8").replace(
    "originating Naive Bayes implementation is in progress.",
    "originating Naive Bayes implementation is author-reviewed; independent review/integration remain separate."
).replace(
    "design-level exact arithmetic and a complete native probability experiment exist in the origin; its final verification is pending.",
    "[Final origin verification](../NAIVE-BAYES-VERIFICATION.md) and [exact author packet](../evidence/naive-bayes-author-review.json) now include independent exact/library oracles, complete native code, changed report and actual-font browser review."
)
ensemble_path.write_text(note, encoding="utf-8")
production = [
    "src/learn/data/topics/naive-bayes-probabilistic-classifiers.jsx",
    "src/learn/data/naive-bayes-models.js",
    "src/learn/data/naive-bayes-examples.js",
    "src/learn/components/lesson-labs/NaiveBayesLabs.jsx",
    "src/learn/components/lesson-labs/naive-bayes-labs.css",
    "src/learn/data/curriculum/blueprints/naive-bayes-probabilistic-classifiers.js",
]
def fingerprint(path):
    data = Path(path).read_bytes()
    return {"path": str(path).replace("\\", "/"), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

archive = {"frozenAt": now, "scope": "Exact six authored sources; original legacy evidence is separate", "sources": []}
snapshot_dir = Path("scratch/naive-bayes-author-frozen")
snapshot_dir.mkdir(exist_ok=True)
for source in production:
    content = Path(source).read_bytes()
    archive["sources"].append({**fingerprint(source), "base64": base64.b64encode(content).decode()})
    (snapshot_dir / Path(source).name).write_bytes(content)
archive_path = Path("docs/teaching/evidence/naive-bayes-author-sources.json")
assert not archive_path.exists()
archive_path.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")
opened = ["reading-" + str(i) + "-320" for i in range(1,12)] + [
    "geometry-unequal-390", "density-crossing-390", "reliability-true-390", "corpus-390",
    "token-evidence-1440", "repeat-presence-390", "copied-risk-320", "changed-solution-390",
    "final-equation-10-320", "final-equation-16-320", "final-gaussian-proof-390", "sources-320",
    "final-report-consequence-320", "final-intro-390", "final-rare-class-answer-390",
    "geometry-unequal-1440", "calibrated-program-1440", "reliability-true-320",
    "complement-pooling-320", "model-fork-320", "invalid-alpha-390",
    "final-copied-node-320", "final-copied-node-390",
]
paths = {
    "native": "scratch/naive-bayes-verification/results.json",
    "validation": "scratch/naive-bayes-verification/js-validation.json",
    "practiceApi": "scratch/naive-bayes-verification/practice-api-results.json",
    "browser": "scratch/naive-bayes-browser/results.json",
    "finalReading": "scratch/naive-bayes-browser/final-reading-results.json",
    "formatConservation": "scratch/naive-bayes-verification/format-conservation.json",
}
payloads = {key: json.loads(Path(path).read_text(encoding="utf-8")) for key, path in paths.items()}
assert len(payloads["native"]["programs"]) == 11
assert [r["width"] for r in payloads["browser"]["records"]] == [1440,390,320]
assert all(r["states"] == 87 and r["practice"] == 12 and not r["errors"] for r in payloads["browser"]["records"])
assert all(not r["errors"] for r in payloads["finalReading"]["records"])
support = [
    str(design_path), "docs/teaching/NAIVE-BAYES-VERIFICATION.md",
    "scripts/generate-naive-bayes-examples.py", "scripts/verify-naive-bayes-examples.mjs",
    "scripts/verify-naive-bayes-native.py", "scripts/verify-naive-bayes-practice.py",
    "scripts/review-naive-bayes-lesson.cjs", "scripts/review-naive-bayes-reading.cjs",
    "scripts/format-naive-bayes-source.cjs", "scripts/archive-naive-bayes-original.mjs",
    "scripts/freeze-naive-bayes-author.py"
]
packet = {
    "topicId": "naive-bayes-probabilistic-classifiers", "authorFrozenAt": now,
    "status": "author-reviewed; separate independent review and parent production integration pending",
    "production": [fingerprint(path) for path in production],
    "sourceArchive": fingerprint(archive_path),
    "originalEvidence": fingerprint("docs/teaching/evidence/naive-bayes-original-review.json"),
    "support": [fingerprint(path) for path in support],
    "checks": {key: {"record": fingerprint(paths[key]), "result": value} for key, value in payloads.items()},
    "openedImages": [fingerprint("scratch/naive-bayes-browser/" + name + ".png") for name in opened],
    "reviewScope": {
        "source": "Full original/revised body, proofs, all program source, practice, model/labs/CSS and annotated references",
        "visual": "Author opened listed ordinary-reading, mechanisms, errors, proof, program, answers and sources; rechecked affected final contexts",
        "lastDisplayAmendment": "After full behavior, copied-alarm source box width 116→160 native units; no model/interaction change. Final focused three-width record verifies actual text containment and reopened final phone images.",
        "resource": "Primary written/API material and official video/course context plus corresponding substantive notes; no full-video viewing claimed",
        "limits": "Bounded synthetic and arithmetic domains. No real deployment benchmark, learner study, user acceptance, independent approval or production integration inferred."
    }
}
packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"frozenAt": now, "production": packet["production"], "opened": len(opened)}, indent=2))
