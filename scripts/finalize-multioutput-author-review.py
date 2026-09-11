"""Bind existing final checks and selected actually opened images; do not rerun them."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import shutil

root = Path(__file__).resolve().parents[1]
directory = root / "scratch/multioutput"
native = json.loads((directory / "native/verification.json").read_text())
execution = json.loads((directory / "native/execution.json").read_text())
behavior = json.loads((directory / "browser/behavior-results.json").read_text())
reading = json.loads((directory / "browser/final-reading-results.json").read_text())
formatting = json.loads((directory / "format-conservation.json").read_text())
assert reading["passed"] and not reading["errors"] and not behavior["errors"]
assert [record["width"] for record in reading["records"]] == [1440,390,320]
for record in behavior["records"]:
    assert len(record["states"]) == 14 and len(record["programs"]) == 15
    assert len(record["anchors"]) == 14
for file, expected in reading["sourceHashes"].items():
    assert hashlib.sha256((root/file).read_bytes()).hexdigest() == expected, file
for file, expected in native["sourceHashes"].items():
    assert reading["sourceHashes"][file] == expected
assert formatting["cssAstConserved"]
assert formatting["originalCssSha256"] == behavior["sourceHashes"][formatting["cssFile"]]
assert formatting["finalCssSha256"] == reading["sourceHashes"][formatting["cssFile"]]

opened = [
    "target-schema-390.png", "reverse-probability-tree-390.png", "threshold-ties-1440.png",
    "masked-error-grid-320.png", "reverse-probability-tree-320.png", "shared-split-units-320.png",
    "ridge-equation-320.png", "conditional-mosaics-1440.png", "shared-split-units-1440.png",
    "grouped-coefficients-320.png", "labelset-figure-1440.png", "continuous-reading-390.png",
    "shared-penalty-equation-320.png", "changed-joint-reading-320.png", "changed-report-output-390.png",
    "first-program-320.png", "split-reading-390.png", "association-reading-1440.png",
]
image_records = []
for name in opened:
    relative = "scratch/multioutput/browser/" + name
    image_records.append({"path":relative,"sha256":hashlib.sha256((root/relative).read_bytes()).hexdigest(),"actuallyOpened":True})
retired = sorted(path.name for path in (directory/"browser").glob("*.png") if path.name not in opened)
archive = root / "docs/teaching/archive/multioutput-author-freeze"
assert not archive.exists(), "Preserve an existing freeze before amending."
for relative in reading["sourceHashes"]:
    destination = archive / relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/relative,destination)
record = {
    "timestamp":datetime.now(timezone.utc).isoformat(),
    "topicId":"multi-label-multi-output-learning",
    "status":"author verified; independent review and parent integration remain separate",
    "sourceCount":len(reading["sourceHashes"]),"sourceHashes":reading["sourceHashes"],
    "sourceArchive":"docs/teaching/archive/multioutput-author-freeze",
    "originalArchive":"docs/teaching/evidence/multioutput-original.json",
    "originalExecution":"docs/teaching/evidence/multioutput-original-execution.json",
    "design":"docs/teaching/MULTIOUTPUT-LESSON-DESIGN.md",
    "designFixtures":"docs/teaching/evidence/multioutput-design-fixtures.json",
    "versions":{package:importlib.metadata.version(package) for package in ["numpy","scipy","scikit-learn","pandas","black"]},
    "execution":execution,"native":native,"behavior":behavior,"finalReading":reading,"formatting":formatting,
    "programs":15,"investigations":6,"inlineFigures":3,"changedPractice":12,
    "openedImages":image_records,
    "retiredUnopenedCaptures":["scratch/multioutput/browser/"+name for name in retired],
    "retention":"Selected actual final images and compact native/browser records remain; generated unused captures are retired after closure. The behavior record’s historical capture names do not imply those images were selected or visually reviewed.",
    "amendmentScope":"Final reading changes reflow mathematically equivalent formulas, clarify the initial mixture setting, add an optional review link and format CSS with conserved AST. Pure models and all program bytes are unchanged; prior passing interaction checks are retained, while final geometry/source hashes are checked separately.",
    "limitations":["The original seventh network-download block has no completed execution; exact code remains archived.","Tiny controlled/hand-authored cohorts teach mechanisms, not calibrated population performance or deployment readiness.","Exact small probability trees do not establish a scalable global joint decoder; path scores and marginals remain distinct.","The optional course video listing was verified but the video itself was not watched.","Full source/output use intentional horizontal scrolling on phones.","Parent independent review, production integration and user acceptance are not claimed."]
}
destination = root / "docs/teaching/evidence/multioutput-author-review.json"
assert not destination.exists(), "Do not overwrite an author packet."
destination.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(record["timestamp"],record["sourceCount"],"frozen sources;",len(opened),"actually opened images")
