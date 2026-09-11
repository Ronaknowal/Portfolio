"""Freeze Decision Theory's owned source and actually inspected author evidence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def fingerprint(filename):
    data = Path(filename).read_bytes()
    return {
        "path": filename,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


records = {}
for name, filename in [
    ("models", "scratch/decision-theory-review/model-results.json"),
    ("native", "scratch/decision-theory-review/native-results.json"),
    ("browser", "scratch/decision-theory-browser/results.json"),
    ("finalBoundaries", "scratch/decision-theory-browser/final-boundaries.json"),
    ("formatConservation", "scratch/decision-theory-review/format-conservation.json"),
]:
    result = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    if name != "formatConservation":
        assert result.get("passed", result.get("status") == "passed")
    records[name] = {"file": fingerprint(filename), "result": result}

final_sources = [
    fingerprint(source["path"])
    for source in records["finalBoundaries"]["result"]["sources"]
]
assert len(final_sources) == 7
for source, checked in zip(
    final_sources, records["finalBoundaries"]["result"]["sources"], strict=True
):
    assert source["path"] == checked["path"]
    assert source["sha256"] == checked["sha256"]
earlier = {
    item["path"]: item["sha256"]
    for item in records["models"]["result"]["sources"]
}
changed = [
    item["path"] for item in final_sources
    if item["sha256"] != earlier[item["path"]]
]
assert changed == ["src/learn/components/lesson-labs/DecisionTheoryLabs.jsx"]
for name in ["browser", "finalBoundaries"]:
    assert [row["width"] for row in records[name]["result"]["records"]] == [
        1440, 390, 320
    ]

opened = [
    "reading-intro-390", "reading-section-3-390", "capacity-two-slots-320",
    "contingent-test-tree-390", "risk-envelope-tie-320",
    "minimax-certificate-plane-320", "utility-chord-390",
    "asymmetric-cost-curve-390", "fixed-state-versus-prior-390",
    "atom-aware-tail-320", "changed-capacity-contingent-policy-390",
    "inline-figure-0-390", "inline-figure-1-390", "final-fallback-sampling-320",
    "equation-0-320", "equation-2-320", "equation-3-320", "equation-4-320",
    "final-cvar-endpoint-320", "complete-capstone-program-390",
    "executed-capstone-output-320", "explained-practice-12-320",
    "annotated-learning-resources-390", "reading-section-1-1440",
]
assert len(opened) == len(set(opened)) == 24
evidence_files = [
    "scripts/generate-decision-theory-examples.py",
    "scripts/format-decision-theory-source.cjs",
    "scripts/fix-decision-prose-spacing.cjs",
    "scripts/verify-decision-theory-models.mjs",
    "scripts/verify-decision-theory-native.py",
    "scripts/review-decision-theory-lesson.cjs",
    "scripts/review-decision-theory-final-boundaries.cjs",
    "scripts/finalize-decision-theory-review.py",
    "docs/teaching/DECISION-THEORY-LESSON-DESIGN.md",
    "docs/teaching/DECISION-THEORY-VERIFICATION.md",
]
packet = {
    "authorFrozenAt": datetime.now(timezone.utc).isoformat(),
    "topicId": "decision-theory-risk-cost-sensitive-decisions",
    "status": "author-verified; independent review, production integration and user acceptance separate",
    "productionSourceCount": len(final_sources),
    "sources": final_sources,
    "records": records,
    "evidenceFiles": [fingerprint(filename) for filename in evidence_files],
    "originalPlan": fingerprint("docs/teaching/evidence/decision-theory-original-plan.json"),
    "openedImages": [
        fingerprint(f"scratch/decision-theory-browser/{name}.png")
        for name in opened
    ],
    "imageReview": "All 24 listed final images were actually opened with view_image. They include ordinary reading, both inline figures, distinct model geometries, exact ties, atom mass, changed allocations, equations, complete program/output, explained practice and resources. The other generated captures are automated evidence, not individually opened review.",
    "finalAmendment": {
        "changedSinceModelRecord": changed,
        "description": "Only the Labs caption changed after the final model/native record: label 101 sampled probabilities, acknowledge thin boundaries between samples and distinguish the directly calculated risk readout. Final boundaries browser record hashes every current production file and asserts the caption at all three widths. No executable model or program changed afterward.",
        "fullBrowserScope": "The comprehensive three-width run passed before the own-entry validation amendment and final caption. The focused current-source run checks valid-array interactions, exact ties, all eight mounted labs, CVaR endpoint text, the sampled-strip caption, fonts and narrow fit. Earlier hashes/results are preserved, not rewritten as final-source tests.",
    },
    "resolvedFindings": [
        "Root identified shared H2 auto-slug behavior; all 13 topic links now have verified real arrivals.",
        "Exact finite action comparisons preserve co-optimal tests/rules/fallback actions and cumulative atom boundaries; all action sets, not just rounded risks, are checked.",
        "Root's general CVaR alpha=0 attainment qualification is explicit; finite-support endpoint behavior is distinguished from an unbounded-below distribution.",
        "Own-entry array validation rejects inherited entries; positive utility underflow is explicitly rejected. Both have actual regressions.",
        "Opened narrow screenshots prompted larger labels and four wrapped equations, preserving all mathematical terms.",
        "Browser harness was adapted to the actual code renderer's text nodes; the focused slider uses valid decimal input. Failed harness runs are not counted as passes.",
    ],
    "knownLimits": [
        "The source arguments establish finite contracts under declared assumptions; finite test enumeration alone is not a general proof or an independent whole-lesson review.",
        "Synthetic costs, utility and intervention probabilities are stipulated, not estimated population truth, causal identification or advice about personal decisions.",
        "The fallback action strip is a 101-point sample; selected-point risks use direct exact finite comparisons. Other geometric views follow their stated finite formulas.",
        "Models interpret canonical written decimal Number inputs through rational arithmetic and reject nonrepresentable nonzero results. They are bounded teaching models, not arbitrary-precision statistical packages.",
        "Complete Python examples use the standard library. Code/output blocks may scroll horizontally on phones; page, formulas and compact figures fit.",
        "Selected source sections and some official transcripts/slides were read; verified video metadata is not full playback. Exact inspected scope is in the design and lesson annotations.",
        "No shared catalogue/order changes or full production build were performed by this author. Root owns independent review and integration; user acceptance remains pending.",
    ],
}
target = Path("docs/teaching/evidence/decision-theory-author-review.json")
assert not target.exists(), "Preserve an existing frozen packet; append an explicit amendment instead."
target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(packet["authorFrozenAt"])
for source in packet["sources"]:
    print(source["path"], source["sha256"])
