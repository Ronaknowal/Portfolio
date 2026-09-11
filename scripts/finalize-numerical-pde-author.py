"""Freeze only source-matched, actually reviewed Numerical PDE author evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def fingerprint(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


native_path = "scratch/numerical-pde-native/native-results.json"
browser_path = "scratch/numerical-pde-browser/results.json"
fixture_path = "scratch/numerical-pde-native/model-fixtures.json"
native = read(native_path)
browser = read(browser_path)
fixtures = read(fixture_path)
assert native["passed"] and browser["passed"] and browser["errors"] == []
assert native["sourceHashes"] == browser["sourceHashes"] == fixtures["sourceHashes"]
assert len(native["sourceHashes"]) == 6
assert all(fingerprint(path) == digest for path, digest in native["sourceHashes"].items())
assert [record["width"] for record in browser["records"]] == [1440, 390, 320]
assert native["counts"]["complete actual stdout"] == 16
assert fixtures["invalid"] == 12 and fixtures["equations"] == 16
for record in browser["records"]:
    assert len(record["programs"]) == 16 and len(record["anchors"]) == 15
    assert len(record["math"]) == 16
    assert all(equation["scroll"] <= equation["width"] + 1 for equation in record["math"])
    assert "Space Grotesk" in record["fonts"] and "JetBrains Mono" in record["fonts"]

# These exact files were opened with view_image after the final 09:23:11 browser run.
# This list records human/agent visual inspection; generating a screenshot alone
# does not qualify it for this manifest.
opened = [
    "mapped-triangle-320", "restriction-reconstruction-390",
    "exact-field-budget-1440", "reading-section-1-390",
    "exactness-trap-390", "high-mode-first-step-320", "series-resistance-field-390",
    "stationary-shock-flux-390", "point-between-nodes-390",
    "rectangle-index-map-320", "coarse-high-mode-1440",
    "incompatible-outflow-390", "upwind-outside-cfl-320",
    "large-amplification-scale-390", "equation-13-320",
    "reading-section-7-1440", "reading-section-11-390",
    "reading-section-12-390", "reading-section-14-320",
    "explained-practice-4-390", "explained-practice-14-320",
    "changed-accepted-output-320", "annotated-resources-390",
    "ordinary-figure-7-0-390", "right-boundary-stencil-390",
    "equation-9-320", "reading-section-3-1440",
    "changed-complete-program-390", "reading-intro-1440", "equation-2-320",
]
captured = {name for record in browser["records"] for name in record["captures"]}
assert len(opened) == len(set(opened)) == 30
assert all(name + ".png" in captured for name in opened)
format_path = "scratch/numerical-pde-review/format-conservation.json"
formatting = read(format_path)
assert len(formatting["records"]) == 5
assert all(record["normalizedASTConserved"] for record in formatting["records"])
example_path = "scratch/numerical-pde-native/example-runs.json"
generated = read(example_path)
assert generated["passed"] and len(generated["programs"]) == 16
production_examples = fixtures["examples"]
for program in generated["programs"]:
    actual = production_examples[program["key"]]
    assert hashlib.sha256(actual["code"].encode("utf-8")).hexdigest() == program["codeSha256"]
    assert actual["expected"].strip() == program["stdout"].strip()

now = datetime.now(timezone.utc).isoformat()
evidence_paths = [
    native_path, browser_path, fixture_path, format_path, example_path,
    "docs/teaching/evidence/numerical-pdes-original-plan.json",
    "docs/teaching/evidence/numerical-pdes-assessed-design.md",
    "docs/teaching/evidence/numerical-pdes-design-independent-review.json",
]
script_paths = [
    "scripts/generate-numerical-pde-examples.py",
    "scripts/verify-numerical-pde-models.mjs",
    "scripts/verify-numerical-pde-native.py",
    "scripts/review-numerical-pde-lesson.cjs",
    "scripts/format-numerical-pde-source.cjs",
    "scripts/finalize-numerical-pde-author.py",
]
packet = {
    "topicId": "numerical-pdes-grids-finite-elements-stability",
    "title": "Numerical PDEs: Grids, Finite Elements & Stability",
    "mathematicsPosition": 57,
    "status": "author-frozen; independent review and integrated production checks are separate",
    "authorFrozenAt": now,
    "productionSourceCount": 6,
    "sourceHashes": native["sourceHashes"],
    "originalBaseline": {
        "publishedBodyExisted": False,
        "completeProgramExisted": False,
        "plannedInventory": "docs/teaching/evidence/numerical-pdes-original-plan.json",
        "assessedParentDesign": "docs/teaching/evidence/numerical-pdes-assessed-design.md",
        "designIndependentReview": "docs/teaching/NUMERICAL-PDES-DESIGN-INDEPENDENT-REVIEW.md",
        "attribution": "Parent design calculations and receiving-author design checks preceded production; neither substitutes for these actual-source checks.",
    },
    "scope": {
        "sections": 15,
        "completeExecutedPrograms": 16,
        "changedPracticeWithSeparateHintsAndSolutions": 15,
        "investigations": 11,
        "inlineFigures": 1,
        "requiredPrerequisites": ["Partial Differential Equations, Conservation & Boundary Conditions",
                                  "Conditioning, Stability & Numerical Analysis",
                                  "Matrix Decompositions (SVD, QR, Cholesky, LU)"],
        "interpretation": "Counts describe the authored scope, not a future template or mastery guarantee.",
    },
    "native": native,
    "modelExport": {"checkedAt": fixtures["checkedAt"], "invalidCases": fixtures["invalid"],
                    "equations": fixtures["equations"],
                    "fixtureCounts": {key: len(value) for key, value in fixtures["fixtures"].items()}},
    "browser": browser,
    "formatConservation": formatting,
    "actuallyOpenedImages": [
        {"path": "scratch/numerical-pde-browser/" + name + ".png",
         "sha256": fingerprint("scratch/numerical-pde-browser/" + name + ".png"),
         "inspection": "Opened after final run using view_image; ordinary viewport, original fonts."}
        for name in opened
    ],
    "evidenceHashes": {path: fingerprint(path) for path in evidence_paths},
    "verificationScriptHashes": {path: fingerprint(path) for path in script_paths},
    "closedAuthorFindings": [
        "Corrected the proposed Robin constant to 20/3 before complete-body readiness.",
        "Bounded triangle input now includes the declared translated vertex; the visible polygon uses actual equal-scale coordinates, with a verified 2:3 changed aspect ratio.",
        "Labels explicitly associate each select with its own text; keyboard controls, reset and initially hidden practice work at all widths.",
        "Narrow equations and prose equality chains wrap readably; no overflow clipping is used to hide notation.",
        "The multi-method diffusion investigation follows the implicit formulas; the point-load investigation follows its local load weights.",
        "Unstable heat comparisons use an independently evaluated operator on each actual previous vector; stable global matrix-power and exact spatial-ODE oracles remain separate.",
    ],
    "limits": [
        "Author checks are not the separate independent agent source review or integrated production build/loading review.",
        "A finite scoped lesson does not teach every numerical PDE method or establish physical-model validity.",
        "Bounds certify the declared manufactured mathematical field; sampled curves are diagnostics, not supremum proofs.",
        "The optional Godunov branch is an exact scalar face calculation, not a full nonlinear PDE solver; two-grid claims are for the explicitly defined operation.",
        "Graphs state actual axes and provenance; code/matrix panes retain intentional local horizontal scrolling on small screens.",
        "Verified official resource pages and selected written/transcript passages do not imply full video playback.",
        "User acceptance remains separate.",
    ],
}
destination = ROOT / "docs/teaching/evidence/numerical-pdes-author-review.json"
if destination.exists():
    raise RuntimeError("An author packet already exists; preserve it and append an explicit amendment instead.")
destination.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"authorFrozenAt": now, "productionSources": 6, "openedImages": len(opened),
                  "native": native["checkedAt"], "browser": browser["checkedAt"]}, indent=2))
