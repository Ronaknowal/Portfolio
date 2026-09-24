"""Bind independently observed evidence to the author's final amended sources."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch/complex-transforms-independent-review"
AUTHOR_PATH = "docs/teaching/evidence/complex-transforms-author-review.json"
ORIGINAL_PATH = "docs/teaching/evidence/complex-transforms-author-review-before-independent-amendment.json"


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fingerprint(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


author = read_json(AUTHOR_PATH)
original = read_json(ORIGINAL_PATH)
native = read_json("scratch/complex-transforms-independent-review/results.json")
browser = read_json("scratch/complex-transforms-independent-review/browser-results.json")
opened = read_json("scratch/complex-transforms-independent-review/opened-images.json")
assert native["status"] == "passed" and browser["passed"]
assert native["productionSources"] == browser["productionSources"] == author["productionSources"]
assert author["productionSources"] == [fingerprint(row["path"]) for row in author["productionSources"]]
for row in original["productionSources"]:
    archive = "scratch/complex-transforms-before-independent-amendment/" + row["path"]
    assert fingerprint(archive)["sha256"] == row["sha256"]
for row in opened["images"]:
    assert fingerprint(row["path"])["sha256"] == row["sha256"]

result = {
    "topicId": author["topicId"],
    "title": author["title"],
    "reviewer": "independent teaching and numerical reviewer",
    "closedAt": datetime.now(timezone.utc).isoformat(),
    "status": "independent review closed; no unresolved material finding; parent integration remains separate",
    "productionSources": author["productionSources"],
    "authorFreeze": {**fingerprint(AUTHOR_PATH), "frozenAt": author["frozenAt"]},
    "originalAuthorFreeze": {
        **fingerprint(ORIGINAL_PATH),
        "frozenAt": original["frozenAt"],
        "allSixArchivedSourcesVerified": True,
        "sourceArchive": "scratch/complex-transforms-before-independent-amendment",
        "productionSources": original["productionSources"],
    },
    "initialFindings": read_json("scratch/complex-transforms-independent-review/initial-findings.json"),
    "findingClosure": [
        {"finding": "Native small-argument Laplace cancellation", "resolution": "Author's bounded series verified against 90-digit expm1 oracle; all original stdout retained."},
        {"finding": "General least-squares proof omitted explicit L2 assumption", "resolution": "Actual preceding paragraph now requires square-integrability; browser and changed projection checks pass."},
        {"finding": "Reachable filtered DFT exceeded fixed vertical plot bounds", "resolution": "All 16 current plotted points fit the data-derived extent at all three widths."},
        {"finding": "Reachable filter transient exceeded fixed vertical plot bounds", "resolution": "All 1,539 current curve points fit the data-derived extent at all three widths."},
    ],
    "inspectedScope": "Full actual lesson, 17 actual programs, pure models, complete labs/CSS/brief, design, author records, all 14 changed practice solutions and selected primary sources; proofs read independently of test counts.",
    "native": native,
    "browser": browser,
    "actuallyOpenedImages": opened,
    "revisitedSources": [
        {"url": "https://www.jirka.org/diffyqs/html/moreonfourier_section.html", "scope": "Piecewise-smooth definitions and midpoint convergence theorem; not a full book audit."},
        {"url": "https://dlmf.nist.gov/6.16", "scope": "Gibbs formulas, first-peak constant and jump factor."},
        {"url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html", "scope": "Parameters and notes for detrending, one-sided sums and spectrum/density scaling."},
    ],
    "evidenceFiles": [fingerprint(path) for path in [
        "scripts/verify-complex-transforms-independent.mjs",
        "scripts/verify-complex-transforms-independent.py",
        "scripts/review-complex-transforms-independent.cjs",
        "scratch/complex-transforms-independent-review/initial-native-payload.json",
        "scratch/complex-transforms-independent-review/initial-findings.json",
        "scratch/complex-transforms-independent-review/results.json",
        "scratch/complex-transforms-independent-review/browser-results.json",
        "scratch/complex-transforms-independent-review/opened-images.json",
    ]],
    "limits": "Finite complementary source/numerical/browser review, not universal binary64 robustness, beginner study, full-video viewing, user approval or production build/loading integration. Author suites are separately attributed; only listed opened images were manually inspected.",
}
destination = ROOT / "docs/teaching/evidence/complex-transforms-independent-review.json"
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
report = ROOT / "docs/teaching/COMPLEX-FOURIER-LAPLACE-INDEPENDENT-REVIEW.md"
text = report.read_text(encoding="utf-8")
old = "The complete production source was read and complementary numerical and browser checks passed on 11 September 2026. Final identity closure is pending the author's amended freeze; this paragraph will be replaced only after the six current hashes match that packet. The reviewer did not edit production source. Parent production integration and user acceptance remain separate."
new = f"The six production sources match the author's amended **11 September 2026, {author['frozenAt'].split('T')[1]}** freeze. The [independent packet](evidence/complex-transforms-independent-review.json) binds their exact hashes, this review's checks and the separate original **06:30:31 UTC** author record. All six archived original files were also verified against the original packet; nothing was silently replaced. No unresolved material finding remains. The reviewer did not edit production source. Parent production integration and user acceptance remain separate."
assert old in text or new in text
report.write_text(text.replace(old, new), encoding="utf-8")
print(json.dumps({"closedAt": result["closedAt"], "authorFreeze": author["frozenAt"], "productionFiles": len(author["productionSources"]), "openedImages": len(opened["images"]), "record": str(destination)}, indent=2))
