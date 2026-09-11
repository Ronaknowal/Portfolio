"""Bind the completed independent review to the author's final six sources."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(path):
    data = Path(path).read_bytes()
    return {"path": str(path).replace("\\", "/"), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


directory = "scratch/single-variable-calculus-independent-review"
author_path = "docs/teaching/evidence/single-variable-calculus-author-review.json"
author = json.loads(Path(author_path).read_text(encoding="utf-8"))
sources = [fingerprint(item["path"]) for item in author["sources"]]
assert sources == author["sources"]
native = json.loads(Path(f"{directory}/results.json").read_text(encoding="utf-8"))
browser = json.loads(Path(f"{directory}/browser-results.json").read_text(encoding="utf-8"))
assert native["status"] == browser["status"] == "passed"
opened_paths = [
    f"{directory}/amended-{case}-{width}.png"
    for case in ["cusp-adjacent", "taylor-roundoff"]
    for width in [1440, 390, 320]
]
opened_paths += [f"{directory}/changed-signed-geometry-320.png", f"{directory}/amended-acceleration-checkpoint-320.png"]
opened_paths += [f"scratch/single-variable-calculus-browser/final-inline-{index}-320.png" for index in range(4)]
report = {
    "reviewedAt": datetime.now(timezone.utc).isoformat(),
    "topicId": "single-variable-calculus-limits-derivatives-integrals",
    "reviewer": "/root/scientific_visual_improvements",
    "status": "independently-reviewed-no-unresolved-material-finding",
    "authorFrozenAt": author["authorFrozenAt"],
    "authorRecord": fingerprint(author_path),
    "finalProductionSources": sources,
    "initialAuthorNativeRecord": json.loads(Path(f"{directory}/initial-author-native-record.json").read_text(encoding="utf-8")),
    "sourceRead": [
        "Complete fourteen-section body, thirteen changed practice questions/hints/solutions and all fifteen actual Python programs with expected stdout.",
        "Complete pure model, eight interactive investigations, four inline figures and topic CSS.",
        "Individual semantic blueprint, lesson design, incoming compounding note and final author verification.",
    ],
    "mathematicalReview": [
        "Strict punctured epsilon/delta quantifiers, equality of the open-neighborhood supremum, and strict rational interior witnesses.",
        "Derivative local errors, product/chain rules, zero inner derivative, inverse hypotheses, radians and implicit local branches.",
        "Fermat/Rolle/mean value theorem conditions, complete endpoint/critical candidates and permitted-interval extrema.",
        "Signed versus absolute accumulation, square Riemann sums, uniform-continuity convergence and both FTC directions with continuity qualifiers.",
        "Substitution on signed paths, parts, rational poles and trig branches; logarithm defined by area, exponential inversion, growth and finite compounding without circularity.",
        "Integral Taylor remainder, exponential/logarithm series limits, analytic truncation versus floating evaluation, and endpoint/tail improper integrals versus principal values.",
        "Finite zero-over-zero L'Hopital hypotheses, distributed mass, work, volume, arc length, integrating-factor uniqueness and every changed practice solution.",
    ],
    "complementaryExecution": native,
    "reviewerBrowserSubset": browser,
    "evidenceFiles": [fingerprint(f"{directory}/{name}") for name in ["fixtures.json", "results.json", "browser-results.json", "initial-author-native-record.json"]],
    "openedImages": [fingerprint(path) for path in opened_paths],
    "visualAssessment": "Twelve files actually opened: eight reviewer captures and four author inline phone captures. Adjacent endpoint labels and exact classification remain readable; Taylor distinguishes roundoff and approximate analytic bounds; the revised velocity checkpoint is visible. Three signed rectangles were checked against independent cubic values and SVG dimensions. Motion legs share one position scale, the product corner has the stated second-order meaning, FTC's triangular residual has area 1/8, and the rod's center 10/9 lies right of its midpoint on the same scale.",
    "findings": [
        {
            "issue": "Rounding of interval midpoints and polynomial heights falsely classified signs or extrema on accepted adjacent endpoints.",
            "before": "Cusp [2,2+2*Number.EPSILON] could have sign zero; motion [1,1+Number.EPSILON] falsely tied both heights.",
            "resolution": "Author uses exact decimal rational candidate comparisons and analytical critical-point partitions, supports scientific notation, and distinguishes rounded plot values from exact classification.",
            "closure": "51 changed/adjacent/subnormal intervals and actual three-width cusp/motion controls passed on the final formatted source.",
        },
        {
            "issue": "Increasing negative velocity was taken to imply strictly positive acceleration everywhere.",
            "resolution": "The checkpoint specifies continuous increasing velocity, claims nonnegative acceleration where differentiable, and supplies a zero-acceleration counterexample.",
            "closure": "Final prose read and actual keyboard-opened checkpoint checked at 1440/390/320.",
        },
        {
            "issue": "The Taylor lab's floating subtraction could exceed its truncation-only analytic bound without a distinction in the labels.",
            "before": "Degree 12 at x=.05 yields a computed difference around 2.22e-16 while the analytic bound is around 2.06e-27.",
            "resolution": "The author preserves the theorem and separately labels floating difference, approximate bound evaluation and roundoff, including possible zero difference.",
            "closure": "110 independent fixed-segment integral remainder cases plus actual roundoff readouts at three widths passed.",
        },
    ],
    "limits": [
        "Finite execution supports implementations; the general real-analysis arguments were read separately and are not proved by these tests.",
        "The largest absolute floating discrepancy occurs for an approximately 1.45e11 integral and has relative size about 3.15e-15. Small subtraction residuals use an absolute tolerance; no uniform relative-error claim is made.",
        "The reviewer's browser execution is a bounded changed-input, keyboard and geometry subset. The author's complete eight-lab, equation and reading run remains separately attributed in the linked author packet.",
        "Only the twelve listed screenshots are claimed as visually opened in this review. Generated captures and truncated tool output do not count as visual inspection.",
        "The reviewer did not edit production sources, shared registries or integration records, and does not claim to have replayed complete external videos or read every cited textbook.",
    ],
}
destination = Path("docs/teaching/evidence/single-variable-calculus-independent-review.json")
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"saved": str(destination), "reviewedAt": report["reviewedAt"], "sources": len(sources), "openedImages": len(opened_paths)}, indent=2))
