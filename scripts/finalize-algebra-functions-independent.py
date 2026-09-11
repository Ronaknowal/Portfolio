"""Close the independent review against the current author freeze."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def fingerprint(relative):
    data = (ROOT / relative).read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


author = read_json("docs/teaching/evidence/algebra-functions-author-review.json")
native = read_json("scratch/algebra-functions-independent/native-results.json")
browser = read_json("scratch/algebra-functions-independent/browser-results.json")
assert native["passed"] and browser["passed"]
for source in author["production"]:
    assert fingerprint(source["path"])["sha256"] == source["sha256"]
assert native["production"] == author["production"]

images = [
    "scratch/algebra-functions-independent/independent-log-geometry-390.png",
    "scratch/algebra-functions-independent/independent-log-geometry-320.png",
    "scratch/algebra-functions-browser/reading-3-390.png",
    "scratch/algebra-functions-browser/practice-inverse-390.png",
    "scratch/algebra-functions-browser/final-fractional-powers-390.png",
    "scratch/algebra-functions-browser/inline-2-390.png",
]
result = {
    "topicId": "algebra-functions-exponentials-logarithms",
    "reviewedAt": datetime.now(timezone.utc).isoformat(),
    "reviewer": "workflow_visual_improvements; independent of the author",
    "authorFrozenAt": author["frozenAt"],
    "status": "closed; no remaining material finding in the bounded review",
    "production": author["production"],
    "readScope": ["complete lesson", "all pure models", "all nine executable examples", "all labs and CSS", "complete design and stable-ID blueprint", "all twelve independent practice tasks", "author verification and amendment"],
    "finding": {
        "location": "src/learn/components/lesson-labs/AlgebraFunctionsLabs.jsx: GrowthScaleLab additive curve",
        "initialLabSha256": "fbbe251af2c1d82e7dd82d07c6320ed56830b98c128b07253c1ecd54ee4214e4",
        "initialAuthorFrozenAt": "2026-09-10T23:17:48.767354+00:00",
        "evidenceType": "direct source inspection and analytic counterexample; no pre-fix browser screenshot was claimed",
        "problem": "A polyline joining only (0,100) and (8,260) after logarithmic transformation implies sqrt(26000) at t=4, although the additive rule and table require 180. It falsely makes additive and exponential models both straight on the semilog plot.",
        "incorrectImpliedValue": 26000 ** 0.5,
        "correctValue": 180,
        "remedy": "Author sampled the actual additive rule at all 81 displayed time points before the log-axis transform. Models, program source/output, body, CSS and blueprint remained unchanged.",
        "finalLabSha256": "477594286e2ff402d95bf578322705ae0cd0b1b0b4d40c8ad3104342498a8c99",
        "finalEvidence": "Independent actual-font Edge at390/320: all81 additive log coordinates, all81 exponential log coordinates and all81 additive linear coordinates match their own rule for three rates. Midpoint log fraction0.6151539763976774 rather than0.5; visible readout180.",
        "resolved": True,
    },
    "independentNative": native,
    "independentBrowser": browser,
    "openedScreenshots": [dict(fingerprint(image), attribution="reviewer-created" if "independent/" in image else "author-created; opened by reviewer") for image in images],
    "scripts": [fingerprint(name) for name in ["scripts/verify-algebra-functions-independent.mjs", "scripts/verify-algebra-functions-independent.py", "scripts/review-algebra-functions-independent.cjs", "scripts/finalize-algebra-functions-independent.py"]],
    "harnessCorrections": ["For the largest plotted vertical range, a subpixel-height additive line requires2e-12 relative coordinate-ratio tolerance to account for cancellation; mathematical values were unchanged.", "Removed intentional Vite WebSocket closure, which otherwise generated a harness-created console error.", "The first390 capture showed page restoration instead of the graph; added actual bounding-rectangle assertion/retry and opened both final replacement captures."],
    "limits": "Bounded source/proof/computation and targeted visual review; not a learner study, full external-source playback, general numerical library certification, deployment or integrated production build. Author's wider browser/keyboard suite is attributed separately.",
}
(ROOT / "docs/teaching/evidence/algebra-functions-independent-review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"closedAt": result["reviewedAt"], "productionHashesMatch": True, "resolvedMaterialFindings": 1, "openedImages": len(images)}))
