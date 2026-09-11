"""Freeze the actual author-reviewed ODE sources and attributable evidence."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "ordinary-differential-equations-linear-systems"
PREFIX = "ordinary-differential-equations"


def fingerprint(relative):
    path = ROOT / relative
    data = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def record(relative):
    return {
        "file": fingerprint(relative),
        "result": json.loads((ROOT / relative).read_text(encoding="utf-8")),
    }


sources = [
    f"src/learn/data/topics/{TOPIC}.jsx",
    f"src/learn/data/{PREFIX}-models.js",
    f"src/learn/data/{PREFIX}-examples.js",
    "src/learn/components/lesson-labs/OrdinaryDifferentialEquationsLabs.jsx",
    "src/learn/components/lesson-labs/OrdinaryDifferentialEquationsFigures.jsx",
    f"src/learn/components/lesson-labs/{PREFIX}-labs.css",
    f"src/learn/data/curriculum/blueprints/{TOPIC}.js",
]
records = {
    "models": record(f"scratch/{PREFIX}-verification/model-results.json"),
    "native": record(f"scratch/{PREFIX}-verification/native-results.json"),
    "browser": record(f"scratch/{PREFIX}-browser/results.json"),
    "formatting": record(f"scratch/{PREFIX}-verification/format-conservation.json"),
}
assert records["models"]["result"]["status"] == "passed"
assert records["native"]["result"]["status"] == "passed"
assert records["browser"]["result"]["passed"] is True
assert records["browser"]["result"]["errors"] == []
for width in records["browser"]["result"]["records"]:
    assert len(width["anchors"]) == 14
    assert len(width["states"]) == 26
    assert len(width["keyboard"]) == 36
    assert len(width["programs"]) == 13
    assert len(width["math"]) == 23
assert {row["width"] for row in records["browser"]["result"]["records"]} == {
    1440, 390, 320
}

# The author opened these final images through view_image and inspected them.
# This is deliberately a subset of the 204 screenshots saved by the browser.
opened_names = [
    "inconsistent-boundary-320.png",
    "reading-figure-8-390.png",
    "reading-figure-9-320.png",
    *[f"equation-{index}-320.png" for index in range(23)],
    "changed-output-390.png",
    "events-complete-program-1440.png",
    "learning-resources-320.png",
    "swapped-stages-320.png",
    "reading-figure-0-320.png",
    "reading-figure-4-320.png",
    "reading-figure-6-1440.png",
    "rk4-actual-stages-320.png",
    "unstable-euler-320.png",
    "changed-practice-6-390.png",
]
scripts = [
    f"scripts/generate-{PREFIX}-examples.py",
    f"scripts/verify-{PREFIX}-models.mjs",
    f"scripts/verify-{PREFIX}-native.py",
    f"scripts/review-{PREFIX}-lesson.cjs",
    f"scripts/format-{PREFIX}-source.cjs",
    f"scripts/finalize-{PREFIX}-review.py",
]
packet = {
    "authorFrozenAt": datetime.now(timezone.utc).isoformat(),
    "topicId": TOPIC,
    "status": "author-verified; independent review, production integration and user acceptance separate",
    "productionSourceCount": len(sources),
    "sources": [fingerprint(path) for path in sources],
    "records": records,
    "verificationScripts": [fingerprint(path) for path in scripts],
    "openedFinalImageCount": len(opened_names),
    "openedFinalImages": [
        fingerprint(f"scratch/{PREFIX}-browser/{name}") for name in opened_names
    ],
    "preservation": {
        "original": record(f"docs/teaching/evidence/{PREFIX}-original-plan.json"),
        "designArithmeticOnly": record(
            f"docs/teaching/evidence/{PREFIX}-design-checks.json"
        ),
        "priorBody": "No prior published body or native program existed at the exact scoped inventory check.",
    },
    "preFinalRepairs": {
        "retainedLastBrowserFailure": record(f"scratch/{PREFIX}-browser/failure.json"),
        "disposition": [
            "390px and 320px displayed equations were reflowed with semantic line breaks and an explicitly defined RK4 weighted average; all 23 final equations fit at all widths.",
            "Endpoint inspection markers were moved outside path clip groups after actual screenshot inspection; final boundary, fundamental-column and forcing captures were reopened.",
            "The logistic implementation handles subnormal initial populations through log-odds and expm1 branches, checked against independent high precision.",
            "The lesson-scoped monospace font uses the existing JetBrains Mono site family; final actual-font browser evidence is recorded.",
        ],
    },
    "scope": {
        "mainSections": 14,
        "optionalDepthDisclosures": 4,
        "browserInvestigations": 8,
        "inlineStaticFigures": 4,
        "completePrograms": 13,
        "independentPracticeGroups": 14,
        "externalReferences": 18,
    },
    "limits": [
        "The browser uses bounded, deterministic teaching models and sampled analytical or computed curves, not a general ODE solver or physical benchmark.",
        "Numerical tolerances and sign-change event detection are explained with their limitations; the checks do not certify arbitrary inputs or undetected interior events.",
        "Code and dense tables retain local horizontal scrolling when needed; the actual page and display equations do not overflow at the tested widths.",
        "Video resource metadata and paired written scope were inspected, not full playback. Specialist Frobenius, DAE and nonsmooth courses remain explicitly scoped follow-up.",
        "Root-owned independent review, integrated production checks and user acceptance are not claimed by this author packet.",
    ],
}
destination = ROOT / f"docs/teaching/evidence/{PREFIX}-author-review.json"
destination.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"file": str(destination), "frozenAt": packet["authorFrozenAt"], "sources": len(sources), "openedImages": len(opened_names)}))
