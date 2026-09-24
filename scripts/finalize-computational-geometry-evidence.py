"""Save checked evidence and fingerprints after the author opens final screenshots."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/teaching/evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)

def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))

native = read("scratch/computational-geometry-verification/results.json")
browser = read("scratch/computational-geometry-browser/results.json")
reading = read("scratch/computational-geometry-browser/reading-results.json")
formatting = read("scratch/computational-geometry-verification/formatting-results.json")
assert native["status"] == "passed" and browser["errors"] == []
for relative, recorded in native["source_sha256"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == recorded

sources = formatting["files"]
fingerprints = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                for relative in sources}
opened = [
    "orientation-1440.png", "orientation-390.png",
    "segments-point-1440.png", "segments-overlap-390.png",
    "precision-input-1440.png", "precision-products-390.png", "precision-input-390.png",
    "hull-fence-1440.png", "hull-fence-390.png", "hull-pop-390.png", "hull-edge-coordinates-390.png",
    "polygon-boundary-1440.png", "polygon-boundary-390.png", "polygon-opening-390.png",
    "concave-envelope-1440.png", "concave-envelope-390.png",
    "intro-390.png", "detail-sources-390.png", "detail-calculation-320.png",
    "detail-hull-invariant-390.png", "detail-ray-rule-390.png",
    "detail-python-program-1440.png", "detail-python-output-390.png",
    "detail-edge-point-labels-320.png", "detail-exact-input-320.png",
] + [f"reading-{index}-390.png" for index in range(1, 9)] + [
    f"reading-{index}-1440.png" for index in [1, 4, 5, 8]
]
for name in opened:
    assert (ROOT / "scratch/computational-geometry-browser" / name).exists()

record = {
    "topic_id": "computational-geometry-robust-predicates-convex-hulls",
    "status": "author-verified; root integration and user acceptance separate",
    "frozen_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": fingerprints,
    "native": native,
    "browser": browser,
    "ordinary_reading": reading,
    "formatting": formatting,
    "actually_opened_images": opened,
    "repairs": [
        "Centered point labels so nearby B and D remain visually separate; coincident labels stay grouped.",
        "Reset crossing restores both the original points and endpoint D selection.",
        "Replaced inline Code helper in the first worked determinant calculation with a scoped multiline calculation; verified at 1440,390,320.",
        "Strengthened browser assertions for all six local hidden-answer groups and the optional LeetCode direction-grouping stage."
    ],
    "independent_root_review": {
        "status": "no actionable source/proof defect reported before freeze",
        "scope": "all-boundary monotone chain, half-open polygon crossing/area, precision/construction, support application, canonical line keys and changed local practice",
        "limit": "source/proof review, not additional numerical tests; root will preserve a separate final-fingerprint record"
    },
    "research": {
        "checked_on": "2026-09-10",
        "record": "docs/teaching/COMPUTATIONAL-GEOMETRY-LESSON-DESIGN.md",
        "official_leetcode_ids": [1037, 1232, 812, 836, 587, 149],
        "alternate_video": "MIT6.046 lecture2 official page, hull notes pages1–3 and opening hull transcript inspected; no full playback",
        "written": "Shewchuk author page; CGAL6.2.1 Kernel, Convex Hull2 and Polygon manual contracts; Python Fraction constructors",
        "limits": "No CGAL execution, LeetCode submission/editorial verification, timing benchmark, production adaptive-predicate implementation or novice study claimed."
    }
}
target = EVIDENCE / "computational-geometry-author-review.json"
target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"record": str(target.relative_to(ROOT)), "frozen_at": record["frozen_at"], "source_sha256": fingerprints}, indent=2))
