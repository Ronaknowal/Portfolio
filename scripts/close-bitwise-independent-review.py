"""Close the independently inspected bitwise extension without changing its author packet."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def hashed(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


author_path = "docs/teaching/evidence/bitwise-foundations-author-review.json"
author = read(author_path)
sources = [{"path": item["path"], "sha256": item["sha256"]} for item in author["sources"]]
for item in sources:
    assert hashed(item["path"]) == item
native_path = "scratch/bitwise-independent-review/native-results.json"
browser_path = "scratch/bitwise-independent-review/browser/results.json"
native, browser = read(native_path), read(browser_path)
assert native["sources"] == browser["sources"] == sources
assert native["status"] == browser["status"] == "passed"
images = ["scratch/bitwise-independent-review/browser/" + name + ".png" for name in [
    "seven-place-shift-320", "five-place-borrow-390", "zero-survivor-1440",
    "changed-high-set-320", "reading-place-values-390", "negative-weight-1440",
    "reading-partition-proof-390", "separating-partition-320",
    "bitmap-transfer-feedback-390", "invalid-keeps-finished-trace-320",
]]
packet = {
    "reviewedAt": datetime.now(timezone.utc).isoformat(),
    "topicId": "arrays-strings-hash-maps",
    "scope": "Bounded independent review of the bitwise extension; preservation of original sections 1–8, programs and problem objects separately checked.",
    "status": "independent review passed; production integration remains separate",
    "reviewer": "/root/testing_documentation_completion",
    "authorFreeze": author["frozenAt"],
    "authorPacket": hashed(author_path),
    "sources": sources,
    "originalArchives": [hashed(path) for path in [
        "docs/teaching/evidence/bitwise-foundations-original.json",
        "docs/teaching/evidence/bitwise-foundations-original-sources.json",
    ]],
    "native": {"record": hashed(native_path), "results": native},
    "browser": {"record": hashed(browser_path), "results": browser},
    "actuallyOpenedImages": [hashed(path) for path in images],
    "scripts": [hashed(path) for path in [
        "scripts/verify-bitwise-independent.mjs", "scripts/verify-bitwise-independent.py",
        "scripts/review-bitwise-independent.cjs", "scripts/close-bitwise-independent-review.py",
    ]],
    "sourceRead": [
        "Complete frozen body and extension design/brief; exact inventory and destination/inbox note.",
        "All five new native programs, model functions, all four lab components and computed partition figure, CSS, four changed practice solutions and five new problem annotations.",
        "Existing sections and six programs read for context; their preservation independently compared with the original byte archive.",
    ],
    "findings": [],
    "productionEditsByReviewer": [],
    "primaryResourcesChecked": [
        "https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types",
        "https://leetcode.com/problems/single-number/",
        "https://leetcode.com/problems/number-of-1-bits/",
        "https://leetcode.com/problems/power-of-two/",
        "https://leetcode.com/problems/hamming-distance/",
        "https://leetcode.com/problems/single-number-iii/",
    ],
    "limits": "No whole-catalogue audit, judge submissions/editorial access, video viewing, physical-phone or complete screen-reader test, timing benchmark or arbitrary-input formal verification. Author broader tests remain separately attributed.",
}
destination = ROOT / "docs/teaching/evidence/bitwise-foundations-independent-review.json"
assert not destination.exists(), "Preserve prior independent evidence."
destination.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"reviewedAt": packet["reviewedAt"], "sources": len(sources), "openedImages": len(images), "status": "passed"}))
