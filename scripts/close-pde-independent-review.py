"""Bind the bounded independent PDE review to its actual amended source."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def hashed(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


author_path = "docs/teaching/evidence/pde-author-review.json"
author = read(author_path)
native_path = "scratch/pde-independent-review/results.json"
browser_path = "scratch/pde-independent-review/browser/results.json"
native, browser = read(native_path), read(browser_path)
for source in author["sourceHashes"]:
    assert hashed(source["path"]) == source
assert native["sourceHashes"] == browser["sourceHashes"] == author["sourceHashes"]
assert native["status"] == browser["status"] == "passed"
archive_path = "docs/teaching/evidence/pde-independent-amendment-originals.json"
archive = read(archive_path)
for item in archive["archived"]:
    assert hashed(item["archive"])["sha256"] == item["sha256"]
old = read(archive["previousPacket"])
assert old["frozenAt"] == "2026-09-11T08:18:02.248Z"
assert old == read("scratch/pde-independent-review/initial-author-packet.json")
images = [
    "scratch/pde-independent-review/browser/" + name + ".png"
    for name in [
        "transport-inflow-390", "wave-dependence-320", "poisson-offset-family-1440",
        "harmonic-top-negative-390", "expansion-gap-320", "reading-11-320",
        "changed-entropy-practice-390", "reading-8-1440",
    ]
] + [
    "scratch/pde-wave-boundary/maximum-final-320.png",
    "scratch/pde-wave-boundary/arithmetic-final-320.png",
]
packet = {
    "reviewedAt": datetime.now(timezone.utc).isoformat(),
    "topicId": author["topicId"],
    "status": "bounded independent review passed; production integration remains separate",
    "reviewer": "/root/testing_documentation_completion",
    "authorFreeze": author["frozenAt"],
    "authorPacket": hashed(author_path),
    "sourceHashes": author["sourceHashes"],
    "preservedOriginalAuthorPacket": hashed(archive["previousPacket"]),
    "preservedArchiveManifest": hashed(archive_path),
    "preservedArchiveEntriesVerified": len(archive["archived"]),
    "initialFindingEvidence": hashed("scratch/pde-independent-review/initial-findings.json"),
    "native": {"record": hashed(native_path), "result": native},
    "browser": {"record": hashed(browser_path), "result": browser},
    "reviewScripts": [hashed(path) for path in [
        "scripts/verify-pde-independent.mjs", "scripts/verify-pde-independent.py",
        "scripts/review-pde-independent.cjs", "scripts/close-pde-independent-review.py",
    ]],
    "actuallyOpenedImages": [hashed(path) for path in images],
    "imageAttribution": "First eight captured by this reviewer; last two captured by author and opened by this reviewer after repair.",
    "resolvedFindings": [
        "Negative wave integral from cancellation at accepted tiny positive time; paired JS/Python positive Bernstein evaluation passes exact stored-input Fraction comparisons.",
        "Heat maximum principle now first works on T'<T, then extends by continuity, respecting the stated interior derivative assumptions.",
        "Forced-rod capstone explicitly restores s/(2k)=1 K/m^2 and SI-coordinate/rate units.",
    ],
    "scope": "Complete actual body, all 15 programs, pure models, labs/CSS, 14 changed practice solutions, design/brief/incoming note; bounded complementary math/native/browser checks.",
    "limits": "No numerical test proves a general PDE theorem. No judge, screen-reader, physical phone or observed learner session; author broader suite separately attributed. No production files or shared ledger edited by reviewer.",
}
destination = ROOT / "docs/teaching/evidence/pde-independent-review.json"
assert not destination.exists(), "Preserve prior independent evidence rather than overwrite it."
destination.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"reviewedAt": packet["reviewedAt"], "sources": 6, "openedImages": len(images), "status": "passed"}))
