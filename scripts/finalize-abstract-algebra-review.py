"""Freeze the completed author evidence without promoting independent/integration status."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/evidence/abstract-algebra-author-review.json"
assert not PACKET.exists(), "Preserve an existing author freeze; use an explicit amendment."


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def fingerprint(path):
    data = (ROOT / path).read_bytes()
    return {"path": path, "sha256": sha256(data).hexdigest(), "bytes": len(data)}


native_path = "scratch/abstract-algebra-verification/results.json"
browser_path = "scratch/abstract-algebra-browser/results.json"
reading_path = "scratch/abstract-algebra-browser/final-reading-results.json"
native, browser, reading = map(read_json, [native_path, browser_path, reading_path])
assert native["status"] == browser["status"] == reading["status"] == "passed"
sources = [fingerprint(row["path"]) for row in native["productionSources"]]
assert len(sources) == 6
for result in [native, reading]:
    assert [{k: row[k] for k in ["path", "sha256"]} for row in sources] == result["productionSources"]
display_only = {
    "src/learn/components/lesson-labs/AbstractAlgebraLabs.jsx",
    "src/learn/components/lesson-labs/abstract-algebra-labs.css",
}
current = {row["path"]: row["sha256"] for row in sources}
changes = []
for row in browser["productionSources"]:
    if row["sha256"] != current[row["path"]]:
        assert row["path"] in display_only
        changes.append({"path": row["path"], "before": row["sha256"], "after": current[row["path"]]})
assert [row["width"] for row in browser["results"]] == [1440, 390, 320]
assert all(row["states"] == 44 and not row["errors"] and not row["documentOverflow"] for row in browser["results"])
assert all(row["publicFont"] and row["actualPrograms"] == 10 and not row["errors"] and not row["documentOverflow"] for row in reading["results"])

opened_names = [
    "final-normal-coset-390.png", "final-cayley-scrolled-320.png",
    "final-projection-proof-320.png", "final-changed-solution-390.png",
    "final-reading-3-390.png", "final-reading-5-320.png",
    "final-reading-6-1440.png", "final-modular-320.png",
    "final-references-390.png", "final-averaged-map-1440.png",
    "final-averaged-map-320.png", "final-composition-320.png",
    "final-program-390.png",
]
image_index = {row["path"]: row["sha256"] for row in reading["images"]}
opened = []
for name in opened_names:
    record = fingerprint(f"scratch/abstract-algebra-browser/{name}")
    assert record["sha256"] == image_index[record["path"]]
    opened.append({**record, "opened": True, "reviewer": "author", "method": "Actual view_image display inspected after final browser capture"})

now = datetime.now(timezone.utc).isoformat()
opened_record = {"recordedAt": now, "images": opened, "scope": "These thirteen final files were actually opened and visually inspected; the larger captured set is not claimed as opened."}
(ROOT / "scratch/abstract-algebra-browser/opened-images.json").write_text(json.dumps(opened_record, indent=2) + "\n", encoding="utf-8")

archive_root = ROOT / "scratch/abstract-algebra-author-freeze"
assert not archive_root.exists(), "Do not overwrite frozen source snapshots."
archives = []
for row in sources:
    target = archive_root / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / row["path"], target)
    archived = fingerprint(target.relative_to(ROOT).as_posix())
    assert archived["sha256"] == row["sha256"]
    archives.append(archived)

hash_table = "\n".join(f"| `{row['path']}` | `{row['sha256']}` |" for row in sources)
verification = f"""# Abstract Algebra, Groups & Symmetry Actions — author verification

Frozen **{now}**. Stable ID `abstract-algebra-groups-symmetry-actions`, mathematics position 55. **Author-verified; independent review, production integration and user acceptance remain separate.** The [immutable author packet](evidence/abstract-algebra-author-review.json) embeds actual native and browser payloads, exact final sources and the thirteen final screenshots actually opened. No broader curriculum completion is claimed.

## Scope and preservation

The [assessed design](ABSTRACT-ALGEBRA-LESSON-DESIGN.md) records the exact inventory, incoming-note assessment, prerequisites, claim-specific research and scope decisions. The [original planned baseline](evidence/abstract-algebra-original-plan.json) is preserved byte-for-byte (SHA-256 `9ac89cb3ff652c61ff86d0da23ed3aef10d896987aa602c8dd3fefa8164aeee9`). This identity had no legacy authored body/program/output to replace. Its square composition, actions and equivariant-map outcomes are retained and developed, with stable title, route and progress identity.

The lesson teaches the shape/transformation/configuration distinction before group terminology. Eleven sections connect left-action composition, subgroups/cosets, orbits and stabilizers, fixed-point counting, homomorphisms/normal quotients, representations, all-input equivariance, finite averaging, legal modular division and independent synthesis. Complete local proofs retain their finite-group, normality and orthogonal-representation hypotheses. Rings use an explicit unital convention; continuous groups and full algebra courses remain further study. The next route stays PDE, with its separate calculus readiness.

Four distinct investigations expose composition routes, unique coloring orbits, representative-dependent products and the two routes through an actual four-channel map. The left Cayley diagram distinguishes eight transformations from a two-configuration action graph with reflection loops. Modular multiplication shows actual preimages. Eight changed tasks have hints and worked solutions; two earlier checkpoints reveal nonempty reasoning. The final changed audit reports exact coefficients `(19/2,35/4,33/4)`, output `(133/4,67/2,37,149/4)`, defect zero and invariant mean `141/4 V`, while distinguishing a symmetry certificate from correct physical calibration.

## Actual native and independent-oracle verification

`node scripts/verify-abstract-algebra.mjs` passed **{native['checkedAt']}** on the exact final six-source fingerprint. [Raw result](../../scratch/abstract-algebra-verification/results.json), [JavaScript driver](../../scripts/verify-abstract-algebra.mjs), [independent Python oracle](../../scripts/verify-abstract-algebra.py).

- All **ten actual displayed Python programs** executed and matched their complete stored stdout. They use the standard library; SymPy 1.14.0 is a verifier dependency. The [generator](../../scripts/generate-abstract-algebra-examples.py) formats complete programs with Black while asserting normalized Python AST conservation before output capture.
- All 64 square compositions and tracked routes agree with separately constructed exact geometric matrices and distance-preserving permutations. All 256 subset candidates/generator choices, ten subgroup coset partitions, 162 ternary action states and 702 reaching-transform records agree with independent enumeration. Fixed-color counts and six Burnside totals agree with full orbit enumeration.
- A separately solved symbolic commutant has dimension three. Thirty-two changed full-square averages agree with independent self/neighbor/opposite relation averages; rotation averages agree with cyclic-offset averages. Exact Frobenius identities and actual Python changed averages pass. On 1,280 map states, both routes, the input-specific defects, full matrix certificates and transformed-input pooled means match independent rational matrix calculations.
- All 649 bounded modular equations and unit/gcd cases pass. Twenty-one invalid model cases reject as specified. Changed capstone and ternary/chiral/modular practice calculations are independently checked; a constant-input false positive is retained as a teaching counterexample.

These are checks of current production exports and actual stored programs, not recycled design fixture counts. Bounded exact integers and binary-fraction controls make zero meaningful. The helpers are educational finite models, not a verified arbitrary-group or arbitrary-range numerical library. General mathematical proofs are separately present and read; finite enumeration alone is not presented as their proof.

## Actual browser and visual evidence

The full [interaction script](../../scripts/review-abstract-algebra-lesson.cjs) passed **{browser['checkedAt']}**, **44 meaningful operated states at each of 1440, 390 and 320 pixels**. Its [raw result](../../scratch/abstract-algebra-browser/results.json) retains that source snapshot. Actual Space Grotesk loaded; all four investigations, resets, changed composition/color/group choices, invalid-draft preservation/repair, all map modes and tied sliders, matrix certificates, numeric positions and values, all hint/solution disclosures and local keyboard scrolling passed. It also compared all ten actual visible questions, complete code and output with their stored examples; checked eleven anchors, ten checkpoint/practice blocks and ten display equations; and found no lesson console/page errors, failed lesson requests or document overflow.

The subsequent opened screenshot pass found a clipped phone option, and completed small display improvements: a visible horizontal-scroll instruction, explicit table column scope, loop labels separated from their curves, shortened select labels and full-row phone controls. Only the owned lab and CSS differ from the full interaction snapshot. The final [reading script](../../scripts/review-abstract-algebra-reading.cjs) passed **{reading['checkedAt']}** on the frozen source, with all three widths, actual font, eleven ordinary reading captures per width, all ten displayed programs and equations, graph-label geometry, separated sensor/position labels, all dropdown option text widths, representative choices, local keyboard scrolling, the deeper projection proof and changed capstone. [Final reading payload](../../scratch/abstract-algebra-browser/final-reading-results.json) records this narrower closure honestly; it does not relabel the earlier full suite as run on the amended files.

Thirteen final captures were actually opened, read and hashed in [opened-image evidence](../../scratch/abstract-algebra-browser/opened-images.json): ordinary subgroup/fixed-point/homomorphism reading, composition, keyboard-scrolled Cayley/action diagrams, corrected normal-coset controls, large sensor outputs on desktop/phone, the projection proof, changed answer, modular map, complete code/output context and annotated references. Long tables/code and the eight-node graph intentionally scroll locally; page-level overflow is absent. Earlier reading failures due to wide equations and sensor labels were repaired before the full passing run. A first oracle failure came from a SymPy floating-versus-rational equality assertion and was corrected in the harness; it is not described as an application algorithm repair.

## Research, handoff and limits

The design records the exact inspected portions of Judson's textbook, Clemson's official course page and matching dihedral/action notes, selected 3Blue1Brown written explanation, and the original Cohen–Welling equivariance paper. Direct useful video links are paired with substantive notes. **No full video playback or transcript review is claimed.** The lesson's references annotate level, left/right-action convention differences and runtime age; they supplement self-contained teaching.

The [geometric-learning destination note](topic-notes/graph-transformers-geometric-deep-learning.md) persists the input/output action contract, finite-versus-continuous averaging, permutation-versus-geometric nonlinearities and boundary/modeling caveats. Its origin is now author-verified; its receiving-topic status stays open. No destination body or shared curriculum file was edited by this authoring closure.

No independent reviewer, production build/integration, screen-reader session, empirical beginner study, arbitrary browser engine or deployment is claimed here. Root owns independent review and shared integration. All source remains frozen for that review; a future amendment must preserve this original packet and source archive separately.

## Frozen production sources

| Source | SHA-256 |
| --- | --- |
{hash_table}

The six exact source bytes are also archived under `scratch/abstract-algebra-author-freeze/`, and fingerprints are embedded in the author packet.
"""
verification_path = "docs/teaching/ABSTRACT-ALGEBRA-VERIFICATION.md"
(ROOT / verification_path).write_text(verification, encoding="utf-8")
supporting_paths = [
    "docs/teaching/ABSTRACT-ALGEBRA-LESSON-DESIGN.md", verification_path,
    "docs/teaching/evidence/abstract-algebra-original-plan.json",
    "docs/teaching/topic-notes/graph-transformers-geometric-deep-learning.md",
    "scripts/verify-abstract-algebra-design.py", "scripts/generate-abstract-algebra-examples.py",
    "scripts/verify-abstract-algebra.mjs", "scripts/verify-abstract-algebra.py",
    "scripts/review-abstract-algebra-lesson.cjs", "scripts/review-abstract-algebra-reading.cjs",
    "scripts/format-abstract-algebra-source.cjs", "scripts/finalize-abstract-algebra-review.py",
]
packet = {
    "topicId": "abstract-algebra-groups-symmetry-actions", "frozenAt": now,
    "status": "author-verified; independent review and production integration pending",
    "productionSources": sources, "supportingSources": list(map(fingerprint, supporting_paths)),
    "sourceArchive": archives, "native": native, "browser": browser,
    "finalReading": reading, "openedImages": opened_record,
    "displayAmendment": {"changedSources": changes, "scope": reading["scope"], "fullBehaviorRerunClaimed": False},
    "rawEvidence": list(map(fingerprint, [native_path, browser_path, reading_path, "scratch/abstract-algebra-browser/opened-images.json"])),
    "originalConservation": "Planned baseline archived unchanged; no legacy body or program existed.",
    "limits": ["Independent review pending", "Production integration belongs to root", "No user acceptance or beginner study claimed", "No full video/transcript review claimed", "Bounded finite educational models; local table/code/graph scrolling is intentional"],
}
PACKET.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
assert all(fingerprint(row["path"])["sha256"] == row["sha256"] for row in sources)
print(json.dumps({"frozenAt": now, "status": packet["status"], "sources": sources, "openedImages": len(opened), "packet": PACKET.relative_to(ROOT).as_posix()}, indent=2))
