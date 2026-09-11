"""Preserve the first freeze and record the independent-review validation fix."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(relative):
    data = (ROOT / relative).read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


prior_path = "docs/teaching/evidence/sets-logic-author-review-before-sparse-validation.json"
packet = read_json(prior_path)
model_path = "src/learn/data/sets-logic-models.js"
previous_sources = packet["sources"]
for source in previous_sources:
    if source["path"] != model_path:
        assert fingerprint(source["path"]) == source
prior_identity = {"authorFrozenAt": packet["authorFrozenAt"], "packet": fingerprint(prior_path)}
before = read_json("scratch/sets-logic-verification/sparse-validation-before.json")
assert all(case["accepted"] for case in before["records"])
models_path = "scratch/sets-logic-verification/model-results.json"
native_path = "scratch/sets-logic-verification/native-results.json"
models, native = read_json(models_path), read_json(native_path)
assert models["passed"] and native["passed"]
assert models["counts"]["sparseInputsRejected"] == 10
for name, count in packet["records"]["models"]["result"]["counts"].items():
    assert models["counts"][name] == count
assert native["counts"]["completePrograms"] == 11
packet["authorFrozenAt"] = datetime.now(timezone.utc).isoformat()
packet["sources"] = [fingerprint(source["path"]) for source in previous_sources]
packet["priorFreeze"] = prior_identity
packet["records"]["models"] = {"file": fingerprint(models_path), "result": models}
packet["records"]["native"] = {"file": fingerprint(native_path), "result": native}
packet["independentReviewAmendment"] = {
    "at": packet["authorFrozenAt"],
    "reviewer": "root; independent of the author",
    "finding": "Sparse arrays bypassed callback-only value checks in inspectQuantifiers, diagonalSubset and inspectRelation. The accepted sparse relation pair wrote matrix[0].undefined=true.",
    "before": before,
    "repair": "Topic-local Boolean-square validation checks every own row and cell index. Relation pairs require both own indices before integer-range checks. Missing input is rejected rather than treated as false or a relation index.",
    "regressions": "Ten sparse row/board/pair/list cases reject TypeError, including selected zero-sized quantifier domains. Existing valid empty sets and all dense states remain accepted and checked.",
    "unchanged": "Other five production files are byte-identical. All11 code/output programs and previous dense model cases pass again.",
    "browserScope": "Original final browser/keyboard/images remain attributed to the prior freeze. This amendment changes only malformed exported inputs absent from every UI-produced board/pair; no new author browser run or screenshot is claimed. Parent independent browser review is separate.",
    "priorModel": next(source for source in previous_sources if source["path"] == model_path),
    "finalModel": fingerprint(model_path),
    "support": [fingerprint("scripts/verify-sets-logic-models.mjs"), fingerprint("scripts/verify-sets-logic-native.py"), fingerprint("scripts/amend-sets-logic-validation.py")],
}
(ROOT / "docs/teaching/evidence/sets-logic-author-review.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"refrozenAt": packet["authorFrozenAt"], "model": fingerprint(model_path), "otherProductionHashesUnchanged": True, "sparseRegressions": 10}))
