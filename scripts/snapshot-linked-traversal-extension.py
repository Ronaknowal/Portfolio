"""Preserve the linked lesson and actual baseline outputs before a scoped extension."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
destination = root / "scratch/linked-traversal-extension-original"
assert not destination.exists(), "Never overwrite original review/source evidence."
paths = [
    "src/learn/data/topics/linked-lists-stacks-queues.jsx",
    "src/learn/data/practice/linked-lists-stacks-queues.js",
    "src/learn/data/curriculum/blueprints/linked-lists-stacks-queues.js",
    "src/learn/data/linked-foundations-examples.js",
    "src/learn/data/linked-foundations-model.js",
    "src/learn/components/lesson-labs/LinkedFoundationsLabs.jsx",
    "docs/teaching/systems-and-structures-design.md",
    "SYSTEMS-STRUCTURES-IMPLEMENTATION.md",
    "docs/teaching/DSA-PRACTICE-OWNERSHIP-REVIEW.md",
]
records = []
for relative in paths:
    source = root / relative
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    records.append({"path": relative, "archive": target.relative_to(root).as_posix(), "sha256": sha256(source.read_bytes()).hexdigest()})
inventory = subprocess.check_output(["node", "scripts/build-curriculum-inventory.mjs", "--topic", "linked-lists-stacks-queues"], cwd=root, text=True, encoding="utf-8")
(destination / "inventory.json").write_text(inventory, encoding="utf-8")
examples = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", "import {linkedExamples} from './src/learn/data/linked-foundations-examples.js'; console.log(JSON.stringify(linkedExamples));"], cwd=root, text=True, encoding="utf-8"))
for name, example in examples.items():
    completed = subprocess.run([sys.executable, "-c", example["code"]], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True)
    assert completed.stdout.strip() == example["output"].strip(), name
(destination / "actual-examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
practice = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", "import practice from './src/learn/data/practice/linked-lists-stacks-queues.js'; console.log(JSON.stringify(practice));"], cwd=root, text=True, encoding="utf-8"))
(destination / "practice.json").write_text(json.dumps(practice, indent=2), encoding="utf-8")
result = {"snapshotAt": datetime.now(timezone.utc).isoformat(), "status": "original files archived and all six actual native outputs passed before mutation", "sources": records, "nativePrograms": list(examples), "practiceProblems": [problem["number"] for group in practice["groups"] for problem in group["problems"]]}
(root / "docs/teaching/evidence/linked-traversal-extension-original.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
