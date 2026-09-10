"""Format new programs without changing their AST; execute every displayed output."""
import ast
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import black

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/optimal-transport-review/native"
examples = json.loads((DIRECTORY / "input.json").read_text(encoding="utf8"))
formatting = []
for key, example in examples.items():
    before = example["code"]
    code = before if key == "originalScaling" else black.format_str(before, mode=black.Mode(line_length=88)).strip()
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(before)), key
    (DIRECTORY / f"{key}.py").write_text(code + "\n", encoding="utf8")
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)
    assert not result.stderr, (key, result.stderr)
    example["code"] = code
    example["expected"] = result.stdout.strip()
    formatting.append({"key": key, "astPreserved": True, "stdout": example["expected"]})
(DIRECTORY / "prepared.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf8")
(DIRECTORY / "capture-results.json").write_text(json.dumps({
    "at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
    "black": black.__version__, "programs": formatting,
}, indent=2) + "\n", encoding="utf8")
print(json.dumps({key: item["expected"] for key, item in examples.items()}, indent=2))
