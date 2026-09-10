"""Keep the original program unchanged; format and execute all complete new examples."""
import ast
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import black

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/rate-distortion-review/native"
examples = json.loads((DIRECTORY / "input.json").read_text(encoding="utf8"))
results = []
for name, example in examples.items():
    original = example["code"]
    code = original if name == "originalBinary" else black.format_str(original, mode=black.Mode(line_length=88)).strip()
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(code)), name
    (DIRECTORY / f"{name}.py").write_text(code + "\n", encoding="utf8")
    run = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)
    assert not run.stderr, (name, run.stderr)
    example.update(code=code, expected=run.stdout.strip())
    results.append({"name": name, "astPreserved": True, "stdout": example["expected"]})
(DIRECTORY / "prepared.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf8")
(DIRECTORY / "capture-results.json").write_text(json.dumps({
    "at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
    "black": black.__version__, "programs": results,
}, indent=2) + "\n", encoding="utf8")
print(json.dumps({name: example["expected"] for name, example in examples.items()}, indent=2))
