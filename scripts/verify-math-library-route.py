"""Execute one canonical downloadable math bridge; retain a source-bound receipt.

Usage: python scripts/verify-math-library-route.py <asset-folder> <program.py>
Use the intended environment's python. This is author evidence, not browser QA.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
folder, filename = sys.argv[1:]
source = (ROOT / "public/learn-assets" / folder / filename).resolve()
source.relative_to(ROOT / "public/learn-assets")
if source.suffix != ".py" or not source.is_file():
    raise ValueError("Expected an existing canonical Python asset")
environment = os.environ.copy()
cache = ROOT / "scratch/math-library-runtime/pytensor-cache" / folder
cache.mkdir(parents=True, exist_ok=True)
# Keep compiler caches in the workspace; numerical model/protocol is unchanged.
environment["PYTENSOR_FLAGS"] = f"base_compiledir={cache.as_posix()}"
environment["PYTHONIOENCODING"] = "utf-8"
started = time.time()
result = subprocess.run([sys.executable, str(source)], cwd=ROOT, env=environment,
                        capture_output=True, text=True, encoding="utf-8", timeout=900)
versions = {}
for package in ("numpy", "scipy", "scikit-learn", "torch", "pymc", "pytensor", "arviz", "arviz-stats", "gudhi"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
receipt = dict(source=source.relative_to(ROOT).as_posix(),
               sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               command=f"python {source.relative_to(ROOT).as_posix()}",
               environment=versions, pytensorCache="workspace-local; default compiler backend",
               exitCode=result.returncode, seconds=round(time.time()-started, 3),
               stdout=result.stdout, stderr=result.stderr)
destination = ROOT / "docs/teaching/evidence" / f"{folder}-library-native.json"
destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(result.stdout)
print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)
