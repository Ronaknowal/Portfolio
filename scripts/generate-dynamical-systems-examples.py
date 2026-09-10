"""Execute authored programs before exporting their exact code and stdout."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scratch/dynamical-systems-review/native"
original = json.loads((ROOT / "docs/teaching/evidence/dynamical-systems-original-content.json").read_text(encoding="utf-8"))
(NATIVE / "original_logistic.py").write_text(original["blocks"][0]["code"], encoding="utf-8")
metadata = [
    ("cooling", "cooling_updates", "Compare a rate with three update intervals", "At decay rate 2, will a step of 1.5 preserve cooling? Predict the sign and size of the update multiplier first.", "All three runs end at time 3. The exact state is independent of the chosen output interval; Euler's update is not. A stable approximation can still be inaccurate."),
    ("scalar", "scalar_stability", "Resolve a zero derivative using the nonlinear rule", "Both x′=−x³ and x′=x³ have derivative zero at the origin. What does the actual solution say?", "The negative cubic approaches zero algebraically. The positive cubic can escape to infinity in finite time. The potential-rate values belong to x′=x−x³, a separate two-basin example."),
    ("transient", "planar_transients", "Watch a stable matrix amplify a disturbance", "Both eigenvalues are negative. Must the Euclidean distance to the origin decrease at every instant?", "The off-diagonal term transfers the initially hidden second component into the first. At time ln(2) the norm exceeds 1, although both components eventually decay. This program needs NumPy and SciPy."),
    ("hopf", "hopf_return", "Return to the same angle and compare radii", "For a=0.25, does a starting radius 0.3 approach zero or a circle? How strongly does a small radial error shrink in one turn?", "One turn takes 2π dimensionless time units. The finite difference near radius 0.5 agrees with exp(−π); this tests a radial return multiplier, not disappearance of a phase offset."),
    ("cycles", "logistic_cycles", "Keep the transient separate from the retained tail", "Will every growth parameter above 3.57 produce the same kind of tail? Compare 3.83 with 3.9.", "The original four growth settings are retained, with a period-three window added for comparison. Rounded finite tails suggest behavior; they do not prove an exact long-run classification."),
    ("original", "original_logistic", "Run the original nearby-start experiment", "The initial difference is one millionth. Does its later size alone establish chaos?", "This is the preserved original program and output. The reference begins at the critical point 0.5, where the derivative is zero: the first difference is quadratic, not a generic linear separation step. A finite large difference alone proves neither chaos nor a long-run exponent."),
    ("exponent", "lyapunov_evidence", "Separate a finite average from a dynamical conclusion", "At r=4, x=0.75 stays fixed but has exponent ln(2). What conclusion would a positive-only classifier get wrong?", "The average discards a declared transient and inspects a finite orbit in binary64 arithmetic. An exactly zero derivative produces −infinity, never an arbitrary small replacement. The forecast horizon uses a stipulated positive rate and a small-error approximation."),
    ("tent", "tent_coordinates", "Use exact fractions to follow the folding mechanism", "Which starting interval follows left then right, and which point returns after those two steps?", "Inverse branches recover [1/4,1/2] and the periodic point 2/5. The dyadic experiment reaches zero in finitely many steps: finite-state arithmetic and real-number chaotic dynamics are different objects."),
    ("lorenz", "lorenz_integration", "Compare two short integrations of one initial state", "If two requested tolerances agree over one time unit, what has been checked and what remains unproved?", "DOP853 is run twice with a common output grid. This checks local numerical reproducibility over the stated short horizon, not decades of forecast accuracy, a physical model, or a theorem of chaos. Needs NumPy and SciPy."),
    ("energy", "oscillator_energy", "Measure numerical energy over the same physical time", "Will the method with bounded oscillating energy be exactly energy-conserving? Compare its ordinary and modified energies.", "Forward Euler adds energy systematically. Kick-then-drift symplectic Euler preserves a step-dependent quadratic form in exact arithmetic; ordinary energy oscillates. Every run ends at time 20."),
    ("products", "jacobian_products", "Multiply changing Jacobians in the actual order", "Each matrix has only zero eigenvalues. Can alternating the two nevertheless magnify a state?", "The product BA has eigenvalue 4. Derivative propagation uses ordered matrix products; averaging the separate eigenvalues loses how one step rotates or transfers the next step's input. Needs NumPy."),
]
examples = {}
records = []
for key, filename, title, question, interpretation in metadata:
    path = NATIVE / (filename + ".py")
    if key != "original":
        subprocess.run([sys.executable, "-m", "black", "--quiet", str(path)], check=True)
    code = path.read_text(encoding="utf-8")
    result = subprocess.run([sys.executable, str(path)], check=True, text=True, capture_output=True, timeout=30)
    if result.stderr:
        raise RuntimeError(result.stderr)
    expected = result.stdout.rstrip("\n")
    examples[key] = dict(title=title, question=question, language="python", code=code, expected=expected, interpretation=interpretation)
    records.append(dict(id=key, file=str(path.relative_to(ROOT)), codeSha256=hashlib.sha256(code.encode()).hexdigest(), stdout=expected))
assert examples["original"]["code"] == original["blocks"][0]["code"]
assert examples["original"]["expected"] == original["blocks"][1]["code"]
target = ROOT / "src/learn/data/dynamical-systems-examples.js"
target.write_text("// Complete Python programs with independently executed expected output.\nexport const dynamicalSystemsExamples = " + json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
record = dict(executedAt=datetime.now(timezone.utc).isoformat(), python=sys.version, records=records, exportedFile=str(target.relative_to(ROOT)), exportedSha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT / "scratch/dynamical-systems-review/native-results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(dict(executedAt=record["executedAt"], programs=len(records), exportedSha256=record["exportedSha256"], outputs={item["id"]: item["stdout"] for item in records}), indent=2))
