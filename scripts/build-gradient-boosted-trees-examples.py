"""Execute complete owned lesson programs and write their literal output bundle."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "scripts/gradient-boosted-trees-programs"
ARCHIVE = ROOT / "docs/teaching/archive/gradient-boosted-trees-before-rewrite"

# Preserve the old substantive recursive learner and exact original random dataset.
# Repairs are individually inspectable; the archive remains unchanged.
recursive = (ARCHIVE / "original-program-1.py").read_text(encoding="utf-8")
recursive = recursive.replace("DecisionStump", "RegressionTree")
recursive = recursive.replace('"""Shallow regression tree for use as a weak learner."""',
                              '"""Greedy regression tree, with an explicit depth cap."""')
recursive = recursive.replace("        feat, thr, lm, rm = self._best_split(X, r)\n",
                              "        feat, thr, lm, rm = self._best_split(X, r)\n"
                              "        if lm is None:  # Constant features cannot create two children.\n"
                              "            return {'leaf': True, 'value': np.mean(r)}\n")
recursive = recursive.replace("        self.F0 = np.mean(y)\n",
                              "        self.trees = []  # Refit replaces, rather than appends to, the ensemble.\n"
                              "        self.F0 = np.mean(y)\n")
recursive = recursive.replace("# gradient of MSE = -(y - F)", "# negative gradient of half-squared loss")
(PROGRAMS / "recursive-regression.py").write_text(recursive + "\n", encoding="utf-8")

specifications = [
    ("residual-corrections", "Fit three actual correction stumps", "Starting from the mean of six targets, what does each half-strength correction change, and how is a new row predicted?"),
    ("recursive-regression", "A complete deeper-tree ensemble", "Can the original 100-row regression example be reproduced with a depth-2 tree, and what predictions does its actual implementation produce?"),
    ("loss-derivatives", "Differentiate the score, not the probability", "For a positive label with score −2, what are the loss gradient and curvature, and what does adding one score unit do?"),
    ("regularized-split", "Price the split and threshold the gradient", "When do L1 shrinkage and a new-leaf cost suppress a split of the five-row fixture?"),
    ("histogram-missing", "Restrict thresholds and learn a missing route", "How do a coarse histogram and a changed missing-row target affect the best split?"),
    ("goss-expectation", "Enumerate the sampling distribution", "Does unbiasedly estimating the gradient sum also preserve its square? Calculate every two-of-four sample."),
    ("exclusive-bundles", "Pack bins only when decoding is possible", "Which nonzero bin ranges can share one column, and what goes wrong when two features are active?"),
    ("ordered-categories", "Encode before revealing the row's target", "After flipping row 3's target, which prefix encodings change and which must stay unchanged?"),
    ("ordered-prefix-models", "Train models on earlier prefixes", "How can each training residual be obtained from a model that has not trained on that row? Trace two constant-learner rounds."),
    ("xgboost-workflow", "Run and preserve XGBoost's selected model", "How do native and sklearn prediction agree after validation selects an iteration, and how do save/reload and a monotonicity check work?"),
    ("lightgbm-workflow", "Run LightGBM with explicit stopping", "How is the validation-selected iteration count used at inference, and how do split count and gain differ?"),
    ("catboost-workflow", "Run CatBoost's explicitly Ordered regression", "Which trees remain after validation selects the best iteration, and what does the locked model achieve on untouched synthetic test rows?"),
    ("catboost-categories", "Use categorical rows and align prediction shapes", "Can categorical classification handle an unseen category, and how can a broadcasting mistake silently corrupt accuracy?"),
    ("native-categorical-splits", "Declare a categorical schema in two more libraries", "How can XGBoost and LightGBM use native categoricals while preserving the meaning of category labels across prediction rows?"),
    ("quantile-planning", "Choose the cost of underprediction", "Which prediction minimizes 80th-percentile pinball loss for the constructed spare-part demand, and why need the optimum not be unique?"),
]

examples = {}
records = []
for key, title, question in specifications:
    path = PROGRAMS / f"{key}.py"
    code = path.read_text(encoding="utf-8").strip()
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError(f"{key}:\n{result.stderr}\n{result.stdout}")
    if result.stderr.strip():
        raise RuntimeError(f"Unexpected stderr for {key}: {result.stderr}")
    output = result.stdout.strip()
    examples[key] = {"title": title, "question": question, "language": "python", "code": code, "expected": output}
    records.append({"key": key, "program": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "codeSha256": hashlib.sha256(code.encode()).hexdigest(),
                    "stdout": output, "returncode": result.returncode})

destination = ROOT / "src/learn/data/gradient-boosted-trees-examples.js"
destination.write_text("// Complete programs executed by scripts/build-gradient-boosted-trees-examples.py.\n"
                       "export const gradientBoostedTreeExamples = "
                       + json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
evidence = ROOT / "scratch/gradient-boosted-trees-verification"
evidence.mkdir(parents=True, exist_ok=True)
record = {"executedAt": datetime.now(timezone.utc).isoformat(), "python": sys.version,
          "programCount": len(records), "programs": records,
          "examplesSha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
(evidence / "program-results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in record.items() if key != "programs"}, indent=2))
