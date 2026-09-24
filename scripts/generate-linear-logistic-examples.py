"""Execute every standalone lesson program before exporting its observed stdout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = []


def example(identifier, title, question, code):
    code = textwrap.dedent(code).strip() + "\n"
    run = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError(f"Program {identifier} failed: {run.stderr}")
    if run.stderr:
        raise RuntimeError(f"Unexpected stderr in {identifier}: {run.stderr}")
    EXAMPLES.append({"id": identifier, "title": title, "question": question, "code": code, "expected": run.stdout.strip()})


example("scalar-fit", "Fit the four shipments", "Will the best line pass through every observation? Check its residual sum as well as its loss.", r'''
    from statistics import mean

    distance = [0, 1, 2, 3]  # hundreds of kilometres
    hours = [1, 2, 2, 4]     # synthetic elapsed hours
    x_mean, y_mean = mean(distance), mean(hours)
    slope = sum((x-x_mean)*(y-y_mean) for x, y in zip(distance, hours)) / sum((x-x_mean)**2 for x in distance)
    intercept = y_mean - slope*x_mean
    predictions = [intercept + slope*x for x in distance]
    residuals = [y-p for y, p in zip(hours, predictions)]
    mse = mean(error**2 for error in residuals)
    baseline_mse = mean((y-y_mean)**2 for y in hours)
    print(f"intercept={intercept:.3f}; slope={slope:.3f}")
    print("predictions:", [round(value, 3) for value in predictions])
    print("residuals:", [round(value, 3) for value in residuals])
    print(f"MSE={mse:.3f}; mean-baseline MSE={baseline_mse:.3f}")
    print(f"sum residuals={sum(residuals):.8f}; training R2={1-mse/baseline_mse:.3f}")
''')

example("least-squares", "Solve without forming a Gram-matrix inverse", "Two columns contain the same distance. Can their coefficients be unique while the fitted predictions are unchanged?", r'''
    import numpy as np

    x = np.array([0., 1., 2., 3.])
    y = np.array([1., 2., 2., 4.])
    design = np.column_stack([np.ones(len(x)), x])
    weights, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    print("ordinary coefficients:", np.round(weights, 3).tolist(), "rank:", rank)
    print("normal-equation residual:", np.round(design.T @ (y-design @ weights), 8).tolist())
    duplicate_design = np.column_stack([np.ones(len(x)), x, x])
    minimum_norm, _, duplicate_rank, _ = np.linalg.lstsq(duplicate_design, y, rcond=None)
    alternative = minimum_norm + np.array([0., 5., -5.])
    print("minimum-norm coefficients:", np.round(minimum_norm, 3).tolist(), "rank:", duplicate_rank)
    print("alternative coefficients:", np.round(alternative, 3).tolist())
    print("same predictions:", np.allclose(duplicate_design @ minimum_norm, duplicate_design @ alternative))
    print("singular values:", np.round(singular_values, 4).tolist())
''')

example("gradient-descent", "Trace real parameter updates", "The largest Hessian eigenvalue determines the stable fixed-step range. Which of these rates will fail?", r'''
    import numpy as np

    x = np.array([0., 1., 2., 3.])
    design = np.column_stack([np.ones(len(x)), x])
    y = np.array([1., 2., 2., 4.])
    hessian = 2*design.T @ design/len(y)
    largest = np.linalg.eigvalsh(hessian)[-1]
    print(f"largest eigenvalue={largest:.6f}; stable rate upper bound={2/largest:.6f}")
    for rate in [0.05, 0.2, 0.3]:
        weights = np.zeros(2)
        for step in range(17):
            errors = design @ weights-y
            if step in [0, 1, 16]:
                print(f"rate={rate:.2f}; step={step:2d}; b={weights[0]:.6f}; w={weights[1]:.6f}; MSE={np.mean(errors**2):.6f}")
            weights = weights-rate*(2*design.T @ errors/len(y))
''')

example("logistic-fit", "Fit stable logistic loss and check another optimizer", "Why can a wrong score of magnitude 1,000 still have a finite, meaningful loss?", r'''
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    x = np.array([-2., -1., -1., 0., 0., 1., 1., 2.])
    y = np.array([0., 0., 1., 0., 1., 0., 1., 1.])
    design = np.column_stack([np.ones(len(x)), x])
    penalty = 0.1  # mean log loss + penalty*w^2/2; intercept unpenalized

    def objective(weights):
        scores = design @ weights
        signed_wrong_scores = (1-2*y)*scores
        loss = np.mean(np.logaddexp(0, signed_wrong_scores)) + penalty*weights[1]**2/2
        gradient = design.T @ (expit(scores)-y)/len(y) + np.array([0., penalty*weights[1]])
        return loss, gradient

    weights = np.zeros(2)
    for _ in range(2000):
        _, gradient = objective(weights)
        weights -= 0.1*gradient
    reference = minimize(objective, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-8})
    if not reference.success:
        raise RuntimeError(reference.message)
    print("GD coefficients:", np.round(weights, 6).tolist())
    print(f"penalized objective={objective(weights)[0]:.6f}")
    print("independent BFGS agreement:", np.allclose(weights, reference.x, atol=1e-7))
    print("probabilities at -2, 0, 2:", np.round(expit(np.array([[1., -2.], [1., 0.], [1., 2.]]) @ weights), 6).tolist())
    print("extreme wrong-label losses:", np.logaddexp(0, np.array([1000., 1000.])).tolist())
''')

example("separation", "Distinguish a decreasing loss from a finite optimum", "Both training labels are already classified correctly. What changes when the weight increases?", r'''
    import numpy as np
    from scipy.optimize import brentq
    from scipy.special import expit

    # Two observations (-1,0), (+1,1), intercept fixed at zero.
    for weight in [0., 2., 6., 12.]:
        print(f"w={weight:4.1f}; positive probability={expit(weight):.6f}; mean loss={np.logaddexp(0, -weight):.6f}")
    for penalty in [0.02, 0.2]:
        optimum = brentq(lambda weight: expit(weight)-1+penalty*weight, 0., 32.)
        objective = np.logaddexp(0, -optimum)+penalty*optimum**2/2
        print(f"lambda={penalty:.2f}; finite optimum={optimum:.6f}; objective={objective:.6f}")
    # Orthogonal one-coordinate squared-loss comparison, a=0.3 and lambda=0.5.
    a, penalty = 0.3, 0.5
    ridge = a/(1+penalty)
    lasso = np.sign(a)*max(abs(a)-penalty, 0)
    print(f"coordinate target={a}; ridge={ridge:.3f}; lasso={lasso:.3f}")
''')

example("held-out-pipeline", "Complete a train, validation and test experiment", "Which observations may choose C and the threshold, and which are only used for the final report?", r'''
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, confusion_matrix

    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, n_redundant=1, weights=[0.65, 0.35], flip_y=0.08, class_sep=0.8, random_state=31)
    X[:, 0] *= 1000  # change a measurement unit without changing its information
    X_develop, X_test, y_develop, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_develop, y_develop, test_size=0.25, stratify=y_develop, random_state=43)
    candidates = []
    for inverse_strength in [0.1, 1., 10.]:
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=inverse_strength, l1_ratio=0, solver="lbfgs", max_iter=2000, tol=1e-9))
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        candidates.append((log_loss(y_valid, probabilities), inverse_strength, pipeline, probabilities))
    validation_loss, chosen_C, model, validation_probabilities = min(candidates, key=lambda item: item[0])
    thresholds = np.r_[0., np.unique(validation_probabilities), 1.]

    def decision_cost(labels, probabilities, threshold):
        predicted = probabilities >= threshold
        return int(np.sum(predicted & (labels == 0)) + 4*np.sum(~predicted & (labels == 1)))

    threshold = min(thresholds, key=lambda value: (decision_cost(y_valid, validation_probabilities, value), value))
    # Keep the training-fitted model frozen: validation chose settings, not training rows.
    probabilities = model.predict_proba(X_test)[:, 1]
    baseline = np.full(len(y_test), y_train.mean())
    print("split sizes:", len(y_train), len(y_valid), len(y_test))
    print(f"chosen C={chosen_C:.1f}; validation log loss={validation_loss:.6f}; threshold={threshold:.6f}")
    print(f"test log loss={log_loss(y_test, probabilities):.6f}; prior baseline={log_loss(y_test, baseline):.6f}")
    print("test confusion [true rows, predicted columns]:", confusion_matrix(y_test, probabilities >= threshold, labels=[0, 1]).tolist())
    print("test cost at chosen threshold:", decision_cost(y_test, probabilities, threshold))
    print("test cost at 0.5:", decision_cost(y_test, probabilities, 0.5))
''')

example("polynomial-map", "Keep nonlinear feature construction inside a pipeline", "Can one affine threshold identify both tails of a one-dimensional input?", r'''
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LogisticRegression

    X = np.array([-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3.]).reshape(-1, 1)
    y = (np.abs(X[:, 0]) >= 2).astype(int)
    query = np.array([-2.5, 0., 2.5]).reshape(-1, 1)
    for degree in [1, 2]:
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LogisticRegression(C=10, l1_ratio=0, max_iter=2000, tol=1e-10))
        model.fit(X, y)
        print("degree", degree, "query probabilities", np.round(model.predict_proba(query)[:, 1], 6).tolist())
    print("These are synthetic mechanism checks, not held-out evidence for a chosen degree.")
''')

example("uncertainty", "Separate uncertainty in a mean from an individual outcome", "Why is a prediction interval wider than a confidence interval at the same feature vector?", r'''
    import numpy as np
    from scipy.stats import t

    x = np.array([0., 1., 2., 3., 4., 5.])
    y = np.array([1., 1.7, 3.2, 3.7, 5.3, 5.8])
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y-design @ weights
    degrees_freedom = len(y)-np.linalg.matrix_rank(design)
    variance = (residual @ residual)/degrees_freedom
    # QR-based covariance factor; do not square the condition number to fit the model.
    _, triangular = np.linalg.qr(design, mode="reduced")
    inverse_factor = np.linalg.solve(triangular, np.eye(2))
    covariance = variance*(inverse_factor @ inverse_factor.T)
    critical = t.ppf(0.975, degrees_freedom)
    for location in [2.5, 8.]:
        query = np.array([1., location])
        prediction = query @ weights
        mean_error = np.sqrt(query @ covariance @ query)
        individual_error = np.sqrt(variance+mean_error**2)
        print(f"x={location:.1f}; fit={prediction:.4f}; mean half-width={critical*mean_error:.4f}; individual half-width={critical*individual_error:.4f}")
    print("Model-based 95% intervals: correct linear mean, independent Gaussian equal-variance errors, fixed design.")
''')

example("sparse-stream", "Fit bounded sparse batches with an explicit feature contract", "Does partial_fit automatically make the data split, units and learning-rate schedule correct?", r'''
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(19)
    X = rng.binomial(1, 0.15, size=(240, 20)).astype(float)
    scores = -0.6 + 1.2*X[:, 0] - 1.8*X[:, 1] + 0.8*X[:, 2]
    probabilities = 1/(1+np.exp(-scores))
    y = (rng.random(len(X)) < probabilities).astype(int)
    train, test = csr_matrix(X[:180]), csr_matrix(X[180:])
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.01, learning_rate="constant", eta0=0.02, random_state=7)
    for epoch in range(30):
        order = rng.permutation(180)
        for start in range(0, 180, 30):
            indices = order[start:start+30]
            model.partial_fit(train[indices], y[indices], classes=np.array([0, 1]))
    prediction = model.predict_proba(test)[:, 1]
    print("training CSR shape:", train.shape, "stored nonzeros:", train.nnz)
    print(f"test log loss={log_loss(y[180:], prediction):.6f}")
    print("Separate SGDClassifier API: its penalty parameter is not LogisticRegression's deprecated parameter.")
''')

example("changed-report", "One acceptable changed-protocol report", "Change a declared setting before running your own report; which conclusions survive a different sample?", r'''
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, confusion_matrix

    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
        n_redundant=1, weights=[.65, .35], flip_y=.12, class_sep=.8, random_state=37)
    train, remaining = train_test_split(np.arange(300), test_size=.4, stratify=y, random_state=42)
    valid, test = train_test_split(remaining, test_size=.5, stratify=y[remaining], random_state=43)
    candidates=[]
    for degree in [1, 2]:
        for strength in [.1, 1.]:
            model=make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                LogisticRegression(C=strength, l1_ratio=0, solver="lbfgs", max_iter=2000, tol=1e-9))
            model.fit(X[train], y[train])
            loss=log_loss(y[valid],model.predict_proba(X[valid])[:,1])
            candidates.append((loss,degree,strength,model))
            print(f"validation degree={degree} C={strength}: log loss={loss:.6f}")
    _,degree,strength,chosen=min(candidates,key=lambda item:item[:3])
    validation_scores=chosen.predict_proba(X[valid])[:,1]
    thresholds=np.unique(np.r_[0., validation_scores, 1.])
    def cost(labels, predictions):
        return int(np.sum((predictions==1)&(labels==0))+4*np.sum((predictions==0)&(labels==1)))
    threshold=min(thresholds,key=lambda cutoff:(cost(y[valid],validation_scores>=cutoff),cutoff))
    test_scores=chosen.predict_proba(X[test])[:,1]
    baseline=np.full(len(test),y[train].mean())
    print("sizes:",len(train),len(valid),len(test))
    print(f"frozen degree={degree} C={strength} threshold={threshold:.6f}")
    print(f"test baseline log loss={log_loss(y[test],baseline):.6f}; model={log_loss(y[test],test_scores):.6f}")
    print("test confusion [true rows, predicted columns]:",confusion_matrix(y[test],test_scores>=threshold,labels=[0,1]).tolist())
    print("test cost:",cost(y[test],test_scores>=threshold))
    print("Limits: sixty synthetic test rows, one declared seed; no universal winner or calibrated-probability guarantee.")
''')

destination = ROOT / "src/learn/data/linear-logistic-examples.js"
destination.write_text("// Complete programs and observed stdout; regenerate with scripts/generate-linear-logistic-examples.py.\nexport const linearLogisticExamples = " + json.dumps(EXAMPLES, ensure_ascii=False, indent=2) + ";\n", encoding="utf8")
record_dir = ROOT / "scratch/linear-logistic-native"
record_dir.mkdir(parents=True, exist_ok=True)
(record_dir / "example-runs.json").write_text(json.dumps({"python": sys.version, "programs": [{"id": item["id"], "codeSha256": hashlib.sha256(item["code"].encode()).hexdigest(), "stdout": item["expected"]} for item in EXAMPLES]}, indent=2) + "\n", encoding="utf8")
print(json.dumps({"programs": len(EXAMPLES), "exported": str(destination)}))
