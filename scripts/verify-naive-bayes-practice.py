"""Small independent solution/API checks complementing the exhaustive model suite."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import sklearn
import sklearn.calibration as calibration
from sklearn.naive_bayes import GaussianNB, ComplementNB

odds = F(1, 2) * F(28, 11) * F(7, 22)
assert odds / (1 + odds) == F(49, 170)
present_weights = [F(1, 2) * F(4, 10) ** 2, F(1, 2) * F(8, 10) ** 2]
assert present_weights[1] / sum(present_weights) == F(4, 5)
assert F(8, 10) / (F(8, 10) + F(4, 10)) == F(2, 3)
x = 1.5 * math.sqrt(math.log(3))
assert abs(-math.log(3) + F(4, 9) * x * x) < 1e-14
actual_weights = [F(9, 10) * F(3, 10), F(1, 10) * F(9, 10)]
fake_weights = [F(9, 10) * F(3, 10) ** 3, F(1, 10) * F(9, 10) ** 3]
assert actual_weights[1] / sum(actual_weights) == F(1, 4)
assert fake_weights[1] / sum(fake_weights) == F(3, 4)
assert F(1, 10) * F(9, 10) + F(9, 10) * F(7, 10) == F(18, 25)
scores = [-2*math.log(a/23)-math.log(b/23) for a, b in [(10,11),(5,9),(10,3)]]
assert scores.index(max(scores)) == 1
rare_odds = F(1, 49) * 9
assert rare_odds / (1 + rare_odds) == F(9, 58)
assert F(9, 58) > F(1, 21)
assert F(2*3,4*5) == F(3, 10)

source_path = Path(inspect.getsourcefile(calibration))
source = source_path.read_text(encoding="utf-8")
fit_source = inspect.getsource(calibration._fit_calibrator)
predict_source = inspect.getsource(calibration._CalibratedClassifier.predict_proba)
assert 'response_method=["decision_function", "predict_proba"]' in predict_source
assert 'calibrator.fit(this_pred' in fit_source
assert not hasattr(GaussianNB(), "decision_function")
assert ComplementNB().norm is False
record = {
    "completedAt": datetime.now(timezone.utc).isoformat(),
    "practice": {
        "opposingWords": "49/170", "observedAbsence": "4/5", "missingSecond": "2/3",
        "gaussianCrossing": x, "copiedTrue": "1/4", "copiedReported": "3/4",
        "copiedActualAccuracy": "18/25", "changedComplementScores": scores,
        "rareDeployment": "9/58", "rareCostThreshold": "1/21", "integratedTwoA": "3/10"
    },
    "installedApi": {
        "sklearn": sklearn.__version__, "calibrationSourceSha256": hashlib.sha256(source.encode()).hexdigest(),
        "sourceFile": str(source_path), "gaussianHasDecisionFunction": False,
        "readSections": "CalibratedClassifierCV.fit; _fit_calibrator sigmoid/isotonic branch; _CalibratedClassifier.predict_proba; response selection and separate temperature branch",
        "conclusion": "The executed sigmoid receives GaussianNB predict_proba outputs. Temperature-specific logit conversion does not apply to sigmoid. FrozenEstimator is preserved through calibration fitting.",
        "fitBranch": fit_source, "predictionBranch": predict_source
    }
}
Path("scratch/naive-bayes-verification/practice-api-results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Changed practice fractions, Gaussian crossings, complement score and installed sigmoid API checks passed.")
