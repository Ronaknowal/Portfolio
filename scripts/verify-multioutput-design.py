"""Evaluate proposed teaching counterexamples before source implementation."""
from fractions import Fraction as F
from itertools import product
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
states = list(product((0, 1), repeat=2))
mass = [F(6, 20), F(5, 20), F(1, 20), F(8, 20)]
marginal = [sum(p * state[j] for p, state in zip(mass, states)) for j in range(2)]
risks = {''.join(map(str, prediction)): {
    'hamming': str(sum(p * sum(a != b for a, b in zip(prediction, state)) / 2 for p, state in zip(mass, states))),
    'subset': str(1 - mass[states.index(prediction)])
} for prediction in states}
assert marginal == [F(9, 20), F(13, 20)]
assert risks['01']['hamming'] == '2/5' and risks['11']['subset'] == '3/5'
assert F(5, 11) < F(1, 2) and F(8, 13) > F(1, 2)

truth = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
prediction = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]])
tp = int(np.sum((truth == 1) & (prediction == 1)))
fp = int(np.sum((truth == 0) & (prediction == 1)))
fn = int(np.sum((truth == 1) & (prediction == 0)))
assert (tp, fp, fn) == (5, 1, 2)
assert F(2 * tp, 2 * tp + fp + fn) == F(10, 13)

x = np.arange(4, dtype=float).reshape(-1, 1)
y = np.array([[0, 0], [0, 100], [2, 100], [2, 100]], dtype=float)
stumps = []
for scale in (1, 100):
    scales = np.array([1, scale])
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x, y / scales)
    errors = np.sum((y / scales - model.predict(x)) ** 2, axis=0)
    stumps.append({'energyScale': scale, 'threshold': float(model.tree_.threshold[0]), 'sseByOutput': errors.tolist()})
assert [row['threshold'] for row in stumps] == [0.5, 1.5]
for column, expected in ((0, 1.5), (1, 0.5)):
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x, y[:, column])
    assert model.tree_.threshold[0] == expected
    assert np.allclose(model.predict(x), y[:, column])

result = {
    'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'design fixtures evaluated; production is not implemented',
    'joint': {'states': states, 'mass': [str(value) for value in mass], 'marginal': list(map(str, marginal)), 'risks': risks, 'marginalDecision': '01', 'greedyAB': '00', 'greedyBA': '11', 'jointMode': '11'},
    'oldExerciseCorrection': {'tp': tp, 'fp': fp, 'fn': fn, 'hamming': '1/4', 'subsetAccuracy': '1/4', 'microF1': '10/13'},
    'averagedMetricReversal': {'A': {'jaccard': '1/2', 'sampleF1': '1/2'}, 'B': {'jaccard': '2/5', 'sampleF1': '4/7'}, 'construction': 'Two five-positive rows. A predicts every label on one row and none on the other; B predicts two correct labels on both rows.'},
    'conditionalIndependence': {'stratumProbabilities': ['1/10', '9/10'], 'mixtureWeight': '1/2', 'pooledBoth': str((F(1, 10)**2 + F(9, 10)**2) / 2), 'pooledConditional': '41/50', 'withinStratumIndependent': True},
    'weightedBce': {'posterior': '1/10', 'positiveWeight': 9, 'negativeWeight': 1, 'scoreOptimum': '1/2'},
    'sharedStumps': stumps,
    'verificationScope': 'Exact finite rational calculations and independent scikit-learn split fits; no empirical superiority or deployment inference.'
}
destination = ROOT / 'docs/teaching/evidence/multioutput-design-fixtures.json'
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
