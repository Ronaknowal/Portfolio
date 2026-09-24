"""Independent arithmetic for the proposed teaching fixtures, before production models."""
from datetime import datetime, timezone
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
records = {}
for ratio in [F(1, 2), F(1), F(2), F(5)]:
    completion = [[F(1), ratio], [1 / ratio, F(1)]]
    assert completion[0][0] * completion[1][1] == completion[0][1] * completion[1][0]
records['rankOneCompatibleCompletions'] = [[[1, .5], [2, 1]], [[1, 2], [.5, 1]]]

user = np.array([.4, -.2])
item = np.array([.5, .3])
mean, user_bias, item_bias, rating, rate, penalty = 3., .1, -.1, 5., .05, .1
parameters = np.r_[user, item, user_bias, item_bias]
def local_loss(vector):
    score = mean + vector[4] + vector[5] + vector[:2] @ vector[2:4]
    return .5 * ((rating - score)**2 + penalty * (vector @ vector))
error = rating - mean - user_bias - item_bias - user @ item
gradient = np.r_[-error * item + penalty * user, -error * user + penalty * item, -error + penalty * user_bias, -error + penalty * item_bias]
finite_difference = []
for index in range(6):
    step = np.zeros(6); step[index] = 1e-6
    finite_difference.append((local_loss(parameters + step) - local_loss(parameters - step)) / 2e-6)
assert np.allclose(gradient, finite_difference, rtol=1e-8, atol=1e-8)
after = parameters - rate * gradient
records['onePairUpdate'] = {'before': parameters.tolist(), 'error': error, 'gradient': gradient.tolist(), 'after': after.tolist(), 'lossBefore': local_loss(parameters), 'lossAfter': local_loss(after)}

items = np.array([[1., 0.], [0., 1.], [1., 1.]])
counts = np.array([2., 0., 1.]); confidence = 1 + 2 * counts
normal = items.T @ (confidence[:, None] * items) + np.eye(2)
right = items.T @ (confidence * (counts > 0))
solution = np.linalg.solve(normal, right)
augmented = np.vstack([np.sqrt(confidence[:, None]) * items, np.eye(2)])
target = np.r_[np.sqrt(confidence) * (counts > 0), [0., 0.]]
independent = np.linalg.lstsq(augmented, target, rcond=None)[0]
assert np.allclose(solution, [31 / 36, 1 / 12])
assert np.allclose(solution, independent)
records['implicitBlock'] = {'normal': normal.tolist(), 'right': right.tolist(), 'solution': solution.tolist(), 'scores': (items @ solution).tolist()}

ranking = [2, 0, 4, 1, 3]
relevant = {0, 1}
hits = [int(item in relevant) for item in ranking[:3]]
dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
ideal = 1 + 1 / math.log2(3)
records['rankingAtThree'] = {'hits': hits, 'precision': sum(hits) / 3, 'recall': sum(hits) / 2, 'ndcg': dcg / ideal, 'reciprocalRank': .5, 'averagePrecisionAtThree': .25}
records['loggingPolicy'] = {'logging': [.8, .2], 'target': [.5, .5], 'rewardProbabilities': [.4, .8]}
policy = records['loggingPolicy']; expectation = second = 0.
for action, reward in itertools.product(range(2), [0, 1]):
    mass = policy['logging'][action] * (policy['rewardProbabilities'][action] if reward else 1 - policy['rewardProbabilities'][action])
    value = policy['target'][action] / policy['logging'][action] * reward
    expectation += mass * value; second += mass * value * value
assert math.isclose(expectation, .6) and math.isclose(second - expectation**2, .765)
policy.update({'expectedLoggedReward': .48, 'expectedTargetReward': expectation, 'importanceVariance': second - expectation**2})
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'attribution': 'Design-stage independent arithmetic; no production/native/browser completion claim.', 'records': records}
destination = ROOT / 'docs/teaching/evidence/recommender-systems-design-calculations.json'
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
