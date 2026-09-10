"""Changed-input exercises, independent arithmetic and actual CPU library state."""
import copy
import io
import itertools
import json
import math
import runpy
from contextlib import redirect_stdout
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch

directory = Path('scratch/learning-rate-schedule-review')
rates = [.06, .12] + [.02 + .1 * (1 + math.cos(math.pi * j / 4)) / 2 for j in range(1, 5)]
assert np.allclose(rates, [.06, .12, .105355, .07, .034645, .02], atol=5e-7, rtol=0)
outcomes = []
for signs in itertools.product((-1, 1), repeat=2):
    error = F(2)
    for rate, sign in zip((F(1, 10), F(3, 10)), signs):
        error = (1 - 4 * rate) * error - rate * sign * F(1, 2)
    outcomes.append(error)
mean = sum(outcomes) / 4
variance = sum((value - mean)**2 for value in outcomes) / 4
second_moment = sum(value**2 for value in outcomes) / 4
assert (mean, variance, second_moment) == (F(-24, 100), F(226, 10000), F(802, 10000))
consumed = [(microbatch, [.2, .05][index]) for index, microbatch in enumerate([boundary for boundary in range(3, 10, 3) if boundary != 6])]
assert consumed == [(3, .2), (9, .05)]
parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
optimizer = torch.optim.SGD([parameter], lr=.4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=.1, threshold_mode='abs', patience=1, cooldown=1, factor=.5, min_lr=.05)
plateau = []
for metric in [1, .9, .89, .9, .9, .9, .7]:
    scheduler.step(metric)
    plateau.append((scheduler.best, scheduler.num_bad_epochs, scheduler.cooldown_counter, optimizer.param_groups[0]['lr']))
assert plateau == [(1, 0, 0, .4), (1, 1, 0, .4), (.89, 0, 0, .4), (.89, 1, 0, .4), (.89, 0, 1, .2), (.89, 0, 0, .2), (.7, 0, 0, .2)]
assert 3 * (1 - F(1, 10) * F(1, 5))**2 == F(28812, 10000)
assert 3 * (1 - F(1, 5) * F(1, 5)) == F(288, 100)

with redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(directory / 'programs/checkpointResume.py'))
create, advance = fixture['create'], fixture['advance']
original = create()
advance(original, 3)
saved = {'weight': original[0].detach().clone(), 'optimizer': copy.deepcopy(original[1].state_dict()),
         'scheduler': copy.deepcopy(original[2].state_dict()), 'generator': original[3].get_state().clone()}
expected = advance(original, 9)
restored = create()
with torch.no_grad(): restored[0].copy_(saved['weight'])
restored[2].load_state_dict(saved['scheduler'])
restored[1].load_state_dict(copy.deepcopy(saved['optimizer']))
restored[3].set_state(saved['generator'])
actual = advance(restored, 9)
assert len(actual) == 9
assert all(left[:2] == right[:2] and torch.equal(left[2], right[2]) for left, right in zip(expected, actual))
changed = create()
with torch.no_grad(): changed[0].copy_(saved['weight'])
# Define this altered policy precisely: restart a new 20-interval clock at the
# already prepared rate. Keep optimizer moments and sampling state identical.
changed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(changed[1], T_max=20, eta_min=.002)
changed[1].load_state_dict(copy.deepcopy(saved['optimizer']))
changed[3].set_state(saved['generator'])
changed_rows = advance((changed[0], changed[1], changed_scheduler, changed[3]), 9)
assert expected[0][:2] == changed_rows[0][:2] and torch.equal(expected[0][2], changed_rows[0][2])
assert expected[1][0] == changed_rows[1][0] and expected[1][1] != changed_rows[1][1]
assert not torch.equal(expected[1][2], changed_rows[1][2])
loaded_horizon = torch.optim.lr_scheduler.CosineAnnealingLR(changed[1], T_max=20, eta_min=.002)
loaded_horizon.load_state_dict(saved['scheduler'])
assert loaded_horizon.T_max == 11

with redirect_stdout(io.StringIO()):
    experiment = runpy.run_path(str(directory / 'programs/controlledExperiment.py'))
X, V = experiment['X'], experiment['V']
training_targets, validation_targets = experiment['train_y'], experiment['validation_y']
def train(policy, peak, seed):
    weight = np.zeros(2)
    for update, sample in enumerate(np.random.default_rng(seed).integers(len(X), size=120)):
        if policy == 'constant': rate = peak
        elif policy == 'linear': rate = peak + (.01 - peak) * update / 119
        elif update < 12: rate = peak * (update + 1) / 12
        else: rate = .01 + (peak - .01) * (1 + math.cos(math.pi * (update - 11) / 108)) / 2
        weight -= rate * (X[sample] @ weight - training_targets[sample]) * X[sample]
    return weight
selection = []
for policy in ['constant', 'linear', 'warmup-cosine']:
    candidates = []
    for peak in [.1, .2]:
        losses = [float(np.mean((V @ train(policy, peak, seed) - validation_targets)**2)) for seed in [2, 7, 19]]
        candidates.append({'peak': peak, 'validation': losses, 'mean': float(np.mean(losses))})
    best = min(candidates, key=lambda row: row['mean'])
    selection.append({'policy': policy, 'candidates': candidates, 'selectedPeak': best['peak']})
# Create untouched test observations only after the candidate selection.
test_rng = np.random.default_rng(101)
test_x = np.linspace(-1, 1, 200)
test_targets = .7 + 1.5 * test_x + test_rng.normal(0, .3, len(test_x))
test_design = np.column_stack((np.ones(len(test_x)), test_x))
for row in selection:
    row['test'] = [float(np.mean((test_design @ train(row['policy'], row['selectedPeak'], seed) - test_targets)**2)) for seed in [23, 29, 37]]
assert all(math.isfinite(value) for row in selection for value in row['test'])
winners = [min(selection, key=lambda row: row['test'][index])['policy'] for index in range(3)]
result = {'torch': torch.__version__, 'exercises': 7, 'finiteRates': rates, 'noiseExact': [str(mean), str(variance), str(second_moment)],
          'plateau': plateau, 'identicalResumeUpdates': len(actual),
          'changedHorizonFirstMismatch': {'remainingIndex': 1, 'originalRate': expected[1][1], 'changedRate': changed_rows[1][1]},
          'comparison': selection, 'testWinners': winners, 'selectionUpdates': 18 * 120, 'finalAssessmentUpdates': 9 * 120}
(directory / 'practice-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
