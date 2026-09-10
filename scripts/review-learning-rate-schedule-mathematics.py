from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import math
import torch

fixtures = json.loads(Path('scratch/learning-rate-schedule-independent/fixtures.json').read_text())
counts = dict(torchVersion=torch.__version__, oneCyclePolicies=0, oneCycleUsedStates=0,
              exactNoisePolicies=0, exactNoiseStates=0, plateauPolicies=0, plateauStates=0)

for case in fixtures['cycles']:
    parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=.02, momentum=.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=.2, total_steps=case['total'],
        pct_start=case['rise'] / case['total'], div_factor=10,
        final_div_factor=100, base_momentum=.85, max_momentum=.95)
    for state in case['states']:
        assert math.isclose(optimizer.param_groups[0]['lr'], state['rate'], abs_tol=1e-14)
        assert math.isclose(optimizer.param_groups[0]['momentum'], state['momentum'], abs_tol=1e-14)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        counts['oneCycleUsedStates'] += 1
    counts['oneCyclePolicies'] += 1

for case in fixtures['noiseCases']:
    rates = [Fraction(str(value)) for value in case['rates']]
    curvature, sigma = Fraction(str(case['curvature'])), Fraction(str(case['noise']))
    for completed, state in enumerate(case['states']):
        used = rates[:completed]
        multipliers = [1 - rate * curvature for rate in used]
        # Closed products, not the model's forward moment recurrence.
        exact_mean = 2 * math.prod(multipliers)
        exact_variance = sum(rate ** 2 * sigma ** 2 * math.prod(multipliers[index + 1:]) ** 2
                             for index, rate in enumerate(used))
        outcomes = []
        for signs in product((-1, 1), repeat=completed):
            error = Fraction(2)
            for rate, sign in zip(used, signs):
                error -= rate * (curvature * error + sigma * sign)
            outcomes.append(error)
        square = sum(value * value for value in outcomes) / len(outcomes)
        assert square == exact_mean ** 2 + exact_variance
        for name, expected in [('mean', exact_mean), ('variance', exact_variance), ('meanSquaredError', square)]:
            assert math.isclose(state[name], float(expected), rel_tol=3e-14, abs_tol=1e-14)
        counts['exactNoiseStates'] += 1
    counts['exactNoisePolicies'] += 1

for case in fixtures['plateaus']:
    options = case['options']
    parameter = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.SGD([parameter], lr=options['initialRate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=options['patience'],
        threshold=options['threshold'], threshold_mode='abs',
        cooldown=options['cooldown'], min_lr=options['minimumRate'], eps=1e-8)
    for metric, state in zip(case['metrics'], case['states']):
        scheduler.step(metric)
        assert scheduler.best == state['best']
        assert scheduler.num_bad_epochs == state['bad']
        assert scheduler.cooldown_counter == state['cooling']
        assert optimizer.param_groups[0]['lr'] == state['rate']
        counts['plateauStates'] += 1
    counts['plateauPolicies'] += 1

# Changed-input hand exercises, checked without the authored schedule/model helper.
rates = [.06, .12] + [.02 + .1 * (1 + math.cos(math.pi * step / 4)) / 2 for step in range(1, 5)]
assert [round(value, 6) for value in rates] == [.06, .12, .105355, .07, .034645, .02]
assert 3 * Fraction(98, 100) ** 2 == Fraction(28812, 10000)
assert 3 * Fraction(96, 100) == Fraction(288, 100)
assert Fraction(1, 5) / (2 * (2 - Fraction(2, 5))) == Fraction(1, 16)
counts['exactHandChecks'] = 4
print(json.dumps(counts))
