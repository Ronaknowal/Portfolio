"""Independent actual PyTorch schedulers, exhaustive noise paths and event oracles."""
from itertools import product
import json
import math
import sys
import numpy as np
import torch

with open(sys.argv[1], encoding='utf-8') as handle:
    fixtures = json.load(handle)
torch.set_num_threads(1)

def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-12), (actual, expected)

runtime_schedules = 0
for case in fixtures['schedules']:
    spec, states = case['specification'], case['states']
    total, kind, peak = spec['total'], spec['kind'], spec['peak']
    minimum = spec['minimum']
    weight = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=peak, momentum=.9)
    if kind == 'one-cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=peak, total_steps=total,
            pct_start=spec['rise']/total, div_factor=10, final_div_factor=100,
            base_momentum=.85, max_momentum=.95)
    elif kind == 'restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            T_0=spec['period'], T_mult=1, eta_min=minimum)
    elif kind == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=spec['period'], gamma=.5)
    elif kind == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(minimum/peak)**(1/(total-1)))
    else:
        scheduler = None
    if scheduler is not None:
        runtime_schedules += 1
        for state in states:
            close(state['rate'], optimizer.param_groups[0]['lr'])
            if kind == 'one-cycle':
                close(state['momentum'], optimizer.param_groups[0]['momentum'])
            weight.grad = torch.ones_like(weight)
            optimizer.step()
            scheduler.step()
    elif kind == 'constant':
        assert all(state['rate'] == peak for state in states)
    elif kind == 'linear':
        np.testing.assert_allclose([state['rate'] for state in states], np.linspace(peak, minimum, total), atol=2e-12)
    elif kind == 'cosine':
        warmup = spec['warmup']
        if warmup:
            expected = np.concatenate((np.linspace(peak/warmup, peak, warmup),
                minimum+(peak-minimum)*(1+np.cos(np.linspace(0, np.pi, total-warmup+1)[1:]))/2))
        else:
            expected = minimum+(peak-minimum)*(1+np.cos(np.linspace(0, np.pi, total)))/2
        np.testing.assert_allclose([state['rate'] for state in states], expected, atol=2e-12)
        close(states[-1]['rate'], minimum)
        if warmup:
            close(states[warmup-1]['rate'], peak)

noise_paths = 0
for case in fixtures['noise']:
    rates, spec = case['rates'], case['specification']
    # Enumerate the whole distribution at each update, not the model's recurrence.
    outcomes = np.array([spec['initialError']], dtype=float)
    for index, state in enumerate(case['states']):
        close(state['mean'], float(outcomes.mean()))
        close(state['variance'], float(outcomes.var()))
        close(state['meanSquaredError'], float(np.mean(outcomes*outcomes)))
        close(state['expectedLoss'], spec['curvature']*float(np.mean(outcomes*outcomes))/2)
        noise_paths += len(outcomes)
        if index < len(rates):
            rate = rates[index]['rate']
            outcomes = np.array([error-rate*(spec['curvature']*error+sign*spec['noise'])
                                 for error, sign in product(outcomes, (-1, 1))])

clock_targets = [1, 3, 2, 4, 0, 2, 3, 1, 4, 2, 1, 3]
clock_updates = 0
for case in fixtures['clocks']:
    spec, trace = case['specification'], case['trace']
    group_size = spec['accumulation']
    batches = [clock_targets[start:start+group_size] for start in range(0, 12, group_size)]
    weight = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=.2)
    successes = 0
    for attempt, batch in enumerate(batches, start=1):
        state = trace['states'][attempt*group_size]
        optimizer.zero_grad()
        loss = torch.mean(.5*(weight-torch.tensor(batch, dtype=torch.float64))**2)
        loss.backward()
        close(state['appliedGradient'], weight.grad.item())
        if spec['skipSecond'] and attempt == 2:
            assert state['rate'] is None and state['scheduleIndex'] is None
        else:
            index = attempt*group_size-1 if spec['policy'] == 'microbatch' else successes+int(spec['policy'] == 'advance-first')
            assert index == state['scheduleIndex']
            if index < trace['updates']:
                optimizer.param_groups[0]['lr'] = trace['rates'][index]
                optimizer.step()
                successes += 1
                clock_updates += 1
            else:
                assert state['action'] == 'budget exhausted: no update'
        close(state['parameter'], weight.item())
        assert state['committed'] == successes
    if spec['policy'] == 'committed':
        assert successes == trace['updates']

plateau_states = 0
for case in fixtures['plateau']:
    weight = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.SGD([weight], lr=.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
        threshold_mode='abs', min_lr=.025, **case['specification'])
    for metric, state in zip(case['metrics'], case['states']):
        scheduler.step(metric)
        close(state['rate'], optimizer.param_groups[0]['lr'])
        close(state['best'], scheduler.best)
        assert state['bad'] == scheduler.num_bad_epochs
        assert state['cooling'] == scheduler.cooldown_counter
        plateau_states += 1

print(json.dumps({'torch': torch.__version__, 'numpy': np.__version__,
    'actualSchedulerConfigurations': runtime_schedules, 'exhaustiveNoiseOutcomes': noise_paths,
    'actualClockUpdates': clock_updates, 'actualPlateauStates': plateau_states}))
