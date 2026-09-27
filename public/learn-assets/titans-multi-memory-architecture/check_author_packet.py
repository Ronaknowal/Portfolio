"""Author numerical checks; independent browser/reviewer checks remain deferred."""
import json
from pathlib import Path
import numpy as np
import torch
from memory_mechanisms import (associative_trace, chunk_trace, gated_sequence,
                               linear_write, nonlinear_outer, scalar_outer)
from neural_memory import read_memory
from rental_memory_study import load_observations, make_keys, replay, score

ROOT = Path(__file__).resolve().parent


def close(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def run_checks():
    torch.set_num_threads(1)
    results = json.loads((ROOT / 'rental-results.json').read_text())
    saved = json.loads((ROOT / 'mechanism-results.json').read_text())
    checks = []
    orthogonal = associative_trace([[1, 0], [0, 1]], [[2], [4]])
    correlated = associative_trace([[1, 0], [1, 1]], [[2], [4]])
    close(orthogonal[-1]['after'], [[1, 2]])
    close(correlated[-1]['after'], [[2.5, 1.5]])
    assert orthogonal == saved['orthogonal'] and correlated == saved['correlated']
    checks.append('Manual orthogonal/correlated write and stored trace agree')
    weight = np.array([[1.0, 2.0]])
    state = np.array([[0.5, -0.5]])
    new, momentum, gradient = linear_write(weight, state, np.zeros(2), np.array([10.0]), 0.5, 0.5, 0.0)
    close(gradient, [[0, 0]])
    close(new, [[1.25, 1.75]])
    reset_decay, _, _ = linear_write(weight, state, np.zeros(2), np.array([10.0]), 0.5, 0.5, 1.0)
    close(reset_decay, [[0.25, -0.25]])
    null, _, _ = linear_write(weight, np.zeros_like(weight), np.ones(2), np.array([10.0]), 0, 0.5, 0)
    close(null, weight)
    checks.append('Zero key, momentum-only change, full-decay-not-reset, no-update null')
    assert chunk_trace() == chunk_trace(True, 1)
    close(chunk_trace()[-1]['weight'], 1.25)
    close(chunk_trace(True)[-1]['weight'], 1.5)
    close(chunk_trace(rate=0)[-1]['weight'], chunk_trace(True, rate=0)[-1]['weight'])
    checks.append('Sequential versus anchor mismatch; size-one and zero-rate equality')
    assert scalar_outer() == {'weight': 0.5, 'prediction': 1.5, 'loss': 0.125, 'derivative': -3.0}
    _, derivative = nonlinear_outer(0.1, True)
    difference = (nonlinear_outer(0.100001) - nonlinear_outer(0.099999))/0.000002
    close(derivative, difference, 1e-9)
    checks.append('Outer derivative by analytic scalar and nonlinear finite differences')
    tokens = [[1,0], [0,1], [1,1], [1,-1]]
    full, state = gated_sequence(tokens)
    first, checkpoint = gated_sequence(tokens[:2])
    tail, _ = gated_sequence(tokens[2:], state=checkpoint)
    assert first + tail == full == saved['gated']
    altered, _ = gated_sequence(tokens[:3]+[[-2,1]])
    assert altered[:3] == full[:3] and altered[3] != full[3]
    fresh, _ = gated_sequence(tokens[2:])
    assert fresh != tail
    # Running request A cannot mutate the separate fresh request B.
    before, _ = gated_sequence([[0,1], [1,0]])
    gated_sequence([[3,2], [-1,2]], state=state)
    after, _ = gated_sequence([[0,1], [1,0]])
    assert before == after
    prefixless, _ = gated_sequence(tokens, prefix=False)
    close(prefixless[0]['output'], full[0]['output'])
    assert prefixless[1]['output'] != full[1]['output']
    disabled, _ = gated_sequence(tokens, rate=0)
    close([row['output'] for row in disabled], np.zeros((4,2)))
    checks.append('Gated topology prefix contrast/null, causality, continuation, isolation, reset, disabled writes')
    days, counts = load_observations()
    mean, scale = counts[:365].mean(), counts[:365].std()
    keys = make_keys(days, counts, mean, scale)
    targets = torch.tensor((counts-mean)/scale, dtype=torch.float64)
    assert len(keys[:358]) == 358 and len(counts[365:548]) == len(counts[548:]) == 183
    close(torch.linalg.vector_norm(keys, dim=1).numpy(), np.ones(724))
    close(results['mean'], mean)
    close(results['scale'], scale)
    for seed, record in results['seeds'].items():
        for window, start, end in [('development',0,183), ('assessment',183,366)]:
            for model in ['frozen','adaptive']:
                observed = counts[365+start:365+end]
                forecast = [row[model] for row in record['trace'][start:end]]
                recomputed = score(forecast, observed)
                for metric in ['mae','rmse']:
                    close(recomputed[metric], record['windows'][window][model][metric])
        assert all(np.isfinite(row[field]) for row in record['trace']
                   for field in ['frozen','adaptive','loss_before_write','gradient_norm','update_norm'])
        weights = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                        for value in record['initial_parameters'])
        replayed, _, _ = replay(weights, keys, targets)
        close([row['prediction_z'] for row in replayed], [row['prediction_z'] for row in record['trace']])
        null_trace, _, _ = replay(weights, keys, targets, stop=372, rate=0, decay=0)
        frozen = read_memory(weights, keys[358:365]).detach().numpy()
        close([row['prediction_z'] for row in null_trace], frozen)
        prefix, at_boundary, momentum = replay(weights, keys, targets, stop=548)
        suffix, _, _ = replay(at_boundary, keys, targets, start=548, momentum=momentum)
        assert prefix + suffix == replayed
        changed_counts = counts.copy()
        changed_counts[380] += 1000
        changed_keys = make_keys(days, changed_counts, mean, scale)
        changed_targets = torch.tensor((changed_counts-mean)/scale, dtype=torch.float64)
        changed_trace, _, _ = replay(weights, changed_keys, changed_targets, stop=383)
        close([row['prediction_z'] for row in changed_trace[:16]],
              [row['prediction_z'] for row in replayed[:16]])
        assert changed_trace[16]['prediction_z'] != replayed[16]['prediction_z']
    checks.append('All real seeds: source/splits, aggregates, replay, zero-update, checkpoint and future-target perturbation')
    # Current target with x=1, initial w=0, eta=1 is a deliberately leaky zero-loss score.
    updated, _, _ = linear_write(np.zeros((1,1)), np.zeros((1,1)), np.ones(1), np.array([4.0]), rate=1)
    close(updated @ np.ones(1), [4])
    checks.append('Score-before-write residual4 versus invalid score-after-write residual0')
    changed_weight, _, _ = linear_write(np.array([[2.0,-1.0]]), np.zeros((1,2)),
        np.array([1.0,2.0]), np.array([3.0]), rate=0.1)
    close(changed_weight, [[2.3,-0.4]])
    close(changed_weight @ np.array([2.0,1.0]), [4.2])
    sequential = 1.0
    anchor = 1.0
    for target in [3.0,-1.0]:
        sequential -= 0.25 * (sequential-target)
        anchor -= 0.25 * (1.0-target)
    close([sequential,anchor],[0.875,1.0])
    assert 2*512*128*2*4*12*8/2**20 == 96
    momentum = -0.5
    reversal = None
    for step in range(1,21):
        momentum = 0.9*momentum+0.05
        if momentum > 0 and reversal is None:
            reversal = step
    assert reversal == 7
    # Loss alone does not determine gradient norm, even for one fixed Jacobian.
    jacobian = np.diag([1.0,0.0])
    close(np.linalg.norm(jacobian.T @ np.array([1.0,0.0])), 1)
    close(np.linalg.norm(jacobian.T @ np.array([0.0,2.0])), 0)
    checks.append('Changed practice arithmetic, exact momentum reversal and fixed-Jacobian counterexample')
    output = {'phase': 'author content checks', 'checks': checks,
              'browser_and_independent_review': 'deferred to implementation',
              'environment': results['environment']}
    (ROOT / 'author-checks.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    run_checks()
