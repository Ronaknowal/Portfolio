"""Independent entropy, constrained optimization, joint and sampling checks.

Consumes real JS model outputs; does not import the JS implementation or the
lesson's Python helper. Exact displayed programs are also executed unchanged.
"""
import json
import math
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import rel_entr
from scipy.stats import entropy

directory = Path('scratch/mutual-information-native-verification')
fixtures = json.loads((directory / 'model-fixtures.json').read_text())
counts = {}
maximum_error = 0.0


def close(actual, expected, tolerance=2e-10):
    global maximum_error
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    maximum_error = max(maximum_error, error)
    assert error <= tolerance, (actual, expected, error, tolerance)


def mi(joint):
    table = np.asarray(joint, dtype=float)
    return entropy(table.sum(axis=1), base=2) + entropy(table.sum(axis=0), base=2) - entropy(table.ravel(), base=2)


for name, example in fixtures['examples'].items():
    path = directory / f'verified-{name}.py'
    path.write_text(example['code'], encoding='utf8')
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == example['expected'].strip(), (name, result.stdout, example['expected'])
counts['exact_programs'] = len(fixtures['examples'])

# Execute the final independent project's changed task, rather than copying
# default program output into its explained solution.
changed_program = fixtures['examples']['iteration']['code'].replace(
    'LABEL = [[.9, .1], [.9, .1], [.1, .9], [.1, .9]]',
    'LABEL = [[.8, .2], [.8, .2], [.2, .8], [.2, .8]]',
).replace('state(encoder, 3)', 'state(encoder, 4)').replace('update(encoder, 3)', 'update(encoder, 4)')
project_path = directory / 'verified-changed-task-project.py'
project_path.write_text(changed_program, encoding='utf8')
project_result = subprocess.run([sys.executable, str(project_path)], capture_output=True, text=True, check=True)
assert 'informative 40 0.698380 0.218212 -0.174466' in project_result.stdout
assert 'symmetric 40 0.000000 0.000000 0.000000' in project_result.stdout
assert 'P(Z=1|X): [0.05363, 0.05363, 0.94637, 0.94637]' in project_result.stdout
(directory / 'changed-task-project-output.txt').write_text(project_result.stdout)
counts['changed_task_complete_program'] = 1

for state in fixtures['finite']:
    p = np.asarray(state['joint'])
    close(state['mi'], mi(p))
    close(state['mi'], np.sum(rel_entr(p, np.outer(p.sum(1), p.sum(0)))) / math.log(2))
    close(state['mi'], mi(p.T))
    close(state['mi'], mi(p[::-1, ::-1]))
counts['finite_tables_entropy_and_KL'] = len(fixtures['finite'])

for state in fixtures['xor']:
    b = state['bias']
    expected = {'a': 1 - entropy([b, 1-b], base=2), 'b': 0, 'both': 1}[state['reveal']]
    close(state['mi'], expected)
    close(state['conditionalMi'], 1)
counts['conditional_XOR_laws'] = len(fixtures['xor'])

for state in fixtures['representations']:
    e, q, mode = state['labelError'], state['noise'], state['mode']
    effective_noise = e + q - 2*e*q
    rate = {'constant': 0, 'signal': 1, 'nuisance': 1, 'both': 2, 'noisy': 1-entropy([q, 1-q], base=2)}[mode]
    relevance = 0 if mode in ['constant', 'nuisance'] else 1 - entropy([effective_noise, 1-effective_noise] if mode == 'noisy' else [e, 1-e], base=2)
    close(state['rate'], rate)
    close(state['relevance'], relevance)
    close(state['objective'], rate - state['beta']*relevance)
counts['closed_form_representations'] = len(fixtures['representations'])

optimized_rows = 0
free_energies = 0
for trace in fixtures['iterations']:
    beta, error = trace['beta'], trace['labelError']
    py_x = np.array([[1-error, error]]*2 + [[error, 1-error]]*2)
    for index, state in enumerate(trace['rows']):
        q = np.array(state['encoder'])
        joint_xz = q/4
        joint_zy = np.einsum('xz,xy->zy', q/4, py_x)
        marginal = joint_zy.sum(1)
        decoder = joint_zy / marginal[:, None]
        close(state['rate'], mi(joint_xz))
        close(state['relevance'], mi(joint_zy))
        close(state['decoder'], decoder)
        # Verify the whole stated F identity with intentionally mismatched
        # auxiliary distributions, directly integrating its original two terms.
        r = np.array([.31, .69])
        w = np.array([[.7, .3], [.2, .8]])
        distortion = np.sum(rel_entr(py_x[:, None, :], w[None, :, :]), axis=2)
        free = np.sum(rel_entr(q, r)/4) + beta*np.sum(q*distortion)/4
        identity = (state['objective'] + beta*(1-entropy([error, 1-error], base=2)))*math.log(2)
        identity += np.sum(rel_entr(marginal, r)) + beta*np.sum(marginal[:, None]*rel_entr(decoder, w))
        close(free, identity)
        free_energies += 1
        if index and beta in [.5, 3] and trace['initialization'] == 'signal' and index in [1, 4, 40]:
            previous = trace['rows'][index-1]
            old_decoder = np.array(previous['decoder'])
            old_marginal = np.array(previous['pZ'])
            for x in range(4):
                cost = np.sum(rel_entr(py_x[x], old_decoder), axis=1)
                def objective(probability):
                    row = np.array([probability, 1-probability])
                    return np.sum(rel_entr(row, old_marginal)) + beta*np.dot(row, cost)
                optimum = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol': 1e-13})
                assert optimum.success
                close(objective(q[x, 0]), optimum.fun, 1e-9)
                close(q[x, 0], optimum.x, 3e-8)
                optimized_rows += 1
counts['IB_full_state_free_energy_identities'] = free_energies
counts['independent_constrained_row_minima'] = optimized_rows

for state in fixtures['bounds']:
    q, pz, decoder = map(np.array, [state['encoder'], state['pZ'], state['decoder']])
    r, s = map(np.array, [state['reference'], state['approximateDecoder']])
    upper = np.sum(rel_entr(q, r))/4/math.log(2)
    cross_entropy = -np.sum(np.array(state['pZY'])*np.log2(s))
    close(state['rateUpper'], upper)
    close(state['predictiveLower'], 1-cross_entropy)
    close(state['rateGap'], upper-state['rate'])
    close(state['predictiveGap'], state['relevance']-(1-cross_entropy))
    close(state['objectiveUpper']-state['objective'], state['rateGap'] + state['beta']*state['predictiveGap'])
counts['variational_gap_identities'] = len(fixtures['bounds'])


def random32(seed):
    state = seed & 0xffffffff or 1
    while True:
        state = (state ^ (state << 13)) & 0xffffffff
        state = (state ^ (state >> 17)) & 0xffffffff
        state = (state ^ (state << 5)) & 0xffffffff
        yield state / 2**32


shuffle_checks = 0
for state in fixtures['samples']:
    size = state['categories']
    pairs = np.array(state['pairs'])
    rng = random32(state['seed'])
    rebuilt = []
    for _ in range(state['count']):
        x, flip, alternative = int(next(rng)*size), next(rng), next(rng)
        y = int(flip*size) if state['mode'] == 'independent' else ((x+1+int(alternative*(size-1))) % size if flip < .2 else x)
        rebuilt.append([x, y])
    assert np.array_equal(pairs, rebuilt)
    table = np.histogram2d(pairs[:, 0], pairs[:, 1], bins=[np.arange(size+1)-.5]*2)[0]
    assert np.array_equal(table, state['counts'])
    close(state['mi'], mi(table))
    rng = random32(state['seed'] ^ 0x5a39b174)
    for actual in state['shuffled']:
        labels = list(pairs[:, 1])
        for i in range(len(labels)-1, 0, -1):
            j = int(next(rng)*(i+1))
            labels[i], labels[j] = labels[j], labels[i]
        shuffled = np.histogram2d(pairs[:, 0], labels, bins=[np.arange(size+1)-.5]*2)[0]
        assert np.array_equal(shuffled.sum(0), table.sum(0))
        assert np.array_equal(shuffled.sum(1), table.sum(1))
        close(actual, mi(shuffled))
        shuffle_checks += 1
    population = np.ones((size, size)) / size**2
    if state['mode'] == 'channel':
        population[:] = .2/(size*(size-1))
        np.fill_diagonal(population, .8/size)
    close(state['populationMi'], mi(population))
counts['sample_count_tables'] = len(fixtures['samples'])
counts['permutations_with_exact_marginal_conservation'] = shuffle_checks

for state in fixtures['gaussian']:
    with localcontext() as context:
        context.prec = 80
        sigma = Decimal(str(state['sigma']))
        expected = float((1+1/(sigma*sigma)).ln()/(2*Decimal(2).ln()))
        close(state['information'], expected, 5e-12)
counts['high_precision_Gaussian_scales'] = len(fixtures['gaussian'])

# Changed-input exercises and independently enumerated processing contracts.
close(mi([[.45, .15], [.1, .3]]), .18149632952867556)
close(math.log2(.15/(.6*.45)), -.8479969065549501)
close(np.array([.25, .75/4]) / (.25+.75/4), [4/7, 3/7])
for prevalence in [.1, .5, .8]:
    for label_error in [.05, .2, .5]:
        for encoding_error in [.1, .3, .5]:
            joint = np.zeros((2, 2, 2))
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        joint[x, y, z] = (prevalence if x else 1-prevalence)*(label_error if y != x else 1-label_error)*(encoding_error if z != x else 1-encoding_error)
            information_xy = mi(joint.sum(2))
            information_zy = mi(joint.sum(0).T)
            conditional = sum(joint[:, :, z].sum()*mi(joint[:, :, z]) for z in range(2))
            close(information_xy-information_zy, conditional)
counts['changed_processing_laws'] = 27
counts['rejected_model_inputs'] = fixtures['invalidCases']
(directory / 'results.json').write_text(json.dumps({'counts': counts, 'maximum_absolute_error': maximum_error, 'python': sys.version}, indent=2))
print(json.dumps({'counts': counts, 'maximum_absolute_error': maximum_error}, indent=2))
