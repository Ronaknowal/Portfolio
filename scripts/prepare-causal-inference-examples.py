"""Execute every complete learner program and retain its actual stdout."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
original = json.loads((ROOT / 'docs/teaching/evidence/causal-inference-original-content.json').read_text(encoding='utf-8'))
examples = {}


def example(key, title, question, code):
    examples[key] = {'title': title, 'question': question, 'code': code.strip() + '\n'}


example('mechanisms', 'Replace the assignment mechanism',
        'Why do selected treated cases have a different risk from a population whose treatment is forced?', '''
from fractions import Fraction as F
from itertools import product

population = [F(1, 2), F(1, 2)]
assignment = [F(1, 5), F(4, 5)]
outcome = [[F(1, 10), F(1, 5)], [F(3, 10), F(2, 5)]]

def joint(forced=None):
    rows = []
    for z, x, y in product([0, 1], repeat=3):
        # Intervention replaces just the probability of X with a point mass.
        px = (assignment[z] if x else 1 - assignment[z])
        if forced is not None:
            px = F(x == forced)
        py = outcome[z][x] if y else 1 - outcome[z][x]
        rows.append((z, x, y, population[z] * px * py))
    assert sum(row[3] for row in rows) == 1
    return rows

observed = joint()
for x in [0, 1]:
    selected = [row for row in observed if row[1] == x]
    observational_risk = sum(y * mass for _, _, y, mass in selected) / sum(r[3] for r in selected)
    intervention_risk = sum(y * mass for _, _, y, mass in joint(forced=x))
    print('X =', x, '| observed risk =', observational_risk,
          '| intervention risk =', intervention_risk)
''')

example('selection', 'Calculate the association created by selection',
        'If X and Y begin independently, what changes after retaining only cases where at least one is present?', '''
from fractions import Fraction as F
from itertools import product

# Independent causes; C is their common effect and D is a copy of C.
rows = [(x, y, int(x or y), F(1, 4)) for x, y in product([0, 1], repeat=2)]

def risk(x, selected=False):
    keep = [(y, mass) for tx, y, c, mass in rows if tx == x and (not selected or c == 1)]
    return sum(y * mass for y, mass in keep) / sum(mass for _, mass in keep)

print('unselected Y risks:', risk(0), risk(1))
print('given C=1, Y risks:', risk(0, True), risk(1, True))
print('selected association:', risk(1, True) - risk(0, True))
print('Selecting D=1 gives the same set because D=C.')
print('Changing X does not change the independent mechanism of Y.')
''')

example('originalOffer', 'Compare raw and adjusted conversion risks',
        'How much of the raw 22-point difference remains after equal population weighting?', original['originalProgram'])

example('nonidentification', 'Two hidden explanations, exactly the same observed data',
        'Can every observed (X,Y) cell be positive while the causal effect remains unidentified?', '''
from fractions import Fraction as F

# U is equally likely zero/one and X=U. Y=1[V < risk(U,X)] with independent V~Uniform(0,1).
# Both models agree whenever X=U, the cells that observation can reveal.
models = {
    'A': [[F(1, 5), F(9, 10)], [F(1, 10), F(4, 5)]],
    'B': [[F(1, 5), F(1, 10)], [F(9, 10), F(4, 5)]],
}
observed_tables = []
for name, risk in models.items():
    observed = {(x, y): F(1, 2) * (risk[x][x] if y else 1-risk[x][x])
                for x in [0, 1] for y in [0, 1]}
    observed_tables.append(observed)
    do_risks = [sum(risk[u][x] for u in [0, 1]) / 2 for x in [0, 1]]
    print(name, 'observed:', {key: str(value) for key, value in observed.items()})
    print(name, 'do risks:', list(map(str, do_risks)), 'ATE:', do_risks[1]-do_risks[0])
assert observed_tables[0] == observed_tables[1]
assert all(value > 0 for value in observed_tables[0].values())
''')

example('ruleThree', 'Why rule three needs its ancestor exception',
        'Does a downstream action remain irrelevant after selecting a downstream result?', '''
from fractions import Fraction as F

# Y~Bernoulli(1/2), Z=Y, W=Z: graph Y -> Z -> W.
def distribution(forced_z=None):
    return [(y, y if forced_z is None else forced_z, F(1, 2)) for y in [0, 1]]

def risk_given_w_one(rows):
    selected = [(y, mass) for y, w, mass in rows if w == 1]
    return sum(y*mass for y, mass in selected) / sum(mass for _, mass in selected)

print('unconditional P(Y=1):', F(1, 2))
print('unconditional P(Y=1 | do(Z=1)):', F(1, 2))
print('P(Y=1 | W=1):', risk_given_w_one(distribution()))
print('P(Y=1 | do(Z=1), W=1):', risk_given_w_one(distribution(1)))
print('Z is an ancestor of conditioned W: do not cut Y->Z for the rule-three test.')
''')

example('frontdoor', 'Use observed factors despite a hidden common cause',
        'Why does frontdoor work here, and what changes when X has a direct effect on Y?', '''
from fractions import Fraction as F
from itertools import product

assignment, mediator = [F(1, 5), F(4, 5)], [F(1, 10), F(4, 5)]

def compare(direct):
    def risk(u, m, x):
        return F(1, 10) + F(1, 2)*m + F(1, 5)*u + direct*x
    rows = []
    for u, x, m, y in product([0, 1], repeat=4):
        mass = F(1, 2) * (assignment[u] if x else 1-assignment[u])
        mass *= mediator[x] if m else 1-mediator[x]
        mass *= risk(u, m, x) if y else 1-risk(u, m, x)
        rows.append((u, x, m, y, mass))
    def conditional_y(m, x):
        selected = [row for row in rows if row[1] == x and row[2] == m]
        return sum(row[3]*row[4] for row in selected) / sum(row[4] for row in selected)
    # Only observed X,M,Y probabilities occur in this calculation.
    inner = [sum(conditional_y(m, x) * F(1, 2) for x in [0, 1]) for m in [0, 1]]
    proposed = [(1-mediator[x])*inner[0] + mediator[x]*inner[1] for x in [0, 1]]
    truth = [sum(F(1, 2)*((1-mediator[x])*risk(u, 0, x) + mediator[x]*risk(u, 1, x))
                 for u in [0, 1]) for x in [0, 1]]
    print('direct =', direct, '| inner =', list(map(str, inner)))
    print('formula:', list(map(str, proposed)), '| true do:', list(map(str, truth)))
    if direct == 0:
        assert proposed == truth
    else:
        assert proposed != truth

compare(F(0))
compare(F(1, 10))
''')

example('counterfactual', 'Keep the same hidden response type across worlds',
        'Do known treatment-specific risks determine whether this treated success would have failed without treatment?', '''
from fractions import Fraction as F

for overlap in [F(0), F(1, 8), F(1, 4)]:
    # Keys are the paired outcomes (Y(0),Y(1)), not independently redrawn outcomes.
    types = {(0, 0): overlap, (0, 1): F(3, 4)-overlap,
             (1, 0): F(1, 4)-overlap, (1, 1): overlap}
    marginals = [sum(pair[x]*mass for pair, mass in types.items()) for x in [0, 1]]
    # X was randomized. Evidence X=1,Y=1 therefore selects types with Y(1)=1.
    compatible = {pair: mass for pair, mass in types.items() if pair[1] == 1}
    evidence = sum(compatible.values())
    posterior = {pair: mass/evidence for pair, mass in compatible.items()}
    failure_without = sum(mass for pair, mass in posterior.items() if pair[0] == 0)
    print('overlap', overlap, '| do risks', list(map(str, marginals)),
          '| would fail without', failure_without)
    assert marginals == [F(1, 4), F(3, 4)]
    assert sum(posterior.values()) == 1
''')

example('estimators', 'Check both routes to the augmented-estimator identity',
        'Which nuisance model can be wrong, and what fails when both are wrong?', '''
from fractions import Fraction as F
from itertools import product

e = [F(1, 5), F(4, 5)]
mu = [[F(1, 10), F(1, 5)], [F(3, 10), F(2, 5)]]

def expected_score(estimated_e, estimated_mu):
    total = F(0)
    for z, x, y in product([0, 1], repeat=3):
        mass = F(1, 2) * (e[z] if x else 1-e[z]) * (mu[z][x] if y else 1-mu[z][x])
        m0, m1 = estimated_mu[z]
        value = m1-m0 + x*(y-m1)/estimated_e[z] - (1-x)*(y-m0)/(1-estimated_e[z])
        total += mass*value
    return total

wrong_e = [F(1, 2), F(1, 2)]
wrong_mu = [[F(3, 20), F(3, 20)], [F(1, 5), F(1, 5)]]
for label, propensity, outcome in [('both correct', e, mu), ('only propensity correct', e, wrong_mu),
                                  ('only outcomes correct', wrong_e, mu), ('both wrong', wrong_e, wrong_mu)]:
    value = expected_score(propensity, outcome)
    print(label + ':', value)
    if label != 'both wrong':
        assert value == F(1, 10)
print('These are population expectations, not guaranteed finite-sample estimates.')
''')

example('instrument', 'Separate assignment, treatment and complier effects',
        'Why can a randomized encouragement identify a local effect that differs from population ATE?', '''
from fractions import Fraction as F

# name, population share, treatment under encouragement0/1, outcome risk under treatment0/1
types = [
    ('always', F(1, 5), [1, 1], [F(1, 10), F(1, 10)]),
    ('complier', F(3, 5), [0, 1], [F(1, 10), F(2, 5)]),
    ('never', F(1, 5), [0, 0], [F(1, 10), F(9, 10)]),
]
# Encouragement Z is randomized, so each Z group retains these type proportions.
treatment_rates = [sum(share*received[z] for _, share, received, _ in types) for z in [0, 1]]
outcome_rates = [sum(share*risk[received[z]] for _, share, received, risk in types) for z in [0, 1]]
first_stage = treatment_rates[1]-treatment_rates[0]
assignment_effect = outcome_rates[1]-outcome_rates[0]
population_ate = sum(share*(risk[1]-risk[0]) for _, share, _, risk in types)
print('treatment rates:', list(map(str, treatment_rates)))
print('outcome rates:', list(map(str, outcome_rates)))
print('assignment effect:', assignment_effect, '| first stage:', first_stage)
print('Wald ratio / complier effect:', assignment_effect / first_stage)
print('population ATE in this fully known model:', population_ate)
assert assignment_effect / first_stage == F(3, 10)
assert population_ate == F(17, 50)
''')

example('uncertainty', 'Resample within the original treatment/activity cells',
        'What sampling variation remains after choosing a valid adjustment?', '''
import math
import random

# Fixed cell sizes and target population weights; outcomes are independent within/between cells.
cells = [(20, 4), (80, 8), (80, 32), (20, 6)]  # low treated/control, high treated/control
weights = [0.5, -0.5, 0.5, -0.5]
observed = sum(w*k/n for w, (n, k) in zip(weights, cells))
rng = random.Random(18)
draws = []
for _ in range(5000):
    # Resampling binary empirical outcomes in a fixed cell equals Bernoulli(k/n) draws.
    estimates = [sum(rng.random() < k/n for _ in range(n))/n for n, k in cells]
    draws.append(sum(w*rate for w, rate in zip(weights, estimates)))
draws.sort()

def percentile(p):
    position = (len(draws)-1)*p
    lower = math.floor(position)
    upper = min(lower+1, len(draws)-1)
    return draws[lower] + (position-lower)*(draws[upper]-draws[lower])

print('adjusted estimate:', round(observed, 4))
print('bootstrap 2.5% and 97.5%:', round(percentile(.025), 4), round(percentile(.975), 4))
print('Resamples:', len(draws), '| fixed-cell percentile approximation; not exact coverage.')
print('This quantifies sampling variation under the design, not hidden-confounding bias.')
''')

results = []
for key, item in examples.items():
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-I', '-c', item['code']],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    item['expected'] = completed.stdout.strip()
    results.append({'key': key, 'stdout': item['expected']})
assert examples['originalOffer']['code'].strip() == original['originalProgram'].strip()
assert examples['originalOffer']['expected'] == original['expected']
destination = ROOT / 'src/learn/data/causal-inference-examples.js'
destination.write_text('export const causalInferenceExamples = ' + json.dumps(examples, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
evidence = ROOT / 'scratch/causal-inference-verification'
evidence.mkdir(parents=True, exist_ok=True)
(evidence / 'prepared-program-results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
print(json.dumps(results, indent=2))
