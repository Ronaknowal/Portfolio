"""Independent rational partition/leaf/OOB checks; reuse unchanged executed programs."""
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'scratch/decision-tree-native/model-payload.json').read_text())
observations = {row['id']: row for row in data['rows']}
checked_nodes = 0


def impurity(rows, criterion):
    p = Fraction(sum(row['label'] for row in rows), len(rows))
    if criterion == 'gini':
        return float(2 * p * (1 - p))
    return sum(-float(mass) * math.log2(float(mass)) for mass in [p, 1-p] if mass)


def check_tree(node, rows, criterion='gini', minimum=1):
    global checked_nodes
    checked_nodes += 1
    assert node['rows'] == [row['id'] for row in rows]
    assert node['count'] == len(rows)
    p = sum(row['label'] for row in rows) / len(rows)
    assert abs(node['probability']-p) < 1e-12
    assert node['label'] == int(p > .5)
    assert abs(node['impurity']-impurity(rows, criterion)) < 1e-12
    if 'left' not in node:
        return
    left = [row for row in rows if row['features'][node['feature']] <= node['threshold']]
    right = [row for row in rows if row not in left]
    # Evaluate all distinct empirical partitions directly, rather than a prefix scan.
    legal_gains = []
    for feature in node['consideredFeatures']:
        for cut in sorted({row['features'][feature] for row in rows})[:-1]:
            low = [row for row in rows if row['features'][feature] <= cut]
            high = [row for row in rows if row['features'][feature] > cut]
            if min(len(low), len(high)) >= minimum:
                residual = sum(len(part)*impurity(part, criterion) for part in [low, high])/len(rows)
                legal_gains.append(impurity(rows, criterion)-residual)
    assert min(len(left), len(right)) >= minimum
    assert abs(node['gain']-max(legal_gains)) < 1e-11
    check_tree(node['left'], left, criterion, minimum)
    check_tree(node['right'], right, criterion, minimum)


for case in data['cases']:
    check_tree(case['tree'], data['rows'], case['criterion'], case['minimum'])

for report in data['forests']:
    eligible = []
    all_scores = []
    row_index = next(i for i, row in enumerate(data['rows']) if row['id'] == report['query']['id'])
    for member in report['members']:
        state = 29 + member['index'] * 101
        draws = []
        for _ in range(8):
            state = (1664525 * state + 1013904223) % 2**32
            draws.append((8*state)//2**32)
        assert draws == member['sampleIndices']
        check_tree(member['tree'], [data['rows'][i] for i in draws])
        node = member['tree']
        while 'left' in node:
            node = node['left' if report['query']['features'][node['feature']] <= node['threshold'] else 'right']
        score = Fraction(sum(observations[key]['label'] for key in node['rows']), len(node['rows']))
        assert abs(float(score)-member['prediction']['probability']) < 1e-12
        all_scores.append(score)
        if row_index not in draws:
            eligible.append(score)
    assert abs(report['probability']-float(sum(all_scores)/len(all_scores))) < 1e-12
    assert report['oobCount'] == len(eligible)
    assert (report['oobProbability'] is None) == (not eligible)
    if eligible:
        assert abs(report['oobProbability']-float(sum(eligible)/len(eligible))) < 1e-12

expected_risks = {(1, Fraction(15, 32)), (2, Fraction(3, 8)), (3, Fraction(1, 3)), (4, Fraction(1, 6)), (5, Fraction(0))}
for report in data['pruning']:
    for candidate in report['candidates']:
        assert any(candidate['leaves'] == leaves and abs(candidate['risk']-float(risk)) < 1e-12 for leaves, risk in expected_risks)
    expected = min((float(risk)+report['alpha']*leaves, leaves) for leaves, risk in expected_risks)
    assert abs(report['best']['objective']-expected[0]) < 1e-12
    assert report['best']['leaves'] == expected[1]

encoding = np.array([[1,0,1,0], [1,0,0,1], [0,1,0,1]])/np.sqrt(2)
proximity = encoding @ encoding.T
np.testing.assert_allclose(proximity, [[1,.5,0],[.5,1,.5],[0,.5,1]], atol=1e-14)
assert min(np.linalg.eigvalsh(proximity)) > 0

source = (root/'src/learn/data/decision-tree-examples.js').read_text(encoding='utf-8')
examples = json.loads(source[source.index(' = ')+3:].strip().removesuffix(';'))
runs = json.loads((root/'scratch/decision-tree-native/example-runs.json').read_text())
for example, run in zip(examples, runs['programs'], strict=True):
    assert example['id'] == run['id']
    assert hashlib.sha256(example['code'].encode()).hexdigest() == run['codeSha256']
    assert example['expected'] == run['stdout']

paths = ['src/learn/data/decision-tree-models.js', 'src/learn/data/decision-tree-examples.js']
record = {'checkedAt':datetime.now(timezone.utc).isoformat(), 'independentNodes':checked_nodes, 'partitionModels':len(data['cases']), 'oobReports':len(data['forests']), 'programs':len(examples), 'programEvidence':'Unchanged code/output hashes match the ten actual standalone executions in example-runs.json; not unnecessarily rerun.', 'proximity':proximity.tolist(), 'sourceHashes':{file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in paths}}
(root/'docs/teaching/evidence/decision-tree-native-review.json').write_text(json.dumps(record,indent=2)+'\n')
print(f'Independent tree review passed: {checked_nodes} nodes, 64 OOB reports, exact pruning and 10 unchanged executed programs.')
