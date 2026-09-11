"""Check the actual displayed endpoint helper; retain the previous source/output."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/numerical-pde-endpoint-amendment'
data = json.loads((DIRECTORY / 'native-fixtures.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest
           for file, digest in data['sourceHashes'].items())
counts = {'complete displayed programs': 0, 'changed endpoint helper states': 0,
          'previous Python endpoint discrepancies': 0}
for key, example in data['examples'].items():
    output = StringIO()
    namespace = {}
    with redirect_stdout(output):
        exec(compile(example['code'], key + '.py', 'exec'), namespace)
    assert output.getvalue().strip() == example['expected'].strip(), key
    assert example['expected'] == data['oldExamples'][key]['expected']
    counts['complete displayed programs'] += 1
    if key != 'poisson':
        continue
    previous = {}
    with redirect_stdout(StringIO()):
        exec(compile(data['oldExamples'][key]['code'], 'old-' + key + '.py', 'exec'), previous)
    for intervals in (3, 7, 11, 17, 31):
        for length in (.3, .42, .43, .7, 1.1, 1.7, 2.3):
            for source in (lambda x: 0.0, lambda x: 3.4, lambda x: 12 * 1.7 * x / length * (1 - x / length)):
                args = (intervals, source, -2.3, 4.1, length)
                nodes, values, residuals = namespace['poisson'](*args)
                old_nodes, old_values, old_residuals = previous['poisson'](*args)
                assert nodes[0] == 0.0 and nodes[-1] == length
                assert values[0] == -2.3 and values[-1] == 4.1
                assert nodes[1:-1] == old_nodes[1:-1]
                assert values == old_values and residuals == old_residuals
                assert all(left < right for left, right in zip(nodes, nodes[1:]))
                counts['changed endpoint helper states'] += 1
                counts['previous Python endpoint discrepancies'] += old_nodes[-1] != length
assert counts['previous Python endpoint discrepancies'] > 0
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'sourceHashes': data['sourceHashes'], 'counts': counts,
          'scope': 'Actual programs and narrowly changed physical endpoints; prior full independent numerical suite remains source-versioned separately.'}
(DIRECTORY / 'native-results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
