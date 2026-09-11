"""Execute the displayed report with a changed separation and measurement interval."""
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'src/learn/data/conditioning-stability-examples.js'
examples = json.loads(source.read_text(encoding='utf-8').split(' = ', 1)[1].rstrip().rstrip(';'))
code = next(row['code'] for row in examples if row['id'] == 'report')
assert 'epsilon = 2.0**-16' in code
assert 'F(1, 2**30)' in code
changed = code.replace('epsilon = 2.0**-16', 'epsilon = 2.0**-18').replace('F(1, 2**30)', 'F(1, 2**34)')
namespace = {}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(changed, 'changed-conditioning-report.py', 'exec'), namespace)
epsilon = F(1, 2**18)
matrix = [[F(1), F(1)], [F(1), 1 + epsilon]]
rhs = list(map(F, namespace['b']))
# Independent two-by-two determinant solve of the actual stored RHS.
det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
center = [(rhs[0] * matrix[1][1] - rhs[1] * matrix[0][1]) / det,
          (matrix[0][0] * rhs[1] - matrix[1][0] * rhs[0]) / det]
assert center == namespace['center']
assert list(map(F, namespace['x'])) == center
assert namespace['arithmetic_error'] == 0
assert F(1, 2**24) / epsilon == F(1, 64) > F(1, 10000)
assert F(1, 2**34) / epsilon == F(1, 65536) < F(1, 10000)
record = {'completedAt': datetime.now(timezone.utc).isoformat(), 'exampleSourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'changedCode': changed, 'stdout': stream.getvalue(), 'independentStoredInputCenter': list(map(str, center)),
          'checks': ['Cramer rule on exact stored RHS', 'exact final central forward error', 'two independent uncertainty-radius decisions']}
destination = ROOT / 'scratch/conditioning-stability-native/changed-report.json'
destination.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(stream.getvalue())
