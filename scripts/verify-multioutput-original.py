"""Execute untouched legacy programs with their explicit notebook dependencies."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['SCIKIT_LEARN_DATA'] = str(ROOT / 'scratch/multioutput-original-data')
import numpy as np
import sklearn

packet = json.loads((ROOT / 'docs/teaching/evidence/multioutput-original.json').read_text(encoding='utf8'))
destination = ROOT / 'docs/teaching/evidence/multioutput-original-execution.json'
records = []
indices = [6] if '--text-only' in sys.argv else list(range(6))
if indices == [6] and destination.exists():
    records = json.loads(destination.read_text(encoding='utf8'))['records']
for index in indices:
    original = packet['programs'][index]
    assert hashlib.sha256((ROOT / original['path']).read_bytes()).hexdigest() == original['sha256']
    namespace = {'np': np}
    setup_indices = list(range(index)) if index in (1, 2, 3, 4) else []
    with redirect_stdout(StringIO()):
        for setup in setup_indices:
            exec(packet['programs'][setup]['code'], namespace)
    output, errors = StringIO(), StringIO()
    failure = None
    try:
        with redirect_stdout(output), redirect_stderr(errors):
            exec(original['code'], namespace)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    records.append({'index': index, 'sourceSha256': original['sha256'], 'setupBlocks': setup_indices, 'restoredImport': 'numpy as np' if index == 6 else None, 'stdout': output.getvalue(), 'stderr': errors.getvalue(), 'failure': failure})
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'records': records, 'pendingOriginalBlocks': [index for index in range(7) if index not in {record['index'] for record in records}], 'note': 'Actual execution preserves original bytes, not their claims. Block 6 also depends on an omitted NumPy import; the namespace restores it explicitly. Its random added labels are synthetic noise, not a genuine text annotation benchmark.'}
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
