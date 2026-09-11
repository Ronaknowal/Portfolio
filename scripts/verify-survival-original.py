"""Execute the five untouched legacy code blocks with explicit notebook setup."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy
import sklearn
import pandas
import lifelines
import sksurv

root = Path(__file__).resolve().parents[1]
packet = json.loads((root / 'docs/teaching/evidence/survival-original.json').read_text(encoding='utf8'))
records = []
for index, original in enumerate(packet['programs']):
    assert hashlib.sha256((root / original['path']).read_bytes()).hexdigest() == original['sha256']
    namespace = {}
    setup_indices = [0] if index in (1, 2) else [3] if index == 4 else []
    output, errors = StringIO(), StringIO()
    failure = None
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            for setup in setup_indices:
                exec(packet['programs'][setup]['code'], namespace)
        with redirect_stdout(output), redirect_stderr(errors):
            exec(original['code'], namespace)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    records.append({'index': index, 'sourceSha256': original['sha256'], 'setupBlocks': setup_indices, 'stdout': output.getvalue(), 'stderr': errors.getvalue(), 'failure': failure})
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'versions': {module.__name__: module.__version__ for module in (np, scipy, sklearn, pandas, lifelines, sksurv)}, 'records': records, 'note': 'Exact old bytes executed in the separate survival environment. np.interp in block0 is the old linear interpolation, not a valid KM step evaluation. Rossi print_summary includes a run timestamp; it is an execution record rather than deterministic expected output. These results do not validate the old surrounding narrative.'}
(root / 'docs/teaching/evidence/survival-original-execution.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
