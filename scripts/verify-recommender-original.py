"""Execute the unchanged legacy NumPy code, retaining actual and claimed output separately."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import implicit
import surprise

ROOT = Path(__file__).resolve().parents[1]
packet = json.loads((ROOT / 'docs/teaching/evidence/recommender-systems-original.json').read_text(encoding='utf8'))
records = []
for index in range(6):
    original = packet['programs'][index]
    assert hashlib.sha256((ROOT / original['path']).read_bytes()).hexdigest() == original['sha256']
    namespace = {}
    with redirect_stdout(StringIO()):
        if index in (1, 2, 3):
            exec(packet['programs'][0]['code'], namespace)
    output = StringIO()
    with redirect_stdout(output):
        exec(original['code'], namespace)
    records.append({'index': index, 'sourceSha256': original['sha256'], 'setupBlock': 0 if index in (1, 2, 3) else None, 'stdout': output.getvalue()})
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'numpy': np.__version__, 'passedExecution': True,
          'implicit': implicit.__version__, 'surprise': surprise.__version__,
          'records': records, 'pendingOriginalLibraryBlocks': [],
          'note': 'Execution is not endorsement of claimed comments or teaching. Dependencies are restored explicitly for fragment execution.'}
(ROOT / 'docs/teaching/evidence/recommender-systems-original-execution.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
