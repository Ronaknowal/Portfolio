import contextlib
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path('docs/teaching/evidence/differential-geometry-original-content.json')
archive = json.loads(path.read_text(encoding='utf8'))
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(archive['blocks'][0]['text'], '<original-differential-geometry>', 'exec'), {})
assert stream.getvalue().rstrip() == archive['blocks'][1]['text']
if 'originalExecution' not in archive:
    archive['originalExecution'] = dict(at=datetime.now(timezone.utc).isoformat(),
        python=sys.version, actualStdout=stream.getvalue(), matchesDisplayedOutput=True,
        codeSha256=hashlib.sha256(archive['blocks'][0]['text'].encode()).hexdigest())
    path.write_text(json.dumps(archive,indent=2),encoding='utf8')
print(stream.getvalue())
