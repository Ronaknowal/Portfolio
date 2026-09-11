"""Save the bounded review and fingerprints; does not mutate production."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
author = json.loads((root / 'docs/teaching/evidence/numerical-methods-author-review.json').read_text(encoding='utf8'))
checks = json.loads((root / 'scratch/numerical-methods-independent-review/results.json').read_text(encoding='utf8'))
production = checks['sourceHashes']
def fingerprint(path):
    data = (root/path).read_bytes()
    return {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
for row in production:
    assert fingerprint(row['path'])['sha256'] == row['sha256']
names = [
    'final-plot-0-320.png', 'final-plot-3-390.png',
    'final-blind-scale-320.png', 'final-nested-program-390.png',
    'final-lost-derivative-geometry-390.png', 'final-actual-error-plot-390.png',
    'final-equation-6-320.png', 'final-plot-6-1440.png', 'final-plot-8-390.png',
]
packet = {
    'topicId': 'numerical-methods-finite-differences-quadrature-root-finding',
    'modulePosition': 38,
    'reviewedAt': datetime.now(timezone.utc).isoformat(),
    'status': 'bounded independent review complete; no unresolved material finding',
    'authorEvidence': 'docs/teaching/evidence/numerical-methods-author-review.json',
    'production': production,
    'reads': ['entire current body and all ten changed practice groups',
        'all thirteen actual complete Python programs, prompts and interpretations',
        'entire pure model, lab and scoped CSS source',
        'archived original body/code/output, individual blueprint, design and verification',
        'resolved incoming Numerical Methods note and outgoing Backpropagation ownership'],
    'complementaryChecks': checks,
    'reviewScripts': [fingerprint(path) for path in [
        'scripts/verify-numerical-methods-independent.mjs',
        'scripts/verify-numerical-methods-independent.py',
    ]],
    'openedImages': [fingerprint('scratch/numerical-methods-browser/'+name) for name in names],
    'imageAttribution': 'These nine author-generated final screenshots were actually opened and inspected by this independent reviewer. This reviewer did not rerun or claim the author\'s full browser interaction matrix.',
    'primaryReferencesInspected': [
        {'url': 'https://dlmf.nist.gov/3.5', 'scope': 'Trapezoid/Simpson assumptions and error constants, Richardson and Gaussian quadrature definitions.'},
        {'url': 'https://fncbook.com/python/adaptive/', 'scope': 'Adaptive refinement and the estimated-error distinction.'},
        {'url': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html', 'scope': 'Output-time, error-control and event-detection API qualifications; fetched documentation1.18.0, execution1.18.1.'},
    ],
    'findings': [],
    'productionEdits': False,
    'limits': ['Finite mathematical/runtime review cannot certify arbitrary functions or arithmetic ranges.',
        'The lesson explicitly excludes complete outward-rounded evaluation enclosures.',
        'Original content preservation is exact for the executed program and output; surrounding useful concepts were also read for coverage.',
        'No production build, bundle/loading audit, deployment or user acceptance is claimed.'],
}
(root/'docs/teaching/evidence/numerical-methods-independent-review.json').write_text(json.dumps(packet,indent=2),encoding='utf8')
print(json.dumps({'reviewedAt':packet['reviewedAt'], 'sources':len(production), 'openedImages':len(names), 'findings':[]},indent=2))
