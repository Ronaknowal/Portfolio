"""Freeze only after matching numerical/browser fingerprints and actual image review."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('.')
evidence = Path('docs/teaching/evidence')
native_path = Path('scratch/complex-transforms-verification/results.json')
browser_path = Path('scratch/complex-transforms-browser/results.json')
native = json.loads(native_path.read_text(encoding='utf-8'))
browser = json.loads(browser_path.read_text(encoding='utf-8'))
assert browser['passed']
assert [row['width'] for row in browser['results']] == [1440,390,320]
sources = browser['sourceHashes']
for source in sources:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256'], source['path']
for path, digest in native['numericSourceHashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
for row in browser['results']:
    assert not row['documentOverflow'] and not row['overflowEquations']
    assert row['errors'] == row['warnings'] == row['failedRequests'] == []
    assert all(not entry['outOfBounds'] for entry in row['geometry'])

# These files were actually opened by the author with view_image, not merely captured.
opened_names = [
    'expanded-inverse-range-1440.png', 'expanded-inverse-range-390.png', 'expanded-inverse-range-320.png',
    'expanded-transient-range-1440.png', 'expanded-transient-range-390.png', 'expanded-transient-range-320.png',
    'reading-3-320.png', 'reading-11-320.png', 'default-10-320.png',
]
opened = []
for name in opened_names:
    path = Path('scratch/complex-transforms-browser') / name
    opened.append({'path': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'review': 'Actually opened; author visual inspection, separate from automated geometry bounds.'})
frozen = datetime.now(timezone.utc).isoformat()
packet = {
    'topicId': 'complex-numbers-fourier-laplace-transforms',
    'title': 'Complex Numbers, Fourier & Laplace Transforms',
    'modulePosition': 51,
    'status': 'author-verified; independent review and parent production integration remain separate',
    'frozenAt': frozen,
    'productionSources': sources,
    'preservation': {'previousPublication': False, 'archive': 'docs/teaching/evidence/complex-transforms-original-plan.json', 'identityAndOrderUnchanged': True, 'existingPrerequisitesRetained': True},
    'native': native,
    'browser': browser,
    'actuallyOpenedImages': opened,
    'previousAuthorReview': 'docs/teaching/evidence/complex-transforms-author-review-before-independent-amendment.json',
    'independentReviewAmendment': {
        'findings': ['Declare square-integrability for the finite least-squares norm argument',
                     'Evaluate the actual native finite Laplace integral stably at a removable limit',
                     'Use actual plotted extents for reachable DFT reconstruction and filter transient states'],
        'conservation': json.loads(Path('scratch/complex-transforms-verification/independent-amendment-conservation.json').read_text()),
        'previousImages': 'The previous packet separately records 18 images opened before these targeted amendments; only the nine newly opened final images are listed here.',
    },
    'design': 'docs/teaching/COMPLEX-FOURIER-LAPLACE-LESSON-DESIGN.md',
    'verification': 'docs/teaching/COMPLEX-FOURIER-LAPLACE-VERIFICATION.md',
    'limits': ['No observed beginner user study or user acceptance is claimed.', 'No full-video viewing is claimed; companion text/transcript and video identity review are distinguished.', 'Finite numerical oracles do not establish general sampling, inversion or convergence theorems.', 'Production bundle/build/loading and shared curriculum integration belong to the parent.'],
}
evidence.mkdir(parents=True, exist_ok=True)
(evidence/'complex-transforms-author-review.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(evidence/'complex-transforms-native-results.json').write_text(json.dumps(native,indent=2)+'\n',encoding='utf-8')
(evidence/'complex-transforms-browser-results.json').write_text(json.dumps(browser,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
format_record=Path('scratch/complex-transforms-verification/format-conservation.json').read_text(encoding='utf-8')
(evidence/'complex-transforms-format-conservation.json').write_text(format_record,encoding='utf-8')
print(json.dumps({'frozenAt':frozen,'productionSources':sources,'openedImages':len(opened),'nativeCheckedAt':native['checkedAt'],'browserCheckedAt':browser['checkedAt']},indent=2))
