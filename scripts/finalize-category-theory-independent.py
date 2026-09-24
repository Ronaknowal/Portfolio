"""Close the independent record only against the repaired author freeze."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('.')
author_path = Path('docs/teaching/evidence/category-theory-author-review.json')
author = json.loads(author_path.read_text(encoding='utf-8'))
initial = json.loads(Path('docs/teaching/evidence/category-theory-author-review-before-entity-fix.json').read_text(encoding='utf-8'))


def fingerprint(path):
    data = Path(path).read_bytes()
    return {'path': str(path).replace('\\', '/'), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


current = [fingerprint(row['path']) for row in author['production']]
assert current == author['production']
assert current[0]['sha256'] == 'b4eec093fa40a2daa11a2c88299f5e9e21148dd980d4d10d5e8fb9fda44f2382'
assert current[1:] == initial['production'][1:]
source = Path(current[0]['path']).read_bytes()
assert b'For q(y)&gt; 0,' in source and b'&gt ;' not in source
old = source.replace(b'For q(y)&gt; 0,', b'For q(y)&gt ; 0,')
line_start = old.index(b'    <Prose>With prior p(x)')
line_end = old.index(b'\n', line_start)
assert old[line_end - 1:line_end] != b'\r'
old = old[:line_end] + b'\r' + old[line_end:]
assert hashlib.sha256(old).hexdigest() == initial['production'][0]['sha256']

image_names = ['default-lab-3-320.png', 'default-lab-5-320.png', 'inline-0-320.png',
               'default-lab-4-320.png', 'naturality-counterexample-1440.png',
               'reading-10-320.png', 'reading-12-390.png', 'conditional-text-320.png']
results = json.loads(Path('scratch/category-theory-independent-review/results.json').read_text(encoding='utf-8'))
assert results['passed']
focused = json.loads(Path('scratch/category-theory-browser/conditional-text-results.json').read_text(encoding='utf-8'))
assert focused['passed'] and focused['bodySha256'] == current[0]['sha256']
assert all('q(y)> 0' in row['text'] and not row['errors'] for row in focused['records'])
record = {
    'topicId': 'category-theory-emerging-use-in-ml',
    'reviewer': 'scientific_visual_improvements',
    'reviewedAt': datetime.now(timezone.utc).isoformat(),
    'status': 'Independent source/proof, complementary execution and named visual review passed; no unresolved material finding.',
    'production': current,
    'authorRecord': fingerprint(author_path),
    'initialAuthorRecord': fingerprint('docs/teaching/evidence/category-theory-author-review-before-entity-fix.json'),
    'initialBodySha256': initial['production'][0]['sha256'],
    'actualScope': ['Full twelve-section body, all eleven actual programs and interpretations, all twelve practice solutions and both checkpoints',
                    'Complete model and lab sources, individual brief, design and author verification; relevant responsive/focus styles',
                    'Primary-source passages independently read: Riehl Yoneda/dual note, products/pullbacks and adjunction; Fritz FinStoch and copying; Backprop as Functor theorem III.2',
                    'Eight named author-generated screenshots actually opened by the reviewer; author full browser runs remain separately attributed'],
    'findings': [{'severity': 'minor learner-visible notation error',
                  'location': 'Section 7, positive-evidence condition for Bayes reversal',
                  'before': 'q(y)&gt ; 0', 'after': 'q(y)&gt; 0',
                  'status': 'Resolved by author; literal symbol inspected in final narrow capture and all three width text results.',
                  'scopeProof': 'Reversing the entity edit and edited-line CRLF/LF change reconstructs the original exact body SHA; other five production sources match initial freeze.'}],
    'complementaryExecution': results,
    'verificationScripts': [fingerprint('scripts/verify-category-theory-independent.mjs'), fingerprint('scripts/verify-category-theory-independent.py')],
    'visualReview': {'attribution': 'Screenshots captured by author and actually opened by this independent reviewer.',
                     'images': [fingerprint('scratch/category-theory-browser/' + name) for name in image_names]},
    'authorFocusedRepairEvidence': focused,
    'limits': ['The reviewer did not rerun or claim the author full 272-state browser matrix as independent evidence.',
               'Finite cases support implementation, not universal proofs or unconstrained numerical guarantees.',
               'Selected primary passages were read; entire books, papers and videos were not reviewed.',
               'No production file was changed by this reviewer. Root owns integrated production checks and rollout status.']
}
Path('docs/teaching/evidence/category-theory-independent-review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'reviewedAt': record['reviewedAt'], 'production': current[0]['sha256'], 'imagesActuallyOpened': len(image_names)}))
