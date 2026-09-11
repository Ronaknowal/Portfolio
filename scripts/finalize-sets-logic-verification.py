"""Record exact source and actually reviewed image identities after author QA."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

sources = [
    'src/learn/data/topics/sets-logic-relations-proof-techniques.jsx',
    'src/learn/data/sets-logic-models.js',
    'src/learn/data/sets-logic-examples.js',
    'src/learn/components/lesson-labs/SetsLogicLabs.jsx',
    'src/learn/components/lesson-labs/sets-logic-labs.css',
    'src/learn/data/curriculum/blueprints/sets-logic-relations-proof-techniques.js',
]
opened = [
    'ordinary-section-1-1440', 'ordinary-section-5-390', 'ordinary-section-7-320',
    'inline-0-390', 'inline-1-390', 'inline-2-390', 'inline-3-390', 'inline-4-390',
    'set-all-overlap-320', 'countermodel-worlds-390', 'quantifier-common-witness-390',
    'quantifier-empty-domains-320', 'relation-classes-390', 'missing-reflexive-loop-320',
    'missing-transitive-edge-390', 'induction-last-square-390', 'diagonal-changed-output-320',
    'keyboard-final-negation-320', 'quantifier-program-390', 'policy-output-320',
    'practice-3-390', 'practice-9-320', 'practice-10-390', 'learning-resources-390',
]
def fingerprint(filename):
    data = Path(filename).read_bytes()
    return {'path': filename, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

records = {}
for name, filename in [
    ('models', 'scratch/sets-logic-verification/model-results.json'),
    ('native', 'scratch/sets-logic-verification/native-results.json'),
    ('browser', 'scratch/sets-logic-browser/results.json'),
    ('keyboard', 'scratch/sets-logic-browser/keyboard-results.json'),
    ('format', 'scratch/sets-logic-verification/format-conservation.json'),
]:
    record = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    if name != 'format':
        assert record['passed']
    records[name] = {'file': fingerprint(filename), 'result': record}
assert [record['width'] for record in records['browser']['result']['records']] == [1440, 390, 320]
packet = {
    'authorFrozenAt': datetime.now(timezone.utc).isoformat(),
    'topicId': 'sets-logic-relations-proof-techniques',
    'status': 'author-verified; independent review and production integration separate',
    'sources': [fingerprint(filename) for filename in sources],
    'records': records,
    'openedImages': [fingerprint(f'scratch/sets-logic-browser/{name}.png') for name in opened],
    'imageReview': 'These 24 final images were actually opened with view_image and read. Earlier provisional images are not substituted for final identities.',
    'knownLimits': [
        'Bounded finite investigations do not automatically prove unrestricted integer or infinite-set statements; the body supplies separate general arguments.',
        'Native examples and comparison tables retain local horizontal scrolling on narrow screens; diagrams and all four displayed equations fit without it.',
        'Official video identity/descriptions and written sources were inspected; full video playback is not claimed.',
        'Classical two-valued logic, natural numbers starting at zero and total functions on an explicit domain are stated conventions.',
        'No legacy body or programs existed. Shared registration belongs to parent; author verification does not imply independent review, production integration or user acceptance.',
    ],
    'resolvedAuthorFindings': [
        'Named selects now have explicit accessible labels, separate from their option text.',
        'The function-image annotation is separated from the lower input node.',
        'All four roster labels fit the overlap; the outside-region caption is separated from the circle boundaries.',
        'The quantified-negation formula uses explicit multi-line layout at320px; 315px pre-fix content no longer overflows its280px available width.',
        'Local paragraph and reset spacing now separate readouts from following explanatory text.',
    ],
    'harnessCorrections': [
        'Native parity/relations helpers return tuples; tests were corrected to use their actual contracts.',
        'CodeBlock renders direct div text rather than pre; browser checks inspect actual renderer text.',
        'Displayed program order differs from object serialization order; checks identify programs by their exact heading.',
        'The first route attempt encountered parent-owned temporary generation failure; final checks used the repaired current server.',
    ],
}
destination = Path('docs/teaching/evidence/sets-logic-author-review.json')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
print(packet['authorFrozenAt'])
for source in packet['sources']:
    print(source['sha256'], source['path'])
