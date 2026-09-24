"""Freeze topic-owned source and actual final evidence after visual inspection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

sources = [
    'src/learn/data/topics/counting-combinatorics-mathematical-induction.jsx',
    'src/learn/data/counting-combinatorics-models.js',
    'src/learn/data/counting-combinatorics-examples.js',
    'src/learn/components/lesson-labs/CountingCombinatoricsLabs.jsx',
    'src/learn/components/lesson-labs/CountingRecurrenceFigures.jsx',
    'src/learn/components/lesson-labs/counting-combinatorics-labs.css',
    'src/learn/data/curriculum/blueprints/counting-combinatorics-mathematical-induction.js',
]
opened = [
    'inline-0-390', 'inline-1-390', 'inline-2-390', 'inline-3-390',
    'unequal-fiber-390', 'moved-allocation-390', 'triple-restored-390',
    'unsupported-proof-chain-390', 'reflected-path-320', 'empty-balanced-path-320',
    'changed-coefficient-320', 'alternating-orbit-320',
    *[f'equation-{index}-320' for index in range(14)],
    'final-tiling-decomposition-1440', 'final-tiling-decomposition-320',
    'final-partition-decomposition-1440', 'final-partition-decomposition-390',
    'final-partition-decomposition-320', 'final-partition-reading-320',
    'ordinary-section-1-1440', 'ordinary-section-9-390', 'practice-2-390',
    'practice-7-320', 'practice-11-390', 'reflection-program-390',
    'capstone-output-320', 'learning-resources-390',
]
assert len(opened) == 40


def fingerprint(filename):
    data = Path(filename).read_bytes()
    return {'path': filename, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


records = {}
for name, filename in [
    ('models', 'scratch/counting-combinatorics-verification/model-results.json'),
    ('native', 'scratch/counting-combinatorics-verification/native-results.json'),
    ('browser', 'scratch/counting-combinatorics-browser/results.json'),
    ('finalRecurrenceFigures', 'scratch/counting-combinatorics-browser/final-recurrence-results.json'),
    ('formatConservation', 'scratch/counting-combinatorics-verification/format-conservation.json'),
]:
    record = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    if name != 'formatConservation':
        assert record['passed']
    records[name] = {'file': fingerprint(filename), 'result': record}
assert [row['width'] for row in records['browser']['result']['records']] == [1440,390,320]
assert [row['width'] for row in records['finalRecurrenceFigures']['result']['records']] == [1440,390,320]
packet = {
    'authorFrozenAt': datetime.now(timezone.utc).isoformat(),
    'topicId': 'counting-combinatorics-mathematical-induction',
    'status': 'author-verified; independent review, production integration and user acceptance separate',
    'productionSourceCount': len(sources),
    'sources': [fingerprint(filename) for filename in sources],
    'records': records,
    'originalPlan': fingerprint('docs/teaching/evidence/counting-original-plan.json'),
    'openedImages': [fingerprint(f'scratch/counting-combinatorics-browser/{name}.png') for name in opened],
    'imageReview': 'All forty listed images were actually opened and read with view_image. The comprehensive run precedes the final two recurrence figures; their separate three-width final record and six opened captures verify those later scoped additions. Earlier provisional images are not substituted.',
    'resolvedFailures': [
        {'kind': 'author formula-string escape defect', 'file': fingerprint('scratch/counting-combinatorics-browser/initial-math-rendering-failure.json'), 'resolution': 'All14 actual templates rewritten with literal backslashes, KaTeX parsing regression and final pixel review.'},
        {'kind': 'seven-pixel320 Stirling equation overflow', 'file': fingerprint('scratch/counting-combinatorics-browser/initial-stirling-fit-failure.json'), 'resolution': 'Separated base-condition lines, verified at all three widths and opened at320.'},
        {'kind': 'focused harness CSS grid assertion', 'resolution': 'Read gridColumnStart for the span shorthand and independently verify actual unit widths; production spans were already correct.'},
    ],
    'knownLimits': [
        'Finite constructions and enumerations illustrate and independently check instances; the general proofs remain mathematical source arguments requiring separate review.',
        'The lesson has bounded exact investigations and no empirical performance graph. Large admitted formulas useBigInt/Python integers; this does not imply constant bit cost.',
        'Code/output blocks retain local horizontal scrolling on narrow screens; diagrams and all14 equations fit within the reading column.',
        'Official videos were verified through titles/pages and matching notes; full playback is not claimed.',
        'No prior authored body or program existed. Core scope and selected optional branches do not exhaust specialist combinatorics or the interview bit-manipulation ownership gap.',
        'No shared catalogue/manifest/order edits or integrated production build were performed by this author.',
    ],
}
target = Path('docs/teaching/evidence/counting-combinatorics-author-review.json')
target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(packet['authorFrozenAt'])
for source in packet['sources']:
    print(source['path'], source['sha256'])
