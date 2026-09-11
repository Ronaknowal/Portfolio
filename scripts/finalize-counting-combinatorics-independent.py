"""Save the independent review's exact source and evidence identities."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def fingerprint(path):
    data = Path(path).read_bytes()
    return {'path': str(path).replace('\\', '/'), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

author_path = 'docs/teaching/evidence/counting-combinatorics-author-review.json'
author = json.loads(Path(author_path).read_text(encoding="utf-8"))
sources = []
for source in author['sources']:
    actual = fingerprint(source['path'])
    assert actual == source, source['path']
    sources.append(actual)

author_images = [
    'final-tiling-decomposition-320.png', 'final-partition-decomposition-1440.png',
    'reflected-path-320.png', 'alternating-orbit-320.png', 'unequal-fiber-390.png',
    'moved-allocation-390.png', 'triple-restored-390.png', 'unsupported-proof-chain-390.png',
    'final-partition-decomposition-320.png', 'practice-7-320.png',
]
directory = 'scratch/counting-combinatorics-independent-review'
opened = [fingerprint(f'scratch/counting-combinatorics-browser/{name}') for name in author_images]
opened.extend(fingerprint(f'{directory}/changed-coefficient-independent-{width}.png') for width in [1440, 390, 320])
assert len(opened) == 13
author_lookup = {entry['path']: entry for entry in author['openedImages']}
for entry in opened[:10]:
    assert entry == author_lookup[entry['path']]

report = {
    'reviewedAt': datetime.now(timezone.utc).isoformat(),
    'topicId': 'counting-combinatorics-mathematical-induction',
    'reviewer': '/root/scientific_visual_improvements',
    'authorFrozenAt': author['authorFrozenAt'],
    'status': 'independently-reviewed-no-unresolved-material-finding',
    'authorRecord': fingerprint(author_path),
    'productionSourcesUnchanged': sources,
    'sourceReview': [
        'Full fourteen-section body and twelve changed practice solutions',
        'All thirteen actual complete Python programs and expected output',
        'Entire pure model, seven lab implementations, CSS and both separate recurrence figures',
        'Individual blueprint, lesson design, author verification and separate late recurrence-figure evidence',
    ],
    'mathematicalReview': [
        'Disjoint partitions versus constant-size continuations; probabilities do not follow merely from counting.',
        'Uniform-fiber division, multiset tagging, onto labeled-box multiplicity and empty objects.',
        'Shifted minima, upper-capacity IE, per-object alternating coefficients and precise pigeonhole/injective-code conclusion.',
        'Induction quantified ranges, constructive witnesses and unsupported certificate chains distinct from impossibility.',
        'First-tile/last-item/first-return decompositions; Catalan first-negative reflection is a bijection with first-positive inverse.',
        'Formal finite coefficient convolution versus analytic substitution; cyclic orbit-stabilizer and fixed-pair proof, rotations without identifying reflections.',
    ],
    'complementaryExecution': json.loads(Path(f'{directory}/results.json').read_text(encoding="utf-8")),
    'reviewerBrowserSubset': json.loads(Path(f'{directory}/browser-results.json').read_text(encoding="utf-8")),
    'evidenceFiles': [fingerprint(f'{directory}/{name}') for name in ['results.json', 'browser-results.json', 'fixtures.json']],
    'openedImages': opened,
    'visualAssessment': 'Thirteen actual files opened. Object groupings, token/run correspondence, overlap signs, missing-base chain, reflected endpoint, ring stabilizers and later recurrence pictures remain readable. Reviewer coefficient changes show count 5→1→5 when the final factor is removed/restored; mobile wraps degree chips with explicit exponent labels and readable contributions.',
    'findings': [{
        'severity': 'documentation-consistency',
        'description': 'Design opening still said implementation/evidence were pending despite the completed disposition and frozen packet.',
        'resolution': 'Author updated the opening to link actual verification and distinguish independent/production review. All seven production files remain byte-identical.',
        'finalDesign': fingerprint('docs/teaching/COUNTING-COMBINATORICS-LESSON-DESIGN.md'),
    }],
    'limits': [
        'Finite complementary execution is not a proof for arbitrary input sizes; general arguments were read separately.',
        'Reviewer browser checks are a bounded changed-state/keyboard subset. Author comprehensive 1440/390/320 results and later recurrence checks remain separately attributed.',
        'No claim of replaying full external videos or reading every cited textbook. Source annotations and relevant locally derived proof obligations were assessed.',
        'No production source, shared registry, progression ledger or root integration files were changed by this reviewer.',
    ],
}
target = Path('docs/teaching/evidence/counting-combinatorics-independent-review.json')
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'saved': str(target), 'reviewedAt': report['reviewedAt'], 'unchangedProductionSources': len(sources), 'openedImages': len(opened)}, indent=2))
