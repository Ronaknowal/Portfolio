"""Record exact independent review and the narrowly amended production source."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json


def fingerprint(filename):
    data = Path(filename).read_bytes()
    return {'path': str(filename), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


directory = Path('scratch/real-analysis-independent')
original_path = directory / 'original-author-freeze/author-review.json'
original = json.loads(original_path.read_text(encoding='utf-8'))
author_path = Path('docs/teaching/evidence/real-analysis-author-review.json')
assert original_path.read_bytes() == author_path.read_bytes(), 'Original author packet must stay unchanged'
archive = {'originalAuthorFrozenAt': original['authorFrozenAt'], 'authorPacket': original, 'sources': []}
for source in original['sources']:
    saved = directory / 'original-author-freeze' / source['path']
    assert fingerprint(saved)['sha256'] == source['sha256']
    archive['sources'].append({**source, 'encoding': 'base64', 'content': base64.b64encode(saved.read_bytes()).decode('ascii')})
archive_path = Path('docs/teaching/evidence/real-analysis-original-author-sources.json')
archive_path.write_text(json.dumps(archive, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

native_path = directory / 'results.json'
browser_path = directory / 'browser/results.json'
native = json.loads(native_path.read_text(encoding='utf-8'))
browser = json.loads(browser_path.read_text(encoding='utf-8'))
assert native['passed'] and browser['passed']
sources = [fingerprint(source['path']) for source in original['sources']]
for result in [native, browser]:
    assert {row['path']: row['sha256'] for row in result['sources']} == {row['path']: row['sha256'] for row in sources}
changed = [new['path'] for old, new in zip(original['sources'], sources, strict=True) if old['sha256'] != new['sha256']]
assert changed == ['src/learn/components/lesson-labs/RealAnalysisLabs.jsx']
opened = [
    'exact-decimal-bracket-320', 'negative-endpoint-320', 'ordinary-intro-1440',
    'strict-tail-390', 'compact-domain-390', 'missed-peak-390', 'paired-derivative-1440',
    'weighted-polynomial-390', 'dyadic-observer-320', 'proof-figure-0-390',
    'proof-figure-1-390', 'changed-capstone-320', 'exact-output-390',
    'learning-resources-390', 'negative-endpoint-1440', 'exact-decimal-bracket-390',
]
assert len(opened) == len(set(opened)) == 16
record = {
    'reviewedAt': datetime.now(timezone.utc).isoformat(),
    'topicId': 'real-analysis-sequences-modes-of-convergence',
    'status': 'Independent final review closed; no unresolved material finding; production integration and user acceptance separate',
    'originalAuthorFrozenAt': original['authorFrozenAt'],
    'originalAuthorPacket': fingerprint(author_path),
    'originalAuthorSources': fingerprint(archive_path),
    'productionSourceCount': len(sources), 'sources': sources,
    'changedProductionFiles': changed,
    'scope': 'Full body/proofs/practice and all six production sources read; all thirteen actual native programs executed; complementary mathematical oracles and independent actual-font three-width changed-state/keyboard/program/plot review. Separate from prior design-only review and original author checks.',
    'native': {'file': fingerprint(native_path), 'result': native},
    'browser': {'file': fingerprint(browser_path), 'result': browser},
    'openedImages': [fingerprint(directory / 'browser' / f'{name}.png') for name in opened],
    'openedImageScope': 'All16 listed final reviewer images were actually opened after the final browser run. The original ambiguous author image below was opened separately; it is not a final-source screenshot.',
    'originalPaintEvidence': fingerprint('scratch/real-analysis-browser/changed-series-endpoint-320.png'),
    'findings': [
        {'issue': 'Rounded dyadic endpoint displays were called exact at an ordinary step16 state', 'reproduction': {'actualDyadicLeft': '1.4141998291015625', 'oldPrintedLeft': '1.414199829'}, 'resolution': 'Exact terminating endpoint decimals at the selected binary precision; squared decimals explicitly rounded. Exact model/certificate unchanged.', 'status': 'resolved'},
        {'issue': 'Author native-select capture had an ambiguous negative glyph', 'arithmeticFinding': False, 'resolution': 'Explicit left/right endpoint and interior labels; closed native control, selected value, signed numerical outputs, keyboard and reset checked.', 'status': 'resolved'},
    ],
    'scriptFiles': [fingerprint(filename) for filename in [
        'scripts/verify-real-analysis-independent.mjs', 'scripts/verify-real-analysis-independent.py',
        'scripts/review-real-analysis-independent.cjs', 'scripts/finalize-real-analysis-independent.py',
        'docs/teaching/REAL-ANALYSIS-INDEPENDENT-REVIEW.md',
    ]],
    'limits': [
        'Finite calculations corroborate mathematical arguments and cannot prove infinite convergence claims by themselves.',
        'Only the Labs display contract changed; source, models, examples and stdout were conserved. The original author packet remains unmodified.',
        'The native select popup was explicitly closed for final captures; no calculation error was inferred from a native painting artifact.',
        'Prior author exhaustive field-state/anchor checks remain separately attributed; this is a complementary targeted pass.',
        'No whole-book/video playback claim, full application build, shared metadata edit, universal-domain audit or observed learner study is implied.',
    ],
}
target = Path('docs/teaching/evidence/real-analysis-independent-review.json')
assert not target.exists(), 'Preserve an earlier final record and append an amendment instead'
target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(record['reviewedAt'])
print('Six final source hashes match native/browser; original author packet and all source bytes preserved.')
