from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import shutil

root = Path(__file__).resolve().parents[1]
native = json.loads((root/'scratch/recommender-native/verification.json').read_text())
browser = json.loads((root/'scratch/recommender-browser/results.json').read_text())
assert not browser['errors']
assert [record['width'] for record in browser['records']] == [1440, 390, 320]
assert native['sourceHashes'] == browser['sourceHashes']
for relative, expected in native['sourceHashes'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative

# These exact final screenshots were opened, not merely generated.
opened = [
    'inline-figure-1-390.png', 'inline-figure-2-390.png', 'inline-figure-3-390.png',
    'factor-update-390.png', 'equation-1-320.png', 'equation-4-320.png',
    'equation-8-320.png', 'equation-9-320.png', 'equation-10-320.png',
    'changed-experiment-output-390.png', 'changed-gradient-practice-320.png',
    'changed-capstone-practice-390.png', 'reading-intro-1440.png',
    'feedback-missing-320.png', 'neighbor-signed-390.png', 'factor-divergence-320.png',
    'implicit-contours-320.png', 'factor-rotation-320.png', 'pair-gap-390.png',
    'slate-retrieval-miss-1440.png', 'policy-no-support-390.png',
]
image_records = []
for filename in opened:
    relative = 'scratch/recommender-browser/'+filename
    data = (root/relative).read_bytes()
    image_records.append(dict(path=relative, sha256=hashlib.sha256(data).hexdigest(), actuallyOpened=True))

archive = root/'docs/teaching/archive/recommender-systems-author-freeze'
archive.mkdir(parents=True, exist_ok=True)
for relative in native['sourceHashes']:
    destination = archive/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.read_bytes() == (root/relative).read_bytes(), 'Do not overwrite a prior frozen source.'
    else:
        shutil.copyfile(root/relative, destination)

original = json.loads((root/'docs/teaching/evidence/recommender-systems-original.json').read_text())
record = dict(
    timestamp=datetime.now(timezone.utc).isoformat(),
    topicId='recommender-systems-collaborative-filtering-matrix-factorization',
    status='author-verified; independent review and parent integration are separate',
    sourceHashes=native['sourceHashes'],
    sourceArchive='docs/teaching/archive/recommender-systems-author-freeze',
    sourceCount=len(native['sourceHashes']),
    originalArchive='docs/teaching/evidence/recommender-systems-original.json',
    originalExecution='docs/teaching/evidence/recommender-systems-original-execution.json',
    designCalculations='docs/teaching/evidence/recommender-systems-design-calculations.json',
    versions={package: importlib.metadata.version(package) for package in ['numpy', 'scipy', 'pandas', 'implicit', 'scikit-surprise', 'black']},
    native=native,
    browser=browser,
    openedImages=image_records,
    programs=14, investigations=8, inlineFigures=3,
    substantialPractice=11, initialCheckpoint=1,
    authorRepairs=[
        'Replaced unsupported original comments/output numbers with separately executed native outputs; preserved all six old blocks and body bytes.',
        'Corrected current implicit user-by-item/confidence API and actual Surprise predictions; no borrowed model outputs or obsolete blanket NumPy incompatibility claim.',
        'Gave select controls explicit accessible names after a real selector-name failure.',
        'Widened the actual divergent-objective plot margin after the 6.50e+13 tick exceeded the SVG boundary.',
        'Reflowed five equivalent equations and defined numerator/norm/error intermediates after measured 320px overflow.',
        'Fixed review harness assumptions about numeric CSS IDs, real div-based code renderer and example ordering; these were harness defects, separately from production repairs.',
        'Formatted actual Python programs with AST conservation and reran all stdout; final model/native and browser packets bind the final seven source hashes.',
    ],
    research='Full scoped primary-source and alternate-resource ledger in RECOMMENDER-SYSTEMS-LESSON-DESIGN.md; video archive listing verified, no full playback claim.',
    limitations=[
        'Tiny invented datasets and chosen objectives demonstrate mechanisms, not population performance, clinical effects or current production benchmarks.',
        'The browser uses bounded double-precision teaching models and rounded readouts; plots declare their computed geometry and sampling. Parameter-range divergence is an explicit status.',
        'Full code and output are preserved in horizontally scrollable code blocks at narrow widths; the explanatory prose gives the important report values in wrapping text.',
        'Historical paper comparisons and architecture descriptions are scoped to actually inspected passages; package versions and seeds are recorded, not promises of identical future environments.',
        'Independent author-to-author review and shared production loading/integration are parent-owned and not claimed by this packet.',
    ],
)
destination = root/'docs/teaching/evidence/recommender-systems-author-review.json'
assert not destination.exists(), 'Preserve a frozen author packet before any later amendment.'
destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(record['timestamp'], record['sourceCount'], 'source files;', len(opened), 'final images opened')
