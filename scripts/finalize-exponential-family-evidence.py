"""Persist completed author evidence; not a substitute for running its checks."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = [
    'src/learn/data/topics/exponential-families-sufficient-statistics.jsx',
    'src/learn/data/exponential-family-models.js',
    'src/learn/data/exponential-family-examples.js',
    'src/learn/components/lesson-labs/ExponentialFamilyLabs.jsx',
    'src/learn/components/lesson-labs/exponential-family-labs.css',
    'src/learn/data/curriculum/blueprints/exponential-families-sufficient-statistics.js',
    'docs/teaching/EXPONENTIAL-FAMILIES-LESSON-DESIGN.md',
    'docs/teaching/topic-notes/entropy-cross-entropy-kl-divergence.md',
    'docs/teaching/evidence/exponential-family-original-content.json',
    'scripts/prepare-exponential-family-examples.py',
    'scripts/verify-exponential-family.py',
    'scripts/format-exponential-family.cjs',
    'scripts/review-exponential-family.cjs',
    'scripts/finalize-exponential-family-evidence.py',
]
opened = [
    'ordinary-lab-1-390.png', 'ordinary-lab-4-1440.png',
    'ordinary-section-4-390.png', 'sources-390.png',
    'ordinary-figure-1-390.png', 'ordinary-figure-2-320.png',
    'ordinary-figure-3-390.png', 'fitted-moments-390.png',
    'uniform-coordinate-density-390.png', 'equation-1-320.png',
    'equation-3-320.png', 'equation-9-320.png',
    'ordinary-lab-2-390.png', 'boundary-moments-320.png',
    'ordinary-intro-1440.png', 'changed-task-solution-390.png',
    'program-output-390.png',
]
def fingerprint(relative):
    raw = (ROOT / relative).read_bytes()
    return {'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

native = json.loads((ROOT/'scratch/exponential-family-verification/native-results.json').read_text())
browser = json.loads((ROOT/'scratch/exponential-family-browser/results.json').read_text())
formatting = json.loads((ROOT/'scratch/exponential-family-verification/formatting-results.json').read_text())
assert len(native['programs']) == 9 and native['preservedOriginalPrograms'] == 1
assert not browser['errors']
assert [record['width'] for record in browser['records']] == [1440,390,320]
assert all(not record['pageOverflow'] and not record['mathErrors'] and not record['overflow'] and not record['svgTextOverflow'] for record in browser['records'])
record = {
    'frozenAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'author-verified; independent cross-review and root integration separate; user acceptance pending',
    'sourceFiles': [fingerprint(path) for path in paths],
    'native': native, 'browser': browser, 'formatting': formatting,
    'actuallyOpenedFinalScreenshots': [fingerprint('scratch/exponential-family-browser/'+name) for name in opened],
    'limits': ['Finite displayed models are calculated rather than empirical evidence for an application.', 'Beta plot is truncated at eta=plus/minus6 and is not renormalized.', 'Source/model/browser verification is not a beginner user study or universal statistical-mastery guarantee.', 'Full video playback was not performed; official page and specified transcript passages were inspected.', 'Root owns shared registrations, production build and integrated-route review.'],
}
destination = ROOT/'docs/teaching/evidence/exponential-family-author-review.json'
destination.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(record['frozenAt'])
print(json.dumps(record['sourceFiles'][:6],indent=2))
