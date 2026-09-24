"""Freeze only after the actual owned numerical/browser checks have passed."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
paths=[
    'src/learn/data/topics/differential-geometry-riemannian-manifolds.jsx',
    'src/learn/data/differential-geometry-models.js',
    'src/learn/data/differential-geometry-examples.js',
    'src/learn/components/lesson-labs/DifferentialGeometryLabs.jsx',
    'src/learn/components/lesson-labs/differential-geometry-labs.css',
    'src/learn/data/curriculum/blueprints/differential-geometry-riemannian-manifolds.js',
]
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))
model=read('scratch/differential-geometry-verification/model-results.json')
native=read('scratch/differential-geometry-verification/native-results.json')
behavior=read('scratch/differential-geometry-browser/results.json')
reading=read('scratch/differential-geometry-browser/final-reading-results.json')
final=read('scratch/differential-geometry-browser/final-program-results.json')
assert model['passed'] and native['passed']
assert model['modelSha256']==digest(paths[1]) and native['examplesSha256']==digest(paths[2])
for evidence in [behavior,reading,final]:
    assert [row['width'] for row in evidence['results']]==[1440,390,320]
    assert all(not row['errors'] and not row['failedRequests'] for row in evidence['results'])
assert all(row['states']==343 for row in behavior['results'])
for row in reading['results']:
    assert row['geometry']['fontReady']
    assert all(e['content']<=e['width']+1 for e in row['geometry']['equations'])
    assert not row['geometry']['svgOverflow']
for row in final['results']:
    assert row['programs']==14 and row['visibleResiduals']
    assert all(e['content']<=e['width']+1 for e in row['measurements']['equations'])
    assert row['measurements']['fontReady']
checked=datetime.fromisoformat(final['checkedAt'].replace('Z','+00:00')).timestamp()
assert all((ROOT/path).stat().st_mtime<=checked for path in paths), 'Source changed after final browser check.'
opened=[
    *['default-visual-'+str(i)+'-390.png' for i in range(1,6)],
    *['default-visual-'+str(i)+'-320.png' for i in range(6,11)],
    *['final-equation-'+str(i)+'-320.png' for i in [3,7,12,15]],
    'final-capstone-reading-320.png','final-capstone-reading-1440.png',
    'final-cross-product-reading-390.png',
]
images=[dict(path='scratch/differential-geometry-browser/'+name,
             sha256=digest('scratch/differential-geometry-browser/'+name),actuallyOpened=True) for name in opened]
record=dict(topicId='differential-geometry-riemannian-manifolds',modulePosition=42,
    status='author-verified; independent and production integration remain parent-owned',
    frozenAt=datetime.now(timezone.utc).isoformat(),
    sources=[dict(path=path,sha256=digest(path)) for path in paths],
    original=read('docs/teaching/evidence/differential-geometry-original-content.json')['originalExecution'],
    originalSourceSha256='4f9a22535e04bf93314583373a2dcde737579ea8a54e5af2426b4bc3ed07989c',
    design='docs/teaching/DIFFERENTIAL-GEOMETRY-LESSON-DESIGN.md',
    verification='docs/teaching/DIFFERENTIAL-GEOMETRY-VERIFICATION.md',
    resolvedNote='docs/teaching/topic-notes/differential-geometry-riemannian-manifolds.md',
    model=model,native=native,browserBehavior=behavior,browserFinalReading=reading,browserFinalPrograms=final,
    openedScreenshots=images,
    resolvedFindings=[
      'Moved atlas seam labels away from axis-scale labels and shortened the candidate annotation to x+v.',
      'Four 320px equations were split without changing mathematical meaning; all17 now fit.',
      'Final capstone prints gradient and feasibility residuals; the actual line-search-limit result remains explicit.',
      'Defined the cross-product area meaning before the first native program relying on it.',
    ],
    evidenceLimits=[
      'Behavior343states/width precedes narrow equation and prose/output amendments; final targeted checks cover those amendments.',
      'Model/native checks are finite, not a proof of arbitrary floating-point input behavior.',
      'Video resource descriptions/pages were inspected, not full videos watched.',
      'No user acceptance or empirical beginner study; parent owns independent/integrated checks.',
    ])
target=ROOT/'docs/teaching/evidence/differential-geometry-author-review.json'
target.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({'frozenAt':record['frozenAt'],'sources':record['sources'],'openedImages':len(images)},indent=2))
