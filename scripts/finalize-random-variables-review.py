"""Freeze the author packet after the recorded images have been opened."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def fingerprint(path):return {'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

sources=[
 'src/learn/data/topics/random-variables-expectation-covariance.jsx',
 'src/learn/data/random-variables-models.js',
 'src/learn/data/random-variables-examples.js',
 'src/learn/components/lesson-labs/RandomVariableLabs.jsx',
 'src/learn/components/lesson-labs/random-variable-labs.css',
 'src/learn/data/curriculum/blueprints/random-variables-expectation-covariance.js',
]
browser=read('scratch/random-variables-browser/results.json')
reading=read('scratch/random-variables-browser/reading-results.json')
final_text=read('scratch/random-variables-browser/final-text-results.json')
for phase in [browser,reading]:
 assert [row['width'] for row in phase['results']]==[1440,390,320]
 for row in phase['results']:
  assert not row['errors'] and not row['warnings'] and not row['failedRequests']
  assert row['geometry']['documentWidth']==row['width'] and row['geometry']['fontReady']
  assert not row['geometry']['svgOverflow']
  assert all(e['content']<=e['width']+1 for e in row['geometry']['equations'])
  assert row['programs']==14 and len(row['anchors'])==12
assert all(row['states']==234 for row in browser['results'])
assert all(not row['errors'] and not row['failed'] for row in final_text['results'])
images=[
 'default-1-320.png','default-2-320.png','joint-nonlinear-witness-320.png',
 'default-5-320.png','conditional-null-group-320.png','sample-independent-copies-320.png',
 'equation-6-320.png','equation-13-320.png','changed-practice-noise-320.png','reading-10-390.png',
 'final-object-legend-320.png','final-object-legend-1440.png',
 'final-bernoulli-bridge-320.png','final-bernoulli-bridge-1440.png',
 'final-noise-labels-320.png','final-noise-labels-390.png','final-noise-labels-1440.png',
 'final-preimage-table-320.png','final-preimage-table-390.png','final-preimage-table-1440.png',
]
result={
 'topicId':'random-variables-expectation-covariance','modulePosition':48,
 'status':'author-verified; independent and integrated production review remain parent-owned',
 'frozenAt':datetime.now(timezone.utc).isoformat(),'sources':[fingerprint(p) for p in sources],
 'preservation':{'priorStatus':'planned','originalSource':None,'originalProgram':None,'titleAndOrderRetained':True,'inheritedDicePracticeRetained':True,'initialPlan':fingerprint('scratch/random-variables-initial-plan.json')},
 'design':'docs/teaching/RANDOM-VARIABLES-LESSON-DESIGN.md',
 'verification':'docs/teaching/RANDOM-VARIABLES-VERIFICATION.md',
 'destinationNote':'docs/teaching/topic-notes/sampling-measurement-experimental-design.md',
 'native':read('scratch/random-variables-verification/native-results.json'),
 'modelContract':read('scratch/random-variables-verification/contract-results.json'),
 'browser':browser,'ordinaryReading':reading,'finalFocusedText':final_text,
 'amendmentAttribution':[
  'The complete behavioral pass exercises 234 states at each width. The local Bernoulli/binomial bridge and extreme-helper variance floor are separately confirmed in the final ordinary-reading pass, including actual loaded-module accepted/rejected calls.',
  'The final focused pass confirms the object-table caption, Bernoulli bridge, specialist source annotation, signed noise-only labels and the shortened continuous branch Mass heading at all three widths. Noise buttons were keyboard activated; numerical models and native program bytes are unchanged by these final labels.',
  'Images of unchanged numerical plots/equations are retained from the successful complete/ordinary pass; the last caption and noise/table-label amendments have their own final captures. All twenty listed images were actually opened by the author.',
 ],
 'imagesOpened':[fingerprint('scratch/random-variables-browser/'+name) for name in images],
 'checks':[fingerprint(p) for p in [
  'scripts/build-random-variables-examples.py','scripts/verify-random-variables-models.mjs',
  'scripts/verify-random-variables-native.py','scripts/review-random-variables-lesson.cjs',
  'scripts/review-random-variables-final-text.cjs','scripts/format-random-variables-source.mjs',
 ]],
 'limitations':['No real-device data or empirical benchmark is claimed.','Browser models deliberately support declared finite teaching ranges; they are not arbitrary-range statistical software.','Video resource identities were checked; full recordings were not claimed as watched.','This packet does not claim independent review, production integration, or user acceptance.'],
}
path=Path('docs/teaching/evidence/random-variables-author-review.json')
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'frozenAt':result['frozenAt'],'sources':result['sources'],'imagesOpened':len(images)}))
