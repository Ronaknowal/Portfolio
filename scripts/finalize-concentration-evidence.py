"""Freeze scoped source fingerprints and actual reviewed evidence, excluding this JSON itself."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def read_json(file):
    return json.loads((ROOT/file).read_text(encoding='utf-8'))
def fingerprint(file):
    data=(ROOT/file).read_bytes()
    return {'path':file,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

runtime=[
 'src/learn/data/topics/concentration-inequalities-hoeffding-bernstein-chernoff.jsx',
 'src/learn/data/concentration-inequalities-models.js',
 'src/learn/data/concentration-inequalities-examples.js',
 'src/learn/components/lesson-labs/ConcentrationInequalityLabs.jsx',
 'src/learn/components/lesson-labs/concentration-inequality-labs.css',
 'src/learn/data/curriculum/blueprints/concentration-inequalities-hoeffding-bernstein-chernoff.js',
]
owned=runtime+[
 'scripts/prepare-concentration-examples.py','scripts/verify-concentration-inequalities.py',
 'scripts/format-concentration-inequalities.cjs','scripts/review-concentration-inequalities.cjs',
 'scripts/review-concentration-reading.cjs','scripts/finalize-concentration-evidence.py',
 'docs/teaching/CONCENTRATION-INEQUALITIES-LESSON-DESIGN.md',
 'docs/teaching/CONCENTRATION-INEQUALITIES-VERIFICATION.md',
 'docs/teaching/topic-notes/concentration-inequalities-hoeffding-bernstein-chernoff.md',
 'docs/teaching/topic-notes/pac-learning-vc-dimension.md',
 'docs/teaching/evidence/concentration-original-content.json',
]
native=read_json('scratch/concentration-verification/native-results.json')
browser=read_json('scratch/concentration-browser/results.json')
reading=read_json('scratch/concentration-browser/reading-results.json')
formatting=read_json('scratch/concentration-verification/formatting-results.json')
for file,digest in native['sourceSha256'].items():
    assert fingerprint(file)['sha256']==digest,('native source changed',file)
assert not browser['errors']
assert [row['width'] for row in browser['records']]==[1440,390]
assert [row['width'] for row in reading['records']]==[1440,390,320]
assert all(not row['boxOverflow'] and not row['pageOverflow'] and not row['mathErrors'] and not row['svgTextOverflow'] for row in reading['records'])
opened=[
 'tail-default-1440.png','witness-optimum-1440.png','reading-4-1440.png',
 'detail-proof-chain-390.png','detail-zero-variance-390.png','tail-default-390.png',
 'detail-equation-8-320.png','detail-equation-10-320.png',
 'witness-optimum-390.png','budget-default-390.png','sampling-1-390.png',
 'sampling-2-390.png','sampling-3-390.png','family-default-390.png',
 'detail-plot-2-320.png','detail-plot-3-320.png','detail-plot-7-320.png',
 'detail-zero-variance-320.png','detail-native-program-390.png',
 'detail-changed-variance-hint-390.png','detail-changed-variance-solution-390.png',
 'detail-native-output-390.png','detail-sources-390.png','budget-default-1440.png',
 'sampling-3-1440.png','family-default-1440.png',
]
result={
 'topicId':'concentration-inequalities-hoeffding-bernstein-chernoff',
 'frozenAt':datetime.now(timezone.utc).isoformat(),
 'status':'author-verified; root integrated review and user acceptance are separate',
 'runtimeSourceFingerprints':[fingerprint(file) for file in runtime],
 'ownedFiles':[fingerprint(file) for file in owned],
 'native':native,'browser':browser,'ordinaryReading':reading,'formatting':formatting,
 'openedScreenshots':[fingerprint('scratch/concentration-browser/'+name) for name in opened],
 'openedScreenshotReview':'Actually opened with view_image, not inferred from a successful capture or element bounds. Models/plots, proof flow, count and variance contracts, figures, program/output, separate hint/solution and references were inspected. No real novice-user study claimed.',
 'independentReview':{'reviewer':'root','scope':'Full lesson proofs/constants, model contracts, variance counterexample, fixed-bucket application, finite-population and countable-budget reasoning. No mathematical blocker found.','findingsResolved':['Optional hints added before all eight substantial practice solutions','Visible investigation questions added before all nine native programs'],'independence':'Complementary source review, not an additional numerical case count.'},
 'limitations':['Graph values are calculated finite-law/analytic quantities, not collected data or benchmarks.','Browser models are separate from the complete standard-library Python programs.','MIT video page and description were inspected; full playback was not reviewed.','Sharper empirical-Bernstein, martingale, mixture and stitching methods remain explicitly routed/deferred with scope reasons.','No deployment or global integration build by this scoped author.'],
}
destination=ROOT/'docs/teaching/evidence/concentration-author-review.json'
destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'frozenAt':result['frozenAt'],'runtimeSourceFingerprints':result['runtimeSourceFingerprints'],'openedScreenshots':len(opened),'ownedFiles':len(owned)},indent=2))
