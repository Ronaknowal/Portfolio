"""Bind a completed, separately attributed reviewer record to unchanged author sources."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

folder=Path('scratch/conditioning-independent-review')
author_path=Path('docs/teaching/evidence/conditioning-stability-author-review.json')
author=json.loads(author_path.read_text(encoding='utf-8'))
native=json.loads((folder/'results.json').read_text(encoding='utf-8'))
browser=json.loads((folder/'browser-results.json').read_text(encoding='utf-8'))
assert native['passed'] and browser['passed']
assert native['productionSources']==browser['productionSources']==author['productionSources']
for source in author['productionSources']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
images=['tiny-cancellation-320.png','positive-addition-tree-390.png','reference-branches-320.png',
        'scaled-backward-witness-320.png','nonunique-measurements-390.png','signed-propagation-320.png',
        'positive-addition-tree-1440.png','ordinary-residual-reading-320.png','changed-report-answer-390.png']
opened=[{'path':(folder/name).as_posix(),'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest(),
         'review':'Actually opened by the independent reviewer using view_image.'} for name in images]
packet={
    'topicId':'conditioning-stability-numerical-analysis',
    'reviewedAt':datetime.now(timezone.utc).isoformat(),
    'authorFrozenAt':author['frozenAt'],
    'authorPacket':{'path':author_path.as_posix(),'sha256':hashlib.sha256(author_path.read_bytes()).hexdigest()},
    'status':'independently reviewed; no unresolved material finding; parent production integration remains separate',
    'productionSources':author['productionSources'],
    'productionEdits':[],
    'sourceReading':['Complete lesson including local proofs, eleven actual programs and all ten closing practice answers',
                     'Full six production files, assessed design and author verification narrative',
                     'Zero-reference/normal-result/singularity conditions, finite perturbation denominator, normwise and componentwise attainment, scaling units, summation path counts, refinement contraction, uniform fixed-time defect bound and reference provenance'],
    'complementaryNative':native,
    'reviewerBrowser':browser,
    'actuallyOpenedImages':opened,
    'findings':[],
    'primarySourcesRevisited':[
        {'url':'https://nhigham.com/2020/03/25/what-is-backward-error/','scope':'Nearby-problem definition, different permitted perturbations and first-order forward-error relation.'},
        {'url':'https://fncbook.com/zerostability/','scope':'Unwanted recurrence mode, fixed-interval boundedness and root-condition discussion; no embedded-program execution or video-viewing claim.'},
        {'url':'https://www.netlib.org/lapack/lug/node79.html','scope':'Componentwise versus normwise perturbation permissions and significance of small/zero entries.'}],
    'limits':['The author comprehensive82-state browser run is separately attributed; this reviewer operated16 targeted states per width.',
              'LP optimality checks are numerical finite evidence, not replacements for the independently read attainment proofs.',
              'Rational/polynomial/operator checks and exact stored-input references are distinguished from rounded display readouts.',
              'Only the nine named reviewer images were actually opened; other captures and author images are not counted as reviewer inspection.',
              'No observed beginner study, user acceptance, full-video review, general library guarantee or production integration is claimed.'],
}
destination=Path('docs/teaching/evidence/conditioning-stability-independent-review.json')
destination.write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'reviewedAt':packet['reviewedAt'],'sourceCount':len(packet['productionSources']),
                  'openedImages':len(opened),'findings':packet['findings']},indent=2))
