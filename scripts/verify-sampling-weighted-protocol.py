from fractions import Fraction as F
from itertools import product
from random import Random
from pathlib import Path
from datetime import datetime, timezone
import json
sizes=(10,10,20,20,30,30)
pairs=((0,1),(2,3),(4,5))
y0=(40,42,60,62,80,82)
weights=[F(sum(sizes[i] for i in pair),sum(sizes)) for pair in pairs]
records=[]
for first in product((-2,0,3),repeat=3):
    effects=tuple(v for e in first for v in (e,e+1))
    values=[]
    for sides in product((0,1),repeat=3):
        differences=[F(y0[pair[side]]+effects[pair[side]]-y0[pair[1-side]]) for pair,side in zip(pairs,sides)]
        values.append(sum(w*d for w,d in zip(weights,differences)))
    truth=F(sum(s*e for s,e in zip(sizes,effects)),sum(sizes))
    assert sum(values)/len(values)==truth
    records.append({'effects':effects,'target':str(truth),'mean':str(sum(values)/len(values))})
rng=Random(50)
treated=[pair[rng.choice((0,1))] for pair in pairs]
assert treated==[1,3,5]
contrasts=[y0[i]+3-y0[pair[0] if i==pair[1] else pair[1]] for pair,i in zip(pairs,treated)]
assert contrasts==[5,5,5] and sum(w*d for w,d in zip(weights,contrasts))==5
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'passed':True,'heterogeneousWeightedProtocols':len(records),'allocationsPerProtocol':8,'records':records,'seed50':{'treated':['C'+str(i+1) for i in treated],'contrasts':contrasts,'weights':list(map(str,weights)),'estimate':'5','constructedEffect':'3'},'method':'Separate exact enumeration of weighted potential outcomes, no browser-model recurrence'}
Path('scratch/sampling-measurement-verification/protocol-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
