"""Changed exact laws and actual native programs; separate from author fixtures."""
import contextlib
import io
import json
import math
from datetime import datetime, timezone
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

directory=Path('scratch/random-variables-independent')
data=json.loads((directory/'cases.json').read_text(encoding='utf-8'))
def q(x): return Q(str(x))
def close(actual,expected):
    expected=float(expected)
    # Relative check protects tiny nonzero moments; exact zero allows roundoff.
    assert math.isclose(actual,expected,rel_tol=2e-11,abs_tol=0 if expected else 2e-12),(actual,expected)
def moments(values,masses):
    total=sum(masses); p=[v/total for v in masses]
    mean=sum(x*w for x,w in zip(values,p))
    variance=sum((x-mean)**2*w for x,w in zip(values,p))
    second=sum(x*x*w for x,w in zip(values,p))
    return mean,variance,second
def check_moments(state,values,masses):
    mean,variance,second=moments(values,masses)
    close(state['mean'],mean); close(state['variance'],variance); close(state['second'],second)
    return mean,variance
for c in data['finite']:
    check_moments(c['state'],list(map(q,c['values'])),list(map(q,c['masses'])))
for c in data['paired']:
    p=[q(r['mass']) for r in c['rows']]; xs=[q(r['x']) for r in c['rows']]; ys=[q(r['y']) for r in c['rows']]
    mx,vx=check_moments(c['state']['x'],xs,p); my,vy=check_moments(c['state']['y'],ys,p)
    covariance=sum((x-mx)*(y-my)*w for x,y,w in zip(xs,ys,p))
    close(c['state']['covariance'],covariance)
    if vx and vy: close(c['state']['correlation'],float(covariance)/math.sqrt(float(vx))/math.sqrt(float(vy)))
    else: assert c['state']['correlation'] is None
for c in data['noise']:
    common,local,a,b=map(q,[c['common'],c['local'],c['a'],c['b']])
    triples=list(product([-1,1],repeat=3)); p=[Q(1,8)]*8
    values=[a*(10+common*s+local*e)+b*(20+common*s+local*f) for s,e,f in triples]
    check_moments(c['state']['combined'],values,p)
    close(c['state']['covariance'],common**2)
    for actual,expected in zip(c['state']['components'],[((a+b)*common)**2,(a*local)**2,(b*local)**2]): close(actual,expected)
for c in data['samples']:
    n,p=c['n'],Q(c['percent'],100)
    law={Q(k,n):Q(0) for k in range(n+1)}
    for bits in product([0,1],repeat=n):
        weight=math.prod(p if bit else 1-p for bit in bits)
        law[Q(sum(bits),n)]+=weight
    for row in c['state']['independent']: close(row['mass'],law[Q(round(row['value']*n),n)])
    close(c['state']['independentVariance'],p*(1-p)/n); close(c['state']['copiedVariance'],p*(1-p))
for c in data['conditional']:
    p=Q(c['p'],100); between=16*p*(1-p)
    close(c['state']['betweenVariance'],between); close(c['state']['withinVariance'],1)
    close(c['state']['betweenCovariance'],between); close(c['state']['withinCovariance'],-1)
    close(c['state']['covariance'],between-1)
programs=[]; namespaces={}
for name,example in data['examples'].items():
    capture=io.StringIO(); namespace={'__name__':'__main__'}
    with contextlib.redirect_stdout(capture): exec(compile(example['code'],name+'.py','exec'),namespace)
    assert capture.getvalue().strip()==example['expected'].strip(),name
    programs.append(name); namespaces[name]=namespace
for values,masses in [([-5,2,7],[Q(1,4),Q(1,2),Q(1,4)]),([0,3,10],[Q(0),Q(1,4),Q(3,4)])]:
    assert namespaces['moments']['expectation'](values,masses)==sum(x*p for x,p in zip(values,masses))
for exponent in [2,3,4,5]:
    outcomes=[(bits,Q(1,2**exponent)) for bits in product([0,1],repeat=exponent)]
    actual=namespaces['mapping']['pushforward'](outcomes,sum)
    assert actual=={k:Q(math.comb(exponent,k),2**exponent) for k in range(exponent+1)}
result={'at':datetime.now(timezone.utc).isoformat(),'passed':True,'unchangedAdmittedStates':data['conserved'],'sparseOrInheritedInputsRejected':data['sparseRejected'],'changedLaws':{key:len(data[key]) for key in ['finite','paired','noise','samples','conditional']},'actualPrograms':programs,'changedNativeHelperChecks':6,'scope':'Independent exact finite laws and program execution. Original author outputs preserved; invalid-array fix has exact conservation checks for admitted states.'}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
