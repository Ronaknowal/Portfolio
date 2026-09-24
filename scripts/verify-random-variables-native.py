"""Independent Fraction laws, binomial coefficients, integrals and native helpers."""
import contextlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import mpmath as mp
import numpy as np

folder=Path('scratch/random-variables-verification')
fixtures=json.loads((folder/'model-fixtures.json').read_text())
examples=json.loads((folder/'actual-examples.json').read_text())
counts=Counter(); maximum_error=0

def close(actual,expected,kind='numeric',atol=3e-11):
    global maximum_error
    error=abs(actual-float(expected))
    assert math.isfinite(actual) and math.isclose(actual,float(expected),rel_tol=2e-11,abs_tol=atol),(kind,actual,expected)
    maximum_error=max(maximum_error,error);counts[kind]+=1

def check_moments(state,values,weights):
    mean=sum(x*p for x,p in zip(values,weights))
    second=sum(x*x*p for x,p in zip(values,weights))
    variance=second-mean*mean
    for key,value in [('mean',mean),('second',second),('variance',variance)]:close(state[key],value,'scalar')
    if 0 < variance < Q(1,10**20):
        assert math.isclose(state['variance'],float(variance),rel_tol=2e-12,abs_tol=0)
        counts['tinyRelativeVariance']+=1

def check_pair(state,rows):
    xs,ys,weights=zip(*rows)
    check_moments(state['x'],xs,weights);check_moments(state['y'],ys,weights)
    mx=sum(x*p for x,p in zip(xs,weights));my=sum(y*p for y,p in zip(ys,weights))
    cov=sum(x*y*p for x,y,p in rows)-mx*my
    close(state['covariance'],cov,'paired')
    vx=sum((x-mx)**2*p for x,p in zip(xs,weights));vy=sum((y-my)**2*p for y,p in zip(ys,weights))
    if vx==0 or vy==0:assert state['correlation'] is None;counts['degenerate']+=1
    else:close(state['correlation'],float(cov)/math.sqrt(float(vx*vy)),'correlation')

for f in fixtures['outcome']:
    p,q=Q(f['p'],100),Q(f['q'],100); law=Counter()
    for a,b in product([0,1],repeat=2):
        value=a+b if f['rule']=='heads' else a if f['rule']=='first' else int(a==b)
        law[value]+=(p if a else 1-p)*(q if b else 1-q)
    check_moments(f['state'],list(law),list(law.values()))
    for row in f['state']['distribution']:
        close(row['mass'],law[row['value']],'pushforward')
        close(row['cumulative'],sum(p for x,p in law.items() if x<=row['value']),'cdf')
for f in fixtures['loss']:
    values,weights={'asymmetric':([-2,0,3],[Q(1,4),Q(1,2),Q(1,4)]),'symmetric':([-1,1],[Q(1,2)]*2),'constant':([2],[Q(1)])}[f['preset']]
    check_moments(f['state'],values,weights)
    close(f['state']['loss'],sum((x-Q(f['c']))**2*p for x,p in zip(values,weights)),'squaredLoss')
for f in fixtures['joint']:
    pairs=list(product([-1,0,1],repeat=2)) if f['preset']=='independent' else [(x,x if f['preset']=='matching' else -x if f['preset']=='opposite' else x*x) for x in [-1,0,1]]
    rows=[(Q(x),Q(f['scale']*y+f['shift']),Q(1,len(pairs))) for x,y in pairs]
    check_pair(f['state'],rows)
    for cell in f['state']['cells']:
        joint=sum(p for x,y,p in rows if x==cell['x'] and y==cell['y'])
        product_mass=sum(p for x,y,p in rows if x==cell['x'])*sum(p for x,y,p in rows if y==cell['y'])
        close(cell['mass'],joint,'jointCell');close(cell['product'],product_mass,'jointCell')
    assert f['state']['independent']==(f['scale']==0 or f['preset']=='independent')
for f in fixtures['noise']:
    common,local,a,b=map(Q,[f['common'],f['local'],f['a'],f['b']])
    rows=[(10+common*s+local*e,20+common*s+local*t,Q(1,8)) for s,e,t in product([-1,1],repeat=3)]
    check_pair(f['state'],rows)
    check_moments(f['state']['combined'],[a*x+b*y for x,y,p in rows],[p for x,y,p in rows])
    close(sum(f['state']['components']),sum((a*x+b*y-(10*a+20*b))**2*p for x,y,p in rows),'noiseDecomposition')
for f in fixtures['conditional']:
    p=Q(f['p'],100);rows=[(Q(g+u),Q(g-u),(p if g==2 else 1-p)/2) for g,u in product([-2,2],[-1,1])]
    check_pair(f['state'],rows)
    for key,value in [('withinVariance',1),('betweenVariance',16*p*(1-p)),('withinCovariance',-1),('betweenCovariance',16*p*(1-p))]:close(f['state'][key],value,'conditional')
    assert sum(g['moments'] is None for g in f['state']['groups'])==int(p in [0,1])
mp.mp.dps=65
for f in fixtures['squared']:
    a,b=mp.mpf(f['lower'])/100,mp.mpf(f['upper'])/100
    mass=(b-a)/(mp.sqrt(b)+mp.sqrt(a)) if b+a else mp.mpf(0)
    close(f['state']['mass'],mass,'preimage',atol=3e-15)
    close(sum(right-left for left,right in f['state']['preimages'])/2,mass,'preimage',atol=3e-15)
for f in fixtures['sample']:
    n,p=f['n'],Q(f['p'],100)
    for k,row in enumerate(f['state']['independent']):
        expected=Q(math.comb(n,k))*p**k*(1-p)**(n-k)
        if expected:assert math.isclose(row['mass'],float(expected),rel_tol=3e-13,abs_tol=0)
        else:assert row['mass']==0
        counts['binomialCell']+=1
    close(f['state']['independentVariance'],p*(1-p)/n,'sampleVariance')
    close(f['state']['copiedVariance'],p*(1-p),'sampleVariance')
for f in fixtures['finite']:
    values=list(map(lambda x:Q(str(x)),f['values']));weights=list(map(lambda x:Q(str(x)),f['masses']))
    check_moments(f['state'],values,weights)

# Execute the exact exported programs, then exercise their actual functions.
namespaces={}
for name,example in examples.items():
    namespace={};output=io.StringIO()
    with contextlib.redirect_stdout(output):exec(compile(example['code'],name,'exec'),namespace)
    assert output.getvalue().rstrip()==example['expected'],name
    namespaces[name]=namespace;counts['actualPrograms']+=1
for n in [1,2,3,4,6,9,16,30]:
    for p in [Q(0),Q(1,7),Q(1,2),Q(4,5),Q(1)]:
        rows=namespaces['sample_mean']['mean_law'](n,p)
        assert sum(mass for value,mass in rows)==1
        assert sum(value*mass for value,mass in rows)==p
        assert sum((value-p)**2*mass for value,mass in rows)==p*(1-p)/n
        counts['nativeBinomial']+=1
for p in [Q(k,13) for k in range(14)]:
    overall,variances,covariances,groups=namespaces['conditioning']['decomposition'](p)
    assert overall==(4*p-2,4*p-2,1+16*p*(1-p),-1+16*p*(1-p))
    assert variances==(1,16*p*(1-p)) and covariances==(-1,16*p*(1-p));counts['nativeConditioning']+=1
for size in range(2,7):
    probabilities=[Q(i+1,sum(range(1,size+1))) for i in range(size)]
    outcomes=list(zip(range(size),probabilities))
    induced=namespaces['mapping']['pushforward'](outcomes,lambda index:(index%3)-1)
    assert sum(induced.values())==1
    for power in [0,1,2,3]:
        values=[Q((i%3)-1)**power for i in range(size)]
        exact=sum(value*p for value,p in zip(values,probabilities))
        assert namespaces['moments']['expectation'](values,probabilities)==exact
        counts['nativePushforwardMoment']+=1
for n in range(2,6):
    for p in [Q(1,3),Q(1,2),Q(3,4)]:
        total=Q(0)
        for sequence in product([0,1],repeat=n):
            weight=p**sum(sequence)*(1-p)**(n-sum(sequence))
            x=list(map(Q,sequence));y=[2*v+3 for v in x]
            total+=weight*namespaces['sample_covariance']['sample_covariance'](x,y)
            counts['nativeSampleSequences']+=1
        assert total==2*p*(1-p)
for common,local in product([Q(0),Q(1,3),Q(2),Q(7,2)],[Q(0),Q(1,2),Q(3)]):
    rows=namespaces['noise']['noise_law'](common,local)
    assert namespaces['noise']['variance']([y-x for x,y in rows])==2*local**2
    counts['nativeNoise']+=1
for k in range(13):
    exact=mp.quad(lambda u:u**(2*k)/2,[-1,1])
    close(float(namespaces['continuous']['moment_y'](k)),exact,'nativeIntegral')
for a,b in [(0,1),(.01,.09),(.04,.49),(.99,1),(.5,.5)]:
    intervals,mass=namespaces['continuous']['squared_uniform_interval'](a,b)
    close(mass,mp.sqrt(str(b))-mp.sqrt(str(a)),'nativeInterval')
for n in [2,3,5,8]:
    for offset in [0,2**20,2**40]:
        values=[offset+i*2 for i in range(n)]
        mean,variance=namespaces['stable']['welford'](values)
        close(mean,Q(sum(values),n),'nativeWelford')
        close(variance,Q(n*n-1,3),'nativeWelford')
for d,k in product(range(1,4),range(1,4)):
    z=np.array([[i,i*i-3,(-1)**i][:d] for i in range(1,6)],dtype=float)
    a=np.array([[Q(i+j+1,3) for j in range(d)] for i in range(k)],dtype=object)
    centered=z-z.mean(axis=0);sigma=centered.T@centered/len(z)
    result=namespaces['matrix']['multiply'](namespaces['matrix']['multiply'](a.tolist(),sigma.tolist()),namespaces['matrix']['transpose'](a.tolist()))
    transformed=z@np.array(a,dtype=float).T
    expected=np.cov(transformed,rowvar=False,bias=True)
    assert np.allclose(np.array(result,dtype=float),expected,atol=1e-10);counts['nativeMatrix']+=1
for alpha,k in product([Q(1),Q(3,2),Q(9,5),Q(3),Q(7,2)],range(5)):
    value=namespaces['tails']['pareto_moment'](alpha,k)
    if k>=alpha:assert value is None
    else:assert value*(alpha-k)==alpha
    counts['nativeMomentBoundary']+=1

# Changed practice answers, independently enumerated or evaluated.
x=[Q(-1),Q(2),Q(4)];p=[Q(1,2),Q(1,4),Q(1,4)]
assert sum(a*b for a,b in zip(x,p))==1
assert sum((a-1)**2*b for a,b in zip(x,p))==Q(9,2)
assert sum((a-2)**2*b for a,b in zip(x,p))==Q(11,2)
assert 9+Q(4,5)**2+4*Q(1,5)**2==Q(49,5)
assert sum(np.array([[1,-.8,-.8],[-.8,1,-.8],[-.8,-.8,1]]).flatten()) < 0
assert Q(3,4)*1+Q(1,4)*9+Q(3,4)*(2-3)**2+Q(1,4)*(6-3)**2==6
assert 4*4+9==25
counts['changedPractice']=7
# Same first two moments do not fix a tail probability.
for values,weights,tail in [([-1,1],[Q(1,2)]*2,Q(0)),([-2,0,2],[Q(1,8),Q(3,4),Q(1,8)],Q(1,4))]:
    assert sum(x*p for x,p in zip(values,weights))==0
    assert sum(x*x*p for x,p in zip(values,weights))==1
    assert sum(p for x,p in zip(values,weights) if abs(x)>=Q(3,2))==tail
    counts['equalMomentsDifferentTails']+=1
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'counts':dict(counts),'maximumAbsoluteError':maximum_error,'sources':'Actual exported programs plus independent exact finite laws, binomial coefficient formula, high-precision interval mass and quadrature, NumPy covariance propagation.'}
(folder/'native-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
