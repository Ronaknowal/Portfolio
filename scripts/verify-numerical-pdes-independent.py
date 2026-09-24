"""Complementary reviewer oracles: Green functions, global extrema, exact cell
integrals, Fourier convolution and gradient-space projection. No production edits.
"""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
from math import comb
import contextlib
import hashlib
import io
import json
import math
import sys
import numpy as np
import mpmath as mp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/numerical-pdes-independent-review'
data = json.loads((DIRECTORY / 'fixtures.json').read_text(encoding='utf8'))
mp.mp.dps = 80
counts = {}
maximum_scaled_error = 0

def close(actual, expected, name, atol=3e-11, rtol=3e-11):
    global maximum_scaled_error
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
    maximum_scaled_error = max(maximum_scaled_error, float(np.max(np.abs(a-b)/(1+np.abs(b)))))
    counts[name] = counts.get(name, 0) + 1

namespaces = {}
for key, example in data['examples'].items():
    code = example['code']
    namespace = {'__name__': '__main__'}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, 'actual_example_' + key + '.py', 'exec'), namespace)
    assert output.getvalue().rstrip() == example['expected'].rstrip(), key
    namespaces[key] = namespace
    counts['actual displayed stdout'] = counts.get('actual displayed stdout', 0) + 1

def global_field_error(values, length, scale, profile, left, right):
    """Find every cell extremum: target derivative is monotone on [0,1]."""
    n = len(values)-1
    L, s, a, b = map(mp.mpf, [length, scale, left, right])
    def target(t):
        shape = t-2*t**3+t**4 if profile == 'quartic' else t*(1-t) if profile == 'quadratic' else 0
        return s*L**2*shape+a*(1-t)+b*t
    def derivative(t):
        return s*L**2*(1-6*t*t+4*t**3 if profile == 'quartic' else 1-2*t if profile == 'quadratic' else 0)+b-a
    maximum = mp.mpf(0)
    for j in range(n):
        lo, hi = mp.mpf(j)/n, mp.mpf(j+1)/n
        u0, u1 = mp.mpf(values[j]), mp.mpf(values[j+1])
        slope = n*(u1-u0)
        probes = [lo, hi]
        if (derivative(lo)-slope)*(derivative(hi)-slope) < 0:
            # Monotonicity brackets the unique stationary point. High precision
            # bisection avoids copying the author's error formulas.
            l, r = lo, hi
            for _ in range(180):
                middle = (l+r)/2
                if derivative(middle) > slope:
                    l = middle
                else:
                    r = middle
            probes.append((l+r)/2)
        for t in probes:
            maximum = max(maximum, abs(u0+slope*(t-lo)-target(t)))
    return maximum

for case in data['poisson']:
    p, result = case['input'], case['result']
    assert result['nodes'][0] == result['curve'][0]['x'] == 0
    assert result['nodes'][-1] == result['curve'][-1]['x'] == p['length']
    n, L, s = p['intervals'], F(p['length']), F(p['scale'])
    h = L/n
    force = [12*s*F(j,n)*(1-F(j,n)) if p['profile']=='quartic' else 2*s if p['profile']=='quadratic' else F(0) for j in range(1,n)]
    green = [[h*h*F(min(i,j)*(n-max(i,j)),n) for j in range(1,n)] for i in range(1,n)]
    ideal = [sum(g*f for g,f in zip(row,force))+F(p['left'])*(1-F(i,n))+F(p['right'])*F(i,n) for i,row in enumerate(green,1)]
    close(result['direct'][1:-1], list(map(float,ideal)), 'exact discrete Green solution')
    residual = [force[j-1]-(2*F(result['values'][j])-F(result['values'][j-1])-F(result['values'][j+1]))/h**2 for j in range(1,n)]
    error = [F(result['values'][i])-ideal[i-1] for i in range(1,n)]
    assert all(e == -sum(g*r for g,r in zip(row,residual)) for e,row in zip(error,green))
    assert max(map(abs,error)) <= F(result['certificate']['algebraicBound'])
    counts['exact Green residual identity'] = counts.get('exact Green residual identity',0)+1
    whole = global_field_error(result['values'], p['length'], p['scale'], p['profile'], p['left'], p['right'])
    assert whole <= mp.mpf(result['certificate']['fieldBound'])
    counts['whole-field stationary-point certificate'] = counts.get('whole-field stationary-point certificate',0)+1

for n in [3,5,7,11]:
    for length in [.3,.7,1.1,1.3,2.7]:
        left,right=-2.1,3.7
        nodes,values,_=namespaces['poisson']['poisson'](n,lambda x:1.2+3*x,left,right,length)
        assert nodes[0]==0 and nodes[-1]==length
        expected=[left*(1-x/length)+right*x/length+1.2*x*(length-x)/2+3*x*(length*length-x*x)/6 for x in nodes]
        close(values,expected,'changed native endpoint and cubic exactness')

for n,L,s,left,right,budget,tol in [(7,.7,2.1,-1,3,11,.02),(15,1.3,.7,2,-4,51,.03),(31,2.1,1.2,-.3,.8,101,.01),(12,1.5,0,3,1,50,.1)]:
    values, used, parts, certified = namespaces['certificate']['jacobi_report'](n,F(tol),budget,L,s,left,right)
    actual = global_field_error(values,L,s,'quartic',left,right)
    assert actual <= mp.mpf(float(sum(parts[1:]))) * (1+mp.mpf('1e-14'))
    assert certified == (sum(parts[1:]) <= F(tol))
    counts['changed native global field report'] = counts.get('changed native global field report',0)+1

for case in data['heat']:
    p,r = case['input'],case['result']
    n=p['intervals']; k=p['mode']; dt=p['finalTime']/p['timeSteps']; alpha=p['alpha']
    # Full sine expansion of the represented initial vector, including its
    # floating leakage: this differs from an ideal single-mode recurrence.
    basis=np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,n))/n)
    initial=np.array(r['frames'][0]['explicit'][1:-1])
    coefficients=2/n*basis.T@initial
    spectrum=4*n*n*np.sin(np.pi*np.arange(1,n)/(2*n))**2
    for frame in [r['frames'][1],r['frames'][13],r['frames'][-1]]:
        mu=alpha*dt*spectrum
        for name,factors in [('explicit',1-mu),('backward',1/(1+mu)),('crank',(1-mu/2)/(1+mu/2))]:
            close(frame[name][1:-1],basis@(coefficients*factors**frame['step']),'full sine spectral propagation',atol=4e-13)
    # Compare original helper with changed n/k/time, scaled final time accounts
    # for alpha because that particular native program has alpha fixed at 1.
    for name in ['explicit','backward','crank']:
        values,_,_,_=namespaces['diffusion']['diffuse'](n,80,alpha*p['finalTime'],k,name)
        close(values,r['frames'][-1][name][1:-1],'changed actual native diffusion')

def exact_pulse(cells, shift=F(0)):
    shift %= 1
    result=[]
    for j in range(cells):
        lo,hi=F(j,cells),F(j+1,cells)
        result.append(cells*sum(max(F(0),min(hi,F(1,2)+shift+copy)-max(lo,F(1,4)+shift+copy)) for copy in [-1,0,1]))
    return result

for case in data['transport']:
    p,r=case['input'],case['result']; n=p['cells']; c=F(p['courant']); v=p['velocity']; steps=p['steps']
    initial=exact_pulse(n) if p['profile']=='pulse' else list(map(F,r['frames'][0]['values']))
    # Closed binomial distribution of independent stay/shift choices; no
    # time-step update loop or matrix-power implementation from the lesson.
    values=[sum(F(comb(steps,m))*c**m*(1-c)**(steps-m)*initial[(j-v*m)%n] for m in range(steps+1)) for j in range(n)]
    close(r['frames'][-1]['values'],list(map(float,values)),'binomial circular transport')
    if p['profile']=='pulse':
        exact=exact_pulse(n,F(v*steps,n)*c)
        close(r['frames'][-1]['exact'],list(map(float,exact)),'exact rational translated cell overlap')
        native,_=namespaces['advection']['transport'](n,float(c),steps,v)
        close(native,list(map(float,values)),'changed native binomial transport')

def integrate_poly_squared(coefficients,a,b):
    return sum(x*y*(b**(i+j+1)-a**(i+j+1))/F(i+j+1) for i,x in enumerate(coefficients) for j,y in enumerate(coefficients))

for case in data['elements']:
    p,r=case['input'],case['result']; nodes=list(map(F,p['nodes'])); s,k,a,left,right=map(F,[p['source'],p['conductivity'],p['point'],p['left'],p['right']])
    def value(x):
        return s/k*(x*(1-x)/2 if p['sourceKind']=='constant' else min(x,a)*(1-max(x,a)))+left*(1-x)+right*x
    ideal=[value(x) for x in nodes]
    close(r['values'],list(map(float,ideal)),'FEM Green nodal values')
    l2=F(0); energy=F(0)
    for lo,hi,vl,vr in zip(nodes,nodes[1:],ideal,ideal[1:]):
        slope=(vr-vl)/(hi-lo); intercept=vl-slope*lo
        splits=sorted(set([lo,hi]+([a] if lo<a<hi else [])))
        for b,c in zip(splits,splits[1:]):
            if p['sourceKind']=='constant':
                coefficients=[left-intercept, right-left+s/(2*k)-slope,-s/(2*k)]
            elif c<=a:
                coefficients=[left-intercept,right-left+s/k*(1-a)-slope]
            else:
                coefficients=[left+s/k*a-intercept,right-left-s/k*a-slope]
            derivative=[F(i)*coefficients[i] for i in range(1,len(coefficients))]
            l2+=integrate_poly_squared(coefficients,b,c)
            energy+=k*integrate_poly_squared(derivative,b,c)
    close(r['l2Error']**2,float(l2),'exact expanded field polynomial integral',atol=1e-13)
    close(r['energyError']**2,float(energy),'exact expanded slope polynomial integral',atol=1e-12)
    _,load,values=namespaces['finiteElement']['assemble_p1'](p['nodes'], source=p['source'],point=p['point'] if p['sourceKind']=='point' else None,strength=p['source'],left=p['left'],right=p['right'],conductivity=p['conductivity'])
    close(values,list(map(float,ideal)),'changed native FEM Green solution')

for case in data['triangles']:
    vertices=np.array(case['vertices']); r=case['result']; matrix=np.array(r['stiffness']); area=r['area']
    # Any affine field's nodal energy must integrate its constant ambient
    # gradient. Polarization identifies every quadratic-form coefficient.
    for a,b,c in [(1,0,0),(0,1,0),(2,-3,4),(-.7,1.4,2)]:
        values=a*vertices[:,0]+b*vertices[:,1]+c
        close(values@matrix@values,area*(a*a+b*b),'ambient affine triangle energy')
        close(values@np.array(r['gradients']),[a,b],'ambient affine gradient reproduction')
    _,_,native=namespaces['triangle']['triangle_stiffness'](vertices)
    close(native,matrix,'changed native triangle')

for case in data['coarse']:
    r=case['result']; n=8
    # Gradient incidence turns energy into an ordinary Euclidean norm. QR
    # projects there without forming or solving the author's coarse matrix.
    D=np.zeros((n,n-1))
    for edge in range(n):
        if edge>0:D[edge,edge-1]=-math.sqrt(n)
        if edge<n-1:D[edge,edge]=math.sqrt(n)
    P=np.array(r['prolongation']); smooth=np.array(r['smooth'][1:-1]); final=np.array(r['final'][1:-1])
    Q,_=np.linalg.qr(D@P)
    corrected_gradient=D@smooth-Q@(Q.T@(D@smooth))
    close(D@final,corrected_gradient,'gradient-space orthogonal coarse projection')
    close(np.linalg.norm(D@final),r['energyNorms']['final'],'coarse actual energy norm')
    close((D@P).T@(D@final),np.zeros(3),'coarse gradient orthogonality')

for case in data['neumann']:
    p,r=case['input'],case['result']; n=p['cells']; s=F(p['source']); left=F(p['leftOutward'])
    # Integrate face balance, then integrate its gradient and subtract mean.
    faces=[-left+s*F(j,n) for j in range(n+1)]
    values=[F(0)]
    for face in faces[1:-1]:values.append(values[-1]-face/n)
    mean=sum(values)/n; values=[v-mean for v in values]
    close(r['values'],list(map(float,values)),'integrated Neumann face balance')
    native,mismatch=namespaces['neumann']['conservative_neumann'](n,s,left,F(p['rightOutward']))
    assert native==values and mismatch==0
    counts['changed exact native Neumann balance']=counts.get('changed exact native Neumann balance',0)+1

for case in data['interfaces']:
    p,r=case['input'],case['result']; pos,k1,k2,a,b=map(F,[p['interfacePosition'],p['leftConductivity'],p['rightConductivity'],p['leftTemperature'],p['rightTemperature']])
    # Minimize the exact two-segment Dirichlet energy as a scalar quadratic.
    temp=(k1*a/pos+k2*b/(1-pos))/(k1/pos+k2/(1-pos))
    close(r['interfaceTemperature'],float(temp),'minimum-energy interface temperature')
    close(r['flux'],float(k1*(a-temp)/pos),'minimum-energy interface flux')

assert F(399,128000)==8*(F(1,1000)-F(5,8192))
assert F(2567,2304000)==F(8)/F(3,2)**2*(F(1,1000)-F(45,65536))
counts['changed exact practice budget thresholds']=2
for item in data['sourceHashes']:
    assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
result={'completedAt':datetime.now(timezone.utc).isoformat(),'passed':True,'sourceHashes':data['sourceHashes'],'counts':counts,'maximumScaledError':maximum_scaled_error,'python':sys.version,'numpy':np.__version__,'mpmathPrecision':mp.mp.dps}
(DIRECTORY/'native-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
