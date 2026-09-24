from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import cmath
import contextlib
import io
import json
import math
import warnings
import numpy as np
from scipy.integrate import quad

directory = Path('scratch/geometry-trigonometry-verification')
data = json.loads((directory/'cases.json').read_text())
counts = {}

def close(actual, expected, atol=3e-12, rtol=3e-12):
    assert np.allclose(np.asarray(actual,dtype=float), np.asarray(expected,dtype=float), atol=atol, rtol=rtol), (actual, expected)

def area(points):
    return abs(sum(Fraction(x1)*y2-Fraction(x2)*y1 for (x1,y1),(x2,y2) in zip(points,points[1:]+points[:1])))/2

for state in data['dissections']:
    points = state['central']
    sides = [(points[(i+1)%4][0]-points[i][0],points[(i+1)%4][1]-points[i][1]) for i in range(4)]
    squares = [x*x+y*y for x,y in sides]
    assert len(set(squares)) == 1
    assert all(sides[i][0]*sides[(i+1)%4][0]+sides[i][1]*sides[(i+1)%4][1] == 0 for i in range(4))
    assert area(points) == squares[0] == state['squareArea']
    assert (state['a']+state['b'])**2 - 4*Fraction(state['a']*state['b'],2) == area(points)
counts['exactPolygonDissections'] = len(data['dissections'])

for state in data['arcs']:
    fraction = Fraction(state['degrees'],360)
    close(state['arc'],float(fraction)*math.tau*state['radius'])
    close(state['area'],float(fraction)*math.pi*state['radius']**2)
    # Independent parametric-speed integral and polar-area integral.
    endpoint=state['degrees']*math.pi/180
    length=quad(lambda t: abs(1j*state['radius']*cmath.exp(1j*t)),0,endpoint)[0]
    close(state['arc'],length)
counts['arcFractionAndQuadratureCases'] = len(data['arcs'])

for state in data['similar']:
    a,o,s=state['adjacent'],state['opposite'],state['scale']
    close(state['sine'],math.sin(cmath.phase(complex(a,o)))); close(state['cosine'],math.cos(cmath.phase(complex(a,o))))
    assert Fraction(str(state['scaledArea'])) == Fraction(a*o,2)*Fraction(str(s))**2
    close(state['hypotenuse'],abs(complex(a,o)))
counts['similarityCases'] = len(data['similar'])

for state in data['circles']:
    point=cmath.rect(1,state['degrees']*math.pi/180)
    close([state['cosine'],state['sine']],[point.real,point.imag])
    if state['degrees']%180 in [90]:
        assert state['tangent'] is None
    else:
        close(state['tangent'],point.imag/point.real)
counts['circleAndPeriodCases'] = len(data['circles'])

for state in data['bearings']:
    point=complex(state['x'],state['y'])
    close(state['radius'],abs(point))
    if point == 0:
        assert state['angle'] is None
    else:
        expected=cmath.phase(point)
        if expected == -math.pi: expected=math.pi
        close(state['angle'],expected)
        close([point.real,point.imag],[cmath.rect(state['radius'],state['angle']).real,cmath.rect(state['radius'],state['angle']).imag])
counts['allIntegerBearingStates'] = len(data['bearings'])

for state in data['frames']:
    p=complex(*state['point']); o=complex(*state['origin']); factor=cmath.exp(1j*math.radians(state['degrees']))
    rotated=o+factor*(p-o)
    local=(p-o)/factor
    close(state['rotated'],[rotated.real,rotated.imag]); close(state['local'],[local.real,local.imag])
    basis=np.array([[factor.real,-factor.imag],[factor.imag,factor.real]])
    close(state['local'],np.linalg.solve(basis,np.array(state['point'])-state['origin']))
    close(state['reconstructed'],state['point'])
    close(abs(rotated-o),abs(p-o))
counts['complexAndLinearSolveFrameCases'] = len(data['frames'])

state=data['ambiguous']
for candidate in state['candidates']:
    close(math.dist([candidate['c'],0],state['C']),7)
    close(candidate['B'],math.acos((7**2+candidate['c']**2-10**2)/(14*candidate['c'])))
    close(candidate['area'],area([[0,0],[candidate['c'],0],state['C']]))
counts['staticSSAReconstructions']=2

namespaces={}
for example in data['examples']:
    namespace={}; output=io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with contextlib.redirect_stdout(output): exec(compile(example['code'],example['id'],'exec'),namespace)
    assert output.getvalue().rstrip()==example['expected'],example['id']
    namespaces[example['id']]=namespace
counts['actualDisplayedPrograms']=len(namespaces)

ssa=namespaces['triangles']['ssa_thirty']
for a in range(1,31):
    for b in range(1,31):
        actual=ssa(a,b)
        # Exact discriminant and sign conditions determine number, independently of atan.
        height=Fraction(b,2); discriminant=Fraction(a*a)-height*height
        expected=0 if discriminant<0 else 1 if discriminant==0 or a>=b else 2
        assert len(actual)==expected,(a,b,actual,expected)
        for c,B,C in actual:
            close(namespaces['triangles']['sas_side'](b,c,30),a)
            close(B+C,150)
            assert 0<B<180 and 0<C<180 and abs(a-b)<c<a+b
counts['changedActualSSACases']=900

for b in [1,3,7,11]:
    for c in [2,5,13]:
        for angle in [1,30,60,90,120,179]:
            actual=namespaces['triangles']['sas_side'](b,c,angle)
            close(actual,math.sqrt(b*b+c*c-2*b*c*math.cos(math.radians(angle))))
counts['changedActualSASCases']=72

for px,py in [(-8,3),(0,0),(3,-5),(7,11)]:
    for ox,oy in [(-2,1),(0,0),(4,-3)]:
        for degrees in [-150,-90,-15,0,30,90,165]:
            p,o=complex(px,py),complex(ox,oy); factor=cmath.rect(1,math.radians(degrees))
            actual=namespaces['frames']['local_coordinates']((px,py),(ox,oy),degrees)
            expected=(p-o)/factor
            close(actual,[expected.real,expected.imag])
            close(namespaces['frames']['world_coordinates'](actual,(ox,oy),degrees),[px,py])
counts['changedActualFrameCases']=84

for scales in [(3,7),(20,10),(1,1),(40,5)]:
    for point in [(-7,3),(0,0),(9,-4),(2,5)]:
        mapped=namespaces['screen']['screen'](point,(100,-200),scales)
        close(namespaces['screen']['world'](mapped,(100,-200),scales),point)
counts['changedActualScreenCases']=16

rejections = [
    lambda: namespaces['arcs']['sector'](0,30), lambda: namespaces['arcs']['sector'](2,0),
    lambda: namespaces['circle']['components'](float('inf')),
    lambda: namespaces['bearing']['polar'](float('nan'),0),
    lambda: namespaces['triangles']['sas_side'](2,3,0), lambda: ssa(2.5,10), lambda: ssa(True,10),
    lambda: namespaces['frames']['local_coordinates']((1,2),(float('inf'),0),30),
    lambda: namespaces['frames']['world_coordinates']((1,2),(float('nan'),0),30),
    lambda: namespaces['screen']['world']((1,2),scales=(0,1)),
    lambda: namespaces['screen']['screen']((1,2),scales=(float('inf'),1)),
    lambda: namespaces['links']['endpoint'](-1,2,30,60),
]
for reject in rejections:
    try: reject()
    except ValueError: pass
    else: raise AssertionError('Accepted invalid native input')
counts['actualNativeBoundaryRejections']=len(rejections)

for l1 in [1,2,5]:
    for l2 in [1,3]:
        for theta in [-120,0,30,90]:
            for phi in [-90,0,60,180]:
                actual=namespaces['links']['endpoint'](l1,l2,theta,phi)
                expected=cmath.rect(1,math.radians(theta))*(l1+cmath.rect(l2,math.radians(phi)))
                close(actual,[expected.real,expected.imag])
                close(np.dot(actual,actual),l1*l1+l2*l2+2*l1*l2*math.cos(math.radians(phi)))
counts['changedActualLinkCases']=96

# Independent changed-practice acceptance values and units.
close(math.dist((-2,1),(4,-7)),10)
close(namespaces['arcs']['sector'](3,120),[2*math.pi/3,2*math.pi,3*math.pi])
assert Fraction(12*5,2)*Fraction(3,2)**2==Fraction(135,2)
close(namespaces['circle']['components'](225)[:2],[-math.sqrt(.5)]*2)
for angle in [210,330]: close(math.sin(math.radians(angle)),-.5)
close(namespaces['bearing']['polar'](-5,-5),[5*math.sqrt(2),-3*math.pi/4])
close(namespaces['triangles']['sas_side'](4,6,60),2*math.sqrt(7))
close([item[0] for item in ssa(5,8)],[4*math.sqrt(3)-3,4*math.sqrt(3)+3])
close(namespaces['frames']['local_coordinates']((3,1),(1,2),90),[-1,-2])
close(namespaces['frames']['world_coordinates']((2,-1),(1,2),90),[2,4])
close(math.hypot(-60/30,60/20),math.sqrt(13))
close(namespaces['links']['endpoint'](3,2,0,60),[4,math.sqrt(3)])
close(namespaces['links']['endpoint'](3,2,90,60),[-math.sqrt(3),4])
close(8+2*math.sin(6*math.pi/12),10)
assert namespaces['frames']['rotate']((2,1),90)==(-1,2)
counts['independentPracticeGroups']=12

result={'at':datetime.now(timezone.utc).isoformat(),'passed':True,'counts':counts,'modelRejectedCases':data['rejected'],'limits':'Finite implementation checks use exact polygon/ratio arithmetic, independent complex geometry, NumPy solves and quadrature; they do not replace proofs for all real inputs. Actual complete learner programs are executed without hidden helpers.'}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
