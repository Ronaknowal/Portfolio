"""Check proposed fixtures, not yet-unwritten production numerical PDE models."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import scipy.linalg as la
import sympy as s

root=Path(__file__).resolve().parents[1]
x=s.symbols('x',real=True)
u=x-2*x**3+x**4
source=12*x*(1-x)
counts={}

def equal(actual,expected,label):
    assert s.simplify(actual-expected)==0,(label,actual,expected)
    counts[label]=counts.get(label,0)+1

def close(actual,expected,label):
    assert np.allclose(actual,expected,rtol=3e-12,atol=2e-12),(label,actual,expected)
    counts[label]=counts.get(label,0)+1

for n in [2,4,6,8,12,16]:
    h=s.Rational(1,n)
    matrix=s.zeros(n-1)
    for j in range(n-1):
        matrix[j,j]=2/h**2
        if j: matrix[j,j-1]=-1/h**2
        if j<n-2: matrix[j,j+1]=-1/h**2
    points=[j*h for j in range(1,n)]
    exact=s.Matrix([u.subs(x,point) for point in points])
    rhs=s.Matrix([source.subs(x,point) for point in points])
    discrete=matrix.inv()*rhs
    barrier=s.Matrix([point*(1-point)/2 for point in points])
    for j,point in enumerate(points):
        equal((rhs-matrix*exact)[j],2*h*h,'quartic_defect')
        equal(discrete[j]-exact[j],h*h*point*(1-point),'exact_nodal_error')
        equal((matrix*barrier)[j],1,'barrier_identity')
    inverse=matrix.inv()
    assert all(entry>=0 for entry in inverse)
    assert max(sum(inverse.row(j)) for j in range(n-1))<=s.Rational(1,8)
    counts['inverse_positive_bound']=counts.get('inverse_positive_bound',0)+1
    equal(max(discrete-exact),h*h/4,'even_grid_nodal_max')
    # Same vector in the unscaled and h^2-scaled equations yields the same bound.
    trial=discrete+s.Rational(1,100)*barrier
    residual=rhs-matrix*trial
    bound=max(abs(v) for v in residual)/8
    scaled=max(abs(v*h*h) for v in residual)/(8*h*h)
    equal(bound,scaled,'equation_scaling_invariance')
    scalar=s.Rational(0);gradient=s.Rational(0)
    q=x*(1-x)
    for j in range(n):
        left,right=j*h,(j+1)*h
        interpolant=q.subs(x,left)*(right-x)/h+q.subs(x,right)*(x-left)/h
        scalar+=s.integrate((q-interpolant)**2,(x,left,right))
        gradient+=s.integrate(s.diff(q-interpolant,x)**2,(x,left,right))
    equal(scalar,h**4/30,'exact_fem_l2_squared')
    equal(gradient,h**2/3,'exact_fem_energy_squared')
    a=np.array(matrix,dtype=float)
    eigen=la.eigh(a,eigvals_only=True)
    formula=np.array([4*n*n*math.sin(k*math.pi/(2*n))**2 for k in range(1,n)])
    close(eigen,formula,'dirichlet_spectrum')
    for ratio in [.25,.5,.6,2,8]:
        dt=ratio/n**2
        be=la.solve(np.eye(n-1)+dt*a,np.eye(n-1))
        cn=la.solve(np.eye(n-1)+dt*a/2,np.eye(n-1)-dt*a/2)
        initial=np.arange(1,n,dtype=float)/n
        new=cn@initial;average=(new+initial)/2
        close(np.dot(new,new)-np.dot(initial,initial),-2*dt*np.dot(average,a@average),'cn_energy_identity')
        close(np.sort(la.eigvalsh(be)),np.sort(1/(1+dt*formula)),'backward_euler_factors')
        close(np.sort(la.eigvalsh(cn)),np.sort((1-dt*formula/2)/(1+dt*formula/2)),'crank_nicolson_factors')

# Exact nonuniform assembly and source functional, independently integrated.
for nodes in [[s.Rational(0),s.Rational(1,4),s.Rational(1,2),s.Rational(3,4),s.Rational(1)], [s.Rational(0),s.Rational(1,3),s.Rational(2,3),s.Rational(1)]]:
    size=len(nodes); stiffness=s.zeros(size); load=s.zeros(size,1); a=s.Rational(1,3)
    for j,(left,right) in enumerate(zip(nodes,nodes[1:])):
        h=right-left; hats=[(right-x)/h,(x-left)/h]
        for p in range(2):
            for q in range(2): stiffness[j+p,j+q]+=s.integrate(s.diff(hats[p],x)*s.diff(hats[q],x),(x,left,right))
        if left<=a<right:
            for p in range(2): load[j+p]+=hats[p].subs(x,a)
    solution=stiffness[1:-1,1:-1].inv()*load[1:-1,0]
    for value,point in zip(solution,nodes[1:-1]): equal(value,min(point,a)*(1-max(point,a)),'point_source_nodal_solution')
    equal(sum(load),1,'point_load_conservation')
    if len(nodes)==5:
        equal(load[1],s.Rational(2,3),'point_load_left_weight');equal(load[2],s.Rational(1,3),'point_load_right_weight')
        left,right=nodes[1:3];h=right-left;theta=(a-left)/h
        exact_peak=a*(1-a);reconstructed=(1-theta)*solution[0]+theta*solution[1]
        equal(exact_peak-reconstructed,s.Rational(1,18),'point_source_between_nodes_peak')

for size in [4,8,16]:
    values=np.array([(j%3)+.2*j for j in range(size)])
    for courant in [0,.25,.5,1]:
        updated=(1-courant)*values+courant*np.roll(values,1)
        close(sum(updated),sum(values),'upwind_mass')
        assert max(updated)<=max(values)+1e-12 and min(updated)>=min(values)-1e-12
        if courant==1: close(updated,np.roll(values,1),'upwind_exact_cell_shift')

equal(1/(s.Rational(1,2)/1+s.Rational(1,2)/10),s.Rational(20,11),'material_interface_flux')
equal(1-s.Rational(20,11)/2,s.Rational(1,11),'material_interface_temperature')
bx=s.Matrix([[-1,1,0],[-1,0,1]])
triangle=s.Rational(1,2)*bx.T*bx
assert triangle==s.Matrix([[1,-s.Rational(1,2),-s.Rational(1,2)],[-s.Rational(1,2),s.Rational(1,2),0],[-s.Rational(1,2),0,s.Rational(1,2)]])
assert triangle*s.ones(3,1)==s.zeros(3,1)
counts['reference_triangle_stiffness']=1
equal(8*(s.Rational(1,1000)-s.Rational(5,8*32**2)),s.Rational('0.0031171875'),'capstone_residual_budget')

record={'checkedAt':datetime.now(timezone.utc).isoformat(),'passed':True,'scope':'Exact and complementary calculations of proposed design fixtures, not final production model/program verification','counts':counts,'versions':{'sympy':s.__version__,'numpy':np.__version__},'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
directory=root/'scratch/numerical-pde-design';directory.mkdir(parents=True,exist_ok=True)
(directory/'results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
