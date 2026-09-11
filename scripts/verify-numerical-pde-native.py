"""Independent matrix, exact-rational and quadrature checks of actual lesson models."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
import contextlib
import hashlib
import io
import json
import math
import subprocess
import sys
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "scratch/numerical-pde-native"
packet = json.loads((folder / "model-fixtures.json").read_text(encoding="utf8"))
counts = {}
def close(actual, expected, name, atol=2e-11, rtol=3e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    counts[name] = counts.get(name, 0) + 1

def laplacian(n, h):
    return (2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1))/h**2

for case in packet["fixtures"]["tridiagonal"]:
    A = np.diag(case["diagonal"])+np.diag(case["off"],1)+np.diag(case["off"],-1)
    close(case["values"], np.linalg.solve(A, case["rhs"]), "SPD solve")

for case in packet["fixtures"]["poisson"]:
    p, r = case["input"], case["result"]
    n, L, s = p["intervals"], F(p["length"]), F(p["scale"])
    h = L/n
    U = list(map(F, r["values"]))
    forcing = [12*s*F(j,n)*(1-F(j,n)) if p["profile"]=="quartic" else 2*s if p["profile"]=="quadratic" else F(0) for j in range(1,n)]
    residual = max(abs(forcing[j-1]-(2*U[j]-U[j-1]-U[j+1])/h**2) for j in range(1,n))
    nodal = s*h*h/4 if p["profile"]=="quartic" else F(0)
    interp = s*h*h*(3 if p["profile"]=="quartic" else 2)/8 if p["profile"]!="linear" else F(0)
    total = nodal+interp+L*L*residual/8
    assert F(r["certificate"]["exactFieldBound"]) == total
    assert F(r["certificate"]["fieldBound"]) >= total
    assert F(r["certificate"]["residualUpper"]) >= residual
    assert r["certificate"]["certified"] == (total <= F(.001))
    A = laplacian(n-1,float(h))
    b = np.array(list(map(float,forcing)))
    b[0] += p["left"]/float(h*h)
    b[-1] += p["right"]/float(h*h)
    close(r["direct"][1:-1], np.linalg.solve(A,b), "Poisson independent solve")
    if p["method"] == "direct":
        close(r["values"][1:-1], np.linalg.solve(A,b), "Poisson chosen direct")
    counts["exact rational certificate"] = counts.get("exact rational certificate",0)+1

for case in packet["fixtures"]["diffusion"]:
    p,r = case["input"],case["result"]
    n=p["intervals"]; h=1/n; dt=p["finalTime"]/p["timeSteps"]
    A=laplacian(n-1,h); I=np.eye(n-1); initial=np.sin(p["mode"]*np.pi*np.arange(1,n)/n)
    operators={"explicit":I-dt*A, "backward":np.linalg.inv(I+dt*A), "crank":np.linalg.solve(I+dt*A/2,I-dt*A/2)}
    for frame in r["frames"]:
        for method, operator in operators.items():
            if np.max(np.abs(np.linalg.eigvalsh(operator))) <= 1+1e-12:
                close(frame[method][1:-1], np.linalg.matrix_power(operator,frame["step"])@initial, "stable diffusion matrix power", atol=2e-12, rtol=1e-9)
            elif frame["step"]:
                previous=np.array(r["frames"][frame["step"]-1][method][1:-1])
                tolerance=1e-13*max(1,np.linalg.norm(operator,np.inf)*np.linalg.norm(previous,np.inf))
                close(frame[method][1:-1],operator@previous,"unstable one-step operator",atol=tolerance,rtol=1e-13)
        close(frame["semidiscrete"][1:-1], expm(-frame["time"]*A)@initial, "spatial ODE exponential", atol=3e-13)
    threshold=2/(h*h*np.linalg.eigvalsh(A)[-1])
    close(r["finiteGridThreshold"],threshold,"finite-grid threshold")

for case in packet["fixtures"]["interfaces"]:
    p,r=case["input"],case["result"]; a=p["interfacePosition"]
    # Solve continuity plus two constant-flux gradient equations as a 2x2 system.
    temp,q=np.linalg.solve([[1,a/p["leftConductivity"]],[1,-(1-a)/p["rightConductivity"]]],[p["leftTemperature"],p["rightTemperature"]])
    close([r["interfaceTemperature"],r["flux"]],[temp,q],"interface independent balance")
for case in packet["fixtures"]["neumann"]:
    p,r=case["input"],case["result"]; n=p["cells"]; h=1/n
    assert r["compatible"] == (F(p["source"]) == F(p["leftOutward"])+F(p["rightOutward"]))
    if r["compatible"]:
        K=laplacian(n,1)/h
        K[0,0]=K[-1,-1]=1/h
        load=np.full(n,p["source"]*h); load[0]-=p["leftOutward"]; load[-1]-=p["rightOutward"]
        augmented=np.block([[K,np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        expected=np.linalg.solve(augmented,np.r_[load,0])[:-1]
        close(r["values"],expected,"Neumann augmented mean solve")
        close(K@np.array(r["values"]),load,"all Neumann balances")

for case in packet["fixtures"]["advection"]:
    p,r=case["input"],case["result"]; n=p["cells"]; c=p["courant"]; v=p["velocity"]; h=1/n
    shift=np.roll(np.eye(n),v,axis=0)
    A=(1-c)*np.eye(n)+c*shift if p["scheme"]=="upwind" else np.eye(n)-v*c/2*(np.roll(np.eye(n),-1,axis=0)-np.roll(np.eye(n),1,axis=0))
    # The centered row action is U[j+1]-U[j-1].
    initial=np.array(r["frames"][0]["values"])
    for frame in r["frames"]:
        expected=np.linalg.matrix_power(A,frame["step"])@initial
        close(frame["values"],expected,"periodic matrix update")
        close(frame["mass"],h*sum(initial),"conservative mass",atol=1e-9)
        if c==1 and p["scheme"]=="upwind":
            close(frame["values"],frame["exact"],"exact one-cell translation")

for case in packet["fixtures"]["fem"]:
    p,r=case["input"],case["result"]; nodes=np.array(p["nodes"]); k=p["conductivity"]; strength=p["source"]; point=p["point"]
    exact=lambda x: strength/k*(x*(1-x)/2 if p["sourceKind"]=="constant" else min(x,point)*(1-max(x,point)))+p["left"]*(1-x)+p["right"]*x
    derivative=lambda x: strength/k*((1-2*x)/2 if p["sourceKind"]=="constant" else (1-point if x<point else -point))+p["right"]-p["left"]
    exact_nodes=np.array([exact(x) for x in nodes])
    close(r["values"],exact_nodes,"constant-k nodal identity")
    l2=energy=0
    for a,b,ua,ub in zip(nodes[:-1],nodes[1:],exact_nodes[:-1],exact_nodes[1:]):
        interpolant=lambda x: ua+(ub-ua)*(x-a)/(b-a)
        breakpoints=[point] if a<point<b else None
        l2+=quad(lambda x:(exact(x)-interpolant(x))**2,a,b,points=breakpoints,epsabs=1e-13)[0]
        energy+=quad(lambda x:k*(derivative(x)-(ub-ua)/(b-a))**2,a,b,points=breakpoints,epsabs=1e-13)[0]
    close(r["l2Error"]**2,l2,"piecewise field quadrature")
    close(r["energyError"]**2,energy,"piecewise gradient quadrature")
    K=np.array(r["stiffness"])
    close(K[1:-1]@np.array(r["values"]),r["load"][1:-1],"assembled interior load")

for case in packet["fixtures"]["rectangles"]:
    p,r=case["input"],case["result"]; a=p["xIntervals"]-1;b=p["yIntervals"]-1
    A=np.kron(np.eye(b),laplacian(a,r["hx"]))+np.kron(laplacian(b,r["hy"]),np.eye(a))
    assert r["nonzeros"]==np.count_nonzero(A)
    row=np.zeros(a*b)
    for neighbor in r["neighbors"]:
        if not neighbor["boundary"]: row[neighbor["index"]]+=neighbor["coefficient"]
    close(row,A[r["selectedIndex"]],"Kronecker stencil row")
for case in packet["fixtures"]["triangles"]:
    points=np.array(case["vertices"]); r=case["result"]
    coefficients=np.linalg.inv(np.c_[np.ones(3),points])
    gradients=coefficients[1:].T
    area=abs(np.linalg.det(np.c_[points[1]-points[0],points[2]-points[0]]))/2
    close(r["gradients"],gradients,"affine interpolation gradients")
    close(r["stiffness"],area*gradients@gradients.T,"triangle energy")
for r in packet["fixtures"]["coarse"]:
    A=laplacian(7,1/8); P=np.array(r["prolongation"]); e=np.array(r["smooth"][1:-1])
    expected=e-P@np.linalg.solve(P.T@A@P,P.T@A@e)
    close(r["final"][1:-1],expected,"coarse energy projection")
    close(P.T@A@expected,0,"projected residual")
    assert r["energyNorms"]["final"]<=r["energyNorms"]["smooth"]+1e-13
for case in packet["fixtures"]["godunov"]:
    left,right=case["left"],case["right"]
    candidates=[left*left/2,right*right/2]
    if left<=0<=right: candidates.append(0)
    expected=min(candidates) if left<=right else max(candidates)
    close(case["result"]["flux"],expected,"convex Godunov extremum")

namespaces={}
for key,example in packet["examples"].items():
    runfile=folder/"programs"/f"{key}-verified.py"
    runfile.write_text(example["code"]+"\n",encoding="utf8")
    run=subprocess.run([sys.executable,"-X","utf8","-I",str(runfile)],text=True,capture_output=True,encoding="utf8",check=True,timeout=120)
    assert run.stdout.rstrip()==example["expected"], (key,run.stdout)
    assert not run.stderr,(key,run.stderr)
    namespace={}
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(example["code"],str(runfile),"exec"),namespace)
    namespaces[key]=namespace
    counts["complete actual stdout"]=counts.get("complete actual stdout",0)+1

# Changed calls to the actual displayed helpers; answers come from different constructions.
for n in [3,7,12]:
    for length in [.5,1.5]:
        nodes,values,*_=namespaces["poisson"]["poisson"](n,lambda x:6,2,-1,length)
        target=[3*x*(length-x)+2-3*x/length for x in nodes]
        close(values,target,"changed displayed Poisson",atol=1e-12)
for n in [5,9]:
    for steps in [3,17]:
        for method in ["explicit","backward","crank"]:
            result=namespaces["diffusion"]["diffuse"](n,steps,.03,2,method)
            values,continuous,semidiscrete,ratio=result
            A=laplacian(n-1,1/n);dt=.03/steps;I=np.eye(n-1)
            op=I-dt*A if method=="explicit" else np.linalg.inv(I+dt*A) if method=="backward" else np.linalg.solve(I+dt*A/2,I-dt*A/2)
            close(values,np.linalg.matrix_power(op,steps)@np.sin(2*np.pi*np.arange(1,n)/n),"changed native diffusion")
for nodes in [[0,.2,.6,1],[0,.1,.3,.7,1]]:
    for point in [.3,.42]:
        K,load,values=namespaces["finiteElement"]["assemble_p1"](nodes,point=point,strength=2,left=1,right=-2,conductivity=3)
        target=[2/3*min(x,point)*(1-max(x,point))+1-3*x for x in nodes]
        close(values,target,"changed native point load")

for nx,ny in [(4,5),(7,4),(11,8)]:
    for modes in [(1,1),(2,2)]:
        Lx,Ly=1.5,.75
        matrix,rhs,values,target=namespaces["poisson2D"]["solve_rectangle"](nx,ny,Lx,Ly,*modes)
        hx,hy=Lx/nx,Ly/ny
        continuous=(modes[0]*math.pi/Lx)**2+(modes[1]*math.pi/Ly)**2
        discrete=4/hx**2*math.sin(modes[0]*math.pi/(2*nx))**2+4/hy**2*math.sin(modes[1]*math.pi/(2*ny))**2
        lift=np.array([.4*i*hx-.2*j*hy for j in range(1,ny) for i in range(1,nx)])
        close(values,lift+(target-lift)*continuous/discrete,"changed native rectangular eigenmode")
for cells in [3,7,11]:
    values,mismatch=namespaces["neumann"]["conservative_neumann"](cells,3,1,2)
    h=F(1,cells)
    faces=[-F(1),*[(a-b)/h for a,b in zip(values,values[1:])],F(2)]
    assert sum(values)==0
    assert all(faces[j+1]-faces[j]==3*h for j in range(cells))
    incompatible,gap=namespaces["neumann"]["conservative_neumann"](cells,3,1,1)
    assert incompatible is None and gap==1
    counts["changed native exact flux balance"]=counts.get("changed native exact flux balance",0)+1
for intervals,length,scale,tolerance,budget in [(9,.75,1.5,F(1,20),73),(16,1.5,2.,F(1,100),350),(32,.5,.5,F(1,1000),900)]:
    values,used,parts,certified=namespaces["certificate"]["jacobi_report"](intervals,tolerance,budget,length,scale,1.,-2.)
    residual,nodal,interpolation,algebraic=parts
    assert used<=budget
    h=F(length)/intervals
    exact_residual=max(abs(12*F(scale)*F(j,intervals)*(1-F(j,intervals))-(2*F(values[j])-F(values[j-1])-F(values[j+1]))/h**2) for j in range(1,intervals))
    assert residual==exact_residual and certified==(sum(parts[1:])<=tolerance)
    assert nodal+interpolation==F(5,8)*F(scale)*h*h
    counts["changed finite-budget native certificate"]=counts.get("changed finite-budget native certificate",0)+1

# Independently compute the changed hand tasks, including the repaired Robin constant.
uA=lambda x:3*x*(1-x)+2-3*x
values=list(map(uA,[F(0),F(1,4),F(1,2),F(3,4),F(1)]))
assert values[1:-1]==[F(29,16),F(5,4),F(5,16)]
assert 2*values[1]-values[2]==F(19,8) and -values[2]+2*values[3]==-F(5,8)
assert F(1,100000)*256/8==F(32,100000)
close(np.linalg.eigvalsh([[-.2,.6],[.6,-.2]]),[-.8,.4],"changed positivity counterexample")
assert -7==1-8 and F(1,9)==1/(1+F(8)) and -F(3,5)==(1-F(8)/2)/(1+F(8)/2)
assert F(2,5)*F(1,4)*F(3,4)==F(3,40)
assert F(5,8)*2*(F(3,2)/64)**2==F(45,65536)
assert F(8)/F(3,2)**2*(F(1,1000)-F(45,65536))==F(2567,2304000)
assert 3*(F(20,3)-1-5)==2  # right Robin flux; left derivative is zero.
counts["changed exact hand-task groups"]=9

record={"checkedAt":datetime.now(timezone.utc).isoformat(),"passed":True,"counts":counts,"versions":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__},"sourceHashes":packet["sourceHashes"]}
(folder/"native-results.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf8")
print(json.dumps(record,indent=2))
