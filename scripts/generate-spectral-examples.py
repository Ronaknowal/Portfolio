"""Authoring generator: format new standalone programs and capture real stdout."""
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import black

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'scratch/spectral-authoring/original-lesson.jsx').open(encoding='utf-8', newline='') as original_file:
    old = original_file.read()
original = re.search(r'<CodeBlock language="python">\{`(.*?)`\}</CodeBlock>', old, re.S).group(1)
examples = []

def add(key, title, question, code, preserve=False):
    code = code.strip('\n')
    if not preserve:
        formatted = black.format_str(code, mode=black.Mode(line_length=78)).rstrip()
        assert ast.dump(ast.parse(formatted)) == ast.dump(ast.parse(code))
        code = formatted
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, key, 'exec'), {'__name__': '__main__'})
    examples.append(dict(id=key, title=title, question=question, code=code, expected=stream.getvalue().rstrip()))

base = '''import numpy as np

def bridge_graph(weight=0.2):
    A = np.zeros((6, 6))
    for i, j in [(0,1), (0,2), (1,2), (3,4), (3,5), (4,5)]:
        A[i,j] = A[j,i] = 1.0
    A[2,3] = A[3,2] = weight
    return A

A = bridge_graph()
d = A.sum(axis=1)
L = np.diag(d) - A
'''

add('original', 'Reproduce the six-document spectrum', 'Predict which nodes should share a sign before running this preserved pilot example.', original, True)
add('energy', 'Verify edge energy on a changed weighted path', 'Which edges contribute zero even though they have positive weight?', '''import numpy as np

edges = [(0,1,2.0), (1,2,0.5), (2,3,3.0)]
x = np.array([2., 2., -1., 1.])
A = np.zeros((4,4))
for i,j,w in edges:
    A[i,j] = A[j,i] = w
L = np.diag(A.sum(axis=1)) - A
terms = [float(w*(x[i]-x[j])**2) for i,j,w in edges]
print("edge contributions:", terms)
print("matrix energy:", float(x @ L @ x))
print("squared norm:", float(x @ x))
print("Rayleigh quotient:", round(float(x @ L @ x / (x @ x)), 6))
print("constant annihilated:", bool(np.allclose(L @ np.ones(4), 0)))
''')
add('sweep', 'Sweep normalized coordinates and check the tiny graph exhaustively', 'Does the smallest sweep conductance equal the global optimum on this fixture? Enumeration here is a check, not a scalable clustering algorithm.', base + '''
def measures(A, selected):
    n = len(A)
    S = sorted(set(selected))
    if not S or len(S) == n:
        raise ValueError("Use a nonempty proper subset")
    T = sorted(set(range(n)) - set(S))
    degree = A.sum(axis=1)
    cut = float(A[np.ix_(S,T)].sum())
    a, b = float(degree[S].sum()), float(degree[T].sum())
    if min(a,b) <= 0:
        raise ValueError("Both volumes must be positive")
    return cut, cut/min(a,b), cut*(1/a+1/b)

inv = 1/np.sqrt(d)
N = inv[:,None] * L * inv[None,:]
values, U = np.linalg.eigh(N)
v = U[:,1]
if v[0] < 0:
    v = -v
y = v/np.sqrt(d)
order = np.argsort(y, kind="stable")
candidates = []
for count in range(1,len(A)):
    if abs(y[order[count]]-y[order[count-1]]) <= 1e-9:
        continue
    S = order[:count].tolist()
    candidates.append((measures(A,S)[1], S))
best_phi, best = min(candidates, key=lambda item:item[0])
all_phi = [measures(A,[i for i in range(6) if mask & (1<<i)])[1] for mask in range(1,63)]
print("sweep side:", "".join("ABCDEF"[i] for i in sorted(best)))
print("sweep conductance:", round(best_phi,6))
print("global tiny-graph conductance:", round(min(all_phi),6))
print("normalized lambda2:", round(float(values[1]),6))
print("Cheeger lower/upper:", round(float(values[1]/2),6), round(float(np.sqrt(2*values[1])),6))
print("triangle Ncut:", round(measures(A,[0,1,2])[2],6))
''')
add('isolates', 'Keep an isolated vertex separate from the identity convention', 'Which diagonal entry does the blind identity formula get wrong?', '''import numpy as np

A = np.array([[0.,2.,0.],[2.,0.,0.],[0.,0.,0.]])
d = A.sum(axis=1)
L = np.diag(d)-A
inv = np.zeros_like(d)
np.divide(1.,np.sqrt(d),out=inv,where=d>0)
N = inv[:,None]*L*inv[None,:]
blind = np.eye(3)-inv[:,None]*A*inv[None,:]
print("product diagonal:", np.round(np.diag(N),6).tolist())
print("blind identity diagonal:", np.diag(blind).tolist())
print("product eigenvalues:", np.round(np.linalg.eigvalsh(N),6).tolist())
print("isolated basis is null:", bool(np.allclose(N @ [0,0,1], 0)))
''')
add('clusters', 'Complete NumPy spectral clustering with explicit Lloyd restarts', 'Watch the distinction between eigenvector columns and node rows. Why does this program reject an empty cluster run?', '''import numpy as np

def spectral_rows(A, k):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all() or (A<0).any() or not np.allclose(A,A.T) or np.any(np.diag(A)!=0):
        raise ValueError("Use a finite symmetric nonnegative adjacency with zero diagonal")
    if type(k) is not int or not 1 <= k <= len(A):
        raise ValueError("k must be an integer from 1 to n")
    d = A.sum(axis=1)
    if np.any(d<=0):
        raise ValueError("Handle isolated vertices before this algorithm")
    inv = 1/np.sqrt(d)
    N = np.eye(len(A))-inv[:,None]*A*inv[None,:]
    values,U = np.linalg.eigh(N)
    rows = U[:,:k]
    lengths = np.linalg.norm(rows,axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Numerically zero embedding row: reconsider k/components")
    return values, rows/lengths[:,None]

def lloyd(points, k, seeds, max_steps=100):
    centers = points[np.array(seeds)].copy()
    previous = None
    for step in range(max_steps):
        distances = ((points[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        labels = distances.argmin(axis=1)
        if any(not np.any(labels==j) for j in range(k)):
            return None
        if previous is not None and np.array_equal(labels,previous):
            loss = float(((points-centers[labels])**2).sum())
            return labels, centers, loss
        centers = np.array([points[labels==j].mean(axis=0) for j in range(k)])
        previous = labels.copy()
    return None  # bounded runtime; unconverged restarts are not silently accepted

A = np.zeros((9,9))
for s in [0,3,6]:
    for i,j in [(s,s+1),(s,s+2),(s+1,s+2)]:
        A[i,j]=A[j,i]=1.
for i,j in [(2,3),(5,6)]:
    A[i,j]=A[j,i]=0.08
values,points = spectral_rows(A,3)
rng = np.random.default_rng(12)
seeds = [[0,3,6]]+[rng.choice(9,3,replace=False).tolist() for _ in range(19)]
runs = [result for seed in seeds if (result:=lloyd(points,3,seed)) is not None]
if not runs:
    raise RuntimeError("No complete converged run; revise initialization or graph")
labels,centers,loss = min(runs,key=lambda result:result[2])
groups = sorted(["".join("ABCDEFGHI"[i] for i in range(9) if labels[i]==j) for j in range(3)])
print("first four eigenvalues:", np.round(values[:4],6).tolist())
print("row lengths:", np.round(np.linalg.norm(points,axis=1),6).tolist())
print("groups:", groups)
print("best occupied-run loss:", round(loss,6))
''')
add('filters', 'Compute three filters and verify their different objectives', 'At the same numerical parameter, should heat and ridge produce exactly the same answer?', base + '''
x = np.array([1.6,0.4,1.,-1.6,-0.4,-1.])
values,U = np.linalg.eigh(L)
c = U.T @ x
heat = U @ (np.exp(-values)*c)
ridge = np.linalg.solve(np.eye(6)+L,x)
projected = U[:,:2] @ (U[:,:2].T @ x)
print("heat:", np.round(heat,4).tolist())
print("ridge:", np.round(ridge,4).tolist())
print("first-two projection:", np.round(projected,4).tolist())
print("input energy:", round(float(x@L@x),6))
print("heat energy:", round(float(heat@L@heat),6))
print("discarded projection energy:", round(float(c[2:]@c[2:]),6))
print("projection squared change:", round(float(np.sum((x-projected)**2)),6))
print("ridge normal equation:", bool(np.allclose((np.eye(6)+L)@ridge,x)))
''')
add('walk', 'Expose the periodic mode and remove it with laziness', 'The graph is connected in both cases. Which operator eigenvalue explains the different limit?', '''import numpy as np

P = np.array([[0.,1.],[1.,0.]])
lazy = (np.eye(2)+P)/2
for name, transition in [("ordinary",P),("lazy",lazy)]:
    mass = np.array([1.,0.])  # row probability vector
    path = [mass.tolist()]
    for _ in range(4):
        mass = mass @ transition
        path.append(mass.tolist())
    print(name, path)
    print("transition eigenvalues:", np.linalg.eigvalsh(transition).tolist())
''')
add('resistance', 'Check spectral resistance against a grounded circuit solve', 'Why do the two endpoint choices below have very different resistance, even though both use edges in the same graph?', base + '''
def effective_resistance(L, a, b):
    n = len(L)
    if a==b:
        return 0.
    values,U = np.linalg.eigh(L)
    if np.count_nonzero(values < 1e-10) != 1:
        raise ValueError("This helper requires one connected component at its numerical tolerance")
    return float(np.sum((U[a,1:]-U[b,1:])**2/values[1:]))

def grounded_resistance(L,a,b):
    keep = [i for i in range(len(L)) if i!=b]
    injection = np.zeros(len(L)); injection[a]=1.; injection[b]=-1.
    potentials = np.zeros(len(L))
    potentials[keep] = np.linalg.solve(L[np.ix_(keep,keep)],injection[keep])
    return float(potentials[a]-potentials[b])

for a,b in [(2,3),(0,1),(0,5)]:
    spectral = effective_resistance(L,a,b)
    grounded = grounded_resistance(L,a,b)
    print("ABCDEF"[a]+"ABCDEF"[b], round(spectral,6), round(grounded,6))
''')
add('rotation', 'Rotate a repeated eigenspace without changing a scalar filter', 'Why can one retained direction change while the whole repeated block remains the same operator?', '''import numpy as np

# Unit-weight triangle: eigenvalues 0,3,3.
L = np.array([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]])
values,U = np.linalg.eigh(L)
B = U[:,1:]
theta = np.pi/4
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
rotated = B @ R
x = np.array([1.,0.,0.])
one = B[:,0]*float(B[:,0]@x)
other = rotated[:,0]*float(rotated[:,0]@x)
print("same repeated-block projector:", bool(np.allclose(B@B.T,rotated@rotated.T)))
print("one-direction outputs equal:", bool(np.allclose(one,other)))
heat = np.exp(-3)*B@B.T
heat_rotated = np.exp(-3)*rotated@rotated.T
print("same repeated-block heat operator:", bool(np.allclose(heat,heat_rotated)))
print("rotated eigenpair residual:", bool(np.allclose(L@rotated,3*rotated)))
''')

destination = ROOT / 'src/learn/data/spectral-graph-examples.js'
destination.write_text('// Complete standalone Python programs; outputs captured by generate-spectral-examples.py.\nexport const spectralGraphExamples = ' + json.dumps(examples,indent=2,ensure_ascii=False) + ';\n',encoding='utf-8')
print(json.dumps({'programs':len(examples),'outputs':{item['id']:item['expected'] for item in examples}},indent=2))
