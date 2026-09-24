"""Execute standalone teaching programs once and save their actual outputs."""
import hashlib
import json
import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / 'scratch' / 'knn-native'
WORK.mkdir(parents=True, exist_ok=True)
programs = []


def example(identifier, title, question, code):
    programs.append(dict(id=identifier, title=title, question=question,
                         code=textwrap.dedent(code).strip()))


example('neighbor-ledger', 'Reconstruct the eight distances and two votes',
        'Which class wins five uniform votes, and why can distance weighting reverse it?', '''
import numpy as np
ids = np.array(['A1','A2','A3','B1','B2','B3','C1','C2'])
X = np.array([[1,2],[1.5,1.8],[1.2,2.5],[5,5],[5.5,4.8],
              [4.8,5.2],[3,3],[3.2,2.8]])
labels = np.array(['A','A','A','B','B','B','C','C'])
query = np.array([3.1,2.9])
distances = np.linalg.norm(X-query, axis=1)
# A declared 12-place distance key resolves decimal-fixture numerical ties.
order = np.lexsort((ids, np.round(distances, 12)))
for index in order:
    print(ids[index], f'{distances[index]:.6f}')
selected = order[:5]
for mode in ['uniform', 'distance']:
    weights = np.ones(5) if mode == 'uniform' else 1/distances[selected]
    weights /= weights.sum()
    probabilities = [weights[labels[selected] == c].sum() for c in 'ABC']
    print(mode, np.round(probabilities, 6), 'ABC'[int(np.argmax(probabilities))])
''')

example('units-cosine', 'Calculate training scales and a direction comparison',
        'Does standardizing change the closest row here? Why does cosine choose a different vector from Euclidean distance?', '''
import numpy as np
from sklearn.preprocessing import StandardScaler
X = np.array([[1,100],[3,1000],[5,1100]], dtype=float)
query = np.array([[1.2,600]])
scaler = StandardScaler().fit(X)
print('Training scales:', np.round(scaler.scale_, 6))
for name, rows, q in [('raw', X, query),
                       ('scaled', scaler.transform(X), scaler.transform(query))]:
    contributions = (rows-q)**2
    distances = np.sqrt(contributions.sum(axis=1))
    print(name, np.round(distances, 6), 'nearest', 'RST'[np.argmin(distances)])
q = np.array([1.,0.])
vectors = np.array([[3.,0.],[1.,1.]])
unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cosine_distance = 1-unit_vectors@q
normalized_squared = ((unit_vectors-q)**2).sum(axis=1)
assert np.allclose(normalized_squared, 2*cosine_distance)
print('Euclidean U,V:', np.linalg.norm(vectors-q, axis=1))
print('Cosine U,V:', np.round(cosine_distance, 6))
''')

ESTIMATOR = '''
import numpy as np

class LocalNeighbors:
    """Dense Euclidean teaching estimator; full stable sort per query."""
    def __init__(self, k=3, weights='uniform', task='classification'):
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError('k must be a positive integer')
        if weights not in ('uniform', 'distance'):
            raise ValueError('unsupported weights')
        if task not in ('classification', 'regression'):
            raise ValueError('unsupported task')
        self.k, self.weights, self.task = k, weights, task

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError('X must be a nonempty row/feature matrix')
        if not np.isfinite(X).all() or y.ndim != 1 or len(y) != len(X):
            raise ValueError('finite X and one target per row are required')
        if self.k > len(X):
            raise ValueError('k exceeds training rows')
        if self.task == 'regression':
            y = y.astype(float)
            if not np.isfinite(y).all():
                raise ValueError('regression targets must be finite')
        else:
            # One consistently comparable scalar label type is required.
            if y.dtype.kind in 'biufc' and not np.isfinite(y).all():
                raise ValueError('classification labels cannot be nonfinite')
            self.classes_, y = np.unique(y, return_inverse=True)
        self.X_, self.y_ = X.copy(), y.copy()
        return self

    def _query(self, X):
        if not hasattr(self, 'X_'):
            raise ValueError('fit before querying')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1] or not np.isfinite(X).all():
            raise ValueError('query shape/finite values do not match training')
        for row in X:
            # Hypot scales its arithmetic instead of squaring tiny/huge differences.
            with np.errstate(over='ignore'):
                distances = np.hypot.reduce(self.X_-row, axis=1)
            if not np.isfinite(distances).all():
                raise ValueError('distance overflow; rescale the coordinates')
            # Stable sorting gives earlier training-row IDs equal-distance priority.
            selected = np.argsort(distances, kind='stable')[:self.k]
            chosen = distances[selected]
            if self.weights == 'uniform':
                weights = np.ones(self.k)
            elif np.any(chosen == 0):
                weights = (chosen == 0).astype(float)
            else:
                # Same normalized inverse weights without reciprocal overflow.
                weights = chosen.min()/chosen
            yield selected, weights/weights.sum()

    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError('probabilities require classification')
        return np.array([np.bincount(self.y_[rows], weights=w,
                                     minlength=len(self.classes_))
                         for rows, w in self._query(X)]).reshape((-1,len(self.classes_)))

    def predict(self, X):
        if self.task == 'classification':
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return np.array([w@self.y_[rows] for rows, w in self._query(X)])
'''

example('scratch-estimator', 'A complete classifier and regressor from distances',
        'At a duplicate input with opposite labels, how should inverse-distance weights behave?', ESTIMATOR + '''
X = np.array([[0.],[0.],[1.],[2.]])
labels = np.array(['A','B','B','B'])
model = LocalNeighbors(k=3, weights='distance').fit(X, labels)
print('Class order:', model.classes_)
print('Duplicate-input probabilities:', model.predict_proba([[0.]]))
print('Prediction:', model.predict([[0.]]))
x = np.arange(6.).reshape(-1,1)
for mode in ['uniform','distance']:
    regressor = LocalNeighbors(k=3, weights=mode, task='regression').fit(x, x[:,0]**2)
    print(mode, np.round(regressor.predict([[2.5],[8.]]), 6))
model.fit(np.array([[0.],[2.],[4.]]), np.array(['C','D','D']))
print('Refit replaces old classes:', model.classes_, model.predict([[3.]]))
''')

WORKFLOW = '''
import numpy as np
from sklearn.datasets import make_moons
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def report(seed, noise):
    X, y = make_moons(n_samples=600, noise=noise, random_state=seed)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=.4, stratify=y, random_state=seed+1)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_hold, y_hold, test_size=.5, stratify=y_hold, random_state=seed+2)
    candidates = []
    for k in [1,5,15,41]:
        for weights in ['uniform','distance']:
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(
                n_neighbors=k, weights=weights, algorithm='brute'))
            model.fit(X_train, y_train)
            loss = log_loss(y_valid, model.predict_proba(X_valid))
            candidates.append((loss,k,weights,model))
            print('validation', k, weights, f'{loss:.6f}')
    loss, k, weights, model = min(candidates, key=lambda row: row[:3])
    baseline = DummyClassifier(strategy='prior').fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    print('Rows train/validation/test:', len(y_train),len(y_valid),len(y_test))
    print('Selected:', k, weights)
    print('Test accuracy:', f'{accuracy_score(y_test,predictions):.6f}')
    print('Test log loss:', f'{log_loss(y_test,probabilities):.6f}')
    print('Baseline test log loss:', f'{log_loss(y_test,baseline.predict_proba(X_test)):.6f}')
    print('Test confusion matrix:')
    print(confusion_matrix(y_test,predictions))
    return candidates
'''
example('held-out-workflow', 'Select a complete pipeline and reserve a test report',
        'Can one-neighbor training fit coexist with poor probability quality on new rows?', WORKFLOW + '\nreport(seed=19, noise=.28)')

KD_CODE = '''
import math
import numpy as np

def build_tree(X, ids=None, depth=0):
    if ids is None:
        X = np.asarray(X,float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError('nonempty finite coordinate matrix required')
        ids = list(range(len(X)))
    if not ids:
        return None
    axis = depth % X.shape[1]
    ordered = sorted(ids, key=lambda i: (X[i,axis],i))
    middle = len(ordered)//2
    return (ordered[middle],axis,
            build_tree(X,ordered[:middle],depth+1),
            build_tree(X,ordered[middle+1:],depth+1))

def nearest(X, tree, query):
    X, query = np.asarray(X,float), np.asarray(query,float)
    if X.ndim != 2 or len(X)==0 or query.shape!=(X.shape[1],):
        raise ValueError('nonempty matching shapes required')
    if not np.isfinite(X).all() or not np.isfinite(query).all():
        raise ValueError('finite coordinates required')
    best = (float('inf'),len(X))
    visits = 0
    def search(node):
        nonlocal best, visits
        if node is None:
            return
        index,axis,left,right = node
        with np.errstate(over='ignore'):
            distance = float(np.hypot.reduce(X[index]-query))
        if not np.isfinite(distance):
            raise ValueError('distance overflow; rescale')
        visits += 1
        best = min(best,(distance,index))
        delta = query[axis]-X[index,axis]
        near,far = (left,right) if delta<=0 else (right,left)
        search(near)
        # Equality still matters to the smallest-row-ID tie convention.
        if abs(delta)<=best[0]:
            search(far)
    search(tree)
    return best, visits
'''
example('kd-tree', 'Build and query an exact median KD-tree',
        'Why does the far-branch condition use ≤ instead of strictly less?', KD_CODE + '''
X = np.array([[1,2],[1.5,1.8],[1.2,2.5],[5,5],[5.5,4.8],
              [4.8,5.2],[3,3],[3.2,2.8]])
tree = build_tree(X)
for query in [[1.3,2.1],[5.1,4.9],[3.1,2.9]]:
    (distance,index),visits = nearest(X,tree,np.array(query))
    brute = min((math.hypot(*(row-query)),i) for i,row in enumerate(X))
    assert index == brute[1] and math.isclose(distance,brute[0],rel_tol=1e-14)
    print(query, 'row', index, 'distance', f'{distance:.6f}', 'visited', visits)
# A query on the root plane must still inspect the other side for tied IDs.
tied = np.array([[-1.,0.],[1.,0.],[0.,5.]])
print('Tie case:', nearest(tied,build_tree(tied),np.array([0.,0.]))[0])
''')

example('library-search', 'Use supported metric/index contracts',
        'What does cosine retrieve here, and what does querying an index without an explicit query change?', '''
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
vectors = np.array([[3.,0.],[1.,1.],[0.,2.]])
search = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute').fit(vectors)
distances, ids = search.kneighbors([[1.,0.]])
print('Cosine IDs:', ids)
print('Cosine distances:', np.round(distances,6))
print('BallTree supports cosine:', 'cosine' in BallTree.valid_metrics)
line = np.array([[0.],[1.],[4.]])
index = NearestNeighbors(n_neighbors=1).fit(line)
print('Explicit training queries include self:', index.kneighbors(line,return_distance=False).ravel())
print('No-query form excludes self:', index.kneighbors(return_distance=False).ravel())
''')

example('candidate-recall', 'Measure recall separately from a weighted decision',
        'Can removing far cases preserve exact top-three retrieval while removing close cases changes the class?', '''
import numpy as np
X = np.array([[1,2],[1.5,1.8],[1.2,2.5],[5,5],[5.5,4.8],
              [4.8,5.2],[3,3],[3.2,2.8]])
labels = np.array(list('AAABBBCC'))
distances = np.linalg.norm(X-[3.1,2.9],axis=1)
def retrieve(pool):
    selected = sorted(pool,key=lambda i:(round(distances[i],12),i))[:3]
    weights = 1/distances[selected]
    mass = np.array([weights[labels[selected]==c].sum() for c in 'ABC'])
    return selected, 'ABC'[mass.argmax()]
exact, _ = retrieve(range(8))
for name,pool in [('all',range(8)),('drop close C',range(6)),
                  ('drop far B',[0,1,2,6,7])]:
    selected,prediction = retrieve(pool)
    recall = len(set(exact)&set(selected))/3
    print(name, selected, 'recall', f'{recall:.6f}', 'class', prediction)
''')

example('dimension', 'Separate a volume identity from a noisy-feature experiment',
        'Why is explaining99% of input variance not a guarantee of preserving a useful label signal?', '''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
rng = np.random.default_rng(7)
for d in [1,2,10,100]:
    side = .01**(1/d)
    distances = np.linalg.norm(rng.normal(size=(4000,d)),axis=1)
    print('dimension',d,'1% side',f'{side:.6f}',
          'Gaussian radius CV',f'{distances.std()/distances.mean():.6f}')
# Deterministic balanced Cartesian data: high-variance noise and tiny label signal.
noise = np.arange(-10,11,dtype=float)
X = np.array([[z,s] for z in noise for s in [-.01,.01]])
y = (X[:,1]>0).astype(int)
pca = PCA(n_components=1).fit(X)
print('Kept variance fraction:',f'{pca.explained_variance_ratio_[0]:.8f}')
print('First component:',pca.components_)
projected = pca.transform(X)
print('Opposite-label rows at same z collapse:',np.allclose(projected[::2],projected[1::2]))
# This is a representation counterexample, not a held-out classifier benchmark.
''')

example('posterior-risk', 'Compute Bayes and independent-neighbor disagreement',
        'If P(Y=1|x)=.8, why can an infinitely close single neighbor still disagree?', '''
import numpy as np
eta = np.array([.1,.8])
mass = np.array([.5,.5])
local_bayes = np.minimum(eta,1-eta)
local_neighbor = 2*eta*(1-eta)
bayes = mass@local_bayes
neighbor = mass@local_neighbor
print('Local Bayes errors:',np.round(local_bayes,6))
print('Local independent-neighbor errors:',np.round(local_neighbor,6))
print('Integrated Bayes:',f'{bayes:.6f}')
print('Integrated neighbor:',f'{neighbor:.6f}')
print('Binary upper bound:',f'{2*bayes*(1-bayes):.6f}')
print('Three independent labels at eta=.8 majority error:',f'{3*.2**2*.8+.2**3:.6f}')
''')

example('haversine-outputs', 'Find a nearby station across the date line',
        'Which station is close to longitude179.8°, and why are raw degree differences misleading?', '''
import numpy as np
from sklearn.neighbors import BallTree, KNeighborsRegressor
# Coordinates are latitude, longitude, converted from degrees to radians.
stations = np.array([[0.,-179.9],[0.,179.],[1.,179.8]])
query = np.array([[0.,179.8]])
tree = BallTree(np.radians(stations),metric='haversine')
angle,ids = tree.query(np.radians(query),k=3)
print('Nearest station IDs:',ids)
print('Spherical distances km:',np.round(angle*6371.,3))
# A separate synthetic analog task: input temperature, outputs pressure and load.
X = np.array([[10.],[20.],[30.]])
Y = np.array([[100.,2.],[110.,4.],[130.,8.]])
model = KNeighborsRegressor(n_neighbors=2).fit(X,Y)
print('Joint output shape:',model.predict([[24.]]).shape)
print('Mean of the same two neighbors:',model.predict([[24.]]))
''')

example('changed-capstone', 'One complete response to the changed held-out task',
        'With a new seed and stronger noise, which validation choice survives, and what does its test report establish?', WORKFLOW + '\nreport(seed=37, noise=.4)')

runs = []
previous_path = WORK/'example-runs.json'
previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
cache = {row['id']: row for row in previous.get('runs', [])} if previous.get('python') == sys.version else {}
for program in programs:
    code_hash = hashlib.sha256(program['code'].encode()).hexdigest()
    old = cache.get(program['id'])
    if old and old['codeSha256'] == code_hash and old['returncode'] == 0:
        program['expected'] = old['expected']
        runs.append(old)
        continue
    path = WORK / (program['id'] + '.py')
    path.write_text(program['code']+'\n', encoding='utf-8')
    result = subprocess.run([sys.executable,str(path)],capture_output=True,text=True,
                            encoding='utf-8',timeout=180,cwd=WORK)
    if result.returncode:
        raise RuntimeError(program['id']+'\n'+result.stderr)
    program['expected'] = result.stdout.strip()
    runs.append(dict(id=program['id'],returncode=result.returncode,
                     codeSha256=hashlib.sha256(program['code'].encode()).hexdigest(),
                     outputSha256=hashlib.sha256(program['expected'].encode()).hexdigest(),
                     stderr=result.stderr,expected=program['expected']))
destination = ROOT/'src/learn/data/knn-examples.js'
destination.write_text('export const knnExamples = '+json.dumps(programs,ensure_ascii=False,indent=2)+';\n',encoding='utf-8')
validation_output = next(program['expected'] for program in programs if program['id']=='held-out-workflow')
validation_rows = [dict(k=int(parts[1]),weights=parts[2],loss=float(parts[3]))
                   for line in validation_output.splitlines()
                   if (parts:=line.split())[0]=='validation']
(ROOT/'src/learn/data/knn-validation-data.js').write_text(
    '// Recorded seed19/noise.28 validation log loss, rounded as in the executed program.\n'
    + 'export const knnValidationRows = '+json.dumps(validation_rows,indent=2)+';\n',encoding='utf-8')
(WORK/'example-runs.json').write_text(json.dumps(dict(python=sys.version,runs=runs),indent=2)+'\n',encoding='utf-8')
print(f'{len(programs)} complete KNN programs have actual outputs; unchanged code/runtime results reused.')
