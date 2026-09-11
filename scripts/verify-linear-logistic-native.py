"""Independent matrix/probability/SciPy oracles and fresh standalone learner runs."""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import scipy
import sklearn
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import t

folder = pathlib.Path(sys.argv[1])
data = json.loads((folder / 'browser-model-values.json').read_text())
def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)

X = np.column_stack([np.ones(4), np.arange(4.)])
for row in data['residuals']:
    y = np.array([1., 2., 2., row['last']])
    weights = np.array([row['intercept'], row['slope']])
    errors = X @ weights - y
    close(row['mse'], errors @ errors / 4)
    close(row['gradient'], X.T @ errors / 2)
    close([row['bestIntercept'], row['bestSlope']], np.linalg.lstsq(X, y, rcond=None)[0])
    close([r['residual'] for r in row['rows']], -errors)
H = X.T @ X / 2
optimum = np.linalg.solve(X.T @ X, X.T @ np.array([1., 2., 2., 4.]))
for trace in data['gradients']:
    transition = np.eye(2) - trace['rate'] * H
    for state in trace['states']:
        expected = optimum - np.linalg.matrix_power(transition, state['step']) @ optimum
        close([state['intercept'], state['slope']], expected)
for row in data['scores']:
    close(row['probability'], expit(row['score']))
    close(row['loss'], np.logaddexp(0, (1 - 2 * row['label']) * row['score']))
probabilities = np.array([.1, .2, .35, .45, .65, .8])
truth = np.array([0, 1, 0, 1, 0, 1])
for row in data['thresholds']:
    predicted = probabilities >= row['threshold']
    for key, mask in [('tp',predicted & (truth == 1)),('fp',predicted & (truth == 0)),('tn',~predicted & (truth == 0)),('fn',~predicted & (truth == 1))]:
        assert row[key] == int(mask.sum())
    close(row['cost'], np.sum(predicted & (truth == 0)) + row['costInput'] * np.sum(~predicted & (truth == 1)))
for row in data['separation']:
    close(row['objective'], np.logaddexp(0, -row['weight']) + row['penalty'] * row['weight'] ** 2 / 2)
    if row['penalty']:
        close(row['optimum'], brentq(lambda w: expit(w)-1+row['penalty']*w, 0, 32))
A = np.column_stack([np.ones(6), np.arange(6.)])
y = np.array([1., 1.7, 3.2, 3.7, 5.3, 5.8])
weights = np.linalg.lstsq(A,y,rcond=None)[0]
variance = np.sum((y-A@weights)**2)/4
covariance = variance * np.linalg.inv(A.T@A)
for row in data['uncertainty']:
    query = np.array([1.,row['input']])
    mean_variance = query @ covariance @ query
    close(row['predicted'], query @ weights)
    close(row['meanHalfWidth'], t.ppf(.975,4)*np.sqrt(mean_variance))
    close(row['individualHalfWidth'], t.ppf(.975,4)*np.sqrt(mean_variance+variance))

runs=[]
for example in data['examples']:
    source=folder/(example['id']+'.py')
    source.write_text(example['code'],encoding='utf-8')
    result=subprocess.run([sys.executable,str(source)],capture_output=True,text=True,encoding='utf-8',timeout=90)
    assert result.returncode==0,(example['id'],result.stderr)
    assert not result.stderr,(example['id'],result.stderr)
    assert result.stdout.strip()==example['expected'].strip(),(example['id'],result.stdout,example['expected'])
    runs.append({'id':example['id'],'codeSha256':hashlib.sha256(example['code'].encode()).hexdigest(),'stdout':result.stdout,'exitCode':result.returncode})
record={'checkedAt':datetime.now(timezone.utc).isoformat(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'checks':{key:len(value) for key,value in data.items() if key!='examples'},'oracles':'NumPy least squares, independent Hessian matrix powers, logaddexp/expit, vectorized confusion counts, Brent roots, matrix covariance and Student t quantiles','runs':runs}
pathlib.Path('docs/teaching/evidence/linear-logistic-native-review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(f"Passed independent numerical families and {len(runs)} fresh standalone learner programs.")
