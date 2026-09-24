import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {pairedSaving, laplacianRowExample} from '../src/learn/data/visual-review-models.js';
import {translationModel} from '../src/learn/data/os-foundations-model.js';

const python = process.env.LESSON_PYTHON || 'python';
const oracle = spawnSync(python, ['-c', `
import json, math, statistics
import numpy as np
# Independent t(4) CDF inversion, rather than copying the UI critical value.
def cdf(t):
    u = t / math.sqrt(t*t + 4)
    return 0.5 + 0.75*u - 0.25*u**3
lo, hi = 0., 10.
for _ in range(60):
    mid = (lo+hi)/2
    if cdf(mid) < 0.975: lo = mid
    else: hi = mid
critical = (lo+hi)/2
intervals = []
for shift in [0, 1, 2]:
    d = [v+shift for v in [2,4,-1,3,2]]
    mean, se = statistics.mean(d), statistics.stdev(d)/math.sqrt(len(d))
    intervals.append([mean, se, mean-critical*se, mean+critical*se])
A = np.zeros((6,6))
for i,j in [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]: A[i,j] = A[j,i] = 1
A[2,3] = A[3,2] = .2
L = np.diag(A.sum(axis=1))-A
rows = [float((L @ np.array(signal))[2]) for signal in [[1]*6, [1,1,1,-1,-1,-1]]]
print(json.dumps(dict(intervals=intervals, rows=rows)))
`], {encoding: 'utf8'});
assert.equal(oracle.status, 0, oracle.stderr);
const expected = JSON.parse(oracle.stdout);
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`);
for (let shift = 0; shift <= 2; shift++) {
  const result = pairedSaving(shift);
  [result.mean, result.se, result.low, result.high].forEach((value, i) => near(value, expected.intervals[shift][i]));
}
[false, true].forEach((split, i) => near(laplacianRowExample(split).total, expected.rows[i]));
for (const process of ['A', 'B']) for (let address = 0; address < 64; address++) {
  const result = translationModel(process, address, 'read');
  if (address >= 48) assert.equal(result.physical, null);
  else {
    const frame = address < 16 ? 0 : address < 32 ? (process === 'A' ? 3 : 5) : 6;
    assert.equal(result.physical, frame * 16 + address % 16);
  }
}
console.log('PASS: three t intervals against independent CDF inversion; two graph rows against NumPy matrix multiplication; all 128 process/address read mappings.');
