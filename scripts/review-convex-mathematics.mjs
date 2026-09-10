import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { allocationCertificateState, ridgeContourPoints, ridgeCurvatureState, softThresholdState } from '../src/learn/data/convex-optimization-models.js';

const directory = resolve('scratch/convex-cross-review');
mkdirSync(directory, { recursive: true });
const ridge = [0, 1e-16, 1e-13, 1e-12, 1e-10, 1e-8, 0.1].map(penalty => {
  const state = ridgeCurvatureState('duplicate', penalty, 1, 1);
  return { penalty, reportedUnique: state.unique, expectedUnique: penalty > 0,
    reportedCurvature: state.smallestCurvature, expectedCurvature: 2 * penalty,
    reportedOptimum: state.optimum, expectedCoefficient: 5 / (6 + penalty) };
});
const almostFeasible = allocationCertificateState(4, 2.5 + 5e-11, 1.5);
const almostStationary = softThresholdState(0, 0, 5e-11);
const program = String.raw`import json
from fractions import Fraction as Q

def transpose_difference(q):
    return [-q[0]] + [q[i-1]-q[i] for i in range(1,len(q))] + [q[-1]]

checks = []
for y,x,penalty,q,expected in [
    ([Q(1,5),Q(-1,10),Q(1,10),Q(3),Q(16,5),Q(14,5)],
     [Q(1,6)]*3+[Q(29,10)]*3,Q(3,10),[Q(-1,9),Q(7,9),Q(1),Q(2,3),Q(-1,3)],Q(137,150)),
    ([Q(0),Q(0),Q(4),Q(4)], [Q(1,4),Q(1,4),Q(15,4),Q(15,4)],
     Q(1,2), [Q(1,2),Q(1),Q(1,2)],Q(15,8)),
]:
    differences = [x[i+1]-x[i] for i in range(len(x)-1)]
    for difference,slope in zip(differences,q):
        assert abs(slope)<=1 and (difference==0 or slope==(1 if difference>0 else -1))
    residual = [fitted-observed+penalty*slope for fitted,observed,slope in zip(x,y,transpose_difference(q))]
    assert residual==[0]*len(x)
    cost = sum((a-b)**2 for a,b in zip(x,y))/2 + penalty*sum(abs(d) for d in differences)
    assert cost==expected
    checks.append(str(cost))

# Exact original ridge fixture; no numerical solver or copied output oracle.
w = [Q(38,41),Q(24,41)]
rows = [[1,0],[1,1],[1,2]]
response = [1,2,2]
prediction = [sum(a*b for a,b in zip(row,w)) for row in rows]
residual = [a-b for a,b in zip(prediction,response)]
gradient = [2*sum(row[j]*error for row,error in zip(rows,residual))+w[j] for j in range(2)]
assert gradient==[0,0]
fit = sum(error**2 for error in residual)
penalty = sum(a*a for a in w)/2
assert fit==Q(425,1681) and penalty==Q(1010,1681) and fit+penalty==Q(35,41)

# Weighted-allocation exercise certificate on arbitrary rational feasible points.
count=0
u,v=Q(7,5),Q(3,5)
for denominator in range(1,21):
    for first in range(2*denominator+1):
        for second in range(2*denominator-first+1):
            a,b=Q(first,denominator),Q(second,denominator)
            assert Q(-16,5)*((a-u)+(b-v))>=0
            assert (a-3)**2+4*(b-1)**2>=Q(16,5)
            count+=1
print(json.dumps({'exactTVCertificates':checks,'exactOriginalRidgeObjective':str(fit+penalty),'weightedFeasibleCertificateChecks':count}))
`;
writeFileSync(resolve(directory, 'exact-certificates.py'), program);
const native = spawnSync(resolve('scratch/lesson-tools/Scripts/python.exe'), ['-I', resolve(directory, 'exact-certificates.py')], { encoding: 'utf8' });
assert.equal(native.status, 0, native.stderr);
const results = {
  reviewedAt: new Date().toISOString(),
  scope: 'Finite independent source/math review, not a replacement for the author browser/native record.',
  ridge,
  continuousCertificateCases: {
    allocation: { candidate: almostFeasible.candidate, reportedFeasible: almostFeasible.feasible, actualViolation: almostFeasible.violation, candidateCost: almostFeasible.objective, actualOptimalCost: almostFeasible.optimumValue },
    threshold: { reportedStationary: almostStationary.stationary, candidate: almostStationary.candidate, subgradient: almostStationary.subgradient, exactOptimum: almostStationary.optimum },
  },
  independentNative: JSON.parse(native.stdout),
};
const finalReview = process.env.CONVEX_REVIEW_PHASE === 'final';
if (finalReview) {
  for (const state of ridge) {
    assert.equal(state.reportedUnique, state.expectedUnique);
    assert.equal(state.reportedCurvature, state.expectedCurvature);
    assert.deepEqual(state.reportedOptimum, [state.expectedCoefficient, state.expectedCoefficient]);
  }
  assert.equal(almostFeasible.feasible, false);
  assert.equal(almostFeasible.gap, null);
  assert.equal(almostFeasible.actualError, null);
  assert.equal(almostStationary.stationary, false);
  assert.throws(() => ridgeContourPoints(ridgeCurvatureState('duplicate', Number.MIN_VALUE, 1, 1), 100));
  results.reportedFindingsResolved = true;
}
writeFileSync(resolve(directory, finalReview ? 'final-results.json' : 'results.json'), JSON.stringify(results, null, 2));
console.log(JSON.stringify(results, null, 2));
