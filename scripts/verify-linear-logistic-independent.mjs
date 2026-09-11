import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { residualReport } from '../src/learn/data/linear-logistic-models.js';

const cases = [];
// Independently expand E[(Y-b)^2] using exact fixture moments
// E[Y] = 9/4 and E[Y^2] = 25/4. No residual loop is reused.
for (const intercept of [-1e154, -1e150, -1000, 0, 0.25, 2.25, 1000, 1e150, 1e154]) {
  const actual = residualReport(intercept, 0).mse;
  const expected = intercept * intercept - 4.5 * intercept + 6.25;
  assert(Number.isFinite(actual));
  assert(Math.abs(actual - expected) <= Math.max(1, expected) * 5e-16);
  cases.push({ intercept, actual, expected });
}
assert.throws(() => residualReport(0, 1e154), RangeError);
const path = 'src/learn/data/linear-logistic-models.js';
const result = {
  checkedAt: new Date().toISOString(),
  modelHash: createHash('sha256').update(fs.readFileSync(path)).digest('hex'),
  initialFinding: {
    observedBeforeRepair: true,
    originalModelHash: '1cc8b6ba6d7df1b1b882476a90da59c67163658127ed328b2a21409c047f2795',
    input: { intercept: 1e154, slope: 0 },
    observedMse: 'Infinity',
    explanation: 'Four finite squares near 1e308 overflowed their total although their mean is representable. This historical observation was reported to the author before this final regression record was created.'
  },
  changedConstantPredictionCases: cases,
  unrepresentableSquaredResidualRejected: true
};
fs.writeFileSync('scratch/linear-logistic-independent-review/model-boundaries.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ checkedAt: result.checkedAt, changedCases: cases.length, modelHash: result.modelHash }));
