import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { projectionCertificateState, resourceAllocationAtPrice, resourceDualAscentState } from '../src/learn/data/duality-kkt-models.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 1e-11 * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
const repairedCases = [[2.75, 1.75, 18], [3.25, 1.75, 17], [4, 1.75, 16]].map(([budget, rate, step]) => {
  const frame = resourceDualAscentState(budget, rate, 0, 20).frames[step];
  assert.ok(frame.subtractedGap < 0, 'The cancellation reproducer should still expose naive subtraction.');
  assert.ok(frame.certificateGap > 0, 'The stable formula must preserve the small positive gap.');
  return { budget, rate, step, subtraction: frame.subtractedGap, stableGap: frame.certificateGap };
});
let optimumChecks = 0;
for (const budget of [0, 0.5, 1, 2, 2.5, 3, 4, 5, 6, 7, 8]) {
  // Independently minimize the two scalar quadratics under their active faces.
  const price = budget >= 7 ? 0 : budget <= 2.5 ? 16 - 4 * budget : (28 - 4 * budget) / 3;
  const x1 = budget >= 7 ? 3 : budget <= 2.5 ? 0 : (2 * budget - 5) / 3;
  const x2 = budget >= 7 ? 4 : budget <= 2.5 ? budget : (budget + 5) / 3;
  const value = (x1 - 3) ** 2 + 2 * (x2 - 4) ** 2;
  const state = resourceAllocationAtPrice(price, budget);
  close(state.allocation[0], x1);
  close(state.allocation[1], x2);
  close(state.dualValue, value);
  close(state.certificateGap, 0);
  optimumChecks += 1;
}
for (const [price, budget, expectedGap] of [[9, 3, 11.25], [16, 0, 0], [20, 2, 40]]) {
  close(resourceAllocationAtPrice(price, budget).certificateGap, expectedGap);
}
const projection = projectionCertificateState(4, [1.5, 2.5], 3);
close(projection.optimalValue, 4.5);
close(projection.certifiedGap, 0);
const files = ['src/learn/data/topics/convex-duality-lagrangian-methods-kkt-conditions.jsx', 'src/learn/data/duality-kkt-models.js', 'src/learn/data/duality-kkt-examples.js'];
const result = { checkedAt: new Date().toISOString(), rootRead: 'Complete body, all ten programs, pure models and six practice solutions; focused independent mathematical review.', repairedCases, optimumChecks, boundaryChecks: 3, changedProjection: true, sourceHashes: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])), finding: 'Naive gap subtraction could display a negative certificate. Author replaced it with a nonnegative decomposition and explained the numerical distinction; targeted reproductions pass.' };
fs.writeFileSync('docs/teaching/evidence/duality-kkt-independent-review.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
