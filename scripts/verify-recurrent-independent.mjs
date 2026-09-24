import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { recurrentSequence, boundaryExperiment, scalarRecurrence, resetPlacement, maxDifference } from '../src/learn/data/recurrent-models.js';

const evidence = JSON.parse(fs.readFileSync('docs/teaching/evidence/recurrent-independent-native.json', 'utf8'));
assert.equal(evidence.passed, true);
for (const [path, hash] of Object.entries(evidence.sourceHashes)) assert.equal(createHash('sha256').update(fs.readFileSync(path)).digest('hex'), hash);
let maximum = 0, gradientMaximum = 0;
function close(a, b, tolerance = 2e-13) {
  const difference = maxDifference(a, b);
  assert.ok(difference < tolerance, `Difference ${difference} exceeds ${tolerance}`);
  return difference;
}
for (const row of evidence.sequences) {
  const initialCopy = JSON.stringify(row.initial), pointsCopy = JSON.stringify(row.points);
  const full = recurrentSequence(row.kind, row.points, row.weights, row.initial);
  maximum = Math.max(maximum, close(full.map(state => state.hidden), row.output), close(full.map(state => state.probabilities), row.probabilities));
  if (row.finalCell) close(full.at(-1).cell, row.finalCell);
  const first = recurrentSequence(row.kind, row.points.slice(0, 3), row.weights, row.initial);
  const last = first.at(-1);
  const tail = recurrentSequence(row.kind, row.points.slice(3), row.weights, { hidden: last.hidden, cell: last.cell });
  close([...first, ...tail].map(state => state.hidden), row.output);
  assert.equal(JSON.stringify(row.initial), initialCopy);
  assert.equal(JSON.stringify(row.points), pointsCopy);
}
for (const row of evidence.limits) close(recurrentSequence('gru', row.points, row.weights, row.initial)[0].hidden, row.expected);
for (const row of evidence.boundaries) {
  const actual = boundaryExperiment(row.points, row.weights, row.boundary, row.mode);
  close(actual.current.map(state => state.hidden), row.output);
  close([actual.loss], [row.loss]);
  gradientMaximum = Math.max(gradientMaximum, close(actual.gradients, row.gradients, 3e-9));
}
const scalar = { inputs: [.7, -.8, .2, .6, -.3, .5], initial: -.4, target: .3, inputWeight: -.8, recurrentWeight: 1.25, bias: .13, rate: .06 };
const calculation = scalarRecurrence(scalar);
for (const name of ['inputWeight', 'recurrentWeight', 'bias']) {
  const epsilon = 1e-6;
  const difference = (scalarRecurrence({ ...scalar, [name]: scalar[name] + epsilon }).loss - scalarRecurrence({ ...scalar, [name]: scalar[name] - epsilon }).loss) / (2 * epsilon);
  close([difference], [calculation.gradient[name]], 3e-9);
}
const zero = scalarRecurrence({ ...scalar, rate: 0 });
assert.equal(zero.loss, zero.updatedLoss);
// Diagonal mixing does NOT suffice for equality when an unscaled bias remains.
const reset = resetPlacement([.2, -.4], [.3, .7], [[2, 0], [0, -3]], [.5, -.8]);
close(reset.before.map((v, i) => v - reset.after[i]), [.35, -.24]);
const result = { passed: true, groups: ['Nine nonzero-state native/NumPy/JS sequences and source immutability', 'All-cell chunk reconstruction', 'Four GRU reset/update limits with hidden bias', 'Changed-shape boundary gradients against autograd', 'Six-step scalar gradient finite difference and zero-rate null', 'Reset placement diagonal-bias counterexample'], maximumNativeDifference: maximum, maximumGradientDifference: gradientMaximum, sourceHashes: Object.fromEntries(['src/learn/data/recurrent-models.js', 'scripts/verify-recurrent-independent.py', 'scripts/verify-recurrent-independent.mjs'].map(path => [path, createHash('sha256').update(fs.readFileSync(path)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/recurrent-independent-models.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
