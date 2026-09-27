import fs from 'node:fs';
import assert from 'node:assert/strict';
import { movementBlockForward, normalizationProbe, feedforwardTrace, gatedFeedforwardTrace, blockCosts } from '../../../../src/learn/data/transformer-block-models.js';
const directory = 'docs/teaching/deep-learning-completion/transformer-block-architecture';
const reference = JSON.parse(fs.readFileSync(`${directory}/independent-fixtures.json`, 'utf8'));
const models = JSON.parse(fs.readFileSync('public/learn-code/transformer-block-architecture/movement-runtime.json', 'utf8'));
let comparisons = 0, maxError = 0;
function near(actual, expected, tolerance = 1e-4) {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); actual.forEach((v, i) => near(v, expected[i], tolerance)); return; }
  const error = Math.abs(actual - expected);
  assert.ok(Number.isFinite(error) && error <= tolerance, `${actual} versus ${expected}`); comparisons++; maxError = Math.max(maxError, error);
}
for (const item of reference.cases) {
  const actual = movementBlockForward(models[item.placement], item.points, item.times, item.padding);
  near(actual.logits, item.logits);
  near(actual.traces.map(t => t.output), item.stages);
  const reversed = movementBlockForward(models[item.placement], [...item.points].reverse(), [...item.times].reverse(), [...item.padding].reverse());
  near(actual.logits, reversed.logits, 1e-10);
  if (item.case === 'interleaved-padding') {
    const plain = reference.cases.find(c => c.placement === item.placement && c.case === 'changed-point-and-tag');
    near(actual.logits, movementBlockForward(models[item.placement], plain.points, plain.times).logits, 1e-10);
  }
}
for (const item of reference.probes) near(normalizationProbe(item.input, item.probe, item.epsilon).gradient, item.gradient, 1e-5);
// Hand-selected signed writes: h=[2,3], rows [1,-2] and [-1,4], sum [-1,8].
const trace = feedforwardTrace([2, 3], [[1, 0], [0, 1]], [[1, -2], [-1, 4]]);
near(trace.contributions, [[2, -4], [-3, 12]], 0); near(trace.output, [-1, 8], 0);
const gated = gatedFeedforwardTrace([2, -1], [[1, 2], [0, 1]], [[.5, -1], [0, 0]], [[1, 0], [0, 1]]);
near(gated.value, [2, 3], 0); near(gated.gateLogits, [1, -2], 0);
near(gated.output, [2 / (1 + Math.exp(-1)), -6 / (1 + Math.exp(2))], 1e-12);
assert.equal(blockCosts(91, 96, 240).parameters, 84048);
assert.equal(blockCosts(182, 96, 240).maps, 2 * blockCosts(91, 96, 240).maps);
assert.equal(blockCosts(182, 96, 240).pairs, 4 * blockCosts(91, 96, 240).pairs);
const result = { passed: true, comparisons, maxAbsoluteError: maxError, checks: ['Six unseen edited/constant/interleaved-padding cases compared with independent functional torch SDPA in float64, including both full block outputs', 'Complete-record reverse null for every case', 'Interspersed masked records reproduce original computation in both layers and valid-only pool', 'Three independent autograd probes including constant input and all-ones null', 'Signed feature contribution sums and changed-practice parameter/scaling counts'], limits: ['No duplicate training campaign.', 'Browser painted geometry and interaction acceptance remain root-owned.'] };
fs.writeFileSync(`${directory}/independent-numerical.json`, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
