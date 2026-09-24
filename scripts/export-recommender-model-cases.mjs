import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as models from '../src/learn/data/recommender-models.js';

const cases = { feedback: [], neighbors: [], updates: [], implicit: [], pairs: [], rotations: [], rankings: [], policies: [] };
for (const mode of ['explicit', 'implicit']) for (const rating of [null, 0, 4]) for (const count of [0, 2, 7]) for (const prediction of [-1, .5, 3]) {
  const input = { mode, rating, count, prediction, alpha: 2 };
  cases.feedback.push({ input, result: models.feedbackCell(input) });
}
for (const item of [2, 4, 6]) for (const centered of [false, true]) for (const signed of [false, true]) for (const minimumOverlap of [1, 2, 4]) for (const shrinkage of [0, 2, 6]) {
  const input = { item, centered, signed, minimumOverlap, shrinkage };
  cases.neighbors.push({ input, result: models.neighborhoodPrediction(input) });
}
for (let index = 0; index < 100; index += 1) {
  const input = { userFactors: [Math.sin(index), Math.cos(2 * index)], itemFactors: [Math.cos(index / 2), Math.sin(index / 3)], userBias: (index % 5 - 2) / 10, itemBias: (index % 7 - 3) / 10, rating: index % 5 + 1, rate: (index % 8 + 1) / 10, penalty: (index % 4) / 5 };
  cases.updates.push({ input, result: models.explicitFactorStep(input) });
  const pairInput = { user: input.userFactors, positive: input.itemFactors, negative: [Math.sin(index / 4), Math.cos(index / 5)], rate: input.rate, penalty: input.penalty };
  cases.pairs.push({ input: pairInput, result: models.bprPairStep(pairInput) });
}
for (const first of [0, 1, 4]) for (const second of [0, 2, 7]) for (const third of [0, 3, 8]) for (const alpha of [0, 2, 7]) for (const penalty of [.1, 1, 3]) for (const includeMissing of [false, true]) {
  const input = { counts: [first, second, third], alpha, penalty, includeMissing };
  cases.implicit.push({ input, result: models.implicitFactorBlock(input) });
}
for (let angle = -180; angle <= 180; angle += 15) cases.rotations.push({ angle, result: models.rotatedFactors(angle) });
function permutations(values) {
  if (!values.length) return [[]];
  return values.flatMap((value, index) => permutations(values.filter((_, other) => other !== index)).map(tail => [value, ...tail]));
}
for (const order of permutations([0, 1, 2, 3, 4])) for (const grades of [[1, 1, 0, 0, 0], [2, 1, 0, 3, 0], [0, 0, 0, 0, 0]]) for (const cutoff of [1, 3, 5]) for (const omit of [false, true]) {
  const input = { order: order.filter(item => !omit || item !== 1), grades, cutoff };
  cases.rankings.push({ input, result: models.evaluateRecommendationList(input) });
}
for (const loggingA of [0, .05, .2, .5, .8, .95, 1]) for (const targetA of [0, .25, .5, .75, 1]) for (const qualityA of [0, .4, 1]) {
  const input = { loggingA, targetA, qualityA, qualityB: .8, requests: 100 };
  cases.policies.push({ input, result: models.exposurePolicy(input) });
}
const empty = models.neighborhoodPrediction({ matrix: [[null, null], [null, null]], item: 1 });
assert.equal(empty.prediction, 3);
assert.equal(empty.usedFallback, true);
assert.equal(models.summarizeRatings([[0, null]]).observed.length, 1);
assert.equal(models.evaluateRecommendationList({ order: [], grades: [1, 1], cutoff: 3 }).bestCandidateRecall, 0);
assert.equal(models.evaluateRecommendationList({ order: [], grades: [1, 1], cutoff: 3, eligible: [] }).recall, null);
for (const rate of [.01, .05, .5, 1]) for (const steps of [0, 1, 8, 15]) {
  const trace = models.factorUpdateTrace({ rate, steps });
  assert.equal(trace.completedSteps, trace.trace.length);
  assert.ok(Number.isFinite(trace.final.loss));
}
let rejected = 0;
for (const operation of [
  () => models.summarizeRatings([[1, , 3]]),
  () => models.summarizeRatings([[NaN]]),
  () => models.feedbackCell({ mode: 'stars' }),
  () => models.neighborhoodPrediction({ item: 0 }),
  () => models.evaluateRecommendationList({ order: [0, 0] }),
  () => models.evaluateRecommendationList({ order: [0], grades: [1], eligible: [] }),
  () => models.implicitFactorBlock({ penalty: 0 }),
  () => models.exposurePolicy({ loggingA: -1 }),
  () => models.exposurePolicy({ loggingA: 1e-310, targetA: .5 }),
]) { assert.throws(operation); rejected += 1; }
const inherited = new Array(2);
Object.setPrototypeOf(inherited, Object.assign(Object.create(Array.prototype), { 0: 1, 1: 2 }));
assert.throws(() => models.summarizeRatings([inherited]));
rejected += 1;
fs.mkdirSync('scratch/recommender-native', { recursive: true });
fs.writeFileSync('scratch/recommender-native/model-cases.json', JSON.stringify(cases));
fs.writeFileSync('scratch/recommender-native/model-contracts.json', JSON.stringify({ timestamp: new Date().toISOString(), invalidCasesRejected: rejected, traceCases: 16, counts: Object.fromEntries(Object.entries(cases).map(([key, values]) => [key, values.length])) }, null, 2));
console.log('Exported actual model cases; bounded validation and trace contracts passed.');
