import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { fittedConstant, regressionSlope, focalTerm, confusionAt, tripletGeometry, candidateCompetition } from '../src/learn/data/loss-functions-models.js';
import { normalizeVector, tensorGroup, normalizeTensor, batchNormalizationStep, normalizationGradient } from '../src/learn/data/normalization-models.js';
import digitData from '../src/learn/data/loss-functions-measurements.js';
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
let assertions = 0;
function close(actual, expected, tolerance = 1e-9) { assertions++; assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`); }
function same(actual, expected) { assertions++; assert.deepEqual(actual, expected); }
for (const v of [2, 10, 37.5, 100]) {
  const values = [0, 0, 0, 0, 0, 0, v];
  close(fittedConstant(values, 'mse'), v / 7); close(fittedConstant(values, 'mae'), 0); close(fittedConstant(values, 'huber'), 1 / 6);
}
for (const method of ['mse', 'mae', 'huber']) close(fittedConstant(Array(7).fill(3), method), 3);
for (const values of [[-100, -8, -2, 0, 1, 2, 100], [-4, -3, -2, -1, 0, 1, 2]]) {
  const fit = fittedConstant(values, 'huber');
  assert.ok(values.reduce((sum, x) => sum + regressionSlope(fit - .01 - x, 'huber'), 0) < 0);
  assert.ok(values.reduce((sum, x) => sum + regressionSlope(fit + .01 - x, 'huber'), 0) > 0); assertions += 2;
}
for (const probability of [.001, .01, .1, .5, .9, .99]) for (const target of [0, 1]) for (const gamma of [0, 2, 5]) {
  const result = focalTerm(probability, target, gamma), logit = Math.log(probability / (1 - probability));
  const reference = z => { const p = 1 / (1 + Math.exp(-z)), pt = target ? p : 1 - p; return -((1 - pt) ** gamma) * Math.log(pt); };
  close(result.slope, (reference(logit + 1e-5) - reference(logit - 1e-5)) / 2e-5, 2e-8);
  if (gamma === 0) close(result.slope, probability - target);
}
close(1000 * focalTerm(.01, 0, 2).slope, .002989966499, 1e-10);
close(focalTerm(.1, 1, 2).slope, -1.102018785065, 1e-10);
const labels = digitData.validation_source_ids.map(id => digitData.specimens.find(row => row.source_id === id).digit === 9 ? 1 : 0);
same(confusionAt(digitData.records[0].validation_probabilities, labels, .1), { tp: 12, fp: 2, fn: 0, tn: 106 });
for (const run of digitData.records) { const c = confusionAt(run.validation_probabilities, labels, .5); same([[c.tn, c.fp], [c.fn, c.tp]], run.confusion_matrix); }
const points = [[0, 0], [1, 0], [.5, 0], [1.2, 0], [2, 0]];
same(tripletGeometry(points, 1).selected, 1); close(tripletGeometry(points, 1).candidates[1].loss, .56);
same(tripletGeometry(points.map(([x, y]) => [x, y + .5]), 1), tripletGeometry(points, 1));
same(tripletGeometry([[0, 0], [1, 0], [.5, 0], [2.5, 0], [2, 0]], 1).selected, null);
same(tripletGeometry([[0, 0], [1, 0], [1, 0], [Math.sqrt(2), 0], [2, 0]], 1).selected, null);
close(tripletGeometry(points, 1, false).candidates[1].loss, .8);
close(candidateCompetition([.8, .2, -.1], .2).loss, .059113895273, 1e-10);
close(candidateCompetition([.2, .8, -.1], .2).loss, 3.059113895273, 1e-10);
for (const t of [.05, .2, 1, 2]) close(candidateCompetition([.4, .4, .4], t).loss, Math.log(3));
assert.throws(() => candidateCompetition([1, 2], 0)); assertions++;
const values = Array.from({ length: 16 }, (_, i) => i + 1), edited = values.map((v, i) => i === 8 ? 19 : v);
same(tensorGroup(0, 'batch'), [0, 1, 8, 9]); same(tensorGroup(0, 'group', 2), [0, 1, 2, 3]);
for (const mode of ['batch', 'layer', 'group', 'instance']) {
  const a = normalizeTensor(values, mode), b = normalizeTensor(edited, mode);
  close(Math.max(...a.slice(0, 8).map((v, i) => Math.abs(v - b[i]))), mode === 'batch' ? .15022057675166323 : 0);
}
same(normalizeTensor(values, 'group', 1), normalizeTensor(values, 'layer'));
same(normalizeTensor(values, 'group', 4), normalizeTensor(values, 'instance'));
for (const [x, gamma, beta, target] of [[[1, 3], [1, 2], [0, 0], [0, 1]], [[-2, .5, 4], [1, -2, .3], [.1, .2, .3], [2, 0, -1]]]) {
  const result = normalizationGradient(x, gamma, beta, target);
  // Perturb only the scalar forward formula, not the derivative helper.
  const forwardLoss = input => { const m = input.reduce((a, b) => a + b) / input.length; const v = input.reduce((a, b) => a + (b - m) ** 2, 0) / input.length; return input.reduce((sum, a, i) => sum + (gamma[i] * (a - m) / Math.sqrt(v + 1e-5) + beta[i] - target[i]) ** 2, 0) / input.length; };
  x.forEach((_, i) => { const hi = x.map((v, j) => v + (i === j ? 1e-5 : 0)), lo = x.map((v, j) => v - (i === j ? 1e-5 : 0)); close(result.inputGradient[i], (forwardLoss(hi) - forwardLoss(lo)) / 2e-5, 2e-8); });
}
const batch = batchNormalizationStep([1, 3, 5, 7], { mean: 0, variance: 1 }, .1, true);
close(batch.next.mean, .4); close(batch.next.variance, 47 / 30); close(batch.output[0], -3 / Math.sqrt(5.00001)); close(batch.evaluationAfter[0], .6 / Math.sqrt(47 / 30 + 1e-5));
same(batchNormalizationStep([2, 4, 6, 8], batch.next, .9, false).next, batch.next);
same(batchNormalizationStep([2, 4, 6, 8], batch.next, 0, true).next, batch.next);
same(normalizeVector([0, 0], 1e-5, true).output, [0, 0]);
const replay = [];
for (const id of ['loss-functions-ce-mse-focal-contrastive-triplet', 'batch-layer-group-rms-normalization']) {
  const packet = `docs/teaching/drafts/${id}`, actual = `scratch/deep-learning-core-implementation/${id}/calculated-inputs.json`;
  same(read(actual), read(`${packet}/calculated-inputs.json`));
  replay.push({ id, original: hash(`${packet}/calculated-inputs.json`), replay: hash(actual), parsedResultsIdentical: true });
}
const sources = ['src/learn/data/loss-functions-models.js', 'src/learn/data/normalization-models.js', 'src/learn/data/loss-functions-measurements.js', 'src/learn/data/normalization-measurements.js', 'scripts/verify-loss-normalization-models.mjs'];
fs.writeFileSync('docs/teaching/evidence/loss-normalization-models.json', JSON.stringify({ passed: true, checkedAt: new Date().toISOString(), assertions, replay, sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])), limits: 'Bounded topic formulas and full prepared CPU programs; not arbitrary production tensors or unseen data.' }, null, 2) + '\n');
console.log(`PASS ${assertions} counted model assertions, full 9 + 12 CPU fits reproduce recorded results.`);
