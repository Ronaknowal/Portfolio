import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { blockOccupancy, causalEdges, erf, featureMemory, forecastTrajectory, graphReach, memoryDefaults, projectedRead, randomFeatureRead, sparseRead } from '../src/learn/data/sparse-attention-models.js';

const id = 'sparse-linear-attention-variants';
const folder = `docs/teaching/deep-learning-completion/${id}`;
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const close = (actual, expected, tolerance = 1e-10) => {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); expected.forEach((value, i) => close(actual[i], value, tolerance)); }
  else assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}; tolerance ${tolerance}`);
};
const maximumError = (a, b) => Math.max(...a.flat(Infinity).map((v, i) => Math.abs(v - b.flat(Infinity)[i])));
const checks = [];
const check = (name, run) => { run(); checks.push({ name, passed: true }); };
check('Sparse normalization, missing denominator, removed mass, constant and translation controls', () => {
  const scores = [0, Math.log(3), Math.log(2), Math.log(4)], values = [[-2, 1], [1, 2], [3, -1], [0, 4]], legal = [true, false, true, true];
  const result = sparseRead(scores, values, legal);
  close(result.dense, [.7, 2.1]); close(result.output, [4 / 7, 15 / 7]); close(result.removedMass, .3);
  close(sparseRead(scores.map(x => x + 1000), values, legal).output, result.output);
  close(sparseRead(scores, values.map(() => [2, -1]), legal).output, [2, -1]);
  assert.equal(sparseRead(scores, values, [false, false, false, false]).output, null);
  close(sparseRead(scores, values, [false, false, true, false]).output, values[2]);
});
check('Directed graph depth and causal hub distinction, including residual reach', () => {
  assert.equal(graphReach(causalEdges(12, 3), 2, 2).edges, 33);
  assert.equal(graphReach(causalEdges(10, 2), 1, 2).paths[9], null);
  assert.deepEqual(graphReach(causalEdges(10, 2, 4), 1, 2).paths[9], [1, 4, 9]);
  assert.equal(graphReach(causalEdges(10, 2, 0), 1, 2).paths[9], null);
  assert.deepEqual(graphReach([[false]], 0, 4).paths[0], [0, 0, 0, 0, 0]);
});
check('Outer products, query-only state invariance, zero overlap and known-write eviction', () => {
  const { query, keys, values } = memoryDefaults;
  const result = featureMemory(query, keys, values);
  close(result.matrix, [[4, 4], [5, 1]]); close(result.normalizer, [4, 4]); close(result.output, [1.1875, .4375]);
  close(featureMemory([7, 1], keys, values).matrix, result.matrix);
  close(featureMemory(query, keys.slice(1), values.slice(1)).output, [4 / 3, 14 / 9]);
  close(featureMemory(query, keys, keys.map(() => [2, -2])).output, [2, -2]);
  assert.equal(featureMemory([0, 0], keys, values).output, null);
});
check('Fixed Gaussian draw, nonmonotone approximation, future exclusion and constant values', () => {
  const fixture = read('src/learn/data/sparse-attention-examples.json').random_features_fresh;
  const evaluate = (count, key = fixture.key_scaled, values = fixture.value) => randomFeatureRead(fixture.query_scaled, key, values, fixture.projections.slice(0, count));
  close(evaluate(8).output, fixture.m8); close(evaluate(64).output, fixture.m64);
  close(evaluate(8).relativeError, .07823265999, 1e-10); close(evaluate(64).relativeError, .13450358670, 1e-10);
  assert.ok(evaluate(64).relativeError > evaluate(8).relativeError);
  const changed = evaluate(8, fixture.changed_key);
  close(changed.output[0], evaluate(8).output[0]); assert.ok(maximumError(changed.output[3], evaluate(8).output[3]) > .04);
  close(evaluate(8, fixture.key_scaled, fixture.value.map(() => [2, -1])).output, Array(4).fill([2, -1]));
});
check('Full versus causal prefix summary and block occupancy endpoints', () => {
  close(projectedRead([4, -2, 3, 8], [.25, .5, 0, .25], 1).full, 2);
  close(projectedRead([4, -2, 3, -4], [.25, .5, 0, .25], 1).full, -1);
  close(projectedRead([4, -2, 3, -4], [.25, .5, 0, .25], 1).prefix, 0);
  const clustered = Array.from({ length: 8 }, (_, r) => Array.from({ length: 8 }, (_, c) => r < 4 && c < 2));
  const dispersed = Array.from({ length: 8 }, (_, r) => Array.from({ length: 8 }, (_, c) => c === 3 * r % 8));
  assert.equal(blockOccupancy(clustered, 2).candidates, 8); assert.equal(blockOccupancy(dispersed, 2).candidates, 32);
  assert.equal(blockOccupancy(dispersed, 1).candidates, 8); assert.equal(blockOccupancy(dispersed, 8).candidates, 64);
  assert.equal(blockOccupancy(Array.from({ length: 8 }, () => Array(8).fill(false)), 8).candidates, 0);
});
const fixtures = read(`${folder}/native-fixtures.json`), models = Object.fromEntries(['dense', 'window', 'kernel'].map(mode => [mode, read(`public/learn-assets/${id}/${mode}-forecast.json`).weights]));
let maximumPortError = 0, maximumStateError = 0, maximumWeightError = 0, maximumIncrementalError = 0;
check('Exact-erf GELU and all 15 frozen native probes, traces and streamed states', () => {
  close(erf(0), 0); close(erf(1), .8427007929497149, 1e-15); close(erf(-2), -.9953222650189527, 2e-15); close(erf(5), .9999999999984626, 2e-15);
  for (const fixture of fixtures) {
    const weights = models[fixture.mode], result = forecastTrajectory(weights, fixture.points, fixture.mode);
    maximumPortError = Math.max(maximumPortError, maximumError(result.predictions, fixture.predictions));
    close(result.predictions, fixture.predictions, 1e-5);
    if (fixture.matrix) {
      const error = maximumError(result.matrices, fixture.matrix);
      maximumStateError = Math.max(maximumStateError, error); close(result.matrices, fixture.matrix, 1e-4);
      close(result.normalizers, fixture.normalizer, 1e-4);
    } else {
      const rows = result.weightRows.map(head => head.at(-1));
      maximumWeightError = Math.max(maximumWeightError, maximumError(rows, fixture.last_weights));
      close(rows, fixture.last_weights, 1e-5);
    }
    let cache = null;
    const stream = fixture.points.map((point, position) => {
      const step = forecastTrajectory(weights, [point], fixture.mode, { cache, start: position });
      cache = step.cache; return step.forecast;
    });
    maximumIncrementalError = Math.max(maximumIncrementalError, maximumError(stream, result.predictions));
    close(stream, result.predictions, 1e-12);
  }
});
check('Fresh near versus remote forecast edits preserve strict earlier outputs and window boundary', () => {
  const points = read('src/learn/data/sparse-attention-examples.json').sourcePoints.slice(0, 27);
  for (const mode of ['dense', 'window', 'kernel']) {
    const baseline = forecastTrajectory(models[mode], points, mode);
    for (const position of [19, 25]) {
      const edited = points.map((row, i) => i === position ? [row[0], 1 - row[1]] : row);
      const changed = forecastTrajectory(models[mode], edited, mode);
      close(changed.predictions.slice(0, position), baseline.predictions.slice(0, position), 0);
      if (mode === 'window' && position === 19) close(changed.forecast, baseline.forecast, 0);
      else assert.ok(maximumError(changed.forecast, baseline.forecast) > 1e-5);
    }
    assert.equal(baseline.payloadBytes, { dense: 5184, window: 960, kernel: 864 }[mode]);
  }
});
check('All current JSX parses; learner program copies exactly match canonical packet', () => {
  for (const path of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/SparseAttentionLabs.jsx', 'src/learn/components/lesson-labs/SparseAttentionFigures.jsx']) parse(fs.readFileSync(path, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  for (const file of fs.readdirSync(`public/learn-code/${id}`)) assert.ok(fs.readFileSync(`public/learn-code/${id}/${file}`).equals(fs.readFileSync(`docs/teaching/drafts/${id}/${file}`)), `Stale deployed ${file}`);
});
const result = { passed: true, checks, nativeFixtures: fixtures.length, maximumPortError, maximumStateError, maximumWeightError, maximumIncrementalError };
fs.writeFileSync(`${folder}/model-checks.json`, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
