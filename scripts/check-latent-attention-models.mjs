import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { latentDefaults, latentRead, rotationOrder, rankOne, latentBudget, latentForecast, difference, matvec } from '../src/learn/data/latent-attention-models.js';

const id = 'multi-head-latent-attention-mla', root = `docs/teaching/deep-learning-completion/${id}`;
const fixtures = JSON.parse(fs.readFileSync(`${root}/native-fixtures.json`, 'utf8'));
let scalarComparisons = 0, maxForecastError = 0;
function near(actual, expected, tolerance = 1e-10) {
  if (Array.isArray(actual)) { assert.equal(actual.length, expected.length); actual.forEach((value, i) => near(value, expected[i], tolerance)); }
  else { assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`); scalarComparisons++; }
}
const state = latentDefaults();
for (const expanded of [false, true]) {
  const rows = latentRead(state, { expanded });
  near(rows.map(r => r.output), fixtures.practice.I1.base.head_outputs);
  near(rows.map(r => r.weights), fixtures.practice.I1.base.weights);
}
near(latentRead(state, { changedBasis: true }).map(r => r.output), latentRead(state).map(r => r.output));
const edit = structuredClone(state); edit.latent[1][0] += .75;
near(latentRead(edit).map(r => r.output), fixtures.practice.I1.changed_latent.head_outputs);
const valueEdit = structuredClone(state); valueEdit.valueUp[0][0][0] += .5;
near(latentRead(valueEdit).map(r => r.weights), latentRead(state).map(r => r.weights));
near(latentRead(valueEdit)[1].output, latentRead(state)[1].output);
assert.throws(() => latentRead({ ...state, queryPosition: 0, positions: [1, 2, 3] }), /No legal key/);
const nonlinearFixture = { queries: [[0, 0]], latent: [[-1, 0], [1, 0]], keyUp: [[[1, 0], [0, 1]]], valueUp: [[[1, 0], [0, 1]]], queryRotary: [[0, 0]], keyRotary: [[0, 0], [0, 0]], positions: [0, 1], queryPosition: 1, frequency: 0 };
near(latentRead(nonlinearFixture, { nonlinear: true, expanded: true })[0].output, [.5, 0]);
near(latentRead(nonlinearFixture, { nonlinear: true })[0].output, [0, 0]);
const rotation = rotationOrder([[1, 1], [0, 2]], [1, -1], Math.PI / 3);
near(rotation.projectThenRotate, fixtures.practice.I2.project_then_rotate);
near(rotation.rotateThenProject, fixtures.practice.I2.rotate_then_project);
near(rotationOrder([[2, 0], [0, 2]], [1, -1], Math.PI / 3).projectThenRotate, fixtures.practice.I2.isotropic_control);
for (const test of fixtures.rankCases) {
  const result = rankOne(test.matrix, test.input);
  near(result.output, test.output); near(result.full, test.full); near(result.matrixError, test.matrixError);
}
const budgetState = { batch: 3, layers: 24, length: 8192, heads: 24, content: 64, value: 64, latent: 192, rotary: 32, bytes: 2, queries: 1 };
const budget = latentBudget(budgetState);
near(budget.compact, 264241152); near(budget.expandedOps, 188743680); near(budget.absorbedOps, 490733568);
near(latentBudget({ ...budgetState, heads: 48 }).compact, budget.compact);
near(latentBudget({ ...budgetState, heads: 48 }).absorbedOps, budget.absorbedOps * 2);
const huge = latentBudget({batch:63,layers:255,length:262143,heads:255,content:2047,value:2047,latent:2047,rotary:254,bytes:4,queries:262142});
assert.equal(huge.exact.absorbedOps, 9600086246924179440n);
assert.equal(huge.exact.compact, 63n*255n*262143n*4n*2301n);
assert.equal(huge.exact.mqa * 8n, 63n*255n*262143n*4n*4094n*8n);
const joined = latentRead(state).flatMap(row => row.output), outputMap = [[1,0,.5,0],[0,1,0,.5]];
near(matvec(outputMap, joined), [joined[0] + .5*joined[2], joined[1] + .5*joined[3]]);
near(matvec([[0,0,0,0],[0,0,0,0]], joined), [0,0]);
near(matvec(outputMap, latentRead(state, {expanded:true}).flatMap(row => row.output)), matvec(outputMap, joined));
const model = JSON.parse(fs.readFileSync(`public/learn-assets/${id}/runtime.json`, 'utf8'));
const full = latentForecast(model.weights, model.points.slice(0, 27));
near(latentForecast(model.weights, model.points.slice(0, 27), { basis: model.completeBasis }).predictions, full.predictions, 2e-6);
assert.ok(difference(full.predictions, latentForecast(model.weights, model.points.slice(0, 27), { wrongScale: true }).predictions) > 1e-6);
const rankFour = model.completeBasis.map(row => row.slice(0, 4));
near(latentForecast(model.weights, model.points.slice(0, 27), { basis: rankFour }).predictions, latentForecast(model.weights, model.points.slice(0, 27), { basis: rankFour, wrongScale: true }).predictions, 0);
for (const test of fixtures.cases) {
  const basis = test.rank === null ? null : model.completeBasis.map(row => row.slice(0, test.rank));
  const options = { basis, shift: test.shift, wrongScale: test.wrong };
  const result = latentForecast(model.weights, test.points, options);
  near(result.predictions, test.predictions, 2e-5);
  near(result.traces.at(-1).map(h => h.weights), test.weights, 2e-5);
  near(result.traces.at(-1).map(h => h.output), test.heads, 2e-5);
  near(result.traces.at(-1).map(h => h.contentScores), test.content, 2e-5);
  near(result.traces.at(-1).map(h => h.rotaryScores), test.rotary, 2e-5);
  near(latentForecast(model.weights, test.points, { ...options, expanded: true }).predictions, result.predictions, 1e-12);
  maxForecastError = Math.max(maxForecastError, difference(result.predictions, test.predictions));
  let cache = null; const incremental = [];
  for (const point of test.points) { const step = latentForecast(model.weights, [point], { ...options, cache }); cache = step.cache; incremental.push(step.predictions[0]); }
  near(incremental, result.predictions, 1e-12);
  near(latentForecast(model.weights, test.points, { ...options, shift: test.shift + 100 }).predictions, result.predictions, 1e-12);
}
for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/LatentAttentionLabs.jsx']) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const receipt = { passed: true, scalarComparisons, maxForecastError, checks: ['fresh NumPy expanded/absorbed fixtures and consistent basis', 'value-only null and latent edit', 'empty legal set rejects', 'projection/rotation order and isotropic null', '15 NumPy SVD input cases including zero and repeated singular values', 'independent cache/operation formulas and doubled-head null', '16 full-model native cases including content/rotary scores', 'expanded/absorbed, incremental and consistent-position-shift parity', 'owned JSX parses'] };
fs.writeFileSync(`${root}/model-checks.json`, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
