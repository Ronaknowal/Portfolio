import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { digitUpdate, lionTrace, sampledCurvature, hessianProbes, prodigyTrace, scheduleFreeTrace, bowlGeometry } from '../src/learn/data/advanced-optimizer-models.js';
const id = 'advanced-optimizers-lion-sophia-prodigy-schedule-free';
const root = `docs/teaching/deep-learning-completion/${id}`;
const native = JSON.parse(fs.readFileSync(`${root}/native-fixtures.json`, 'utf8'));
const snapshots = JSON.parse(fs.readFileSync(`docs/teaching/drafts/${id}/fitted-optimizer-states.json`, 'utf8'));
let comparisons = 0, maximumError = 0;
function near(actual, expected, tolerance = 1e-10) {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); actual.forEach((v, i) => near(v, expected[i], tolerance)); }
  else { const error = Math.abs(actual - expected); assert.ok(Number.isFinite(error) && error <= tolerance, `${actual} != ${expected}`); maximumError = Math.max(maximumError, error); comparisons++; }
}
for (const item of native.cases) {
  const snapshot = snapshots.find(s => s.method === item.method && s.seed === item.seed);
  assert.deepEqual(JSON.parse(fs.readFileSync(`public/learn-assets/${id}/model-${item.method}-${item.seed}.json`, 'utf8')), snapshot);
  const result = digitUpdate(snapshot, item.pixels, item.target);
  near(result.before.probabilities, item.before); near(result.training.probabilities, item.training);
  near(result.gradient, item.gradient); near(result.after.probabilities, item.after);
  near(result.next.parameters, item.parameters);
  for (const key of Object.keys(item.state)) near(result.next.state[key], item.state[key], 1e-9);
  const changedTarget = digitUpdate(snapshot, item.pixels, (item.target + 1) % 10);
  near(result.before.probabilities, changedTarget.before.probabilities, 0);
  assert.notDeepEqual(result.gradient, changedTarget.gradient);
  if (item.method === 'adamw_cosine') { near(result.next.parameters, snapshot.parameters, 0); near(result.after.probabilities, result.before.probabilities, 0); }
}
const a = native.calculated;
near(lionTrace({})[0].next, a.lion.fresh.next_parameter); near(lionTrace({})[0].nextMemory, a.lion.fresh.next_momentum);
assert.equal(lionTrace({ momentum: 0, gradients: [0], decay: 0 })[0].next, -.4);
assert.notEqual(lionTrace({ gradients: [2] })[0].direction, lionTrace({ gradients: [4] })[0].direction);
const curvature = sampledCurvature([3, 1], [.25, .6], [1, 0]);
near(curvature.expectation, .96375); near(curvature.trueSquare, .680625); near(curvature.expectation, curvature.exact);
near(sampledCurvature([3, 1], [.25, .6], [0, 1]).expectation, curvature.expectation, 0);
const probes = hessianProbes(4, -2, 1); near(probes.values.slice(0, 2).map(r => r.estimate), [[2, -1], [6, 3]]); near(probes.eigenvalues, [0, 5]);
for (const [key, scale] of [['fresh', .3], ['fresh_scale_contrast', 1]]) {
  const rows = prodigyTrace(1, -2, .01, scale).rows;
  near(rows.map(r => r.after), a.prodigy[key].trace.map(r => r.parameter));
  near(rows.map(r => r.distance_used), a.prodigy[key].trace.map(r => r.distance_used));
}
near(prodigyTrace(1, 1, 1e-6).rows.map(r => r.after), Array(12).fill(1), 0);
for (const [key, beta] of [['fresh', .9], ['fresh_beta_zero', 0], ['fresh_beta_one', 1]]) {
  const rows = scheduleFreeTrace(-1, 1, [-.5, .5, 1, -1], beta);
  for (const field of ['training', 'gradient', 'fast', 'average', 'loss']) near(rows.map(r => r[field]), a.schedule_free[key].map(r => r[field]));
}
const geometry = bowlGeometry(); near(geometry.H, a.geometry.hessian); near(geometry.descent, a.geometry.gd_rate_008); near(geometry.diagonal, a.geometry.diagonal_newton);
const study = JSON.parse(fs.readFileSync(`docs/teaching/drafts/${id}/study-results.json`, 'utf8'));
assert.deepEqual(JSON.parse(fs.readFileSync(`public/learn-assets/${id}/histories.json`, 'utf8')), { selected: study.selected, runs: study.runs });
for (const name of ['optimizer_rules.py', 'optimizer_study.py', 'optimizer_calculations.py', 'optimizer_library_bridge.py', 'digits-400.csv', 'study-results.json', 'fitted-optimizer-states.json', 'calculated-inputs.json', 'data-provenance.md', 'lion_pytorch.py', 'sophia.py', 'LICENSE-lion.txt', 'LICENSE-sophia.txt']) assert.equal(fs.readFileSync(`public/learn-assets/${id}/${name}`).equals(fs.readFileSync(`docs/teaching/drafts/${id}/${name}`)), true, name);
for (const name of ['AdvancedOptimizerDiagrams', 'AdvancedOptimizerLabs', 'AdvancedOptimizerStudy']) parse(fs.readFileSync(`src/learn/components/lesson-labs/${name}.jsx`, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
parse(fs.readFileSync(`src/learn/data/topics/${id}.jsx`, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const result = { passed: true, scalarComparisons: comparisons, maximumError, checks: ['All12 fresh stateful digit updates versus native NumPy with torch-verified gradients, every stored state array', 'Label-independent before probabilities and changed gradients', 'Cosine update401 exact frozen-parameter null', 'Lion magnitude reversal and exact zero-history null', 'Independent model-label outcome expectation and true-label contrast', 'Hutchinson negative sample from PSD matrix', 'Full Prodigy altered-scale native histories and stationary null', 'Schedule-Free native traces at beta0/.9/1', 'Rotated bowl matrix and update endpoints', 'Every deployed model and explicit learner download equals its canonical source', 'Lossless measured histories and all JSX source parse'] };
fs.writeFileSync(`${root}/model-checks.json`, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
