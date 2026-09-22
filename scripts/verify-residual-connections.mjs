// Source + numeric author check. Does not build, alter other topics or retrain.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { correctionFixture, correctionModel, scalarStack, operatorPlacement, projectionModel, gateModel, nonlinearPaths } from '../src/learn/data/residual-connections-model.js';
const root = 'public/learn-assets/residual-connections';
const packet = 'docs/teaching/drafts/residual-connections-skip-connections';
const evidence = 'docs/teaching/evidence/residual-connections-author.json';
const paths = ['src/learn/data/topics/residual-connections-skip-connections.jsx', 'src/learn/components/lesson-labs/ResidualConnectionsLabs.jsx', 'src/learn/components/lesson-labs/residual-connections.css', 'src/learn/data/residual-connections-model.js', 'src/learn/data/curriculum/blueprints/residual-connections-skip-connections.js', `${root}/residual-experiments.py`, `${root}/calculated-inputs.json`, `${root}/browser-measurements.json`, 'scripts/build-residual-connections-lesson.mjs', 'scripts/verify-residual-connections.mjs'];
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const checks = [];
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync(evidence, JSON.stringify({ passed: false, status: 'running', started: new Date().toISOString() }));
const check = (name, fn) => { fn(); checks.push(name); };
function near(actual, expected, tolerance = 1e-10) {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); actual.forEach((value, i) => near(value, expected[i], tolerance)); }
  else assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
}
const recorded = JSON.parse(fs.readFileSync(`${root}/calculated-inputs.json`)), mechanics = recorded.mechanisms;
check('Both JSX sources parse; blueprint/model parse', () => paths.filter(path => /\.[jm]sx?$/.test(path)).forEach(path => parse(fs.readFileSync(path, 'utf8'), { sourceType: 'module', plugins: ['jsx'] })));
check('Complete native re-execution is identical to prepared evidence; program and CSV identity', () => {
  assert.deepEqual(recorded, JSON.parse(fs.readFileSync(`${packet}/calculated-inputs.json`)));
  for (const name of ['residual-experiments.py', 'digits-400.csv']) assert.equal(hash(`${root}/${name}`), hash(`${packet}/${name}`));
});
check('Original correction, gradients and simultaneous update match float64 autograd', () => {
  const value = correctionModel(correctionFixture()), expected = mechanics.one_update;
  for (const [key, nativeKey] of Object.entries({ correction: 'correction', output: 'output', loss: 'loss', upstream: 'upstream', branchGradient: 'branch_input_gradient', totalGradient: 'total_input_gradient', weightGradient: 'weight_gradient', newWeight: 'new_weight', newOutput: 'new_output', newLoss: 'new_loss' })) near(value[key], expected[nativeKey]);
  const changed = correctionFixture(); changed[0][0] = -0.1;
  near(correctionModel(changed).loss, 0.305); near(correctionModel(changed).output, [1.6, -0.5]);
  changed[0][0] = 0.3; near(correctionModel(changed).loss, 1.105);
});
check('Weight and input derivatives pass centered differences away from fixture', () => {
  const epsilon = 1e-6;
  for (const weight of [[[0.3, -0.7], [0.9, 0.2]], [[-1, 1], [1, -1]], [[0, 0], [0, 0]]]) {
    for (const input of [[2, -1], [0.3, 1.4], [0, 0]]) {
      const model = correctionModel(weight, 0.1, input);
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const plus = structuredClone(weight), minus = structuredClone(weight); plus[i][j] += epsilon; minus[i][j] -= epsilon;
        near((correctionModel(plus, 0.1, input).loss - correctionModel(minus, 0.1, input).loss) / (2 * epsilon), model.weightGradient[i][j], 1e-8);
      }
      for (let i = 0; i < 2; i++) {
        const plus = [...input], minus = [...input]; plus[i] += epsilon; minus[i] -= epsilon;
        near((correctionModel(weight, 0.1, plus).loss - correctionModel(weight, 0.1, minus).loss) / (2 * epsilon), model.totalGradient[i], 1e-8);
      }
    }
  }
});
check('Depth extremes, exact cancellations, alternating sign and every intermediate', () => {
  for (const fixture of mechanics.scalar_derivatives) near(scalarStack(fixture.branch_slope, 10).gain, fixture.ten_block_derivative);
  for (const slope of [-2, -1.71, -1, -0.5, 0, 0.132, 1]) for (const depth of [1, 2, 9, 20]) {
    let product = 1; const model = scalarStack(slope, depth);
    for (let i = 1; i <= depth; i++) { product *= 1 + slope; near(model.trace[i].value, product); }
  }
  assert.throws(() => scalarStack(0, 0)); assert.throws(() => scalarStack(0, 2.5));
});
check('Operator placement matches native Jacobians and changed-direction differences', () => {
  near(operatorPlacement('post-relu').output, mechanics.zero_branch_order.post_relu_output);
  near(operatorPlacement('inside').jacobian, mechanics.zero_branch_order.pure_jacobian);
  near(operatorPlacement('post-ln').output, mechanics.zero_branch_norm.post_output);
  near(operatorPlacement('post-ln').jacobian, mechanics.zero_branch_norm.post_jacobian);
  for (const shift of [-3, -0.31, 0, 1.27, 3]) {
    near(operatorPlacement('post-ln', shift).output, mechanics.zero_branch_norm.post_output);
    near(operatorPlacement('post-ln', shift).shiftDirection, [0, 0]);
  }
  near(operatorPlacement('post-relu', 0, true).output, [1, 2]);
});
check('Projection shape/nulls and changed input/backward fixtures', () => {
  near(projectionModel([1, 1]).skip, mechanics.projection.skip_output);
  near(projectionModel([1, 1]).gradient, mechanics.projection.skip_gradient);
  near(projectionModel([1, 1], true).skip, mechanics.changed_projection.output);
  near(projectionModel([1, 1], true).gradient, mechanics.changed_projection.skip_gradient);
  assert.equal(projectionModel([1, 1], false, false).output, null);
  near(projectionModel([0, 0]).skip, [2, -1, 0]);
  near(projectionModel([1, 1], false, true, true).correction, [0.5, 0.5, 0.5]);
});
check('Scalar and per-feature gate mechanisms; actual frozen-branch step', () => {
  for (const native of mechanics.gates) {
    const value = gateModel(native.scale);
    near(value.output, native.output); near(value.scaleGradient, native.scale_gradient); near(value.branchGradient, native.branch_weight_gradient); near(value.nextScale, native.scale_after_sgd_0_1);
  }
  near(gateModel(0, 'channel').scaleGradient, [0, 1.4]);
  near(gateModel(0, 'channel', 0.4).nextScale, [-0.04, -0.14]);
  near(gateModel(0, 'scalar', 0.4).nextScale, -0.18);
  assert.notDeepEqual(gateModel(0, 'channel', 0.4).nextOutput, gateModel(0, 'scalar', 0.4).nextOutput);
});
check('Nonlinear counterexample, zero null and Euler native endpoints', () => {
  near(nonlinearPaths(1).actual, mechanics.path_expansion.actual); near(nonlinearPaths(1).invalid, mechanics.path_expansion.invalid_distributed);
  near(nonlinearPaths(0).actual, 0); near(nonlinearPaths(0).invalid, 0);
  for (const point of mechanics.euler) near(scalarStack(-point.step, 10).gain, point.after_ten);
  near(scalarStack(-2, 9).gain, -1); near(scalarStack(0, 10).gain, 1);
});
check('All 39 fits, strict split, actual baselines, movements, shape counts and omission records', () => {
  assert.equal(recorded.fits.length, 39); assert.equal(recorded.training_source_ids.length, 280); assert.equal(recorded.validation_source_ids.length, 120);
  assert.equal(new Set([...recorded.training_source_ids, ...recorded.validation_source_ids]).size, 400);
  for (const fit of recorded.fits) {
    assert.deepEqual(fit.trace.map(row => row.step), [0, 1, 25, 100, 250]);
    assert.equal(fit.trainable_count, 2410 + fit.depth * 2176 + (fit.mode === 'rezero' ? fit.depth : 0));
    if (fit.mode === 'rezero') {
      const base = recorded.fits.find(row => row.depth === 0 && row.seed === fit.seed);
      assert.deepEqual(fit.trace[0].training, base.trace[0].training);
      Object.entries(fit.parameter_displacements).filter(([name]) => name.startsWith('blocks.') && /(?:lower|upper)\.weight$/.test(name)).forEach(([, value]) => assert.ok(value > 0));
    }
    assert.equal(fit.validation_block_ablations.length, fit.mode === 'plain' ? 0 : fit.depth);
  }
  const fit = recorded.fits.find(row => row.seed === 1 && row.depth === 6 && row.mode === 'residual');
  assert.equal(fit.trace.at(-1).validation.correct, 116);
  assert.equal(Math.min(...fit.validation_block_ablations.map(row => row.correct)), 28);
  assert.equal(Math.max(...fit.validation_block_ablations.map(row => row.correct)), 114);
});
check('Browser measurements contain actual fits and attributed training pixels', () => {
  const browser = JSON.parse(fs.readFileSync(`${root}/browser-measurements.json`)); assert.deepEqual(browser.fits, recorded.fits);
  const csv = fs.readFileSync(`${root}/digits-400.csv`, 'utf8').trim().split(/\r?\n/).slice(1).map(line => line.split(',').map(Number));
  for (const specimen of browser.specimens) { const row = csv.find(row => row[0] === specimen.source_id); assert.ok(recorded.training_source_ids.includes(specimen.source_id)); assert.equal(specimen.digit, row.at(-1)); assert.deepEqual(specimen.pixels, row.slice(1, 65)); }
});
check('Complete section/practice/visual ownership and no unsafe broad SVG geometry', () => {
  const lesson = fs.readFileSync(paths[0], 'utf8'), css = fs.readFileSync(paths[2], 'utf8');
  assert.equal((lesson.match(/<H2>/g) || []).length, 13); assert.equal((lesson.match(/<H3>/g) || []).length, 7);
  assert.equal((lesson.match(/<details>/g) || []).length, 16); assert.equal((lesson.match(/<\/details>/g) || []).length, 16);
  for (const name of ['CorrectionLab', 'GradientFigure', 'DepthLab', 'OrderLab', 'ProjectionLab', 'OpeningLab', 'EvidenceLab', 'PathsFigure', 'DenoisingFigure', 'EulerLab', 'MemoryFigure', 'Program']) assert.ok(lesson.includes(`<Residual${name} />`), name);
  assert.ok(!/\bsvg\s*\{/.test(css)); assert.ok(!lesson.includes('\\('));
});
const result = { passed: true, checkedAt: new Date().toISOString(), groups: checks, nativeReplay: 'All 39 CPU fits were rerun separately; the full decoded record matches revision-3 evidence exactly.', versions: recorded.versions, sourceHashes: Object.fromEntries(paths.map(path => [path, hash(path)])), browserStatus: 'Pending coordinated production build and parent browser checks; these author checks do not certify rendering.' };
fs.writeFileSync(evidence, `${JSON.stringify(result, null, 2)}\n`);
console.log(`Residual author checks passed: ${checks.length} substantive groups. ${evidence}`);
