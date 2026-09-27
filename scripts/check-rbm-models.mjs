import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { binaryStates, rbmDefault, tinyDistribution, hiddenEnumeration, hiddenProbabilities, statistics, updateTiny, tinyTransition, probabilityFlow, tinyDraw, particleTrace, persistenceTrace, completion, rbmMetrics, exactSample } from '../src/learn/data/rbm-models.js';
const id = 'boltzmann-machines-restricted-boltzmann-machines-rbm';
const root = 'docs/teaching/deep-learning-completion/' + id;
const native = JSON.parse(fs.readFileSync(root + '/native-fixtures.json'));
const recorded = JSON.parse(fs.readFileSync('docs/teaching/drafts/' + id + '/calculated-inputs.json'));
const checks = [];
function near(name, actual, expected, tolerance = 1e-10) {
  const a = Array.isArray(actual) ? actual.flat(Infinity) : [actual],
    b = Array.isArray(expected) ? expected.flat(Infinity) : [expected];
  assert.equal(a.length, b.length, name + ' length');
  const error = Math.max(0, ...a.map((x, i) => Math.abs(x - b[i])));
  assert.ok(Number.isFinite(error) && error <= tolerance, name + ': ' + error);
  checks.push({
    name,
    passed: true,
    maxAbsoluteError: error,
    tolerance
  });
}
const base = rbmDefault(),
  distribution = tinyDistribution(base),
  stats = statistics(base, [0, 0, 0, 1]);
near('All original visible probabilities', distribution.p, native.exact.visible_probability);
near('All eight joint masses through normalized probability', distribution.joint.map(x => x.probability), native.exact.joint_mass.flat().map(x => x / native.exact.z));
near('Marginals and hidden posterior marginal', [...distribution.marginals, distribution.hidden], [.7, .7, .8]);
near('Five old-state gradient coordinates', stats.gradient, native.exact.gradient.flat(Infinity));
near('Updated data log likelihood', statistics(updateTiny(base, stats.gradient, .1), [0, 0, 0, 1]).logLikelihood, native.exact.updated_log_probability);
near('Full alternating Gibbs transition', tinyTransition(base), native.exact.transition);
const flow = probabilityFlow(base, [0, 0, 0, 1], 10);
for (const row of native.exact.traces) {
  near('Exact mass at step ' + row.step, flow.traces[row.step].mass, row.distribution);
  near('Total variation at step ' + row.step, flow.traces[row.step].tv, row.total_variation);
}
for (const test of native.tiny) {
  near('Asymmetric/zero/extreme bounded native distribution ' + test.model.w, tinyDistribution(test.model).p, test.probabilities);
  near('Asymmetric/zero/extreme expected statistics ' + test.model.w, statistics(test.model, test.counts).gradient, test.gradient);
  const stationary = tinyDistribution(test.model).p;
  near('Stationarity for altered model ' + test.model.w, probabilityFlow(test.model, stationary, 10).traces.at(-1).mass, stationary);
  for (const offset of [-100, 100]) near('Energy offset null ' + offset + ' ' + test.model.w, tinyDistribution(test.model, offset).p, stationary);
}
near('a1=ln2 normalized probability competition', tinyDistribution({
  ...base,
  a: [Math.log(2), 0]
}).p, [1 / 17, 2 / 17, 4 / 17, 10 / 17]);
near('Model-as-data zero gradient', statistics(base, distribution.p).gradient, [0, 0, 0, 0, 0]);
near('Zero learning rate unchanged', updateTiny(base, stats.gradient, 0).w, base.w);
near('One missing pixel posterior', completion([1, 0], [true, false], base).probabilities, [1, 5 / 7]);
near('Unobserved placeholder null', completion([1, 1], [true, false], base).probabilities, [1, 5 / 7]);
assert.throws(() => statistics(base, [0, 0, 0, 0]), /positive total/);
assert.throws(() => binaryStates(11), /0–10/);
assert.deepEqual(tinyDraw([1, 0], base, [.8, .3, .6]).next, [1, 0]);
assert.equal(tinyDraw([0, 0], base, [.5, .5, .5]).hidden, 0);
assert.deepEqual(particleTrace([1, 1], base, 20, 31), particleTrace([1, 1], base, 20, 31));
const batches = [[[1, 1], [1, 1], [1, 0], [0, 1]], [[0, 0], [0, 1], [1, 0], [1, 0]]],
  persistence = persistenceTrace(base, batches, 31);
assert.deepEqual(persistence[1].old, persistence[0].pcd.map(row => row.next));
assert.deepEqual(persistence[1].cd.map(row => row.previous), batches[1]);
checks.push({
  name: 'Zero-data, enumeration bound, hand random draw, threshold equality, seed replay and retained-particle provenance',
  passed: true
});
const images = JSON.parse(fs.readFileSync('public/learn-code/' + id + '/assessment-images.json'));
assert.equal(images.length, 80);
assert.deepEqual(images.map(r => r.id), recorded.protocol.roles.assessment);
const csv = fs.readFileSync('docs/teaching/drafts/' + id + '/digits-400.csv', 'utf8').trim().split(/\r?\n/).slice(1).map(r => r.split(',').map(Number));
for (const image of images) {
  const original = csv.find(r => r[0] === image.id);
  assert.equal(image.digit, original[65]);
  assert.deepEqual(image.pixels, original.slice(1, 65).map(x => Number(x >= 8)));
}
checks.push({
  name: 'All80 exported real images retain original ID, label and every thresholded pixel',
  passed: true
});
for (const test of native.cases) {
  const payload = JSON.parse(fs.readFileSync('public/learn-code/' + id + '/model-' + test.model + '.json')),
    model = payload.model;
  near(test.model + ' ' + test.kind + ' completion', completion(test.visible, test.observed, model).probabilities, test.completion);
  near(test.model + ' ' + test.kind + ' hidden conditional', hiddenProbabilities(test.visible, model), test.hidden);
  const metrics = rbmMetrics(test.visible, model);
  near(test.model + ' ' + test.kind + ' full-input NLL', metrics.nll, test.metrics.per_image_nll[0]);
  near(test.model + ' ' + test.kind + ' reconstruction MSE', metrics.mse, test.metrics.mean_reconstruction_mse);
}
for (const fit of recorded.fits) {
  const key = fit.method + '-' + fit.seed,
    payload = JSON.parse(fs.readFileSync('public/learn-code/' + id + '/model-' + key + '.json'));
  const model = payload.model,
    prepared = hiddenEnumeration(model),
    mask = Array.from({
      length: 64
    }, (_, i) => i % 8 < 4);
  const all = images.map(image => completion(image.pixels, mask, model, prepared).probabilities);
  near(key + ' every stored assessment completion', all, fit.assessment_completion);
  assert.deepEqual(payload.samples, fit.exact_samples);
  assert.deepEqual(payload.hiddenSamples, fit.hidden_sample_states);
  const conditional = exactSample(model, 19, 8, images[0].pixels, mask);
  assert.ok(conditional.every(sample => sample.pixels.every((bit, i) => (bit === 0 || bit === 1) && (!mask[i] || bit === images[0].pixels[i]))));
  const clamped = exactSample(model, 19, 8, images[0].pixels, Array(64).fill(true));
  assert.ok(clamped.every(sample => JSON.stringify(sample.pixels) === JSON.stringify(images[0].pixels)));
}
for (const [i, row] of native.library.entries()) {
  near('Library bridge matched hidden response ' + i, binaryStates(2).map(v => hiddenProbabilities(v, row.model)[0]), row.hidden);
  near('Library bridge exact normalized probabilities ' + i, tinyDistribution(row.model).p, row.probabilities);
}
for (const file of ['rbm-study.py', 'bernoulli_rbm_bridge.py', 'calculated-inputs.json', 'digits-400.csv', 'data-provenance.md']) assert.deepEqual(fs.readFileSync('public/learn-code/' + id + '/' + file), fs.readFileSync('docs/teaching/drafts/' + id + '/' + file));
checks.push({
  name: 'Every public reproduction file is byte-identical; all9 sample galleries and clamped conditional sample contracts preserved',
  passed: true
});
for (const file of ['RbmLabs', 'RbmDigitLabs', 'RbmElements', 'RbmStructureFigures']) parse(fs.readFileSync('src/learn/components/lesson-labs/' + file + '.jsx', 'utf8'), {
  sourceType: 'module',
  plugins: ['jsx']
});
parse(fs.readFileSync('src/learn/data/topics/' + id + '.jsx', 'utf8'), {
  sourceType: 'module',
  plugins: ['jsx']
});
checks.push({
  name: 'Complete generated article and all4 topic components parse',
  passed: true
});
fs.writeFileSync(root + '/model-checks.json', JSON.stringify({
  topicId: id,
  passed: true,
  checks,
  limits: ['Actual scalar/array calculations checked; painted browser interactions and integration belong to root.', 'All recorded fits reused; no new benchmark fit or claimed GPU performance.']
}, null, 2) + '\n');
console.log(checks.length + ' RBM browser-model, native-reference, source and JSX checks passed.');
