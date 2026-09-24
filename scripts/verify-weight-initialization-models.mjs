import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
import { activationMoments, directionalGeometry, gatedIdentity, idealSignal, symmetryState, widthConfiguration } from '../src/learn/data/weight-initialization-models.js';
const path = 'docs/teaching/evidence/weight-initialization-author.json';
writeFileSync(path, JSON.stringify({ passed: false, status: 'running' }));
const checks = [], check = (name, action) => { action(); checks.push(name); };
const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} ≠ ${expected}`);
const packet = JSON.parse(readFileSync('docs/teaching/drafts/weight-initialization-xavier-kaiming-p/calculated-inputs.json'));
const data = JSON.parse(readFileSync('src/learn/data/weight-initialization-measurements.json'));
check('Topic and lab JSX parse; complete11section manuscript is present', () => {
  for (const file of ['src/learn/data/topics/weight-initialization-xavier-kaiming-p.jsx', 'src/learn/components/lesson-labs/WeightInitializationLabs.jsx']) parse(readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  const body = readFileSync('src/learn/data/topics/weight-initialization-xavier-kaiming-p.jsx', 'utf8');
  assert.equal((body.match(/<H2>/g) || []).length, 11);
  assert.equal((body.match(/<details>/g) || []).length, 14);
});
check('Every compact empirical value is preserved from the native packet', () => {
  // JSON.stringify normalizes IEEE negative zero; its mathematical value is unchanged.
  assert.equal(JSON.stringify(data.fixtures), JSON.stringify(packet.fixtures)); assert.deepEqual(data.digitFits, packet.digit_fits); assert.deepEqual(data.widthFits, packet.width_fits);
  assert.equal(data.propagation.length, packet.propagation.records.length);
  data.propagation.forEach((row, i) => row.layers.forEach((layer, j) => { assert.equal(layer.q, packet.propagation.records[i].layers[j].second_moment); assert.equal(layer.g, packet.propagation.records[i].layers[j].gradient_rms); }));
});
check('Live moments match both native fixtures and identity/constant/null cases', () => {
  for (const [values, key] of [[[-2, -1, 1, 2], 'relu'], [[-3, -1, 1, 3], 'changed_relu']]) {
    const result = activationMoments(values);
    for (const [position, field] of [['before', 'input'], ['after', 'output']]) { close(result[position].mean, packet.fixtures[`${key}_${field}`].mean); close(result[position].secondMoment, packet.fixtures[`${key}_${field}`].second_moment); close(result[position].variance, packet.fixtures[`${key}_${field}`].variance); }
    assert.deepEqual(activationMoments(values, 'identity').before, activationMoments(values, 'identity').after);
  }
  close(activationMoments([4, 4, 4, 4]).after.variance, 0); close(activationMoments([-1, -1, -1, -1]).after.secondMoment, 0);
});
check('Directional norm/identity/zero/gated fixtures and equal average squared gains', () => {
  for (const row of packet.fixtures.geometry_interventions) { const result = directionalGeometry(row.smaller_gain, [0, 1], 5); close(result.larger, row.larger_gain); close(result.normGain, row.five_layer_smaller_gain); close(result.averageSquaredGain, 1); }
  close(directionalGeometry(1, [1.1, -2], 8).normGain, 1); assert.equal(directionalGeometry(.2, [0, 0]).normGain, null);
  assert.deepEqual(gatedIdentity([-1, 1], [1, 1]).diagonal, [0, Math.SQRT2]); assert.equal(gatedIdentity([0, 1], [1, 1]).atKink, true);
  for (let s = 1; s <= 20; s++) { const result = directionalGeometry(s / 20, [1, -1], 8); assert.ok(Number.isFinite(result.normGain)); close(result.averageSquaredGain, 1); }
});
check('Symmetry gradients match native autograd; changed-input derivatives pass independent finite differences', () => {
  for (const row of packet.fixtures.symmetry) { const result = symmetryState(row.weight, [.3, .3]); close(result.output, row.prediction); result.hiddenGradient.forEach((value, i) => close(value, row.weight_gradient[i])); }
  const zero = symmetryState([.1, .3], [0, 0]); zero.headGradient.forEach((value, i) => close(value, packet.fixtures.zero_readout.readout_gradient[i])); assert.deepEqual(zero.hiddenGradient.map(Math.abs), [0, 0]);
  for (const input of [-1, 0, 1.7]) for (const target of [-.5, 1]) { const weights = [.2, -.6], outgoing = [.7, -.4], result = symmetryState(weights, outgoing, input, target), h = 1e-6; for (let i = 0; i < 2; i++) { const changed = (row, delta) => row.map((value, j) => value + (i === j ? delta : 0)); close(result.hiddenGradient[i], (symmetryState(changed(weights, h), outgoing, input, target).loss - symmetryState(changed(weights, -h), outgoing, input, target).loss) / (2 * h), 1e-8); close(result.headGradient[i], (symmetryState(weights, changed(outgoing, h), input, target).loss - symmetryState(weights, changed(outgoing, -h), input, target).loss) / (2 * h), 1e-8); } }
});
check('Width formulas match native fixtures including zero-rate and exact base-width null', () => {
  for (const fixture of packet.fixtures.width_checks) { const result = widthConfiguration(fixture.width, fixture.base_rate); close(result.inputRate, fixture.input_rate); close(result.hiddenRate, fixture.hidden_rate); close(result.readoutRate, fixture.readout_rate); close(result.readoutMultiplier, fixture.readout_input_multiplier); close(result.readoutStd, fixture.raw_readout_std); }
  assert.deepEqual(widthConfiguration(32, .003, 'mu'), widthConfiguration(32, .003, 'standard')); close(widthConfiguration(512, 0).hiddenRate, 0);
  close(idealSignal(.02, 30).ratio, 1); close(idealSignal(0, 30).ratio, 0); close(idealSignal(.01, 20).ratio, 2 ** -20);
});
check('Recorded data split is400distinct observations with no train/validation overlap', () => {
  assert.equal(packet.training_source_ids.length, 280); assert.equal(packet.validation_source_ids.length, 120); assert.equal(new Set([...packet.training_source_ids, ...packet.validation_source_ids]).size, 400);
  data.specimens.forEach(sample => { assert.equal(sample.pixels.length, 64); sample.pixels.forEach(pixel => assert.ok(Number.isInteger(pixel) && pixel >= 0 && pixel <= 16)); });
});
const files = ['src/learn/data/topics/weight-initialization-xavier-kaiming-p.jsx', 'src/learn/data/weight-initialization-models.js', 'src/learn/data/weight-initialization-measurements.json', 'src/learn/components/lesson-labs/WeightInitializationLabs.jsx', 'src/learn/components/lesson-labs/weight-initialization.css', 'scripts/generate-weight-initialization-lesson.mjs'];
writeFileSync(path, JSON.stringify({ passed: true, checks, sources: Object.fromEntries(files.map(file => [file, createHash('sha256').update(readFileSync(file)).digest('hex')])), limits: 'Author model/native-packet/source checks; browser integration and independent review are separate.' }, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, substantiveGroups: checks.length, checks }));
