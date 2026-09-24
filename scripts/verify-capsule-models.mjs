import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { parse } from '@babel/parser';
import { capsuleVotes, emVotes, capsuleIndex, routeCapsules, squashProbe, squash, norm, softmax, diagonalCapsuleEM, shiftCapsuleImage, loadCapsuleWeights, encodeCapsuleImage, decodeCapsules } from '../src/learn/data/capsule-models.js';
const report = 'docs/teaching/evidence/capsule-author.json', assets = 'public/learn-assets/capsule-networks/';
writeFileSync(report, JSON.stringify({ passed: false, status: 'running' }));
const groups = [], check = (name, action) => { action(); groups.push(name); }, close = (actual, expected, tolerance = 1e-11) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
function arrays(actual, expected, tolerance = 1e-11) { assert.equal(actual.length, expected.length); actual.forEach((value, i) => Array.isArray(value) ? arrays(value, expected[i], tolerance) : close(value, expected[i], tolerance)); }
const mechanics = JSON.parse(readFileSync(assets + 'mechanics-results.json')), native = JSON.parse(readFileSync('docs/teaching/evidence/capsule-native-inference.json'));
check('Complete11section source and lab JSX parse;17 closed manuscript disclosures and full implementation/reference flow retained', () => {
  for (const file of ['src/learn/data/topics/capsule-networks.jsx', 'src/learn/components/lesson-labs/CapsuleLabs.jsx']) parse(readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  const source = readFileSync('src/learn/data/topics/capsule-networks.jsx', 'utf8'); assert.equal((source.match(/<H2>/g) || []).length, 11); assert.equal((source.match(/<details>/g) || []).length, 17); assert.match(source, /Follow routing all the way into a trainable program/); assert.match(source, /Capsule Routing via Variational Bayes/);
});
check('All256 grouping coordinates and both batches preserve channel, location, type and coordinate identity', () => {
  const seen = new Set();
  for (let row = 0; row < 4; row++) for (let column = 0; column < 4; column++) for (let type = 0; type < 4; type++) for (let coordinate = 0; coordinate < 4; coordinate++) { const item = capsuleIndex(row, column, type, coordinate); seen.add(`${item.child},${coordinate}`); assert.equal(Math.floor(item.channel / 4), type); assert.equal(item.child % 4, type); assert.equal(Math.floor(item.child / 4) % 4, column); assert.equal(Math.floor(item.child / 16), row); for (const batch of [0, 1]) assert.equal(batch * 256 + item.channel * 16 + row * 4 + column, batch * 256 + (type * 4 + coordinate) * 16 + row * 4 + column); }
  assert.equal(seen.size, 256); assert.equal(capsuleIndex(1, 2, 3, 0).child, 27); assert.equal(capsuleIndex(2, 1, 0, 0).child, 36);
});
check('Allrouting trace fields match independent NumPy fixtures; zero, symmetry, permutation, rotation and temperature boundaries preserve the intended contract', () => {
  const opposite = structuredClone(capsuleVotes); opposite[1][0] = [-2, 0];
  for (const [input, key, steps] of [[capsuleVotes, 'routing', 8], [opposite, 'opposing_edit', 3], [capsuleVotes.map(row => row.map(() => [0, 0])), 'zero_votes', 3], [capsuleVotes.map(row => row.map(() => [1, 0])), 'identical_parents', 3]]) routeCapsules(input, steps).forEach((result, i) => { for (const field of ['logits', 'coupling', 'sums', 'outputs', 'lengths', 'agreement']) arrays(result[field], mechanics[key][i][field]); close(result.entropy, mechanics[key][i].mean_row_entropy); result.coupling.forEach(row => close(row.reduce((sum, value) => sum + value, 0), 1)); });
  const baseline = routeCapsules(capsuleVotes).at(-1), rotated = routeCapsules(capsuleVotes.map(row => row.map(([x, y]) => [-y, x]))).at(-1);
  arrays(rotated.lengths, baseline.lengths); arrays(rotated.coupling, baseline.coupling); arrays(rotated.outputs, baseline.outputs.map(([x, y]) => [-y, x])); arrays(routeCapsules([capsuleVotes[2], capsuleVotes[0], capsuleVotes[1]]).at(-1).outputs, baseline.outputs);
  for (const children of [2, 4]) close(routeCapsules(Array.from({ length: children }, () => [[1, 0], [1, 0]]), 8).at(-1).lengths[0], children === 2 ? .5 : .8);
  for (const temperature of [.5, 1, 2]) arrays(softmax([0, 2], temperature), [1 / (1 + Math.exp(2 / temperature)), 1 / (1 + Math.exp(-2 / temperature))]); assert.throws(() => softmax([0, 1], 0)); assert.throws(() => routeCapsules(capsuleVotes, 0));
});
check('Actual finite perturbations match independent formula evaluations; radial/tangent derivatives and zero behavior agree with native probes', () => {
  for (const fixture of mechanics.squash_probes) { const result = squashProbe(fixture.input, .017, 'radial'); close(result.radial, fixture.radial_gradient); close(result.tangent, fixture.tangential_gradient); arrays(result.output, fixture.output); for (const direction of ['radial', 'tangent']) { const point = squashProbe(fixture.input, .017, direction); close(point.change, norm(squash(point.changed).map((value, i) => value - squash(fixture.input)[i]))); close(squashProbe(fixture.input, 0, direction).change, 0); } }
});
check('Every EM sufficient statistic and responsibility matches native; inactive and absent evidence have honest distinct outcomes', () => {
  diagonalCapsuleEM(emVotes, [1, 1, .5]).forEach((row, i) => { const expected = mechanics.em.history[i]; for (const [field, nativeField] of [['mass', 'mass'], ['means', 'means'], ['variance', 'variance'], ['effective', 'effective_weights'], ['parentActivation', 'activation'], ['responsibility', 'responsibility_in'], ['next', 'responsibility_out']]) arrays(row[field], expected[nativeField]); });
  const edited = structuredClone(emVotes); edited[2] = [[-4, 4], [4, -4]]; arrays(diagonalCapsuleEM(edited, [1, 1, 0]).at(-1).means, diagonalCapsuleEM(emVotes, [1, 1, 0]).at(-1).means); assert.deepEqual(diagonalCapsuleEM(emVotes, [0, 0, 0]), []);
  const tiny = diagonalCapsuleEM(emVotes, [0, 0, 1e-20]);
  arrays(tiny[0].mass, [5e-21, 5e-21], 1e-35);
  arrays(tiny[0].means, [[1e-8, 0], [1.6e-8, 0]], 1e-23);
  tiny.forEach(row => row.mass.forEach(mass => assert.ok(mass > 0 && mass <= 1e-12)));
  arrays(diagonalCapsuleEM(emVotes, [0, 0, 2e-12])[0].mass, [1e-12, 1e-12], 1e-26);
});
const bytes = readFileSync(assets + 'frozen-model.f32'), metadata = JSON.parse(readFileSync(assets + 'frozen-model.json'));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), state = loadCapsuleWeights(buffer, metadata);
let largestError = 0;
check('Exact binary weights and all14 changed-image encodes/decodes match nativefloat64; latent mask null and corruption contracts pass', () => {
  assert.equal(createHash('sha256').update(bytes).digest('hex'), metadata.sha256); assert.equal(Object.values(state).reduce((sum, item) => sum + item.length, 0), 47184);
  for (const fixture of native) { const result = encodeCapsuleImage(fixture.image, state); result.capsules.forEach((row, i) => row.forEach((value, j) => { largestError = Math.max(largestError, Math.abs(value - fixture.capsules[i][j])); })); arrays(result.capsules, fixture.capsules, 1e-12); arrays(result.reconstruction, fixture.reconstruction, 1e-12); assert.equal(result.predicted, fixture.predicted); if (fixture.kind.startsWith('shift-')) { const [dy, dx] = fixture.kind.slice(6).split(',').map(Number); arrays(shiftCapsuleImage(metadata.examples[fixture.specimen].image, dy, dx), fixture.image, 0); } }
  const result = encodeCapsuleImage(metadata.examples[0].image, state), changed = structuredClone(result.capsules); changed[5][0] += .137; arrays(decodeCapsules(changed, 4, state), decodeCapsules(result.capsules, 4, state), 0); changed[4][2] -= .137; assert.notDeepEqual(decodeCapsules(changed, 4, state), result.reconstruction);
  assert.throws(() => loadCapsuleWeights(buffer.slice(0, 100), metadata)); assert.throws(() => loadCapsuleWeights(buffer, { ...metadata, model: 'other' }));
});
check('Every compact figure value retains its immutable run identity and same-count/different-prediction contrast', () => {
  const compact = JSON.parse(readFileSync('src/learn/data/capsule-measurements.json')), full = JSON.parse(readFileSync(assets + 'calculated-inputs.json'));
  compact.runs.forEach((row, i) => { assert.deepEqual(row.trajectory, full.runs[i].trajectory); for (const [count, result] of Object.entries(row.inference_iterations)) { const { lengths, ...expected } = full.runs[i].inference_iterations[count]; assert.deepEqual(result, expected); } });
  const row = compact.runs.find(item => item.seed === 3 && item.training_iterations === 1); assert.equal(row.inference_iterations['1'].correct, row.inference_iterations['2'].correct); assert.equal(row.inference_iterations['1'].predictions.filter((value, i) => value !== row.inference_iterations['2'].predictions[i]).length, 1);
});
const durations = []; for (let i = 0; i < 40; i++) { const start = performance.now(); encodeCapsuleImage(metadata.examples[i % 2].image, state); durations.push(performance.now() - start); } durations.sort((a, b) => a - b);
const files = ['src/learn/data/topics/capsule-networks.jsx', 'src/learn/data/capsule-models.js', 'src/learn/data/capsule-measurements.json', 'src/learn/components/lesson-labs/CapsuleLabs.jsx', 'src/learn/components/lesson-labs/capsules.css', 'scripts/generate-capsule-lesson.mjs', assets + 'frozen-model.f32', assets + 'frozen-model.json'];
writeFileSync(report, JSON.stringify({ passed: true, groups, largestNativeDoubleError: largestError, boundedInference: { runtime: process.version, warmedRuns: 40, medianMs: durations[20], p95Ms: durations[38], limit: 'Node timing only; final browser latency measured separately. No training occurs in the reader.' }, sources: Object.fromEntries(files.map(file => [file, createHash('sha256').update(readFileSync(file)).digest('hex')])), limits: 'Author source/model checks; independent review, production browser/perceptual checks remain separate.' }, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, groups: groups.length, largestNativeDoubleError: largestError, medianInferenceMs: durations[20], p95InferenceMs: durations[38] }));
