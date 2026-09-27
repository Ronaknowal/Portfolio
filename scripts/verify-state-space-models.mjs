import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { linearSystem, systemDefault, selectiveMemory, selectionDefault, ssdOperator, ssdDefault, trajectoryForward, maximumDifference, fftConvolution } from '../src/learn/data/state-space-models.js';
const evidence = 'docs/teaching/evidence/state-space-models.json';
fs.writeFileSync(evidence, JSON.stringify({ passed: false, note: 'Verification started; completion not established.' }));
const close = (a, b, tolerance = 1e-10) => assert.ok(maximumDifference(a, b) <= tolerance, `${maximumDifference(a, b)} > ${tolerance}`);
const hashes = paths => Object.fromEntries(paths.map(path => [path, crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')]));
const checks = [];
const base = systemDefault();
close(linearSystem(base).recurrent, [.5625, -.921875, 1.39453125, .4423828125]);
close(linearSystem({ ...base, inputs: [2, 0, 1, 0] }).recurrent, [1.125, .40625, .7890625, .322265625]);
close(linearSystem({ ...base, rates: [0, -2], write: [1, 0], read: [1, 0], direct: 0, interval: .5, initial: [3, 0], inputs: [2, -1] }).recurrent, [4, 3.5]);
close(linearSystem({ ...base, read: [0, 0], direct: 1, initial: [4, -3] }).recurrent, base.inputs);
for (let length = 1; length <= 16; length++) for (const rate of [-3, -1, -1e-12, 0]) {
  const settings = { ...base, rates: [rate, -.21], interval: .05 + length / 10, initial: [1, -2], inputs: Array.from({ length }, (_, i) => Math.sin(i + 1) * 8) };
  const result = linearSystem(settings);
  assert.ok(result.difference < 1e-10);
  const changed = { ...settings, inputs: settings.inputs.map((value, i) => i === length - 1 ? -value : value) };
  close(result.recurrent.slice(0, -1), linearSystem(changed).recurrent.slice(0, -1));
}
close(fftConvolution([1, 2, 3], [4, 5, 6]), [4, 13, 28]);
checks.push('Hand-derived default/worked/singular/direct fixtures; 64 boundary configurations with initial response and independent FFT; causality');
const memory = selectionDefault(), trace = selectiveMemory(memory);
close(trace.map(row => row.state), [2.97, 2.8603, 2.881697, -1.95118303]);
close(trace.map(row => row.fixed), [1.5, -3.25, .875, -.5625]);
close(selectiveMemory({ ...memory, initial: 4, gates: [0, 0, 0, 0] }).map(row => row.state), [4, 4, 4, 4]);
close(selectiveMemory({ ...memory, independent: true, retention: .5, initial: 4, gates: [0, 0, 0, 0] }).map(row => row.state), [2, 1, .5, .25]);
assert.equal(selectiveMemory({ ...memory, marked: [false, false, false, false] })[3].target, null);
checks.push('Selective/constant traces, closed writes versus independent decay, missing target');
const ssd = ssdDefault();
close(ssdOperator(ssd).recurrent, [[2, 1], [4, -.5], [-.25, 2.75], [1, 5.9]]);
for (let length = 1; length <= 8; length++) for (let chunkSize = 1; chunkSize <= 8; chunkSize++) {
  const settings = { chunkSize, initial: [[1, -2], [3, -.5]], decay: Array.from({ length }, (_, t) => t % 3 === 0 ? 0 : .8), write: Array.from({ length }, (_, t) => [Math.sin(t + 1), Math.cos(t)]), read: Array.from({ length }, (_, t) => [t / 2 - 1, 1]), values: Array.from({ length }, (_, t) => [t, -t / 3]) };
  assert.ok(ssdOperator(settings).difference < 1e-10);
}
checks.push('Hand-derived SSD fixture; all 64 length/chunk combinations with zero decays, uneven chunks and nonzero initial matrix');
const resource = JSON.parse(fs.readFileSync('public/learn-code/state-space-models-s4-mamba-mamba-2/trajectory-inference.json'));
const compact = JSON.parse(fs.readFileSync('src/learn/data/state-space-measurements.json'));
const recorded = JSON.parse(fs.readFileSync('public/learn-code/state-space-models-s4-mamba-mamba-2/trajectory-results.json'));
assert.equal(compact.models.length, 4);
for (const shown of compact.models) {
  const original = recorded.models.find(run => run.kind === shown.kind && run.seed === shown.seed);
  assert.deepEqual(shown.history, original.history);
  assert.deepEqual(shown.confusion, original.metrics.test.confusion);
  assert.equal(shown.selected_epoch, original.selected_epoch);
  assert.equal(shown.history.length, 100);
  assert.ok(shown.confusion.every(row => row.reduce((sum, value) => sum + value, 0) === 4));
}
assert.deepEqual(resource.records.map(row => row.id), recorded.roles.validation);
assert.ok(compact.paths.every(row => recorded.roles.fit.includes(row.id)));
checks.push('All four measured learning curves and confusion matrices exactly match retained results; validation and fitting roles preserved');
const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/state-space-native.json'));
assert.equal(native.passed, true);
assert.equal(native.cases.length, 108);
assert.equal(resource.records.length, 50);
let maxLogitError = 0, maxProbabilityError = 0;
const elapsed = [];
for (const fixture of native.cases) {
  let points = resource.records.find(row => row.id === fixture.row).points.map(point => [...point]);
  if (fixture.edit === 'reflect') points[22][0] = 1 - points[22][0];
  if (fixture.edit === 'reverse') points.reverse();
  if (fixture.edit === 'zero') points = points.map(() => [0, 0]);
  if (fixture.edit === 'one') points = points.map(() => [1, 1]);
  const start = performance.now();
  const actual = trajectoryForward(points, resource.models[fixture.kind], fixture.kind);
  elapsed.push(performance.now() - start);
  const error = maximumDifference(actual.logits, fixture.logits);
  maxLogitError = Math.max(maxLogitError, error);
  close(actual.logits, fixture.logits, 1e-4);
  const exp = fixture.logits.map(value => Math.exp(value - Math.max(...fixture.logits)));
  const expected = exp.map(value => value / exp.reduce((a, b) => a + b));
  maxProbabilityError = Math.max(maxProbabilityError, maximumDifference(actual.probabilities, expected));
  close(actual.probabilities, expected, 1e-5);
}
checks.push('108 actual frozen PyTorch inference cases: every validation record, reflected coordinate, reversed order and unit-coordinate extremes for both models');
const lesson = fs.readFileSync('src/learn/data/topics/state-space-models-s4-mamba-mamba-2.jsx', 'utf8');
assert.ok(!/Inline figure:|predict which outputs|record whether you expect|green equality/.test(lesson));
assert.equal((lesson.match(/<StateSpace(?:System|Selection|SSD|Trajectory)Lab/g) || []).length, 4);
assert.equal((lesson.match(/<StateSpaceProgram /g) || []).length, 3);
assert.equal((lesson.match(/<summary>Hint<\/summary>/g) || []).length, 10);
assert.ok(lesson.includes('Mamba-3') && lesson.includes('Woodbury') && lesson.includes('.46875 MiB'));
checks.push('Four live investigations, three full on-demand program routes, ten retained hints, prepared figure/prediction instructions removed');
// Demonstrate that the parity assertion rejects a material fault without mutating source.
assert.throws(() => close([1, 2], [1, 2.01], 1e-4));
const paths = ['src/learn/data/state-space-models.js', 'src/learn/components/lesson-labs/StateSpaceLabs.jsx', 'src/learn/components/lesson-labs/StateSpaceFigures.jsx', 'src/learn/components/lesson-labs/state-space-labs.css', 'src/learn/data/topics/state-space-models-s4-mamba-mamba-2.jsx', 'public/learn-code/state-space-models-s4-mamba-mamba-2/trajectory-inference.json', 'scripts/verify-state-space-models.mjs'];
paths.push('src/learn/data/state-space-measurements.json', 'scripts/generate-state-space-lesson.mjs', 'scripts/prepare-state-space-assets.py', 'public/learn-code/state-space-models-s4-mamba-mamba-2/trajectory-results.json');
const report = { passed: true, checks, nativeCases: 108, maxLogitError, maxProbabilityError, nodeForwardTiming: { medianMs: elapsed.sort((a, b) => a - b)[54], maxMs: Math.max(...elapsed), context: 'Local Node CPU only; not a browser/GPU throughput benchmark' }, sourceHashes: hashes(paths) };
fs.writeFileSync(evidence, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ checks: checks.length, maxLogitError, maxProbabilityError, timing: report.nodeForwardTiming }));
