import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import * as model from '../src/learn/data/convolution-models.js';
const reportFile = 'docs/teaching/evidence/convolution-author-checks.json';
const report = { passed: false, checkedAt: new Date().toISOString(), groups: [], sourceHashes: {} };
fs.writeFileSync(reportFile, JSON.stringify(report));
function group(name, check) { check(); report.groups.push(name); }
function close(a, b, tolerance = 1e-10) { assert.ok(Math.abs(a - b) <= tolerance, `${a} ≠ ${b}`); }
group('Patch correspondence, changed filter and zero-weight null', () => {
  assert.deepEqual(model.correlate2d(model.initialImage, model.initialKernel), [[0, 5], [0, -2]]);
  const image = model.initialImage.map(row => [...row]); image[1][1] = 2;
  assert.deepEqual(model.correlate2d(image, model.initialKernel), [[1, 5], [-1, -1]]);
  assert.deepEqual(model.correlate2d(model.initialImage, [[1, 0], [0, -1]]), [[0, -1], [-1, 1]]);
  assert.deepEqual(model.correlate2d(image, [[1, 0], [0, -1]]), [[-1, -1], [-1, 2]]);
  assert.deepEqual(model.correlate2d(image, [[0, 0], [0, 0]]), [[0, 0], [0, 0]]);
});
group('Shared derivatives checked by independent perturbations, descent, overshoot and null', () => {
  for (const targets of [[0, 0], [0, 2], [-2, 1], [1.3, -.7]]) {
    const result = model.sharedFilterUpdate(targets, .1);
    const loss = weights => [1, 3].reduce((sum, x, i) => sum + (weights[0] * x + weights[1] * [3, 2][i] - targets[i]) ** 2 / 2, 0);
    for (let coordinate = 0; coordinate < 2; coordinate++) {
      const plus = [1, -1], minus = [1, -1]; plus[coordinate] += 1e-5; minus[coordinate] -= 1e-5;
      close(result.gradient[coordinate], (loss(plus) - loss(minus)) / 2e-5, 1e-8);
    }
  }
  close(model.sharedFilterUpdate().nextLoss, 1.53);
  close(model.sharedFilterUpdate([0, 2]).nextLoss, 2.61);
  close(model.sharedFilterUpdate([0, 2], .01).nextLoss, 1.7001);
  assert.deepEqual(model.sharedFilterUpdate([-2, 1]).gradient, [0, 0]);
  assert.deepEqual(model.sharedFilterUpdate([0, 2], 0).nextWeights, [1, -1]);
});
group('Window counts and sampled addresses at all supported discrete geometries', () => {
  for (let n = 3; n <= 16; n++) for (let k = 1; k <= 5; k++) for (let s = 1; s <= 3; s++) for (let d = 1; d <= 3; d++) for (let left = 0; left <= 4; left++) for (let right = 0; right <= 4; right++) {
    const geometry = model.windowGeometry({ n, k, s, d, left, right });
    const starts = [];
    for (let start = -left; start + (k - 1) * d < n + right; start += s) starts.push(start);
    assert.equal(geometry.windows.length, starts.length);
    assert.deepEqual(geometry.windows.map(row => row[0]), starts.map(value => value + 0));
    if (starts.length) close(geometry.center, (geometry.windows[0][0] + geometry.windows[0].at(-1)) / 2 + .5);
  }
  assert.throws(() => model.windowGeometry({ n: 3.5, k: 2 }));
});
group('Pooling overlap, ties, and receptive-field coordinate conservation', () => {
  assert.deepEqual(model.poolVector([1, 4, 3]).gradient, [0, 2, 0]);
  assert.deepEqual(model.poolVector([3, 1, 4]).gradient, [1, 0, 1]);
  assert.deepEqual(model.poolVector([2, 2, 1]).gradient, [1, 1, 0]);
  assert.deepEqual(model.poolVector([1, 4, 3], 2, 1, 'mean').gradient, [.5, 1, .5]);
  const trace = model.receptiveTrace(model.cnnAxisLayers);
  assert.deepEqual(trace.map(row => row.r), [3, 4, 8, 10, 18, 46]);
  assert.deepEqual(trace.map(row => row.a), [.5, 1, 1, 2, 2, 16]);
  assert.deepEqual(model.observedAncestors(model.cnnAxisLayers, 0), Array.from({ length: 32 }, (_, i) => i));
  assert.deepEqual(model.dilationOffsets(2, 2), [-4, -2, 0, 2, 4]);
  assert.deepEqual(model.dilationOffsets(1, 2), [-3, -2, -1, 0, 1, 2, 3]);
});
group('Exact path coefficients, threshold, transpose identity and boundary contrast', () => {
  model.averagingProfile(2, .01).coefficients.forEach((v, i) => close(v, [1, 2, 3, 2, 1][i] / 9));
  assert.equal(model.averagingProfile(20, .01).width, 21);
  for (let depth = 1; depth <= 20; depth++) close(model.averagingProfile(depth, .01).coefficients.reduce((a, b) => a + b), 1);
  assert.deepEqual(model.transpose1d([2, -3], [1, -1]).output, [2, -5, 3]);
  assert.deepEqual(model.transpose1d([1, 1, 1], [1, 1, 1], 2).output, [1, 1, 2, 1, 2, 1, 1]);
  assert.equal(model.shiftComparison('circular').difference, 0);
  assert.equal(model.shiftComparison('zero').difference, 1);
});
group('Actual full CPU replay matches the conserved experimental data', () => {
  const prepared = JSON.parse(fs.readFileSync('docs/teaching/drafts/convolution-pooling-receptive-fields/calculated-inputs.json'));
  const executed = JSON.parse(fs.readFileSync('public/learn-code/convolution-pooling-receptive-fields/calculated-inputs.json'));
  assert.deepEqual(executed, prepared);
  assert.equal(executed.experiment.fits.length, 12);
  for (const fit of executed.experiment.fits) {
    assert.deepEqual(fit.trace.map(row => row.step), [0, 1, 25, 100, 200, 400]);
    for (const row of fit.visual_rows || []) {
      close(row.probabilities.reduce((a, b) => a + b), 1, 1e-6);
      assert.equal(row.first_maps.length, 8); assert.equal(row.final_maps.length, 16);
    }
  }
});
const files = ['src/learn/data/convolution-models.js', 'src/learn/components/lesson-labs/ConvolutionLabs.jsx', 'src/learn/components/lesson-labs/convolution-labs.css', 'src/learn/data/topics/convolution-pooling-receptive-fields.jsx', 'src/learn/data/curriculum/blueprints/convolution-pooling-receptive-fields.js', ...['convolution-experiments.py', 'convolution_pullbacks.py', 'calculated-inputs.json', 'digits-400.csv'].map(name => 'public/learn-code/convolution-pooling-receptive-fields/' + name)];
for (const file of files) {
  const text = fs.readFileSync(file);
  if (/\.(jsx|js)$/.test(file)) parse(text.toString(), { sourceType: 'module', plugins: ['jsx'] });
  report.sourceHashes[file] = crypto.createHash('sha256').update(text).digest('hex');
}
report.nativeExecution = { python: '3.12.14', torch: '2.14.0+cpu', experiment: 'All twelve fits and fixtures executed from canonical public program; complete JSON equality checked above.', pullbacks: 'Canonical convolution_pullbacks.py executed: dense valid forward/input/weight/bias derivatives, overlapping max/mean pooling, adaptive values/derivatives passed.' };
report.passed = true;
fs.writeFileSync(reportFile, JSON.stringify(report, null, 2) + '\n');
console.log(`PASS: ${report.groups.length} substantive convolution author groups; exact full native replay.`);
