import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import * as model from '../src/learn/data/depthwise-convolution-models.js';
const id = 'depthwise-separable-dilated-convolutions';
const reportPath = 'docs/teaching/evidence/depthwise-convolution-author.json';
const report = { passed: false, checkedAt: new Date().toISOString(), groups: [], sourceHashes: {} };
fs.writeFileSync(reportPath, JSON.stringify(report));
const group = (name, work) => { work(); report.groups.push(name); };
const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const clone = value => JSON.parse(JSON.stringify(value));
group('Channel arithmetic, locality, true zero null and independent numerical gradients', () => {
  const { inputs, filters, mixing } = clone(model.channelFixture);
  assert.deepEqual(model.channelForward(inputs, filters, mixing).output, [-3, -3]);
  inputs[1][0] = 4;
  assert.deepEqual(model.channelForward(inputs, filters, mixing).output, [-4, -3]);
  assert.deepEqual(model.channelForward(inputs, filters, [0, 0]).output, [0, 0]);
  for (const target of [-3, -.2, 1, 3]) {
    const result = model.channelGradient(inputs, filters, mixing, target, .01);
    const loss = (d, p) => (model.channelForward(inputs, d, p).output[0] - target) ** 2 / 2;
    for (let channel = 0; channel < 2; channel++) {
      const plus = [...mixing], minus = [...mixing]; plus[channel] += 1e-5; minus[channel] -= 1e-5;
      close(result.mixingGradient[channel], (loss(filters, plus) - loss(filters, minus)) / 2e-5, 1e-7);
      for (let tap = 0; tap < 2; tap++) {
        const dp = clone(filters), dm = clone(filters); dp[channel][tap] += 1e-5; dm[channel][tap] -= 1e-5;
        close(result.filterGradient[channel][tap], (loss(dp, mixing) - loss(dm, mixing)) / 2e-5, 1e-7);
      }
    }
  }
  const fixture = model.channelFixture;
  const gradient = rate => model.channelGradient(fixture.inputs, fixture.filters, fixture.mixing, 1, rate);
  close(gradient(.01).after, -1.9824); close(gradient(.01).nextLoss, 4.44735488);
  close(gradient(.1).nextLoss, 8.6528); close(gradient(0).nextLoss, 8);
});
group('Two independent rank probes, noncanonical valid basis, dependent-probe limitation', () => {
  assert.equal(model.rankProbe([[1, 0]], [[1], [0]]).error, 3);
  assert.equal(model.rankProbe([[1, 0]], [[1], [0]], true).error, 0);
  assert.equal(model.rankProbe([[1, 0], [0, 1]], [[1, 0], [0, 1]]).error, 0);
  assert.equal(model.rankProbe([[1, 1], [1, -1]], [[.5, .5], [.5, -.5]]).error, 0);
  assert.ok(model.rankProbe([[1, 0], [0, 1]], [[1, 0], [0, 0]]).error > 0);
});
group('Every stencil center/dilation and finite-map tap count matches independently enumerated addresses', () => {
  for (let center = 0; center < 9; center++) for (let dilation = 1; dilation <= 8; dilation++) {
    const value = model.sampleStencil(model.initialSignal, center, dilation);
    let expected = 0;
    for (let cell = 0; cell < 9; cell++) if ([center - dilation, center, center + dilation].includes(cell)) expected += cell + 1;
    close(value.sum, expected);
  }
  for (let row = 0; row < 8; row++) for (let column = 0; column < 8; column++) for (let dilation = 1; dilation <= 8; dilation++) {
    let count = 0;
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) if (Math.abs(r - row) % dilation === 0 && Math.abs(r - row) <= dilation && Math.abs(c - column) % dilation === 0 && Math.abs(c - column) <= dilation) count++;
    assert.equal(model.finiteTaps(row, column, dilation).length, count);
  }
  close(model.sampleStencil([1, 2, 3, 4, 5, 20, 7, 8, 9], 4, 2).sum, 15);
  close(model.sampleStencil([1, 2, 3, 4, 5, 6, 20, 8, 9], 4, 2).sum, 28);
});
group('Serial support agrees with explicit path enumeration and rate-order null', () => {
  for (let a = 1; a <= 9; a++) for (let b = 1; b <= 9; b++) for (let c = 1; c <= 9; c++) {
    const brute = new Set();
    for (const i of [-1, 0, 1]) for (const j of [-1, 0, 1]) for (const k of [-1, 0, 1]) brute.add(i * a + j * b + k * c);
    assert.deepEqual(model.samplingSupport([a, b, c]).sites, [...brute].sort((x, y) => x - y));
    assert.deepEqual(model.samplingSupport([a, b, c]).sites, model.samplingSupport([c, b, a]).sites);
  }
  assert.deepEqual(model.samplingSupport([1, 4]).holes, [-2, 2]);
  assert.equal(model.samplingSupport([1, 2, 4]).sites.length, 15);
  assert.throws(() => model.samplingSupport([1.5]));
});
group('Parallel branches expose far-cell effects, mean-preserving swap and zero projection', () => {
  const weights = [0, .1, .2, .3, 1];
  assert.deepEqual(model.parallelContext(model.initialSignal, weights).branches, [5, 15, 15, 15, 5]);
  close(model.parallelContext(model.initialSignal, weights).output, 14);
  close(model.parallelContext([2, 1, 3, 4, 5, 6, 7, 8, 9], weights).output, 14.3);
  close(model.parallelContext([1, 2, 3, 4, 5, 6, 7, 8, 20], weights).output, 18.522222222222222);
  close(model.parallelContext(model.initialSignal, [0, 0, 0, 0, 0]).output, 0);
});
group('Saved JavaScript inference matches independent native loops, edited pixels and all four retained ranks', () => {
  const records = JSON.parse(fs.readFileSync(`public/learn-code/${id}/digit-inference.json`));
  const reference = JSON.parse(fs.readFileSync(`public/learn-code/${id}/author-check-results.json`));
  let maxDense = 0;
  for (const run of records.runs) for (const example of run.examples) {
    const saved = reference.saved_model_checks.find(row => row.source_id === example.source_id && row.dilation === run.dilation);
    const dense = model.savedDigitLogits(run, example.input);
    dense.forEach((value, index) => { maxDense = Math.max(maxDense, Math.abs(value - example.dense_logits[index])); close(value, example.dense_logits[index], 3e-5); });
    const edited = clone(example.input); edited[0][0] = 1;
    model.savedDigitLogits(run, edited).forEach((value, index) => close(value, saved.edit.dense_logits[index], 1e-10));
    for (const factor of saved.factorized_outputs) model.savedDigitLogits(run, example.input, factor.multiplier).forEach((value, index) => close(value, factor.logits[index], 1e-10));
    for (let input = 0; input < 8; input++) for (let output = 0; output < 12; output++) model.reconstructedKernel(run, 9, input, output).forEach((row, r) => row.forEach((value, c) => close(value, run.dense_state['spatial.weight'][output][input][r][c], 2e-6)));
  }
  report.maximumDenseFloat32Difference = maxDense;
});
group('Owned JSX parses; full practice/math and deferred source/model ownership are retained', () => {
  for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/DepthwiseConvolutionLabs.jsx']) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  const body = fs.readFileSync(`src/learn/data/topics/${id}.jsx`, 'utf8');
  assert.equal((body.match(/<H2>/g) || []).length, 10);
  assert.ok((body.match(/<MathBlock>/g) || []).length >= 9);
  assert.ok((body.match(/<summary>Solution/g) || []).length >= 8);
  assert.ok(body.includes('The improved '));
  assert.ok(!body.includes('Those improved pages are still prepared'));
  assert.ok(!body.includes('**Investigation:'));
  const labs = fs.readFileSync('src/learn/components/lesson-labs/DepthwiseConvolutionLabs.jsx', 'utf8');
  assert.ok(!labs.includes('import measurements'));
  assert.ok(labs.includes('AbortController'));
  assert.ok(!/prediction.*(?:guess|submit)|commitGuess/i.test(labs));
});
const sources = [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/DepthwiseConvolutionLabs.jsx', 'src/learn/components/lesson-labs/depthwise-convolutions.css', 'src/learn/data/depthwise-convolution-models.js', `src/learn/data/curriculum/blueprints/${id}.js`, 'scripts/generate-depthwise-convolution-lesson.mjs', 'scripts/verify-depthwise-convolution-models.mjs'];
for (const file of sources) report.sourceHashes[file] = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
report.passed = true;
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, groups: report.groups.length, maximumDenseFloat32Difference: report.maximumDenseFloat32Difference }));
