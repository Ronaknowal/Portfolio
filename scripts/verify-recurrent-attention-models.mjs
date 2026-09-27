import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
import { memoryRead, localAttention, cancellationRead, copyDistribution, encodeAttention, traceAttention } from '../src/learn/data/recurrent-attention-models.js';

const draft = 'docs/teaching/drafts/attention-mechanism-bahdanau-luong/';
const assets = 'public/learn-code/attention-mechanism-bahdanau-luong/';
const evidencePath = 'docs/teaching/evidence/recurrent-attention-models.json';
fs.writeFileSync(evidencePath, JSON.stringify({ passed: false, reason: 'Verification started; success is written only after all assertions pass.' }) + '\n');
const analytic = JSON.parse(fs.readFileSync(draft + 'analytic-results.json'));
const mechanics = JSON.parse(fs.readFileSync(draft + 'mechanics-results.json'));
const checks = [];
let maximumTraceError = 0;
function close(actual, expected, tolerance = 1e-10) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length);
    actual.forEach((value, index) => close(value, expected[index], tolerance));
  } else {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
  }
}
for (const fixture of Object.values(analytic.fixtures)) {
  const result = memoryRead(fixture.query, fixture.keys, fixture.values, fixture.valid, fixture.rate);
  for (const key of ['scores', 'attention', 'context', 'loss']) close(result[key], fixture[key]);
  for (const [current, saved] of [['probabilities', 'class_probabilities'], ['scoreGradient', 'score_gradient'], ['queryGradient', 'query_gradient'], ['nextQuery', 'next_query'], ['nextLoss', 'next_loss']]) close(result[current], fixture[saved]);
}
assert.throws(() => memoryRead([0, 1], undefined, undefined, [false, false, false]));
checks.push('All ten calculated reads, signed gradients, actual updates, masked and equal-value nulls');

for (const [key, center, radius, normalize] of [['local_worked', 3, 2, false], ['local_worked_renormalized', 3, 2, true], ['local_fresh', 2.5, 1.5, false], ['local_fresh_shift', 3.5, 1.5, false]]) {
  const result = localAttention(center, radius, normalize);
  for (const field of ['base', 'weights', 'sum', 'context']) close(result[field], analytic[key][field]);
}
for (const center of [1, 1.123, 2.37, 3.999, 5]) for (const radius of [1, 1.19, 1.5, 2.431, 3]) {
  const original = localAttention(center, radius);
  close(localAttention(center, radius, true).sum, 1);
  assert.ok(original.sum > 0 && original.sum <= 1);
  original.valid.forEach((valid, index) => { if (!valid) close(original.weights[index], 0); });
}
close(localAttention(2.5, 1.5, false, [0, .5, 1, -.5, -3]).context, analytic.local_fresh.context);
close(cancellationRead(-1.7).attention, cancellationRead(1.9).attention);
assert.notDeepEqual(cancellationRead(-1.7, true).attention, cancellationRead(1.9, true).attention);
const copied = copyDistribution(['Ada', 'met', 'Ada'], [.2, .3, .5], { Ada: .1, met: .6, left: .3 }, .4);
copied.forEach(row => close(row.probability, analytic.pointer.output[row.word]));
checks.push('Continuous window boundaries, normalization, excluded-score null, query cancellation and repeated-copy aggregation');

const start = performance.now();
for (const kind of ['additive', 'general']) {
  const { weights } = JSON.parse(fs.readFileSync(assets + `${kind}-seed-one.json`));
  for (const [name, expected] of Object.entries(mechanics.traces[kind])) {
    if (!expected.rows) continue;
    const options = { prefix: name === 'fresh_forced_prefix' ? 'b' : '', padding: name.includes('padding') || name.includes('mask') ? 2 : 0, admitted: name === 'fresh_broken_mask' ? [true, true] : [] };
    const trace = traceAttention(weights, kind, encodeAttention(weights, expected.input.lemma, expected.input.feature), options);
    assert.equal(trace.prediction, expected.prediction);
    assert.equal(trace.rows.length, expected.rows.length);
    for (let step = 0; step < trace.rows.length; step++) {
      const row = trace.rows[step], reference = expected.rows[step];
      assert.equal(row.emittedToken, reference.emitted_token);
      for (const field of ['query', 'state', 'attention', 'context', 'probabilities']) {
        close(row[field], reference[field], 1e-9);
        maximumTraceError = Math.max(maximumTraceError, ...row[field].map((value, index) => Math.abs(value - reference[field][index])));
      }
    }
  }
  const encoded = encodeAttention(weights, 'prone', 'third_person');
  const base = traceAttention(weights, kind, encoded);
  const forced = traceAttention(weights, kind, encoded, { prefix: 'ab' });
  close(base.rows[0].probabilities, forced.rows[0].probabilities);
  const padded = traceAttention(weights, kind, encoded, { padding: 2 });
  close(base.rows.map(row => row.probabilities), padded.rows.map(row => row.probabilities));
  assert.equal(traceAttention(weights, kind, encoded, { cap: 1 }).rows.length, 1);
  assert.throws(() => encodeAttention(weights, '', 'past'));
}
checks.push('All sixteen saved full recurrent traces versus independent NumPy; changed source, prefix causality and padding invariance');
const elapsedMilliseconds = performance.now() - start;
const source = fs.readFileSync('src/learn/data/topics/attention.jsx', 'utf8');
assert.ok(!source.includes('**Visual:') && !source.includes('**Investigation:'));
assert.ok(!source.includes('after making your own prediction'));
assert.equal((source.match(/<summary>Hint<\/summary>/g) || []).length, 8);
assert.equal((source.match(/<summary>Solution<\/summary>/g) || []).length, 8);
checks.push('All prepared representation anchors replaced; eight separate practice hints and solutions preserved');
const numbered = renderPreparedLesson('1. First dependency.\n\nAn explanation between steps.\n\n2. Second dependency.\n3. Third dependency.').jsx;
assert.match(numbered, /<ol start=\{1\}>/);
assert.match(numbered, /<ol start=\{2\}>/);
assert.equal((numbered.match(/<li>/g) || []).length, 3);
checks.push('Ordered dependency steps preserve numbering across explanatory paragraphs');
const files = ['src/learn/data/recurrent-attention-models.js', 'src/learn/data/recurrent-attention-measurements.json', 'src/learn/data/topics/attention.jsx', 'src/learn/components/lesson-labs/RecurrentAttentionLabs.jsx', 'src/learn/components/lesson-labs/recurrent-attention-labs.css', 'scripts/generate-recurrent-attention-lesson.mjs', 'scripts/lib/prepared-lesson-renderer.mjs', 'scripts/verify-recurrent-attention-models.mjs'];
const evidence = { passed: true, date: '2026-09-26', checks, maximumTraceError, traceCheckMilliseconds: elapsedMilliseconds, timingScope: 'Combined Node verification work, not a browser latency benchmark', reviewedFiles: Object.fromEntries(files.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify({ checks: checks.length, maximumTraceError, elapsedMilliseconds }));
