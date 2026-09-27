import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
const topicId = 'batch-layer-group-rms-normalization';
const sourceFiles = ['src/learn/data/topics/' + topicId + '.jsx', 'src/learn/components/lesson-labs/NormalizationIntuitionFigures.jsx', 'src/learn/components/lesson-labs/normalization-intuition.css'];
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const receipt = { topicId, passed: false, sourceFiles, checks: [], browser: 'pending root', independentReview: 'pending complementary reviewer' };
const save = () => fs.writeFileSync('docs/teaching/concept-intuition/' + topicId + '/author-checks.json', JSON.stringify(receipt, null, 2) + '\n');
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`);
save();
try {
  const centered = values => { const mean = values.reduce((a, b) => a + b) / values.length; const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length; return { mean, variance, output: values.map(value => (value - mean) / Math.sqrt(variance + 1e-5)) }; };
  const before = centered([1, 3, 5, 7]), after = centered([1, 3, 15, 17]);
  assert.deepEqual([before.mean, before.variance, after.mean, after.variance], [4, 5, 9, 50]);
  assert.ok(Math.abs(before.output[0] - after.output[0]) > .2);
  close(centered([1, 3]).output[0], -1 / Math.sqrt(1 + 1e-5));
  for (const values of [[1, 3], [11, 13]]) { const m = centered(values); close(values.reduce((sum, x) => sum + x * x, 0) / 2, m.variance + m.mean ** 2); }
  receipt.checks.push({ name: 'Causal leakage fixture and mean-square decomposition', passed: true });
  close([4, 8, 2].reduce((buffer, mean) => .9 * buffer + .1 * mean, 0), 1.244);
  close(.729 + .081 + .09 + .1, 1); close(.081 * 4 + .09 * 8 + .1 * 2, 1.244);
  receipt.checks.push({ name: 'Recursive running buffer agrees with weight expansion', passed: true });
  const x = [1, 3, -2], offset = centered(x.map(value => value + 8)).output;
  centered(x).output.forEach((value, index) => close(value, offset[index]));
  close(1 / (1 + 1), .5); close(.01 / (1 + .01), .009900990099009901);
  close(Math.hypot(1.2, 1.6), 2);
  receipt.checks.push({ name: 'Offset invariance, epsilon attenuation and weight magnitude', passed: true });
  for (const file of sourceFiles.filter(file => file.endsWith('.jsx'))) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  receipt.checks.push({ name: 'Changed JSX parses', passed: true });
  const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/concept-intuition-baseline.json'));
  const preserved = Object.entries(baseline.actualFileHashes).filter(([file]) => file.startsWith('public/learn-assets/' + topicId + '/') || /^src\/learn\/data\/normalization-(models|data)\.js$/.test(file));
  assert.ok(preserved.length > 3);
  for (const [file, expected] of preserved) assert.equal(hash(file), expected, file);
  receipt.checks.push({ name: 'Native programs, models and measured data unchanged', passed: true, files: preserved.length });
  receipt.sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, hash(file)]));
  receipt.reviewedOn = new Date().toISOString(); receipt.passed = true;
} catch (error) { receipt.failure = error.stack; process.exitCode = 1; }
save(); console.log(JSON.stringify({ passed: receipt.passed, checks: receipt.checks, failure: receipt.failure }));
