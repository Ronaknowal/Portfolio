import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
import { perceptronUpdateExample, evaluateBatchNeuronExample, neuronOutputShares } from '../src/learn/data/perceptron-intuition.js';

const topicId = 'perceptrons-neurons-activation-functions';
const destination = 'docs/teaching/concept-intuition/' + topicId + '/author-checks.json';
const sourceFiles = [
  'src/learn/data/topics/' + topicId + '.jsx',
  'src/learn/data/perceptron-intuition.js',
  'src/learn/components/lesson-labs/PerceptronIntuitionFigures.jsx',
  'src/learn/components/lesson-labs/perceptron-intuition.css',
];
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const checks = [];
const receipt = { topicId, passed: false, sourceFiles, checks, browser: 'pending root', independentReview: 'pending complementary reviewer' };
const save = () => fs.writeFileSync(destination, JSON.stringify(receipt, null, 2) + '\n');
save();
try {
  const update = perceptronUpdateExample();
  assert.deepEqual([update.before, update.after, ...update.nextWeights, update.nextBias], [-1, 2, -.5, 1, .5]);
  assert.equal(update.after - update.before, .5 * (1 + 4 + 1));
  checks.push({ name: 'Mistake correction matches independent signed-score expansion', passed: true });
  assert.deepEqual(evaluateBatchNeuronExample(), [[4, -2.5], [-7, 3.5]]);
  checks.push({ name: 'Every displayed matrix result matches hand arithmetic', passed: true });
  const first = neuronOutputShares([1, 2, 3]).shares;
  const second = neuronOutputShares([1, 2, 4]).shares;
  const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12);
  const denominator = Math.exp(1) + Math.exp(2) + Math.exp(3);
  first.forEach((value, i) => close(value, Math.exp(i + 1) / denominator));
  neuronOutputShares([1001, 1002, 1003]).shares.forEach((value, i) => close(value, first[i]));
  close(first[0] / first[1], second[0] / second[1]);
  assert.ok(second[0] < first[0] && second[1] < first[1] && second[2] > first[2]);
  close(first.reduce((a, b) => a + b), 1);
  checks.push({ name: 'Softmax common total, common shift and unchanged pair ratio', passed: true });
  for (const file of sourceFiles.filter(file => /\.(jsx|js)$/.test(file))) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  checks.push({ name: 'Changed JavaScript and JSX parse', passed: true });
  const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/concept-intuition-baseline.json'));
  const preserved = Object.entries(baseline.actualFileHashes).filter(([file]) =>
    file.startsWith('public/learn-assets/perceptrons/') ||
    /^src\/learn\/data\/perceptron-(models|examples|data|mechanism-program)\.js$/.test(file));
  assert.ok(preserved.length >= 4);
  for (const [file, expected] of preserved) assert.equal(hash(file), expected, file);
  checks.push({ name: 'Prior models, examples, measured data and downloads unchanged', passed: true, files: preserved.length });
  receipt.sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, hash(file)]));
  receipt.reviewedOn = new Date().toISOString();
  receipt.passed = true;
} catch (error) { receipt.failure = error.stack; process.exitCode = 1; }
save();
console.log(JSON.stringify({ passed: receipt.passed, checks, failure: receipt.failure }));
