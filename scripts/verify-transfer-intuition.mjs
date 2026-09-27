import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
const topicId = 'transfer-learning-fine-tuning-strategies';
const sourceFiles = ['src/learn/data/topics/' + topicId + '.jsx', 'src/learn/components/lesson-labs/TransferIntuitionFigures.jsx', 'src/learn/components/lesson-labs/transfer-intuition.css'];
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const receipt = { topicId, passed: false, sourceFiles, checks: [], browser: 'pending root', independentReview: 'pending complementary reviewer' };
const save = () => fs.writeFileSync('docs/teaching/concept-intuition/' + topicId + '/author-checks.json', JSON.stringify(receipt, null, 2) + '\n');
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`);
save();
try {
  const rows = [[-1,-1],[-1,1],[1,-1],[1,1]];
  for (const feature of [-1,1]) assert.equal(new Set(rows.filter(row => row[0] === feature).map(([x,y]) => x*y > 0)).size, 2);
  // Both class convex hulls contain their common midpoint (0,0); strict affine separation is impossible.
  assert.deepEqual(rows[0].map((value, i) => (value + rows[3][i]) / 2), [0,0]);
  assert.deepEqual(rows[1].map((value, i) => (value + rows[2][i]) / 2), [0,0]);
  receipt.checks.push({ name: 'Preserved nonlinear label signal versus identical projected features', passed: true });
  const correction = ([x,y]) => [x-y, 2*(x-y)];
  assert.deepEqual([[2,1],[0,2],[3,3]].map(correction), [[1,2],[-2,-4],[0,0]]);
  receipt.checks.push({ name: 'Rank-one corrections and equal-coordinate geometric scales', passed: true });
  const tanhFromExp = value => (Math.exp(2 * value) - 1) / (Math.exp(2 * value) + 1);
  close(Math.tanh(-1), tanhFromExp(-1));
  const output = [1 + .5 * Math.tanh(-1), 2 - .25 * Math.tanh(-1)];
  close(output[0], .6192029220221176); close(output[1], 2.1903985389889413);
  assert.ok(Math.abs(Math.tanh(-2) - 2 * Math.tanh(-1)) > .5);
  receipt.checks.push({ name: 'Bottleneck correction, residual addition and nonlinearity', passed: true });
  for (const file of sourceFiles.filter(file => file.endsWith('.jsx'))) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  receipt.checks.push({ name: 'Changed JSX parses', passed: true });
  const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/concept-intuition-baseline.json'));
  const preserved = Object.entries(baseline.actualFileHashes).filter(([file]) => file.startsWith('src/learn/assets/transfer-learning/') || file.startsWith('public/learn-assets/transfer-learning/') || /^src\/learn\/data\/transfer-learning-(model|mechanism-program)\.js$/.test(file) || /^src\/learn\/data\/transfer-learning-(experiment|specimens)\.json$/.test(file));
  assert.ok(preserved.length >= 5);
  for (const [file, expected] of preserved) assert.equal(hash(file), expected, file);
  receipt.checks.push({ name: 'Existing mechanism and measured-data source identities preserved', passed: true, files: preserved.length });
  receipt.sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, hash(file)]));
  receipt.reviewedOn = new Date().toISOString(); receipt.passed = true;
} catch (error) { receipt.failure = error.stack; process.exitCode = 1; }
save(); console.log(JSON.stringify({ passed: receipt.passed, checks: receipt.checks, failure: receipt.failure }));
