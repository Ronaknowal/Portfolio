import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { activation, geometry, geometryInitial, normalCdf, sensitivity, xorRows, boundarySegment, parseNumeric, triangle } from '../src/learn/data/perceptron-models.js';
import { perceptronData } from '../src/learn/data/perceptron-data.js';
import { perceptronExamples } from '../src/learn/data/perceptron-examples.js';
const checks = [];
const check = (name, fn) => {
  fn();
  checks.push(name);
};
const close = (a, b, tol = 1e-10) => assert.ok(Math.abs(a - b) <= tol, `${a} != ${b}`);
const fixture = JSON.parse(fs.readFileSync('docs/teaching/drafts/perceptrons-neurons-activation-functions/calculated-inputs.json', 'utf8'));
check('geometry fixtures, scale invariance, ties and zero norm', () => {
  for (const f of fixture.investigationFixtures.geometry) {
    const r = geometry({
      x1: f.point[0],
      x2: f.point[1],
      w1: f.weight[0],
      w2: f.weight[1],
      b: f.bias
    });
    for (const key of ['score', 'hard', 'sigmoid']) close(r[key], f[key]);
    if (f.distance === null) assert.equal(r.distance, null);else close(r.distance, f.distance);
  }
  for (const c of [.25, .5, 1, 1.75, 3]) close(geometry(geometryInitial, c).distance, 1.6);
});
check('clipped boundary coordinates lie on the equation', () => {
  for (const [w1, w2, b] of [[1.5, -2, -1], [0, 1, 2], [1, 0, 3], [-4, .25, 4], [0, 0, 1]]) {
    for (const [x, y] of boundarySegment(w1, w2, b)) close(w1 * x + w2 * y + b, 0);
  }
});
check('XOR repairs and inactive null match all four saved corners', () => {
  for (const f of fixture.investigationFixtures.xorRepairs) xorRows(f.bias, f.coefficient).forEach((r, i) => {
    close(r.h2, f.hidden2[i]);
    close(r.q, f.output[i]);
  });
});
check('incoming weights and slopes reproduce independent PyTorch fixture', () => {
  for (const f of fixture.investigationFixtures.localSensitivity) {
    const r = sensitivity(f.function, f.score, f.weight);
    for (const [key, other] of [['value', 'value'], ['slope', 'slope'], ['sensitivity', 'sensitivity']]) close(r[key], f[other]);
  }
  assert.equal(sensitivity('sigmoid', 0, 4).category, 'at least 1');
  assert.equal(sensitivity('relu', 2, -2).category, 'negative');
});
const oracle = JSON.parse(fs.readFileSync('docs/teaching/evidence/perceptron-activation-oracle.json', 'utf8'));
let maxError = 0;
check('nine dense value/slope curves agree with independent PyTorch float64 oracle', () => {
  for (const [name, data] of Object.entries(oracle.functions)) oracle.z.forEach((z, i) => {
    const result = activation(name, z);
    for (const [key, expected] of [['value', data.values[i]], ['slope', data.slopes[i]]]) {
      maxError = Math.max(maxError, Math.abs(result[key] - expected));
      close(result[key], expected, 2e-12);
    }
  });
});
check('CDF endpoints, symmetry, numerical stability and corners', () => {
  for (const z of [0, 1, 2, 4, 6]) close(normalCdf(z) + normalCdf(-z), 1);
  assert.equal(activation('relu', 0).corner, true);
  assert.equal(activation('leaky_relu', 0).slope, .1);
  for (const name of ['sigmoid', 'mish', 'silu']) for (const z of [-1000, 1000]) assert.ok(Number.isFinite(activation(name, z).value));
});
check('numeric text admits rational entries and rejects invalid ranges without clamping', () => {
  close(parseNumeric('-4/3', -4, 2).value, -4 / 3);
  for (const text of ['', 'NaN', 'Infinity', '1/0', '7', '--2', '1/2/3']) assert.ok(parseNumeric(text, -4, 4).error);
  for (const v of [-4, -3.75, .25, 4]) close(parseNumeric(String(v), -4, 4).value, v);
});
check('triangular pulse has exact corners and no spurious outer slope', () => {
  for (const [x, y] of [[-1, 0], [0, 0], [.5, .5], [1, 1], [1.5, .5], [2, 0], [3, 0]]) close(triangle(x), y);
});
check('all measured rows join validation IDs to the original labels, with no split leakage', () => {
  assert.equal(perceptronData.digits.length, 400);
  assert.equal(perceptronData.runs.length, 18);
  assert.equal(new Set([...perceptronData.splits.trainSourceIds, ...perceptronData.splits.validationSourceIds]).size, 400);
  const byId = new Map(perceptronData.digits.map(row => [row.sourceId, row]));
  for (const row of perceptronData.runs) {
    const correct = row.validationPredictions.filter((prediction, i) => prediction === byId.get(perceptronData.splits.validationSourceIds[i]).digit).length;
    assert.equal(row.correct, correct);
    close(row.validationAccuracy, correct / 120, 1e-7);
    assert.deepEqual(row.trace.map(p => p.step), [0, 1, 10, 50, 100, 200]);
    close(row.trace.at(-1).trainLoss, row.trainLoss);
  }
});
check('executed printed digit counts and losses match all 18 plotted rows', () => {
  for (const line of perceptronExamples[1].expected.split('\n').slice(1)) {
    const [name, seed, loss, count] = line.trim().split(/\s+/),
      run = perceptronData.runs.find(r => r.activation === name && r.seed === Number(seed));
    close(run.trainLoss, Number(loss), .00000051);
    assert.equal(`${run.correct}/120`, count);
  }
  assert.ok(perceptronExamples[0].expected.includes('9 [ 3  2 -4] 0'));
  assert.ok(perceptronExamples[2].expected.includes('(2, 6)\n144'));
});
const sourceFiles = ['src/learn/data/perceptron-models.js', 'src/learn/data/perceptron-data.js', 'src/learn/data/perceptron-examples.js'];
const result = {
  status: 'passed',
  checks,
  denseValuesAndSlopes: oracle.z.length * 9 * 2,
  maxAbsoluteError: maxError,
  oracleTorch: oracle.torch,
  sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]))
};
fs.writeFileSync('docs/teaching/evidence/perceptron-models.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
