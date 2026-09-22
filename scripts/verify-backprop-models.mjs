import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { lineFit, fitDefaults, sharedSquare, finiteDifference, biasGradient, directionalProducts, checkpointSchedule } from '../src/learn/data/backprop-models.js';
import { backpropTraining } from '../src/learn/data/backprop-training.js';

let assertions = 0;
function close(actual, expected, tolerance = 1e-10) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
  assertions++;
}
const fit = lineFit(fitDefaults);
for (const [actual, expected] of [[fit.loss, .5], [fit.weightGradient, -2], [fit.biasGradient, -1], [fit.nextWeight, 1.2], [fit.nextBias, .1], [fit.nextLoss, .17]]) close(actual, expected);
close(lineFit({ ...fitDefaults, rate: 1 }).nextLoss, 12.5);
close(lineFit({ ...fitDefaults, rate: 0 }).delta, 0);
// Independent differentiation of the full objective at varied, non-preset inputs.
for (const inputs of [fitDefaults, { weight: -.83, bias: .42, rate: .37, target1: 2.17, target2: -1.3 }, { weight: 3, bias: -3, rate: 1, target1: -3, target2: 5 }]) {
  const model = lineFit(inputs);
  const loss = (w, b) => ((w + b - inputs.target1) ** 2 + (2 * w + b - inputs.target2) ** 2) / 2;
  const h = 1e-5;
  close(model.weightGradient, (loss(inputs.weight + h, inputs.bias) - loss(inputs.weight - h, inputs.bias)) / (2 * h), 1e-8);
  close(model.biasGradient, (loss(inputs.weight, inputs.bias + h) - loss(inputs.weight, inputs.bias - h)) / (2 * h), 1e-8);
  close(model.nextLoss, loss(model.nextWeight, model.nextBias));
}
for (const x of [-3, -2, -.37, 0, 1.5, 3]) for (const c of [-3, -1, 0, .42, 2, 3]) {
  const model = sharedSquare(x, c);
  close(model.gradient, model.firstPath + model.secondPath);
  close(model.gradient, 2 * model.slotContribution);
  close(model.loss, (1 + c) * x * x);
  if (c === -1 || x === 0) { close(model.gradient, 0); close(model.loss, 0); }
}
assert.deepEqual(biasGradient, [9, 12]); assertions++;
const sine = exponent => finiteDifference({ exponent });
assert.ok(sine(-5).absoluteError < sine(-3).absoluteError / 100); assertions++;
assert.ok(sine(-9).absoluteError > sine(-5).absoluteError * 10); assertions++;
assert.ok(sine(-15).absoluteError > .01); assertions++;
const offset = finiteDifference({ kind: 'linear', point: 1, offset: 1e12, exponent: -5 });
close(offset.lower, 1000000000001); close(offset.upper, 1000000000001); close(offset.estimate, 0); close(offset.analytic, 1);
const zero = finiteDifference({ kind: 'square', point: 0, exponent: -5 });
close(zero.absoluteError, 0); assert.equal(zero.relativeError, null); assertions++;
close(directionalProducts.left, 5.944663510874394); close(directionalProducts.right, directionalProducts.left);
// Before a segment is reversed, all of each operation's two required states exist.
for (const [stageIndex, operations] of [[1, [8, 7, 6, 5]], [3, [4, 3, 2, 1]]]) {
  const stage = checkpointSchedule[stageIndex];
  const slots = new Set([...stage.boundaries, ...stage.regenerated]);
  for (const operation of operations) { assert.ok(slots.has(operation) && slots.has(operation - 1)); assertions++; }
  assert.equal(slots.size, stage.boundaries.length + stage.regenerated.length); assertions++;
}
const saved = JSON.parse(fs.readFileSync('docs/teaching/drafts/backpropagation-automatic-differentiation/calculated-inputs.json', 'utf8'));
for (const key of ['xorTraining', 'digitTraining']) { assert.deepEqual(backpropTraining[key], saved[key]); assertions++; }
close(backpropTraining.xorTraining[0].mse, 2.903633084230992);
close(backpropTraining.digitTraining[0].trainLoss, 3.597476697698935);
assert.deepEqual(backpropTraining.digitTraining.map(row => row.validationCorrect), [17, 17, 40, 101, 105, 112]); assertions++;
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const files = ['src/learn/data/backprop-models.js', 'src/learn/data/backprop-training.js', 'scripts/verify-backprop-models.mjs'];
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/backprop-models.json', JSON.stringify({ timestamp: new Date().toISOString(), assertions, scope: 'Independent small-model calculations, numerical contrasts, source-bound training observations, checkpoint dependency schedule.', sourceHashes: Object.fromEntries(files.map(file => [file, hash(file)])) }, null, 2) + '\n');
console.log(`Backprop models: ${assertions} assertions passed.`);
