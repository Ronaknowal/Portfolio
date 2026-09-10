import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/mutual-information-models.js';
import { mutualInformationExamples } from '../src/learn/data/mutual-information-examples.js';
const directory = 'scratch/mutual-information-native-verification';
mkdirSync(directory, {
  recursive: true
});
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const finite = [];
for (let a = 0; a <= 4; a += 1) for (let b = 0; b <= 4; b += 1) {
  for (let c = 0; c <= 4; c += 1) for (let d = 0; d <= 4; d += 1) {
    const total = a + b + c + d;
    if (!total) continue;
    const joint = [[a / total, b / total], [c / total, d / total]];
    const state = model.finiteInformation(joint);
    assert.ok(Object.isFrozen(state) && Object.isFrozen(state.cells[0][0]));
    close(state.mi, state.hy - state.conditionalEntropy);
    assert.ok(state.mi >= -1e-13 && state.mi <= Math.min(state.hx, state.hy) + 1e-13);
    finite.push(state);
  }
}
// A real positive joint cell can have an independent-reference product that
// underflows to zero. The pointwise log ratio still has a finite value.
const tiny = model.finiteInformation([[1e-310, 0], [0, 1]]);
assert.equal(tiny.cells[0][0].independent, 0);
close(tiny.cells[0][0].information, 310 * Math.log2(10), 1e-10);
assert.ok(Number.isFinite(tiny.mi));
const xor = [];
for (const bias of [0.05, 0.1, 0.25, 0.5, 0.7, 0.95]) {
  for (const reveal of ['a', 'b', 'both']) xor.push(model.xorInformation(bias, reveal));
}
const representations = [];
for (const error of [0, 0.1, 0.3, 0.5]) for (const noise of [0, 0.15, 0.2, 0.5]) {
  for (const mode of ['constant', 'signal', 'nuisance', 'both', 'noisy']) {
    representations.push(model.bottleneckRepresentation(mode, noise, error, 3));
  }
}
const iterations = [];
for (const beta of [0, 0.5, 1, 2, 3, 8, 12]) for (const error of [0.05, 0.1, 0.45]) {
  for (const initial of ['signal', 'symmetric', 'nuisance']) {
    const trace = model.bottleneckIterations(beta, 40, initial, error);
    for (const row of trace.rows) {
      row.encoder.forEach(probabilities => close(probabilities[0] + probabilities[1], 1));
      if (row.step) assert.ok(row.objective <= trace.rows[row.step - 1].objective + 1e-12);
    }
    iterations.push(trace);
  }
}
assert.ok(model.bottleneckIterations(3, 40, 'signal').current.objective < -0.6);
close(model.bottleneckIterations(3, 40, 'symmetric').current.objective, 0);
const bounds = [];
for (const r of [0.02, 0.2, 0.5, 0.8, 0.98]) for (const d of [0.02, 0.26, 0.4, 0.8, 0.98]) {
  for (const beta of [0, 0.5, 3, 12]) bounds.push(model.variationalInformationBounds(r, d, beta));
}
const samples = [];
for (const categories of [2, 4, 8]) for (const count of [20, 100, 2000]) {
  for (const mode of ['independent', 'channel']) {
    samples.push(model.sampledInformation({
      categories,
      count,
      mode,
      seed: 831
    }));
  }
}
for (const seed of [1, 2147483647, 0x5a39b174]) {
  const small = model.sampledInformation({
    count: 20,
    seed
  });
  const large = model.sampledInformation({
    count: 100,
    seed
  });
  assert.deepEqual(small.pairs, large.pairs.slice(0, 20));
  samples.push(small);
}
const gaussian = [1e-310, 1e-100, 0.05, 0.25, 0.5, 1, 2, 4].map(sigma => ({
  sigma,
  information: model.gaussianNoiseInformation(sigma)
}));
assert.equal(model.gaussianNoiseInformation(0), Infinity);
let invalidCases = 0;
for (const operation of [() => model.finiteInformation([]), () => model.finiteInformation([[0.5], [0.2, 0.3]]), () => model.finiteInformation([[1, -0.1]]), () => model.finiteInformation([[0.5, NaN]]), () => model.finiteInformation([[0.1, 0.2]]), () => model.binaryChannel(-0.1), () => model.bottleneckIterations(3, 41), () => model.bottleneckIterations(3, 1.5), () => model.bottleneckIterations(3, 1, 'other'), () => model.bottleneckIterations(3, 1, 'signal', 0), () => model.variationalInformationBounds(0), () => model.variationalInformationBounds(0.5, 1), () => model.sampledInformation({
  seed: 0
}), () => model.sampledInformation({
  count: 19
}), () => model.sampledInformation({
  categories: 3
}), () => model.gaussianNoiseInformation(Infinity)]) {
  assert.throws(operation);
  invalidCases += 1;
}
writeFileSync(`${directory}/model-fixtures.json`, JSON.stringify({
  finite,
  tiny,
  xor,
  representations,
  iterations,
  bounds,
  samples,
  gaussian,
  invalidCases,
  examples: mutualInformationExamples
}));
const verification = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-mutual-information-native.py'], {
  encoding: 'utf8',
  maxBuffer: 5_000_000
});
process.stdout.write(verification.stdout || '');
process.stderr.write(verification.stderr || '');
assert.equal(verification.status, 0, 'Independent Python verification failed.');
console.log(`JavaScript contracts passed: ${finite.length} finite tables, ${iterations.length} complete IB traces, ${bounds.length} bounds, ${samples.length} samples, ${invalidCases} rejected inputs.`);
