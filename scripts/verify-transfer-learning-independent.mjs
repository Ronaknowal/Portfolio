import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { loraFixture, loraForward, loraStep, batchNormFixture, batchNormForward, selectTransferCandidate } from '../src/learn/data/transfer-learning-model.js';

// Complement the author's finite-difference and CPU checks with invariances.
const receipt = 'docs/teaching/evidence/transfer-learning-independent.json';
const report = { status: 'running', reviewer: 'root, independent of audit_classical_controls author', checks: [] };
const save = () => fs.writeFileSync(receipt, JSON.stringify(report, null, 2) + '\n');
save();
const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
try {
  for (const rank of [1, 2]) {
    const state = { ...loraFixture(rank), B: rank === 1 ? [[.3], [-.2]] : [[.3, -.6], [-.2, .5]], x: [1.7, -.8], target: [.1, .6] };
    const base = loraForward(state);
    const rescaled = loraForward({ ...state, A: state.A.map(row => row.map(v => 2 * v)), B: state.B.map(row => row.map(v => v / 2)) });
    base.output.forEach((value, i) => near(value, rescaled.output[i]));
    base.inputGradient.forEach((value, i) => near(value, rescaled.inputGradient[i]));
    base.gradA.flat().forEach((value, i) => near(value / 2, rescaled.gradA.flat()[i]));
    base.gradB.flat().forEach((value, i) => near(value * 2, rescaled.gradB.flat()[i]));
    const zeroRate = loraStep({ ...state, rate: 0 });
    assert.deepEqual(zeroRate.after.output, base.output);
    const zeroScale = loraForward({ ...state, alpha: 0 });
    assert.deepEqual(zeroScale.output, state.x);
    assert.ok(zeroScale.gradA.flat().every(v => v === 0) && zeroScale.gradB.flat().every(v => v === 0));
  }
  report.checks.push('Rank-1/2 factor rescaling preserves the function/input gradient while transforming factor gradients; zero rate/scale give exact nulls');
  for (const training of [true, false]) {
    const state = { ...batchNormFixture(), input: [-3, 7], mean: 1.2, variance: 2.3, training, recording: true, trainable: true, optimizer: true };
    const base = batchNormForward(state), noGraph = batchNormForward({ ...state, recording: false }), noOptimizer = batchNormForward({ ...state, optimizer: false });
    assert.deepEqual(base.output, noGraph.output);
    assert.deepEqual(base.output, noOptimizer.output);
    near(base.nextMean, noGraph.nextMean); near(base.nextVariance, noGraph.nextVariance);
    assert.equal(noGraph.inputGradient, null); assert.equal(noOptimizer.canUpdate, false);
    const offset = batchNormForward({ ...state, input: state.input.map(v => v + 4), mean: state.mean + 4 });
    base.output.forEach((v, i) => near(v, offset.output[i]));
    near(offset.nextMean - base.nextMean, 4);
  }
  report.checks.push('BatchNorm mode, gradient recording and optimizer ownership are independent; translated inputs and memory retain output');
  const runs = JSON.parse(fs.readFileSync('src/learn/data/transfer-learning-experiment.json')).runs;
  const choose = records => selectTransferCandidate(records, 400).winner.method;
  assert.equal(choose(runs), 'lora2');
  assert.equal(choose([...runs].reverse()), 'lora2');
  const poisonedOtherSeeds = runs.map(row => row.seed === 1 ? row : { ...row, validation: { ...row.validation, ce: -1000 } });
  assert.equal(choose(poisonedOtherSeeds), 'lora2');
  assert.equal(selectTransferCandidate(runs, 84).winner, null);
  report.checks.push('Candidate selection is unaffected by record order or other-seed metrics, and an impossible budget stays empty');
  report.sourceHashes = Object.fromEntries(['src/learn/data/transfer-learning-model.js', 'src/learn/components/lesson-labs/TransferLearningLabs.jsx', 'src/learn/components/lesson-labs/transfer-learning.css', 'src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx'].map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  report.status = 'passed'; report.checkedAt = new Date().toISOString();
} catch (error) { report.status = 'failed'; report.failure = error.stack; process.exitCode = 1; }
save(); console.log(JSON.stringify(report));
