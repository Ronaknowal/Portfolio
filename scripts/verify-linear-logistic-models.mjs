import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/linear-logistic-models.js';
import { linearLogisticExamples } from '../src/learn/data/linear-logistic-examples.js';

const folder = 'scratch/linear-logistic-verification';
fs.mkdirSync(folder, { recursive: true });
const snapshot = { residuals: [], gradients: [], scores: [], thresholds: [], separation: [], uncertainty: [], examples: linearLogisticExamples };
for (const last of [4, 8, -2]) for (const intercept of [-1, 0, .9, 4]) for (const slope of [-1, 0, .9, 3]) snapshot.residuals.push({ intercept, slope, last, ...model.residualReport(intercept, slope, last) });
for (const rate of [.05, .2, .3]) snapshot.gradients.push({ rate, states: model.gradientTrace(rate, 40) });
for (const score of [-1000, -50, -7, -.1, 0, .1, 7, 50, 1000]) for (const label of [0, 1]) snapshot.scores.push({ score, label, probability: model.sigmoid(score), loss: model.binaryScoreLoss(score, label) });
for (let tick = 0; tick <= 100; tick++) for (const cost of [0, 1, 4, 10]) snapshot.thresholds.push({ threshold: tick / 100, costInput: cost, ...model.thresholdReport(tick / 100, cost) });
for (const penalty of [0, .02, .2]) for (let tick = 0; tick <= 48; tick++) snapshot.separation.push({ weight: tick / 4, penalty, optimum: model.separationOptimum(penalty), ...model.separationReport(tick / 4, penalty) });
for (let tick = -10; tick <= 90; tick++) snapshot.uncertainty.push(model.uncertaintyReport(tick / 10));
for (const bad of [NaN, Infinity, -Infinity, '2', null]) {
  assert.throws(() => model.residualReport(bad, 1)); assert.throws(() => model.sigmoid(bad)); assert.throws(() => model.thresholdReport(bad)); assert.throws(() => model.uncertaintyReport(bad));
}
assert.throws(() => model.residualReport(1e308, 1e308));
assert.throws(() => model.logisticReport({ intercept: 1000 }));
assert.throws(() => model.gradientTrace(.1));
assert.throws(() => model.gradientTrace(.2, 41));
assert.throws(() => model.binaryScoreLoss(0, 2));
assert.equal(model.thresholdReport(1).precision, null);
assert.equal(model.separationOptimum(0), null);
fs.writeFileSync(path.join(folder, 'browser-model-values.json'), JSON.stringify(snapshot));
const run = spawnSync(path.resolve('scratch/lesson-tools/Scripts/python.exe'), ['scripts/verify-linear-logistic-native.py', folder], { encoding: 'utf8', timeout: 180000, env: { ...process.env, PYTHONIOENCODING: 'utf-8', OMP_NUM_THREADS: '1' } });
if (run.status !== 0) throw new Error(run.stderr + run.stdout);
console.log(run.stdout.trim());
