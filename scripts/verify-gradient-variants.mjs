import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { gradientVariantsExamples } from '../src/learn/data/gradient-variants-examples.js';
import * as models from '../src/learn/data/gradient-variants-models.js';

const folder = 'scratch/gradient-variants-verification';
mkdirSync(folder, { recursive: true });
const cases = { batch: [], momentum: [], adaptive: [], decay: [], layer: [] };
for (const theta of [-4, -1, 0, 1, 3.25, 4]) {
  for (let mask = 1; mask < 16; mask++) {
    const selected = [0, 1, 2, 3].filter(index => mask & (1 << index));
    for (const rate of [0, 0.2, 0.8]) for (const reduction of ['mean', 'sum']) {
      cases.batch.push(models.batchGradientState(theta, selected, rate, reduction));
    }
  }
}
for (const method of ['sgd', 'momentum', 'nesterov']) {
  for (const rate of [0.005, 0.05, 0.08, 0.2, 0.25]) {
    for (const beta of [0, 0.5, 0.8, 0.95]) for (const curvature of [1, 7, 20, 30]) {
      cases.momentum.push(models.momentumTrajectoryState(method, rate, beta, 24, curvature));
    }
  }
}
for (const method of ['adagrad', 'rmsprop', 'adam']) {
  for (const profile of Object.keys(models.optimizerGradientProfiles)) {
    for (const beta1 of [0, 0.9]) for (const beta2 of [0, 0.9, 0.999]) {
      for (const epsilon of [1e-8, 0.1]) for (const correction of [false, true]) for (const rate of [0.001, 0.1, 0.5]) {
        cases.adaptive.push(models.adaptiveHistoryState(method, profile, rate, beta1, beta2, epsilon, correction));
      }
    }
  }
}
for (const preset of Object.keys(models.optimizerDecayPresets)) {
  for (const rate of [0.01, 0.1, 0.3]) for (const decay of [0, 0.1, 0.5, 1]) {
    for (const steps of [1, 2, 4, 8]) cases.decay.push(models.decayComparisonState(preset, rate, decay, steps));
  }
}
for (const method of ['sgd', 'lars', 'lamb']) {
  for (const scale of [0.02, 0.1, 1, 2]) for (const rate of [0.01, 0.1, 0.5]) {
    for (const decay of [0, 0.2, 1]) for (const coefficient of [0.01, 0.1, 1]) {
      for (const preset of ['ordinary', 'zeroWeight', 'zeroGradient']) cases.layer.push(models.layerScaleState(method, scale, rate, decay, coefficient, preset));
    }
  }
}
const invalid = [
  () => models.batchGradientState(1, []), () => models.batchGradientState(1, [0, 0]),
  () => models.batchGradientState(1, [4]), () => models.batchGradientState(NaN),
  () => models.batchGradientState(1, [1], -0.1), () => models.batchGradientState(1, [1], 0.2, 'median'),
  () => models.momentumTrajectoryState('bad'), () => models.momentumTrajectoryState('sgd', 0),
  () => models.momentumTrajectoryState('sgd', 0.1, 1), () => models.momentumTrajectoryState('sgd', 0.1, 0.5, 2.5),
  () => models.momentumTrajectoryState('sgd', 0.1, 0.5, 2, Infinity),
  () => models.adaptiveHistoryState('bad'), () => models.adaptiveHistoryState('adam', 'bad'),
  () => models.adaptiveHistoryState('adam', 'sparse', 0.1, 0.9, 1),
  () => models.adaptiveHistoryState('adam', 'sparse', 0.1, 0.9, 0.9, 0),
  () => models.adaptiveHistoryState('adam', 'sparse', 0.1, 0.9, 0.9, 1e-6, 1),
  () => models.decayComparisonState('bad'), () => models.decayComparisonState('zero', 0.1, -1),
  () => models.decayComparisonState('zero', 0.1, 0.1, 0),
  () => models.layerScaleState('bad'), () => models.layerScaleState('lars', 0),
  () => models.layerScaleState('lars', 0.1, 0.1, 0, 0),
  () => models.layerScaleState('lamb', 0.1, 0.1, 0, 0.1, 'bad'),
];
invalid.forEach(call => assert.throws(call));
assert.equal(models.formatOptimizerNumber(1e-14), '1.000e-14');
assert.equal(models.formatOptimizerNumber(0), '0');
writeFileSync(`${folder}/cases.json`, JSON.stringify(cases));
writeFileSync(`${folder}/programs.json`, JSON.stringify(gradientVariantsExamples));
const run = spawnSync(process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe',
  ['-I', 'scripts/verify-gradient-variants-native.py', folder], { encoding: 'utf8', timeout: 60000 });
assert.equal(run.status, 0, run.stderr || run.stdout);
console.log(run.stdout);
writeFileSync(`${folder}/model-boundaries.json`, JSON.stringify({ checkedAt: new Date().toISOString(),
  invalidGroups: invalid.length, tinyNonzeroFormatting: 'passed', status: 'passed' }, null, 2));
console.log(`${invalid.length} invalid model groups passed.`);
