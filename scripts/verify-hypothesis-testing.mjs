import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/hypothesis-testing-models.js';
import { hypothesisExamples as examples } from '../src/learn/data/hypothesis-testing-examples.js';
import { coverageIntervals as originalCoverage } from '../src/learn/components/lesson-labs/math.js';

const folder = 'scratch/hypothesis-testing-verification';
mkdirSync(folder, { recursive: true });
const cases = { tails: [], distributions: [], quantiles: [], coverage: [], power: [], proportions: [], family: [], looks: [], signFlips: [], clusters: [], predictions: [], effects: [] };
for (const shift of [-2, -1.75, -0.25, 0, 0.25, 1, 3]) for (const spread of [0.25, 0.5, 1, 2]) for (const reference of [-1, 0, 0.25, 1, 2]) for (const alpha of [0.01, 0.05, 0.1]) for (const alternative of ['two-sided', 'greater', 'less']) {
  const state = model.pairedTailState({ shift, spread, reference, alpha, alternative });
  assert.equal(state.reject, !state.nullInside);
  cases.tails.push(state);
}
for (const t of [...Array.from({ length: 1201 }, (_, i) => (i - 600) / 15), -1e6, -1e4, 1e4, 1e6]) cases.distributions.push({ t, density: model.t4Density(t), tTail: model.t4Survival(t), normalTail: Math.abs(t) <= 40 ? model.normalSurvival(t) : null });
for (const p of [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]) cases.quantiles.push({ p, value: model.t4Quantile(p) });
let preservedPilotRows = 0;
for (let n = 5; n <= 100; n += 5) for (const level of [90, 95, 99]) for (const batch of [0, 1, 7, 20, 1000000]) for (const mode of ['known', 'estimated']) {
  const state = model.intervalCoverageState(n, level, batch, mode);
  if (mode === 'known') {
    const original = originalCoverage(n, model.NORMAL_CRITICAL[level], batch);
    assert.deepEqual(state.intervals.map(({ mean, low, high }) => ({ mean, low, high })), original);
    preservedPilotRows += original.length;
  }
  cases.coverage.push(state);
}
for (const n of [5, 10, 20, 25, 50, 100, 150, 200]) for (const sigma of [2, 2.5, 4, 7, 10]) for (const effect of [0, 0.25, 0.5, 1, 1.5, 2, 3, 3.75, 4]) for (const alpha of [0.01, 0.05, 0.1]) for (const alternative of ['greater', 'two-sided']) cases.power.push(model.plannedPowerState(n, sigma, effect, alpha, alternative));
for (const n of [5, 10, 15, 20]) for (let index = 0; index <= 100; index++) cases.proportions.push(model.proportionCoverageState(n, index % (n + 1), index / 100));
for (let n = 1; n <= 100; n++) for (const alpha of [0.01, 0.05, 0.1]) cases.family.push(model.familyErrorState(n, alpha));
for (let n = 1; n <= 12; n++) for (const alpha of [0.01, 0.05, 0.1]) cases.looks.push(model.optionalLooksState(n, alpha));
for (const preset of ['original', 'allPositive', 'centered']) for (let mask = 0; mask < 32; mask++) cases.signFlips.push(model.signFlipState(preset, mask));
for (const clusters of [2, 5, 1000]) for (const repeats of [1, 2000, 10000]) for (const correlation of [0, 0.2, 1]) cases.clusters.push(model.clusterPrecisionState(clusters, repeats, correlation));
for (let n = 5; n <= 100; n += 5) for (const level of [90, 95, 99]) cases.predictions.push(model.predictionWidths(n, level));
for (const preset of Object.keys(model.EFFECT_PRESETS)) for (const tolerance of [0.25, 0.5, 0.75, 1, 1.75, 2]) cases.effects.push(model.practicalEffectState(preset, tolerance));
const invalid = [
  () => model.sampleSummary([]), () => model.sampleSummary([1]), () => model.sampleSummary([1, NaN]), () => model.sampleSummary([1, Infinity]),
  () => model.intervalCoverageState(0), () => model.intervalCoverageState(7), () => model.intervalCoverageState(true), () => model.intervalCoverageState(25, 94), () => model.intervalCoverageState(25, 95, -1), () => model.intervalCoverageState(25, 95, 0, 'constructor'),
  () => model.pairedTailState({ shift: Infinity }), () => model.pairedTailState({ spread: 0 }), () => model.pairedTailState({ alpha: 0 }), () => model.pairedTailState({ alternative: 'constructor' }),
  () => model.t4Quantile(1), () => model.t4Survival(NaN), () => model.normalSurvival(41),
  () => model.plannedPowerState(2), () => model.plannedPowerState(25, 0), () => model.plannedPowerState(25, 4, -1),
  () => model.proportionIntervals(-1, 10), () => model.proportionIntervals(11, 10), () => model.proportionIntervals(0, false), () => model.proportionCoverageState(10, 0, 1.1),
  () => model.optionalLooksState(13), () => model.optionalLooksState(2.5), () => model.familyErrorState(0), () => model.clusterPrecisionState(5, 2000, -1),
  () => model.signFlipState('constructor'), () => model.signFlipState('original', 32), () => model.practicalEffectState('original', 0),
];
invalid.forEach(call => assert.throws(call, RangeError));
assert.notEqual(model.inferenceNumber(1e-12), '0');
assert.equal(model.sampleSummary([2, 2]).se, 0);
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const stdout = {};
for (const [key, example] of Object.entries(examples)) {
  const run = spawnSync(python, ['-X', 'utf8', '-c', example.code], { encoding: 'utf8' });
  assert.equal(run.status, 0, `${key}: ${run.stderr}`);
  assert.equal(run.stderr, '', `${key}: no runtime warnings`);
  assert.equal(run.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), `${key}: complete expected stdout`);
  stdout[key] = example.expected;
}
writeFileSync(`${folder}/cases.json`, JSON.stringify({ cases, examples }));
const native = spawnSync(python, ['-X', 'utf8', 'scripts/verify-hypothesis-testing-native.py', `${folder}/cases.json`], { encoding: 'utf8', maxBuffer: 6_000_000 });
assert.equal(native.status, 0, native.stdout + native.stderr);
const results = { at: new Date().toISOString(), status: 'passed', preservedPilotRows, invalidCases: invalid.length, stdoutPrograms: Object.keys(stdout).length, modelCases: Object.fromEntries(Object.entries(cases).map(([key, values]) => [key, values.length])), native: JSON.parse(native.stdout), stdout };
writeFileSync(`${folder}/native-results.json`, JSON.stringify(results, null, 2));
console.log(JSON.stringify({ ...results, stdout: undefined }, null, 2));
