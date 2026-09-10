import fs from 'node:fs';
import assert from 'node:assert/strict';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { buildRateSchedule, buildScheduleClockTrace, buildPlateauTrace, scheduleNoiseMoments, parseValidationMetrics } from '../src/learn/data/learning-rate-schedule-models.js';
import { learningRateScheduleExamples } from '../src/learn/data/learning-rate-schedule-examples.js';

const directory = 'scratch/learning-rate-schedule-review';
fs.mkdirSync(directory, { recursive: true });
const schedules = [];
for (const total of [3, 4, 8, 12, 24, 48]) {
  for (const kind of ['constant', 'cosine', 'linear', 'exponential', 'step', 'one-cycle', 'restart']) {
    for (const peak of [.01, .2, 1]) {
      const specification = { total, kind, peak, minimum: peak / 20, warmup: Math.min(3, total - 1), rise: Math.max(2, Math.floor(total / 3)), period: 4 };
      schedules.push({ specification, states: buildRateSchedule(specification) });
    }
  }
}
for (const total of [2, 3, 5, 17]) for (const warmup of [0, 1, total - 1]) {
  const specification = { total, warmup, kind: 'cosine', minimum: 0, peak: .2 };
  schedules.push({ specification, states: buildRateSchedule(specification) });
}
const noise = [];
for (const curvature of [.5, 2, 4, 8]) for (const sigma of [0, .5, 2]) {
  for (const kind of ['constant', 'cosine', 'one-cycle']) {
    const rates = buildRateSchedule({ total: 8, warmup: 2, rise: 3, kind });
    noise.push({ rates, specification: { curvature, noise: sigma, initialError: 3 }, states: scheduleNoiseMoments(rates, { curvature, noise: sigma }) });
  }
}
const clocks = [];
for (const accumulation of [1, 2, 3]) for (const skipSecond of [false, true]) for (const policy of ['committed', 'microbatch', 'advance-first']) {
  const specification = { accumulation, skipSecond, policy };
  clocks.push({ specification, trace: buildScheduleClockTrace(specification) });
}
const plateau = [];
const metricsSets = [[1, .9, .9, .89, .88, .88, .9, .87, .87, .87, .87, .87], [0, 0, 0, 0, 0], [1, .75, .5, .25, 0], Array(24).fill(.5)];
for (const metrics of metricsSets) for (const patience of [0, 1, 3]) for (const threshold of [0, .02, .125]) for (const cooldown of [0, 1, 3]) {
  const specification = { patience, threshold, cooldown };
  plateau.push({ metrics, specification, states: buildPlateauTrace(metrics, specification) });
}
let rejected = 0;
for (const specification of [{ total: 1 }, { total: 81 }, { kind: 'unknown' }, { kind: 'exponential', minimum: 0 }, { peak: Infinity }, { minimum: -.1 }, { total: 3, warmup: 3 }, { kind: 'one-cycle', rise: 1 }, { kind: 'step', period: 0 }]) {
  assert.throws(() => buildRateSchedule(specification), RangeError); rejected += 1;
}
for (const text of ['', 'NaN 1', 'Infinity', '-1', '1 nope', '11', Array(25).fill(1).join(' ')]) { assert.throws(() => parseValidationMetrics(text), RangeError); rejected += 1; }
for (const input of [[], [NaN], [11]]) { assert.throws(() => buildPlateauTrace(input), RangeError); rejected += 1; }
assert.deepEqual(parseValidationMetrics('1, .5 0\n.125'), [1, .5, 0, .125]);
const fixturePath = `${directory}/model-fixtures.json`;
fs.writeFileSync(fixturePath, JSON.stringify({ schedules, noise, clocks, plateau }));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const oracle = spawnSync(python, ['scripts/verify-learning-rate-schedule-native.py', fixturePath], { encoding: 'utf8', timeout: 120000 });
assert.equal(oracle.status, 0, oracle.stderr || oracle.stdout);
const examples = [];
for (const [key, example] of Object.entries(learningRateScheduleExamples)) {
  const filename = `${directory}/programs/${key}.py`;
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  fs.writeFileSync(filename, example.code + '\n');
  const result = spawnSync(python, ['-I', filename], { encoding: 'utf8', timeout: 60000 });
  assert.equal(result.status, 0, `${key}: ${result.stderr}`);
  assert.equal(result.stderr, '', `${key}: unexpected warning`);
  assert.equal(result.stdout.replaceAll('\r\n', '\n').trimEnd(), example.expected.trimEnd(), `${key}: displayed stdout`);
  examples.push({ key, passed: true });
}
const result = { checkedAt: new Date().toISOString(), scheduleCases: schedules.length, noiseCases: noise.length, clockCases: clocks.length, plateauCases: plateau.length, rejected, independent: JSON.parse(oracle.stdout), examples };
fs.writeFileSync(`${directory}/verification-results.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
