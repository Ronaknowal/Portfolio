import fs from 'node:fs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import * as models from '../src/learn/data/sampling-measurement-models.js';
import { samplingMeasurementExamples } from '../src/learn/data/sampling-measurement-examples.js';

const directory = 'scratch/sampling-measurement-verification';
fs.mkdirSync(directory, { recursive: true });
const cases = { samples: [], inclusion: [], readings: [], assignments: [], factorial: [], missing: [] };
for (let code = 0; code < 81; code += 1) {
  const population = Array.from({ length: 4 }, (_, i) => [-3, 0, 7][Math.floor(code / 3 ** i) % 3]);
  for (const frame of [[0, 1, 2, 3], [0, 2, 3], [2]]) {
    for (let n = 1; n <= frame.length; n += 1) cases.samples.push(models.finiteSampleState(population, frame, n));
  }
  for (const mode of ['equal', 'unequal', 'uncovered']) cases.inclusion.push(models.inclusionDesignState(mode, population));
}
cases.samples.push(models.finiteSampleState([5], [0], 1));
for (const G of [1, 2, 4, 8, 16]) for (const m of [1, 2, 4, 8, 16]) {
  for (const u of [0, .25, 1, 4, 9]) for (const e of [0, .25, 1, 4, 9]) {
    for (const b of [-3, 0, 2]) cases.readings.push(models.groupedMeasurementState(G, m, u, e, b));
  }
}
for (let code = 0; code < 243; code += 1) {
  const baseline = [2, ...Array.from({ length: 5 }, (_, i) => [-3, 0, 5][Math.floor(code / 3 ** i) % 3])];
  for (const design of ['complete', 'prognostic', 'mixed']) for (const effect of ['constant', 'heterogeneous']) cases.assignments.push(models.assignmentState(design, effect, baseline));
}
for (let interaction = -6; interaction <= 6; interaction += 1) {
  for (let share = 0; share <= 4; share += 1) cases.factorial.push(models.factorialState(interaction, share / 4));
}
for (let code = 0; code < 256; code += 1) {
  const values = Array.from({ length: 4 }, (_, i) => [null, 0, 3, 10][Math.floor(code / 4 ** i) % 4]);
  cases.missing.push({ values, state: models.boundedMissingMean(values) });
}
let rejected = 0;
for (const call of [
  () => models.finiteSampleState([], null, 1),
  () => models.finiteSampleState([1, 2], [0, 0], 1),
  () => models.finiteSampleState([1, 2], [3], 1),
  () => models.finiteSampleState([1, 2], null, 1.5),
  () => models.finiteSampleState([1, 2], null, true),
  () => models.finiteSampleState([1, Infinity], null, 1),
  () => models.finiteSampleState([1, 2], [], 1),
  () => models.finiteSampleState([1, 2], null, 3),
  () => models.inclusionDesignState('absent'),
  () => models.inclusionDesignState('equal', [1, 2, 3]),
  () => models.groupedMeasurementState(0, 1),
  () => models.groupedMeasurementState(1, 1, -1),
  () => models.groupedMeasurementState(1, 1, 1, NaN),
  () => models.assignmentState('other'),
  () => models.assignmentState('complete', 'other'),
  () => models.factorialState(7),
  () => models.factorialState(0, 1.1),
  () => models.boundedMissingMean([]),
  () => models.boundedMissingMean([undefined]),
  () => models.boundedMissingMean([11]),
  () => models.finiteMoments([1, 2], [.3, .3]),
  () => models.finiteMoments([0, 1e-200]),
  () => models.groupedMeasurementState(16, 16, Number.MIN_VALUE, 0),
  () => models.groupedMeasurementState(1, 16, 0, Number.MIN_VALUE),
  () => models.groupedMeasurementState(1, 1, 0, 0, 1e-200),
]) {
  assert.throws(call, RangeError);
  rejected += 1;
}
assert(models.finiteMoments([0, 1e-100]).variance > 0);
assert(models.groupedMeasurementState(16, 16, 1e-100, 1e-100, 1e-100).variance > 0);
fs.writeFileSync(`${directory}/cases.json`, JSON.stringify({ cases, examples: samplingMeasurementExamples }));
const run = spawnSync(process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe', ['-X', 'utf8', 'scripts/verify-sampling-measurement.py'], { encoding: 'utf8', maxBuffer: 4e6 });
if (run.status !== 0) throw new Error(run.stderr || run.stdout);
const evidence = { ...JSON.parse(run.stdout), rejectedInputs: rejected, timestamp: new Date().toISOString() };
evidence.hashes = Object.fromEntries(['src/learn/data/sampling-measurement-models.js', 'src/learn/data/sampling-measurement-examples.js'].map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
fs.writeFileSync(`${directory}/results.json`, JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));
