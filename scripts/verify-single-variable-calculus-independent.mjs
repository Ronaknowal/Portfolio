import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/single-variable-calculus-models.js';
import { singleVariableCalculusExamples as examples } from '../src/learn/data/single-variable-calculus-examples.js';

const directory = 'scratch/single-variable-calculus-independent-review';
fs.mkdirSync(directory, { recursive: true });
const oldRecord = `${directory}/initial-author-native-record.json`;
if (!fs.existsSync(oldRecord)) fs.copyFileSync('scratch/single-variable-calculus-verification/results.json', oldRecord);
const data = { limits: [], extrema: [], accumulations: [], taylors: [], improper: [], growth: [], examples };
const buffer = new ArrayBuffer(8);
const view = new DataView(buffer);
function adjacent(value, direction) {
  view.setFloat64(0, value);
  view.setBigUint64(0, view.getBigUint64(0) + BigInt(direction));
  return view.getFloat64(0);
}
for (const delta of [.0025, .005, .01, .025, .05, .1, .125, .2]) {
  const threshold = delta * (4 + delta);
  for (const epsilon of [adjacent(threshold, -1), threshold, adjacent(threshold, 1)]) {
    for (const kind of ['smooth', 'hole', 'jump']) data.limits.push(model.limitGuarantee(kind, epsilon, delta));
  }
}
const intervals = [[0, Number.MIN_VALUE], [0, 1e-200], [1e-200, 2e-200],
  [adjacent(1, -1), 1], [1, adjacent(1, 1)], [adjacent(1, -1), adjacent(1, 1)],
  [adjacent(2, -1), 2], [2, adjacent(2, 1)], [adjacent(2, -1), adjacent(2, 1)],
  [adjacent(3, -1), 3], [3, adjacent(3, 1)], [adjacent(3, -1), adjacent(3, 1)],
  [adjacent(4, -1), 4], [.017, .831], [.531, 3.173], [1.173, 2.999], [0, 4]];
for (const kind of ['motion', 'inflection', 'cusp']) {
  for (const [left, right] of intervals) data.extrema.push(model.extremaCandidates(kind, left, right));
}
for (const upper of [.271, .999, 1, 1.001, 2.137, 2.999, 3, 3.001, 3.913, 4]) {
  for (const n of [1, 3, 7, 19, 53, 128]) for (const tag of ['left', 'midpoint', 'right']) data.accumulations.push(model.motionAccumulation(upper, n, tag));
}
for (const kind of ['exp', 'log']) for (const degree of [0, 1, 3, 7, 12]) {
  for (const input of [-.875, -.31, -.05, .001, .05, .1, .15, .2, .73, 1, 1.5]) data.taylors.push(model.taylorApproximation(kind, degree, input));
}
for (const kind of ['tail', 'endpoint']) for (const p of [.0001, .713, .9999999999999999, 1, 1.0000000000000002, 1.313, 2.917]) {
  for (const q of [.003, 1.713, 5.97]) data.improper.push(model.improperPowerIntegral(kind, p, q));
}
for (const rate of [-.713, -.000001, 0, .000001, .317, .783]) for (const period of [.271, .913, 1.731]) data.growth.push(model.exponentialRate(rate, period));
data.modelSource = { path: 'src/learn/data/single-variable-calculus-models.js', sha256: crypto.createHash('sha256').update(fs.readFileSync('src/learn/data/single-variable-calculus-models.js')).digest('hex') };
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify(data));
const run = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-single-variable-calculus-independent.py'], { encoding: 'utf8' });
process.stdout.write(run.stdout);
process.stderr.write(run.stderr);
if (run.status !== 0) process.exit(run.status ?? 1);
