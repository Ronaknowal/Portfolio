import fs from 'node:fs';
import assert from 'node:assert/strict';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/real-analysis-models.js';
import { realAnalysisExamples as examples } from '../src/learn/data/real-analysis-examples.js';

const directory = 'scratch/real-analysis-verification';
fs.mkdirSync(directory, { recursive: true });
const data = { examples, brackets: [], bernstein: [], triangles: [], tails: [], cauchy: [], powers: [], derivatives: [], series: [], typewriter: [] };
for (const target of [2, 3, 5]) for (const steps of [0, 1, 7, 19, 24]) data.brackets.push(model.dyadicBracket(target, steps));
for (const n of [1, 3, 7, 13, 64, 256]) for (const x of [0, 1 / 16, 3 / 8, 13 / 16, 1]) data.bernstein.push(model.bernsteinApproximation(n, x, 7 / 16, 3));
for (const n of [3, 7, 19, 127, 4096]) for (const scaling of ['unit-height', 'unit-area', 'shrinking-height']) data.triangles.push(model.triangleFamily(n, scaling, 12, 5 / 16));
for (const p of [1, 7, 127]) for (const q of [3, 13, 999]) for (const index of [1, 7, 999]) data.tails.push({ p, q, index, state: model.sequenceTail(p, q, index) });
for (const n of [1, 3, 17, 128, 512]) for (const family of ['harmonic', 'telescoping']) data.cauchy.push(model.cauchyBlock(n, family));
for (const n of [1, 3, 32, 256]) for (const domain of ['closed-unit', 'open-unit', 'compact-subinterval']) for (const x of [0, 0.25, 0.75]) data.powers.push(model.powerFamily(n, domain, 0.75, x));
for (const n of [1, 3, 8, 23, 48]) for (const power of [1, 2]) for (const x of [-Math.PI, -0.3, 0, 0.4, Math.PI]) data.derivatives.push(model.derivativeFamily(n, power, x));
for (const n of [1, 7, 32, 256, 1024]) for (const x of [-1, -0.75, 0, 0.5, 1]) data.series.push(model.inverseSquareSeries(n, x));
for (const block of [0, 1, 3, 7, 12]) for (const position of [...new Set([0, 2 ** block - 1, Math.floor(2 ** block / 2)])]) for (const observer of [0, 0.25, 0.5, 0.999]) data.typewriter.push(model.typewriterInterval(block, position, observer));
data.tinyBound = model.bernsteinApproximation(256, Number.MIN_VALUE, 0.3, 1);
assert(data.tinyBound.pointwiseBound > 0);
assert.equal(model.bernsteinApproximation(256, 0.5, 0.3, 0).pointwiseBound, 0);
assert.equal(model.bernsteinApproximation(256, 0, 0.3, 1).pointwiseBound, 0);
assert.equal(model.bernsteinApproximation(256, 1, 0.3, 1).pointwiseBound, 0);
const invalid = [
  () => model.bernsteinApproximation(256, 0.5, 0.3, Number.MIN_VALUE),
  () => model.sequenceTail(0, 10, 1), () => model.sequenceTail(1, NaN, 1), () => model.sequenceTail(1, 10, 1.5),
  () => model.dyadicBracket(4, 3), () => model.dyadicBracket(2, 25),
  () => model.cauchyBlock(1, 'other'), () => model.cauchyBlock(Infinity, 'harmonic'),
  () => model.powerFamily(2, 'open-unit', 0.5, 1), () => model.powerFamily(2, 'closed-unit', 1, 0.5),
  () => model.triangleValue(-1, 4, 'unit-height'), () => model.triangleFamily(1, 'unit-height'),
  () => model.triangleFamily(4, 'unit-height', 129), () => model.derivativeFamily(3, 3),
  () => model.derivativeFamily(3, 1, 4), () => model.inverseSquareSeries(3, 1.01),
  () => model.inverseSquareSeries(3.5, 1), () => model.bernsteinWeights(257, 0.5),
  () => model.bernsteinWeights(3, '0.5'), () => model.bernsteinApproximation(3, 0.5, 1.1),
  () => model.typewriterInterval(-1, 0, 0), () => model.typewriterInterval(2, 0.5, 0.5),
  () => model.typewriterInterval(2, 4, 0.5), () => model.typewriterInterval(2, 0, 1)
];
for (const call of invalid) assert.throws(call, RangeError);
data.invalidInputsRejected = invalid.length;
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(data));
const result = spawnSync(path.resolve('scratch/lesson-tools/Scripts/python.exe'), ['scripts/verify-real-analysis-native.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.status !== 0) throw new Error('Real analysis verification failed');
