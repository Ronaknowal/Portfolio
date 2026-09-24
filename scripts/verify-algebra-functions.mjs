import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/algebra-functions-models.js';
import { algebraFunctionsExamples } from '../src/learn/data/algebra-functions-examples.js';

const directory = 'scratch/algebra-functions-verification';
fs.mkdirSync(directory, { recursive: true });
const payload = { equations: [], functions: [], compositions: [], quadratics: [], growth: [], logs: [], examples: algebraFunctionsExamples };
for (let a = -7; a <= 7; a++) for (let b = -5; b <= 5; b++) for (let c = -5; c <= 5; c++) payload.equations.push(model.equationState(a, b, c));
payload.equations.push(model.equationState(30, -30, 30), model.equationState(-30, 30, -30), model.equationState(3, 6, 21));
for (let i = -16; i <= 16; i++) {
  for (const kind of ['affine', 'square', 'reciprocal']) for (const restricted of [false, true]) payload.functions.push(model.functionProbeState(kind, i / 4, restricted));
  payload.compositions.push(model.compositionState(i / 4));
}
for (let h = -4; h <= 8; h++) for (let k = -18; k <= 8; k++) payload.quadratics.push(model.quadraticState(h / 2, k / 2));
for (const rate of [-.5, -.2, 0, .1, .2, .5, 1]) for (let n = 0; n <= 8; n++) for (const factor of [.125, .5, 1, 2, 3, 8]) payload.growth.push(model.growthState(rate, n, factor));
for (const base of [.5, 2, 10]) for (let i = -12; i <= 12; i++) payload.logs.push(model.logarithmState(base, i / 4));
const errors = [
  () => model.parseEquationDraft({ a: '', b: '2', c: '3' }),
  () => model.parseEquationDraft({ a: '2.5', b: '2', c: '3' }),
  () => model.parseEquationDraft({ a: 'Infinity', b: '2', c: '3' }),
  () => model.equationState(31, 0, 1), () => model.equationState(NaN, 0, 1),
  () => model.functionProbeState('bad', 1), () => model.functionProbeState('reciprocal', 1e-300),
  () => model.compositionState(Infinity), () => model.quadraticState(0, -10),
  () => model.growthState(-1, 2), () => model.growthState(.2, 2.5), () => model.growthState(.2, 1, 0),
  () => model.logarithmState(1, 2), () => model.logarithmState(2, Infinity), () => model.algebraNumber(Infinity),
  () => model.functionProbeState('square', 1e-200), () => model.compositionState(1e-200),
  () => model.quadraticState(1, -1e-300), () => model.growthState(1e-320, 2), () => model.logarithmState(2, 1e-300),
];
for (const fail of errors) assert.throws(fail, RangeError);
assert.equal(model.algebraNumber(1e-12), '1.0000e-12');
assert.equal(model.functionProbeState('reciprocal', 0).y, null);
assert.equal(model.parseEquationDraft({ a: '+3', b: ' 6 ', c: '21' }).solution, 5);
payload.rejectionCases = errors.length;
fs.writeFileSync(`${directory}/model-cases.json`, JSON.stringify(payload));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const result = spawnSync(python, ['scripts/verify-algebra-functions.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout); process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status || 1);
