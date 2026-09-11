import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/geometry-trigonometry-models.js';
import { geometryTrigonometryExamples } from '../src/learn/data/geometry-trigonometry-examples.js';

const directory = 'scratch/geometry-trigonometry-verification';
fs.mkdirSync(directory, { recursive: true });
const payload = { dissections: [], arcs: [], similar: [], circles: [], bearings: [], frames: [], ambiguous: model.ambiguousTriangleState(), examples: geometryTrigonometryExamples };
for (let a = 1; a <= 20; a++) for (let b = 1; b <= 20; b++) payload.dissections.push(model.dissectionState(a, b));
for (let radius = 1; radius <= 3; radius += .5) for (let degrees = 15; degrees <= 330; degrees += 15) payload.arcs.push(model.arcState(radius, degrees));
for (const shape of ['3-4-5', '5-12-13', 'equal-legs']) for (let scale = .5; scale <= 3; scale += .25) payload.similar.push(model.similarityState(shape, scale));
for (let degrees = -360; degrees <= 360; degrees += 15) payload.circles.push(model.circleComponents(degrees));
for (let x = -6; x <= 6; x++) for (let y = -6; y <= 6; y++) payload.bearings.push(model.bearingState(x, y));
for (const px of [-4, -2, 0, 2, 4]) for (const py of [-4, -2, 0, 2, 4]) for (const [ox, oy] of [[-2, -2], [0, 0], [1, -1], [2, 2]]) for (let degrees = -180; degrees <= 180; degrees += 15) for (const mode of ['active', 'passive']) payload.frames.push(model.frameState(px, py, ox, oy, degrees, mode));
const rejects = [
  () => model.arcState(0, 30), () => model.arcState(2, 360), () => model.arcState(2, 0), () => model.arcState(2, 15.1),
  () => model.circleComponents(1e-200), () => model.circleComponents(1e-323), () => model.circleComponents(361), () => model.circleComponents(NaN),
  () => model.bearingState(1e-200, 1), () => model.bearingState(7, 0), () => model.bearingState(0, Infinity),
  () => model.similarityState('wrong', 1), () => model.similarityState('3-4-5', -1), () => model.similarityState('3-4-5', .3),
  () => model.frameState(5, 0, 0, 0, 0, 'active'), () => model.frameState(0, 0, 3, 0, 0, 'active'), () => model.frameState(0, 0, 0, 0, 1, 'active'), () => model.frameState(0, 0, 0, 0, 0, 'wrong'),
  () => model.dissectionState(0, 1), () => model.formatGeometry(Infinity),
];
for (const reject of rejects) assert.throws(reject, RangeError);
assert.equal(model.formatGeometry(1e-12), '1.00e-12');
assert.equal(model.circleComponents(90).tangent, null);
assert.equal(model.bearingState(0, 0).angle, null);
payload.rejected = rejects.length;
fs.writeFileSync(`${directory}/cases.json`, JSON.stringify(payload));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const result = spawnSync(python, ['scripts/verify-geometry-trigonometry.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout); process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status || 1);
