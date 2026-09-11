import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as models from '../src/learn/data/conditioning-stability-models.js';
import { conditioningStabilityExamples } from '../src/learn/data/conditioning-stability-examples.js';

const directory = path.resolve('scratch/conditioning-stability-native');
fs.mkdirSync(directory, { recursive: true });
const payload = { examples: conditioningStabilityExamples, rounding: [], cancellation: [], sensitivity: [], backward: [], summation: [], propagation: [], refinement: [], exactFloats: [] };
for (let bin = 0; bin <= 1; bin += 1) for (let half = 0; half <= 16; half += 1) payload.rounding.push({ bin, half, state: models.roundingCellState(bin, half) });
for (const sign of [-1, 0, 1]) for (let exponent = 0; exponent <= 60; exponent += 1) payload.cancellation.push({ sign, exponent, state: models.cancellationState(exponent, sign) });
for (let exponent = 1; exponent <= 7; exponent += 1) for (let change = -2; change <= 2; change += 1) for (const singular of [false, true]) payload.sensitivity.push({ exponent, change, singular, state: models.measurementSensitivityState(exponent, change, singular) });
for (let power = 1; power <= 9; power += 1) for (const scaled of [false, true]) payload.backward.push({ power, scaled, state: models.backwardWitnessState(power, scaled) });
for (const preset of Object.keys(models.summationPresets)) payload.summation.push({ preset, state: models.summationState(preset) });
for (const q of [-0.5, 0.5, 0.9, 1, 1.1]) for (const mode of ['pulse', 'constant', 'alternating']) for (let steps = 1; steps <= 24; steps += 1) payload.propagation.push({ q, mode, steps, state: models.propagationState(q, mode, steps) });
for (const steps of [4, 8, 16, 32]) payload.refinement.push(models.unstableRefinementState(steps));
for (const value of [0, -0, Number.MIN_VALUE, -Number.MIN_VALUE, 1e-308, 1e-100, 0.1, -0.3, 1, 1e16, Number.MAX_VALUE]) payload.exactFloats.push({ value, fraction: models.storedNumberFraction(value).map(String) });
let rejections = 0;
for (const action of [
  () => models.roundingCellState(2), () => models.roundingCellState(0, 0.5),
  () => models.cancellationState(-1), () => models.cancellationState(61), () => models.cancellationState(4, NaN),
  () => models.measurementSensitivityState(0), () => models.measurementSensitivityState(4, 3), () => models.measurementSensitivityState(4, 0, 1),
  () => models.backwardWitnessState(0), () => models.backwardWitnessState(5, 'yes'),
  () => models.summationState('constructor'), () => models.summationState('__proto__'), () => models.summationState(''),
  () => models.propagationState(0.8), () => models.propagationState(1, 'constructor'), () => models.propagationState(1, 'pulse', 25),
  () => models.unstableRefinementState(7), () => models.storedNumberFraction(Infinity), () => models.storedNumberFraction(NaN)
]) { assert.throws(action); rejections += 1; }
assert.equal(models.formatConditioning(1e-12), '1.000000e-12');
assert.equal(models.formatConditioning(0), '0');
payload.javascriptRejections = rejections;
fs.writeFileSync(path.join(directory, 'production-payload.json'), JSON.stringify(payload, (_, value) => typeof value === 'bigint' ? String(value) : value, 2));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const result = spawnSync(python, ['scripts/verify-conditioning-stability-native.py'], { cwd: process.cwd(), encoding: 'utf8' });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0, 'Independent native verifier failed');
