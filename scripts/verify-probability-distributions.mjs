import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { probabilityDistributionExamples } from '../src/learn/data/probability-distributions-examples.js';
import { arrivalWindowState, bayesPopulationState, chooseCount, eventConditionState, mixedDelayState, pairedEvidenceState, probabilityNumber, urnCountState } from '../src/learn/data/probability-distributions-models.js';

const destination = resolve('scratch/probability-distributions-review/native');
mkdirSync(destination, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
const examples = [];
for (const [name, example] of Object.entries(probabilityDistributionExamples)) {
  const filename = resolve(destination, `${name}.py`);
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-X', 'utf8', '-I', filename], { encoding: 'utf8', timeout: 60000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr || result.error}`);
  const stdout = result.stdout.replace(/\r\n/g, '\n').trim();
  assert.equal(stdout, example.expected.trim(), `${name}: displayed output differs`);
  examples.push({ name, stdout });
}

const cases = { events: [], bayes: [], pairs: [], urns: [], delays: [], arrivals: [] };
const faces = mask => Array.from({ length: 6 }, (_, index) => index + 1).filter(face => mask & (1 << (face - 1)));
for (let first = 0; first < 64; first += 1) {
  for (let second = 0; second < 64; second += 1) {
    cases.events.push({ first: faces(first), second: faces(second), state: eventConditionState(faces(first), faces(second)) });
  }
}
for (const prior of [0, 0.001, 0.01, 0.1, 0.5, 0.9, 1]) {
  for (const sensitivity of [0, 0.01, 0.2, 0.5, 0.8, 0.95, 1]) {
    for (const falsePositive of [0, 0.01, 0.1, 0.5, 0.8, 0.99, 1]) {
      cases.bayes.push({ prior, sensitivity, falsePositive, state: bayesPopulationState(prior, sensitivity, falsePositive) });
      for (const copy of [0, 0.25, 0.5, 0.75, 1]) {
        cases.pairs.push({ prior, sensitivity, falsePositive, copy, state: pairedEvidenceState(prior, sensitivity, falsePositive, copy) });
      }
    }
  }
}
for (let marked = 0; marked <= 6; marked += 1) {
  for (let draws = 0; draws <= 6; draws += 1) {
    for (const replacement of [false, true]) {
      for (let threshold = 0; threshold <= 6; threshold += 1) {
        cases.urns.push({ marked, draws, replacement, threshold, state: urnCountState(marked, draws, replacement, threshold) });
      }
    }
  }
}
for (const width of [0.1, 0.2, 0.4, 1, 2]) {
  for (const atom of [0, 0.2, 0.3, 0.9, 1]) {
    for (const left of [0, 0.25, 0.5, 0.75, 1]) {
      for (const right of [0, 0.25, 0.5, 0.75, 1].filter(value => value >= left)) {
        for (const unit of ['seconds', 'milliseconds']) {
          cases.delays.push({ width, atom, left, right, unit, state: mixedDelayState(width, atom, left, right, unit) });
        }
      }
    }
  }
}
for (let rateIndex = 1; rateIndex <= 24; rateIndex += 1) {
  for (let windowIndex = 0; windowIndex <= 20; windowIndex += 1) {
    for (const quantile of [0.01, 0.1, 0.5, 0.9, 0.99]) {
      const rate = rateIndex / 4;
      const window = windowIndex / 10;
      cases.arrivals.push({ rate, window, quantile, state: arrivalWindowState(rate, window, quantile) });
    }
  }
}
const invalid = [
  () => eventConditionState([1, 1]), () => eventConditionState([7]), () => eventConditionState(null),
  () => bayesPopulationState(-0.1), () => bayesPopulationState(0.5, Infinity), () => pairedEvidenceState(0.5, 0.8, 0.1, 1.1),
  () => urnCountState(7), () => urnCountState(2, 1.5), () => urnCountState(2, 3, 'false'),
  () => mixedDelayState(0), () => mixedDelayState(0.2, 0.3, 0.8, 0.2), () => mixedDelayState(0.2, 0, 0, 1, 'minutes'),
  () => arrivalWindowState(0), () => arrivalWindowState(1, 3), () => arrivalWindowState(1, 1, 1), () => chooseCount(31, 1),
  () => probabilityNumber(NaN),
];
invalid.forEach(check => assert.throws(check, RangeError));
assert.equal(probabilityNumber(null), 'undefined: zero evidence');
assert.notEqual(probabilityNumber(1e-20), '0');
assert.notEqual(probabilityNumber(1 - 1e-12), '1');
writeFileSync(resolve(destination, 'model-cases.json'), JSON.stringify(cases));
const oracle = spawnSync(python, ['-X', 'utf8', '-I', 'scripts/verify-probability-distributions-native.py', destination], { encoding: 'utf8', timeout: 60000 });
assert.equal(oracle.status, 0, oracle.stderr || oracle.error);
console.log(oracle.stdout.trim());
const result = { checkedAt: new Date().toISOString(), status: 'passed', examples, modelCases: Object.fromEntries(Object.entries(cases).map(([key, values]) => [key, values.length])), invalidCases: invalid.length };
writeFileSync(resolve(destination, 'verification.json'), JSON.stringify(result, null, 2) + '\n');
console.log(`Passed ${examples.length} exact native programs, independent model oracles and ${invalid.length} invalid-input cases.`);
