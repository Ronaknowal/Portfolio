import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as m from '../src/learn/data/stochastic-processes-models.js';
import { stochasticProcessesExamples as examples } from '../src/learn/data/stochastic-processes-examples.js';

const directory = 'scratch/stochastic-processes-native-verification';
fs.mkdirSync(directory, { recursive: true });
const laws = [];
for (const kind of ['fresh', 'frozen', 'alternating']) {
  for (const length of [2, 3, 4, 8]) laws.push(m.finiteProcessLaw(kind, length));
}
const markov = [];
for (const a of [0, 1e-6, 0.02, 0.2, 0.5, 0.8, 1]) {
  for (const b of [0, 1e-6, 0.03, 0.3, 0.5, 0.9, 1]) {
    for (const initialSunny of [0, 0.25, 0.5, 1]) {
      markov.push(m.markovState({ a, b, initialSunny, steps: 60 }));
    }
  }
}
const absorption = [];
for (let boundary = 3; boundary <= 8; boundary += 1) {
  for (const upward of [0.1, 0.25, 0.5, 0.75, 0.9]) {
    for (let start = 0; start <= boundary; start += 1) {
      absorption.push(m.absorptionState({ boundary, upward, start, steps: 16 }));
    }
  }
}
const arrivals = [];
for (const rates of [[0, 0], [0, 4], [1, 0], [1, 4], [2.5, 2.5], [0.01, 0.01], [6, 6]]) {
  for (const interval of [[0, 1], [1, 2], [2, 3], [0, 3]]) {
    for (const seed of [0, 11, 97]) {
      for (const routing of ['independent', 'alternating']) arrivals.push(m.arrivalState({ rates, interval, seed, routing }));
    }
  }
}
arrivals.push(m.arrivalState({ rates: [6, 6], maxEvents: 1 }));
arrivals.push(m.arrivalState({ rates: [6, 6], maxEvents: 1, interval: [0, 0] }));
const shortInterval = m.arrivalState({ rates: [6, 0.01], interval: [2, 2 + 2 ** -51] });
assert.equal(shortInterval.intervalMean, 0.01 * 2 ** -51);
const split = [];
for (const mean of [0, 0.01, 1, 2.5, 12, 36]) {
  for (const probability of [0, 0.1, 0.4, 0.5, 1]) {
    for (let a = 0; a <= 5; a += 1) {
      for (let b = 0; b <= 5; b += 1) split.push({ mean, a, b, probability, state: m.splitCountLaw(mean, a, b, probability) });
    }
  }
}
const clocks = [];
for (const alpha of [0.05, 0.5, 1, 6]) {
  for (const beta of [0.05, 0.5, 2, 6]) {
    for (const horizon of [0.1, 4, 12]) {
      for (const initial of [0, 1]) clocks.push(m.jumpClockState({ alpha, beta, horizon, initial, seed: 11 }));
    }
  }
}
clocks.push(m.jumpClockState({ alpha: 6, beta: 6, horizon: 12, maxJumps: 1 }));
const brownian = [];
for (const horizon of [0.1, 1, 2, 4]) {
  for (const drift of [-0.5, 0, 0.3]) {
    for (const scale of [0.05, 0.7, 1, 4]) {
      for (const level of [2, 4, 8]) brownian.push(m.brownianState({ horizon, drift, scale, level, seed: 11 }));
    }
  }
}
const basis = [];
for (const count of [2, 4, 8, 16]) {
  for (const horizon of [0.1, 1, 2, 4]) {
    for (const scale of [0.2, 1, 2]) {
      basis.push({ count, horizon, scale, paths: Array.from({ length: count }, (_, index) =>
        m.brownianFromNormals(Array.from({ length: count }, (__, j) => Number(index === j)), { horizon, scale })) });
    }
  }
}
const bridges = [];
for (const left of [-1, 0, 0.2]) {
  for (const right of [-0.1, 0, 0.7]) {
    for (const duration of [0.01, 0.5, 1, 12]) {
      for (const scale of [0.05, 0.8, 1, 4]) {
        for (const fraction of [0, 0.25, 0.5, 1]) {
          bridges.push(m.bridgeState({ left, right, duration, scale, fraction, barrier: 0.7 }));
        }
      }
    }
  }
}
const transitions = [
  [[0, 1, 1], [2, 2, 0]],
  [[0], [1], [2]],
  [[0, 0, 1, 1, 0], [1, 0, 1]],
].map(paths => ({ paths, state: m.transitionEstimate(paths) }));

const invalid = [
  () => m.markovState({ a: NaN }), () => m.markovState({ b: 1e-8 }),
  () => m.markovState({ steps: 61 }), () => m.markovState({ initialSunny: -0.1 }),
  () => m.absorptionState({ boundary: 2 }), () => m.absorptionState({ start: 5 }),
  () => m.absorptionState({ upward: 0 }), () => m.absorptionState({ steps: -1 }),
  () => m.arrivalState({ rates: [0.001, 1] }), () => m.arrivalState({ interval: [2, 1] }),
  () => m.arrivalState({ horizon: 0 }), () => m.arrivalState({ rates: [1, Infinity] }),
  () => m.arrivalState({ seed: -1 }), () => m.arrivalState({ maxEvents: 0 }),
  () => m.jumpClockState({ alpha: 0 }), () => m.jumpClockState({ initial: 2 }),
  () => m.jumpClockState({ maxJumps: 301 }), () => m.jumpClockState({ horizon: 13 }),
  () => m.brownianState({ level: 9 }), () => m.brownianState({ pathIndex: 8 }),
  () => m.brownianState({ scale: 0 }), () => m.brownianFromNormals([Infinity]),
  () => m.brownianFromNormals([]), () => m.bridgeState({ duration: 0 }),
  () => m.bridgeState({ fraction: 1.1 }), () => m.bridgeState({ left: NaN }),
  () => m.transitionEstimate([[3]]), () => m.transitionEstimate([]),
  () => m.parseProcessNumber('', 0, 1, 'p'), () => m.parseProcessNumber('1e-9999', 0, 1, 'p'),
  () => m.parseProcessNumber('0xff', 0, 300, 'p'), () => m.parseProcessNumber('Infinity', 0, 1, 'p'),
  () => m.integratedRate(Number.MIN_VALUE, [Number.MIN_VALUE, 1], 1),
  () => m.bridgeState({ fraction: Number.MIN_VALUE }),
];
invalid.forEach(check => assert.throws(check, RangeError));
function frozen(value) {
  if (!value || typeof value !== 'object') return;
  assert.ok(Object.isFrozen(value));
  Object.values(value).forEach(frozen);
}
[laws, markov, absorption, arrivals, clocks, brownian, bridges].forEach(group => group.forEach(frozen));
for (const seed of [0, 11, 4294967295]) {
  const draw = m.processUniforms(seed), same = m.processUniforms(seed);
  for (let index = 0; index < 10000; index += 1) {
    const value = draw();
    assert.ok(value > 0 && value < 1);
    assert.equal(value, same());
  }
}
fs.writeFileSync(directory + '/fixtures.json', JSON.stringify({
  laws, markov, absorption, arrivals, split, clocks, brownian, basis, bridges, transitions,
  examples, invalidInputs: invalid.length, uniformDrawChecks: 30000, shortInterval,
}));
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const result = spawnSync(python, ['scripts/verify-stochastic-processes-native.py'], {
  encoding: 'utf8', maxBuffer: 8 * 1024 * 1024,
});
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status || 1);
