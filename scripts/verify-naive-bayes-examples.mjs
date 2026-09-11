import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { naiveBayesExamples } from '../src/learn/data/naive-bayes-examples.js';
import {
  tokenEvidenceState, presenceEvidenceState, gaussianObservationState,
  gaussianGeometryState, copiedAlarmState, reliabilityState,
  normalizeNaiveBayesScores, tokenizeNaiveBayesMessage
} from '../src/learn/data/naive-bayes-models.js';

const target = 'scratch/naive-bayes-verification';
fs.mkdirSync(target, { recursive: true });
const encode = (_, value) => value === Infinity ? '+Infinity' : value === -Infinity ? '-Infinity' : value;
const cases = { token: [], presence: [], gaussian: [], geometry: [], copied: [], reliability: [] };
const words = ['free', 'money', 'win', 'meeting', 'agenda'];
for (const alpha of [0, 0.25, 1, 10]) {
  for (let length = 0; length <= 4; length += 1) {
    for (let number = 0; number < 5 ** length; number += 1) {
      let remainder = number;
      const tokens = [];
      for (let j = 0; j < length; j += 1) {
        tokens.push(words[remainder % 5]);
        remainder = Math.floor(remainder / 5);
      }
      const state = tokenEvidenceState(tokens.join(' '), alpha);
      cases.token.push({ text: state.text, alpha, probabilities: state.final.probabilities,
        scores: state.final.scores, frames: state.frames.map(frame => ({ probabilities: frame.probabilities, scores: frame.scores })) });
    }
  }
}
for (let number = 0; number < 3 ** 5; number += 1) {
  let remainder = number;
  const counts = Array.from({ length: 5 }, () => {
    const value = remainder % 3;
    remainder = Math.floor(remainder / 3);
    return value;
  });
  cases.presence.push(presenceEvidenceState(counts));
}
for (let i = -50; i <= 50; i += 2) {
  for (const width of [0.1, 0.4, 1]) cases.gaussian.push(gaussianObservationState(i / 10, width));
}
for (const mode of ['equal', 'unequal']) {
  for (const point of [[0, 0], [-2, -2], [2, 2], [-8, -8], [6, 6], [-4, 4], [0.25, -0.75]]) {
    cases.geometry.push(gaussianGeometryState(mode, point));
  }
}
for (const prior of [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) {
  for (let copies = 1; copies <= 5; copies += 1) {
    for (const positive of [true, false]) cases.copied.push(copiedAlarmState(copies, positive, prior));
  }
}
for (let bins = 2; bins <= 6; bins += 1) {
  for (const compression of [false, true]) cases.reliability.push(reliabilityState(bins, compression));
}
const rejected = [];
const invalid = [
  ['sparse scores', () => normalizeNaiveBayesScores([, 1])],
  ['NaN score', () => normalizeNaiveBayesScores([NaN, 1])],
  ['positive infinity', () => normalizeNaiveBayesScores([Infinity, 1])],
  ['overflow difference', () => normalizeNaiveBayesScores([-Number.MAX_VALUE, Number.MAX_VALUE])],
  ['negative alpha', () => tokenEvidenceState('free', -1)],
  ['tiny unsupported alpha', () => tokenEvidenceState('free', 1e-300)],
  ['nonfinite alpha', () => tokenEvidenceState('free', Infinity)],
  ['oversize message', () => tokenEvidenceState('x'.repeat(301))],
  ['too many tokens', () => tokenEvidenceState(Array(31).fill('a').join(' '))],
  ['sparse presence', () => presenceEvidenceState([, 0, 0, 0, 0])],
  ['fractional count', () => presenceEvidenceState([0.5, 0, 0, 0, 0])],
  ['oversize probe', () => gaussianGeometryState('equal', [9, 0])],
  ['invalid mode', () => gaussianGeometryState('constructor')],
  ['invalid copies', () => copiedAlarmState(2.1)],
  ['invalid bins', () => reliabilityState(0)]
];
for (const [name, action] of invalid) {
  let threw = false;
  try { action(); } catch { threw = true; }
  if (!threw) throw new Error('Expected rejection: ' + name);
  rejected.push(name);
}
if (normalizeNaiveBayesScores([-Infinity, -Infinity]).probabilities !== null) throw new Error('Undefined posterior concealed');
const unknown = tokenizeNaiveBayesMessage('FREE, zephyronic free!');
if (unknown.knownTokens.join(' ') !== 'free free' || unknown.unknownTokens.join(' ') !== 'zephyronic') throw new Error('Tokenizer contract');
fs.writeFileSync(target + '/model-cases.json', JSON.stringify(cases, encode));
fs.writeFileSync(target + '/examples.json', JSON.stringify(naiveBayesExamples));
fs.writeFileSync(target + '/js-validation.json', JSON.stringify({ at: new Date().toISOString(), rejected,
  caseCounts: Object.fromEntries(Object.entries(cases).map(([key, value]) => [key, value.length])) }, null, 2));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-naive-bayes-native.py'], {
  encoding: 'utf8', timeout: 180000, maxBuffer: 10 * 1024 * 1024
});
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.error || result.status !== 0) process.exit(result.status || 1);
