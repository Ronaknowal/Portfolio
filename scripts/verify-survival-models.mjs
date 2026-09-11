// Topic-owned, bounded numerical verification. Unchanged displayed executions
// are reused by exact source identity; Python supplies separate finite oracles.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/survival-models.js';
import { survivalExamples } from '../src/learn/data/survival-examples.js';
const directory = 'scratch/survival/native-verification';
fs.mkdirSync(directory, {
  recursive: true
});
const hash = source => createHash('sha256').update(fs.readFileSync(source)).digest('hex');
const modelSource = 'src/learn/data/survival-models.js';
const exampleSource = 'src/learn/data/survival-examples.js';
const executionSource = 'scratch/survival/program-execution.json';
const previous = JSON.parse(fs.readFileSync(executionSource, 'utf8'));
assert.equal(hash(exampleSource), previous.sha256, 'Stored program executions must bind current examples.');
for (const [name, example] of Object.entries(survivalExamples)) {
  assert.equal(example.expected, previous.outputs[name], `${name}: reused stdout`);
}
const kmInputs = [{
  times: model.pumpTimes,
  events: model.pumpEvents
}, {
  times: [1, 2, 2, 4, 5],
  events: [true, true, false, true, false]
}];
for (const times of [[1, 2, 3, 4], [1, 1, 2, 4], [2, 2, 2, 2]]) {
  for (const mask of [0, 1, 3, 5, 10, 15]) {
    kmInputs.push({
      times,
      events: times.map((_, index) => Boolean(mask & 1 << index))
    });
  }
}
const km = kmInputs.map(input => {
  const table = model.kaplanMeier(input.times, input.events);
  const points = [...new Set([0, ...input.times.flatMap(time => [time / 2, time])])].sort((a, b) => a - b);
  return {
    input,
    table,
    evaluations: points.map(time => ({
      time,
      survival: model.survivalAt(table, time),
      area: model.restrictedMean(table, time).value
    }))
  };
});
const coxInputs = [{
  times: model.coxTimes,
  events: model.coxEvents,
  features: model.coxFeatures
}, {
  times: [2, 3, 1],
  events: [false, false, true],
  features: [0, 1, 2]
}, {
  times: [1, 2, 2, 4, 5, 6, 6, 8],
  events: [true, true, false, true, false, true, true, false],
  features: [1, -2, 0.5, 2, -1, 0, 1, -0.5]
}];
const cox = coxInputs.flatMap(input => ['efron', 'breslow'].flatMap(ties => [-0.8, 0, Math.log(2)].map(beta => {
  const arguments_ = {
    ...input,
    beta,
    ties
  };
  return {
    input: arguments_,
    actual: model.coxRiskSets(arguments_)
  };
})));
const logrank = kmInputs.slice(0, 14).map((input, index) => {
  const groups = input.times.map((_, position) => (position + index) % 2);
  return {
    input: {
      ...input,
      groups
    },
    actual: model.logrankTable(input.times, input.events, groups)
  };
});
const logrankPractice = {
  times: [1, 1, 2, 2, 2, 2, 2, 2],
  events: [true, true, false, false, false, false, false, false],
  groups: [1, 0, 1, 1, 0, 0, 0, 0]
};
logrank.push({
  input: logrankPractice,
  actual: model.logrankTable(logrankPractice.times, logrankPractice.events, logrankPractice.groups)
});
const pairInputs = [{
  times: [1, 2, 3],
  events: [false, true, true],
  scores: [9, 1, 2]
}, {
  times: [1, 1, 2, 2, 3],
  events: [true, true, false, true, false],
  scores: [1, 2, 2, 2, 0]
}, {
  times: [1, 2, 3, 4],
  events: [false, false, false, false],
  scores: [0, 3, 1, 2]
}, {
  times: [2, 2, 2],
  events: [true, true, true],
  scores: [1, 0, -1]
}, {
  times: [1, 2, 3],
  events: [true, true, false],
  scores: [1, 1 + 5e-9, 0]
}, {
  times: [1, 1, 2],
  events: [true, false, true],
  scores: [0, 1, 0]
}];
const pairs = pairInputs.map(input => ({
  input,
  actual: model.concordancePairs(input.times, input.events, input.scores)
}));
const cifInputs = [{
  times: [1, 2, 3, 4],
  statuses: [1, 2, 0, 1]
}, {
  times: [1, 1, 2, 2, 3, 4],
  statuses: [1, 2, 0, 2, 1, 0]
}, {
  times: [1, 2, 3],
  statuses: [0, 0, 0]
}, {
  times: [2, 2, 2, 2],
  statuses: [1, 2, 0, 1]
}, {
  times: [1, 2, 3, 4],
  statuses: [1, 0, 1, 0]
}, {
  times: [1, 2, 3, 4],
  statuses: [2, 0, 2, 0]
}];
const cif = cifInputs.map(input => ({
  input,
  actual: model.competingIncidence(input.times, input.statuses)
}));
const clocks = [{
  scale: 7,
  shape: 0.5,
  age: 0,
  interval: 2,
  multiplier: 0.5,
  mode: 'time'
}, {
  scale: 9,
  shape: 1,
  age: 8,
  interval: 5,
  multiplier: 2,
  mode: 'hazard'
}, {
  scale: 12,
  shape: 1.5,
  age: 4,
  interval: 3,
  multiplier: 2,
  mode: 'time'
}, {
  scale: 20,
  shape: 3,
  age: 0,
  interval: 6,
  multiplier: 0.25,
  mode: 'hazard'
}, {
  scale: 14,
  shape: 4,
  age: 3,
  interval: 4,
  multiplier: 4,
  mode: 'time'
}].map(input => ({
  input,
  actual: model.weibullClock(input)
}));
const ph = ['switch', 'mixture'].flatMap(mode => [0, 3.9, 4, 6, 11].map(time => ({
  input: {
    mode,
    time
  },
  actual: model.proportionalComparison({
    mode,
    time
  })
})));
const brier = [0, 0.35, 0.5, 1].flatMap(prediction => [0.05, 0.25, 0.8, 1].map(lateCensorProbability => {
  const input = {
    prediction,
    lateCensorProbability
  };
  return {
    input,
    actual: model.censorWeightedBrier(input)
  };
}));
const constants = [[0, 0, 5], [0.2, 0, 8], [0, 0.3, 4], [0.15, 0.35, 3], [0.6, 0.2, 0]].map(([firstRate, secondRate, horizon]) => {
  const input = {
    firstRate,
    secondRate,
    horizon
  };
  return {
    input,
    actual: model.constantCompeting(input)
  };
});
const observation = [1, 4, 5.5, 9].flatMap(cutoff => [null, 1, 3].map(toggled => ({
  input: {
    cutoff,
    toggled
  },
  actual: model.observedPumps({
    cutoff,
    toggled
  })
})));
observation.push({
  input: {
    cutoff: 4,
    allCensored: true
  },
  actual: model.observedPumps({
    cutoff: 4,
    allCensored: true
  })
});
const invalidCases = [['empty records', () => model.kaplanMeier([], [])], ['sparse records', () => model.kaplanMeier([, 1], [false, true])], ['non-Boolean indicators', () => model.kaplanMeier([1], [1])], ['mismatched records', () => model.kaplanMeier([1], [true, false])], ['negative time', () => model.kaplanMeier([-1], [true])], ['unsupported area horizon', () => model.restrictedMean(km[0].table, 10)], ['unsupported curve horizon', () => model.survivalAt(km[0].table, 10)], ['illegal tied-likelihood mode', () => model.coxRiskSets({
  ties: 'exact'
})], ['missing feature', () => model.coxRiskSets({
  features: [0]
})], ['zero censor support', () => model.censorWeightedBrier({
  lateCensorProbability: 0
})], ['illegal cause', () => model.competingIncidence([1], [3])], ['illegal group', () => model.logrankTable([1], [true], [2])], ['unavailable clock mode', () => model.weibullClock({
  mode: 'probability'
})]];
for (const [, operation] of invalidCases) assert.throws(operation);
const packet = {
  capturedAt: new Date().toISOString(),
  modelSource,
  modelHash: hash(modelSource),
  exampleSource,
  exampleHash: hash(exampleSource),
  reusedPrograms: {
    record: executionSource,
    recordHash: hash(executionSource),
    ...previous
  },
  examples: survivalExamples,
  km,
  cox,
  logrank,
  pairs,
  cif,
  clocks,
  ph,
  brier,
  constants,
  observation,
  invalidCases: invalidCases.map(([name]) => name)
};
const snapshot = path.join(directory, 'model-cases.json');
fs.writeFileSync(snapshot, JSON.stringify(packet, (_, value) => value === Infinity ? 'Infinity' : value, 2) + '\n');
const python = process.env.LESSON_PYTHON || 'scratch/survival-tools/Scripts/python.exe';
const result = spawnSync(python, ['scripts/verify-survival-models.py', snapshot], {
  encoding: 'utf8',
  maxBuffer: 1024 * 1024
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.status, 0, 'Independent native oracle failed.');
