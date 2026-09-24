import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { labelJointDecisions, multioutputMetrics, pooledLabelAssociation, sharedFeatureShrinkage, sharedOutputStump, thresholdInspection } from '../src/learn/data/multioutput-models.js';
import { multioutputExamples } from '../src/learn/data/multioutput-examples.js';
const directory = 'scratch/multioutput/native';
fs.mkdirSync(directory, { recursive: true });
const cases = { metrics: [], joint: [], association: [], thresholds: [], stumps: [], shrink: [] };
let seed = 198;
function random() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; }
for (let index = 0; index < 80; index += 1) {
  const rows = 1 + index % 5, columns = 2 + index % 3;
  const truth = Array.from({ length: rows }, () => Array.from({ length: columns }, () => random() < .15 ? null : Number(random() > .5)));
  const prediction = truth.map(row => row.map(() => Number(random() > .5)));
  cases.metrics.push({ truth, prediction, result: multioutputMetrics(truth, prediction) });
}
for (const truth of [[[0, 0]], [[null, null]], [[1, 0], [null, null]]]) cases.metrics.push({ truth, prediction: truth.map(row => row.map(() => 0)), result: multioutputMetrics(truth, truth.map(row => row.map(() => 0))) });
for (let index = 0; index < 40; index += 1) {
  const counts = index === 0 ? [4, 1, 2, 3] : Array.from({ length: 4 }, () => Math.floor(random() * 8));
  if (!counts.some(Boolean)) counts[0] = 1;
  for (const order of ['AB', 'BA']) cases.joint.push({ counts, order, result: labelJointDecisions(counts, order) });
}
for (const highShare of [0, .1, .25, .5, .9, 1]) cases.association.push(pooledLabelAssociation({ highShare }));
for (const threshold of [0, .19, .2, .21, .79, .8, .81, 1, 1.01]) {
  const input = { scores: [.8, .8, .2], targets: [1, 0, 1], threshold };
  cases.thresholds.push({ input, result: thresholdInspection(input) });
}
for (const energyScale of [.1, 1, 10, 50, 100, 1000]) for (const changed of [false, true]) {
  const input = { energyScale, ...(changed ? { temperature: [-2, 3, 1, 4], energy: [200, -100, 40, 90] } : {}) };
  cases.stumps.push({ input, result: sharedOutputStump(input) });
}
for (const first of [-3, 0, 3]) for (const second of [0, 4]) for (const penalty of [0, 1, 5, 8]) cases.shrink.push(sharedFeatureShrinkage({ first, second, penalty }));
const invalid = [() => multioutputMetrics([[1,,]], [[1,0]]), () => labelJointDecisions([0,0,0,0]), () => labelJointDecisions([1,2,3,4], 'AC'), () => thresholdInspection({ scores: [.2,.3], targets: [1] }), () => sharedOutputStump({ energyScale: 0 }), () => sharedFeatureShrinkage({ first: Infinity })];
const inherited = [, 0]; Object.setPrototypeOf(inherited, Object.assign(Object.create(Array.prototype), { 0: 1 }));
invalid.push(() => multioutputMetrics([inherited], [[1, 0]]));
invalid.forEach(run => assert.throws(run));
const execution = JSON.parse(fs.readFileSync(`${directory}/execution.json`, 'utf8'));
for (const [key, example] of Object.entries(multioutputExamples)) {
  assert.equal(crypto.createHash('sha256').update(example.code).digest('hex'), execution.examples[key].codeSha256);
  assert.equal(crypto.createHash('sha256').update(example.expected).digest('hex'), execution.examples[key].stdoutSha256);
}
fs.writeFileSync(`${directory}/model-cases.json`, JSON.stringify({ cases, invalid: invalid.length, examples: multioutputExamples }, null, 2));
console.log(JSON.stringify({ exported: Object.fromEntries(Object.entries(cases).map(([key, value]) => [key, value.length])), invalid: invalid.length, actualProgramsConserved: Object.keys(multioutputExamples).length }));
