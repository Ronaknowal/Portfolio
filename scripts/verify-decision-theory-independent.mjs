import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import * as model from '../src/learn/data/decision-theory-models.js';
import { decisionTheoryExamples } from '../src/learn/data/decision-theory-examples.js';

const directory = 'scratch/decision-theory-independent';
fs.mkdirSync(directory, { recursive: true });
const authorPath = 'docs/teaching/evidence/decision-theory-author-review.json';
const author = JSON.parse(fs.readFileSync(authorPath, 'utf8'));
const sources = author.sources || author.productionSources;
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const archive = `${directory}/author-baseline.json`;
if (!fs.existsSync(archive)) {
  fs.copyFileSync(authorPath, archive);
  for (const source of sources) {
    assert.equal(digest(source.path), source.sha256, source.path);
    const target = `${directory}/author-sources/${source.path}`;
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(source.path, target);
  }
}
for (const source of sources) assert.equal(digest(source.path), source.sha256, source.path);
const fixtures = { tails: [], allocations: [], information: [], mixtures: [], provisioning: [], rules: [], contingent: [], binary: [], fallback: [], utility: [] };
for (const values of [[-7, -1, 2, 9, 17], [8, -3, 8, 0, 40], [4, 4, 4, 4, 4]])
  for (const weights of [[0.05, 0.15, 0.2, 0.25, 0.35], [0, 0.1, 0.3, 0.6, 0]])
    for (const alpha of [0, 0.05, 0.2, 0.4, 0.73, 0.9, 0.99]) fixtures.tails.push({ values, weights, alpha, result: model.finiteTailRisk(values, weights, alpha) });
for (const probabilities of [[0.03, 0.125, 0.22, 0.45, 0.92], [0.7, 0.7, 0.05, 0, 1]])
  for (let capacity = 0; capacity <= probabilities.length; capacity++)
    for (const [handling, damage] of [[10, 80], [7, 20], [0, 0]]) fixtures.allocations.push({ probabilities, capacity, handling, damage, result: model.allocateInspections(probabilities, capacity, handling, damage) });
for (const p of [0.015, 0.075, 0.125, 0.45]) for (const sensitivity of [0.15, 0.55, 0.95]) for (const falsePositive of [0.05, 0.35]) {
  fixtures.information.push({ p, sensitivity, falsePositive, result: model.informationValue(p, sensitivity, falsePositive, 0.75), degraded: model.informationValue(p, 0.3 + 0.4 * sensitivity, 0.3 + 0.4 * falsePositive, 0.75) });
}
for (let index = 0; index <= 50; index++) fixtures.mixtures.push({ weight: index / 50, result: model.mixedStateRisk(index / 50) });
for (const quantity of [-3, -1, 0, 2, 4, 7, 9]) for (const [under, over] of [[2, 3], [9, 1]]) fixtures.provisioning.push({ quantity, under, over, values: [-1, 2, 7], weights: [0.1, 0.6, 0.3], result: model.provisioningRisk(quantity, [-1, 2, 7], [0.1, 0.6, 0.3], under, over) });
for (const p of [0.04, 0.2, 0.37, 0.8, 0.96]) fixtures.rules.push({ p, result: model.signalRuleRisks(p) });
for (const probabilities of [[0.07, 0.21, 0.42, 0.8], [0, 0.125, 0.6, 1]]) for (let capacity = 0; capacity <= 4; capacity++) for (let index = 0; index < 4; index++) fixtures.contingent.push({ probabilities, capacity, index, result: model.contingentInspection(index, 0.8, capacity, probabilities) });
for (const p of [0, 0.075, 0.2, 0.7, 1]) {
  fixtures.binary.push({ p, losses: [[-2, 18], [6, 6]], result: model.binaryDecision(p, [[-2, 18], [6, 6]]) });
  fixtures.fallback.push({ p, result: model.abstentionDecision(p, 0.6) });
}
for (const p of [0.1, 0.3, 0.7, 0.9]) fixtures.utility.push({ p, result: model.utilityLottery(16, 121, p, 64) });
assert.deepEqual(model.abstentionDecision(0.075, 0.6).actions, [0, 2]);
assert.deepEqual(model.abstentionDecision(0.7, 0.6).actions, [1, 2]);
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify({ fixtures, sources, examples: decisionTheoryExamples }, null, 2));
console.log(JSON.stringify(Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length]))));
