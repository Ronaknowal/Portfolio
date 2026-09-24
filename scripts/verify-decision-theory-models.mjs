import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import katex from 'katex';
import * as model from '../src/learn/data/decision-theory-models.js';
import { decisionTheoryExamples } from '../src/learn/data/decision-theory-examples.js';

const fixtures = { binary: [], signals: [], provisioning: [], allocations: [], information: [], tails: [], utility: [], contingent: [], thresholds: [] };
const directory = 'scratch/decision-theory-review';
fs.mkdirSync(directory, { recursive: true });
let equations = 0;
function inspect(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'TaggedTemplateExpression' && node.tag.type === 'MemberExpression' && node.tag.property.name === 'raw') {
    const formula = node.quasi.quasis.map(part => part.value.raw).join('');
    katex.renderToString(formula, { displayMode: true, throwOnError: true });
    equations += 1;
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(inspect);
    else if (value && typeof value === 'object') inspect(value);
  }
}
for (const file of ['src/learn/data/topics/decision-theory-risk-cost-sensitive-decisions.jsx', 'src/learn/components/lesson-labs/DecisionTheoryLabs.jsx', 'src/learn/components/lesson-labs/DecisionTheoryFigures.jsx']) inspect(parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] }));
for (const p of [0, 0.01, 0.08, 0.125, 0.2, 0.5, 0.9, 1]) for (const losses of [
  [[0, 80], [10, 10]], [[0, 60], [12, 12]], [[4, 4], [4, 4]], [[0, 0], [3, 8]], [[-3, 5], [1, -2]], [[1, 4], [3, 6]],
]) fixtures.binary.push({ p, losses, result: model.binaryDecision(p, losses) });
for (let index = 0; index <= 20; index += 1) fixtures.signals.push({ prior: index / 20, result: model.signalRuleRisks(index / 20) });
for (const under of [0.5, 1, 3, 4, 6]) for (const quantity of [-1, 0, 1, 2, 2.5, 3, 4, 7, 9, 10, 12]) {
  fixtures.provisioning.push({ quantity, under, values: [1, 3, 9], masses: [0.25, 0.5, 0.25], result: model.provisioningRisk(quantity, [1, 3, 9], [0.25, 0.5, 0.25], under, 1) });
}
for (const probabilities of [model.inspectionProbabilities, [0, 0.125, 0.125, 1], [0.2, 0.2, 0.2], [0, 0, 0], [1, 1, 1]]) for (let capacity = 0; capacity <= probabilities.length; capacity += 1) fixtures.allocations.push({ probabilities, capacity, result: model.allocateInspections(probabilities, capacity) });
for (const p of [0, 0.01, 0.02, 0.08, 0.125, 0.3, 1]) for (const sensitivity of [0, 0.2, 0.8, 1]) for (const falsePositive of [0, 0.1, 0.8, 1]) {
  const result = model.informationValue(p, sensitivity, falsePositive, 2);
  assert(result.value >= -2e-14 && result.value <= result.perfectValue + 2e-14);
  fixtures.information.push({ p, sensitivity, falsePositive, result });
}
for (const values of [[0, 10, 100], [-10, 0, 20], [4, 4, 4], [0, 0, 10], [100, 0, 10]]) for (const weights of [[0.8, 0.15, 0.05], [0.25, 0.5, 0.25], [0, 0, 1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]) for (const alpha of [0, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999999]) fixtures.tails.push({ values, weights, alpha, result: model.finiteTailRisk(values, weights, alpha) });
for (const low of [0, 1, 50]) for (const high of [50, 150]) for (const p of [0, 0.2, 0.5, 1]) fixtures.utility.push({ low, high, p, result: model.utilityLottery(low, high, p, 95) });
for (let capacity = 0; capacity <= 6; capacity += 1) for (let index = 0; index < 6; index += 1) for (const price of [0, 1, 12]) fixtures.contingent.push({ capacity, index, price, result: model.contingentInspection(index, price, capacity) });
const scores = [0.05, 0.1, 0.2, 0.2, 0.4, 0.8], labels = [0, 1, 0, 1, 0, 1];
for (const threshold of [0, 0.05, 0.1, 0.2, 0.4, 0.8, 2]) fixtures.thresholds.push({ scores, labels, threshold, result: model.thresholdLoss(scores, labels, threshold) });
let invalid = 0;
for (const operation of [
  () => model.evaluateFiniteActions([0.5, , 0.5], [[0, 1, 2]]),
  () => model.evaluateFiniteActions([0.5, 0.5], [[0, ,]]),
  () => model.evaluateFiniteActions([0.2, 0.2], [[0, 1]]),
  () => model.evaluateFiniteActions([1], [[Infinity]]),
  () => model.binaryDecision(-0.1), () => model.binaryDecision(NaN),
  () => model.signalRuleRisks(2), () => model.provisioningRisk(1, [1], [1], 0, 1),
  () => model.allocateInspections([0.2, , 0.8], 1), () => model.allocateInspections([0.2], 2),
  () => model.informationValue(0.2, -1), () => model.informationValue(0.2, 1, 0, -1),
  () => model.mixedStateRisk(Infinity), () => model.utilityLottery(-1),
  () => model.finiteTailRisk([0, 1], [0.5, 0.5], 1), () => model.finiteTailRisk([0, 1], [1.1, -0.1], 0.9),
  () => model.finiteTailRisk([0, , 1], [0.5, 0, 0.5], 0.9),
  () => model.contingentInspection(6), () => model.thresholdLoss([0.5], [2], 0.5),
  () => model.signalRuleRisks(Number.MIN_VALUE),
  () => model.utilityLottery(0, Number.MIN_VALUE, 0.5, 0),
  () => {
    const inherited = [0.5, ,];
    Object.setPrototypeOf(inherited, Object.assign(Object.create(Array.prototype), { 1: 0.5 }));
    model.evaluateFiniteActions(inherited, [[0, 1]]);
  },
]) { assert.throws(operation); invalid += 1; }
assert.equal(model.formatDecisionNumber(1e-12), '1.00e-12');
assert.equal(model.formatDecisionNumber(0), '0');
assert.equal(model.formatDecisionNumber(null), 'undefined');
assert.equal(model.contingentInspection(3).total, model.contingentInspection(4).total);
assert.equal(model.contingentInspection(3, 10.4).total, model.allocateInspections().risk);
assert.equal(model.contingentInspection(4, 1, 1).total, model.contingentInspection(5, 1, 1).total);
assert.deepEqual(model.signalRuleRisks(0.2).actions, [0, 2]);
assert.deepEqual(model.signalRuleRisks(0.8).actions, [1, 2]);
assert.deepEqual(model.abstentionDecision(0.7).actions, [1, 2]);
assert.equal(model.finiteTailRisk([0, 10, 100], [0.1, 0.7, 0.2], 0.8).valueAtRisk, 10);
fs.writeFileSync(`${directory}/model-fixtures.json`, JSON.stringify(fixtures, null, 2));
fs.writeFileSync(`${directory}/actual-examples.json`, JSON.stringify(decisionTheoryExamples, null, 2));
const sources = ['src/learn/data/topics/decision-theory-risk-cost-sensitive-decisions.jsx', 'src/learn/data/decision-theory-models.js', 'src/learn/data/decision-theory-examples.js', 'src/learn/components/lesson-labs/DecisionTheoryLabs.jsx', 'src/learn/components/lesson-labs/DecisionTheoryFigures.jsx', 'src/learn/components/lesson-labs/decision-theory-labs.css', 'src/learn/data/curriculum/blueprints/decision-theory-risk-cost-sensitive-decisions.js'].map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
const result = { timestamp: new Date().toISOString(), status: 'passed', cases: Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length])), invalid, equations, programs: Object.keys(decisionTheoryExamples).length, sources };
fs.writeFileSync(`${directory}/model-results.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result));
