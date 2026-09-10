import fs from 'node:fs';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { parse } from '@babel/parser';
import { topicCatalogue } from '../src/learn/data/curriculum/topic-catalogue.js';
import { TRANSPORT_COSTS, twoLocationTransport, orderedTransport, cumulativeTransport, sinkhornScaling, equalWeightEntropicPlan, sinkhornComparison, stabilityComparison } from '../src/learn/data/optimal-transport-models.js';
import { optimalTransportExamples } from '../src/learn/data/optimal-transport-examples.js';
const directory = 'scratch/optimal-transport-review';
const fixtures = { twoLocation: [], ordered: [], scaling: [], equalWeight: [], comparisons: [], stability: [], cumulative: [] };
for (const costKind of Object.keys(TRANSPORT_COSTS)) for (let a = 0; a <= 10; a += 1) for (let b = 0; b <= 10; b += 1) {
  for (const fraction of [0, 0.37, 1]) {
    const state = twoLocationTransport({ sourceFirst: a / 10, targetFirst: b / 10, fraction, costKind });
    assert.ok(state.residual < 1e-12);
    assert.ok(Object.isFrozen(state.plan[0]));
    fixtures.twoLocation.push(state);
  }
}
let seed = 904;
const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
const weighted = length => {
  const weights = Array.from({ length }, () => Math.floor(random() * 5));
  if (!weights.some(Boolean)) weights[0] = 1;
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map(value => value / total);
};
for (let trial = 0; trial < 60; trial += 1) {
  const source = Array.from({ length: 1 + trial % 6 }, () => Math.floor(random() * 10) - 3);
  const target = Array.from({ length: 1 + trial % 5 }, () => Math.floor(random() * 10) - 3);
  const a = weighted(source.length), b = weighted(target.length);
  for (const order of [1, 2, 3]) fixtures.ordered.push({ source, target, a, b, ...orderedTransport(source, a, target, b, order) });
}
for (const epsilon of [.5, 1, 2]) for (const a of [.2, .5, .8]) for (const b of [.3, .5, .7]) {
  fixtures.scaling.push(sinkhornScaling({ source: [a, 1 - a], target: [b, 1 - b], epsilon, traceSteps: 12 }));
}
// Exercise the actual unequal-weight correction lab, including its sharp setting.
for (const epsilon of [.1, .5, 2]) fixtures.scaling.push(sinkhornScaling({ source: [.6, .4], target: [.3, .7], epsilon, traceSteps: 40 }));
for (const source of [[1, 0], [0, 1]]) for (const target of [[1, 0], [.2, .8], [0, 1]]) fixtures.scaling.push(sinkhornScaling({ source, target }));
for (const costs of Object.values(TRANSPORT_COSTS)) for (const epsilon of [.1, .3, .5, 1, 2, 4]) fixtures.equalWeight.push({ epsilon, costs, ...equalWeightEntropicPlan(costs, epsilon) });
for (const epsilon of [.1, .5, 1, 2, 4]) for (const [shift, scale] of [[0, .5], [1, 1], [0, 1]]) fixtures.comparisons.push({ epsilon, ...sinkhornComparison(epsilon, shift, scale) });
for (let offset = 0; offset <= 1000; offset += 100) fixtures.stability.push(stabilityComparison(.5, offset));
for (const scenario of ['nearby', 'far', 'identical']) for (const spacing of [.5, 1, 2, 3]) fixtures.cumulative.push(cumulativeTransport(scenario, spacing));
const invalid = [
  () => twoLocationTransport({ sourceFirst: -0.1 }), () => twoLocationTransport({ fraction: 2 }), () => twoLocationTransport({ costKind: 'invented' }),
  () => sinkhornScaling({ epsilon: 0 }), () => sinkhornScaling({ iterations: Infinity }), () => sinkhornScaling({ source: [.4, .4] }),
  () => sinkhornScaling({ costs: [[0, Infinity], [0, 1]] }), () => sinkhornScaling({ costs: [[1], [2]] }),
  () => sinkhornScaling({ target: [NaN, 1] }), () => sinkhornScaling({ traceSteps: 1000 }),
  () => orderedTransport([0], [1], [1], [1], 0.5), () => cumulativeTransport('missing'), () => stabilityComparison(.5, 1001),
];
invalid.forEach(check => assert.throws(check, RangeError));
const rareMass = orderedTransport([0, 100], [1e-15, 1 - 1e-15], [-100, 100], [5e-16, 1 - 5e-16], 4);
assert.ok(Math.abs(rareMass.cost / 1e-7 - 1) < 1e-12, 'A tiny positive remainder must retain its displacement cost');
assert.equal(rareMass.plan[0][0], 5e-16);
assert.equal(rareMass.plan[0][1], 5e-16);
const durations = [];
for (let trial = 0; trial < 15; trial += 1) { const start = performance.now(); sinkhornComparison(.1, 1, 1); durations.push(performance.now() - start); }
fs.mkdirSync(directory, { recursive: true });
fs.writeFileSync(directory + '/model-fixtures.json', JSON.stringify(fixtures));
fs.writeFileSync(directory + '/examples.json', JSON.stringify(optimalTransportExamples));
// Source structure and link identity checks are separate from numerical truth.
const draft = fs.readFileSync('src/learn/data/topics/optimal-transport-wasserstein-distance-sinkhorn.jsx', 'utf8');
const ast = parse(draft, { sourceType: 'module', plugins: ['jsx'] });
const localLinks = [], proseBlocks = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXAttribute' && node.name.name === 'href' && node.value?.value?.startsWith('/learn/topic/')) localLinks.push(node.value.value.split('/').at(-1));
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'Prose') {
    proseBlocks.push(node);
    for (const child of node.children) if (child.type === 'JSXElement') assert.ok(!['MathBlock', 'Prose', 'LessonTable', 'details', 'div'].includes(child.openingElement.name.name), 'Invalid paragraph nesting');
  }
  for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
}
visit(ast);
for (const id of localLinks) assert.ok(topicCatalogue[id], 'Unknown local topic link: ' + id);
execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-optimal-transport-native.py'], { stdio: 'inherit' });
const result = { at: new Date().toISOString(), modelCases: Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length])), invalidContracts: invalid.length, proseBlocks: proseBlocks.length, localLinks, nodeOnlyWorstCaseMilliseconds: { minimum: Math.min(...durations), maximum: Math.max(...durations), median: durations.sort((a, b) => a - b)[7], note: '15 warm local Node measurements of the bounded5000-sweep2x2 case; not browser or universal timing.' } };
fs.writeFileSync(directory + '/model-results.json', JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
