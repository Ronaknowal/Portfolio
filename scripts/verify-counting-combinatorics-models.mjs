import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parse } from '@babel/parser';
import katex from 'katex';
import * as model from '../src/learn/data/counting-combinatorics-models.js';
import { countingCombinatoricsExamples } from '../src/learn/data/counting-combinatorics-examples.js';

const fixtures = { choices: [], allocations: [], overlaps: [], induction: [], catalan: [], coefficients: [], rotations: [], binomials: [] };
const body = fs.readFileSync('src/learn/data/topics/counting-combinatorics-mathematical-induction.jsx', 'utf8');
let mathTemplates = 0;
function inspect(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'TaggedTemplateExpression' && node.tag.type === 'MemberExpression' && node.tag.property.name === 'raw') {
    const formula = node.quasi.quasis.map(part => part.value.raw).join('');
    assert(![...formula].some(character => character.charCodeAt(0) < 32 && !['\n', '\r', '\t'].includes(character)));
    katex.renderToString(formula, { displayMode: true, throwOnError: true });
    mathTemplates += 1;
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(inspect);
    else if (value && typeof value === 'object') inspect(value);
  }
}
inspect(parse(body, { sourceType: 'module', plugins: ['jsx'] }));
assert.equal(mathTemplates, 14);
for (let labels = 0; labels <= 5; labels += 1) {
  for (let length = 0; length <= 4; length += 1) {
    for (const repeats of [false, true]) fixtures.choices.push({ labels, length, repeats, result: model.choiceFibers(labels, length, repeats) });
  }
}
for (let code = 0; code < 256; code += 1) {
  const capacities = [0, 1, 2, 3].map(index => Math.floor(code / 4 ** index) % 4);
  for (const total of [0, 2, 5, 8, 13]) {
    const minimums = capacities.map((_, index) => (code + index) % 3 === 0 ? 1 : 0);
    fixtures.allocations.push({ total, capacities, minimums, count: model.allocationCount(total, capacities, minimums), values: model.enumerateAllocations(total, capacities, minimums) });
  }
}
for (const total of [0, 1]) fixtures.allocations.push({ total, capacities: [], minimums: [], count: model.allocationCount(total, []), values: model.enumerateAllocations(total, []) });
for (let mask = 0; mask < 4096; mask += 1) {
  const memberships = Array.from({ length: 4 }, (_, row) => Array.from({ length: 3 }, (_, column) => Boolean(mask & (1 << (3 * row + column)))));
  fixtures.overlaps.push({ memberships, result: model.overlapContributions(memberships) });
}
for (const [name, preset] of Object.entries(model.inductionPresets)) {
  for (let mask = 0; mask < 2 ** preset.small; mask += 1) {
    const enabled = Array.from({ length: preset.small }, (_, index) => Boolean(mask & (1 << index)));
    for (let target = preset.lower; target <= 70; target += 1) fixtures.induction.push({ name, enabled, target, result: model.inductionCoverage(name, enabled, target) });
  }
}
for (let pairs = 0; pairs <= 6; pairs += 1) {
  const result = model.balancedWordCounts(pairs);
  fixtures.catalan.push({ pairs, result, paths: result.words.map(model.parenthesisPath) });
}
for (let code = 0; code < 125; code += 1) {
  const capacities = [0, 1, 2].map(index => Math.floor(code / 5 ** index) % 5);
  fixtures.coefficients.push({ capacities, rows: model.coefficientStages(capacities) });
}
for (let length = 1; length <= 8; length += 1) fixtures.rotations.push({ length, result: model.rotationOrbits(length) });
for (let n = 0; n <= 100; n += 1) for (const k of [-1, 0, 1, Math.floor(n / 2), n, n + 1]) fixtures.binomials.push({ n, k, count: model.binomialCount(n, k) });
for (const [n, k] of [[1000, 500], [10000, 50], [10000, 0]]) fixtures.binomials.push({ n, k, count: model.binomialCount(n, k) });
const invalid = [
  () => model.binomialCount(2.5, 1), () => model.binomialCount(4, NaN),
  () => model.choiceFibers(2, 2, 1), () => model.choiceFibers(9, 2),
  () => model.allocationCount(3, [2, , 2]), () => model.allocationCount(3, [2], [,]),
  () => model.allocationCount(3, [-1]), () => model.allocationCount(3, [2], []),
  () => model.enumerateAllocations(40, [40, 40, 40, 40, 40, 40]),
  () => model.allocationWord([1, , 2]), () => model.overlapContributions([[true, , false]]),
  () => model.overlapContributions([,]), () => model.overlapContributions([[true, false, 1]]),
  () => model.inductionCoverage('unknown', [], 20), () => model.inductionCoverage('fourSeven', [true, , true, true], 20),
  () => model.inductionCoverage('fourSeven', [true, true, true, true], 17),
  () => model.parenthesisPath('(()'), () => model.parenthesisPath('(a)'),
  () => model.balancedWordCounts(8), () => model.coefficientStages([1, , 2]),
  () => model.coefficientStages(Array(6).fill(20)), () => model.rotationOrbits(0),
  () => model.rotationOrbits(Infinity),
];
invalid.forEach(check => assert.throws(check));
assert.equal(model.allocationCount(10000, [10000, 10000, 10000, 10000]), 166766685001n);
const directory = 'scratch/counting-combinatorics-verification';
fs.mkdirSync(directory, { recursive: true });
fs.writeFileSync(`${directory}/model-fixtures.json`, JSON.stringify(fixtures, (_, value) => typeof value === 'bigint' ? value.toString() : value));
fs.writeFileSync(`${directory}/examples.json`, JSON.stringify(countingCombinatoricsExamples, null, 2));
const record = { checkedAt: new Date().toISOString(), passed: true, mathTemplates, invalidInputs: invalid.length, counts: Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length])), scope: 'Actual JS states exported for a separate Python oracle; invalid input contract and actual formula parsing assertions run here.' };
fs.writeFileSync(`${directory}/model-results.json`, JSON.stringify(record, null, 2));
console.log(record);
