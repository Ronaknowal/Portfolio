import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { groupedDefaults, groupedRead, cacheBudget, conversionRead, groupedForecast, groupedRollout } from '../src/learn/data/grouped-query-models.js';

const id = 'grouped-query-attention-gqa-multi-query-attention-mqa';
const root = `docs/teaching/deep-learning-completion/${id}`;
const fixtures = JSON.parse(fs.readFileSync(`${root}/native-fixtures.json`, 'utf8'));
let assertions = 0, maximumForecastError = 0;
function near(actual, expected, tolerance = 1e-10) {
  if (Array.isArray(actual)) { assert.equal(actual.length, expected.length); actual.forEach((value, i) => near(value, expected[i], tolerance)); }
  else { assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`); assertions++; }
}
const fixture = groupedDefaults();
near(groupedRead(fixture).map(r => r.output), fixtures.mechanisms.manual_outputs.baseline[0].map(v => v[0]));
const valueEdit = structuredClone(fixture); valueEdit.values[0][0] = [3, -1];
near(groupedRead(valueEdit).map(r => r.output), fixtures.mechanisms.manual_outputs.shared_value_edit[0].map(v => v[0]));
near(groupedRead(valueEdit).map(r => r.weights), groupedRead(fixture).map(r => r.weights));
const keyEdit = structuredClone(fixture); keyEdit.keys[0][0] = [2, 0];
near(groupedRead(keyEdit)[1].output, groupedRead(fixture)[1].output);
const swap = { ...fixture, keys: [...fixture.keys].reverse(), values: [...fixture.values].reverse(), mapping: fixture.mapping.map(g => 1 - g) };
near(groupedRead(swap).map(r => r.output), groupedRead(fixture).map(r => r.output));
assert.throws(() => groupedRead({ ...fixture, positions: [3, 4, 5] }), /No legal key/); assertions++;
const counts = cacheBudget({ batch: 2, layers: 12, length: 1024, queryHeads: 12, kvHeads: 3, keyWidth: 64, valueWidth: 32, bytes: 2 });
near(counts.total, 14155776); near(counts.total / 2 ** 20, 13.5);
assert.throws(() => cacheBudget({ queryHeads: 5, kvHeads: 2 }), /divisible/); assertions++;
near(conversionRead([1, 2], [[2, 0], [0, 2]], [[1, 3], [5, -1]]).map(r => r.converted), [2, 2]);
const tied = conversionRead([1, 2], [[2, 0], [2, 0]], [[1, 3], [1, 3]]);
tied.forEach(row => near(row.original, row.converted));
const extended = structuredClone(fixture);
extended.positions.push(3);
extended.keys.forEach(group => group.push([4, 4]));
extended.values.forEach(group => group.push([4, -4]));
near(groupedRead(extended).map(r => r.output), groupedRead(fixture).map(r => r.output));
const reordered = { ...extended, positions: [...extended.positions].reverse(), keys: extended.keys.map(r => [...r].reverse()), values: extended.values.map(r => [...r].reverse()) };
near(groupedRead(reordered).map(r => r.output), groupedRead(extended).map(r => r.output));
assert.ok(Math.abs(groupedRead({ ...extended, queryPosition: 3 })[0].output[0] - groupedRead(extended)[0].output[0]) > .1); assertions++;
const candidate = conversionRead([1, 2], [[2, 0], [0, 2]], [[1, 3], [5, -1]], { keys: [2, 0], values: [2, 4] });
near(candidate[0].minimumKeyDistance, 4); near(candidate[0].minimumValueDistance, 16);
near(candidate[0].keyDistance, 8); near(candidate[0].valueDistance, 36);
candidate.forEach(r => near(r.outputDelta, r.converted - r.original));
// The two-head least-squares identity gives an independent expected excess.
near(candidate[0].keyDistance - candidate[0].minimumKeyDistance, 2 * (1 ** 2 + (-1) ** 2));
near(candidate[0].valueDistance - candidate[0].minimumValueDistance, 2 * ((-1) ** 2 + 3 ** 2));
for (const test of fixtures.cases) {
  const { weights } = JSON.parse(fs.readFileSync(`public/learn-assets/${id}/runtime-${test.heads}.json`, 'utf8'));
  const result = groupedRollout(weights, test.points, test.heads, 5, test.shift);
  near(result.predictions, test.prediction, 2e-5);
  near(result.rollout, test.rollout, 2e-5);
  near(result.attention.at(-1), test.weights, 2e-5);
  near(result.heads.at(-1), test.headOutputs, 2e-5);
  maximumForecastError = Math.max(maximumForecastError, ...result.predictions.flatMap((p, i) => p.map((v, d) => Math.abs(v - test.prediction[i][d]))));
  let cache = null; const incremental = [];
  for (const point of test.points) { const step = groupedForecast(weights, [point], test.heads, test.shift, cache); cache = step.cache; incremental.push(step.predictions[0]); }
  near(incremental, result.predictions, 1e-12);
  near(groupedForecast(weights, test.points, test.heads, test.shift + 100).predictions, result.predictions, 1e-12);
}
for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/GroupedQueryLabs.jsx', 'src/learn/components/lesson-labs/GroupedQueryDiagrams.jsx']) {
  parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] }); assertions++;
}
const receipt = { passed: true, assertions, maximumForecastError, checks: ['NumPy manual fixture parity', 'key/value nulls and exact consistent relabeling', 'empty legal set rejects', 'cache bytes and divisibility', 'conversion nonlinearity and tied null', '12 native model/attention/head/rollout comparisons', 'incremental and global-shift equivalence', 'owned JSX parses'] };
fs.writeFileSync(`${root}/model-checks.json`, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
