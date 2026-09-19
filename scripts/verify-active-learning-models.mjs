import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
import { activeLearningFixtures as f, entropy, uncertaintyScores, committeeDecomposition, thresholdQuestion,
  thresholdState, validateThresholdInputs, coveringDistances, farthestFirst, entropyBatch, validateGeometry,
} from '../src/learn/data/active-learning-models.js';
import { activeLearningProgram } from '../src/learn/data/active-learning-examples.js';

let comparisons = 0;
function close(actual, expected, label) {
  comparisons += 1;
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
}

// Independent expected survivor count: average the survivor count for each possible true hypothesis.
for (const hypotheses of [f.hypotheses, [0.5, 1.5, 4.5, 7.5].map(threshold => ({ threshold, weight: 1 })), f.hypotheses.map((row, index) => ({ ...row, weight: index + 1 }))]) {
  for (let query = -1; query <= 9; query += 0.5) {
    const total = hypotheses.reduce((sum, row) => sum + row.weight, 0);
    const expected = hypotheses.reduce((sum, truth) => sum + truth.weight / total * hypotheses.filter(candidate => (query >= candidate.threshold) === (query >= truth.threshold)).length, 0);
    close(thresholdQuestion(hypotheses, [], query).expectedCount, expected, 'Expected survivors by averaging possible worlds');
  }
}
let observations = [...f.observations];
for (const [query, answer, count] of [[4, 0, 4], [6, 1, 2], [5, 0, 1]]) {
  observations.push({ x: query, label: answer });
  assert.equal(thresholdState(f.hypotheses, observations).length, count);
}
assert.equal(thresholdState(f.hypotheses, [{ x: 8, label: 0 }]).length, 0);
close(thresholdQuestion(f.hypotheses, f.observations, 0).expectedCount, 8, 'Null query');
close(thresholdQuestion([{ threshold: 1, weight: 1e308 }, { threshold: 3, weight: 1e308 }], [], 2).expectedCount, 1, 'Large relative weights');
assert.throws(() => validateThresholdInputs([{ threshold: 1, weight: 0 }, { threshold: 3, weight: 1 }], [], [2]));
assert.throws(() => validateThresholdInputs(f.hypotheses, [], [1, 1]));

// Independent route for disagreement: average Kullback-Leibler divergence to the mixture.
for (let left = 0; left <= 20; left += 1) {
  for (let right = 0; right <= 20; right += 1) {
    const rows = [[left / 20, 1 - left / 20], [right / 20, 1 - right / 20]];
    const mean = rows[0].map((value, index) => (value + rows[1][index]) / 2);
    const divergence = rows.reduce((sum, row) => sum + row.reduce((part, value, index) => part + (value === 0 ? 0 : value * Math.log(value / mean[index])), 0), 0) / 2;
    close(committeeDecomposition(rows).disagreement, divergence, 'D equals mean KL');
  }
}
close(committeeDecomposition(f.shared).disagreement, 0, 'Identical-member null');
close(committeeDecomposition([[1, 0], [0, 1]]).disagreement, Math.log(2), 'Deterministic disagreement');
close(uncertaintyScores([0.45, 0.3, 0.25]).entropy, 1.0670938948757507, 'Multiclass row');
assert.throws(() => entropy([0.5, 0.4]));
assert.throws(() => committeeDecomposition([[NaN, 0], [0, 1]]));
assert.deepEqual(committeeDecomposition([[0.5, 0.5], [0.5, 0.5]]).votes, [2, 0]);

// Radius by exhaustive squared-distance matrix, independent of the nearest-center helper.
const oracleRadius = (anchors, candidates, ids) => Math.sqrt(Math.max(...candidates.map(point => Math.min(...[...anchors, ...candidates.filter(candidate => ids.includes(candidate.id))].map(center => (point.x - center.x) ** 2 + (point.y - center.y) ** 2)))));
assert.deepEqual(farthestFirst(f.anchors, f.candidates, 2).selected, ['C', 'D']);
assert.deepEqual(entropyBatch(f.candidates, 2), ['A', 'B']);
close(coveringDistances(f.anchors, f.candidates, ['A', 'B']).radius, 4, 'Redundant batch');
close(farthestFirst(f.anchors, f.candidates, 2).radius, Math.sqrt(1.01), 'Diverse batch');
const changed = f.candidates.map(point => ({ ...point, y: point.id === 'C' ? 2 : point.y }));
assert.deepEqual(farthestFirst(f.anchors, changed, 2).selected, ['D', 'C']);
const coincident = ['A', 'B', 'C'].map(id => ({ id, x: 0, y: 0, probability: 0.5 }));
assert.deepEqual(farthestFirst(f.anchors, coincident, 2).selected, ['A', 'B']);
close(farthestFirst(f.anchors, coincident, 2).radius, 0, 'Coincident null');
for (let fixture = 0; fixture < 40; fixture += 1) {
  const candidates = Array.from({ length: 5 }, (_, index) => ({ id: String(index), x: ((fixture * 3 + index * 7) % 17) - 8, y: ((fixture * 11 + index * 5) % 19) - 9, probability: 0.5 }));
  const result = farthestFirst(f.anchors, candidates, 2);
  close(result.radius, oracleRadius(f.anchors, candidates, result.selected), 'Radius independently assembled');
  const alternatives = candidates.flatMap((left, index) => candidates.slice(index + 1).map(right => oracleRadius(f.anchors, candidates, [left.id, right.id])));
  assert.ok(result.radius <= 2 * Math.min(...alternatives) + 1e-10, 'Metric radius approximation bound');
  assert.equal(new Set(result.selected).size, 2);
}
assert.throws(() => validateGeometry([], f.candidates, 2));
assert.throws(() => validateGeometry(f.anchors, f.candidates, 1.5));
assert.throws(() => validateGeometry(f.anchors, [...f.candidates, f.candidates[0]], 2));

const sourcePaths = ['src/learn/data/active-learning-models.js', 'src/learn/data/topics/active-learning.jsx', 'src/learn/components/lesson-labs/ActiveLearningFigures.jsx', 'src/learn/components/lesson-labs/ActiveLearningLabs.jsx'];
const hashes = {};
for (const path of sourcePaths) {
  const source = fs.readFileSync(path, 'utf8');
  assert.ok(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(source), `Control character in ${path}`);
  const ast = parse(source, { sourceType: 'module', plugins: ['jsx'] });
  function visit(node) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'StringLiteral') assert.ok(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(node.value), `Cooked escape in ${path}`);
    Object.values(node).forEach(value => { if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value); });
  }
  visit(ast);
  hashes[path] = createHash('sha256').update(source).digest('hex');
}
assert.equal(activeLearningProgram, fs.readFileSync('public/learn/downloads/active-learning/banknote-active-learning.py', 'utf8').replaceAll('\r\n', '\n'));
const body = fs.readFileSync('src/learn/data/topics/active-learning.jsx', 'utf8');
assert.equal((body.match(/className="active-section-anchor"/g) || []).length, 10);
assert.equal((body.match(/className="active-answer"/g) || []).length, 16);
assert.ok(!body.includes('[Inline figure:'), 'All design markers must become real visuals');
const record = { status: 'passed', checkedAt: new Date().toISOString(), comparisons, scope: ['Weighted query expectations by possible-world averaging', 'Committee entropy difference by mean KL', 'Coverage by exhaustive squared-distance matrix and small exact optimum', 'Specified contrasts, nulls, contradictions and invalid inputs', 'Complete generated prose structure and runtime-string escaping'], sourceHashes: hashes };
fs.writeFileSync('docs/teaching/evidence/active-learning-models.json', JSON.stringify(record, null, 2) + '\n');
console.log(`PASS: ${comparisons} independent numerical comparisons, geometry bounds, edge contracts, complete reader structure and escaping.`);
