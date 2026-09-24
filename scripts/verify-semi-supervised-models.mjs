import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import * as model from '../src/learn/data/semi-supervised-models.js';

const packet = 'docs/teaching/drafts/semi-supervised-learning-label-propagation-self-training-co-training';
const reference = JSON.parse(readFileSync(`${packet}/checked-results.json`));
const checks = [];
const near = (actual, expected) => {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); expected.forEach((value, i) => near(actual[i], value)); }
  else if (typeof expected === 'number') assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} ≠ ${expected}`);
  else assert.equal(actual, expected);
};
function check(name, test) { test(); checks.push(name); }
const line = model.graphDefault;
check('Harmonic chain, exact equilibrium and synchronous trace against NumPy packet', () => {
  const result = model.propagateGraph(line);
  near(result.scores.slice(1, 3), reference.graph.line.unlabeled);
  near(result.trace.slice(0, 6), reference.graph.line.trace);
  assert.ok(result.residual < 1e-12);
});
check('Shortcut reverses B; severed chain separates anchors', () => {
  near(model.propagateGraph({ ...line, edges: [...line.edges, [1, 3, 2]] }).scores.slice(1, 3), reference.graph.bridge.unlabeled);
  near(model.propagateGraph({ ...line, edges: [[0, 1, 1], [2, 3, 1]] }).scores.slice(1, 3), reference.graph.separated.unlabeled);
});
check('Soft solutions, raw/normalized quantities and intermediate states against independent NumPy', () => {
  for (const alpha of [0, 0.2, 0.8]) {
    const result = model.propagateGraph(line, 'soft', alpha);
    const fixture = reference.graph.spreading[alpha.toFixed(1)];
    near(result.equilibrium, fixture.scores);
    result.scores.forEach((value, i) => fixture.has_signal[i] ? near(value, fixture.normalized[i][1]) : assert.equal(value, null));
    near(result.trace.slice(0, Math.min(result.trace.length, fixture.trace.length)), fixture.trace.slice(0, result.trace.length));
    assert.ok(result.residual < 1e-12);
  }
});
check('Unanchored islands, degree-zero nodes and no-anchor null', () => {
  const island = { nodes: [...line.nodes, { name: 'E', label: null }, { name: 'F', label: null }], edges: [...line.edges, [4, 5, 1]] };
  for (const mode of ['hard', 'soft']) near(model.propagateGraph(island, mode).scores.slice(4), [null, null]);
  const empty = { nodes: [{ name: 'A', label: 1 }, { name: 'B', label: null }], edges: [] };
  for (const mode of ['hard', 'soft']) near(model.propagateGraph(empty, mode).scores, [1, null]);
  near(model.propagateGraph({ ...line, nodes: line.nodes.map(node => ({ ...node, label: null })) }).scores, [null, null, null, null]);
});
check('Single class, contradictory anchors, exact tie and high-alpha convergence', () => {
  near(model.propagateGraph({ ...line, nodes: line.nodes.map(node => ({ ...node, label: node.label === null ? null : 1 })) }).scores, [1, 1, 1, 1]);
  const tied = { nodes: [{ name: 'A', label: 0 }, { name: 'B', label: null }, { name: 'C', label: 1 }], edges: [[0, 1, 5], [1, 2, 5]] };
  assert.equal(model.scoreSide(model.propagateGraph(tied).scores[1]), 'tie');
  const soft = model.propagateGraph(tied, 'soft', 0.99);
  assert.ok(soft.scores[0] > 0 && soft.scores[2] < 1);
  assert.ok(soft.converged && soft.trace.length <= 5001);
  assert.equal(model.propagateGraph(tied).scores[0], 0);
});
check('Graph parser accepts arbitrary named edits and rejects malformed/duplicate/invalid rows', () => {
  const graph = model.parseGraph('left,0\ncenter,?\nright,1', 'left,center,2\ncenter,right,1');
  near(model.propagateGraph(graph).scores, [0, 1 / 3, 1]);
  for (const edges of ['A,A,1', 'A,B,-1', 'A,B,6', 'A,B,1\nB,A,2', 'A,Z,1', 'A,B,']) assert.throws(() => model.parseGraph('A,0\nB,?', edges));
  assert.throws(() => model.parseGraph('A,0\nA,1', ''));
});
const baseline = model.parsePrototype('-2,0\n2,1', '-1,0,1,3', 0.8, 1.25);
check('Prototype default and edited contrast, including final refit', () => {
  const run = model.prototypeTraining(baseline);
  near(run.history[0].after, [-1.5, 2]);
  near(run.final, [-1, 2]);
  near(run.finalBoundary, 0.5);
  assert.equal(run.afterQuery, 1);
  near(run.history[1].proposals[0].probabilities[0], 0.8519528019683106);
  const changed = model.prototypeTraining({ ...baseline, pool: [-1, 0, 1, 9] });
  near(changed.finalBoundary, 1.5);
  assert.equal(changed.afterQuery, 0);
  const capped = model.prototypeTraining(baseline, 1);
  near(capped.final, [-1.5, 2]);
  assert.equal(capped.capped, true);
});
check('Prototype null, duplicates, empty pool, equality, stable softmax and reversed classes', () => {
  for (const pool of [[0, 0], []]) {
    const result = model.prototypeTraining({ ...baseline, pool });
    near(result.final, [-2, 2]);
    assert.equal(result.movement, 'unchanged');
  }
  const tie = model.prototypeTraining({ ...baseline, pool: [0], threshold: 0.5 });
  assert.equal(tie.history[0].accepted[0].label, 0);
  const overlapping = model.prototypeTraining(model.parsePrototype('0,0\n0,1', '0', 0.8, 0));
  assert.equal(overlapping.finalBoundary, null);
  assert.equal(overlapping.queryChange, 'tie');
  const reversed = model.prototypeTraining(model.parsePrototype('2,0\n-2,1', '-1,0,1,-3', 0.8, -1.25));
  near(reversed.final, [1, -2]);
  near(model.prototypeScores(10, [-10, -9]).reduce((a, b) => a + b, 0), 1);
  assert.throws(() => model.parsePrototype('0,0\n1,0', '1', .8, 0));
  assert.throws(() => model.parsePrototype('-2,0\n2,1', 'Infinity', .8, 0));
});
check('Changed practice calculations independently derived in manuscript', () => {
  const changed = { ...line, edges: [[0, 1, 2], [1, 2, 1], [2, 3, 1]] };
  near(model.propagateGraph(changed).scores.slice(1, 3), [1 / 5, 3 / 5]);
  const exercise = model.prototypeTraining(model.parsePrototype('-3,0\n3,1', '-2,2,8', .8, 0));
  near(exercise.finalBoundary, 11 / 12);
});
check('Categorical transfer trace agrees with complete independent Python algorithm', () => {
  const result = model.categoricalCoTraining(model.coTrainingRows);
  const expected = reference.teaching.co_training.baseline;
  assert.deepEqual(result.labels, expected.labels);
  assert.deepEqual(result.rules, expected.rules);
  result.history.forEach((round, i) => {
    assert.deepEqual(round.rulesBefore, expected.history[i].rules_before);
    assert.deepEqual(round.labelsAfter, expected.history[i].labels_after);
    assert.deepEqual(round.conflicts, expected.history[i].conflicts);
    assert.deepEqual(round.offers.map(({ row, donor, recipient, label }) => ({ row, donor, recipient, label })), expected.history[i].offers);
  });
  assert.equal(model.coTrainingAnswer(result, model.coTrainingRows, 3, 0), '0');
  assert.equal(model.coTrainingAnswer(result, model.coTrainingRows, 6, 0), 'conflict');
});
check('Changed bridge, duplicate view, no anchors and broken-bridge transfer', () => {
  const changed = model.coTrainingRows.map((row, i) => i === 2 ? { ...row, views: ['blue', 'triangle'] } : row);
  assert.equal(model.categoricalCoTraining(changed).rules[0].green, 1);
  const duplicate = model.coTrainingRows.map(row => ({ ...row, views: [row.views[0], row.views[0]] }));
  assert.deepEqual(model.categoricalCoTraining(duplicate).labels[0].flatMap((label, i) => label === null ? [i] : []), [3, 4, 5]);
  const empty = model.coTrainingRows.map(row => ({ ...row, label: null }));
  assert.equal(model.categoricalCoTraining(empty).history[0].offers.length, 0);
  const broken = model.coTrainingRows.map((row, i) => i === 2 ? { ...row, views: ['violet', 'triangle'] } : row);
  const result = model.categoricalCoTraining(broken);
  assert.equal(result.rules[0].green, undefined);
  assert.equal(result.rules[0].orange, 1);
  assert.equal(result.rules[1].hexagon, 1);
});
check('Conflicting category abstains, observed labels persist and row permutation preserves batch semantics', () => {
  const rows = [{ views: ['same', 'one'], label: 0 }, { views: ['same', 'two'], label: 1 }, { views: ['same', 'three'], label: null }];
  assert.equal(model.categoricalCoTraining(rows).rules[0].same, undefined);
  const original = model.categoricalCoTraining(model.coTrainingRows);
  const order = [5, 2, 0, 6, 1, 4, 3];
  const permuted = model.categoricalCoTraining(order.map(index => model.coTrainingRows[index]));
  order.forEach((source, index) => [0, 1].forEach(view => assert.equal(permuted.labels[view][index], original.labels[view][source])));
  assert.equal(model.categoricalCoTraining(model.coTrainingRows, 1).capped, true);
  assert.throws(() => model.parseViews(' | red | 0\nblue | square | 1'));
});
const sources = ['src/learn/data/semi-supervised-models.js', `${packet}/checked-results.json`, 'scripts/verify-semi-supervised-models.mjs'];
writeFileSync('docs/teaching/evidence/semi-supervised-models.json', JSON.stringify({ status: 'passed', checks, sourceHashes: Object.fromEntries(sources.map(path => [path, createHash('sha256').update(readFileSync(path)).digest('hex')])), limits: 'Deterministic tiny teaching models; banknote library fitting verified separately. Numerical comparisons do not establish visual or independent pedagogical review.' }, null, 2) + '\n');
console.log(`Passed ${checks.length} grouped model checks.`);
