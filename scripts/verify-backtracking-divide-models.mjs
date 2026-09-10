import assert from 'node:assert/strict';
import { parseBoundedIntegers, queenConflicts, subsetTree, summarizeSubarray, traceQueens, traceSubsetChoices } from '../src/learn/data/backtracking-divide-models.js';
function* tuples(alphabet, length, prefix = []) {
  if (!length) {
    yield prefix;
    return;
  }
  for (const value of alphabet) yield* tuples(alphabet, length - 1, [...prefix, value]);
}
let subsetCases = 0;
for (let length = 0; length <= 3; length++) {
  for (const values of tuples([1, 2, 3], length)) {
    for (let target = 0; target <= 10; target++) {
      const expected = [...tuples([0, 1], length)].filter(bits => bits.reduce((sum, bit, index) => sum + bit * values[index], 0) === target).map(bits => bits.flatMap((bit, index) => bit ? [index] : [])).map(JSON.stringify).sort();
      for (const prune of [false, true]) {
        const trace = traceSubsetChoices(values, target, prune);
        assert.deepEqual(trace.answers.map(JSON.stringify).sort(), expected);
        assert.deepEqual(trace.frames.at(-1).path, []);
        assert.deepEqual(trace.frames.at(-1).calls, []);
        assert.ok(Object.isFrozen(trace.frames[0].answers));
        for (const frame of trace.frames) {
          assert.equal(new Set(frame.path).size, frame.path.length);
          assert.ok(frame.path.every((position, index) => index === 0 || position > frame.path[index - 1]));
          for (const answer of frame.answers) assert.equal(answer.reduce((sum, position) => sum + values[position], 0), target);
          if (frame.phase === 'prune') {
            const call = frame.calls.at(-1);
            for (const bits of tuples([0, 1], length - call.index)) {
              const completion = bits.reduce((sum, bit, index) => sum + bit * values[call.index + index], call.sum);
              assert.notEqual(completion, target, 'every discarded completion must fail');
            }
          }
        }
        subsetCases++;
      }
      const tree = subsetTree(values);
      assert.equal(tree.nodes.length, 2 ** (length + 1) - 1);
      for (const node of tree.nodes) assert.ok(node.x >= 18 && node.x <= tree.width - 18);
    }
  }
}
assert.equal(traceSubsetChoices().entered, 13);
assert.equal(traceSubsetChoices([2, 4, 5], 5, false).entered, 15);
let queenFrames = 0;
for (let n = 1; n <= 5; n++) {
  const expected = [...tuples(Array.from({
    length: n
  }, (_, index) => index), n)].filter(columns => new Set(columns).size === n && columns.every((column, row) => columns.every((other, otherRow) => row === otherRow || Math.abs(column - other) !== Math.abs(row - otherRow)))).map(JSON.stringify).sort();
  const trace = traceQueens(n);
  assert.deepEqual(trace.answers.map(JSON.stringify).sort(), expected);
  assert.deepEqual(trace.frames.at(-1).queens, []);
  for (const frame of trace.frames) {
    frame.queens.forEach((column, row) => assert.equal(queenConflicts(frame.queens.slice(0, row), row, column).length, 0));
    if (frame.phase === 'reject') assert.ok(frame.conflicts.length > 0);
    if (frame.phase === 'solution') assert.equal(frame.queens.length, n);
    queenFrames++;
  }
}
function bestInterval(values, low, high, kind) {
  let best = null;
  for (let start = low; start < high; start++) {
    for (let end = start + 1; end <= high; end++) {
      if (kind === 'prefix' && start !== low || kind === 'suffix' && end !== high) continue;
      const sum = values.slice(start, end).reduce((a, b) => a + b, 0);
      if (!best || sum > best.sum) best = {
        sum,
        low: start,
        high: end
      };
    }
  }
  return best;
}
let summaryCases = 0;
for (let length = 1; length <= 6; length++) {
  for (const values of tuples([-2, 0, 3], length)) {
    const result = summarizeSubarray(values);
    assert.equal(result.nodes.length, 2 * length - 1);
    assert.equal(result.combines, length - 1);
    for (const node of result.nodes) {
      assert.equal(node.total, values.slice(node.low, node.high).reduce((a, b) => a + b, 0));
      for (const kind of ['prefix', 'suffix', 'best']) assert.deepEqual(node[kind], bestInterval(values, node.low, node.high, kind));
      if (node.crossing) assert.equal(node.crossing.sum, values.slice(node.crossing.low, node.crossing.high).reduce((a, b) => a + b, 0));
    }
    summaryCases++;
  }
}
for (const bad of ['', '1,', '1.5', 'NaN', '1,2,3,4,5,6,7,8,9']) assert.throws(() => parseBoundedIntegers(bad));
for (const bad of [[], [Infinity], [10]]) assert.throws(() => summarizeSubarray(bad));
assert.throws(() => traceSubsetChoices([0], 0));
assert.throws(() => traceSubsetChoices([1], -1));
assert.throws(() => traceQueens(6));
console.log(`Backtracking/divide models passed: ${subsetCases} subset cases with discarded-completion proofs, ${queenFrames} queen frames, ${summaryCases} arrays with every subtree checked against exhaustive intervals.`);
