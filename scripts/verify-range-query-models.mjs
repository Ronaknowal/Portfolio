import assert from 'node:assert/strict';
import { segmentState, segmentQuery, segmentAssign, segmentGeometry, fenwickState, fenwickBlocks, fenwickPrefix, fenwickAdd, lazyState, lazyRangeOperation, lazyEffectiveValues, sparseMinimum, weightedIncreasingPlan, parseRangeValues } from '../src/learn/data/range-query-models.js';
import { movingMaximumTrace, signedShortestTrace, parseDequeValues } from '../src/learn/data/range-deque-models.js';

let seed = 918273;
function random(maximum) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed % maximum;
}
const sum = values => values.reduce((total, value) => total + value, 0);
const totals = { arrays: 0, intervals: 0, lazyOperations: 0, dequeCases: 0, weightedCases: 0 };
for (let sample = 0; sample < 320; sample++) {
  const values = Array.from({ length: random(9) }, () => random(15) - 7);
  totals.arrays++;
  for (const operation of ['sum', 'min', 'max']) {
    let state = segmentState(values, operation);
    const expected = operation === 'sum' ? sum : operation === 'min' ? items => Math.min(...items) : items => Math.max(...items);
    for (let left = 0; left <= values.length; left++) {
      for (let right = left; right <= values.length; right++) {
        const trace = segmentQuery(state, left, right);
        assert.equal(trace.result, expected(values.slice(left, right)));
        const final = trace.frames.at(-1);
        const geometry = segmentGeometry(state).nodes;
        const cover = [...final.leftNodes, ...final.rightNodes].flatMap(id => {
          const node = geometry.find(item => item.id === id);
          return Array.from({ length: node.right - node.left }, (_, offset) => node.left + offset);
        });
        assert.deepEqual(cover, Array.from({ length: right - left }, (_, offset) => left + offset));
        if (left < right && operation === 'min') assert.equal(sparseMinimum(values, left, right).result, expected(values.slice(left, right)));
        totals.intervals++;
      }
    }
    if (values.length) {
      const index = random(values.length);
      const changed = [...values];
      changed[index] = random(15) - 7;
      state = segmentAssign(state, index, changed[index]).state;
      assert.equal(segmentQuery(state, 0, values.length).result, expected(changed));
    }
  }
  let fenwick = fenwickState(values);
  for (const block of fenwickBlocks(fenwick)) assert.equal(block.value, sum(values.slice(block.left, block.right)));
  for (let end = 0; end <= values.length; end++) {
    const trace = fenwickPrefix(fenwick, end);
    assert.equal(trace.result, sum(values.slice(0, end)));
    const covered = trace.frames.at(-1).visited.flatMap(index => {
      const block = fenwickBlocks(fenwick)[index - 1];
      return Array.from({ length: block.right - block.left }, (_, offset) => block.left + offset);
    }).sort((a, b) => a - b);
    assert.deepEqual(covered, Array.from({ length: end }, (_, index) => index));
  }
  if (values.length) {
    const index = random(values.length);
    const updated = [...values];
    updated[index] += 3;
    const trace = fenwickAdd(fenwick, index, 3);
    const expectedVisited = fenwickBlocks(fenwick).filter(block => block.left <= index && index < block.right).map(block => block.internal);
    assert.deepEqual(trace.frames.at(-1).visited, expectedVisited);
    fenwick = trace.state;
    for (const block of fenwickBlocks(fenwick)) assert.equal(block.value, sum(updated.slice(block.left, block.right)));
  }
  let lazy = lazyState(values);
  const plain = [...values];
  for (let step = 0; step < 25; step++) {
    const first = random(values.length + 1);
    const second = random(values.length + 1);
    const left = Math.min(first, second);
    const right = Math.max(first, second);
    const kind = ['add', 'set', 'query'][random(3)];
    const amount = random(9) - 4;
    const before = JSON.stringify(lazy);
    const trace = lazyRangeOperation(lazy, kind, left, right, amount);
    assert.equal(JSON.stringify(lazy), before, 'input snapshots remain unchanged');
    if (kind === 'query') assert.equal(trace.result, sum(plain.slice(left, right)));
    else for (let index = left; index < right; index++) plain[index] = kind === 'set' ? amount : plain[index] + amount;
    lazy = trace.state;
    assert.deepEqual(lazyEffectiveValues(lazy), plain);
    assert.equal(lazy.tree[1], sum(plain));
    assert.equal(trace.frames.at(-1).dirty.length, 0);
    for (const frame of trace.frames) {
      assert(Object.isFrozen(frame.state.tree));
      if (frame.phase === 'push') assert.equal(frame.state.multipliers[frame.current], 1);
    }
    totals.lazyOperations++;
  }
}

function enumerateArrays(length, alphabet, visit, prefix = []) {
  if (!length) return visit(prefix);
  for (const value of alphabet) enumerateArrays(length - 1, alphabet, visit, [...prefix, value]);
}
for (let length = 0; length <= 6; length++) enumerateArrays(length, [-2, 0, 3], values => {
  for (let width = 1; width <= 8; width++) {
    const trace = movingMaximumTrace(values, width);
    const expected = [];
    for (let left = 0; left + width <= values.length; left++) {
      const value = Math.max(...values.slice(left, left + width));
      let index = left + width - 1;
      while (values[index] !== value) index--;
      expected.push({ left, right: left + width, value, index });
    }
    assert.deepEqual(trace.answers, expected);
    assert.equal(trace.pushes, values.length);
    assert(trace.pops <= trace.pushes);
    for (const frame of trace.frames.filter(item => ['append', 'answer'].includes(item.phase))) {
      assert(frame.deque.every((index, offset) => index >= frame.left && (!offset || values[frame.deque[offset - 1]] > values[index])));
    }
    totals.dequeCases++;
  }
  for (const target of [1, 3, 5, 9]) {
    let expected = null;
    for (let left = 0; left < values.length; left++) {
      for (let right = left + 1; right <= values.length; right++) {
        const total = sum(values.slice(left, right));
        if (total >= target && (!expected || right - left < expected.length)) expected = { left, right, length: right - left, sum: total };
      }
    }
    const trace = signedShortestTrace(values, target);
    assert.deepEqual(trace.best, expected);
    assert.equal(trace.pushes, values.length + 1);
    assert(trace.pops <= trace.pushes);
    for (const frame of trace.frames.filter(item => item.phase === 'append')) assert(frame.deque.every((index, offset) => !offset || trace.prefixes[frame.deque[offset - 1]] < trace.prefixes[index]));
    totals.dequeCases++;
  }
});
for (let sample = 0; sample < 800; sample++) {
  const values = Array.from({ length: random(9) }, () => random(7) - 3);
  const weights = values.map(() => random(13) - 6);
  let best = 0;
  for (let mask = 0; mask < 2 ** values.length; mask++) {
    const indices = values.map((_, index) => index).filter(index => mask & 2 ** index);
    if (indices.every((index, offset) => !offset || values[indices[offset - 1]] < values[index])) best = Math.max(best, sum(indices.map(index => weights[index])));
  }
  const plan = weightedIncreasingPlan(values, weights);
  assert.equal(plan.score, best);
  assert.equal(sum(plan.witness.map(index => weights[index])), best);
  assert(plan.witness.every((index, offset) => !offset || plan.witness[offset - 1] < index && values[plan.witness[offset - 1]] < values[index]));
  for (const frame of plan.frames) assert(frame.stored.score >= frame.previous.score);
  totals.weightedCases++;
}
assert.deepEqual(parseRangeValues(''), []);
assert.deepEqual(parseDequeValues('1, -1, 5'), [1, -1, 5]);
for (const invalid of ['1.5', 'NaN', '1,2,3,4,5,6,7,8,9']) assert.throws(() => parseRangeValues(invalid));
assert.throws(() => segmentQuery(segmentState([1]), -1, 1));
assert.throws(() => fenwickAdd(fenwickState([]), 0, 1));
assert.throws(() => sparseMinimum([1], 0, 0));
assert.throws(() => signedShortestTrace([1], 0));
console.log(JSON.stringify(totals));
