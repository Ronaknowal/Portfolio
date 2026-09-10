import assert from 'node:assert/strict';
import { HEAP_SAMPLE, HEAP_RAW_SAMPLE, HEAP_LIMIT, TRIE_SAMPLE, heapOperationTrace, heapViolations, heapLayout, heapConstructionProfile, topKStreamTrace, parseHeapValues, parseHeapKey, buildTrie, trieWords, validateTrie, trieOperationTrace, trieLayout, parseTrieWords, parseTrieQuery } from '../src/learn/data/heap-trie-models.js';
const last = trace => trace.at(-1);
const sorted = values => [...values].sort((a, b) => a - b);
const get = (trie, id) => trie.nodes.find(node => node.id === id);
let heapCases = 0,
  streamCases = 0,
  trieCases = 0;
function* arrays(alphabet, length) {
  if (length === 0) {
    yield [];
    return;
  }
  for (const prefix of arrays(alphabet, length - 1)) for (const value of alphabet) yield [...prefix, value];
}
function validHeap(values) {
  for (let child = 1; child < values.length; child++) assert(values[Math.floor((child - 1) / 2)] <= values[child]);
  assert.deepEqual(heapViolations(values), []);
}
function validTrie(trie) {
  const validation = validateTrie(trie);
  assert.equal(validation.valid, true, validation.errors.join('; '));
}
function verifyHeapTrace(values, operation, key = 0) {
  const baseline = [...values],
    trace = heapOperationTrace(values, operation, key),
    end = last(trace);
  assert.deepEqual(values, baseline, 'Input mutation');
  const expected = sorted(operation === 'push' ? [...values, key] : operation === 'pop' ? sorted(values).slice(1) : values);
  assert.deepEqual(sorted(end.heap), expected);
  validHeap(end.heap);
  assert.equal(end.result, operation === 'pop' && !values.length ? 'empty' : 'complete');
  if (operation === 'pop') assert.equal(end.popped, values.length ? Math.min(...values) : null);
  let comparisons = 0,
    swaps = 0;
  for (const frame of trace) {
    assert(frame.comparisonCount >= comparisons && frame.comparisonCount <= comparisons + 1);
    assert(frame.swapCount >= swaps && frame.swapCount <= swaps + 1);
    if (frame.swappedIndices.length) {
      assert.equal(frame.swappedIndices.length, 2);
      assert.equal(frame.swapCount, swaps + 1);
    }
    for (const index of [...frame.activeIndices, ...frame.comparedIndices, ...frame.swappedIndices]) assert(index >= 0 && index < frame.heap.length);
    if (operation === 'build') assert.deepEqual(sorted(frame.heap), sorted(values));else if (operation === 'push') {
      assert([values.length, values.length + 1].includes(frame.heap.length));
      assert.deepEqual(sorted(frame.heap), sorted(frame.heap.length === values.length ? values : [...values, key]));
    } else {
      assert([values.length, Math.max(0, values.length - 1)].includes(frame.heap.length));
      assert.deepEqual(sorted(frame.heap), frame.heap.length === values.length ? sorted(values) : sorted(values).slice(1));
    }
    comparisons = frame.comparisonCount;
    swaps = frame.swapCount;
  }
  const layout = heapLayout(end.heap);
  assert.equal(layout.nodes.length, end.heap.length);
  assert.equal(layout.edges.length, Math.max(0, end.heap.length - 1));
  for (const node of layout.nodes) {
    assert.equal(node.depth, Math.floor(Math.log2(node.index + 1)));
    assert(node.x >= 23 && node.x <= layout.width - 23);
    assert(node.y >= 23 && node.y + 43 <= layout.height);
  }
  for (const edge of layout.edges) {
    assert.equal(edge.parent, Math.floor((edge.child - 1) / 2));
    const parent = layout.nodes[edge.parent],
      child = layout.nodes[edge.child];
    assert.equal(child.depth, parent.depth + 1);
    assert(edge.child === 2 * edge.parent + 1 ? child.x < parent.x : child.x > parent.x);
  }
  heapCases++;
  return trace;
}
for (let length = 0; length <= 5; length++) for (const values of arrays([-1, 0, 1], length)) {
  const heap = last(verifyHeapTrace(values, 'build')).heap;
  verifyHeapTrace(heap, 'pop');
  for (const key of [-2, 0, 2]) verifyHeapTrace(heap, 'push', key);
  let current = [...heap],
    drained = [];
  while (current.length) {
    const end = last(heapOperationTrace(current, 'pop'));
    drained.push(end.popped);
    current = end.heap;
  }
  assert.deepEqual(drained, sorted(values));
}
assert.deepEqual(last(heapOperationTrace(HEAP_RAW_SAMPLE, 'build')).heap, HEAP_SAMPLE);
assert.deepEqual(last(heapOperationTrace(HEAP_SAMPLE, 'push', 0)).heap, [0, 2, 1, 7, 5, 9]);
assert.deepEqual(last(heapOperationTrace(HEAP_SAMPLE, 'pop')).heap, [2, 5, 9, 7]);
const equalChoice = heapOperationTrace([3, 1, 1], 'build').find(frame => frame.phase === 'Choose the smaller child');
assert.deepEqual(equalChoice.activeIndices, [0, 1], 'Equal child priorities must select left');
assert.equal(last(heapOperationTrace([2, 1], 'pop')).result, 'invalid');
assert.deepEqual(last(heapOperationTrace([2, 1], 'push', 0)).heap, [2, 1]);
const maximum = Array.from({
  length: HEAP_LIMIT
}, (_, index) => index);
assert.equal(last(heapOperationTrace(maximum, 'push', -1)).result, 'limit');
assert.deepEqual(last(heapOperationTrace(maximum, 'push', -1)).heap, maximum);
assert.equal(last(heapOperationTrace([...maximum, 13], 'build')).result, 'limit');
assert.throws(() => heapOperationTrace([NaN], 'build'), TypeError);
assert.throws(() => heapOperationTrace([], 'push', Infinity), TypeError);
assert.throws(() => heapOperationTrace([], 'sort'), TypeError);
const profile = heapConstructionProfile(15);
assert.deepEqual(profile.rows.map(row => [row.height, row.count, row.downwardBudget]), [[0, 8, 0], [1, 4, 4], [2, 2, 4], [3, 1, 3]]);
assert.equal(profile.totalDownwardBudget, 11);
for (const size of [0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 31, 100, 1023]) {
  const result = heapConstructionProfile(size);
  const height = index => index >= size ? -1 : 1 + Math.max(height(2 * index + 1), height(2 * index + 2));
  assert.deepEqual(result.heights, Array.from({
    length: size
  }, (_, index) => height(index)));
  assert.equal(result.rows.reduce((sum, row) => sum + row.count, 0), size);
  if (size) assert(result.totalDownwardBudget < size);
}
function verifyStream(values, k) {
  const trace = topKStreamTrace(values, k),
    baseline = [...values];
  for (const frame of trace) {
    const prefix = values.slice(0, frame.processed),
      expected = sorted(prefix).slice(-k);
    assert.deepEqual(sorted(frame.heap.map(entry => entry.value)), expected);
    validHeap(frame.heap.map(entry => entry.value));
    assert.equal(frame.heap.length, Math.min(k, frame.processed));
    assert.equal(frame.kth, frame.processed >= k ? expected[0] : null);
    const sources = [...frame.heap, ...frame.discarded];
    assert.deepEqual(sources.map(entry => entry.sourceIndex).sort((a, b) => a - b), Array.from({
      length: frame.processed
    }, (_, index) => index));
    for (const entry of sources) assert.equal(entry.value, values[entry.sourceIndex]);
    if (frame.current && !frame.prefixComplete) assert.equal(frame.current.sourceIndex, frame.processed);
  }
  assert.equal(last(trace).result, 'complete');
  assert.deepEqual(values, baseline);
  streamCases++;
}
for (let length = 0; length <= 5; length++) for (const values of arrays([-1, 0, 1], length)) for (let k = 1; k <= 6; k++) verifyStream(values, k);
verifyStream([5, 1, 9, 3, 9, 2], 3);
verifyStream([99, -99, 0, 99, -99], 8);
verifyStream(Array.from({
  length: 16
}, (_, i) => i), 1);
const defaultTopK = last(topKStreamTrace());
assert.deepEqual(sorted(defaultTopK.heap.map(item => item.value)), [5, 9, 9]);
assert.equal(defaultTopK.kth, 5);
for (const k of [0, -1, 1.5, 9, NaN]) assert.throws(() => topKStreamTrace([1], k), TypeError);
assert.throws(() => topKStreamTrace(Array(17).fill(1), 3), TypeError);
const possibleWords = ['', 'a', 'ab', 'ac', 'b', 'ba'];
function verifyTrie(words, query, operation) {
  const source = buildTrie(words),
    baseline = structuredClone(source),
    trace = trieOperationTrace(source, query, operation),
    end = last(trace),
    set = new Set(words);
  let expectedResult;
  if (operation === 'insert') {
    expectedResult = set.has(query) ? 'duplicate' : 'inserted';
    set.add(query);
  } else if (operation === 'delete') {
    expectedResult = set.has(query) ? 'deleted' : 'absent';
    set.delete(query);
  } else if (operation === 'exact') expectedResult = set.has(query) ? 'found' : 'absent';else expectedResult = query === '' || words.some(word => word.startsWith(query)) ? 'found' : 'absent';
  assert.equal(end.result, expectedResult);
  assert.deepEqual(trieWords(end.trie), [...set].sort());
  if (operation === 'prefix') assert.deepEqual(end.matches, [...set].filter(word => word.startsWith(query)).sort());
  assert.deepEqual(source, baseline, 'Trie operation mutated its input');
  for (const frame of trace) {
    validTrie(frame.trie);
    assert(query.startsWith(frame.consumed));
    if (frame.result !== 'limit') assert(frame.trie.nodes.length <= 32);
  }
  // Every node of the finished pruned trie must represent a prefix of a stored word,
  // with the empty root always retained even when no word remains.
  for (const node of end.trie.nodes) assert(node.prefix === '' || [...set].some(word => word.startsWith(node.prefix)));
  if (operation === 'delete') for (const node of end.trie.nodes) assert(source.nodes.some(original => original.id === node.id && original.prefix === node.prefix));
  const layout = trieLayout(end.trie),
    positions = new Map(layout.nodes.map(node => [node.id, node]));
  assert.equal(layout.nodes.length, end.trie.nodes.length);
  assert.equal(layout.edges.length, end.trie.nodes.length - 1);
  for (const edge of layout.edges) {
    const from = positions.get(edge.fromId),
      to = positions.get(edge.toId);
    assert.equal(from.children[edge.character], to.id);
    assert.equal(to.prefix, from.prefix + edge.character);
    assert.equal(to.depth, from.depth + 1);
    assert(to.x > from.x);
  }
  for (const node of layout.nodes) {
    assert(node.x >= 20 && node.x + 20 <= layout.width);
    assert(node.y >= 20 && node.y + 38 <= layout.height);
  }
  trieCases++;
}
for (let mask = 0; mask < 2 ** possibleWords.length; mask++) {
  const words = possibleWords.filter((_, index) => mask & 1 << index);
  for (const query of [...possibleWords, 'c', 'aba']) for (const operation of ['exact', 'prefix', 'insert', 'delete']) verifyTrie(words, query, operation);
}
for (const query of ['car', 'cart', 'ca', 'cat', 'dog', '']) for (const operation of ['exact', 'prefix', 'insert', 'delete']) verifyTrie(TRIE_SAMPLE, query, operation);
assert.deepEqual(trieWords(last(trieOperationTrace(buildTrie(), 'car', 'delete')).trie), ['cart', 'cat', 'dog']);
const cartDeleted = last(trieOperationTrace(buildTrie(), 'cart', 'delete'));
assert.equal(cartDeleted.removedIds.length, 1);
assert.deepEqual(trieWords(cartDeleted.trie), ['car', 'cat', 'dog']);
assert.equal(last(trieOperationTrace(buildTrie([]), '', 'prefix')).result, 'found');
assert.equal(last(trieOperationTrace(buildTrie([]), '', 'exact')).result, 'absent');
assert.equal(last(trieOperationTrace(buildTrie(['']), '', 'exact')).result, 'found');
const fullTrie = buildTrie(['aaaaaa', 'bbbbbb', 'cccccc', 'dddddd', 'eeeeee']);
assert.equal(fullTrie.nodes.length, 31);
assert.equal(last(trieOperationTrace(fullTrie, 'ffffff', 'insert')).result, 'limit');
assert.deepEqual(last(trieOperationTrace(fullTrie, 'ffffff', 'insert')).trie, fullTrie);
assert.throws(() => trieOperationTrace(buildTrie(), 'a', 'unknown'), TypeError);
for (const mutate of [trie => {
  trie.nodes[1].prefix = 'wrong';
}, trie => {
  trie.nodes[0].children.z = 'missing';
}, trie => {
  trie.nodes[1].children.z = trie.rootId;
}, trie => {
  trie.nodes.push({
    ...trie.nodes[1]
  });
}, trie => {
  trie.nodes.push({
    id: 'orphan',
    prefix: 'orphan',
    terminal: true,
    children: {}
  });
}]) {
  const broken = buildTrie();
  mutate(broken);
  assert.equal(validateTrie(broken).valid, false);
}
for (const value of ['', '1, -2, 0', '-99,99']) assert.equal(parseHeapValues(value).valid, true);
for (const value of ['1,', '1,,2', 'NaN', '1e2', '100', '-100', Array(13).fill(1).join(',')]) assert.equal(parseHeapValues(value).valid, false, value);
for (const value of ['', '1,2', 'abc', 'Infinity']) assert.equal(parseHeapKey(value).valid, false, value);
assert.deepEqual(parseHeapKey('0'), {
  valid: true,
  key: 0,
  error: null
});
for (const value of ['', 'ε', 'car, cart, cat, dog', 'a,a']) assert.equal(parseTrieWords(value).valid, true, value);
for (const value of ['car,', 'CAR', 'a,,b', 'a b', 'abcdefg', 'é', 'aaaaaa,bbbbbb,cccccc,dddddd,eeeeee,ffffff']) assert.equal(parseTrieWords(value).valid, false, value);
assert.deepEqual(parseTrieWords('a,a,ε').words, ['a', '']);
for (const value of ['', 'ε', 'car']) assert.equal(parseTrieQuery(value).valid, true, value);
for (const value of ['Car', 'a b', 'abcdefg', 'é', ' car']) assert.equal(parseTrieQuery(value).valid, false, value);
const heapIsolation = heapOperationTrace(),
  firstHeap = structuredClone(heapIsolation[0]);
last(heapIsolation).heap[0] = 99;
assert.deepEqual(heapIsolation[0], firstHeap);
const trieIsolation = trieOperationTrace(buildTrie(), 'car', 'delete'),
  firstTrie = structuredClone(trieIsolation[0]);
last(trieIsolation).trie.nodes[0].terminal = true;
assert.deepEqual(trieIsolation[0], firstTrie);
const streamIsolation = topKStreamTrace(),
  firstCommit = structuredClone(streamIsolation[2]);
last(streamIsolation).heap[0].value = 99;
assert.deepEqual(streamIsolation[2], firstCommit);
console.log(`Heap/trie models verified: ${heapCases} heap operations, ${streamCases} streams checked at every prefix, ${trieCases} trie operations; independent sorted/set oracles, exact identity/source conservation, layout, construction bounds, parsing, limits and snapshot isolation.`);
