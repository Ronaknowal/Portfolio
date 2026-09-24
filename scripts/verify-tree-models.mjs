import assert from 'node:assert/strict';
import { TREE_SAMPLE, TREE_LIMIT, buildTree, inorderKeys, validateBST, treeMetrics, treeLayout, parseTreeKeys, parseTreeTarget, searchTrace, insertionTrace, traversalTrace, deletionTrace, invalidBoundsExample, rotationExample } from '../src/learn/data/tree-models.js';

let cases = 0;
const last = trace => trace.at(-1);
const sortedSet = values => [...new Set(values)].sort((a, b) => a - b);
const get = (tree, id) => tree.nodes.find(node => node.id === id);
const valid = tree => { const result = validateBST(tree); assert.equal(result.valid, true, result.errors.join('\n')); };

// Independent reference representation: nested nodes built by recursive partition,
// rather than the production model's iterative pointer insertion and snapshots.
function reference(values) {
  if (!values.length) return null;
  const [key, ...rest] = values;
  return { key, left: reference(rest.filter(item => item < key)), right: reference(rest.filter(item => item > key)) };
}
function referenceWalk(node, order) {
  if (order === 'level-order') {
    let level = node ? [node] : [], out = [];
    while (level.length) { out.push(...level.map(item => item.key)); level = level.flatMap(item => [item.left, item.right].filter(Boolean)); }
    return out;
  }
  if (!node) return [];
  const left = referenceWalk(node.left, order), right = referenceWalk(node.right, order);
  return order === 'preorder' ? [node.key, ...left, ...right] : order === 'inorder' ? [...left, node.key, ...right] : [...left, ...right, node.key];
}
function referenceSearch(node, key) {
  let lower = null, upper = null;
  const keys = [], intervals = [];
  while (node) {
    keys.push(node.key); intervals.push([lower, upper]);
    if (node.key === key) return { keys, intervals, found: true };
    if (key < node.key) { upper = node.key; node = node.left; } else { lower = node.key; node = node.right; }
  }
  return { keys, intervals, found: false, lower, upper };
}
function referenceHeight(node) { return node ? 1 + Math.max(referenceHeight(node.left), referenceHeight(node.right)) : -1; }
function verifyLayout(tree) {
  const layout = treeLayout(tree), positions = new Map(layout.nodes.map(node => [node.id, node]));
  assert.equal(positions.size, tree.nodes.length);
  assert.equal(layout.edges.length, Math.max(0, tree.nodes.length - 1));
  for (const node of layout.nodes) {
    assert(node.x >= 22 && node.x <= layout.width - 22);
    assert(node.y >= 22 && node.y <= layout.height - 22);
    assert.equal(node.y, 55 + 88 * node.depth);
    for (const [side, childId] of [['left', node.leftId], ['right', node.rightId]]) {
      if (childId === null) continue;
      const child = positions.get(childId);
      assert.equal(child.depth, node.depth + 1);
      assert(side === 'left' ? child.x < node.x : child.x > node.x);
      assert(layout.edges.some(edge => edge.fromId === node.id && edge.toId === childId && edge.side === side));
    }
  }
}
function verifySearch(tree, values, key) {
  const oracle = referenceSearch(reference(values), key), trace = searchTrace(tree, key), result = last(trace);
  assert.equal(result.result, oracle.found ? 'found' : 'absent');
  assert.deepEqual(result.visitedIds.map(id => get(tree, id).key), oracle.keys);
  assert.equal(result.comparisonCount, oracle.keys.length);
  const comparisons = trace.filter(frame => frame.comparisonCount > 0 && frame.activeId !== null);
  assert.deepEqual(comparisons.map(frame => [frame.lower, frame.upper]), oracle.intervals);
  if (!oracle.found) assert.deepEqual([result.lower, result.upper], [oracle.lower, oracle.upper]);
  for (const frame of trace) { assert.deepEqual(frame.tree, tree); assert.equal(new Set(frame.visitedIds).size, frame.visitedIds.length); }
}
function verifyTraversals(tree, values) {
  for (const order of ['preorder', 'inorder', 'postorder', 'level-order']) {
    const trace = traversalTrace(tree, order), expected = referenceWalk(reference(values), order);
    assert.deepEqual(last(trace).output, expected);
    assert.deepEqual(last(trace).frontier, []);
    let previous = [];
    for (const frame of trace) {
      assert.deepEqual(frame.tree, tree);
      assert.deepEqual(frame.output.slice(0, previous.length), previous);
      assert(frame.output.length === previous.length || frame.output.length === previous.length + 1);
      assert.deepEqual(frame.outputIds.map(id => get(tree, id).key), frame.output);
      assert.equal(new Set(frame.outputIds).size, frame.outputIds.length);
      assert.equal(new Set(frame.frontier.map(item => item.nodeId)).size, frame.frontier.length);
      if (order !== 'level-order') {
        for (let index = 1; index < frame.frontier.length; index++) {
          const parent = get(tree, frame.frontier[index - 1].nodeId);
          assert([parent.leftId, parent.rightId].includes(frame.frontier[index].nodeId));
        }
        assert(frame.frontier.length <= treeMetrics(tree).height + 1);
      }
      previous = frame.output;
    }
    cases++;
  }
}
function verifyDeletion(tree, values, key) {
  const baseline = structuredClone(tree), trace = deletionTrace(tree, key), result = last(trace), present = values.includes(key);
  assert.deepEqual(tree, baseline, 'Deletion mutated the input tree');
  valid(result.tree); verifyLayout(result.tree);
  assert.deepEqual(inorderKeys(result.tree), sortedSet(values.filter(item => item !== key)));
  assert.equal(result.result, present ? 'deleted' : 'absent');
  assert.equal(result.tree.nodes.length, tree.nodes.length - Number(present));
  assert.equal(result.tree.nextId, tree.nextId, 'Deletion must not allocate identities');
  const removed = tree.nodes.filter(node => !result.tree.nodes.some(item => item.id === node.id));
  assert.equal(removed.length, Number(present));
  if (present) assert.equal(removed[0].id, result.removedId); else assert.deepEqual(result.tree, tree);
  for (const frame of trace) {
    if (!frame.transient) valid(frame.tree);
    else {
      assert.equal(validateBST(frame.tree).valid, false, 'The explicitly intermediate copy must expose its duplicate');
      assert.equal(frame.tree.nodes.length, tree.nodes.length);
      assert.equal(get(frame.tree, frame.targetId).key, get(frame.tree, frame.successorId).key);
      assert.equal(frame.tree.nodes.filter(node => node.key === get(frame.tree, frame.targetId).key).length, 2);
    }
  }
  cases++;
}
function* permutations(values) {
  if (!values.length) { yield []; return; }
  for (let index = 0; index < values.length; index++) for (const rest of permutations(values.filter((_, i) => index !== i))) yield [values[index], ...rest];
}

const sample = buildTree();
assert.deepEqual(inorderKeys(sample), [1, 3, 4, 6, 7, 8, 10, 13, 14]);
assert.deepEqual(treeMetrics(sample), { size: 9, height: 3, diameter: 6, leafCount: 4 });
assert.deepEqual(treeMetrics(buildTree([])), { size: 0, height: -1, diameter: 0, leafCount: 0 });
assert.deepEqual(treeMetrics(buildTree([8])), { size: 1, height: 0, diameter: 0, leafCount: 1 });
verifyTraversals(sample, TREE_SAMPLE);
verifyLayout(sample);
for (const key of [-99, 0, ...TREE_SAMPLE, 5, 9, 99]) verifySearch(sample, TREE_SAMPLE, key);
assert.equal(last(searchTrace(sample, 14)).comparisonCount, 3);
const ascending = buildTree(sortedSet(TREE_SAMPLE));
assert.equal(treeMetrics(ascending).height, 8);
assert.equal(last(searchTrace(ascending, 14)).comparisonCount, 9);
const insertFive = last(insertionTrace(sample, 5));
assert.deepEqual([insertFive.lower, insertFive.upper], [4, 6]);
assert.equal(insertFive.tree.nodes.length, 10);
assert.equal(get(insertFive.tree, 'n10').key, 5);
assert.equal(get(insertFive.tree, 'n7').rightId, 'n10');
assert.deepEqual(last(insertionTrace(sample, 6)).tree, sample);
assert.equal(last(insertionTrace(sample, 6)).result, 'duplicate');

for (const order of permutations([-2, -1, 0, 1, 2])) {
  const tree = buildTree(order), baseline = structuredClone(tree);
  valid(tree); verifyLayout(tree); verifyTraversals(tree, order);
  assert.deepEqual(inorderKeys(tree), sortedSet(order));
  assert.equal(treeMetrics(tree).height, referenceHeight(reference(order)));
  for (const key of [-3, ...order, 3]) {
    verifySearch(tree, order, key); verifyDeletion(tree, order, key);
    const inserted = last(insertionTrace(tree, key));
    valid(inserted.tree);
    assert.deepEqual(inorderKeys(inserted.tree), sortedSet([...order, key]));
    assert.equal(inserted.result, order.includes(key) ? 'duplicate' : 'inserted');
    assert.equal(inserted.tree.nodes.length, tree.nodes.length + Number(!order.includes(key)));
    cases++;
  }
  assert.deepEqual(tree, baseline, 'A trace operation mutated the input tree');
}
for (const values of [[], [8], [8, 3], [8, 10], TREE_SAMPLE, [20, 10, 40, 30, 50, 35]]) for (const key of [...values, -99]) verifyDeletion(buildTree(values), values, key);

const rootDelete = last(deletionTrace(sample, 8));
assert.equal(rootDelete.tree.rootId, 'n1');
assert.equal(get(rootDelete.tree, 'n1').key, 10);
assert.equal(get(rootDelete.tree, 'n1').rightId, 'n6');
assert.equal(rootDelete.removedId, 'n3');
const deeper = last(deletionTrace(buildTree([20, 10, 40, 30, 50, 35]), 20));
assert.equal(deeper.tree.rootId, 'n1');
assert.equal(get(deeper.tree, 'n1').key, 30);
assert.equal(get(deeper.tree, 'n3').leftId, 'n6');
assert.equal(get(deeper.tree, 'n6').key, 35);
assert.equal(deeper.removedId, 'n4');
assert.deepEqual(inorderKeys(deeper.tree), [10, 30, 35, 40, 50]);

// Reject malformed global structure, not merely immediate parent ordering.
assert.equal(validateBST(invalidBoundsExample()).valid, false);
for (const mutate of [
  tree => { get(tree, 'n4').leftId = tree.rootId; },
  tree => { get(tree, 'n3').leftId = 'n4'; },
  tree => { get(tree, 'n4').leftId = 'missing'; },
  tree => { tree.nodes.push({ id: 'orphan', key: 22, leftId: null, rightId: null }); },
  tree => { tree.nodes.push({ ...tree.nodes[0] }); },
  tree => { get(tree, 'n2').key = 8; },
]) { const broken = structuredClone(sample); mutate(broken); assert.equal(validateBST(broken).valid, false); }

const full = buildTree(Array.from({ length: TREE_LIMIT }, (_, i) => i));
assert.equal(last(insertionTrace(full, 90)).result, 'limit');
assert.deepEqual(last(insertionTrace(full, 90)).tree, full);
assert.equal(last(insertionTrace(full, 2)).result, 'duplicate');
assert.equal(treeMetrics(full).height, TREE_LIMIT - 1);
assert.equal(treeMetrics(full).diameter, TREE_LIMIT - 1);
assert.equal(last(searchTrace(full, TREE_LIMIT - 1)).comparisonCount, TREE_LIMIT);
for (const text of ['', '  ', '1, 2, -3', '-99,99', '8,8']) assert.equal(parseTreeKeys(text).valid, true);
for (const text of ['1,,2', '1,', 'a', '2.5', 'Infinity', 'NaN', '100', '-100', '9007199254740992', Array.from({length:13},(_,i)=>i).join(',')]) assert.equal(parseTreeKeys(text).valid, false, text);
for (const text of ['', '1,2', 'NaN', '1e2', '0.2']) assert.equal(parseTreeTarget(text).valid, false, text);
assert.deepEqual(parseTreeTarget(' -9 '), { valid: true, key: -9, error: null });
assert.throws(() => buildTree([1, NaN]), TypeError);
assert.throws(() => searchTrace(sample, Infinity), TypeError);
assert.throws(() => traversalTrace(sample, 'diagonal'), TypeError);

const rotation = rotationExample();
valid(rotation.before); valid(rotation.after);
assert.deepEqual(inorderKeys(rotation.before), [10, 20, 25, 30, 40]);
assert.deepEqual(inorderKeys(rotation.after), inorderKeys(rotation.before));
assert.deepEqual(rotation.before.nodes.map(node => [node.id, node.key]), rotation.after.nodes.map(node => [node.id, node.key]));
assert.equal(rotation.after.rootId, 'n2');
assert.equal(get(rotation.after, 'n1').leftId, rotation.transferredId);
assert.equal(get(rotation.after, 'n2').rightId, 'n1');
verifyLayout(rotation.before); verifyLayout(rotation.after);
// Trace snapshots are isolated from one another and from the original model.
const isolation = deletionTrace(sample, 8), originalFirst = structuredClone(isolation[0]);
last(isolation).tree.nodes[0].key = 999;
assert.deepEqual(isolation[0], originalFirst);
assert.equal(sample.nodes[0].key, 8);
console.log(`Tree models verified: ${cases} traversal/insertion/deletion cases, 120 insertion permutations, ancestor-bound search traces, identity-preserving deletion, layout, rotation, parsing, limits and snapshot isolation.`);
