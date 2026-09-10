import assert from 'node:assert/strict';
import { assignPersistent, createPersistentStore, historyLookup, HISTORY_WRITES, parsePersistentValues, persistentQuery, prefixRankModel, reachablePersistentNodes } from '../src/learn/data/persistent-structures-models.js';
let intervalChecks = 0;
let rankChecks = 0;
let seed = 1234567;
// Exact identity/edge contract used by the compact inline figure.
const figureStore = assignPersistent(createPersistentStore(), 0, 2, 9).store;
assert.deepEqual(figureStore.versions.map(version => version.root), [8, 11]);
assert.deepEqual([8, 11, 7, 10].map(id => [figureStore.nodes[id].left, figureStore.nodes[id].right]), [[2, 7], [2, 10], [3, 6], [9, 6]]);
assert.deepEqual([8, 11, 2, 7, 10, 6, 3, 9].map(id => figureStore.nodes[id].sum), [15, 20, 3, 12, 17, 8, 4, 9]);
function random(limit) {
  seed = 1664525 * seed + 1013904223 >>> 0;
  return seed % limit;
}
function expectedPathLength(size, index) {
  let low = 0;
  let high = size;
  let count = 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (index < middle) high = middle;else low = middle;
    count++;
  }
  return count;
}
for (let size = 1; size <= 8; size++) {
  for (let trial = 0; trial < 100; trial++) {
    const initial = Array.from({
      length: size
    }, () => random(11) - 5);
    let store = createPersistentStore(initial);
    const snapshots = [initial];
    assert.equal(store.nodes.length, 2 * size - 1);
    for (let step = 0; step < 7; step++) {
      const source = random(snapshots.length);
      const index = random(size);
      const value = random(11) - 5;
      const before = JSON.stringify(store);
      const priorStore = store;
      const result = assignPersistent(store, source, index, value);
      assert.equal(JSON.stringify(priorStore), before);
      const changed = snapshots[source][index] !== value;
      assert.equal(result.copied.length, changed ? expectedPathLength(size, index) : 0);
      assert(result.copied.length <= Math.ceil(Math.log2(size)) + 1);
      store = result.store;
      const snapshot = [...snapshots[source]];
      snapshot[index] = value;
      snapshots.push(snapshot);
      for (let version = 0; version < snapshots.length; version++) {
        for (let low = 0; low <= size; low++) {
          for (let high = low; high <= size; high++) {
            const query = persistentQuery(store, version, low, high);
            assert.equal(query.sum, snapshots[version].slice(low, high).reduce((a, b) => a + b, 0));
            const indices = query.cover.flatMap(id => Array.from({
              length: store.nodes[id].high - store.nodes[id].low
            }, (_, i) => store.nodes[id].low + i));
            assert.deepEqual(indices, Array.from({
              length: high - low
            }, (_, i) => low + i));
            intervalChecks++;
          }
        }
      }
      for (const [oldId, freshId] of result.copied) {
        const old = store.nodes[oldId];
        const fresh = store.nodes[freshId];
        assert.equal(old.low, fresh.low);
        assert.equal(old.high, fresh.high);
        if (old.left !== null) {
          const middle = Math.floor((old.low + old.high) / 2);
          assert.equal(index < middle ? old.right : old.left, index < middle ? fresh.right : fresh.left);
        }
      }
      assert.equal(reachablePersistentNodes(store, [0]).size, 2 * size - 1);
      assert.equal(reachablePersistentNodes(store, []).size, 0);
      for (const node of store.nodes) assert(Object.isFrozen(node));
    }
  }
}
// All arrays up to length five over {-1,0,1}, every subrange and rank.
for (let size = 1; size <= 5; size++) {
  for (let code = 0; code < 3 ** size; code++) {
    let digits = code;
    const values = Array.from({
      length: size
    }, () => {
      const value = digits % 3 - 1;
      digits = Math.floor(digits / 3);
      return value;
    });
    for (let low = 0; low < size; low++) for (let high = low + 1; high <= size; high++) {
      const sorted = values.slice(low, high).sort((a, b) => a - b);
      for (let rank = 1; rank <= sorted.length; rank++) {
        const model = prefixRankModel(values, low, high, rank);
        assert.equal(model.answer, sorted[rank - 1]);
        assert.equal(model.counts.reduce((a, b) => a + b, 0), high - low);
        rankChecks++;
      }
    }
  }
}
for (const history of HISTORY_WRITES) for (let snapshot = 0; snapshot <= 4; snapshot++) {
  const eligible = history.filter(([time]) => time <= snapshot).at(-1);
  assert.equal(historyLookup(history, snapshot).value, eligible?.[1] ?? 0);
}
assert.equal(persistentQuery(createPersistentStore([]), 0, 0, 0).sum, 0);
assert.throws(() => assignPersistent(createPersistentStore([]), 0, 0, 1));
assert.throws(() => parsePersistentValues('1, nope'));
assert.throws(() => parsePersistentValues('100'));
assert.throws(() => persistentQuery(createPersistentStore(), 0, 4, 2));
assert.throws(() => prefixRankModel([1, 2], 0, 2, 3));
console.log(`PASS: ${intervalChecks} historical interval/cover checks, ${rankChecks} sorted rank oracles, 800 branching histories, allocation/identity/immutability/invalid contracts.`);
