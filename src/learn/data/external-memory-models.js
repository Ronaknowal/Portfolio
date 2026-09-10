function boundedInteger(value, low, high, label) {
  if (!Number.isInteger(value) || value < low || value > high) throw new RangeError(`${label}: use an integer from ${low} through ${high}.`);
}
export function bufferTrace(pageSize = 4, capacity = 2, pattern = 'sequential') {
  boundedInteger(pageSize, 1, 8, 'Records per page');
  boundedInteger(capacity, 1, 6, 'Buffer frames');
  const patterns = {
    sequential: Array.from({
      length: 16
    }, (_, address) => [address, 'read']),
    strided: Array.from({
      length: 16
    }, (_, index) => [index % 4 * 4 + Math.floor(index / 4), 'read']),
    reuse: [0, 4, 0, 8, 4, 0, 12, 1, 4].map(address => [address, 'read']),
    writes: [[0, 'write'], [1, 'write'], [4, 'read'], [0, 'write'], [8, 'write'], [12, 'read'], [0, 'read']]
  };
  if (!(pattern in patterns)) throw new RangeError('Choose a known request pattern.');
  const requests = patterns[pattern];
  const resident = new Map();
  let reads = 0,
    writes = 0,
    hits = 0;
  const frames = [];
  const record = (action, address = null, page = null, evicted = null) => frames.push({
    action,
    address,
    page,
    evicted,
    reads,
    writes,
    hits,
    resident: [...resident].map(([id, dirty]) => ({
      id,
      dirty
    }))
  });
  record('Cold buffer: no pages resident');
  for (const [address, operation] of requests) {
    const page = Math.floor(address / pageSize);
    const hit = resident.has(page);
    let dirty = resident.get(page) || false;
    let evicted = null;
    if (hit) {
      hits++;
      resident.delete(page);
    } else {
      reads++;
      if (resident.size === capacity) {
        const [id, wasDirty] = resident.entries().next().value;
        evicted = {
          id,
          dirty: wasDirty
        };
        if (wasDirty) writes++;
        resident.delete(id);
      }
    }
    dirty ||= operation === 'write';
    resident.set(page, dirty);
    record(`${operation} record ${address}: ${hit ? 'hit' : 'load page'}${evicted ? `; evict ${evicted.dirty ? 'dirty' : 'clean'} P${evicted.id}` : ''}`, address, page, evicted);
  }
  for (const [id, dirty] of resident) {
    if (!dirty) continue;
    writes++;
    resident.set(id, false);
    record(`Final flush of dirty P${id}`, null, id);
  }
  return {
    pageSize,
    capacity,
    requests,
    frames
  };
}
class BTree {
  constructor(degree) {
    this.degree = degree;
    this.nextId = 0;
    this.root = this.node();
    this.events = [];
  }
  node(keys = [], children = []) {
    return {
      id: `P${this.nextId++}`,
      keys,
      children
    };
  }
  snapshot(action, pages = []) {
    this.events.push({
      action,
      pages,
      root: structuredClone(this.root)
    });
  }
  search(key) {
    const visited = [];
    let node = this.root;
    while (true) {
      visited.push(node.id);
      let index = 0;
      while (index < node.keys.length && node.keys[index] < key) index++;
      if (node.keys[index] === key) return {
        found: true,
        visited
      };
      if (!node.children.length) return {
        found: false,
        visited
      };
      node = node.children[index];
    }
  }
  split(parent, index) {
    const child = parent.children[index];
    const middle = this.degree - 1;
    const separator = child.keys[middle];
    const right = this.node(child.keys.slice(middle + 1), child.children.length ? child.children.slice(this.degree) : []);
    child.keys = child.keys.slice(0, middle);
    if (child.children.length) child.children = child.children.slice(0, this.degree);
    parent.keys.splice(index, 0, separator);
    parent.children.splice(index + 1, 0, right);
    this.snapshot(`Split ${child.id}; promote ${separator} into ${parent.id}; create ${right.id}`, [parent.id, child.id, right.id]);
  }
  insert(key) {
    if (this.search(key).found) {
      this.snapshot(`${key} already exists: set unchanged`);
      return false;
    }
    if (this.root.keys.length === 2 * this.degree - 1) {
      this.root = this.node([], [this.root]);
      this.split(this.root, 0);
    }
    let node = this.root;
    while (node.children.length) {
      let index = 0;
      while (index < node.keys.length && key > node.keys[index]) index++;
      if (node.children[index].keys.length === 2 * this.degree - 1) {
        this.split(node, index);
        if (key > node.keys[index]) index++;
      }
      node = node.children[index];
    }
    const index = node.keys.findIndex(value => value > key);
    node.keys.splice(index < 0 ? node.keys.length : index, 0, key);
    this.snapshot(`Insert ${key} into leaf ${node.id}`, [node.id]);
    return true;
  }
  merge(parent, index) {
    const left = parent.children[index],
      right = parent.children[index + 1];
    const separator = parent.keys.splice(index, 1)[0];
    left.keys.push(separator, ...right.keys);
    left.children.push(...right.children);
    parent.children.splice(index + 1, 1);
    this.snapshot(`Merge ${left.id}, parent key ${separator} and ${right.id} into ${left.id}`, [parent.id, left.id, right.id]);
    return left;
  }
  remove(key) {
    if (!this.search(key).found) {
      this.snapshot(`${key} is absent: set unchanged`);
      return false;
    }
    const visit = (node, target) => {
      let index = 0;
      while (index < node.keys.length && target > node.keys[index]) index++;
      if (node.keys[index] === target) {
        if (!node.children.length) {
          node.keys.splice(index, 1);
          this.snapshot(`Remove ${target} from leaf ${node.id}`, [node.id]);
        } else if (node.children[index].keys.length >= this.degree) {
          let predecessor = node.children[index];
          while (predecessor.children.length) predecessor = predecessor.children.at(-1);
          const replacement = predecessor.keys.at(-1);
          node.keys[index] = replacement;
          this.snapshot(`Replace internal ${target} by predecessor ${replacement}; remove its leaf occurrence next`, [node.id, predecessor.id]);
          visit(node.children[index], replacement);
        } else if (node.children[index + 1].keys.length >= this.degree) {
          let successor = node.children[index + 1];
          while (successor.children.length) successor = successor.children[0];
          const replacement = successor.keys[0];
          node.keys[index] = replacement;
          this.snapshot(`Replace internal ${target} by successor ${replacement}; remove its leaf occurrence next`, [node.id, successor.id]);
          visit(node.children[index + 1], replacement);
        } else visit(this.merge(node, index), target);
        return;
      }
      let child = node.children[index];
      if (child.keys.length === this.degree - 1) {
        const left = node.children[index - 1],
          right = node.children[index + 1];
        if (left && left.keys.length >= this.degree) {
          child.keys.unshift(node.keys[index - 1]);
          node.keys[index - 1] = left.keys.pop();
          if (left.children.length) child.children.unshift(left.children.pop());
          this.snapshot(`Borrow from left sibling through parent ${node.id} into ${child.id}`, [node.id, left.id, child.id]);
        } else if (right && right.keys.length >= this.degree) {
          child.keys.push(node.keys[index]);
          node.keys[index] = right.keys.shift();
          if (right.children.length) child.children.push(right.children.shift());
          this.snapshot(`Borrow from right sibling through parent ${node.id} into ${child.id}`, [node.id, right.id, child.id]);
        } else {
          if (!right) index--;
          child = this.merge(node, index);
        }
      }
      visit(child, target);
    };
    visit(this.root, key);
    if (this.root.keys.length === 0 && this.root.children.length) {
      const previous = this.root.id;
      this.root = this.root.children[0];
      this.snapshot(`Root ${previous} is empty: promote ${this.root.id}; every leaf moves up one level`, [this.root.id]);
    }
    return true;
  }
}
export const btreePresets = {
  insert: {
    initial: [],
    operations: [10, 20, 5, 6, 12, 30, 7, 17].map(key => ['insert', key])
  },
  delete: {
    initial: Array.from({
      length: 16
    }, (_, index) => index + 1),
    operations: [6, 13, 7, 4, 2, 16, 15, 14, 12, 11, 10, 9, 8, 5, 3, 1].map(key => ['delete', key])
  },
  duplicates: {
    initial: [10, 5, 20],
    operations: [['insert', 10], ['delete', 99], ['delete', 10], ['insert', 12]]
  }
};
export function btreeTrace(degree = 2, preset = 'insert', extra = []) {
  boundedInteger(degree, 2, 4, 'Minimum degree');
  if (!(preset in btreePresets) || !Array.isArray(extra) || extra.length > 12) throw new RangeError('Choose a known tree scenario and at most 12 added operations.');
  const tree = new BTree(degree);
  const {
    initial,
    operations
  } = btreePresets[preset];
  for (const key of initial) tree.insert(key);
  tree.events = [];
  tree.snapshot('Initial valid B-tree');
  const completed = [];
  for (const [operation, key] of [...operations, ...extra]) {
    boundedInteger(key, 0, 99, 'Key');
    if (!['insert', 'delete'].includes(operation)) throw new RangeError('Choose insert or delete.');
    const before = tree.search(key);
    tree.snapshot(`${operation} ${key}: lookup visits ${before.visited.join(' → ')}`, before.visited);
    if (operation === 'insert') tree.insert(key);else tree.remove(key);
    tree.snapshot(`Complete ${operation} ${key}: full set/occupancy invariant restored`);
    completed.push({
      operation,
      key,
      root: structuredClone(tree.root)
    });
  }
  return {
    degree,
    frames: tree.events,
    completed,
    root: structuredClone(tree.root),
    search: key => tree.search(key)
  };
}
export function treeKeys(root) {
  if (!root.children.length) return [...root.keys];
  return root.children.flatMap((child, index) => [...treeKeys(child), ...(index < root.keys.length ? [root.keys[index]] : [])]);
}
function balancedGroups(values, capacity) {
  const count = Math.ceil(values.length / capacity);
  const small = Math.floor(values.length / count);
  let offset = 0;
  return Array.from({
    length: count
  }, (_, index) => {
    const size = small + (index < values.length % count ? 1 : 0);
    const group = values.slice(offset, offset + size);
    offset += size;
    return group;
  });
}
export const bplusRecords = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
export function bplusRange(low = 10, high = 27, leafCapacity = 3, fanout = 3) {
  boundedInteger(low, 0, 40, 'Range start');
  boundedInteger(high, 0, 40, 'Range end');
  boundedInteger(leafCapacity, 2, 4, 'Leaf capacity');
  boundedInteger(fanout, 2, 4, 'Fanout');
  let nextId = 0;
  const leaves = balancedGroups(bplusRecords, leafCapacity).map(keys => ({
    id: `L${nextId++}`,
    keys,
    children: []
  }));
  leaves.forEach((leaf, index) => {
    leaf.next = leaves[index + 1]?.id || null;
  });
  let level = leaves;
  const firstKey = node => node.children.length ? firstKey(node.children[0]) : node.keys[0];
  while (level.length > 1) {
    level = balancedGroups(level, fanout).map(children => ({
      id: `I${nextId++}`,
      keys: children.slice(1).map(firstKey),
      children
    }));
  }
  const root = level[0];
  const visited = [],
    result = [],
    frames = [];
  if (low > high) return {
    root,
    leaves,
    visited,
    result,
    frames: [{
      action: 'Empty reversed range: no page needed',
      page: null,
      result: []
    }]
  };
  let node = root;
  while (node.children.length) {
    visited.push(node.id);
    let index = 0;
    while (index < node.keys.length && low >= node.keys[index]) index++;
    frames.push({
      action: `Read ${node.id}; start ${low} chooses child ${index + 1}; separator equality routes right`,
      page: node.id,
      result: []
    });
    node = node.children[index];
  }
  let index = leaves.findIndex(leaf => leaf.id === node.id);
  while (index < leaves.length) {
    const leaf = leaves[index++];
    visited.push(leaf.id);
    result.push(...leaf.keys.filter(key => key >= low && key <= high));
    frames.push({
      action: `Read ${leaf.id}; retain records in inclusive [${low},${high}]`,
      page: leaf.id,
      result: [...result]
    });
    if (leaf.keys.at(-1) > high) break;
  }
  return {
    root,
    leaves,
    visited,
    result,
    frames
  };
}
export function mergePlan(recordCount = 32, pageSize = 4, memoryPages = 3) {
  boundedInteger(recordCount, 0, 96, 'Record count');
  boundedInteger(pageSize, 1, 8, 'Records per page');
  boundedInteger(memoryPages, 3, 8, 'Memory pages');
  const memoryRecords = pageSize * memoryPages,
    fanIn = memoryPages - 1;
  // A fixed permutation of 0..N−1; data ordering is deterministic, not a random benchmark.
  const input = Array.from({
    length: recordCount
  }, (_, index) => recordCount - 1 - index);
  let runs = [];
  for (let offset = 0; offset < input.length; offset += memoryRecords) runs.push(input.slice(offset, offset + memoryRecords).sort((a, b) => a - b));
  const pages = values => Math.ceil(values.length / pageSize);
  const stages = [];
  if (!input.length) return {
    recordCount,
    pageSize,
    memoryPages,
    memoryRecords,
    fanIn,
    input,
    stages,
    totalReads: 0,
    totalWrites: 0,
    result: []
  };
  stages.push({
    name: 'Form sorted runs',
    inputRuns: [input],
    runs,
    groups: [],
    reads: pages(input),
    writes: runs.reduce((sum, run) => sum + pages(run), 0)
  });
  while (runs.length > 1) {
    const groups = [],
      outputRuns = [];
    for (let start = 0; start < runs.length; start += fanIn) {
      const group = runs.slice(start, start + fanIn);
      const positions = group.map(() => 0),
        output = [];
      while (true) {
        let selected = -1;
        for (let index = 0; index < group.length; index++) {
          if (positions[index] >= group[index].length) continue;
          if (selected < 0 || group[index][positions[index]] < group[selected][positions[selected]]) selected = index;
        }
        if (selected < 0) break;
        output.push(group[selected][positions[selected]++]);
      }
      groups.push({
        inputIndices: group.map((_, index) => start + index),
        output
      });
      outputRuns.push(output);
    }
    stages.push({
      name: `Merge pass ${stages.length}`,
      inputRuns: runs,
      runs: outputRuns,
      groups,
      reads: runs.reduce((sum, run) => sum + pages(run), 0),
      writes: outputRuns.reduce((sum, run) => sum + pages(run), 0)
    });
    runs = outputRuns;
  }
  return {
    recordCount,
    pageSize,
    memoryPages,
    memoryRecords,
    fanIn,
    input,
    stages,
    totalReads: stages.reduce((sum, stage) => sum + stage.reads, 0),
    totalWrites: stages.reduce((sum, stage) => sum + stage.writes, 0),
    result: runs[0]
  };
}
export function shadowCommitTrace(earlyRoot = false) {
  const pages = {
    oldLeaf: {
      value: 5
    },
    oldRoot: {
      child: 'oldLeaf'
    },
    newLeaf: {
      value: 9
    },
    newRoot: {
      child: 'newLeaf'
    }
  };
  const durable = new Set(['oldLeaf', 'oldRoot']),
    pending = new Set();
  let committedRoot = 'oldRoot';
  const frames = [];
  const record = action => {
    const missing = [];
    const inspect = id => {
      if (!durable.has(id)) {
        missing.push(id);
        return;
      }
      if (pages[id].child) inspect(pages[id].child);
    };
    inspect(committedRoot);
    frames.push({
      action,
      durable: [...durable],
      pending: [...pending],
      committedRoot,
      missing,
      recoveredValue: missing.length ? null : pages[pages[committedRoot].child].value
    });
  };
  record('Old durable root reaches value 5');
  pending.add('newLeaf');
  pending.add('newRoot');
  record('Prepare new leaf and root in volatile memory');
  if (earlyRoot) {
    committedRoot = 'newRoot';
    record('Wrong ordering: atomically publish the new root before its pages are durable');
  }
  for (const id of ['newLeaf', 'newRoot']) {
    pending.delete(id);
    durable.add(id);
    record(`Complete durable write of ${id}`);
  }
  if (!earlyRoot) {
    committedRoot = 'newRoot';
    record('Atomically publish durable root metadata only after both pages are durable');
  }
  record('Acknowledge commit; old pages still retained in this model');
  return {
    pages,
    earlyRoot,
    frames
  };
}
