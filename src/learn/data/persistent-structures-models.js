// Finite exact-integer teaching model. Nodes are immutable; the append-only arena
// records allocation identity, while version roots determine logical ownership.
export const DEFAULT_PERSISTENT_VALUES = Object.freeze([2, 1, 4, 3, 5]);
function integer(value, low, high, label) {
  if (!Number.isInteger(value) || value < low || value > high) {
    throw new RangeError(`${label} must be an integer from ${low} through ${high}.`);
  }
  return value;
}
export function parsePersistentValues(text) {
  const pieces = text.trim().split(/[\s,]+/).filter(Boolean);
  if (!pieces.length || pieces.length > 8 || pieces.some(piece => !/^-?\d+$/.test(piece))) {
    throw new RangeError('Enter 1–8 integers separated by commas or spaces.');
  }
  return pieces.map(piece => integer(Number(piece), -99, 99, 'Each value'));
}
export function createPersistentStore(values = DEFAULT_PERSISTENT_VALUES) {
  if (!Array.isArray(values) || values.length > 8) throw new RangeError('Use an array of at most 8 integers.');
  values.forEach(value => integer(value, -99, 99, 'Each value'));
  const nodes = [];
  function build(low, high) {
    if (low === high) return null;
    const middle = Math.floor((low + high) / 2);
    const left = high - low === 1 ? null : build(low, middle);
    const right = high - low === 1 ? null : build(middle, high);
    const node = Object.freeze({
      id: nodes.length,
      low,
      high,
      left,
      right,
      sum: high - low === 1 ? values[low] : nodes[left].sum + nodes[right].sum
    });
    nodes.push(node);
    return node.id;
  }
  const root = build(0, values.length);
  return Object.freeze({
    size: values.length,
    nodes: Object.freeze(nodes),
    versions: Object.freeze([Object.freeze({
      root,
      parent: null,
      label: 'initial'
    })])
  });
}
export function assignPersistent(store, version, index, value) {
  integer(version, 0, store.versions.length - 1, 'Source version');
  integer(index, 0, store.size - 1, 'Index');
  integer(value, -99, 99, 'Value');
  if (store.versions.length >= 8) throw new RangeError('Eight versions reached; reset to start another investigation.');
  const nodes = [...store.nodes];
  const copied = [];
  function update(id) {
    const old = nodes[id];
    if (old.high - old.low === 1) {
      if (old.sum === value) return id;
      const fresh = Object.freeze({
        ...old,
        id: nodes.length,
        sum: value
      });
      nodes.push(fresh);
      copied.push([id, fresh.id]);
      return fresh.id;
    }
    const middle = Math.floor((old.low + old.high) / 2);
    const left = index < middle ? update(old.left) : old.left;
    const right = index >= middle ? update(old.right) : old.right;
    if (left === old.left && right === old.right) return id;
    const fresh = Object.freeze({
      ...old,
      id: nodes.length,
      left,
      right,
      sum: nodes[left].sum + nodes[right].sum
    });
    nodes.push(fresh);
    copied.push([id, fresh.id]);
    return fresh.id;
  }
  const root = update(store.versions[version].root);
  const versions = [...store.versions, Object.freeze({
    root,
    parent: version,
    label: `a[${index}] = ${value}`
  })];
  return {
    store: Object.freeze({
      size: store.size,
      nodes: Object.freeze(nodes),
      versions: Object.freeze(versions)
    }),
    copied: Object.freeze(copied.map(pair => Object.freeze(pair)))
  };
}
export function persistentQuery(store, version, low, high) {
  integer(version, 0, store.versions.length - 1, 'Version');
  integer(low, 0, store.size, 'Left endpoint');
  integer(high, low, store.size, 'Right endpoint');
  const cover = [];
  function query(id) {
    if (id === null || low === high) return 0;
    const node = store.nodes[id];
    if (node.high <= low || high <= node.low) return 0;
    if (low <= node.low && node.high <= high) {
      cover.push(id);
      return node.sum;
    }
    return query(node.left) + query(node.right);
  }
  return {
    sum: query(store.versions[version].root),
    cover
  };
}
export function reachablePersistentNodes(store, versionIds) {
  const reached = new Set();
  function visit(id) {
    if (id === null || reached.has(id)) return;
    reached.add(id);
    const node = store.nodes[id];
    visit(node.left);
    visit(node.right);
  }
  for (const version of versionIds) {
    integer(version, 0, store.versions.length - 1, 'Version');
    visit(store.versions[version].root);
  }
  return reached;
}

// The layout groups physical objects by their owned interval. New objects on a
// changed path occupy another column within that interval, never duplicate IDs.
export function persistentLayout(store, visibleIds) {
  const byInterval = new Map();
  for (const id of [...visibleIds].sort((a, b) => a - b)) {
    const node = store.nodes[id];
    const key = `${node.low}:${node.high}`;
    if (!byInterval.has(key)) byInterval.set(key, []);
    byInterval.get(key).push(id);
  }
  const positions = new Map();
  let depthMax = 0;
  function place(low, high, depth) {
    if (low === high) return;
    depthMax = Math.max(depthMax, depth);
    const ids = byInterval.get(`${low}:${high}`) || [];
    ids.forEach((id, column) => positions.set(id, {
      x: 55 + (low + high) / 2 * 120 + column * 68,
      y: 120 + depth * 130
    }));
    if (high - low > 1) {
      const middle = Math.floor((low + high) / 2);
      place(low, middle, depth + 1);
      place(middle, high, depth + 1);
    }
  }
  place(0, store.size, 0);
  return {
    positions,
    width: Math.max(360, ...[...positions.values()].map(point => point.x + 85)),
    height: 185 + depthMax * 130
  };
}
export const HISTORY_WRITES = Object.freeze([Object.freeze([Object.freeze([0, 0]), Object.freeze([1, 5]), Object.freeze([3, 9])]), Object.freeze([Object.freeze([0, 0]), Object.freeze([2, 7])]), Object.freeze([Object.freeze([0, 0])])]);
export function historyLookup(history, snapshot) {
  integer(snapshot, 0, 4, 'Snapshot');
  let low = 0;
  let high = history.length;
  const trace = [];
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    const eligible = history[middle][0] <= snapshot;
    trace.push({
      low,
      high,
      middle,
      eligible
    });
    if (eligible) low = middle + 1;else high = middle;
  }
  return {
    position: low - 1,
    value: low ? history[low - 1][1] : 0,
    trace
  };
}
export function prefixRankModel(values, low, high, rank) {
  if (!Array.isArray(values) || !values.length || values.length > 8) throw new RangeError('Use 1–8 values.');
  values.forEach(value => integer(value, -99, 99, 'Each value'));
  integer(low, 0, values.length - 1, 'Left endpoint');
  integer(high, low + 1, values.length, 'Right endpoint');
  integer(rank, 1, high - low, 'Rank');
  const alphabet = [...new Set(values)].sort((a, b) => a - b);
  const earlier = alphabet.map(value => values.slice(0, low).filter(item => item === value).length);
  const later = alphabet.map(value => values.slice(0, high).filter(item => item === value).length);
  const counts = later.map((count, i) => count - earlier[i]);
  let begin = 0;
  let end = alphabet.length;
  let k = rank;
  const steps = [];
  while (end - begin > 1) {
    const middle = Math.floor((begin + end) / 2);
    const leftCount = counts.slice(begin, middle).reduce((sum, count) => sum + count, 0);
    const left = k <= leftCount;
    steps.push({
      begin,
      end,
      middle,
      k,
      leftCount,
      direction: left ? 'left' : 'right'
    });
    if (left) end = middle;else {
      begin = middle;
      k -= leftCount;
    }
  }
  steps.push({
    begin,
    end,
    k,
    value: alphabet[begin],
    direction: 'answer'
  });
  return {
    alphabet,
    earlier,
    later,
    counts,
    steps,
    answer: alphabet[begin]
  };
}
