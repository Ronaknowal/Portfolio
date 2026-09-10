/** Bounded educational DSU models. Parent arrows point toward a self-parent root. */
export const UNION_FIND_SIZE = 8;
export const BALANCED_UNIONS = [[0, 1], [2, 3], [0, 2], [4, 5], [6, 7], [4, 6], [0, 4]];
export const CHAIN_UNIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]];
function checkIndex(state, value) {
  if (!Number.isInteger(value) || value < 0 || value >= state.parent.length) {
    throw new RangeError(`Choose an integer from 0 to ${state.parent.length - 1}.`);
  }
}
export function createUnionFind(n = UNION_FIND_SIZE) {
  if (!Number.isInteger(n) || n < 0 || n > 25) throw new RangeError('Use 0–25 elements.');
  return freezeState({
    parent: Array.from({
      length: n
    }, (_, index) => index),
    size: Array(n).fill(1),
    count: n
  });
}
function freezeState(state) {
  return Object.freeze({
    parent: Object.freeze([...state.parent]),
    size: Object.freeze([...state.size]),
    count: state.count
  });
}
export function parentPath(state, node) {
  checkIndex(state, node);
  const path = [node];
  while (state.parent[node] !== node) {
    node = state.parent[node];
    if (!Number.isInteger(node) || node < 0 || node >= state.parent.length || path.includes(node)) {
      throw new Error('Parent pointers must form rooted trees.');
    }
    path.push(node);
  }
  return path;
}
export function representative(state, node) {
  return parentPath(state, node).at(-1);
}
export function unionFindGroups(state) {
  const groups = new Map();
  state.parent.forEach((_, node) => {
    const root = representative(state, node);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(node);
  });
  return [...groups].map(([root, members]) => ({
    root,
    members
  }));
}
function frame(state, message, focus = [], extra = {}) {
  return Object.freeze({
    state: freezeState(state),
    message,
    focus: Object.freeze([...focus]),
    ...extra
  });
}
export function traceFind(initial, node, compress = true) {
  checkIndex(initial, node);
  let state = {
    parent: [...initial.parent],
    size: [...initial.size],
    count: initial.count
  };
  const path = parentPath(initial, node);
  const root = path.at(-1);
  const frames = [frame(state, `Find the representative of ${node}.`, [node], {
    phase: 'start',
    path: [node]
  })];
  for (let position = 1; position < path.length; position++) {
    frames.push(frame(state, `Follow ${path[position - 1]} → ${path[position]}.`, path.slice(0, position + 1), {
      phase: 'follow',
      path: path.slice(0, position + 1)
    }));
  }
  frames.push(frame(state, `${root} points to itself: representative ${root}.`, path, {
    phase: 'root',
    path
  }));
  if (compress) {
    for (const item of path.slice(0, -1)) {
      if (state.parent[item] !== root) {
        const previous = state.parent[item];
        state.parent[item] = root;
        frames.push(frame(state, `Rewrite parent[${item}]: ${previous} → ${root}. Membership is unchanged.`, [item, root], {
          phase: 'rewrite',
          path
        }));
      }
    }
  }
  frames.push(frame(state, `Return ${root}. ${path.length - 1} upward hops on this lookup.`, [root], {
    phase: 'done',
    path
  }));
  return {
    state: freezeState(state),
    frames,
    root,
    path,
    hops: path.length - 1
  };
}
export function traceUnion(initial, left, right, policy = 'size', compress = false) {
  checkIndex(initial, left);
  checkIndex(initial, right);
  if (!['size', 'unweighted'].includes(policy)) throw new Error('Unknown union policy.');
  const first = traceFind(initial, left, compress);
  const second = traceFind(first.state, right, compress);
  const frames = [...first.frames, ...second.frames];
  let state = {
    parent: [...second.state.parent],
    size: [...second.state.size],
    count: second.state.count
  };
  let a = first.root;
  let b = second.root;
  if (a === b) {
    frames.push(frame(state, `${left} and ${right} already share root ${a}. No merge; count stays ${state.count}.`, [a]));
    return {
      state: freezeState(state),
      frames,
      merged: false
    };
  }
  if (policy === 'size') {
    if (state.size[a] < state.size[b]) [a, b] = [b, a];
    frames.push(frame(state, `Root ${a} has ${state.size[a]} members; root ${b} has ${state.size[b]}. Attach ${b} below ${a}.`, [a, b]));
    state.parent[b] = a;
    state.size[a] += state.size[b];
  } else {
    frames.push(frame(state, `Unweighted rule: attach the first root ${a} below the second root ${b}.`, [a, b]));
    state.parent[a] = b;
    state.size[b] += state.size[a];
  }
  state.count--;
  frames.push(frame(state, `Merged whole components. ${state.count} components remain.`, [a, b]));
  return {
    state: freezeState(state),
    frames,
    merged: true
  };
}
export function buildUnionFind(pairs = BALANCED_UNIONS, policy = 'size', n = UNION_FIND_SIZE) {
  let state = createUnionFind(n);
  for (const [a, b] of pairs) state = traceUnion(state, a, b, policy).state;
  return state;
}

/** Forest coordinates: children below parents; wide states scroll at native label size. */
export function forestLayout(state) {
  const children = state.parent.map(() => []);
  const roots = [];
  state.parent.forEach((parent, node) => {
    if (parent === node) roots.push(node);else children[parent].push(node);
  });
  const positions = {};
  let nextLeaf = 0;
  let maximumDepth = 0;
  function place(node, depth) {
    maximumDepth = Math.max(maximumDepth, depth);
    const childPositions = children[node].map(child => place(child, depth + 1));
    const x = childPositions.length ? (childPositions[0] + childPositions.at(-1)) / 2 : 38 + nextLeaf++ * 62;
    positions[node] = {
      x,
      y: 45 + depth * 78,
      depth
    };
    return x;
  }
  roots.forEach(root => {
    place(root, 0);
    nextLeaf += 0.45;
  });
  return {
    positions,
    width: Math.max(280, 38 + nextLeaf * 62),
    height: 96 + maximumDepth * 78
  };
}
export function createIslandGrid() {
  return Object.freeze({
    dsu: createUnionFind(25),
    active: Object.freeze(Array(25).fill(false)),
    count: 0
  });
}
export function activateIslandCell(initial, index) {
  checkIndex(initial.dsu, index);
  if (initial.active[index]) return {
    state: initial,
    mergedRoots: [],
    message: `Cell (${Math.floor(index / 5)}, ${index % 5}) is already open. Count stays ${initial.count}.`
  };
  const active = [...initial.active];
  active[index] = true;
  let dsu = initial.dsu;
  let count = initial.count + 1;
  const row = Math.floor(index / 5);
  const column = index % 5;
  const mergedRoots = [];
  const neighbors = [[row - 1, column], [row, column - 1], [row, column + 1], [row + 1, column]];
  for (const [r, c] of neighbors) {
    if (r < 0 || r >= 5 || c < 0 || c >= 5 || !active[r * 5 + c]) continue;
    const neighbor = r * 5 + c;
    const oldRoot = representative(dsu, neighbor);
    const result = traceUnion(dsu, index, neighbor, 'size', true);
    dsu = result.state;
    if (result.merged) {
      count--;
      mergedRoots.push(oldRoot);
    }
  }
  const state = Object.freeze({
    dsu,
    active: Object.freeze(active),
    count
  });
  return {
    state,
    mergedRoots,
    message: `Opened (${row}, ${column}): ${initial.count} + 1 new cell − ${mergedRoots.length} successful joins = ${count} ${count === 1 ? 'island' : 'islands'}.`
  };
}
export function activeIslandGroups(state) {
  return unionFindGroups(state.dsu).map(group => ({
    ...group,
    members: group.members.filter(node => state.active[node])
  })).filter(group => group.members.length);
}
