// Exact, bounded teaching models. Trace copies and geometric inspection are
// deliberately separate from the native algorithm's time/space contract.
function snapshot(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(snapshot));
  if (value && typeof value === 'object') {
    return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, snapshot(item)])));
  }
  return value;
}
export const rangeReadings = Object.freeze([2, 1, 3, 4, 0, 5, 2, 1]);
export function parseRangeValues(text) {
  if (!text.trim()) return [];
  const parts = text.trim().split(/[\s,]+/);
  if (parts.some(part => !/^-?\d+$/.test(part))) throw new Error('Use comma-separated whole numbers.');
  const values = parts.map(Number);
  validateValues(values);
  return values;
}
function validateValues(values) {
  if (!Array.isArray(values) || values.length > 8 || values.some(value => !Number.isInteger(value) || Math.abs(value) > 99)) {
    throw new Error('Use at most eight integers from −99 through 99.');
  }
}
function validateRange(length, left, right) {
  if (!Number.isInteger(left) || !Number.isInteger(right) || left < 0 || right < left || right > length) {
    throw new Error(`Use whole-number boundaries 0 ≤ left ≤ right ≤ ${length}.`);
  }
}
function validatePosition(length, index) {
  if (!Number.isInteger(index) || index < 0 || index >= length) throw new Error('Choose an existing array index.');
}
function aggregate(operation) {
  if (operation === 'sum') return {
    identity: 0,
    combine: (left, right) => left + right
  };
  if (operation === 'min') return {
    identity: Infinity,
    combine: Math.min
  };
  if (operation === 'max') return {
    identity: -Infinity,
    combine: Math.max
  };
  throw new Error('Choose sum, min or max.');
}
export function displayRangeValue(value) {
  return value === Infinity ? '+∞' : value === -Infinity ? '−∞' : String(value);
}
function paddedSize(length) {
  let size = 1;
  while (size < length) size *= 2;
  return size;
}
export function segmentState(values = [2, 1, 3, 4], operation = 'sum') {
  validateValues(values);
  const {
    identity,
    combine
  } = aggregate(operation);
  const size = paddedSize(values.length);
  const tree = Array(size * 2).fill(identity);
  values.forEach((value, index) => tree[size + index] = value);
  for (let node = size - 1; node > 0; node--) tree[node] = combine(tree[node * 2], tree[node * 2 + 1]);
  return snapshot({
    values,
    operation,
    size,
    tree
  });
}
export function segmentGeometry(state) {
  const nodes = [];
  const spacing = state.size <= 4 ? 60 : 72;
  const width = Math.max(104, state.size * spacing + 12);
  const offset = (width - state.size * spacing) / 2;
  function visit(id, left, right, depth, parent) {
    nodes.push({
      id,
      left,
      right,
      depth,
      parent,
      x: offset + (left + right) * spacing / 2,
      y: 16 + depth * 85,
      value: state.tree[id],
      padding: left >= state.values.length
    });
    if (right - left > 1) {
      const middle = (left + right) / 2;
      visit(id * 2, left, middle, depth + 1, id);
      visit(id * 2 + 1, middle, right, depth + 1, id);
    }
  }
  visit(1, 0, state.size, 0, null);
  return {
    nodes,
    width,
    height: Math.log2(state.size) * 85 + 88
  };
}
export function segmentQuery(state, left, right) {
  validateRange(state.values.length, left, right);
  const {
    identity,
    combine
  } = aggregate(state.operation);
  let low = state.size + left;
  let high = state.size + right;
  let leftValue = identity;
  let rightValue = identity;
  const leftNodes = [];
  const rightNodes = [];
  const frames = [];
  function emit(phase, current, message) {
    frames.push(snapshot({
      state,
      phase,
      current,
      low,
      high,
      leftValue,
      rightValue,
      leftNodes,
      rightNodes,
      message
    }));
  }
  emit('start', null, `Query [${left},${right}). Both accumulators begin at the ${state.operation} identity.`);
  while (low < high) {
    if (low % 2 === 1) {
      const current = low;
      leftValue = combine(leftValue, state.tree[current]);
      leftNodes.push(current);
      low++;
      emit('take-left', current, `Take node ${current} at the left boundary; append its interval to the left accumulator.`);
    }
    if (high % 2 === 1) {
      const current = --high;
      rightValue = combine(state.tree[current], rightValue);
      rightNodes.unshift(current);
      emit('take-right', current, `Take node ${current} before the right boundary; prepend its interval to the right accumulator.`);
    }
    low = Math.floor(low / 2);
    high = Math.floor(high / 2);
    emit('ascend', null, 'The remaining boundaries move to their parents. Already taken intervals are excluded.');
  }
  const result = combine(leftValue, rightValue);
  emit('done', null, `Combine left then right: ${displayRangeValue(result)}. The selected intervals cover the query exactly once.`);
  return snapshot({
    kind: 'query',
    left,
    right,
    result,
    state,
    frames
  });
}
export function segmentAssign(state, index, value) {
  validatePosition(state.values.length, index);
  if (!Number.isInteger(value) || Math.abs(value) > 99) throw new Error('Assign an integer from −99 through 99.');
  const values = [...state.values];
  const tree = [...state.tree];
  const {
    combine
  } = aggregate(state.operation);
  const frames = [];
  const currentState = () => ({
    ...state,
    values,
    tree
  });
  function emit(phase, current, message) {
    frames.push(snapshot({
      state: currentState(),
      phase,
      current,
      leftNodes: [],
      rightNodes: [],
      message
    }));
  }
  emit('start', null, `Assign array index ${index} to ${value}; this is replacement, not an increment.`);
  values[index] = value;
  let node = state.size + index;
  tree[node] = value;
  emit('write-leaf', node, `Leaf ${node} now holds ${value}. Its ancestors are being repaired before the operation completes.`);
  while (node > 1) {
    node = Math.floor(node / 2);
    tree[node] = combine(tree[node * 2], tree[node * 2 + 1]);
    emit('pull', node, `Recompute node ${node} from its left and right child summaries.`);
  }
  emit('done', null, 'Assignment complete. All segment summaries are current.');
  return snapshot({
    kind: 'assign',
    index,
    value,
    state: currentState(),
    frames
  });
}
export function fenwickState(values = rangeReadings) {
  validateValues(values);
  const tree = [0, ...values];
  for (let index = 1; index <= values.length; index++) {
    const parent = index + (index & -index);
    if (parent <= values.length) tree[parent] += tree[index];
  }
  return snapshot({
    values,
    tree
  });
}
export function fenwickBlocks(state) {
  return state.tree.slice(1).map((value, index) => {
    const internal = index + 1;
    const lowbit = internal & -internal;
    return {
      internal,
      lowbit,
      left: internal - lowbit,
      right: internal,
      value,
      binary: internal.toString(2).padStart(4, '0')
    };
  });
}
export function fenwickPrefix(state, end) {
  validateRange(state.values.length, 0, end);
  const frames = [];
  const visited = [];
  let cursor = end;
  let result = 0;
  frames.push(snapshot({
    state,
    cursor,
    current: null,
    visited,
    result,
    message: `Sum [0,${end}). A prefix length is also the first internal query index.`
  }));
  while (cursor > 0) {
    const current = cursor;
    const lowbit = cursor & -cursor;
    result += state.tree[cursor];
    visited.push(cursor);
    cursor -= lowbit;
    frames.push(snapshot({
      state,
      cursor,
      current,
      visited,
      result,
      message: `Add block [${current - lowbit},${current}) with sum ${state.tree[current]}. Subtract lowbit ${lowbit}: next internal index ${cursor}.`
    }));
  }
  frames.push(snapshot({
    state,
    cursor,
    current: null,
    visited,
    result,
    message: `Reached zero: prefix sum is ${result}. Visited blocks partition [0,${end}).`
  }));
  return snapshot({
    kind: 'prefix',
    end,
    result,
    state,
    frames
  });
}
export function fenwickAdd(state, index, delta) {
  validatePosition(state.values.length, index);
  if (!Number.isInteger(delta) || Math.abs(delta) > 20 || Math.abs(state.values[index] + delta) > 99) throw new Error('Use an integer delta from −20 through 20 that keeps the value in −99…99.');
  const values = [...state.values];
  const tree = [...state.tree];
  const visited = [];
  const frames = [];
  let cursor = index + 1;
  frames.push(snapshot({
    state,
    cursor,
    current: null,
    visited,
    message: `Add ${delta} at external index ${index}; its internal index is ${cursor}.`
  }));
  values[index] += delta;
  while (cursor <= values.length) {
    const current = cursor;
    tree[cursor] += delta;
    visited.push(cursor);
    cursor += cursor & -cursor;
    frames.push(snapshot({
      state: {
        values,
        tree
      },
      cursor,
      current,
      visited,
      message: `Block ${current} contains the changed element: add ${delta}. Add lowbit to visit internal index ${cursor} next. Other ancestors await their writes.`
    }));
  }
  frames.push(snapshot({
    state: {
      values,
      tree
    },
    cursor,
    current: null,
    visited,
    message: 'The next index is beyond the tree. Every containing block is updated.'
  }));
  return snapshot({
    kind: 'add',
    index,
    delta,
    state: {
      values,
      tree
    },
    frames
  });
}
export function lazyState(values = [2, 1, 3, 4]) {
  const ordinary = segmentState(values, 'sum');
  return snapshot({
    values,
    size: ordinary.size,
    tree: ordinary.tree,
    multipliers: Array(ordinary.size * 2).fill(1),
    additions: Array(ordinary.size * 2).fill(0)
  });
}
export function lazyEffectiveValues(state) {
  const values = [];
  function visit(node, left, right, multiplier, addition) {
    if (right - left === 1) {
      if (left < state.values.length) values[left] = multiplier * state.tree[node] + addition;
      return;
    }
    const nextMultiplier = multiplier * state.multipliers[node];
    const nextAddition = multiplier * state.additions[node] + addition;
    const middle = (left + right) / 2;
    visit(node * 2, left, middle, nextMultiplier, nextAddition);
    visit(node * 2 + 1, middle, right, nextMultiplier, nextAddition);
  }
  visit(1, 0, state.size, 1, 0);
  return values;
}
export function lazyRangeOperation(original, kind, left, right, amount = 0) {
  validateRange(original.values.length, left, right);
  if (!['query', 'add', 'set'].includes(kind)) throw new Error('Choose range query, add or set.');
  if (kind !== 'query' && (!Number.isInteger(amount) || Math.abs(amount) > 20)) throw new Error('Use an integer range value from −20 through 20.');
  const previousValues = lazyEffectiveValues(original);
  if (kind === 'add' && previousValues.slice(left, right).some(value => Math.abs(value + amount) > 99)) throw new Error('This update would leave the teaching bounds −99…99.');
  const tree = [...original.tree];
  const multipliers = [...original.multipliers];
  const additions = [...original.additions];
  const dirty = [];
  const taken = [];
  const frames = [];
  let result = 0;
  function state() {
    return {
      ...original,
      tree,
      multipliers,
      additions
    };
  }
  function emit(phase, current, message) {
    frames.push(snapshot({
      state: state(),
      effective: lazyEffectiveValues(state()),
      phase,
      current,
      dirty,
      taken,
      result,
      message
    }));
  }
  function apply(node, low, high, multiplier, addition) {
    tree[node] = multiplier * tree[node] + addition * (high - low);
    if (high - low > 1) {
      multipliers[node] = multiplier * multipliers[node];
      additions[node] = multiplier * additions[node] + addition;
    }
  }
  function push(node, low, high) {
    if (high - low <= 1 || multipliers[node] === 1 && additions[node] === 0) return;
    const middle = (low + high) / 2;
    const multiplier = multipliers[node];
    const addition = additions[node];
    apply(node * 2, low, middle, multiplier, addition);
    apply(node * 2 + 1, middle, high, multiplier, addition);
    multipliers[node] = 1;
    additions[node] = 0;
    emit('push', node, `Push map x→${multiplier}x+${addition} from node ${node} to both children, then clear the parent tag. Logical values do not change.`);
  }
  function visit(node, low, high) {
    if (right <= low || high <= left || left === right) {
      emit('outside', node, `Node ${node}, [${low},${high}), contributes nothing: it is outside the requested range.`);
      return;
    }
    if (left <= low && high <= right) {
      if (kind === 'query') {
        result += tree[node];
        taken.push(node);
        emit('consume', node, `Node ${node} is fully covered. Its stored sum ${tree[node]} already includes its pending tag; do not add that tag again.`);
      } else {
        apply(node, low, high, kind === 'set' ? 0 : 1, amount);
        const explanation = high - low > 1
          ? 'Update its sum now and compose a tag for children; no descent is needed.'
          : 'Update this leaf’s sum directly. It has no children and needs no pending tag.';
        emit('apply', node, `${kind === 'set' ? 'Assign' : 'Add'} ${amount} on [${low},${high}). ${explanation}`);
      }
      return;
    }
    push(node, low, high);
    if (kind !== 'query') dirty.push(node);
    const middle = (low + high) / 2;
    visit(node * 2, low, middle);
    visit(node * 2 + 1, middle, high);
    if (kind !== 'query') {
      tree[node] = tree[node * 2] + tree[node * 2 + 1];
      dirty.pop();
      emit('pull', node, `Recompute node ${node} from its now-current children. Ancestors marked for repair are completed on return.`);
    }
  }
  emit('start', null, `${kind} on [${left},${right})${kind === 'query' ? '' : ` with ${amount}`}. Pending tags are already included in their node's stored sum.`);
  visit(1, 0, original.size);
  const finalValues = lazyEffectiveValues(state());
  const finalState = {
    ...state(),
    values: finalValues
  };
  emit('done', null, kind === 'query' ? `Query complete: sum ${result}. Pushing changed storage ownership, not array values.` : 'Update complete. Stored parent sums are correct; tagged children may still await propagation.');
  return snapshot({
    kind,
    left,
    right,
    amount,
    result: kind === 'query' ? result : null,
    state: finalState,
    frames
  });
}
export function sparseMinimum(values, left, right) {
  validateValues(values);
  validateRange(values.length, left, right);
  if (left === right) throw new Error('A sparse minimum query needs a nonempty interval.');
  const levels = [[...values]];
  for (let length = 2; length <= values.length; length *= 2) {
    const previous = levels.at(-1);
    const half = length / 2;
    levels.push(Array.from({
      length: values.length - length + 1
    }, (_, index) => Math.min(previous[index], previous[index + half])));
  }
  const power = Math.floor(Math.log2(right - left));
  const length = 2 ** power;
  const blocks = [[left, left + length], [right - length, right]];
  return snapshot({
    values,
    levels,
    left,
    right,
    power,
    length,
    blocks,
    result: Math.min(levels[power][left], levels[power][right - length])
  });
}
export function weightedIncreasingPlan(values = [3, 1, 2, 2, 4], weights = [4, 2, 5, 20, 3]) {
  validateValues(values);
  validateValues(weights);
  if (values.length !== weights.length) throw new Error('Each input value needs one weight.');
  const coordinates = [...new Set(values)].sort((a, b) => a - b);
  const size = paddedSize(coordinates.length);
  const empty = {
    score: 0,
    endpoint: null
  };
  const tree = Array.from({
    length: size * 2
  }, () => ({
    ...empty
  }));
  const parents = Array(values.length).fill(null);
  function better(first, second) {
    if (first.score !== second.score) return first.score > second.score ? first : second;
    if (first.endpoint === null) return first;
    if (second.endpoint === null) return second;
    return first.endpoint <= second.endpoint ? first : second;
  }
  function prefix(end) {
    let low = size;
    let high = size + end;
    let best = empty;
    while (low < high) {
      if (low % 2) best = better(best, tree[low++]);
      if (high % 2) best = better(best, tree[--high]);
      low = Math.floor(low / 2);
      high = Math.floor(high / 2);
    }
    return best;
  }
  const frames = [];
  for (let index = 0; index < values.length; index++) {
    const rank = coordinates.indexOf(values[index]);
    const prior = prefix(rank);
    parents[index] = prior.endpoint;
    const candidate = {
      score: prior.score + weights[index],
      endpoint: index
    };
    const previous = tree[size + rank];
    let node = size + rank;
    tree[node] = better(tree[node], candidate);
    while (node > 1) {
      node = Math.floor(node / 2);
      tree[node] = better(tree[node * 2], tree[node * 2 + 1]);
    }
    frames.push(snapshot({
      index,
      value: values[index],
      weight: weights[index],
      rank,
      prior,
      candidate,
      previous,
      stored: tree[size + rank],
      byRank: tree.slice(size, size + coordinates.length),
      best: tree[1]
    }));
  }
  const witness = [];
  let endpoint = tree[1].endpoint;
  while (endpoint !== null) {
    witness.push(endpoint);
    endpoint = parents[endpoint];
  }
  witness.reverse();
  return snapshot({
    values,
    weights,
    coordinates,
    frames,
    parents,
    score: tree[1].score,
    witness
  });
}
