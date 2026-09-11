function snapshot(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(snapshot));
  if (value && typeof value === 'object') {
    return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, snapshot(item)])));
  }
  return value;
}
function integers(values, minimumLength, maximumLength, minimum, maximum, message) {
  if (!Array.isArray(values) || values.length < minimumLength || values.length > maximumLength) throw new Error(message);
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index) || !Number.isSafeInteger(values[index]) || values[index] < minimum || values[index] > maximum) throw new Error(message);
  }
}
export const defaultChainDimensions = Object.freeze([8, 2, 12, 3, 6]);
export function matrixChainPlan(dimensions = defaultChainDimensions) {
  integers(dimensions, 2, 7, 1, 20, 'Use 2–7 positive integer dimensions, each from 1 through 20.');
  const count = dimensions.length - 1;
  const costs = Array.from({
    length: count
  }, () => Array(count + 1).fill(null));
  const splits = Array.from({
    length: count
  }, () => Array(count + 1).fill(null));
  const cells = [];
  for (let length = 1; length <= count; length += 1) {
    for (let left = 0; left + length <= count; left += 1) {
      const right = left + length;
      const candidates = [];
      if (length === 1) costs[left][right] = 0;
      for (let split = left + 1; split < right; split += 1) {
        const first = costs[left][split];
        const second = costs[split][right];
        const merge = dimensions[left] * dimensions[split] * dimensions[right];
        const total = first + second + merge;
        candidates.push({
          split,
          first,
          second,
          merge,
          total
        });
        if (costs[left][right] === null || total < costs[left][right]) {
          costs[left][right] = total;
          splits[left][right] = split;
        }
      }
      cells.push({
        left,
        right,
        length,
        cost: costs[left][right],
        split: splits[left][right],
        candidates
      });
    }
  }
  const operations = [];
  const tokens = [];
  function recover(left, right) {
    if (right === left + 1) {
      tokens.push(`A${left}`);
      return {
        left,
        right,
        split: null,
        rows: dimensions[left],
        columns: dimensions[right]
      };
    }
    const split = splits[left][right];
    tokens.push('(');
    const first = recover(left, split);
    tokens.push(' × ');
    const second = recover(split, right);
    tokens.push(')');
    operations.push({
      left,
      split,
      right,
      cost: dimensions[left] * dimensions[split] * dimensions[right]
    });
    return {
      left,
      right,
      split,
      rows: dimensions[left],
      columns: dimensions[right],
      first,
      second
    };
  }
  const tree = recover(0, count);
  return snapshot({
    dimensions,
    count,
    costs,
    splits,
    cells,
    tree,
    operations,
    expression: tokens.join(''),
    result: costs[0][count]
  });
}
export function balloonPlan(values = [2, 4, 3]) {
  integers(values, 0, 8, 0, 20, 'Use at most eight integer balloon values from 0 through 20.');
  const padded = [1, ...values, 1];
  const count = padded.length;
  const costs = Array.from({
    length: count
  }, () => Array(count).fill(0));
  const splits = Array.from({
    length: count
  }, () => Array(count).fill(null));
  for (let gap = 2; gap < count; gap += 1) {
    for (let left = 0; left + gap < count; left += 1) {
      const right = left + gap;
      let best = null;
      for (let last = left + 1; last < right; last += 1) {
        const candidate = costs[left][last] + costs[last][right] + padded[left] * padded[last] * padded[right];
        if (best === null || candidate > best) {
          best = candidate;
          splits[left][right] = last;
        }
      }
      costs[left][right] = best;
    }
  }
  const order = [];
  function recover(left, right) {
    const last = splits[left][right];
    if (last === null) return;
    recover(left, last);
    recover(last, right);
    order.push(last - 1);
  }
  recover(0, count - 1);
  const live = values.map((value, id) => ({
    value,
    id
  }));
  const replay = [{
    live: live.slice(),
    removed: null,
    earned: 0,
    total: 0
  }];
  let total = 0;
  for (const removed of order) {
    const position = live.findIndex(balloon => balloon.id === removed);
    const first = position ? live[position - 1].value : 1;
    const second = position + 1 < live.length ? live[position + 1].value : 1;
    const earned = first * values[removed] * second;
    total += earned;
    live.splice(position, 1);
    replay.push({
      live: live.slice(),
      removed,
      first,
      value: values[removed],
      second,
      earned,
      total
    });
  }
  return snapshot({
    values,
    padded,
    costs,
    splits,
    order,
    replay,
    result: costs[0][count - 1]
  });
}
export const defaultTreeWeights = Object.freeze([5, 9, 2, 4, 1, 6, 3]);
export const defaultTreeEdges = Object.freeze([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]].map(Object.freeze));
export function treePresetEdges(count, shape = 'binary') {
  if (!Number.isInteger(count) || count < 0 || count > 9 || !['binary', 'chain', 'star'].includes(shape)) throw new Error('Choose a binary, chain or star tree with at most nine nodes.');
  return Array.from({
    length: Math.max(0, count - 1)
  }, (_, index) => {
    const child = index + 1;
    const parent = shape === 'binary' ? Math.floor(index / 2) : shape === 'chain' ? index : 0;
    return [parent, child];
  });
}
export function treeBoundaryPlan(weights = defaultTreeWeights, edges = defaultTreeEdges, root = 0) {
  integers(weights, 0, 9, -9, 20, 'Use at most nine integer weights from −9 through 20.');
  const count = weights.length;
  if (!Array.isArray(edges) || edges.length !== Math.max(0, count - 1)) throw new Error('A simple tree needs exactly n−1 edges (zero for an empty tree).');
  if (!Number.isInteger(root) || root < 0 || root >= Math.max(1, count)) throw new Error('Choose an existing root (0 for an empty tree).');
  const adjacency = Array.from({
    length: count
  }, () => []);
  const seenEdges = new Set();
  for (let index = 0; index < edges.length; index += 1) {
    const edge = edges[index];
    integers(edge, 2, 2, 0, count - 1, 'Every edge must contain two existing node IDs.');
    const [first, second] = edge;
    const key = `${Math.min(first, second)},${Math.max(first, second)}`;
    if (first === second || seenEdges.has(key)) throw new Error('Self-loops and duplicate edges are not a simple tree.');
    seenEdges.add(key);
    adjacency[first].push(second);
    adjacency[second].push(first);
  }
  const children = Array.from({
    length: count
  }, () => []);
  const parents = Array(count).fill(null);
  const depth = Array(count).fill(0);
  const order = [];
  const seen = new Set();
  const stack = count ? [root] : [];
  while (stack.length) {
    const node = stack.pop();
    if (seen.has(node)) throw new Error('The input contains a cycle.');
    seen.add(node);
    order.push(node);
    for (const next of adjacency[node]) {
      if (next === parents[node]) continue;
      if (seen.has(next)) throw new Error('The input contains a cycle.');
      parents[next] = node;
      depth[next] = depth[node] + 1;
      children[node].push(next);
      stack.push(next);
    }
  }
  if (seen.size !== count) throw new Error('Every node must belong to one connected tree.');
  const postorder = order.slice().reverse();
  const free = Array(count).fill(0);
  const blocked = Array(count).fill(0);
  const take = Array(count).fill(0);
  for (const node of postorder) {
    blocked[node] = children[node].reduce((sum, child) => sum + free[child], 0);
    take[node] = weights[node] + children[node].reduce((sum, child) => sum + blocked[child], 0);
    free[node] = Math.max(blocked[node], take[node]);
  }
  return snapshot({
    weights,
    edges,
    root,
    count,
    parents,
    children,
    depth,
    postorder,
    free,
    blocked,
    take,
    result: count ? free[root] : 0
  });
}
export function treeBoundaryWitness(plan, node = plan.root, parentSelected = false) {
  if (typeof parentSelected !== 'boolean') throw new Error('Parent-selected must be a Boolean.');
  if (!Number.isInteger(node) || node < 0 || node >= Math.max(1, plan.count)) throw new Error('Choose an existing subtree root.');
  const selected = [];
  const requests = [];
  const stack = plan.count ? [[node, parentSelected]] : [];
  while (stack.length) {
    const [current, blocked] = stack.pop();
    const choose = !blocked && plan.take[current] > plan.blocked[current];
    requests.push({
      node: current,
      parentSelected: blocked,
      selected: choose
    });
    if (choose) selected.push(current);
    for (const child of plan.children[current].slice().reverse()) stack.push([child, choose]);
  }
  return snapshot({
    selected,
    requests,
    result: plan.count ? parentSelected ? plan.blocked[node] : plan.free[node] : 0
  });
}
export function createDigitCounter(bound = 213) {
  if (!Number.isSafeInteger(bound) || bound < 0 || bound > 999999) throw new Error('Use an integer bound from 0 through 999999.');
  const digits = String(bound).split('').map(Number);
  const cache = new Map();
  function nextState(state, digit) {
    const next = {
      position: state.position + 1,
      tight: state.tight && digit === digits[state.position],
      started: state.started || digit !== 0,
      used: state.used
    };
    if (next.started) next.used |= 1 << digit;
    return next;
  }
  function reason(state, digit) {
    if (state.tight && digit > digits[state.position]) return 'above bound';
    if ((state.started || digit !== 0) && state.used & 1 << digit) return 'digit already used';
    return null;
  }
  function count(state) {
    if (state.position === digits.length) return Number(state.started);
    const key = `${state.position},${Number(state.tight)},${Number(state.started)},${state.used}`;
    if (cache.has(key)) return cache.get(key);
    let total = 0;
    for (let digit = 0; digit <= 9; digit += 1) {
      if (reason(state, digit) === null) total += count(nextState(state, digit));
    }
    cache.set(key, total);
    return total;
  }
  const initial = {
    position: 0,
    tight: true,
    started: false,
    used: 0
  };
  const total = count(initial);
  function inspect(prefix = '') {
    if (typeof prefix !== 'string' || !/^\d*$/.test(prefix) || prefix.length > digits.length) throw new Error('Use a decimal prefix no longer than the bound.');
    let state = initial;
    const history = [{
      ...state,
      prefix: ''
    }];
    for (let position = 0; position < prefix.length; position += 1) {
      const digit = Number(prefix[position]);
      const invalid = reason(state, digit);
      if (invalid) throw new Error(`Prefix rejected: ${invalid}.`);
      state = nextState(state, digit);
      history.push({
        ...state,
        prefix: prefix.slice(0, position + 1)
      });
    }
    const complete = state.position === digits.length;
    const branches = complete ? [] : Array.from({
      length: 10
    }, (_, digit) => {
      const invalid = reason(state, digit);
      const next = invalid ? null : nextState(state, digit);
      return {
        digit,
        allowed: !invalid,
        reason: invalid,
        padding: !state.started && digit === 0,
        count: next ? count(next) : 0,
        next
      };
    });
    let completion = null;
    if (count(state) > 0) {
      let cursor = state;
      let spelling = prefix;
      while (cursor.position < digits.length) {
        for (let digit = 0; digit <= 9; digit += 1) {
          if (reason(cursor, digit) !== null) continue;
          const next = nextState(cursor, digit);
          if (count(next) === 0) continue;
          spelling += digit;
          cursor = next;
          break;
        }
      }
      completion = Number(spelling);
    }
    return snapshot({
      bound,
      digits,
      prefix,
      state,
      history,
      branches,
      complete,
      total,
      remaining: count(state),
      completion,
      cachedStates: cache.size
    });
  }
  return Object.freeze({
    bound,
    total,
    inspect
  });
}
