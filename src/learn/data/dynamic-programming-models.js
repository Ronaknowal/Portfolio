function freezeSnapshot(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(freezeSnapshot));
  if (value && typeof value === 'object') {
    return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, freezeSnapshot(item)])));
  }
  return value;
}
export function rewardTrace(rewards = [4, 7, 2, 9], method = 'memo') {
  if (!['memo', 'table'].includes(method)) throw new Error('Choose memo or table.');
  if (rewards.length > 7 || rewards.some(value => !Number.isInteger(value) || value < -9 || value > 20)) {
    throw new Error('Use at most seven integers from −9 through 20. Empty input is allowed.');
  }
  const count = rewards.length;
  const cache = Array(count + 2).fill(null);
  const stack = [];
  const frames = [];
  let calls = 0;
  let hits = 0;
  function emit(index, phase, message, candidates = null) {
    frames.push(freezeSnapshot({
      index,
      phase,
      message,
      candidates,
      cache,
      stack,
      calls,
      hits
    }));
  }
  emit(null, 'start', 'No result has been evaluated. Blank cells mean unknown, not zero.');
  function solve(index) {
    calls += 1;
    stack.push(index);
    if (cache[index] !== null) {
      hits += 1;
      emit(index, 'hit', `Reuse F(${index}) = ${cache[index]}; do not expand its choices again.`);
      stack.pop();
      return cache[index];
    }
    emit(index, 'enter', `Ask for the best reward in the free suffix beginning at ${index}.`);
    if (index >= count) {
      cache[index] = 0;
      emit(index, 'base', 'There are no sessions left; choosing nothing earns zero.');
    } else {
      const skip = solve(index + 1);
      const take = rewards[index] + solve(index + 2);
      cache[index] = Math.max(skip, take);
      emit(index, 'write', `F(${index}) = max(skip ${skip}, take ${take}) = ${cache[index]}.`, {
        skip,
        take
      });
    }
    stack.pop();
    return cache[index];
  }
  if (method === 'memo') solve(0);else {
    cache[count] = 0;
    cache[count + 1] = 0;
    emit(count, 'base', `Initialize the two empty suffixes F(${count}) and F(${count + 1}) to zero.`);
    for (let index = count - 1; index >= 0; index -= 1) {
      const skip = cache[index + 1];
      const take = rewards[index] + cache[index + 2];
      cache[index] = Math.max(skip, take);
      emit(index, 'write', `Both dependencies are ready: max(${skip}, ${rewards[index]} + ${cache[index + 2]}) = ${cache[index]}.`, {
        skip,
        take
      });
    }
  }
  emit(0, 'done', `Answer F(0) = ${cache[0]}. ${method === 'memo' ? `${calls} requests, including ${hits} cache hits.` : 'Each non-base table cell was evaluated once.'}`);
  return freezeSnapshot({
    rewards,
    method,
    frames,
    result: cache[0]
  });
}
export const defaultGrid = Object.freeze([Object.freeze([1, 6, 2, 1]), Object.freeze([2, 1, 5, 2]), Object.freeze([4, 1, 1, 1])]);
export function gridPlan(grid = defaultGrid, blocked = []) {
  if (!grid.length || !grid[0].length || grid.length > 5 || grid[0].length > 5 || grid.some(row => row.length !== grid[0].length || row.some(cost => !Number.isInteger(cost) || Math.abs(cost) > 20))) {
    throw new Error('Use a rectangular 1–5 by 1–5 grid of integer costs from −20 through 20.');
  }
  const rows = grid.length;
  const columns = grid[0].length;
  if (blocked.some(cell => !/^\d+,\d+$/.test(cell) || Number(cell.split(',')[0]) >= rows || Number(cell.split(',')[1]) >= columns)) {
    throw new Error('Blocked cells must name valid row,column coordinates.');
  }
  const forbidden = new Set(blocked);
  const costs = Array.from({
    length: rows
  }, () => Array(columns).fill(null));
  const parents = Array.from({
    length: rows
  }, () => Array(columns).fill(null));
  const ways = Array.from({
    length: rows
  }, () => Array(columns).fill(0));
  const steps = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const candidates = [];
      if (!forbidden.has(`${row},${column}`)) {
        if (row === 0 && column === 0) {
          costs[row][column] = grid[row][column];
          ways[row][column] = 1;
        } else {
          for (const [priorRow, priorColumn] of [[row - 1, column], [row, column - 1]]) {
            if (priorRow >= 0 && priorColumn >= 0 && costs[priorRow][priorColumn] !== null) {
              candidates.push({
                row: priorRow,
                column: priorColumn,
                cost: costs[priorRow][priorColumn]
              });
              ways[row][column] += ways[priorRow][priorColumn];
            }
          }
          if (candidates.length) {
            const chosen = candidates.reduce((best, candidate) => candidate.cost < best.cost ? candidate : best);
            costs[row][column] = chosen.cost + grid[row][column];
            parents[row][column] = [chosen.row, chosen.column];
          }
        }
      }
      steps.push(freezeSnapshot({
        row,
        column,
        candidates,
        costs,
        ways,
        parents
      }));
    }
  }
  const path = [];
  let cursor = costs[rows - 1][columns - 1] === null ? null : [rows - 1, columns - 1];
  while (cursor) {
    path.push(cursor);
    cursor = parents[cursor[0]][cursor[1]];
  }
  path.reverse();
  return freezeSnapshot({
    grid,
    blocked,
    costs,
    ways,
    parents,
    steps,
    path,
    result: costs[rows - 1][columns - 1]
  });
}
export function lcsPlan(first = 'CABAC', second = 'ABC') {
  if (!/^[A-Za-z]{0,6}$/.test(first) || !/^[A-Za-z]{0,6}$/.test(second)) throw new Error('Use zero to six ASCII letters in each input; case matters.');
  const lengths = Array.from({
    length: first.length + 1
  }, () => Array(second.length + 1).fill(0));
  for (let row = 1; row <= first.length; row += 1) {
    for (let column = 1; column <= second.length; column += 1) {
      lengths[row][column] = first[row - 1] === second[column - 1] ? 1 + lengths[row - 1][column - 1] : Math.max(lengths[row - 1][column], lengths[row][column - 1]);
    }
  }
  const trace = [];
  const pairs = [];
  let row = first.length;
  let column = second.length;
  while (row && column) {
    const match = first[row - 1] === second[column - 1];
    const action = match ? 'match' : lengths[row - 1][column] >= lengths[row][column - 1] ? 'up' : 'left';
    trace.push({
      row,
      column,
      action,
      pairs: pairs.slice().reverse()
    });
    if (match) {
      pairs.push([row - 1, column - 1]);
      row -= 1;
      column -= 1;
    } else if (action === 'up') row -= 1;else column -= 1;
  }
  trace.push({
    row,
    column,
    action: 'done',
    pairs: pairs.slice().reverse()
  });
  pairs.reverse();
  return freezeSnapshot({
    first,
    second,
    lengths,
    trace,
    pairs,
    witness: pairs.map(([index]) => first[index]).join(''),
    result: lengths[first.length][second.length]
  });
}
export function capacityTrace(items = [{
  weight: 2,
  value: 3
}, {
  weight: 3,
  value: 4
}], capacity = 6, direction = 'descending') {
  if (!Number.isInteger(capacity) || capacity < 0 || capacity > 8 || items.length > 4 || items.some(item => !Number.isInteger(item.weight) || item.weight < 1 || item.weight > 8 || !Number.isInteger(item.value) || item.value < 0 || item.value > 20) || !['ascending', 'descending'].includes(direction)) throw new Error('Use capacity 0–8, at most four positive-weight items and a valid direction.');
  const best = Array(capacity + 1).fill(0);
  const witnesses = Array.from({
    length: capacity + 1
  }, () => []);
  const generations = Array(capacity + 1).fill(-1);
  const frames = [freezeSnapshot({
    best,
    witnesses,
    generations,
    item: null,
    destination: null,
    source: null,
    message: 'Empty selection is allowed at every capacity. All values start at zero.'
  })];
  items.forEach((item, itemIndex) => {
    const capacities = Array.from({
      length: Math.max(0, capacity - item.weight + 1)
    }, (_, index) => item.weight + index);
    if (direction === 'descending') capacities.reverse();
    for (const destination of capacities) {
      const source = destination - item.weight;
      const sourceGeneration = generations[source];
      const sourceValue = best[source];
      const skip = best[destination];
      const take = item.value + sourceValue;
      if (take > skip) {
        best[destination] = take;
        witnesses[destination] = [...witnesses[source], itemIndex];
      }
      generations[destination] = itemIndex;
      frames.push(freezeSnapshot({
        best,
        witnesses,
        generations,
        item: itemIndex,
        destination,
        source,
        sourceGeneration,
        sourceValue,
        skip,
        take,
        message: `Capacity ${destination}: max(skip ${skip}, item ${itemIndex} value ${item.value} + source ${sourceValue}) = ${best[destination]}. Source ${source} ${sourceGeneration === itemIndex ? 'was already updated for this item: reuse is permitted.' : 'has not been updated for this item: it describes earlier items.'}`
      }));
    }
  });
  return freezeSnapshot({
    items,
    capacity,
    direction,
    frames,
    result: best[capacity],
    witness: witnesses[capacity]
  });
}
export const routeNames = Object.freeze(['A', 'B', 'C', 'D']);
export const routeCosts = Object.freeze([Object.freeze([0, 1, 1, 8]), Object.freeze([1, 0, 1, 9]), Object.freeze([1, 1, 0, 1]), Object.freeze([8, 9, 1, 0])]);
export function subsetRoutePlan(costs = routeCosts) {
  const count = costs.length;
  if (count < 1 || count > 7 || costs.some(row => row.length !== count || row.some(value => value !== null && (!Number.isInteger(value) || Math.abs(value) > 20)))) {
    throw new Error('Use a square cost matrix for one to seven vertices, integer costs −20–20 or null for no edge.');
  }
  const fullMask = (1 << count) - 1;
  const best = Array.from({
    length: fullMask + 1
  }, () => Array(count).fill(null));
  const parents = Array.from({
    length: fullMask + 1
  }, () => Array(count).fill(null));
  best[1][0] = 0;
  for (let mask = 1; mask <= fullMask; mask += 1) {
    for (let endpoint = 0; endpoint < count; endpoint += 1) {
      if (best[mask][endpoint] === null) continue;
      for (let next = 0; next < count; next += 1) {
        if ((mask & 1 << next) !== 0 || costs[endpoint][next] === null) continue;
        const nextMask = mask | 1 << next;
        const candidate = best[mask][endpoint] + costs[endpoint][next];
        if (best[nextMask][next] === null || candidate < best[nextMask][next]) {
          best[nextMask][next] = candidate;
          parents[nextMask][next] = endpoint;
        }
      }
    }
  }
  function witness(mask, endpoint) {
    if (best[mask]?.[endpoint] == null) return [];
    const path = [];
    while (endpoint !== null) {
      path.push(endpoint);
      const prior = parents[mask][endpoint];
      mask &= ~(1 << endpoint);
      endpoint = prior;
    }
    return path.reverse();
  }
  const reachableEnds = best[fullMask].map((cost, endpoint) => ({
    cost,
    endpoint
  })).filter(item => item.cost !== null);
  const winner = reachableEnds.reduce((bestEnd, candidate) => !bestEnd || candidate.cost < bestEnd.cost ? candidate : bestEnd, null);
  return {
    costs,
    best: freezeSnapshot(best),
    parents: freezeSnapshot(parents),
    fullMask,
    result: winner?.cost ?? null,
    path: winner ? witness(fullMask, winner.endpoint) : [],
    witness
  };
}
