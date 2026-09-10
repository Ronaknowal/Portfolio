// Bounded exact teaching models. Geometry/counts are not measured runtime.
function immutable(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) immutable(child);
    Object.freeze(value);
  }
  return value;
}
export const intervalPresets = immutable({
  appointments: [[0, 6, 1], [1, 3, 1], [3, 5, 1], [5, 7, 1], [6, 9, 1], [7, 9, 1]],
  shortestTrap: [[0, 4, 1], [4, 8, 1], [3, 5, 1]],
  weightedTrap: [[0, 5, 10], [0, 2, 4], [2, 5, 4]]
});
export function parseIntervals(text) {
  if (!text.trim()) return [];
  const rows = text.trim().split(/[;\n]+/).filter(row => row.trim());
  if (rows.length > 6) throw new Error('Use at most six intervals.');
  const intervals = rows.map((row, index) => {
    const parts = row.trim().split(/[ ,]+/);
    if (parts.length < 2 || parts.length > 3 || parts.some(part => !/^\d+$/.test(part))) {
      throw new Error('Each row needs start, finish and optionally value, as whole numbers.');
    }
    const [start, finish, value = 1] = parts.map(Number);
    return {
      id: String.fromCharCode(65 + index),
      start,
      finish,
      value
    };
  });
  validateIntervals(intervals);
  return intervals;
}
function validateIntervals(intervals) {
  if (!Array.isArray(intervals) || intervals.length > 6 || new Set(intervals.map(item => item.id)).size !== intervals.length) {
    throw new Error('Use at most six distinct interval IDs.');
  }
  for (const item of intervals) {
    if (![item.start, item.finish, item.value].every(Number.isInteger) || item.start < 0 || item.finish > 16 || item.start >= item.finish || item.value < 1 || item.value > 30) {
      throw new Error('Use 0 ≤ start < finish ≤ 16 and a value from 1 through 30.');
    }
  }
}
export function compatible(first, second) {
  return first.finish <= second.start || second.finish <= first.start;
}
export function selectIntervals(intervals, rule = 'finish') {
  validateIntervals(intervals);
  if (!['finish', 'start', 'duration'].includes(rule)) throw new Error('Choose a supported candidate rule.');
  const score = item => rule === 'finish' ? item.finish : rule === 'start' ? item.start : item.finish - item.start;
  const ordered = [...intervals].sort((a, b) => score(a) - score(b) || a.finish - b.finish || a.start - b.start || a.id.localeCompare(b.id));
  const selected = [];
  const frames = [{
    current: null,
    selected: [],
    examined: [],
    conflicts: [],
    message: 'No interval has been examined.'
  }];
  const examined = [];
  for (const item of ordered) {
    const conflicts = selected.filter(other => !compatible(item, other));
    if (!conflicts.length) selected.push(item);
    examined.push(item.id);
    frames.push({
      current: item.id,
      selected: selected.map(other => other.id),
      examined: [...examined],
      conflicts: conflicts.map(other => other.id),
      message: conflicts.length ? `Reject ${item.id}: it overlaps selected ${conflicts.map(other => other.id).join(', ')}.` : `Keep ${item.id}: it is compatible with every selected interval.`
    });
  }
  return immutable({
    intervals: intervals.map(item => ({
      ...item
    })),
    rule,
    order: ordered.map(item => item.id),
    frames,
    selected: selected.map(item => item.id)
  });
}
export function intervalOptimum(intervals, objective = 'count') {
  validateIntervals(intervals);
  if (!['count', 'value'].includes(objective)) throw new Error('Choose count or value.');
  let best = {
    score: 0,
    selected: []
  };
  for (let mask = 0; mask < 2 ** intervals.length; mask++) {
    const selected = intervals.filter((_, index) => mask & 1 << index);
    const feasible = selected.every((item, index) => selected.slice(index + 1).every(other => compatible(item, other)));
    const score = selected.reduce((sum, item) => sum + (objective === 'count' ? 1 : item.value), 0);
    if (feasible && score > best.score) best = {
      score,
      selected: selected.map(item => item.id)
    };
  }
  return immutable(best);
}
export function assignRooms(intervals) {
  validateIntervals(intervals);
  const rooms = [];
  for (const item of [...intervals].sort((a, b) => a.start - b.start || a.finish - b.finish || a.id.localeCompare(b.id))) {
    let selectedRoom = -1;
    for (let room = 0; room < rooms.length; room++) {
      if (rooms[room].at(-1).finish <= item.start && (selectedRoom < 0 || rooms[room].at(-1).finish < rooms[selectedRoom].at(-1).finish)) selectedRoom = room;
    }
    if (selectedRoom < 0) {
      rooms.push([{ ...item }]);
    } else {
      rooms[selectedRoom].push({ ...item });
    }
  }
  let witness = {
    time: 0,
    ids: []
  };
  for (const item of intervals) {
    const active = intervals.filter(other => other.start <= item.start && item.start < other.finish);
    if (active.length > witness.ids.length) witness = {
      time: item.start,
      ids: active.map(other => other.id)
    };
  }
  return immutable({
    rooms,
    witness
  });
}
export const defaultJobs = immutable([{
  id: 'A',
  processing: 4,
  deadline: 9
}, {
  id: 'B',
  processing: 3,
  deadline: 5
}, {
  id: 'C',
  processing: 2,
  deadline: 7
}]);
export function deadlineSchedule(jobs, order = jobs.map(job => job.id)) {
  if (!Array.isArray(jobs) || !jobs.length || jobs.length > 5 || new Set(jobs.map(job => job.id)).size !== jobs.length || jobs.some(job => !Number.isInteger(job.processing) || job.processing < 1 || job.processing > 8 || !Number.isInteger(job.deadline) || job.deadline < 0 || job.deadline > 30)) {
    throw new Error('Use one to five jobs, processing times 1–8 and deadlines 0–30.');
  }
  if (order.length !== jobs.length || new Set(order).size !== jobs.length || order.some(id => !jobs.some(job => job.id === id))) throw new Error('The order must contain each job exactly once.');
  let time = 0;
  const scheduled = order.map(id => {
    const job = jobs.find(item => item.id === id);
    const start = time;
    time += job.processing;
    return {
      ...job,
      start,
      completion: time,
      lateness: time - job.deadline
    };
  });
  const maximumLateness = Math.max(...scheduled.map(job => job.lateness));
  return immutable({
    jobs: jobs.map(job => ({
      ...job
    })),
    order: [...order],
    scheduled,
    maximumLateness,
    maximumTardiness: Math.max(0, maximumLateness),
    total: time
  });
}
export function exchangeAdjacent(schedule, position) {
  if (!Number.isInteger(position) || position < 0 || position + 1 >= schedule.order.length) throw new Error('Choose an adjacent pair in the schedule.');
  const order = [...schedule.order];
  const first = schedule.scheduled[position];
  const second = schedule.scheduled[position + 1];
  [order[position], order[position + 1]] = [order[position + 1], order[position]];
  return immutable({
    before: schedule,
    after: deadlineSchedule(schedule.jobs, order),
    pair: [first.id, second.id],
    inverted: first.deadline > second.deadline
  });
}
function gcd(first, second) {
  while (second) [first, second] = [second, first % second];
  return Math.abs(first);
}
export function rational(numerator, denominator = 1) {
  const divisor = gcd(numerator, denominator);
  return {
    numerator: numerator / divisor,
    denominator: denominator / divisor
  };
}
export function addRational(first, second) {
  return rational(first.numerator * second.denominator + second.numerator * first.denominator, first.denominator * second.denominator);
}
export function formatRational(value) {
  return value.denominator === 1 ? String(value.numerator) : `${value.numerator}/${value.denominator}`;
}
export const capacityItems = immutable([{
  id: 'A',
  weight: 10,
  value: 60
}, {
  id: 'B',
  weight: 20,
  value: 100
}, {
  id: 'C',
  weight: 30,
  value: 120
}]);
export function allocateCapacity(items = capacityItems, capacity = 50, divisible = true) {
  if (!Array.isArray(items) || items.length > 6 || new Set(items.map(item => item.id)).size !== items.length || items.some(item => !Number.isInteger(item.weight) || item.weight < 1 || item.weight > 30 || !Number.isInteger(item.value) || item.value < 0 || item.value > 150) || !Number.isInteger(capacity) || capacity < 0 || capacity > 90 || typeof divisible !== 'boolean') {
    throw new Error('Use up to six items, weights 1–30, values 0–150 and capacity 0–90.');
  }
  const ordered = [...items].sort((a, b) => b.value * a.weight - a.value * b.weight || a.id.localeCompare(b.id));
  let remaining = capacity;
  let total = rational(0);
  const taken = [];
  for (const item of ordered) {
    const amount = divisible ? Math.min(remaining, item.weight) : item.weight <= remaining ? item.weight : 0;
    const value = rational(item.value * amount, item.weight);
    total = addRational(total, value);
    remaining -= amount;
    taken.push({
      ...item,
      amount,
      fraction: rational(amount, item.weight),
      contribution: value,
      remaining
    });
  }
  return immutable({
    items: items.map(item => ({
      ...item
    })),
    capacity,
    divisible,
    taken,
    remaining,
    total
  });
}
export function wholeCapacityOptimum(items, capacity) {
  allocateCapacity(items, capacity, false);
  let best = {
    value: 0,
    weight: 0,
    selected: []
  };
  for (let mask = 0; mask < 2 ** items.length; mask++) {
    const selected = items.filter((_, index) => mask & 1 << index);
    const weight = selected.reduce((sum, item) => sum + item.weight, 0);
    const value = selected.reduce((sum, item) => sum + item.value, 0);
    if (weight <= capacity && value > best.value) best = {
      value,
      weight,
      selected: selected.map(item => item.id)
    };
  }
  return immutable(best);
}
export function huffmanTree(frequencies = [2, 3, 7, 9]) {
  if (!Array.isArray(frequencies) || frequencies.length > 6 || frequencies.some(value => !Number.isInteger(value) || value < 1 || value > 30)) throw new Error('Use up to six positive frequencies from 1 through 30.');
  let serial = 0;
  const queue = frequencies.map((frequency, index) => ({
    id: serial++,
    frequency,
    symbol: String.fromCharCode(65 + index),
    left: null,
    right: null
  }));
  const merges = [];
  while (queue.length > 1) {
    queue.sort((a, b) => a.frequency - b.frequency || a.id - b.id);
    const left = queue.shift();
    const right = queue.shift();
    const node = {
      id: serial++,
      frequency: left.frequency + right.frequency,
      symbol: null,
      left,
      right
    };
    merges.push({
      left: left.frequency,
      right: right.frequency,
      sum: node.frequency
    });
    queue.push(node);
  }
  const root = queue[0] || null;
  const codes = [];
  const nodes = [];
  let nextLeaf = 0;
  function visit(node, code, depth) {
    if (!node) return;
    if (node.symbol !== null) {
      const x = 35 + nextLeaf++ * 70;
      nodes.push({
        id: node.id,
        x,
        y: 30 + depth * 72,
        frequency: node.frequency,
        symbol: node.symbol,
        parent: null
      });
      codes.push({
        symbol: node.symbol,
        frequency: node.frequency,
        code,
        depth
      });
      return x;
    }
    const leftX = visit(node.left, code + '0', depth + 1);
    const rightX = visit(node.right, code + '1', depth + 1);
    nodes.find(item => item.id === node.left.id).parent = node.id;
    nodes.find(item => item.id === node.right.id).parent = node.id;
    const x = (leftX + rightX) / 2;
    nodes.push({
      id: node.id,
      x,
      y: 30 + depth * 72,
      frequency: node.frequency,
      symbol: null,
      parent: null
    });
    return x;
  }
  visit(root, '', 0);
  return immutable({
    root,
    merges,
    nodes,
    codes: codes.sort((a, b) => a.symbol.localeCompare(b.symbol)),
    cost: codes.reduce((sum, item) => sum + item.frequency * item.depth, 0),
    width: Math.max(280, frequencies.length * 70),
    height: 76 + Math.max(0, ...codes.map(item => item.depth)) * 72
  });
}
export function reachablePrefix(values) {
  if (!Array.isArray(values) || !values.length || values.length > 8 || values.some(value => !Number.isInteger(value) || value < 0 || value > 8)) throw new Error('Use one to eight jump limits from 0 through 8.');
  let reach = 0;
  const frames = [];
  for (let index = 0; index < values.length; index++) {
    if (index > reach) {
      frames.push({
        index,
        before: reach,
        reach,
        blocked: true
      });
      return immutable({
        values: [...values],
        frames,
        reachable: false
      });
    }
    const before = reach;
    reach = Math.min(values.length - 1, Math.max(reach, index + values[index]));
    frames.push({
      index,
      before,
      reach,
      blocked: false
    });
  }
  return immutable({
    values: [...values],
    frames,
    reachable: true
  });
}
