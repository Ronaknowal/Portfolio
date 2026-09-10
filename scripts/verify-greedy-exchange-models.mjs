import assert from 'node:assert/strict';
import { allocateCapacity, assignRooms, capacityItems, compatible, deadlineSchedule, defaultJobs, exchangeAdjacent, huffmanTree, intervalOptimum, intervalPresets, parseIntervals, reachablePrefix, selectIntervals, wholeCapacityOptimum } from '../src/learn/data/greedy-exchange-models.js';

let seed = 701;
function random(limit) {
  seed = (1664525 * seed + 1013904223) >>> 0;
  return (seed >>> 8) % limit;
}
function* permutations(items) {
  if (!items.length) yield [];
  for (let index = 0; index < items.length; index++) {
    for (const tail of permutations(items.filter((_, position) => position !== index))) yield [items[index], ...tail];
  }
}
function intervalOracle(items, objective) {
  let best = 0;
  for (let mask = 0; mask < 2 ** items.length; mask++) {
    let feasible = true;
    let score = 0;
    const occupancy = Array(17).fill(0);
    for (let index = 0; index < items.length; index++) {
      if (!(mask & (1 << index))) continue;
      const item = items[index];
      score += objective === 'count' ? 1 : item.value;
      for (let time = item.start; time < item.finish; time++) {
        occupancy[time]++;
        if (occupancy[time] > 1) feasible = false;
      }
    }
    if (feasible) best = Math.max(best, score);
  }
  return best;
}

let intervalCases = 0;
for (let sample = 0; sample < 1500; sample++) {
  const items = Array.from({ length: random(7) }, (_, index) => {
    const start = random(9);
    return { id: String.fromCharCode(65 + index), start, finish: start + 1 + random(7), value: 1 + random(20) };
  });
  const before = JSON.stringify(items);
  for (const objective of ['count', 'value']) assert.equal(intervalOptimum(items, objective).score, intervalOracle(items, objective));
  const optimum = intervalOracle(items, 'count');
  for (const rule of ['finish', 'start', 'duration']) {
    const trace = selectIntervals(items, rule);
    assert.equal(trace.frames.length, items.length + 1);
    assert.ok(Object.isFrozen(trace.frames[0].selected));
    for (const frame of trace.frames) {
      const chosen = items.filter(item => frame.selected.includes(item.id));
      assert.equal(intervalOracle(chosen, 'count'), chosen.length);
      assert.ok(frame.conflicts.every(id => frame.selected.includes(id)));
    }
    if (rule === 'finish') assert.equal(trace.selected.length, optimum);
  }
  const rooms = assignRooms(items);
  const depth = Math.max(0, ...Array.from({ length: 17 }, (_, time) => items.filter(item => item.start <= time && time < item.finish).length));
  assert.equal(rooms.rooms.length, depth);
  assert.equal(rooms.witness.ids.length, depth);
  assert.deepEqual(rooms.rooms.flat().map(item => item.id).sort(), items.map(item => item.id).sort());
  for (const room of rooms.rooms) assert.equal(intervalOracle(room, 'count'), room.length);
  assert.equal(JSON.stringify(items), before);
  intervalCases++;
}
const traps = Object.fromEntries(Object.entries(intervalPresets).map(([name, rows]) => [name, parseIntervals(rows.map(row => row.join(',')).join('\n'))]));
assert.equal(selectIntervals(traps.appointments).selected.length, 4);
assert.equal(selectIntervals(traps.appointments, 'start').selected.length, 2);
assert.equal(selectIntervals(traps.shortestTrap, 'duration').selected.length, 1);
assert.equal(intervalOptimum(traps.shortestTrap).score, 2);
assert.equal(intervalOptimum(traps.weightedTrap, 'value').score, 10);
assert.equal(selectIntervals(traps.weightedTrap).selected.reduce((sum, id) => sum + traps.weightedTrap.find(item => item.id === id).value, 0), 8);
assert.ok(compatible({ start: 0, finish: 2 }, { start: 2, finish: 3 }));

let scheduleCases = 0;
let invertedExchanges = 0;
for (let sample = 0; sample < 500; sample++) {
  const jobs = Array.from({ length: 1 + random(4) }, (_, index) => ({ id: String.fromCharCode(65 + index), processing: 1 + random(5), deadline: random(16) }));
  let optimum = Infinity;
  for (const order of permutations(jobs.map(job => job.id))) {
    let time = 0;
    const expected = order.map(id => { const job = jobs.find(item => item.id === id); time += job.processing; return time - job.deadline; });
    const schedule = deadlineSchedule(jobs, order);
    assert.equal(schedule.maximumLateness, Math.max(...expected));
    optimum = Math.min(optimum, Math.max(...expected));
    for (let pair = 0; pair + 1 < order.length; pair++) {
      const exchange = exchangeAdjacent(schedule, pair);
      assert.deepEqual(exchange.after.scheduled.slice(pair + 2), schedule.scheduled.slice(pair + 2));
      if (exchange.inverted) {
        assert.ok(exchange.after.maximumLateness <= schedule.maximumLateness);
        invertedExchanges++;
      }
    }
    scheduleCases++;
  }
  const byDeadline = [...jobs].sort((a, b) => a.deadline - b.deadline).map(job => job.id);
  assert.equal(deadlineSchedule(jobs, byDeadline).maximumLateness, optimum);
}
assert.equal(deadlineSchedule(defaultJobs).maximumLateness, 2);
assert.equal(deadlineSchedule(defaultJobs, ['B', 'C', 'A']).maximumLateness, 0);

function fractionalVertexOracle(items, capacity) {
  let best = 0;
  for (let mask = 0; mask < 2 ** items.length; mask++) {
    let weight = 0;
    let value = 0;
    for (let index = 0; index < items.length; index++) if (mask & (1 << index)) { weight += items[index].weight; value += items[index].value; }
    if (weight > capacity) continue;
    best = Math.max(best, value);
    for (let partial = 0; partial < items.length; partial++) {
      if (mask & (1 << partial)) continue;
      const amount = Math.min(items[partial].weight, capacity - weight);
      best = Math.max(best, value + items[partial].value * amount / items[partial].weight);
    }
  }
  return best;
}
let capacityCases = 0;
for (let sample = 0; sample < 1200; sample++) {
  const items = Array.from({ length: random(5) }, (_, index) => ({ id: String.fromCharCode(65 + index), weight: 1 + random(9), value: random(31) }));
  const capacity = random(25);
  const result = allocateCapacity(items, capacity);
  assert.ok(Math.abs(result.total.numerator / result.total.denominator - fractionalVertexOracle(items, capacity)) < 1e-10);
  assert.equal(result.taken.reduce((sum, item) => sum + item.amount, 0) + result.remaining, capacity);
  assert.ok(result.taken.filter(item => item.amount > 0 && item.amount < item.weight).length <= 1);
  const table = Array(capacity + 1).fill(0);
  for (const item of items) for (let space = capacity; space >= item.weight; space--) table[space] = Math.max(table[space], table[space - item.weight] + item.value);
  assert.equal(wholeCapacityOptimum(items, capacity).value, table[capacity]);
  const whole = allocateCapacity(items, capacity, false);
  assert.ok(whole.taken.every(item => item.amount === 0 || item.amount === item.weight));
  assert.ok(whole.total.numerator / whole.total.denominator <= table[capacity]);
  capacityCases++;
}
assert.equal(allocateCapacity(capacityItems, 50).total.numerator, 240);
assert.equal(allocateCapacity(capacityItems, 50, false).total.numerator, 160);
assert.equal(wholeCapacityOptimum(capacityItems, 50).value, 220);

const mergeMemo = new Map();
function mergeOracle(weights) {
  if (weights.length < 2) return 0;
  const key = [...weights].sort((a, b) => a - b).join(',');
  if (mergeMemo.has(key)) return mergeMemo.get(key);
  let best = Infinity;
  for (let first = 0; first < weights.length; first++) for (let second = first + 1; second < weights.length; second++) {
    const combined = weights[first] + weights[second];
    const rest = weights.filter((_, index) => index !== first && index !== second);
    best = Math.min(best, combined + mergeOracle([...rest, combined]));
  }
  mergeMemo.set(key, best);
  return best;
}
for (let sample = 0; sample < 400; sample++) {
  const weights = Array.from({ length: random(7) }, () => 1 + random(9));
  const model = huffmanTree(weights);
  assert.equal(model.cost, mergeOracle(weights));
  assert.equal(model.cost, model.merges.reduce((sum, item) => sum + item.sum, 0));
  assert.equal(model.nodes.length, Math.max(0, 2 * weights.length - 1));
  for (const code of model.codes) {
    assert.equal(code.code.length, code.depth);
    assert.ok(model.codes.every(other => other === code || !other.code.startsWith(code.code)));
  }
  assert.ok(model.nodes.every(node => node.x >= 19 && node.x <= model.width - 19 && node.y + 38 <= model.height));
}
assert.equal(huffmanTree().cost, 38);
assert.equal(huffmanTree([1, 2, 4, 8]).cost, 25);
assert.deepEqual(huffmanTree([1, 2, 4, 8]).codes.map(item => item.depth), [3, 3, 2, 1]);
let reachCases = 0;
for (let sample = 0; sample < 2000; sample++) {
  const values = Array.from({ length: 1 + random(8) }, () => random(5));
  const seen = new Set([0]);
  const queue = [0];
  while (queue.length) {
    const source = queue.shift();
    for (let jump = 1; jump <= values[source]; jump++) {
      const target = source + jump;
      if (target < values.length && !seen.has(target)) { seen.add(target); queue.push(target); }
    }
  }
  assert.equal(reachablePrefix(values).reachable, seen.has(values.length - 1));
  reachCases++;
}
for (const action of [() => parseIntervals('2,2'), () => parseIntervals('-1,3'), () => parseIntervals('0,2,0'), () => parseIntervals('0,x'), () => deadlineSchedule([{ id: 'A', processing: 0, deadline: 1 }]), () => exchangeAdjacent(deadlineSchedule(defaultJobs), 2), () => allocateCapacity(capacityItems, -1), () => allocateCapacity([{ id: 'A', weight: 0, value: 1 }], 1), () => huffmanTree([0]), () => reachablePrefix([])]) assert.throws(action);
console.log(JSON.stringify({ intervalCases, scheduleCases, invertedExchanges, capacityCases, huffmanCases: 400, reachCases }));
