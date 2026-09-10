// Exact bounded teaching traces. Counts describe operations, never elapsed time.
export function validateIntegers(values, { sorted = false, nonnegative = false, limit = 16 } = {}) {
  if (!Array.isArray(values) || values.length > limit || values.some(value => !Number.isSafeInteger(value) || Math.abs(value) > 1000 || (nonnegative && value < 0))) {
    throw new RangeError(`Use at most ${limit} small ${nonnegative ? 'nonnegative ' : ''}integers.`);
  }
  if (sorted && values.some((value, index) => index > 0 && values[index - 1] > value)) {
    throw new RangeError('Values must be in nondecreasing order.');
  }
}

export function boundaryTrace(values, target, side = 'left') {
  validateIntegers(values, { sorted: true });
  if (!Number.isSafeInteger(target) || Math.abs(target) > 1000 || !['left', 'right'].includes(side)) {
    throw new RangeError('Choose a bounded integer target and a left or right boundary.');
  }
  let low = 0;
  let high = values.length;
  const steps = [];
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    const belongsBefore = side === 'left' ? values[middle] < target : values[middle] <= target;
    const nextLow = belongsBefore ? middle + 1 : low;
    const nextHigh = belongsBefore ? high : middle;
    steps.push({ low, high, middle, belongsBefore, nextLow, nextHigh, done: false });
    low = nextLow;
    high = nextHigh;
  }
  steps.push({ low, high, middle: null, done: true });
  return { values: [...values], target, side, steps, boundary: low };
}

export const mergeRecords = {
  left: [{ key: 2, id: 'A' }, { key: 4, id: 'B' }, { key: 4, id: 'C' }],
  right: [{ key: 1, id: 'D' }, { key: 4, id: 'E' }, { key: 6, id: 'F' }],
};

export function mergeChoice(state, lane, records = mergeRecords) {
  if (!['left', 'right'].includes(lane)) throw new RangeError('Choose a source lane.');
  const chosen = records[lane][state[lane]];
  const otherLane = lane === 'left' ? 'right' : 'left';
  const other = records[otherLane][state[otherLane]];
  if (!chosen) return { ...state, feedback: 'That lane is empty; take the other head.' };
  if (other && chosen.key > other.key) {
    return { ...state, feedback: `${chosen.key}${chosen.id} would precede the smaller ${other.key}${other.id}. Take the smaller remaining head.` };
  }
  const breaksStability = lane === 'right' && other && chosen.key === other.key;
  return {
    ...state,
    [lane]: state[lane] + 1,
    output: [...state.output, { ...chosen }],
    stable: state.stable && !breaksStability,
    feedback: breaksStability
      ? `The keys remain sorted, but ${chosen.id} has passed earlier equal-key record ${other.id}. Stability is now lost.`
      : `Moved ${chosen.key}${chosen.id}. Every unchosen head is at least this key.`,
  };
}

export function initialMergeState() {
  return { left: 0, right: 0, output: [], stable: true, feedback: 'Predict which head is safe. Letters identify original record order A through F.' };
}

export function pairTrace(values, target) {
  validateIntegers(values, { sorted: true, limit: 8 });
  if (!Number.isSafeInteger(target) || Math.abs(target) > 1000) throw new RangeError('Use a small integer target.');
  let left = 0;
  let right = values.length - 1;
  const steps = [];
  let answer = null;
  while (left < right) {
    const sum = values[left] + values[right];
    const action = sum < target ? 'left' : sum > target ? 'right' : 'found';
    steps.push({ left, right, sum, action, done: action === 'found' });
    if (action === 'found') {
      answer = [left, right];
      break;
    }
    if (action === 'left') left += 1;
    else right -= 1;
  }
  if (!answer) steps.push({ left, right, sum: null, action: 'absent', done: true });
  return { values: [...values], target, steps, answer };
}

export function windowTrace(values, target) {
  validateIntegers(values, { nonnegative: true, limit: 10 });
  if (!Number.isSafeInteger(target) || target <= 0 || target > 1000) throw new RangeError('Use a positive bounded target.');
  let left = 0;
  let right = 0;
  let total = 0;
  let best = null;
  const steps = [{ left, right, total, best, action: 'start', note: 'The half-open window [0,0) is empty.' }];
  while (right < values.length) {
    total += values[right];
    right += 1;
    steps.push({ left, right, total, best: best && [...best], action: 'add', note: `Added index ${right - 1}. The window is [${left},${right}).` });
    while (total >= target) {
      if (!best || right - left < best[1] - best[0]) best = [left, right];
      steps.push({ left, right, total, best: [...best], action: 'qualifies', note: `Sum ${total} reaches ${target}. Record this length, then try dropping index ${left}.` });
      total -= values[left];
      left += 1;
      steps.push({ left, right, total, best: [...best], action: 'remove', note: `Removed index ${left - 1}. A shorter window can still qualify; test its sum again.` });
    }
  }
  steps.push({ left, right, total, best: best && [...best], action: 'done', note: best ? `Shortest length is ${best[1] - best[0]}; one interval is [${best[0]},${best[1]}).` : 'No nonempty window reaches the target.' });
  return { values: [...values], target, steps, best };
}

export const rateJobs = [3, 6, 7];

export function rateState(speed, budget, jobs = rateJobs) {
  validateIntegers(jobs, { nonnegative: true, limit: 8 });
  if (!jobs.length || jobs.some(value => value === 0) || !Number.isSafeInteger(speed) || speed < 1 || speed > 1000 || !Number.isSafeInteger(budget) || budget < 0 || budget > 1000) {
    throw new RangeError('Use positive jobs/speed and a nonnegative bounded budget.');
  }
  const slots = jobs.map(job => Math.floor((job + speed - 1) / speed));
  const total = slots.reduce((sum, count) => sum + count, 0);
  return { speed, budget, slots, total, feasible: total <= budget };
}

export function rateSearch(budget, jobs = rateJobs) {
  rateState(1, budget, jobs);
  const maximum = Math.max(...jobs);
  const options = Array.from({ length: maximum }, (_, index) => rateState(index + 1, budget, jobs));
  const first = options.find(option => option.feasible);
  return { options, answer: first ? first.speed : null };
}
