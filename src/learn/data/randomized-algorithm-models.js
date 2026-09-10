// Finite teaching experiments: exact small integer state, floating display probabilities.
// No browser pseudo-random generator is used as evidence for a probability theorem.
function integerInRange(value, lower, upper, name) {
  if (!Number.isInteger(value) || value < lower || value > upper) {
    throw new RangeError(`${name} must be an integer from ${lower} to ${upper}.`);
  }
}
export function rejectionMap(sourceSize, targetSize, reject = true) {
  integerInRange(sourceSize, 2, 64, 'Source size');
  integerInRange(targetSize, 1, sourceSize, 'Target size');
  const limit = sourceSize - sourceSize % targetSize;
  const outcomes = Array.from({
    length: sourceSize
  }, (_, raw) => ({
    raw,
    result: reject && raw >= limit ? null : raw % targetSize
  }));
  const counts = Array.from({
    length: targetSize
  }, (_, result) => outcomes.filter(outcome => outcome.result === result).length);
  const accepted = reject ? limit : sourceSize;
  return {
    outcomes,
    counts,
    accepted,
    limit,
    expectedAttempts: sourceSize / accepted
  };
}
export function shuffleTrace(choices, values = ['A', 'B', 'C', 'D']) {
  const state = [...values];
  const trace = [{
    values: [...state],
    activeEnd: state.length - 1,
    choice: null
  }];
  choices.forEach((choice, step) => {
    const activeEnd = state.length - 1 - step;
    if (activeEnd < 1) throw new RangeError('The shuffle is already complete.');
    integerInRange(choice, 0, activeEnd, 'Chosen index');
    [state[activeEnd], state[choice]] = [state[choice], state[activeEnd]];
    trace.push({
      values: [...state],
      activeEnd: activeEnd - 1,
      choice
    });
  });
  return trace;
}
export function reservoirTrace(capacity, choices, size = 6) {
  integerInRange(size, 1, 8, 'Stream size');
  integerInRange(capacity, 1, size, 'Reservoir size');
  const sample = Array.from({
    length: capacity
  }, (_, index) => index);
  const trace = [{
    sample: [...sample],
    seen: capacity,
    draw: null,
    replaced: null
  }];
  choices.forEach((draw, step) => {
    const incoming = capacity + step;
    if (incoming >= size) throw new RangeError('The stream is exhausted.');
    integerInRange(draw, 0, incoming, 'Draw');
    const replaced = draw < capacity ? sample[draw] : null;
    if (draw < capacity) sample[draw] = incoming;
    trace.push({
      sample: [...sample],
      seen: incoming + 1,
      draw,
      replaced
    });
  });
  return trace;
}
export function reservoirDistribution(size, capacity) {
  integerInRange(size, 1, 8, 'Stream size');
  integerInRange(capacity, 1, size, 'Reservoir size');
  let distribution = new Map([[Array.from({
    length: capacity
  }, (_, index) => index).join(','), 1]]);
  for (let incoming = capacity; incoming < size; incoming += 1) {
    const next = new Map();
    for (const [key, probability] of distribution) {
      const prior = key.split(',').map(Number);
      for (let draw = 0; draw <= incoming; draw += 1) {
        const sample = [...prior];
        if (draw < capacity) sample[draw] = incoming;
        const result = sample.sort((left, right) => left - right).join(',');
        next.set(result, (next.get(result) || 0) + probability / (incoming + 1));
      }
    }
    distribution = next;
  }
  return [...distribution].sort(([left], [right]) => left.localeCompare(right)).map(([key, probability]) => ({
    sample: key.split(',').map(Number),
    probability
  }));
}
export function quickselectTrace(values, rank, choices) {
  if (!values.length || values.length > 30 || values.some(value => !Number.isFinite(value))) {
    throw new RangeError('Use 1–30 finite values.');
  }
  integerInRange(rank, 0, values.length - 1, 'Rank');
  let active = [...values];
  let wanted = rank;
  let work = 0;
  const trace = [{
    active,
    wanted,
    work,
    result: null,
    pivot: null,
    lower: [],
    equal: [],
    upper: []
  }];
  for (const choice of choices) {
    if (!active.length) throw new RangeError('Selection is complete.');
    integerInRange(choice, 0, active.length - 1, 'Pivot index');
    const pivot = active[choice];
    const lower = active.filter(value => value < pivot);
    const equal = active.filter(value => value === pivot);
    const upper = active.filter(value => value > pivot);
    work += active.length;
    let result = null;
    if (wanted < lower.length) active = lower;else if (wanted < lower.length + equal.length) {
      result = pivot;
      active = [];
    } else {
      wanted -= lower.length + equal.length;
      active = upper;
    }
    trace.push({
      active,
      wanted,
      work,
      result,
      pivot,
      lower,
      equal,
      upper
    });
  }
  return trace;
}
const selectionExpectations = new Map();
export function expectedSelectionWork(size, rank) {
  integerInRange(size, 1, 30, 'Size');
  integerInRange(rank, 0, size - 1, 'Rank');
  const key = `${size}:${rank}`;
  if (selectionExpectations.has(key)) return selectionExpectations.get(key);
  let expectation = size;
  for (let pivot = 0; pivot < size; pivot += 1) {
    if (pivot < rank) expectation += expectedSelectionWork(size - pivot - 1, rank - pivot - 1) / size;else if (pivot > rank) expectation += expectedSelectionWork(pivot, rank) / size;
  }
  selectionExpectations.set(key, expectation);
  return expectation;
}
export const productFixtures = {
  correct: {
    title: 'Correct product',
    left: [[1, 2], [0, 1]],
    right: [[2, 0], [1, 3]],
    claimed: [[4, 6], [1, 3]]
  },
  cancellation: {
    title: 'Errors that cancel for [1, 1]',
    left: [[1, 2], [0, 1]],
    right: [[2, 0], [1, 3]],
    claimed: [[5, 5], [1, 3]]
  },
  even: {
    title: 'Even integer error',
    left: [[1, 2], [0, 1]],
    right: [[2, 0], [1, 3]],
    claimed: [[6, 6], [1, 3]]
  }
};
export function matrixVector(matrix, vector) {
  return matrix.map(row => row.reduce((sum, value, index) => sum + value * vector[index], 0));
}
export function productProbe(fixtureName, bits) {
  const fixture = productFixtures[fixtureName];
  if (!fixture || bits.length !== 2 || bits.some(bit => bit !== 0 && bit !== 1)) {
    throw new RangeError('Choose a fixture and two binary coordinates.');
  }
  const intermediate = matrixVector(fixture.right, bits);
  const actual = matrixVector(fixture.left, intermediate);
  const claimed = matrixVector(fixture.claimed, bits);
  const residual = actual.map((value, index) => value - claimed[index]);
  return {
    ...fixture,
    intermediate,
    actual,
    claimedProbe: claimed,
    residual,
    passes: residual.every(value => value === 0)
  };
}
export function chooseCount(total, selected) {
  if (selected < 0 || selected > total) return 0;
  let result = 1;
  for (let index = 1; index <= selected; index += 1) result = result * (total - index + 1) / index;
  return result;
}
export function amplification(rounds, failure = 0.25) {
  integerInRange(rounds, 1, 25, 'Rounds');
  if (!(failure >= 0 && failure <= 1)) throw new RangeError('Failure probability must be in [0,1].');
  const distribution = Array.from({
    length: rounds + 1
  }, (_, errors) => ({
    errors,
    probability: chooseCount(rounds, errors) * failure ** errors * (1 - failure) ** (rounds - errors)
  }));
  return {
    distribution,
    independentAllFail: failure ** rounds,
    identicalAllFail: failure,
    majorityError: distribution.filter(outcome => outcome.errors > rounds / 2).reduce((sum, outcome) => sum + outcome.probability, 0)
  };
}
export function sampleBudget(epsilon, delta) {
  if (!(epsilon > 0 && epsilon <= 1 && delta > 0 && delta < 1)) throw new RangeError('Require 0<epsilon≤1 and 0<delta<1.');
  return Math.ceil(Math.log(2 / delta) / (2 * epsilon ** 2));
}
