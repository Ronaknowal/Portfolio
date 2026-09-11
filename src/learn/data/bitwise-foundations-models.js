// Bounded mathematical word models. Arithmetic avoids JavaScript's implicit
// signed-32-bit coercion; this is not an emulator for Python's integer runtime.
function checkedInteger(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
  return value;
}
function checkedWidth(width) {
  return checkedInteger(width, 1, 16, 'Width');
}
function checkedWord(value, width) {
  checkedWidth(width);
  return checkedInteger(value, 0, 2 ** width - 1, 'Word');
}
export function wordBits(value, width) {
  checkedWord(value, width);
  return Array.from({
    length: width
  }, (_, column) => {
    const position = width - column - 1;
    return {
      position,
      bit: Math.floor(value / 2 ** position) % 2,
      weight: 2 ** position
    };
  });
}
function combineWords(left, right, width, predicate) {
  const leftBits = wordBits(left, width);
  const rightBits = wordBits(right, width);
  return leftBits.reduce((value, cell, index) => value + (predicate(cell.bit, rightBits[index].bit) ? cell.weight : 0), 0);
}
export function selectedMembers(mask, width) {
  return wordBits(mask, width).filter(cell => cell.bit === 1).map(cell => cell.position).reverse();
}
export function toggleMember(mask, width, position) {
  checkedWord(mask, width);
  checkedInteger(position, 0, width - 1, 'Position');
  const weight = 2 ** position;
  return mask + (Math.floor(mask / weight) % 2 === 0 ? weight : -weight);
}
export function packedSetView(left, right, width, operation) {
  const predicates = {
    intersection: (a, b) => a === 1 && b === 1,
    union: (a, b) => a === 1 || b === 1,
    difference: (a, b) => a === 1 && b === 0,
    symmetric: (a, b) => a !== b,
    complement: a => a === 0
  };
  if (!Object.hasOwn(predicates, operation)) throw new RangeError('Unknown set operation.');
  const result = combineWords(left, right, width, predicates[operation]);
  return {
    left,
    right,
    width,
    operation,
    result,
    leftMembers: selectedMembers(left, width),
    rightMembers: selectedMembers(right, width),
    resultMembers: selectedMembers(result, width)
  };
}
export function wordInterpretation(unsigned, width, shift) {
  checkedWord(unsigned, width);
  checkedInteger(shift, 0, width, 'Shift');
  const modulus = 2 ** width;
  const signed = unsigned >= modulus / 2 ? unsigned - modulus : unsigned;
  const logical = Math.floor(unsigned / 2 ** shift);
  const arithmetic = Math.floor(signed / 2 ** shift);
  const arithmeticWord = (arithmetic % modulus + modulus) % modulus;
  const leftWord = unsigned * 2 ** shift % modulus;
  const cells = wordBits(unsigned, width).map(cell => ({
    ...cell,
    signedWeight: cell.position === width - 1 ? -cell.weight : cell.weight
  }));
  const rightOrigins = cells.map(cell => ({
    position: cell.position,
    source: cell.position + shift < width ? cell.position + shift : null
  }));
  const leftOrigins = cells.map(cell => ({
    position: cell.position,
    source: cell.position - shift >= 0 ? cell.position - shift : null
  }));
  return {
    unsigned,
    signed,
    width,
    shift,
    cells,
    logical,
    arithmetic,
    arithmeticWord,
    leftWord,
    unboundedLeft: unsigned * 2 ** shift,
    rightOrigins,
    leftOrigins
  };
}
export function parityTrace(values, width = 8) {
  checkedWidth(width);
  if (!Array.isArray(values) || values.length < 1 || values.length > 12) {
    throw new RangeError('Use 1 to 12 events.');
  }
  Array.from(values).forEach(value => checkedWord(value, width));
  const counts = new Map();
  const states = [{
    consumed: 0,
    value: null,
    before: 0,
    accumulator: 0
  }];
  let accumulator = 0;
  values.forEach((value, index) => {
    const before = accumulator;
    accumulator = combineWords(accumulator, value, width, (a, b) => a !== b);
    counts.set(value, (counts.get(value) || 0) + 1);
    states.push({
      consumed: index + 1,
      value,
      before,
      accumulator
    });
  });
  const frequencies = [...counts].sort((a, b) => a[0] - b[0]);
  const singletons = frequencies.filter(([, count]) => count === 1).map(([value]) => value);
  const othersPaired = frequencies.every(([, count]) => count === 1 || count === 2);
  return {
    values: [...values],
    width,
    states,
    frequencies,
    singletons,
    oneSingletonPromise: singletons.length === 1 && othersPaired,
    twoSingletonPromise: singletons.length === 2 && othersPaired
  };
}
export function sparseBitTrace(value, width = 8) {
  checkedWord(value, width);
  const states = [];
  let current = value;
  let removed = 0;
  while (current > 0) {
    const next = combineWords(current, current - 1, width, (a, b) => a === 1 && b === 1);
    const clearedPosition = Math.log2(current - next);
    states.push({
      current,
      minusOne: current - 1,
      next,
      removed,
      clearedPosition
    });
    current = next;
    removed += 1;
  }
  states.push({
    current: 0,
    minusOne: null,
    next: 0,
    removed,
    clearedPosition: null
  });
  return {
    value,
    width,
    states,
    population: removed,
    isPowerOfTwo: removed === 1
  };
}
export function twoSingletonPartition(values, width = 8) {
  const trace = parityTrace(values, width);
  const total = trace.states.at(-1).accumulator;
  if (total === 0) throw new RangeError('Zero XOR contradicts the two-distinct-singletons promise.');
  const separatingBit = wordBits(total, width).filter(cell => cell.bit === 1).at(-1).weight;
  const groups = [[], []];
  values.forEach(value => groups[Math.floor(value / separatingBit) % 2].push(value));
  const answers = groups.map(group => group.reduce((value, event) => combineWords(value, event, width, (a, b) => a !== b), 0));
  return {
    total,
    separatingBit,
    groups,
    answers,
    promiseHolds: trace.twoSingletonPromise
  };
}
