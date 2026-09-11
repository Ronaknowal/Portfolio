function integer(value, name, maximum = 10000) {
  if (!Number.isSafeInteger(value) || value < 0 || value > maximum) {
    throw new RangeError(`${name} must be an integer from 0 to ${maximum}.`);
  }
}
function denseArray(values, name, maximum) {
  if (!Array.isArray(values) || values.length > maximum) {
    throw new TypeError(`${name} must be an array of at most ${maximum} entries.`);
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new TypeError(`${name} cannot contain holes.`);
  }
}
export function binomialCount(n, k) {
  integer(n, 'n');
  if (!Number.isSafeInteger(k)) throw new TypeError('k must be an integer.');
  if (k < 0 || k > n) return 0n;
  const smaller = Math.min(k, n - k);
  let result = 1n;
  for (let index = 1; index <= smaller; index += 1) {
    result = result * BigInt(n - smaller + index) / BigInt(index);
  }
  return result;
}
export function choiceFibers(labelCount, length, repeats = false) {
  integer(labelCount, 'Label count', 5);
  integer(length, 'Length', 4);
  if (typeof repeats !== 'boolean') throw new TypeError('Repeats must be Boolean.');
  const labels = 'ABCDE'.slice(0, labelCount).split('');
  const descriptions = [];
  function extend(prefix) {
    if (prefix.length === length) {
      descriptions.push(prefix);
      return;
    }
    for (const label of labels) {
      if (repeats || !prefix.includes(label)) extend([...prefix, label]);
    }
  }
  extend([]);
  const groups = new Map();
  for (const description of descriptions) {
    const key = [...description].sort().join('');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(description);
  }
  const fibers = [...groups.entries()].map(([key, members]) => ({
    key,
    members
  }));
  const sizes = [...new Set(fibers.map(group => group.members.length))].sort((a, b) => a - b);
  return {
    labels,
    descriptions,
    fibers,
    sizes,
    uniform: sizes.length === 1
  };
}
export function allocationCount(total, capacities, minimums = capacities.map(() => 0)) {
  integer(total, 'Total');
  denseArray(capacities, 'Capacities', 12);
  denseArray(minimums, 'Minimums', 12);
  if (capacities.length !== minimums.length) throw new RangeError('Bounds must have matching lengths.');
  capacities.forEach((value, index) => {
    integer(value, `Capacity ${index}`);
    integer(minimums[index], `Minimum ${index}`);
  });
  if (!capacities.length) return total === 0 ? 1n : 0n;
  if (capacities.some((value, index) => value < minimums[index])) return 0n;
  const shiftedTotal = total - minimums.reduce((sum, value) => sum + value, 0);
  if (shiftedTotal < 0) return 0n;
  const shiftedCaps = capacities.map((value, index) => value - minimums[index]);
  let count = 0n;
  for (let mask = 0; mask < 2 ** capacities.length; mask += 1) {
    let remaining = shiftedTotal;
    let selected = 0;
    shiftedCaps.forEach((capacity, index) => {
      if (mask & 1 << index) {
        remaining -= capacity + 1;
        selected += 1;
      }
    });
    if (remaining >= 0) {
      // n may exceed the public binomial bound by at most 11 here.
      let term = 1n;
      const choose = capacities.length - 1;
      for (let index = 1; index <= choose; index += 1) {
        term = term * BigInt(remaining + index) / BigInt(index);
      }
      count += selected % 2 ? -term : term;
    }
  }
  return count;
}
export function enumerateAllocations(total, capacities, minimums = capacities.map(() => 0)) {
  const count = allocationCount(total, capacities, minimums);
  if (total > 40 || capacities.length > 6 || count > 20000n) {
    throw new RangeError('Enumeration is limited to total 40, six containers and 20000 outcomes.');
  }
  const result = [];
  function extend(index, remaining, prefix) {
    if (index === capacities.length) {
      if (remaining === 0) result.push(prefix);
      return;
    }
    for (let value = minimums[index]; value <= Math.min(capacities[index], remaining); value += 1) {
      extend(index + 1, remaining - value, [...prefix, value]);
    }
  }
  if (count) extend(0, total, []);
  return result;
}
export function allocationWord(values) {
  denseArray(values, 'Allocation', 6);
  values.forEach((value, index) => integer(value, `Allocation ${index}`, 40));
  return values.map(value => '★'.repeat(value)).join('|');
}
export function overlapContributions(memberships) {
  denseArray(memberships, 'Memberships', 20);
  for (const row of memberships) {
    denseArray(row, 'Membership row', 3);
    if (row.length !== 3 || row.some(value => typeof value !== 'boolean')) {
      throw new TypeError('Each membership row requires three Boolean entries.');
    }
  }
  const terms = [];
  for (let mask = 1; mask < 8; mask += 1) {
    const sets = [0, 1, 2].filter(index => mask & 1 << index);
    const members = memberships.flatMap((row, index) => sets.every(set => row[set]) ? [index + 1] : []);
    terms.push({
      sets,
      members,
      sign: sets.length % 2 ? 1 : -1
    });
  }
  terms.sort((first, second) => first.sets.length - second.sets.length);
  const stages = [1, 2, 3].map(stage => {
    const weights = memberships.map((_, index) => terms.reduce((sum, term) => term.sets.length <= stage && term.members.includes(index + 1) ? sum + term.sign : sum, 0));
    return {
      weights,
      total: weights.reduce((sum, value) => sum + value, 0)
    };
  });
  const union = memberships.flatMap((row, index) => row.some(Boolean) ? [index + 1] : []);
  return {
    terms,
    stages,
    union
  };
}
export const inductionPresets = {
  fourSeven: {
    small: 4,
    large: 7,
    lower: 18,
    bases: [[1, 2], [3, 1], [5, 0], [0, 3]]
  },
  threeFive: {
    small: 3,
    large: 5,
    lower: 8,
    bases: [[1, 1], [3, 0], [0, 2]]
  }
};
export function inductionCoverage(presetName, enabled, target) {
  if (!Object.hasOwn(inductionPresets, presetName)) throw new RangeError('Unknown induction preset.');
  const preset = inductionPresets[presetName];
  denseArray(enabled, 'Enabled bases', 4);
  if (enabled.length !== preset.small || enabled.some(value => typeof value !== 'boolean')) {
    throw new TypeError('Supply one Boolean for each base case.');
  }
  integer(target, 'Target', 80);
  if (target < preset.lower) throw new RangeError('Target is below the stated theorem range.');
  const baseIndex = (target - preset.lower) % preset.small;
  const base = preset.lower + baseIndex;
  const steps = (target - base) / preset.small;
  const witness = enabled[baseIndex] ? [preset.bases[baseIndex][0] + steps, preset.bases[baseIndex][1]] : null;
  let actualWitness = null;
  for (let largeCount = 0; largeCount * preset.large <= target; largeCount += 1) {
    const remaining = target - largeCount * preset.large;
    if (remaining % preset.small === 0) {
      actualWitness = [remaining / preset.small, largeCount];
      break;
    }
  }
  const chain = Array.from({
    length: steps + 1
  }, (_, index) => target - index * preset.small);
  return {
    ...preset,
    baseIndex,
    base,
    steps,
    chain,
    supported: Boolean(witness),
    witness,
    actualWitness
  };
}
export function parenthesisPath(word) {
  if (typeof word !== 'string' || word.length > 14 || !/^[()]*$/.test(word)) {
    throw new TypeError('Use at most 14 parentheses.');
  }
  const steps = [...word].map(symbol => symbol === '(' ? 1 : -1);
  const heights = [0];
  steps.forEach(step => heights.push(heights.at(-1) + step));
  if (heights.at(-1) !== 0) throw new RangeError('The word must have equal opening and closing counts.');
  const firstBad = heights.findIndex(height => height < 0);
  const valid = firstBad === -1;
  const firstReturn = valid && word.length ? heights.findIndex((height, index) => index > 0 && height === 0) : null;
  let reflected = null;
  let reflectedHeights = null;
  if (!valid) {
    reflected = steps.map((step, index) => index < firstBad ? -step : step);
    reflectedHeights = [0];
    reflected.forEach(step => reflectedHeights.push(reflectedHeights.at(-1) + step));
  }
  return {
    word,
    steps,
    heights,
    valid,
    firstBad,
    firstReturn,
    reflected,
    reflectedHeights,
    inside: firstReturn === null ? null : word.slice(1, firstReturn - 1),
    after: firstReturn === null ? null : word.slice(firstReturn)
  };
}
export function balancedWordCounts(pairs) {
  integer(pairs, 'Pairs', 6);
  const words = [];
  function extend(word, opens, closes) {
    if (opens === pairs && closes === pairs) {
      words.push(word);
      return;
    }
    if (opens < pairs) extend(`${word}(`, opens + 1, closes);
    if (closes < pairs) extend(`${word})`, opens, closes + 1);
  }
  extend('', 0, 0);
  const valid = words.filter(word => parenthesisPath(word).valid);
  const bad = words.filter(word => !parenthesisPath(word).valid);
  const catalan = Array(pairs + 1).fill(0n);
  catalan[0] = 1n;
  for (let size = 1; size <= pairs; size += 1) {
    for (let inside = 0; inside < size; inside += 1) catalan[size] += catalan[inside] * catalan[size - 1 - inside];
  }
  return {
    words,
    valid,
    bad,
    catalan,
    total: binomialCount(2 * pairs, pairs),
    excluded: pairs ? binomialCount(2 * pairs, pairs - 1) : 0n
  };
}
export function coefficientStages(capacities) {
  denseArray(capacities, 'Capacities', 12);
  capacities.forEach((value, index) => integer(value, `Capacity ${index}`, 20));
  if (capacities.reduce((sum, value) => sum + value, 0) > 100) throw new RangeError('Total degree must not exceed 100.');
  const stages = [[1n]];
  for (const capacity of capacities) {
    const previous = stages.at(-1);
    const next = Array(previous.length + capacity).fill(0n);
    previous.forEach((count, degree) => {
      for (let chosen = 0; chosen <= capacity; chosen += 1) next[degree + chosen] += count;
    });
    stages.push(next);
  }
  return stages;
}
export function rotationOrbits(length) {
  integer(length, 'Ring length', 8);
  if (length === 0) throw new RangeError('The ring has at least one site.');
  const words = Array.from({
    length: 2 ** length
  }, (_, value) => value.toString(2).padStart(length, '0'));
  const groups = new Map();
  const fixed = Array(length).fill(0);
  for (const word of words) {
    const rotations = Array.from({
      length
    }, (_, shift) => word.slice(shift) + word.slice(0, shift));
    const key = [...rotations].sort()[0];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(word);
    rotations.forEach((rotated, shift) => {
      if (rotated === word) fixed[shift] += 1;
    });
  }
  return {
    words,
    orbits: [...groups.entries()].map(([key, members]) => ({
      key,
      members
    })),
    fixed
  };
}
