export function parseProofValues(text, {
  categories = false
} = {}) {
  const values = text.trim() ? text.split(',').map(part => {
    if (!/^-?\d+$/.test(part.trim())) throw new Error('Use comma-separated integers, or leave the list empty.');
    return Number(part.trim());
  }) : [];
  if (values.length > 8 || values.some(value => !Number.isSafeInteger(value) || value < -20 || value > 20)) {
    throw new Error('Use at most eight integers from −20 to 20.');
  }
  if (categories && values.some(value => ![0, 1, 2].includes(value))) {
    throw new Error('This partition accepts only the categories 0, 1 and 2.');
  }
  return values;
}
function validateValues(values) {
  if (!Array.isArray(values) || values.length > 8 || values.some(value => !Number.isSafeInteger(value) || value < -20 || value > 20)) {
    throw new Error('Use at most eight integers from −20 to 20.');
  }
}
export function searchClaim(values, target, index, claim = 'prefix') {
  const bounds = Number.isInteger(index) && 0 <= index && index <= values.length;
  if (claim === 'bounds') return bounds;
  if (claim === 'whole') return bounds && values.every(value => value !== target);
  if (claim === 'prefix') return bounds && values.slice(0, index).every(value => value !== target);
  throw new Error('Choose one of the displayed claims.');
}
export function searchProofTrace(values, target, skip = false) {
  validateValues(values);
  if (!Number.isSafeInteger(target) || Math.abs(target) > 20 || typeof skip !== 'boolean') {
    throw new Error('Use an integer target from −20 to 20 and a displayed update rule.');
  }
  const states = [];
  let index = 0;
  let action = 'Initialize i = 0. No positions have been rejected.';
  function save(result = null) {
    states.push({
      index,
      action,
      result,
      finished: result !== null,
      variant: values.length - index,
      prefixValid: searchClaim(values, target, index)
    });
  }
  save();
  while (index < values.length) {
    if (values[index] === target) {
      action = `Return i = ${index}: the current value matches.`;
      save(index);
      return states;
    }
    const previous = index;
    index = Math.min(index + (skip ? 2 : 1), values.length);
    action = `Value at ${previous} is not the target. Move i from ${previous} to ${index}.`;
    save();
  }
  action = 'Guard i < n is false. Return −1.';
  save(-1);
  return states;
}

// Examine every index-boundary state for this fixed input. A missing witness
// is finite evidence, not a universal proof or a formal-verifier result.
export function searchObligations(values, target, claim, skip = false) {
  searchProofTrace(values, target, skip);
  const result = {
    initialization: null,
    preservation: null,
    matchExit: null,
    absentExit: null,
    examined: values.length + 1
  };
  if (!searchClaim(values, target, 0, claim)) result.initialization = {
    index: 0
  };
  for (let index = 0; index <= values.length; index++) {
    if (!searchClaim(values, target, index, claim)) continue;
    if (index === values.length) {
      if (values.includes(target)) result.absentExit = {
        index,
        missed: values.indexOf(target)
      };
    } else if (values[index] === target) {
      if (values.indexOf(target) !== index && result.matchExit === null) {
        result.matchExit = {
          index,
          earlier: values.indexOf(target)
        };
      }
    } else {
      const next = Math.min(index + (skip ? 2 : 1), values.length);
      if (!searchClaim(values, target, next, claim) && result.preservation === null) {
        result.preservation = {
          index,
          next
        };
      }
    }
  }
  return result;
}
function cloneOccurrences(items) {
  return items.map(item => ({
    ...item
  }));
}
export function compactionTrace(values, removed) {
  validateValues(values);
  if (!Number.isSafeInteger(removed) || Math.abs(removed) > 20) throw new Error('Use a removed integer from −20 to 20.');
  const original = values.map((value, origin) => ({
    value,
    origin
  }));
  const working = cloneOccurrences(original);
  const states = [];
  let read = 0;
  let write = 0;
  let action = 'Initialize read = write = 0. The kept prefix is empty.';
  let source = null;
  let destination = null;
  function save() {
    const expected = original.slice(0, read).filter(item => item.value !== removed);
    states.push({
      read,
      write,
      working: cloneOccurrences(working),
      action,
      source,
      destination,
      variant: values.length - read,
      prefixValid: JSON.stringify(working.slice(0, write)) === JSON.stringify(expected),
      unreadValid: JSON.stringify(working.slice(read)) === JSON.stringify(original.slice(read)),
      boundsValid: 0 <= write && write <= read && read <= values.length
    });
  }
  save();
  while (read < values.length) {
    source = read;
    destination = null;
    const item = working[read];
    if (item.value !== removed) {
      destination = write;
      working[write] = {
        ...item
      };
      action = `Keep original occurrence #${item.origin}: copy position ${read} to ${write}, then advance write.`;
      write++;
    } else {
      action = `Reject occurrence #${item.origin}: leave write at ${write}.`;
    }
    read++;
    action += ' Advance read to the next boundary.';
    save();
  }
  return {
    original,
    states
  };
}
export function partitionTrace(values, skipIncoming = false) {
  validateValues(values);
  if (values.some(value => ![0, 1, 2].includes(value)) || typeof skipIncoming !== 'boolean') {
    throw new Error('Use categories 0, 1 and 2, and a displayed update rule.');
  }
  const working = values.map((value, origin) => ({
    value,
    origin
  }));
  const states = [];
  let low = 0;
  let middle = 0;
  let high = values.length;
  let action = 'Initially every occurrence belongs to the unknown region.';
  let changed = [];
  function save() {
    states.push({
      low,
      middle,
      high,
      working: cloneOccurrences(working),
      action,
      changed: [...changed],
      variant: high - middle,
      checks: {
        bounds: 0 <= low && low <= middle && middle <= high && high <= values.length,
        zero: working.slice(0, low).every(item => item.value === 0),
        one: working.slice(low, middle).every(item => item.value === 1),
        two: working.slice(high).every(item => item.value === 2),
        occurrences: working.map(item => item.origin).sort((a, b) => a - b).every((origin, index) => origin === index)
      }
    });
  }
  function swap(left, right) {
    [working[left], working[right]] = [working[right], working[left]];
    changed = [left, right];
  }
  save();
  while (middle < high) {
    const value = working[middle].value;
    if (value === 0) {
      action = `Category 0 at ${middle}: swap with low = ${low}; advance low and middle.`;
      swap(low, middle);
      low++;
      middle++;
    } else if (value === 1) {
      action = `Category 1 at ${middle}: extend the known-one region by advancing middle.`;
      changed = [middle];
      middle++;
    } else {
      high--;
      action = `Category 2 at ${middle}: decrement high to ${high}, then swap. `;
      swap(middle, high);
      if (skipIncoming) {
        middle++;
        action += 'Faulty rule also advances middle without classifying the incoming occurrence.';
      } else {
        action += 'Keep middle here: the incoming occurrence is still unknown.';
      }
    }
    save();
  }
  return states;
}
export function euclidTrace(first, second) {
  if (![first, second].every(value => Number.isInteger(value) && value >= 0 && value <= 96) || first + second === 0) {
    throw new Error('Use integers from 0 to 96, with at least one positive value.');
  }
  const divisorCandidates = Array.from({
    length: Math.max(first, second)
  }, (_, index) => index + 1);
  const states = [];
  let a = first;
  let b = second;
  do {
    const terminal = b === 0;
    states.push({
      a,
      b,
      quotient: terminal ? null : Math.floor(a / b),
      remainder: terminal ? null : a % b,
      commonDivisors: divisorCandidates.filter(divisor => a % divisor === 0 && b % divisor === 0),
      terminal
    });
    if (terminal) break;
    [a, b] = [b, a % b];
  } while (true);
  return states;
}
