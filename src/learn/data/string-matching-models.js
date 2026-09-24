export const defaultMatchingText = 'abababacaba';
export const defaultMatchingPattern = 'ababaca';
export function symbols(value) {
  if (typeof value !== 'string') throw new TypeError('Use a string.');
  return Array.from(value);
}
export function validateMatchingInput(text, pattern) {
  if (symbols(text).length > 40 || symbols(pattern).length > 18) {
    throw new RangeError('Use at most 40 text code points and 18 pattern code points in this drawing.');
  }
}
export function prefixFunction(sequence) {
  const prefix = Array(sequence.length).fill(0);
  for (let position = 1; position < sequence.length; position++) {
    let matched = prefix[position - 1];
    while (matched > 0 && sequence[position] !== sequence[matched]) {
      matched = prefix[matched - 1];
    }
    if (sequence[position] === sequence[matched]) matched++;
    prefix[position] = matched;
  }
  return prefix;
}
export function prefixTrace(pattern) {
  validateMatchingInput('', pattern);
  const letters = symbols(pattern);
  const table = Array(letters.length).fill(null);
  const states = [];
  let position = 0,
    matched = 0;
  function record(kind, action, compared = null) {
    states.push({
      kind,
      action,
      position,
      matched,
      table: [...table],
      compared
    });
  }
  if (!letters.length) {
    record('done', 'The empty pattern has an empty prefix table.');
    return {
      letters,
      states,
      table
    };
  }
  table[0] = 0;
  record('initial', 'A one-symbol prefix has no nonempty proper border: π[0] = 0.');
  for (position = 1; position < letters.length; position++) {
    matched = table[position - 1];
    record('candidate', `At i=${position}, try extending the known border of length ${matched}.`);
    while (true) {
      const compared = [position, matched];
      if (letters[position] === letters[matched]) {
        matched++;
        record('equal', `Equal symbols extend the candidate to length ${matched}.`, compared);
        break;
      }
      if (matched === 0) {
        record('unequal', 'Even the first pattern symbol differs; no nonempty border can extend here.', compared);
        break;
      }
      const previous = matched;
      matched = table[matched - 1];
      record('fallback', `Mismatch: replace length ${previous} with π[${previous - 1}]=${matched}. Keep i=${position}.`, compared);
    }
    table[position] = matched;
    record('commit', `Store π[${position}]=${matched}. The prefix and suffix of that length are equal.`);
  }
  return {
    letters,
    states,
    table
  };
}
export function kmpTrace(text, pattern) {
  validateMatchingInput(text, pattern);
  const letters = symbols(text),
    needle = symbols(pattern);
  const table = prefixFunction(needle),
    states = [],
    matches = [];
  let consumed = 0,
    matched = 0,
    comparisons = 0;
  function record(kind, action, compared = null) {
    states.push({
      kind,
      action,
      consumed,
      matched,
      comparisons,
      matches: [...matches],
      compared
    });
  }
  record('initial', 'No text consumed. q=0: no pattern prefix is currently retained.');
  if (!needle.length) {
    matches.push(...Array.from({
      length: letters.length + 1
    }, (_, index) => index));
    record('done', `Empty pattern: report all ${letters.length + 1} boundaries, without character comparisons.`);
    return {
      letters,
      needle,
      table,
      states,
      matches
    };
  }
  while (consumed < letters.length) {
    const compared = [consumed, matched];
    comparisons++;
    if (letters[consumed] === needle[matched]) {
      consumed++;
      matched++;
      record('equal', `Equal: consume text[${compared[0]}]; q becomes ${matched}.`, compared);
      if (matched === needle.length) {
        matches.push(consumed - needle.length);
        record('match', `Report start ${consumed - needle.length}. The complete pattern ends just before position ${consumed}.`);
        matched = table[matched - 1];
        record('overlap', `Retain q=π[${needle.length - 1}]=${matched} so an overlapping match can continue.`);
      }
    } else if (matched > 0) {
      const previous = matched;
      matched = table[matched - 1];
      record('fallback', `Mismatch: q ${previous} → ${matched}. Text[${consumed}] has NOT been consumed.`, compared);
    } else {
      consumed++;
      record('skip', `No candidate prefix fits text[${compared[0]}]; consume this symbol with q=0.`, compared);
    }
  }
  record('done', `Text exhausted. Exact starts: ${matches.length ? matches.join(', ') : 'none'}.`);
  return {
    letters,
    needle,
    table,
    states,
    matches
  };
}
export function streamTrace(chunks, pattern, resetAtSeam = false) {
  if (!Array.isArray(chunks) || chunks.length > 12) throw new RangeError('Use up to 12 decoded chunks.');
  const text = chunks.join('');
  validateMatchingInput(text, pattern);
  chunks.forEach(symbols);
  const needle = symbols(pattern),
    table = prefixFunction(needle);
  let consumed = 0,
    matched = 0;
  const matches = needle.length ? [] : [0];
  const states = [{
    fed: 0,
    consumed,
    matched,
    matches: [...matches],
    added: [...matches]
  }];
  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
    if (resetAtSeam) matched = 0;
    const added = [];
    for (const letter of symbols(chunks[chunkIndex])) {
      if (!needle.length) {
        consumed++;
        matches.push(consumed);
        added.push(consumed);
        continue;
      }
      while (matched > 0 && letter !== needle[matched]) matched = table[matched - 1];
      if (letter === needle[matched]) matched++;
      consumed++;
      if (matched === needle.length) {
        const start = consumed - needle.length;
        matches.push(start);
        added.push(start);
        matched = table[matched - 1];
      }
    }
    states.push({
      fed: chunkIndex + 1,
      consumed,
      matched,
      matches: [...matches],
      added
    });
  }
  return {
    chunks,
    needle,
    table,
    states,
    matches
  };
}
function positiveModulo(value, modulus) {
  return (value % modulus + modulus) % modulus;
}
export function polynomialHash(sequence, base, modulus) {
  let value = 0;
  for (const letter of sequence) value = (value * base + letter.codePointAt(0)) % modulus;
  return value;
}
export function rollingTrace(text, pattern, base = 3, modulus = 7) {
  validateMatchingInput(text, pattern);
  if (!Number.isInteger(base) || !Number.isInteger(modulus) || base < 2 || base >= modulus || modulus > 1009) {
    throw new RangeError('Use integer 2 ≤ base < modulus ≤ 1009. These are teaching bounds.');
  }
  const letters = symbols(text),
    needle = symbols(pattern),
    width = needle.length;
  const target = polynomialHash(needle, base, modulus),
    rows = [],
    matches = [];
  if (!width) {
    matches.push(...Array.from({
      length: letters.length + 1
    }, (_, index) => index));
    return {
      letters,
      needle,
      base,
      modulus,
      target,
      power: 1,
      rows,
      matches
    };
  }
  let power = 1;
  for (let count = 1; count < width; count++) power = power * base % modulus;
  let value = polynomialHash(letters.slice(0, width), base, modulus);
  for (let start = 0; start + width <= letters.length; start++) {
    const window = letters.slice(start, start + width);
    const candidate = value === target;
    let checks = 0,
      exact = false;
    if (candidate) {
      exact = true;
      for (let index = 0; index < width; index++) {
        checks++;
        if (window[index] !== needle[index]) {
          exact = false;
          break;
        }
      }
      if (exact) matches.push(start);
    }
    let update = null;
    if (start + width < letters.length) {
      const outgoing = letters[start].codePointAt(0);
      const incoming = letters[start + width].codePointAt(0);
      const remainder = positiveModulo(value - outgoing * power, modulus);
      const next = (remainder * base + incoming) % modulus;
      update = {
        outgoing,
        incoming,
        remainder,
        next
      };
    }
    rows.push({
      start,
      window,
      value,
      candidate,
      exact,
      checks,
      update
    });
    if (update) value = update.next;
  }
  return {
    letters,
    needle,
    base,
    modulus,
    target,
    power,
    rows,
    matches
  };
}
export function zFunction(sequence) {
  const values = Array(sequence.length).fill(0);
  let left = 0,
    right = 0;
  for (let position = 1; position < sequence.length; position++) {
    if (position < right) values[position] = Math.min(right - position, values[position - left]);
    while (position + values[position] < sequence.length && sequence[values[position]] === sequence[position + values[position]]) {
      values[position]++;
    }
    if (position + values[position] > right) {
      left = position;
      right = position + values[position];
    }
  }
  return values;
}
