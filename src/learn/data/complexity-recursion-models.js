export const loopPatterns = {
  square: {
    label: 'All ordered pairs',
    code: 'for i in range(n):\n    for j in range(n):\n        work(i, j)',
    bound: 'Θ(n²)'
  },
  triangle: {
    label: 'Earlier-index pairs',
    code: 'for i in range(n):\n    for j in range(i):\n        work(i, j)',
    bound: 'Θ(n²)'
  },
  doubling: {
    label: 'Doubling inner index',
    code: 'for i in range(n):\n    j = 1\n    while j < n:\n        work(i, j)\n        j *= 2',
    bound: 'Θ(n log n)'
  }
};
export function loopWork(pattern, size) {
  if (!Object.hasOwn(loopPatterns, pattern) || !Number.isInteger(size) || size < 0 || size > 32) {
    throw new RangeError('Choose a supported loop and integer size from 0 to 32.');
  }
  const rows = [];
  for (let outer = 0; outer < size; outer += 1) {
    const columns = [];
    if (pattern === 'doubling') {
      for (let inner = 1; inner < size; inner *= 2) columns.push(inner);
    } else {
      const limit = pattern === 'square' ? size : outer;
      for (let inner = 0; inner < limit; inner += 1) columns.push(inner);
    }
    rows.push({
      outer,
      columns
    });
  }
  return {
    rows,
    total: rows.reduce((total, row) => total + row.columns.length, 0)
  };
}
export function parseSumValues(text) {
  if (text.trim() === '') return [];
  const fields = text.split(',').map(value => value.trim());
  if (fields.length > 8 || fields.some(value => !/^-?\d+$/.test(value))) {
    throw new RangeError('Enter up to eight comma-separated integers, or clear the field for an empty list.');
  }
  const values = fields.map(Number);
  if (values.some(value => !Number.isInteger(value) || Math.abs(value) > 99)) {
    throw new RangeError('Each integer must be between −99 and 99.');
  }
  return values;
}
export function sumCallTrace(values) {
  if (!Array.isArray(values) || values.length > 8 || values.some(value => !Number.isInteger(value) || Math.abs(value) > 99)) {
    throw new RangeError('The trace supports at most eight integers between −99 and 99.');
  }
  const frames = [];
  const states = [];
  let calls = 0;
  function record(event, message, result = null) {
    states.push({
      event,
      message,
      frames: frames.map(frame => ({
        ...frame
      })),
      calls,
      result
    });
  }
  function visit(index) {
    calls += 1;
    const frame = {
      index,
      phase: 'enter',
      value: null,
      child: null
    };
    frames.push(frame);
    record('enter', `Enter suffix(${index}): ${values.length - index} values remain.`);
    if (index === values.length) {
      frame.phase = 'return';
      frame.value = 0;
      record('base', 'The empty suffix contributes 0. No child is called.');
    } else {
      frame.phase = 'waiting';
      record('call', `suffix(${index}) saves ${values[index]} + ? and calls suffix(${index + 1}).`);
      frame.child = visit(index + 1);
      frame.phase = 'return';
      frame.value = values[index] + frame.child;
      record('combine', `suffix(${index}) resumes: ${values[index]} + ${frame.child} = ${frame.value}.`);
    }
    frames.pop();
    return frame.value;
  }
  const result = visit(0);
  record('finished', `Return ${result} to the caller. No suffix frame remains.`, result);
  return states;
}
export const recurrencePatterns = {
  chainConstant: {
    label: 'One child: size − 1, local work 1',
    formula: 'T(n) = T(n − 1) + 1',
    bound: 'Θ(n)',
    shrink: 'one',
    children: 1,
    cost: 'one'
  },
  chainLinear: {
    label: 'One child: size − 1, local work size',
    formula: 'T(n) = T(n − 1) + n',
    bound: 'Θ(n²)',
    shrink: 'one',
    children: 1,
    cost: 'size'
  },
  halfConstant: {
    label: 'One half-size child, local work 1',
    formula: 'T(n) = T(n/2) + 1',
    bound: 'Θ(log n)',
    shrink: 'half',
    children: 1,
    cost: 'one'
  },
  halfLinear: {
    label: 'One half-size child, local work size',
    formula: 'T(n) = T(n/2) + n',
    bound: 'Θ(n)',
    shrink: 'half',
    children: 1,
    cost: 'size'
  },
  twoHalfLinear: {
    label: 'Two half-size children, local work size',
    formula: 'T(n) = 2T(n/2) + n',
    bound: 'Θ(n log n)',
    shrink: 'half',
    children: 2,
    cost: 'size'
  }
};
export function recurrenceLevels(pattern, size) {
  if (!Object.hasOwn(recurrencePatterns, pattern) || !Number.isInteger(size) || size < 1 || size > 32 || !Number.isInteger(Math.log2(size))) {
    throw new RangeError('Choose a supported recurrence and a power of two from 1 to 32.');
  }
  const configuration = recurrencePatterns[pattern];
  const levels = [];
  let subproblemSize = size;
  let nodes = 1;
  while (true) {
    const localWork = subproblemSize === 1 || configuration.cost === 'one' ? 1 : subproblemSize;
    levels.push({
      depth: levels.length,
      nodes,
      size: subproblemSize,
      localWork,
      work: nodes * localWork,
      leaf: subproblemSize === 1
    });
    if (subproblemSize === 1) break;
    subproblemSize = configuration.shrink === 'one' ? subproblemSize - 1 : subproblemSize / 2;
    nodes *= configuration.children;
  }
  return {
    levels,
    totalWork: levels.reduce((total, level) => total + level.work, 0),
    totalCalls: levels.reduce((total, level) => total + level.nodes, 0),
    peakFrames: levels.length
  };
}
