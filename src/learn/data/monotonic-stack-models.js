function validateValues(values, nonnegative = false) {
  if (!Array.isArray(values) || values.length > 12) throw new RangeError('Use at most 12 integer values.');
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new TypeError('Every array position must be present.');
    const value = values[index];
    if (!Number.isInteger(value) || value < (nonnegative ? 0 : -20) || value > 20) throw new RangeError(nonnegative ? 'Heights must be integers from 0 to 20.' : 'Readings must be integers from −20 to 20.');
  }
}
export function parseStackValues(text, nonnegative = false) {
  if (typeof text !== 'string') throw new TypeError('Enter comma-separated integers.');
  if (!text.trim()) return [];
  const parts = text.split(',').map(part => part.trim());
  if (parts.some(part => !/^-?\d+$/.test(part))) throw new TypeError('Enter comma-separated integers without empty positions.');
  const values = parts.map(Number);
  validateValues(values, nonnegative);
  return values;
}
export function nextGreaterTrace(values, inclusive = false) {
  validateValues(values);
  if (typeof inclusive !== 'boolean') throw new TypeError('The comparison policy must be boolean.');
  const stack = [];
  const distances = Array(values.length).fill(null);
  const frames = [];
  let pushes = 0;
  let pops = 0;
  const save = (phase, current, note, resolved = null) => frames.push({
    phase,
    current,
    note,
    resolved,
    stack: [...stack],
    distances: [...distances],
    pushes,
    pops
  });
  save('start', null, 'No readings processed. Each answer is unresolved, not yet a final zero.');
  for (let current = 0; current < values.length; current += 1) {
    save('arrive', current, `Read index ${current}, value ${values[current]}. Inspect the newest unresolved index.`);
    while (stack.length && (inclusive ? values[current] >= values[stack.at(-1)] : values[current] > values[stack.at(-1)])) {
      const previous = stack.pop();
      distances[previous] = current - previous;
      pops += 1;
      save('pop', current, `Index ${previous} is answered by ${current}: distance ${current} − ${previous} = ${current - previous}.`, previous);
    }
    stack.push(current);
    pushes += 1;
    save('push', current, `Push index ${current}; it has no processed future answer yet.`);
  }
  for (const index of stack) distances[index] = 0;
  save('finish', null, 'End of input: the indices still on the stack have no qualifying future value. Their final distances are zero.');
  return {
    values: [...values],
    inclusive,
    frames,
    distances,
    pushes,
    pops
  };
}
export function smallerBoundaryTrace(heights, direction = 'left') {
  validateValues(heights, true);
  if (!['left', 'right'].includes(direction)) throw new TypeError('Choose left or right.');
  const stack = [];
  const boundaries = Array(heights.length).fill(null);
  const indices = Array.from({
    length: heights.length
  }, (_, index) => direction === 'left' ? index : heights.length - 1 - index);
  const frames = [];
  const save = (phase, current, note, removed = null) => frames.push({
    phase,
    current,
    note,
    removed,
    stack: [...stack],
    boundaries: [...boundaries]
  });
  save('start', null, `Scan toward the ${direction === 'left' ? 'right' : 'left'} to find each nearest strictly smaller ${direction} boundary.`);
  for (const current of indices) {
    save('arrive', current, `Inspect bar ${current}, height ${heights[current]}. Equal heights cannot be strictly smaller boundaries.`);
    while (stack.length && heights[stack.at(-1)] >= heights[current]) {
      const removed = stack.pop();
      save('pop', current, `Discard candidate ${removed}: bar ${current} is nearer for future queries and no taller.`, removed);
    }
    boundaries[current] = stack.length ? stack.at(-1) : direction === 'left' ? -1 : heights.length;
    stack.push(current);
    save('push', current, `Boundary for bar ${current}: ${boundaries[current]}. Then push ${current} as a new candidate.`);
  }
  save('finish', null, 'All directional boundaries are known. Missing boundaries denote positions outside the array.');
  return {
    heights: [...heights],
    direction,
    frames,
    boundaries
  };
}
export function histogramState(heights) {
  validateValues(heights, true);
  const leftTrace = smallerBoundaryTrace(heights, 'left');
  const rightTrace = smallerBoundaryTrace(heights, 'right');
  const candidates = heights.map((height, index) => {
    const left = leftTrace.boundaries[index];
    const right = rightTrace.boundaries[index];
    return {
      index,
      height,
      left,
      right,
      start: left + 1,
      end: right,
      width: right - left - 1,
      area: height * (right - left - 1)
    };
  });
  let best = null;
  for (const candidate of candidates) if (candidate.area > (best?.area ?? 0)) best = candidate;
  return {
    heights: [...heights],
    leftTrace,
    rightTrace,
    candidates,
    best,
    area: best?.area ?? 0
  };
}
