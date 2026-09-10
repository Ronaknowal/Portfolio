// Candidate histories for two different moving-range contracts. This renderer
// model copies small arrays; the native algorithms use deque endpoint operations.
function snapshot(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(snapshot));
  if (value && typeof value === 'object') return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, snapshot(item)])));
  return value;
}
function validate(values) {
  if (!Array.isArray(values) || values.length > 8 || values.some(value => !Number.isInteger(value) || Math.abs(value) > 20)) throw new Error('Use at most eight integers from −20 through 20.');
}
export function parseDequeValues(text) {
  if (!text.trim()) return [];
  const parts = text.trim().split(/[\s,]+/);
  if (parts.some(part => !/^-?\d+$/.test(part))) throw new Error('Use comma-separated whole numbers.');
  const values = parts.map(Number);
  validate(values);
  return values;
}
export function movingMaximumTrace(values = [4, 2, 2, 5, 1, 3, 0, 2], width = 3) {
  validate(values);
  if (!Number.isInteger(width) || width < 1 || width > 8) throw new Error('Use a window width from 1 through 8.');
  const deque = [];
  const answers = [];
  const frames = [];
  let pushes = 0;
  let pops = 0;
  function emit(index, phase, removed, message) {
    frames.push(snapshot({
      index,
      phase,
      removed,
      deque,
      answers,
      pushes,
      pops,
      left: Math.max(0, index - width + 1),
      right: index + 1,
      message
    }));
  }
  emit(-1, 'start', null, 'No values processed. The deque stores candidate indices, not every value in the window.');
  for (let index = 0; index < values.length; index++) {
    emit(index, 'arrive', null, `Index ${index}, value ${values[index]}, arrives. A full width-${width} window ends here only after ${width} values.`);
    while (deque.length && deque[0] <= index - width) {
      const removed = deque.shift();
      pops++;
      emit(index, 'expire', removed, `Expire index ${removed}: it lies before the current window, regardless of its value.`);
    }
    while (deque.length && values[deque.at(-1)] <= values[index]) {
      const removed = deque.pop();
      pops++;
      emit(index, 'dominate', removed, `Remove index ${removed} from the back: newer ${index} has value ${values[index]} ≥ ${values[removed]} and expires later. Equal maxima keep the newest identity.`);
    }
    deque.push(index);
    pushes++;
    emit(index, 'append', null, `Append index ${index}. Candidate values now strictly decrease from front to back.`);
    if (index + 1 >= width) {
      const endpoint = deque[0];
      answers.push({
        left: index - width + 1,
        right: index + 1,
        value: values[endpoint],
        index: endpoint
      });
      emit(index, 'answer', null, `Window [${index - width + 1},${index + 1}) has maximum ${values[endpoint]} at newest tied index ${endpoint}.`);
    }
  }
  emit(values.length - 1, 'done', null, `Complete: ${answers.length} full windows, ${pushes} pushes and ${pops} removals. A width longer than the array produces no full window.`);
  return snapshot({
    values,
    width,
    frames,
    answers,
    pushes,
    pops
  });
}
export function signedShortestTrace(values = [1, -1, 5], target = 5) {
  validate(values);
  if (!Number.isInteger(target) || target < 1 || target > 100) throw new Error('Use a positive integer target from 1 through 100.');
  const prefixes = [0];
  for (const value of values) prefixes.push(prefixes.at(-1) + value);
  const deque = [];
  const frames = [];
  let best = null;
  let pushes = 0;
  let pops = 0;
  function emit(index, phase, removed, candidate, message) {
    frames.push(snapshot({
      index,
      phase,
      removed,
      candidate,
      deque,
      best,
      pushes,
      pops,
      threshold: prefixes[index] - target,
      message
    }));
  }
  emit(0, 'start', null, null, 'Prefix P[0]=0 describes the empty prefix. Candidates are earlier start boundaries for a nonempty range.');
  for (let index = 0; index < prefixes.length; index++) {
    emit(index, 'arrive', null, null, `Consider end boundary ${index}, P=${prefixes[index]}. A start needs prefix ≤ ${prefixes[index] - target}.`);
    while (deque.length && prefixes[index] - prefixes[deque[0]] >= target) {
      const start = deque.shift();
      pops++;
      const candidate = {
        left: start,
        right: index,
        length: index - start,
        sum: prefixes[index] - prefixes[start]
      };
      if (!best || candidate.length < best.length || candidate.length === best.length && candidate.left < best.left) best = candidate;
      emit(index, 'record-front', start, candidate, `Range [${start},${index}) reaches the target with length ${candidate.length}. Record it and remove this start: a later end would only lengthen its range.`);
    }
    while (deque.length && prefixes[deque.at(-1)] >= prefixes[index]) {
      const removed = deque.pop();
      pops++;
      emit(index, 'dominate-back', removed, null, `New boundary ${index} has prefix ${prefixes[index]} ≤ old ${prefixes[removed]}. It gives every future end at least as much sum in a shorter range, so remove ${removed}.`);
    }
    deque.push(index);
    pushes++;
    emit(index, 'append', null, null, `Append boundary ${index} after checking earlier starts. Candidate prefix totals strictly increase from front to back.`);
  }
  emit(values.length, 'done', null, null, best ? `Shortest range [${best.left},${best.right}) has length ${best.length} and sum ${best.sum}.` : 'No nonempty range reaches the target.');
  return snapshot({
    values,
    target,
    prefixes,
    frames,
    best,
    pushes,
    pops
  });
}
