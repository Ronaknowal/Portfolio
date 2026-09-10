function immutable(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(immutable));
  if (value && typeof value === 'object') return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, immutable(item)])));
  return value;
}
export function parseBoundedIntegers(text, {
  maximumLength = 8,
  minimum = -9,
  maximum = 9,
  allowEmpty = false
} = {}) {
  if (!text.trim()) {
    if (allowEmpty) return [];
    throw new Error('Enter at least one integer.');
  }
  const parts = text.split(',').map(item => item.trim());
  if (parts.length > maximumLength || parts.some(item => !/^-?\d+$/.test(item))) throw new Error(`Use at most ${maximumLength} comma-separated whole numbers.`);
  const values = parts.map(Number);
  if (values.some(value => !Number.isSafeInteger(value) || value < minimum || value > maximum)) throw new Error(`Each number must be from ${minimum} through ${maximum}.`);
  return values;
}
export function traceSubsetChoices(values = [2, 4, 5], target = 5, prune = true) {
  if (values.length > 3 || values.some(value => !Number.isInteger(value) || value < 1 || value > 9)) throw new Error('Use at most three positive integers from 1 through 9.');
  if (!Number.isInteger(target) || target < 0 || target > 30) throw new Error('Target must be an integer from 0 through 30.');
  const frames = [];
  const path = [];
  const answers = [];
  const calls = [];
  const visited = {};
  let entered = 0;
  const suffix = Array(values.length + 1).fill(0);
  for (let i = values.length - 1; i >= 0; i--) suffix[i] = suffix[i + 1] + values[i];
  function emit(phase, message, prefix) {
    frames.push(immutable({
      phase,
      message,
      prefix,
      path,
      answers,
      calls,
      visited,
      entered
    }));
  }
  function search(index, sum, prefix) {
    entered++;
    calls.push({
      index,
      sum,
      prefix
    });
    visited[prefix] = 'entered';
    emit('enter', `Enter index ${index}, sum ${sum}.`, prefix);
    if (index === values.length) {
      if (sum === target) {
        answers.push([...path]);
        visited[prefix] = 'solution';
        emit('solution', `Copy the selected positions as answer ${answers.length}.`, prefix);
      } else {
        visited[prefix] = 'miss';
        emit('reject', `No choices remain; ${sum} does not equal ${target}.`, prefix);
      }
    } else if (prune && (sum > target || sum + suffix[index] < target)) {
      visited[prefix] = 'pruned';
      emit('prune', sum > target ? `Prune: ${sum} already exceeds ${target}; every remaining value is positive.` : `Prune: even taking the remaining ${suffix[index]} reaches only ${sum + suffix[index]}.`, prefix);
    } else {
      path.push(index);
      emit('choose', `Choose position ${index} (value ${values[index]}).`, prefix);
      search(index + 1, sum + values[index], prefix + '1');
      path.pop();
      emit('undo', `Undo position ${index}; the parent's path is restored.`, prefix);
      emit('skip', `Explore skipping position ${index}.`, prefix);
      search(index + 1, sum, prefix + '0');
    }
    calls.pop();
    emit('return', `Return from index ${index}.`, prefix);
  }
  search(0, 0, 'r');
  emit('done', `Finished: ${answers.length} answers; ${entered} entered calls. Path restored to empty.`, 'r');
  return immutable({
    values,
    target,
    prune,
    frames,
    answers,
    entered
  });
}
export function subsetTree(values) {
  const nodes = [];
  const width = Math.max(280, 2 ** values.length * 70);
  function visit(index, sum, prefix, low, high, parent) {
    nodes.push({
      id: prefix,
      parent,
      depth: index,
      sum,
      x: (low + high) / 2,
      y: 32 + index * 78
    });
    if (index < values.length) {
      const middle = (low + high) / 2;
      visit(index + 1, sum + values[index], prefix + '1', low, middle, prefix);
      visit(index + 1, sum, prefix + '0', middle, high, prefix);
    }
  }
  visit(0, 0, 'r', 0, width, null);
  return {
    nodes,
    width,
    height: 74 + values.length * 78
  };
}
export function queenConflicts(queens, row, column) {
  return queens.flatMap((queenColumn, queenRow) => queenColumn === column || Math.abs(queenRow - row) === Math.abs(queenColumn - column) ? [{
    row: queenRow,
    column: queenColumn,
    kind: queenColumn === column ? 'column' : 'diagonal'
  }] : []);
}
export function traceQueens(n = 4) {
  if (!Number.isInteger(n) || n < 1 || n > 5) throw new Error('Board size must be from 1 through 5.');
  const queens = [];
  const answers = [];
  const frames = [];
  let candidates = 0;
  function emit(phase, row, column, conflicts = []) {
    const message = phase === 'try' ? `Try row ${row}, column ${column}.` : phase === 'reject' ? `Reject (${row}, ${column}): ${conflicts.map(conflict => `${conflict.kind} conflict with (${conflict.row}, ${conflict.column})`).join('; ')}.` : phase === 'place' ? `Place queen (${row}, ${column}); explore the next row.` : phase === 'undo' ? `Remove queen (${row}, ${column}); restore the previous rows.` : phase === 'solution' ? `Copy complete solution ${answers.length}.` : phase === 'done' ? `All branches finished: ${answers.length} solutions, ${candidates} candidate tests.` : 'Start with an empty board. Each row will receive one queen.';
    frames.push(immutable({
      phase,
      row,
      column,
      queens,
      answers,
      conflicts,
      candidates,
      message
    }));
  }
  function search(row) {
    if (row === n) {
      answers.push([...queens]);
      emit('solution', row, null);
      return;
    }
    for (let column = 0; column < n; column++) {
      candidates++;
      const conflicts = queenConflicts(queens, row, column);
      emit('try', row, column, conflicts);
      if (conflicts.length) {
        emit('reject', row, column, conflicts);
        continue;
      }
      queens.push(column);
      emit('place', row, column);
      search(row + 1);
      queens.pop();
      emit('undo', row, column);
    }
  }
  emit('start', 0, null);
  search(0);
  emit('done', 0, null);
  return immutable({
    n,
    frames,
    answers,
    candidates
  });
}
function better(first, second) {
  return first.sum > second.sum || first.sum === second.sum && (first.low < second.low || first.low === second.low && first.high <= second.high) ? first : second;
}
export function summarizeSubarray(values = [-2, 4, -1, 3, -5, 2]) {
  if (!values.length || values.length > 8 || values.some(value => !Number.isInteger(value) || value < -9 || value > 9)) throw new Error('Use one to eight integers from −9 through 9.');
  const nodes = [];
  function solve(low, high) {
    const id = `${low}:${high}`;
    if (high - low === 1) {
      const range = {
        sum: values[low],
        low,
        high
      };
      const leaf = {
        id,
        low,
        high,
        total: values[low],
        prefix: range,
        suffix: range,
        best: range,
        left: null,
        right: null,
        crossing: null
      };
      nodes.push(leaf);
      return leaf;
    }
    const middle = Math.floor((low + high) / 2);
    const left = solve(low, middle);
    const right = solve(middle, high);
    const crossing = {
      sum: left.suffix.sum + right.prefix.sum,
      low: left.suffix.low,
      high: right.prefix.high
    };
    const prefix = better(left.prefix, {
      sum: left.total + right.prefix.sum,
      low,
      high: right.prefix.high
    });
    const suffix = better(right.suffix, {
      sum: left.suffix.sum + right.total,
      low: left.suffix.low,
      high
    });
    const best = better(better(left.best, right.best), crossing);
    const node = {
      id,
      low,
      high,
      total: left.total + right.total,
      prefix,
      suffix,
      best,
      crossing,
      left: left.id,
      right: right.id
    };
    nodes.push(node);
    return node;
  }
  const root = solve(0, values.length);
  return immutable({
    values,
    nodes,
    root,
    combines: nodes.filter(node => node.left).length
  });
}
