// Small, transparent fixtures for the introductory views. These are constructed
// teaching examples, independent of the measured classifier and its weights.
export function retainedTrace(retention, inputs = [5, 0, 0]) {
  let state = 0;
  let sum = 0;
  return inputs.map((input, step) => {
    const previous = state;
    const retained = retention * previous;
    const written = (1 - retention) * input;
    state = retained + written;
    sum += input;
    return { step, input, previous, retained, written, state, average: sum / (step + 1) };
  });
}

export function impulseTrails(inputs = [2, 0, 1, 0], retention = .5) {
  const contributions = inputs.map((input, birth) => inputs.map((_, step) =>
    step < birth ? null : input * retention ** (step - birth)));
  return {
    contributions,
    kernel: inputs.map((_, lag) => retention ** lag),
    outputs: inputs.map((_, step) => contributions.reduce((sum, row) => sum + (row[step] ?? 0), 0)),
  };
}

export function markedMemory(inputs = [4, 9, -7, 6], markers = [true, false, false, true]) {
  let selected = 0;
  return inputs.map((input, step) => {
    if (markers[step]) selected = input;
    return { input, marked: markers[step], selected, delayed: step < 2 ? 0 : inputs[step - 2] };
  });
}

export function matrixWriteExample() {
  const old = [[2, 1], [0, 0]];
  const decay = .5, writeWeights = [0, 1], value = [3, -1], readWeights = [1, 1];
  const retained = old.map(row => row.map(entry => decay * entry));
  const write = writeWeights.map(weight => value.map(entry => weight * entry));
  const state = retained.map((row, i) => row.map((entry, j) => entry + write[i][j]));
  const read = value.map((_, j) => state.reduce((sum, row, i) => sum + readWeights[i] * row[j], 0));
  return { old, decay, writeWeights, value, readWeights, retained, write, state, read };
}
