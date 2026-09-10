// Bounded teaching models; these do not execute Python or implement arbitrary ndarrays.
export const sensorValues = [18, 20, 24, 26, 30, 32];
export const shapeText = shape => shape.length ? `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})` : '()';
export const product = shape => shape.reduce((a, b) => a * b, 1);

export function coordinates(shape) {
  return Array.from({ length: product(shape) }, (_, flat) => {
    const coordinate = Array(shape.length);
    for (let i = shape.length - 1; i >= 0; i--) { coordinate[i] = flat % shape[i]; flat = Math.floor(flat / shape[i]); }
    return coordinate;
  });
}

export function flatIndex(coordinate, shape) {
  return coordinate.reduce((flat, value, i) => flat * shape[i] + value, 0);
}

export const selectionCases = [
  { id: 'scalar', label: 'One reading: X[1, 0]', code: 'X[1, 0]', shape: [], source: [2], why: 'The row index chooses time 1; the column index chooses sensor A. Two integer indices leave one numeric scalar.' },
  { id: 'row', label: 'One time: X[1, :]', code: 'X[1, :]', shape: [2], source: [2, 3], why: 'Fix time 1, then keep every sensor. The integer row index removes axis 0; two sensor values remain.' },
  { id: 'column', label: 'One sensor: X[:, 0]', code: 'X[:, 0]', shape: [3], source: [0, 2, 4], why: 'Keep all three times, fix sensor A. The integer column index removes axis 1, giving a one-dimensional array.' },
  { id: 'column2d', label: 'Keep the column axis: X[:, :1]', code: 'X[:, :1]', shape: [3, 1], source: [0, 2, 4], why: 'The slice :1 keeps a column axis of length one. Values match X[:, 0], but their shapes have different broadcasting contracts.' },
  { id: 'rows', label: 'Whole rows: X[X[:, 0] >= 24]', code: 'X[X[:, 0] >= 24]', shape: [2, 2], source: [2, 3, 4, 5], why: 'The row mask is [False, True, True]. It keeps times 1 and 2 with BOTH sensor readings, including their companion columns.' },
  { id: 'cells', label: 'Individual cells: X[X >= 24]', code: 'X[X >= 24]', shape: [4], source: [2, 3, 4, 5], why: 'A condition on every cell selects true cells into a one-dimensional result. The time/sensor grid is no longer preserved.' },
  { id: 'transpose', label: 'Swap the axes: X.T', code: 'X.T', shape: [2, 3], source: [0, 2, 4, 1, 3, 5], why: 'Transpose swaps which axis means time and which means sensor. Each output row now follows one sensor across all three times.' },
  { id: 'reshape', label: 'Regroup the sequence: X.reshape(2, 3)', code: 'X.reshape(2, 3)', shape: [2, 3], source: [0, 1, 2, 3, 4, 5], why: 'Default C-order reshape reads the original row-by-row sequence and groups it into two rows of three. This does NOT put one sensor in each row.' },
];

export function selectionResult(id) {
  const selected = selectionCases.find(item => item.id === id);
  if (!selected) throw new Error('Unknown selection');
  return { ...selected, values: selected.source.map(index => sensorValues[index]) };
}

export const memoryCases = [
  { id: 'view', label: 'Basic slice (view)', code: 'picked = X[:, 0]', shares: true },
  { id: 'copy', label: 'Explicit copy', code: 'picked = X[:, 0].copy()', shares: false },
  { id: 'advanced', label: 'Integer-array selection (copy)', code: 'picked = X[[0, 1, 2], 0]', shares: false },
];

export function memoryState(kind, writeIndex = null, newValue = 99) {
  const choice = memoryCases.find(item => item.id === kind);
  if (!choice) throw new Error('Unknown memory case');
  const original = [...sensorValues];
  const picked = [18, 24, 30];
  if (writeIndex !== null) {
    if (![0, 1, 2].includes(writeIndex)) throw new Error('Invalid selected index');
    picked[writeIndex] = newValue;
    if (choice.shares) original[writeIndex * 2] = newValue;
  }
  return { ...choice, original, picked, offsets: choice.shares ? [0, 16, 32] : [0, 8, 16], buffer: choice.shares ? 'A' : 'B' };
}

export const broadcastCases = [
  { id: 'sensor', label: 'One offset per sensor', aShape: [3, 2], a: sensorValues, bShape: [2], b: [2, 4], code: 'X - np.array([2, 4])', meaning: 'Use offset 2 for sensor A and offset 4 for sensor B at every time.' },
  { id: 'time', label: 'One offset per time, with an axis', aShape: [3, 2], a: sensorValues, bShape: [3, 1], b: [1, 2, 3], code: 'X - np.array([1, 2, 3])[:, None]', meaning: 'At time 0 use 1 for both sensors; at time 1 use 2; at time 2 use 3.' },
  { id: 'invalid', label: 'One offset per time, missing the axis', aShape: [3, 2], a: sensorValues, bShape: [3], b: [1, 2, 3], code: 'X - np.array([1, 2, 3])', meaning: 'Three time offsets accidentally align with the last axis, which has two sensors.' },
  { id: 'scalar', label: 'One offset for every reading', aShape: [3, 2], a: sensorValues, bShape: [], b: [2], code: 'X - 2', meaning: 'The same scalar offset is used for every reading.' },
  { id: 'outer', label: 'A valid but surprising square result', aShape: [3, 1], a: [18, 24, 30], bShape: [3], b: [18, 24, 30], code: 'v[:, None] - v', meaning: 'Each reading is compared with every reading. This computes pairwise differences, not three self-differences.' },
];

export function broadcastShape(aShape, bShape) {
  const dimensions = Math.max(aShape.length, bShape.length);
  const aAligned = [...Array(dimensions - aShape.length).fill(1), ...aShape];
  const bAligned = [...Array(dimensions - bShape.length).fill(1), ...bShape];
  const shape = [];
  for (let i = 0; i < dimensions; i++) {
    const a = aAligned[i], b = bAligned[i];
    if (a !== b && a !== 1 && b !== 1) return { valid: false, aAligned, bAligned, mismatchAxis: i };
    shape.push(a === 1 ? b : a);
  }
  return { valid: true, aAligned, bAligned, shape };
}

export function broadcastResult(preset) {
  const plan = broadcastShape(preset.aShape, preset.bShape);
  if (!plan.valid) return { ...preset, ...plan, cells: [] };
  const cells = coordinates(plan.shape).map(coordinate => {
    const source = shape => coordinate.slice(coordinate.length - shape.length).map((value, i) => shape[i] === 1 ? 0 : value);
    // A scalar has no coordinate axes, so explicitly preserve the empty coordinate.
    const aCoordinate = preset.aShape.length ? source(preset.aShape) : [];
    const bCoordinate = preset.bShape.length ? source(preset.bShape) : [];
    const aIndex = flatIndex(aCoordinate, preset.aShape), bIndex = flatIndex(bCoordinate, preset.bShape);
    return { coordinate, aCoordinate, bCoordinate, aIndex, bIndex, value: preset.a[aIndex] - preset.b[bIndex] };
  });
  return { ...preset, ...plan, cells, values: cells.map(cell => cell.value) };
}

export function reductionResult(axis = 0, keepdims = false) {
  if (![0, 1].includes(axis)) throw new Error('Only the two pictured axes are supported');
  const groups = axis === 0 ? [[0, 2, 4], [1, 3, 5]] : [[0, 1], [2, 3], [4, 5]];
  return { groups, values: groups.map(indices => indices.reduce((sum, index) => sum + sensorValues[index], 0) / indices.length),
    shape: keepdims ? (axis === 0 ? [1, 2] : [3, 1]) : [groups.length] };
}
