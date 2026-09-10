export const linearMapPresets = {
  shear: {
    name: 'Shear right',
    matrix: [[1, 1], [0, 1]]
  },
  identity: {
    name: 'Identity',
    matrix: [[1, 0], [0, 1]]
  },
  rotate: {
    name: 'Quarter-turn counterclockwise',
    matrix: [[0, -1], [1, 0]]
  },
  reflect: {
    name: 'Reflect across the vertical axis',
    matrix: [[-1, 0], [0, 1]]
  },
  stretch: {
    name: 'Double horizontal coordinates',
    matrix: [[2, 0], [0, 1]]
  },
  collapse: {
    name: 'Collapse onto the horizontal axis',
    matrix: [[1, 1], [0, 0]]
  },
  zero: {
    name: 'Send every input to zero',
    matrix: [[0, 0], [0, 0]]
  }
};
export function dot(left, right) {
  if (left.length !== right.length || left.some(value => !Number.isFinite(value)) || right.some(value => !Number.isFinite(value))) {
    throw new Error('Use equal-length vectors of finite real numbers.');
  }
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}
export function projectVector(vector, direction) {
  const product = dot(vector, direction);
  const squaredLength = dot(direction, direction);
  const length = Math.sqrt(dot(vector, vector));
  if (squaredLength === 0) {
    return {
      product,
      length,
      directionLength: 0,
      coefficient: null,
      projection: null,
      residual: null,
      cosine: null
    };
  }
  const coefficient = product / squaredLength;
  const projection = direction.map(value => coefficient * value);
  return {
    product,
    length,
    directionLength: Math.sqrt(squaredLength),
    coefficient,
    projection,
    residual: vector.map((value, index) => value - projection[index]),
    cosine: length === 0 ? null : Math.max(-1, Math.min(1, product / (length * Math.sqrt(squaredLength))))
  };
}
export function applyMatrix(matrix, vector) {
  return matrix.map(row => dot(row, vector));
}
export function parseSmallMatrix(text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('Enter at least one row, such as 2,1;0,3.');
  const rows = text.trim().split(';').map(row => row.split(',').map(cell => {
    if (!cell.trim()) throw new Error('Every cell needs a number.');
    const value = Number(cell.trim());
    if (!Number.isFinite(value) || Math.abs(value) > 20) throw new Error('Use finite numbers from −20 to 20.');
    return value;
  }));
  if (rows.length > 3 || rows[0].length > 3 || rows.some(row => row.length !== rows[0].length)) {
    throw new Error('Use a rectangular matrix with one to three rows and columns.');
  }
  return rows;
}
export function matrixProduct(left, right) {
  if (!left.length || !right.length || !left[0].length || !right[0].length || left.some(row => row.length !== left[0].length) || right.some(row => row.length !== right[0].length) || left[0].length !== right.length) {
    throw new Error('The left column count must equal the right row count; both matrices must be rectangular.');
  }
  return left.map(row => right[0].map((_, column) => dot(row, right.map(values => values[column]))));
}
export function productCell(left, right, row, column, terms) {
  const result = matrixProduct(left, right);
  if (!Number.isInteger(row) || !Number.isInteger(column) || !result[row] || column < 0 || column >= result[row].length || !Number.isInteger(terms) || terms < 0 || terms > right.length) throw new Error('Choose a valid output cell and term count.');
  const contributions = right.map((values, index) => ({
    index,
    left: left[row][index],
    right: values[column],
    product: left[row][index] * values[column]
  }));
  return {
    result,
    contributions,
    partial: contributions.slice(0, terms).reduce((sum, term) => sum + term.product, 0)
  };
}
export const tensorAxisNames = ['session', 'time', 'channel'];
export const tensorShape = [2, 2, 3];
export function measurementTensor(kind = 'ramp') {
  if (!['ramp', 'repeat', 'impulse'].includes(kind)) throw new Error('Unknown teaching dataset.');
  return Array.from({
    length: 2
  }, (_, session) => Array.from({
    length: 2
  }, (_, time) => Array.from({
    length: 3
  }, (_, channel) => {
    if (kind === 'repeat') return 10 * time + channel;
    if (kind === 'impulse') return session === 1 && time === 0 && channel === 2 ? 12 : 0;
    return 6 * session + 3 * time + channel;
  })));
}
export function reduceMeasurementTensor(tensor, axis) {
  if (!Number.isInteger(axis) || axis < 0 || axis > 2) throw new Error('Choose session, time or channel.');
  if (tensor.length !== 2 || tensor.some(session => session.length !== 2 || session.some(time => time.length !== 3 || time.some(value => !Number.isFinite(value))))) {
    throw new Error('The teaching tensor must have shape (2,2,3) and finite values.');
  }
  const remainingAxes = [0, 1, 2].filter(index => index !== axis);
  const shape = remainingAxes.map(index => tensorShape[index]);
  const cells = Array.from({
    length: shape[0]
  }, (_, row) => Array.from({
    length: shape[1]
  }, (_, column) => {
    const sources = Array.from({
      length: tensorShape[axis]
    }, (_, reducedIndex) => {
      const coordinate = [0, 0, 0];
      coordinate[axis] = reducedIndex;
      coordinate[remainingAxes[0]] = row;
      coordinate[remainingAxes[1]] = column;
      return {
        coordinate,
        value: tensor[coordinate[0]][coordinate[1]][coordinate[2]]
      };
    });
    const sum = sources.reduce((total, source) => total + source.value, 0);
    return {
      sources,
      sum,
      mean: sum / sources.length
    };
  }));
  return {
    shape,
    axisNames: remainingAxes.map(index => tensorAxisNames[index]),
    cells
  };
}
export function formatVectorNumber(value) {
  const rounded = Number(value.toFixed(3));
  return Object.is(rounded, -0) ? '0' : String(rounded);
}
