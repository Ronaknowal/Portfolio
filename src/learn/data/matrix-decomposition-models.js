// Bounded real 2-by-2 teaching models. These are not arbitrary matrix solvers.
export function multiply2(left, right) {
  return left.map(row => right[0].map((_, column) => row.reduce((sum, value, index) => sum + value * right[index][column], 0)));
}
export function transpose2(matrix) {
  return matrix[0].map((_, column) => matrix.map(row => row[column]));
}
export function apply2(matrix, vector) {
  return matrix.map(row => row.reduce((sum, value, index) => sum + value * vector[index], 0));
}
function assertFiniteValues(values) {
  if (!values.every(Number.isFinite)) throw new Error('Use finite numerical values.');
}
export const eliminationPresets = {
  coupled: {
    title: 'Coupled equations',
    matrix: [[2, 1], [4, 3]],
    target: [5, 11]
  },
  zeroPivot: {
    title: 'Zero first entry; invertible matrix',
    matrix: [[0, 2], [1, 3]],
    target: [4, 7]
  },
  singular: {
    title: 'Dependent equations',
    matrix: [[1, 2], [2, 4]],
    target: [3, 6]
  },
  inconsistent: {
    title: 'Dependent left sides; conflicting targets',
    matrix: [[1, 2], [2, 4]],
    target: [3, 7]
  }
};
export function eliminationTrace(presetName = 'coupled') {
  const preset = eliminationPresets[presetName];
  if (!preset) throw new Error('Choose an available elimination example.');
  const matrix = preset.matrix.map(row => [...row]);
  const target = [...preset.target];
  const upper = matrix.map(row => [...row]);
  const transformedTarget = [...target];
  const permutation = [[1, 0], [0, 1]];
  const lower = [[1, 0], [0, 1]];
  const steps = [];
  function record(message, stage, solution = null) {
    steps.push({
      message,
      stage,
      solution,
      augmented: upper.map((row, index) => [...row, transformedTarget[index]]),
      lower: lower.map(row => [...row]),
      permutation: permutation.map(row => [...row])
    });
  }
  record('The two rows are equations. Changes to a row must also change its target.', 'initial');
  if (Math.abs(upper[1][0]) > Math.abs(upper[0][0])) {
    [upper[0], upper[1]] = [upper[1], upper[0]];
    [transformedTarget[0], transformedTarget[1]] = [transformedTarget[1], transformedTarget[0]];
    [permutation[0], permutation[1]] = [permutation[1], permutation[0]];
    record('Swap both complete equations to use the larger available first-column pivot. P records this swap.', 'pivot');
  } else {
    record('The first row already has a largest-magnitude available pivot. P remains the identity.', 'pivot');
  }
  if (upper[0][0] === 0) {
    record('There is no nonzero pivot in the first column. This preset cannot have a unique two-variable solution.', 'singular');
    return {
      matrix,
      target,
      steps,
      lower,
      upper,
      permutation,
      solution: null
    };
  }
  const multiplier = upper[1][0] / upper[0][0];
  lower[1][0] = multiplier;
  upper[1][1] -= multiplier * upper[0][1];
  upper[1][0] = 0;
  transformedTarget[1] -= multiplier * transformedTarget[0];
  record(`Subtract ${multiplier} times row 1 from row 2, including its target. L stores the multiplier needed to reconstruct the original system.`, 'eliminate');
  if (upper[1][1] === 0) {
    record(transformedTarget[1] === 0 ? 'The last equation is 0 = 0: one constraint is redundant and the solution is not unique.' : `The last equation is 0 = ${transformedTarget[1]}: these targets are inconsistent.`, 'singular');
    return {
      matrix,
      target,
      steps,
      lower,
      upper,
      permutation,
      solution: null
    };
  }
  const second = transformedTarget[1] / upper[1][1];
  record(`The last equation contains only x₂. Divide by its coefficient: x₂ = ${second}.`, 'back-bottom', [null, second]);
  const first = (transformedTarget[0] - upper[0][1] * second) / upper[0][0];
  const solution = [first, second];
  record(`Substitute x₂ into the first equation: x₁ = ${first}. Check the original equations, not only the transformed rows.`, 'back-top', solution);
  return {
    matrix,
    target,
    steps,
    lower,
    upper,
    permutation,
    solution
  };
}
export function qrGeometry(secondColumn = [1, 0]) {
  if (!Array.isArray(secondColumn) || secondColumn.length !== 2) throw new Error('Use a two-component column.');
  assertFiniteValues(secondColumn);
  if (secondColumn.some(value => Math.abs(value) > 2)) throw new Error('This view supports coordinates from -2 to 2.');
  const firstColumn = [1, 1];
  const firstNorm = Math.sqrt(2);
  const firstUnit = firstColumn.map(value => value / firstNorm);
  const overlap = firstUnit.reduce((sum, value, index) => sum + value * secondColumn[index], 0);
  const projection = firstUnit.map(value => value * overlap);
  const perpendicular = secondColumn.map((value, index) => value - projection[index]);
  const remainingNorm = Math.hypot(...perpendicular);
  const dependent = remainingNorm < 1e-12;
  const secondUnit = dependent ? null : perpendicular.map(value => value / remainingNorm);
  const orthogonal = secondUnit ? [[firstUnit[0], secondUnit[0]], [firstUnit[1], secondUnit[1]]] : null;
  const triangular = [[firstNorm, overlap], [0, dependent ? 0 : remainingNorm]];
  return {
    firstColumn,
    secondColumn: [...secondColumn],
    firstUnit,
    overlap,
    projection,
    perpendicular,
    remainingNorm: dependent ? 0 : remainingNorm,
    dependent,
    secondUnit,
    orthogonal,
    triangular
  };
}
export function covarianceGeometry(correlation = 0.5) {
  assertFiniteValues([correlation]);
  if (correlation < -1 || correlation > 1) throw new Error('Correlation must be between -1 and 1.');
  const secondIndependentWeight = Math.sqrt(Math.max(0, 1 - correlation * correlation));
  const lower = [[2, 0], [correlation, secondIndependentWeight]];
  const covariance = [[4, 2 * correlation], [2 * correlation, 1]];
  const circle = Array.from({
    length: 65
  }, (_, index) => {
    const angle = index * 2 * Math.PI / 64;
    return [Math.cos(angle), Math.sin(angle)];
  });
  return {
    correlation,
    lower,
    covariance,
    positiveDefinite: Math.abs(correlation) < 1,
    determinant: 4 * (1 - correlation * correlation),
    circle,
    transformed: circle.map(point => apply2(lower, point))
  };
}
function rotation(degrees) {
  const radians = degrees * Math.PI / 180;
  return [[Math.cos(radians), -Math.sin(radians)], [Math.sin(radians), Math.cos(radians)]];
}
export function svdGeometry({
  inputAngle = 30,
  outputAngle = -20,
  smallerScale = 1,
  vectorAngle = 45,
  retained = 2
} = {}) {
  assertFiniteValues([inputAngle, outputAngle, smallerScale, vectorAngle, retained]);
  if (smallerScale < 0 || smallerScale > 3 || ![0, 1, 2].includes(retained)) throw new Error('Choose a scale from 0 to 3 and a rank budget of 0, 1 or 2.');
  const right = rotation(inputAngle);
  const left = rotation(outputAngle);
  const diagonal = [[3, 0], [0, smallerScale]];
  const rightTranspose = transpose2(right);
  const matrix = multiply2(multiply2(left, diagonal), rightTranspose);
  const truncatedDiagonal = [[retained >= 1 ? 3 : 0, 0], [0, retained >= 2 ? smallerScale : 0]];
  const approximation = multiply2(multiply2(left, truncatedDiagonal), rightTranspose);
  const radians = vectorAngle * Math.PI / 180;
  const input = [Math.cos(radians), Math.sin(radians)];
  const rotated = apply2(rightTranspose, input);
  const scaled = apply2(diagonal, rotated);
  const output = apply2(left, scaled);
  const approximationOutput = apply2(approximation, input);
  const discarded = [3, smallerScale].slice(retained);
  const frobeniusError = Math.hypot(...discarded);
  const spectralError = discarded[0] ?? 0;
  const circle = Array.from({
    length: 65
  }, (_, index) => {
    const angle = index * 2 * Math.PI / 64;
    return [Math.cos(angle), Math.sin(angle)];
  });
  return {
    left,
    right,
    diagonal,
    matrix,
    approximation,
    input,
    rotated,
    scaled,
    output,
    approximationOutput,
    singularValues: [3, smallerScale],
    rank: smallerScale === 0 ? 1 : 2,
    retained,
    frobeniusError,
    spectralError,
    circle,
    transformed: circle.map(point => apply2(matrix, point)),
    truncated: circle.map(point => apply2(approximation, point))
  };
}
