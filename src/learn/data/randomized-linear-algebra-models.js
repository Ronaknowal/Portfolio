// Bounded teaching models: at most 6×6 spectral fixtures, eight observations,
// and 32 trace probes. These are not general-purpose numerical routines.
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const normSquared = matrix => matrix.flat().reduce((sum, value) => sum + value * value, 0);
const zeros = (rows, columns) => Array.from({
  length: rows
}, () => Array(columns).fill(0));
const identity = size => zeros(size, size).map((row, index) => row.map((_, column) => Number(index === column)));
const transpose = matrix => matrix[0].map((_, column) => matrix.map(row => row[column]));
const multiply = (left, right) => {
  const columns = transpose(right);
  return left.map(row => columns.map(column => dot(row, column)));
};
const difference = (left, right) => left.map((row, index) => row.map((value, column) => value - right[index][column]));
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
}
function seededUniform(seed) {
  integer(seed, 1, 999, 'Seed');
  let state = seed;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) + 0.5) / 4294967296;
  };
}
function gaussianColumns(rows, columns, seed) {
  const uniform = seededUniform(seed);
  // Generate column first: increasing the width retains previous probes.
  const probes = Array.from({
    length: columns
  }, () => Array.from({
    length: rows
  }, () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform())));
  return Array.from({
    length: rows
  }, (_, row) => probes.map(column => column[row]));
}
function orthonormalColumns(matrix) {
  const columns = transpose(matrix);
  const scale = Math.max(...columns.map(column => Math.hypot(...column)));
  const basis = [];
  for (const original of columns) {
    let remainder = [...original];
    // Reorthogonalize to avoid retaining numerical copies of a direction.
    for (let pass = 0; pass < 2; pass += 1) {
      for (const direction of basis) {
        const coefficient = dot(direction, remainder);
        remainder = remainder.map((value, row) => value - coefficient * direction[row]);
      }
    }
    const length = Math.hypot(...remainder);
    if (length > 1e-11 * scale) basis.push(remainder.map(value => value / length));
  }
  return Array.from({
    length: matrix.length
  }, (_, row) => basis.map(column => column[row]));
}
function symmetricEigenvectors(matrix) {
  const size = matrix.length;
  const diagonalized = matrix.map(row => [...row]);
  const vectors = identity(size);
  const tolerance = 1e-13 * Math.max(1, Math.sqrt(normSquared(matrix)));
  for (let sweep = 0; sweep < 80; sweep += 1) {
    let largest = 0;
    for (let first = 0; first < size; first += 1) {
      for (let second = first + 1; second < size; second += 1) {
        const cross = diagonalized[first][second];
        largest = Math.max(largest, Math.abs(cross));
        if (Math.abs(cross) <= tolerance) continue;
        const firstDiagonal = diagonalized[first][first];
        const secondDiagonal = diagonalized[second][second];
        const angle = 0.5 * Math.atan2(2 * cross, secondDiagonal - firstDiagonal);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        for (let row = 0; row < size; row += 1) {
          if (row !== first && row !== second) {
            const firstEntry = diagonalized[row][first];
            const secondEntry = diagonalized[row][second];
            diagonalized[row][first] = diagonalized[first][row] = cosine * firstEntry - sine * secondEntry;
            diagonalized[row][second] = diagonalized[second][row] = sine * firstEntry + cosine * secondEntry;
          }
          const firstVector = vectors[row][first];
          const secondVector = vectors[row][second];
          vectors[row][first] = cosine * firstVector - sine * secondVector;
          vectors[row][second] = sine * firstVector + cosine * secondVector;
        }
        diagonalized[first][first] = cosine ** 2 * firstDiagonal - 2 * sine * cosine * cross + sine ** 2 * secondDiagonal;
        diagonalized[second][second] = sine ** 2 * firstDiagonal + 2 * sine * cosine * cross + cosine ** 2 * secondDiagonal;
        diagonalized[first][second] = diagonalized[second][first] = 0;
      }
    }
    if (largest <= tolerance) {
      const order = Array.from({
        length: size
      }, (_, index) => index).sort((left, right) => diagonalized[right][right] - diagonalized[left][left]);
      return {
        values: order.map(index => Math.max(0, diagonalized[index][index])),
        vectors: vectors.map(row => order.map(index => row[index]))
      };
    }
  }
  throw new Error('The bounded spectral fixture did not converge.');
}
export const sketchSpectra = {
  fast: {
    title: 'Fast decay',
    values: [12, 4, 1, 0.3, 0.08, 0.02]
  },
  slow: {
    title: 'Slow decay',
    values: [12, 10, 8, 6, 4, 2]
  },
  flat: {
    title: 'Equal singular values',
    values: [6, 6, 6, 6, 6, 6]
  },
  rankTwo: {
    title: 'Exactly rank two',
    values: [12, 4, 0, 0, 0, 0]
  },
  zero: {
    title: 'Zero matrix',
    values: [0, 0, 0, 0, 0, 0]
  }
};
function rotatedSpectrum(values) {
  function rotationBasis(offset) {
    const basis = identity(values.length);
    for (let step = 0; step < 9; step += 1) {
      const first = step % values.length;
      const second = (first + 1 + step % 3) % values.length;
      const angle = offset + 0.13 * step;
      const firstRow = [...basis[first]];
      const secondRow = [...basis[second]];
      basis[first] = firstRow.map((value, column) => Math.cos(angle) * value - Math.sin(angle) * secondRow[column]);
      basis[second] = secondRow.map((value, column) => Math.sin(angle) * firstRow[column] + Math.cos(angle) * value);
    }
    return basis;
  }
  return multiply(rotationBasis(0.31).map(row => row.map((value, column) => value * values[column])), transpose(rotationBasis(0.77)));
}
export function weightedProbeState(weights) {
  if (!Array.isArray(weights) || weights.length !== 3) throw new RangeError('Use three column weights.');
  weights.forEach(value => integer(value, -2, 2, 'Column weight'));
  const columns = [[3, 0], [0, 1], [3, 1]];
  const contributions = columns.map((column, index) => column.map(value => value * weights[index]));
  const output = contributions.reduce((sum, column) => sum.map((value, row) => value + column[row]), [0, 0]);
  const length = Math.hypot(...output);
  const direction = length === 0 ? null : output.map(value => value / length);
  const projected = columns.map(column => direction ? direction.map(value => value * dot(direction, column)) : [0, 0]);
  const residualSquared = normSquared(difference(columns, projected));
  return {
    weights: [...weights],
    columns,
    contributions,
    output,
    direction,
    projected,
    residualSquared
  };
}
export function spectralSketchState(preset, rank, oversampling, iterations, seed) {
  if (!Object.hasOwn(sketchSpectra, preset)) throw new RangeError('Unknown spectrum.');
  integer(rank, 1, 4, 'Target rank');
  integer(oversampling, 0, 6 - rank, 'Oversampling');
  integer(iterations, 0, 3, 'Subspace iterations');
  const spectrum = [...sketchSpectra[preset].values];
  const matrix = rotatedSpectrum(spectrum);
  const width = rank + oversampling;
  const omega = gaussianColumns(6, width, seed);
  let basis = orthonormalColumns(multiply(matrix, omega));
  for (let iteration = 0; iteration < iterations && basis[0].length; iteration += 1) {
    const rightBasis = orthonormalColumns(multiply(transpose(matrix), basis));
    basis = orthonormalColumns(multiply(matrix, rightBasis));
  }
  const observedRank = basis[0].length;
  let compressed = [];
  let projected = zeros(6, 6);
  let approximation = zeros(6, 6);
  let smallSpectrum = [];
  let smallVectors = [];
  if (observedRank) {
    compressed = multiply(transpose(basis), matrix);
    projected = multiply(basis, compressed);
    const decomposition = symmetricEigenvectors(multiply(compressed, transpose(compressed)));
    smallSpectrum = decomposition.values.map(Math.sqrt);
    smallVectors = decomposition.vectors;
    const retained = smallVectors.map(row => row.slice(0, rank));
    const smallApproximation = multiply(retained, multiply(transpose(retained), compressed));
    approximation = multiply(basis, smallApproximation);
  }
  const rangeErrorSquared = normSquared(difference(matrix, projected));
  const truncationErrorSquared = normSquared(difference(projected, approximation));
  const errorSquared = normSquared(difference(matrix, approximation));
  const floorSquared = spectrum.slice(rank).reduce((sum, value) => sum + value * value, 0);
  const totalSquared = normSquared(matrix);
  return {
    preset,
    rank,
    oversampling,
    iterations,
    seed,
    spectrum,
    matrix,
    omega,
    width,
    basis,
    observedRank,
    compressed,
    smallSpectrum,
    smallVectors,
    projected,
    approximation,
    rangeErrorSquared,
    truncationErrorSquared,
    errorSquared,
    floorSquared,
    totalSquared,
    relativeError: totalSquared ? Math.sqrt(errorSquared / totalSquared) : null,
    passes: 2 + 2 * iterations
  };
}
export const sketchObservations = [0, 1, 2, 3, 4, 5, 6, 20].map((x, index) => ({
  x,
  y: 2 + 0.5 * x + [0, 0.2, -0.1, 0.4, -0.3, 0.1, -0.2, 3][index]
}));
function fitLine(observations) {
  if (observations.length < 2) return null;
  const meanX = observations.reduce((sum, point) => sum + point.x, 0) / observations.length;
  const meanY = observations.reduce((sum, point) => sum + point.y, 0) / observations.length;
  const spread = observations.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
  if (spread === 0) return null;
  const slope = observations.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0) / spread;
  return {
    intercept: meanY - slope * meanX,
    slope
  };
}
export function rowSketchState(selectedRows) {
  if (!Array.isArray(selectedRows) || new Set(selectedRows).size !== selectedRows.length) {
    throw new RangeError('Choose distinct observation indices.');
  }
  selectedRows.forEach(index => integer(index, 0, 7, 'Observation index'));
  const observations = sketchObservations.map(point => ({
    ...point
  }));
  const selected = observations.filter((_, index) => selectedRows.includes(index));
  const fullFit = fitLine(observations);
  const sketchFit = fitLine(selected);
  const residual = (points, fit) => points.reduce((sum, point) => sum + (point.y - fit.intercept - fit.slope * point.x) ** 2, 0);
  const meanX = observations.reduce((sum, point) => sum + point.x, 0) / observations.length;
  const spread = observations.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
  const leverage = observations.map(point => 1 / observations.length + (point.x - meanX) ** 2 / spread);
  return {
    observations,
    selectedRows: [...selectedRows],
    fullFit,
    sketchFit,
    leverage,
    fullResidualSquared: residual(observations, fullFit),
    sketchResidualSquared: sketchFit ? residual(selected, sketchFit) : null,
    originalResidualSquared: sketchFit ? residual(observations, sketchFit) : null
  };
}
export const traceMatrices = {
  diagonal: {
    title: 'Diagonal: every sign probe is exact',
    matrix: [[4, 0], [0, 2]]
  },
  coupled: {
    title: 'Off-diagonal coupling',
    matrix: [[4, 1], [1, 2]]
  },
  zeroTrace: {
    title: 'Cancellation: trace equals zero',
    matrix: [[1, 2], [2, -1]]
  }
};
export function traceProbeState(preset, seed, count) {
  if (!Object.hasOwn(traceMatrices, preset)) throw new RangeError('Unknown trace matrix.');
  integer(count, 0, 32, 'Probe count');
  const uniform = seededUniform(seed);
  const matrix = traceMatrices[preset].matrix.map(row => [...row]);
  const exactTrace = matrix[0][0] + matrix[1][1];
  let total = 0;
  const samples = Array.from({
    length: count
  }, (_, index) => {
    const input = [uniform() < 0.5 ? -1 : 1, uniform() < 0.5 ? -1 : 1];
    const output = matrix.map(row => dot(row, input));
    const value = dot(input, output);
    total += value;
    return {
      count: index + 1,
      input,
      output,
      value,
      mean: total / (index + 1)
    };
  });
  const mean = count ? total / count : null;
  return {
    matrix,
    exactTrace,
    samples,
    mean,
    absoluteError: mean === null ? null : Math.abs(mean - exactTrace),
    variancePerProbe: 4 * matrix[0][1] ** 2
  };
}
