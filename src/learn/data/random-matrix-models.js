// Bounded numerical teaching models; simulated spectra are not measured data.
function integer(value, name, lower, upper) {
  if (!Number.isInteger(value) || value < lower || value > upper) {
    throw new RangeError(name + ' is outside the supported integer range.');
  }
}
export function formatRandomMatrix(value, digits = 3) {
  if (!Number.isFinite(value)) return value === Infinity ? 'unbounded' : 'undefined';
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
export function randomMatrixGenerator(seed) {
  integer(seed, 'Seed', 1, 2147483647);
  let state = seed >>> 0;
  function uniform() {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) + 0.5) / 4294967296;
  }
  return {
    uniform,
    normal: () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform())
  };
}
export function symmetricRandomMatrixEigen(matrix) {
  const size = matrix.length;
  integer(size, 'Matrix size', 1, 48);
  if (matrix.some(row => !Array.isArray(row) || row.length !== size || row.some(value => !Number.isFinite(value) || Math.abs(value) > 1e6))) {
    throw new RangeError('Use a finite square matrix with entries bounded by 1e6.');
  }
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      if (matrix[row][column] !== matrix[column][row]) throw new RangeError('The matrix must be symmetric.');
    }
  }
  const work = matrix.map(row => [...row]);
  const vectors = Array.from({
    length: size
  }, (_, row) => Array.from({
    length: size
  }, (_, column) => Number(row === column)));
  const scale = Math.max(1, ...matrix.flat().map(Math.abs));
  let converged = size === 1;
  for (let sweep = 0; sweep < 60 && !converged; sweep++) {
    let largest = 0;
    for (let left = 0; left < size; left++) {
      for (let right = left + 1; right < size; right++) {
        const off = work[left][right];
        largest = Math.max(largest, Math.abs(off));
        if (Math.abs(off) < 1e-13 * scale) continue;
        const ratio = (work[right][right] - work[left][left]) / (2 * off);
        const tangent = ratio === 0 ? 1 : Math.sign(ratio) / (Math.abs(ratio) + Math.hypot(1, ratio));
        const cosine = 1 / Math.hypot(1, tangent);
        const sine = tangent * cosine;
        work[left][left] -= tangent * off;
        work[right][right] += tangent * off;
        work[left][right] = work[right][left] = 0;
        for (let index = 0; index < size; index++) {
          if (index !== left && index !== right) {
            const oldLeft = work[index][left];
            const oldRight = work[index][right];
            work[index][left] = work[left][index] = cosine * oldLeft - sine * oldRight;
            work[index][right] = work[right][index] = sine * oldLeft + cosine * oldRight;
          }
          const oldVector = vectors[index][left];
          vectors[index][left] = cosine * oldVector - sine * vectors[index][right];
          vectors[index][right] = sine * oldVector + cosine * vectors[index][right];
        }
      }
    }
    converged = largest < 1e-11 * scale;
  }
  if (!converged) throw new RangeError('The eigensolver did not converge within its teaching-model limit.');
  const order = Array.from({
    length: size
  }, (_, index) => index).sort((left, right) => work[left][left] - work[right][right]);
  return {
    values: order.map(index => work[index][index]),
    vectors: order.map(index => vectors.map(row => row[index]))
  };
}
export function randomMatrixSpectrum({
  rows = 64,
  columns = 16,
  seed = 7,
  law = 'gaussian',
  centered = false,
  spike = 1,
  duplicate = false
} = {}) {
  integer(rows, 'Rows', 4, 192);
  integer(columns, 'Columns', 2, 48);
  if (!['gaussian', 'sign'].includes(law) || typeof centered !== 'boolean' || typeof duplicate !== 'boolean') throw new RangeError('Select a supported law and boolean convention.');
  if (!Number.isFinite(spike) || spike < 1 || spike > 8) throw new RangeError('Use a population spike from 1 to 8.');
  const generator = randomMatrixGenerator(seed);
  const data = Array.from({
    length: rows
  }, () => Array.from({
    length: columns
  }, (_, column) => {
    const value = law === 'gaussian' ? generator.normal() : generator.uniform() < .5 ? -1 : 1;
    return value * (column === 0 ? Math.sqrt(spike) : 1);
  }));
  if (duplicate) data.forEach(row => {
    row[columns - 1] = row[0];
  });
  const means = Array.from({
    length: columns
  }, (_, column) => data.reduce((sum, row) => sum + row[column], 0) / rows);
  const active = data.map(row => row.map((value, column) => value - (centered ? means[column] : 0)));
  const denominator = centered ? rows - 1 : rows;
  const covariance = Array.from({
    length: columns
  }, () => Array(columns).fill(0));
  for (let left = 0; left < columns; left++) {
    for (let right = left; right < columns; right++) {
      const value = active.reduce((sum, row) => sum + row[left] * row[right], 0) / denominator;
      covariance[left][right] = covariance[right][left] = value;
    }
  }
  const eigen = symmetricRandomMatrixEigen(covariance);
  const tolerance = 1e-9 * Math.max(1, eigen.values.at(-1));
  if (eigen.values[0] < -tolerance) throw new RangeError('A Gram matrix produced an invalid negative spectrum.');
  const values = eigen.values.map(value => Math.abs(value) < tolerance ? 0 : value);
  return {
    rows,
    columns,
    seed,
    law,
    centered,
    spike,
    duplicate,
    data,
    covariance,
    means,
    denominator,
    gamma: columns / denominator,
    values,
    zeroCount: values.filter(value => value === 0).length,
    meanEigenvalue: values.reduce((sum, value) => sum + value, 0) / columns,
    secondMoment: values.reduce((sum, value) => sum + value * value, 0) / columns,
    largest: values.at(-1),
    alignment: values.at(-1) - values.at(-2) > tolerance ? eigen.vectors.at(-1)[0] ** 2 : null
  };
}
export function marchenkoPastur(gamma, variance = 1) {
  if (!Number.isFinite(gamma) || gamma < .02 || gamma > 20 || !Number.isFinite(variance) || variance < .01 || variance > 100) throw new RangeError('Use supported positive aspect ratio and variance.');
  return {
    gamma,
    variance,
    lower: variance * (1 - Math.sqrt(gamma)) ** 2,
    upper: variance * (1 + Math.sqrt(gamma)) ** 2,
    atom: Math.max(0, 1 - 1 / gamma)
  };
}
export function mpIntervalMass(gamma, lower, upper, variance = 1) {
  const law = marchenkoPastur(gamma, variance);
  if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower > upper) throw new RangeError('Use finite ordered interval endpoints.');
  const start = Math.max(lower, law.lower);
  const end = Math.min(upper, law.upper);
  if (start >= end) return 0;
  const width = law.upper - law.lower;
  const angle = value => Math.asin(Math.sqrt(Math.max(0, Math.min(1, (value - law.lower) / width))));
  // An antiderivative after lambda = a + (b-a) sin²(theta).
  // atan2 retains the narrow endpoint contribution when gamma is close to one.
  const root = Math.sqrt(gamma);
  const primitive = theta => ((1 + gamma) * theta + root * Math.sin(2 * theta) - Math.abs(1 - gamma) * Math.atan2((1 + root) * Math.sin(theta), Math.abs(1 - root) * Math.cos(theta))) / (Math.PI * gamma);
  return Math.max(0, primitive(angle(end)) - primitive(angle(start)));
}
export function randomMatrixBins(values, gamma, count = 10) {
  integer(count, 'Bins', 4, 24);
  if (!Array.isArray(values) || values.length === 0 || values.some(value => !Number.isFinite(value) || value < 0)) throw new RangeError('Use a nonempty nonnegative spectrum.');
  const law = marchenkoPastur(gamma);
  const maximum = Math.max(law.upper, ...values) * 1.03;
  return Array.from({
    length: count
  }, (_, index) => {
    const lower = maximum * index / count;
    const upper = maximum * (index + 1) / count;
    return {
      lower,
      upper,
      observed: values.filter(value => value > 0 && value >= lower && (index === count - 1 ? value <= upper : value < upper)).length / values.length,
      theoretical: mpIntervalMass(gamma, lower, upper)
    };
  });
}
export function spikeLimits(gamma, population) {
  if (!Number.isFinite(gamma) || gamma <= 0 || gamma >= 1 || !Number.isFinite(population) || population < 1 || population > 8) throw new RangeError('This spike model requires 0 < gamma < 1 and 1 <= population <= 8.');
  const threshold = 1 + Math.sqrt(gamma);
  const separated = population > threshold;
  return {
    threshold,
    sample: separated ? population * (1 + gamma / (population - 1)) : (1 + Math.sqrt(gamma)) ** 2,
    alignment: separated ? (1 - gamma / (population - 1) ** 2) / (1 + gamma / (population - 1)) : 0
  };
}
export function gaussianSpectrumBound(rows, columns, delta) {
  integer(rows, 'Rows', 2, 10000000);
  integer(columns, 'Columns', 1, rows);
  if (!Number.isFinite(delta) || delta <= 0 || delta >= 1) throw new RangeError('Use a failure probability strictly between zero and one.');
  const margin = Math.sqrt(2 * (Math.log(2) - Math.log(delta)));
  const lower = Math.max(0, Math.sqrt(rows) - Math.sqrt(columns) - margin);
  const upper = Math.sqrt(rows) + Math.sqrt(columns) + margin;
  return {
    margin,
    lower: lower ** 2 / rows,
    upper: upper ** 2 / rows
  };
}
export function wignerSpectrum(size = 32, seed = 7, law = 'gaussian') {
  integer(size, 'Size', 2, 48);
  if (!['gaussian', 'sign'].includes(law)) throw new RangeError('Select Gaussian or sign entries.');
  const generator = randomMatrixGenerator(seed);
  const matrix = Array.from({
    length: size
  }, () => Array(size).fill(0));
  for (let row = 0; row < size; row++) {
    for (let column = row; column < size; column++) {
      const value = law === 'gaussian' ? generator.normal() * Math.sqrt((row === column ? 2 : 1) / size) : row === column ? 0 : (generator.uniform() < .5 ? -1 : 1) / Math.sqrt(size);
      matrix[row][column] = matrix[column][row] = value;
    }
  }
  const eigen = symmetricRandomMatrixEigen(matrix);
  return {
    size,
    seed,
    law,
    matrix,
    values: eigen.values,
    secondMoment: eigen.values.reduce((sum, value) => sum + value ** 2, 0) / size
  };
}
export function twoLevelSpectrum(offset, difference, coupling) {
  if ([offset, difference, coupling].some(value => !Number.isFinite(value) || Math.abs(value) > 100)) throw new RangeError('Use finite two-level parameters bounded by 100.');
  const radius = Math.hypot(difference, coupling);
  return {
    lower: offset - radius,
    upper: offset + radius,
    gap: 2 * radius
  };
}
const calibrationCache = new Map();
export function matrixNullCalibration(nullModel = 'iid', observedModel = 'iid') {
  if (![nullModel, observedModel].every(value => ['iid', 'duplicate', 'spike'].includes(value)) || nullModel === 'spike') throw new RangeError('Choose a supported null and observed model.');
  if (!calibrationCache.has(nullModel)) {
    calibrationCache.set(nullModel, Array.from({
      length: 59
    }, (_, index) => randomMatrixSpectrum({
      rows: 40,
      columns: 12,
      seed: 2000 + index * 7919,
      duplicate: nullModel === 'duplicate'
    }).largest));
  }
  const maxima = calibrationCache.get(nullModel);
  const observed = randomMatrixSpectrum({
    rows: 40,
    columns: 12,
    seed: 9381,
    duplicate: observedModel === 'duplicate',
    spike: observedModel === 'spike' ? 3 : 1
  }).largest;
  const exceedances = maxima.filter(value => value >= observed).length;
  return {
    nullModel,
    observedModel,
    maxima: [...maxima],
    observed,
    exceedances,
    rankScore: (1 + exceedances) / 60,
    bulkEdge: (1 + Math.sqrt(12 / 40)) ** 2
  };
}
