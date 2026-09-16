/** Bounded, dependency-free mathematical models for the ICA lesson.
 *
 * Every quantity the page states about the constructed four-state mixture is
 * computed here from the shared fixture, never typed as a drawing constant.
 * Geometry is a mathematical claim: the eigen-decomposition, the whitener and
 * the fixed-point step all live here so scripts/verify-ica-models.mjs can check
 * them against independent oracles. Bad input is refused, never substituted.
 */

const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${name} must be a finite number.`);
  return value;
};

const isPair = value => Array.isArray(value) && value.length === 2;

const checkVector = (vector, name) => {
  if (!isPair(vector)) throw new Error(`${name} must be a pair of numbers.`);
  vector.forEach((value, index) => finite(value, `${name}[${index}]`));
  return vector;
};

const checkMatrix = (matrix, name) => {
  if (!isPair(matrix) || !matrix.every(isPair)) throw new Error(`${name} must be a 2 by 2 matrix.`);
  matrix.forEach((row, i) => row.forEach((value, j) => finite(value, `${name}[${i}][${j}]`)));
  return matrix;
};

/** Sensor recipe of the hand model: sensor i receives row i of A. */
export const MIXING = [[2, 1], [1, 2]];

/** The four equiprobable source states, in the packet's declared order. */
export const SOURCE_STATES = [
  { id: 'A', source: [-1, -1] },
  { id: 'B', source: [-1, 1] },
  { id: 'C', source: [1, -1] },
  { id: 'D', source: [1, 1] },
];

export const STATE_PROBABILITY = 1 / 4;

export function applyMatrix(matrix, vector) {
  checkMatrix(matrix, 'Matrix');
  checkVector(vector, 'Vector');
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
  ];
}

export function multiplyMatrices(left, right) {
  checkMatrix(left, 'Left matrix');
  checkMatrix(right, 'Right matrix');
  return [0, 1].map(i => [0, 1].map(j => left[i][0] * right[0][j] + left[i][1] * right[1][j]));
}

export function invert2(matrix) {
  checkMatrix(matrix, 'Matrix');
  const determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  if (Math.abs(determinant) < 1e-12) throw new Error('A singular mixing matrix has no inverse; this model needs an invertible square mixture.');
  return [
    [matrix[1][1] / determinant, -matrix[0][1] / determinant],
    [-matrix[1][0] / determinant, matrix[0][0] / determinant],
  ];
}

/** Second moments of a finite equiprobable set of two-dimensional rows. */
export function covarianceOfRows(rows) {
  if (!Array.isArray(rows) || rows.length < 2 || rows.length > 4096) throw new Error('Use between 2 and 4096 rows.');
  rows.forEach((row, index) => checkVector(row, `Row ${index}`));
  const n = rows.length;
  const mean = [0, 1].map(axis => rows.reduce((total, row) => total + row[axis], 0) / n);
  const entry = (i, j) => rows.reduce((total, row) => total + (row[i] - mean[i]) * (row[j] - mean[j]), 0) / n;
  return { mean, covariance: [[entry(0, 0), entry(0, 1)], [entry(1, 0), entry(1, 1)]] };
}

/** Closed-form symmetric eigen-decomposition, eigenvalues descending.
 *  The diagonal branch must return two different axes; a helper that returns
 *  one direction twice would silently draw a family of ellipses as a line.
 */
export function symmetricEigen2(matrix) {
  checkMatrix(matrix, 'Matrix');
  const [[a, b], [c, d]] = matrix;
  if (Math.abs(b - c) > 1e-12) throw new Error('Eigen-decomposition here expects a symmetric matrix.');
  const half = (a + d) / 2;
  const spread = Math.hypot((a - d) / 2, b);
  const values = [half + spread, half - spread];
  let vectors;
  if (Math.abs(b) > 1e-14) {
    vectors = values.map(value => {
      const raw = [b, value - a];
      const length = Math.hypot(raw[0], raw[1]);
      return [raw[0] / length, raw[1] / length];
    });
  } else {
    vectors = a >= d ? [[1, 0], [0, 1]] : [[0, 1], [1, 0]];
  }
  // One deterministic sign per direction: first nonzero entry positive.
  vectors = vectors.map(vector => {
    const lead = Math.abs(vector[0]) > 1e-14 ? vector[0] : vector[1];
    return lead < 0 ? [-vector[0], -vector[1]] : vector;
  });
  return { values, vectors };
}

/** K = D^(-1/2) V^T with the larger eigenvalue first, matching section 3. */
export function whitenerFromCovariance(covariance) {
  const { values, vectors } = symmetricEigen2(covariance);
  if (values.some(value => value <= 1e-12)) {
    throw new Error('A zero or negative eigenvalue cannot be whitened; a constant or redundant channel supplies no direction.');
  }
  return {
    values,
    vectors,
    whitener: vectors.map((vector, index) => vector.map(entry => entry / Math.sqrt(values[index]))),
  };
}

/** The full chain of section 3 for the constructed four-state distribution. */
export function whitenedFixture(mixing = MIXING) {
  checkMatrix(mixing, 'Mixing matrix');
  const observed = SOURCE_STATES.map(state => ({ ...state, observed: applyMatrix(mixing, state.source) }));
  const { mean, covariance } = covarianceOfRows(observed.map(state => state.observed));
  const { values, vectors, whitener } = whitenerFromCovariance(covariance);
  const centred = observed.map(state => [state.observed[0] - mean[0], state.observed[1] - mean[1]]);
  const whitened = centred.map(row => applyMatrix(whitener, row));
  const recovery = [[Math.SQRT1_2, Math.SQRT1_2], [Math.SQRT1_2, -Math.SQRT1_2]];
  const states = observed.map((state, index) => ({
    ...state,
    whitened: whitened[index],
    recovered: applyMatrix(recovery, whitened[index]),
  }));
  return {
    states,
    mean,
    covariance,
    eigenvalues: values,
    eigenvectors: vectors,
    whitener,
    recovery,
    whitenedCovariance: covarianceOfRows(whitened).covariance,
    unmixing: multiplyMatrices(recovery, whitener),
  };
}

export const CONTRASTS = {
  // g = G', with its own derivative. The cube choice makes section 5's trace
  // exact; the linear choice is practice 3's broken program, kept so the model
  // can demonstrate the failure rather than describe it.
  cube: { key: 'cube', label: 'g(u) = u³', g: u => u ** 3, gPrime: u => 3 * u ** 2 },
  tanh: { key: 'tanh', label: 'g(u) = tanh u', g: u => Math.tanh(u), gPrime: u => 1 - Math.tanh(u) ** 2 },
  linear: { key: 'linear', label: 'g(u) = u', g: u => u, gPrime: () => 1 },
};

/** One FastICA update on a finite whitened fixture. */
export function fastIcaStep(points, direction, contrastKey = 'cube') {
  if (!Array.isArray(points) || points.length < 2 || points.length > 4096) throw new Error('Use between 2 and 4096 whitened observations.');
  points.forEach((point, index) => checkVector(point, `Whitened point ${index}`));
  checkVector(direction, 'Direction');
  const contrast = CONTRASTS[contrastKey];
  if (!contrast) throw new Error(`Unknown contrast "${contrastKey}". Use cube, tanh or linear.`);
  const norm = Math.hypot(direction[0], direction[1]);
  if (Math.abs(norm - 1) > 1e-9) throw new Error('The search direction must be a unit vector; normalize it before the update.');
  const n = points.length;
  const rows = points.map((point, index) => {
    const projection = point[0] * direction[0] + point[1] * direction[1];
    return {
      index,
      point,
      projection,
      cube: contrast.g(projection),
      weighted: [point[0] * contrast.g(projection), point[1] * contrast.g(projection)],
      derivative: contrast.gPrime(projection),
    };
  });
  const weightedMean = [0, 1].map(axis => rows.reduce((total, row) => total + row.weighted[axis], 0) / n);
  const derivativeMean = rows.reduce((total, row) => total + row.derivative, 0) / n;
  const correction = direction.map(value => derivativeMean * value);
  const raw = [weightedMean[0] - correction[0], weightedMean[1] - correction[1]];
  const rawNorm = Math.hypot(raw[0], raw[1]);
  if (rawNorm < 1e-12) {
    throw new Error('The update vector is zero, so normalization is undefined: a linear contrast on whitened data gives exactly this, because variance has no preferred direction.');
  }
  const next = raw.map(value => value / rawNorm);
  const alignment = next[0] * direction[0] + next[1] * direction[1];
  const signAligned = alignment < 0 ? next.map(value => -value) : next.slice();
  return {
    contrast,
    rows,
    weightedMean,
    derivativeMean,
    correction,
    raw,
    rawNorm,
    next,
    signAligned,
    alignment,
    convergence: 1 - Math.abs(alignment),
  };
}

/** Deflation: remove the components already found, then renormalize. */
export function deflate(vector, found) {
  checkVector(vector, 'Vector');
  if (!Array.isArray(found) || found.length > 2) throw new Error('At most two previous directions exist in this two-dimensional fixture.');
  found.forEach((previous, index) => {
    checkVector(previous, `Previous direction ${index}`);
    if (Math.abs(Math.hypot(previous[0], previous[1]) - 1) > 1e-9) throw new Error('Previous directions must be unit vectors.');
  });
  const residual = found.reduce((current, previous) => {
    const projection = current[0] * previous[0] + current[1] * previous[1];
    return [current[0] - projection * previous[0], current[1] - projection * previous[1]];
  }, vector.slice());
  const length = Math.hypot(residual[0], residual[1]);
  if (length < 1e-12) throw new Error('The residual after removing earlier directions is zero; diagnose that before normalizing.');
  return { residual, unit: residual.map(value => value / length), length };
}

export const SOURCE_FAMILIES = {
  binary: { key: 'binary', label: 'Binary ±1', kurtosis: -2, curveMax: 2.2, continuous: false },
  laplace: { key: 'laplace', label: 'Standardized Laplace', kurtosis: 3, curveMax: 3.3, continuous: true },
  gaussian: { key: 'gaussian', label: 'Standard Gaussian', kurtosis: 0, curveMax: 1, continuous: true },
};

const familyOf = key => {
  const family = SOURCE_FAMILIES[key];
  if (!family) throw new Error(`Unknown source family "${key}". Use binary, laplace or gaussian.`);
  return family;
};

const checkAngle = angle => {
  finite(angle, 'Angle');
  if (angle < 0 || angle > 180) throw new Error('Enter an angle in degrees from 0 to 180.');
  return angle;
};

/** kappa(y) = kappa_s (cos^4 t + sin^4 t) for two equal independent sources. */
export function projectionKurtosis(familyKey, angleDegrees) {
  const family = familyOf(familyKey);
  const radians = (checkAngle(angleDegrees) * Math.PI) / 180;
  const shape = Math.cos(radians) ** 4 + Math.sin(radians) ** 4;
  return family.kurtosis * shape;
}

const rotate = (angleDegrees, point) => {
  const radians = (angleDegrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return [cos * point[0] + sin * point[1], -sin * point[0] + cos * point[1]];
};

/** The exact population picture the rotation investigation shows. */
export function rotationModel(familyKey, angleDegrees, samples = 361) {
  const family = familyOf(familyKey);
  const angle = checkAngle(angleDegrees);
  if (!Number.isInteger(samples) || samples < 3 || samples > 361) throw new Error('Use between 3 and 361 curve samples.');
  const kurtosis = projectionKurtosis(familyKey, angle);
  const support = family.continuous ? [] : SOURCE_STATES.map(state => ({
    id: state.id,
    source: state.source,
    projected: rotate(angle, state.source),
    probability: STATE_PROBABILITY,
  }));
  const contours = family.continuous ? [1, 2, 3].map(level => ({
    level,
    points: family.key === 'laplace'
      ? [[level, 0], [0, level], [-level, 0], [0, -level]].map(point => rotate(angle, point))
      : Array.from({ length: 64 }, (unused, index) => {
        const t = (2 * Math.PI * index) / 64;
        return rotate(angle, [level * Math.cos(t), level * Math.sin(t)]);
      }),
  })) : [];
  const curve = Array.from({ length: samples }, (unused, index) => {
    const theta = (180 * index) / (samples - 1);
    const value = projectionKurtosis(familyKey, theta);
    return { angle: theta, kurtosis: value, magnitude: Math.abs(value) };
  });
  return {
    family,
    angle,
    kurtosis,
    magnitude: Math.abs(kurtosis),
    companionKurtosis: projectionKurtosis(familyKey, angle),
    variance: 1,
    covariance: [[1, 0], [0, 1]],
    support,
    contours,
    curve,
    direction: [Math.cos((angle * Math.PI) / 180), Math.sin((angle * Math.PI) / 180)],
    companionDirection: [-Math.sin((angle * Math.PI) / 180), Math.cos((angle * Math.PI) / 180)],
  };
}

/** Grade a rotation prediction against the inputs committed with it. */
export function gradeRotation(committed) {
  if (!committed || typeof committed !== 'object') throw new Error('A committed rotation snapshot is required.');
  const { family, oldAngle, newAngle, prediction } = committed;
  if (!['smaller', 'same', 'larger'].includes(prediction)) throw new Error('Prediction must be smaller, same or larger.');
  const before = Math.abs(projectionKurtosis(family, oldAngle));
  const after = Math.abs(projectionKurtosis(family, newAngle));
  const difference = after - before;
  const actual = Math.abs(difference) <= 1e-10 ? 'same' : difference > 0 ? 'larger' : 'smaller';
  return { before, after, difference, actual, correct: actual === prediction };
}

const KEEP_SETS = { both: [0, 1], first: [0], second: [1], none: [] };

export const KEEP_LABELS = {
  both: 'Keep both components',
  first: 'Keep source 1 only',
  second: 'Keep source 2 only',
  none: 'Keep neither component',
};

/** The learner may enter amplitudes in [-4, 4]; a compensated rescaling by up
 *  to 4 can legitimately carry an active amplitude to 16, so the model's own
 *  bound is wider than the entry bound the investigation enforces.
 */
export const ENTERED_AMPLITUDE_LIMIT = 4;
export const MODEL_AMPLITUDE_LIMIT = 16;

const checkAmplitude = (value, name, limit = MODEL_AMPLITUDE_LIMIT) => {
  finite(value, name);
  if (value < -limit || value > limit) throw new Error(`${name} must lie between −${limit} and ${limit}.`);
  return value;
};

export function checkEnteredAmplitude(value, name = 'Amplitude') {
  return checkAmplitude(value, name, ENTERED_AMPLITUDE_LIMIT);
}

/** Contributions, the reconstruction from a keep-set, and what it removed. */
export function contributionModel({ mixing = MIXING, sources, keep = 'both' }) {
  checkMatrix(mixing, 'Mixing matrix');
  checkVector(sources, 'Source amplitudes');
  sources.forEach((value, index) => checkAmplitude(value, `Source ${index + 1} amplitude`));
  const kept = KEEP_SETS[keep];
  if (!kept) throw new Error(`Unknown keep-set "${keep}". Use both, first, second or none.`);
  const columns = [0, 1].map(j => [mixing[0][j], mixing[1][j]]);
  const contributions = columns.map((column, j) => ({
    index: j,
    column,
    amplitude: sources[j],
    contribution: column.map(value => value * sources[j]),
    columnNorm: Math.hypot(column[0], column[1]),
    varianceShare: (column[0] ** 2 + column[1] ** 2) * sources[j] ** 2,
  }));
  const observed = [0, 1].map(sensor => contributions.reduce((total, item) => total + item.contribution[sensor], 0));
  const retained = [0, 1].map(sensor => kept.reduce((total, j) => total + contributions[j].contribution[sensor], 0));
  const removed = observed.map((value, sensor) => value - retained[sensor]);
  return { mixing, sources: sources.slice(), keep, keptIndices: kept, columns, contributions, observed, retained, removed };
}

/** Grade a numeric sensor prediction against the committed amplitudes. */
export function gradeContribution(committed, tolerance = 1e-9) {
  if (!committed || typeof committed !== 'object') throw new Error('A committed contribution snapshot is required.');
  const { sources, keep, prediction, sensor = 0, mixing = MIXING } = committed;
  finite(prediction, 'Prediction');
  if (sensor !== 0 && sensor !== 1) throw new Error('Predict sensor 1 or sensor 2.');
  const model = contributionModel({ mixing, sources, keep });
  const actual = model.retained[sensor];
  // Preserve the inclusive decimal boundary after floating-point subtraction.
  const roundingSlack = 4 * Number.EPSILON * Math.max(1, Math.abs(prediction), Math.abs(actual));
  return { model, actual, prediction, difference: prediction - actual, correct: Math.abs(prediction - actual) <= tolerance + roundingSlack };
}

/** Rescale one source, optionally compensating its mixing column by 1/c. */
export function rescaleComponent({ mixing = MIXING, sources, index, scale, compensate = true }) {
  checkMatrix(mixing, 'Mixing matrix');
  checkVector(sources, 'Source amplitudes');
  sources.forEach((value, position) => checkEnteredAmplitude(value, `Source ${position + 1} amplitude`));
  if (index !== 0 && index !== 1) throw new Error('Choose component 1 or component 2.');
  finite(scale, 'Scale');
  if (scale === 0) throw new Error('A zero scale cannot be compensated: dividing the mixing column by zero is undefined.');
  if (Math.abs(scale) < 0.25 || Math.abs(scale) > 4) throw new Error('Use a scale with magnitude between 0.25 and 4.');
  const nextSources = sources.slice();
  nextSources[index] *= scale;
  const nextMixing = mixing.map(row => row.slice());
  if (compensate) {
    nextMixing[0][index] /= scale;
    nextMixing[1][index] /= scale;
  }
  return { mixing: nextMixing, sources: nextSources, scale, index, compensate };
}

/** Exact counterexample of section 2: zero covariance with total dependence. */
export const DEPENDENCE_POINTS = [
  { u: -1, v: 1, probability: 1 / 3 },
  { u: 0, v: 0, probability: 1 / 3 },
  { u: 1, v: 1, probability: 1 / 3 },
];

export function dependenceSummary(points = DEPENDENCE_POINTS) {
  if (!Array.isArray(points) || points.length < 2 || points.length > 16) throw new Error('Use between 2 and 16 labelled outcomes.');
  points.forEach((point, index) => {
    finite(point.u, `Outcome ${index} u`);
    finite(point.v, `Outcome ${index} v`);
    finite(point.probability, `Outcome ${index} probability`);
    if (point.probability <= 0) throw new Error('Outcome probabilities must be positive.');
  });
  const total = points.reduce((sum, point) => sum + point.probability, 0);
  if (Math.abs(total - 1) > 1e-12) throw new Error('Outcome probabilities must sum to one.');
  const meanU = points.reduce((sum, point) => sum + point.probability * point.u, 0);
  const meanV = points.reduce((sum, point) => sum + point.probability * point.v, 0);
  const covariance = points.reduce((sum, point) => sum + point.probability * (point.u - meanU) * (point.v - meanV), 0);
  const jointZero = points.filter(point => point.u === 0 && point.v === 0).reduce((sum, point) => sum + point.probability, 0);
  const marginalU = points.filter(point => point.u === 0).reduce((sum, point) => sum + point.probability, 0);
  const marginalV = points.filter(point => point.v === 0).reduce((sum, point) => sum + point.probability, 0);
  return {
    points: points.map(point => ({ ...point, product: point.u * point.v })),
    meanU,
    meanV,
    covariance,
    jointZero,
    marginalProduct: marginalU * marginalV,
    marginalU,
    marginalV,
    independent: Math.abs(jointZero - marginalU * marginalV) <= 1e-12,
  };
}

/** Absolute Pearson correlation between two equal-length recorded series. */
export function absoluteCorrelation(values, reference) {
  if (!Array.isArray(values) || !Array.isArray(reference) || values.length !== reference.length) {
    throw new Error('Correlation needs two series of the same length.');
  }
  if (values.length < 2 || values.length > 40000) throw new Error('Use between 2 and 40000 paired samples.');
  const n = values.length;
  const meanA = values.reduce((a, b) => a + b, 0) / n;
  const meanB = reference.reduce((a, b) => a + b, 0) / n;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let index = 0; index < n; index += 1) {
    const da = finite(values[index], 'Series value') - meanA;
    const db = finite(reference[index], 'Reference value') - meanB;
    covariance += da * db;
    varianceA += da * da;
    varianceB += db * db;
  }
  if (varianceA <= 0 || varianceB <= 0) throw new Error('A constant series has no correlation with anything.');
  return Math.abs(covariance / Math.sqrt(varianceA * varianceB));
}
