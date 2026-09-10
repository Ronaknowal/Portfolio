// Small deterministic real-matrix investigations, not a general eigensolver.
const radians = degrees => degrees * Math.PI / 180;
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const apply = (matrix, vector) => matrix.map(row => dot(row, vector));
const clean = value => Math.abs(value) < 1e-14 ? 0 : value;
function assertAngle(angle) {
  if (!Number.isFinite(angle) || angle < 0 || angle > 360) {
    throw new Error('Choose an angle from 0 to 360 degrees.');
  }
}
export const eigenDirectionPresets = {
  diagonalStretch: {
    title: 'Stretch the two diagonal directions',
    matrix: [[2, 1], [1, 2]],
    interpretation: 'The lines at 45° and 135° are preserved, with factors 3 and 1.'
  },
  reflection: {
    title: 'Reflect across the horizontal axis',
    matrix: [[1, 0], [0, -1]],
    interpretation: 'Horizontal inputs stay; vertical inputs reverse. Their eigenvalues are 1 and −1.'
  },
  projection: {
    title: 'Project onto the horizontal axis',
    matrix: [[1, 0], [0, 0]],
    interpretation: 'Vertical nonzero inputs disappear. Zero is a valid eigenvalue; the zero input itself is not an eigenvector.'
  },
  shear: {
    title: 'Shear horizontally',
    matrix: [[1, 1], [0, 1]],
    interpretation: 'Only the horizontal line is preserved. A repeated eigenvalue need not supply two independent directions.'
  },
  rotation: {
    title: 'Rotate by a quarter turn',
    matrix: [[0, -1], [1, 0]],
    interpretation: 'Every nonzero real input turns 90°. There is no real eigendirection; complex eigenvalues are ±i.'
  },
  scalar: {
    title: 'Double every coordinate',
    matrix: [[2, 0], [0, 2]],
    interpretation: 'Every line is preserved. The repeated eigenvalue 2 has a two-dimensional eigenspace.'
  }
};
export function eigenDirectionState(presetName = 'diagonalStretch', angle = 30) {
  const preset = eigenDirectionPresets[presetName];
  if (!preset) throw new Error('Choose a listed transformation.');
  assertAngle(angle);
  const input = [clean(Math.cos(radians(angle))), clean(Math.sin(radians(angle)))];
  const output = apply(preset.matrix, input);
  const alongFactor = dot(input, output) / dot(input, input);
  const along = input.map(value => alongFactor * value);
  const perpendicular = output.map((value, index) => value - along[index]);
  const residualNorm = Math.hypot(...perpendicular);
  return {
    matrix: preset.matrix.map(row => [...row]),
    input,
    output,
    along,
    alongFactor,
    perpendicular,
    residualNorm,
    isEigenDirection: residualNorm < 1e-10,
    isZeroOutput: Math.hypot(...output) < 1e-10,
    interpretation: preset.interpretation
  };
}
export const repeatedMapPresets = {
  decay: {
    title: 'Two decaying coordinates',
    matrix: [[0.7, 0], [0, 0.3]],
    eigenvalues: '0.7, 0.3',
    conclusion: 'Every starting vector tends to zero; each component is multiplied by a scalar of magnitude below one.'
  },
  alternating: {
    title: 'Decay with a sign flip',
    matrix: [[-0.8, 0], [0, 0.4]],
    eigenvalues: '−0.8, 0.4',
    conclusion: 'The first coordinate alternates in sign while its magnitude shrinks. A negative eigenvalue does not imply growth.'
  },
  growth: {
    title: 'One growing coordinate',
    matrix: [[1.2, 0], [0, 0.5]],
    eigenvalues: '1.2, 0.5',
    conclusion: 'A nonzero horizontal component grows. A purely vertical start still decays: the initial state matters.'
  },
  rotation: {
    title: 'Quarter-turn without decay',
    matrix: [[0, -1], [1, 0]],
    eigenvalues: 'i, −i; both magnitudes 1',
    conclusion: 'Length stays constant and the vector cycles. Bounded behavior is different from convergence to zero.'
  },
  shear: {
    title: 'Repeated unit eigenvalue with a shear',
    matrix: [[1, 1], [0, 1]],
    eigenvalues: '1, 1',
    conclusion: 'A nonzero vertical component adds to the horizontal one every step. Unit-magnitude eigenvalues alone do not guarantee bounded iterates.'
  },
  transient: {
    title: 'Temporary growth, eventual decay',
    matrix: [[0.6, 2], [0, 0.6]],
    eigenvalues: '0.6, 0.6',
    conclusion: 'The off-diagonal coupling can amplify the state first. The factor k·0.6^(k−1) eventually decays: long-run stability is not a bound on every intermediate step.'
  }
};
export const repeatedMapStarts = {
  mixed: {
    title: '(1, 1): both coordinates',
    vector: [1, 1]
  },
  horizontal: {
    title: '(1, 0): horizontal only',
    vector: [1, 0]
  },
  vertical: {
    title: '(0, 1): vertical only',
    vector: [0, 1]
  }
};
export function repeatedMapTrace(presetName = 'decay', startName = 'mixed', steps = 12) {
  const preset = repeatedMapPresets[presetName];
  const start = repeatedMapStarts[startName];
  if (!preset || !start) throw new Error('Choose a listed map and starting vector.');
  if (!Number.isInteger(steps) || steps < 0 || steps > 30) throw new Error('Use 0 to 30 update steps.');
  let vector = [...start.vector];
  const states = [{
    step: 0,
    vector,
    norm: Math.hypot(...vector)
  }];
  for (let step = 1; step <= steps; step++) {
    vector = apply(preset.matrix, vector);
    states.push({
      step,
      vector,
      norm: Math.hypot(...vector)
    });
  }
  return {
    matrix: preset.matrix.map(row => [...row]),
    states,
    eigenvalues: preset.eigenvalues,
    conclusion: preset.conclusion
  };
}
export const pcaDirectionDatasets = {
  diagonalCloud: {
    title: 'Unequal diagonal variance',
    points: [[-2, -1], [-1, -2], [1, 2], [2, 1]],
    conclusion: 'The diagonal at 45° has sample variance 6; its perpendicular has variance 2/3. The leading direction retains 90% of total sample variance.'
  },
  isotropic: {
    title: 'Equal variance in every direction',
    points: [[-1, 0], [1, 0], [0, -1], [0, 1]],
    conclusion: 'Both eigenvalues are 2/3. There is no unique best line; every unit direction has the same variance.'
  },
  line: {
    title: 'All variation on one line',
    points: [[-2, -2], [-1, -1], [1, 1], [2, 2]],
    conclusion: 'The perpendicular eigenvalue is zero. The 45° line retains all variation, while projection onto 135° erases it.'
  }
};
export function pcaDirectionState(datasetName = 'diagonalCloud', angle = 0) {
  const dataset = pcaDirectionDatasets[datasetName];
  if (!dataset) throw new Error('Choose a listed dataset.');
  assertAngle(angle);
  const points = dataset.points.map(point => [...point]);
  const mean = [0, 1].map(index => points.reduce((sum, point) => sum + point[index], 0) / points.length);
  const centered = points.map(point => point.map((value, index) => value - mean[index]));
  const direction = [clean(Math.cos(radians(angle))), clean(Math.sin(radians(angle)))];
  const scores = centered.map(point => dot(point, direction));
  const projections = scores.map(score => direction.map((value, index) => mean[index] + score * value));
  const covariance = [0, 1].map(row => [0, 1].map(column => centered.reduce((sum, point) => sum + point[row] * point[column], 0) / (points.length - 1)));
  const variance = scores.reduce((sum, value) => sum + value ** 2, 0) / (points.length - 1);
  const totalVariance = covariance[0][0] + covariance[1][1];
  const squaredReconstructionError = points.reduce((sum, point, index) => sum + point.reduce((error, value, axis) => error + (value - projections[index][axis]) ** 2, 0), 0);
  return {
    points,
    mean,
    centered,
    direction,
    scores,
    projections,
    covariance,
    variance,
    totalVariance,
    retainedFraction: variance / totalVariance,
    squaredReconstructionError,
    conclusion: dataset.conclusion
  };
}
