// Small deterministic fixtures for convexity, certificates, ridge curvature and
// proximal geometry. No production solver or empirical timing is implied.
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const subtract = (left, right) => left.map((value, index) => value - right[index]);
const squaredNorm = vector => dot(vector, vector);
function bounded(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must lie between ${minimum} and ${maximum}.`);
  }
}
export const convexCurves = {
  quadratic: {
    title: "Quadratic: x²",
    value: x => x * x,
    derivative: x => 2 * x,
    convex: true
  },
  absolute: {
    title: "Kink: |x|",
    value: Math.abs,
    derivative: x => x === 0 ? null : Math.sign(x),
    convex: true
  },
  quartic: {
    title: "Quartic: x⁴ / 4",
    value: x => x ** 4 / 4,
    derivative: x => x ** 3,
    convex: true
  },
  doubleWell: {
    title: "Two wells: x⁴ / 4 − x²",
    value: x => x ** 4 / 4 - x * x,
    derivative: x => x ** 3 - 2 * x,
    convex: false
  }
};
export function convexChordState(preset, left, right, fraction) {
  const curve = convexCurves[preset];
  if (!curve) throw new RangeError("Unknown curve.");
  bounded(left, -2, 2, "Left input");
  bounded(right, -2, 2, "Right input");
  bounded(fraction, 0, 1, "Mixing fraction");
  if (left >= right) throw new RangeError("Choose left input below right input.");
  const input = (1 - fraction) * left + fraction * right;
  const value = curve.value(input);
  const chordValue = (1 - fraction) * curve.value(left) + fraction * curve.value(right);
  return {
    preset,
    left,
    right,
    fraction,
    input,
    value,
    chordValue,
    gap: chordValue - value,
    slope: curve.derivative(input),
    samples: Array.from({
      length: 81
    }, (_, index) => {
      const x = -2 + index / 20;
      return [x, curve.value(x)];
    }),
    endpoints: [[left, curve.value(left)], [right, curve.value(right)]]
  };
}
export function allocationCertificateState(budget, first, second) {
  bounded(budget, 0, 8, 'Budget');
  bounded(first, 0, 8, "First allocation");
  bounded(second, 0, 8, "Second allocation");
  const target = [4, 3];
  const candidate = [first, second];
  const feasible = first + second <= budget;
  const optimum = budget >= 7 ? target.slice() : budget <= 1 ? [budget, 0] : [(budget + 1) / 2, (budget - 1) / 2];
  const gradient = subtract(candidate, target);
  const objective = squaredNorm(gradient) / 2;
  const optimumValue = squaredNorm(subtract(optimum, target)) / 2;
  const vertices = [[0, 0], [budget, 0], [0, budget]];
  const directionalValues = vertices.map(vertex => dot(gradient, subtract(vertex, candidate)));
  const minimumDirection = Math.min(...directionalValues);
  const lowerBound = objective + minimumDirection;
  return {
    budget,
    target,
    candidate,
    feasible,
    optimum,
    objective,
    optimumValue,
    gradient,
    vertices,
    directionalValues,
    lowerBound,
    gap: feasible ? Math.max(0, objective - lowerBound) : null,
    actualError: feasible ? Math.max(0, objective - optimumValue) : null,
    violation: Math.max(0, first + second - budget),
    minimizingVertex: vertices[directionalValues.indexOf(minimumDirection)]
  };
}
export const ridgeDesigns = {
  full: {
    title: "Original intercept and input columns",
    rows: [[1, 0], [1, 1], [1, 2]]
  },
  duplicate: {
    title: "Two identical feature columns",
    rows: [[1, 1], [1, 1], [1, 1]]
  }
};
export function ridgeCurvatureState(preset, penalty, stepFactor, iterations) {
  if (!ridgeDesigns[preset]) throw new RangeError("Unknown feature matrix.");
  bounded(penalty, 0, 2, "Ridge penalty");
  bounded(stepFactor, 0.1, 2.2, "Step times largest curvature");
  bounded(iterations, 0, 24, "Iteration count");
  if (!Number.isInteger(iterations)) throw new RangeError("Iteration count must be an integer.");
  const rows = ridgeDesigns[preset].rows.map(row => row.slice());
  const response = [1, 2, 2];
  const gram = [0, 1].map(first => [0, 1].map(second => rows.reduce((sum, row) => sum + row[first] * row[second], 0) + (first === second ? penalty : 0)));
  const normalResponse = [0, 1].map(column => rows.reduce((sum, row, index) => sum + row[column] * response[index], 0));
  const hessian = gram.map(row => row.map(value => 2 * value));
  const trace = hessian[0][0] + hessian[1][1];
  const spread = Math.hypot(hessian[0][0] - hessian[1][1], 2 * hessian[0][1]);
  // Duplicate columns have known eigendirections. Preserve a small positive
  // penalty instead of losing it by subtracting nearly equal eigenvalues.
  const smallestCurvature = preset === 'duplicate' ? 2 * penalty : (trace - spread) / 2;
  const largestCurvature = preset === 'duplicate' ? 12 + 2 * penalty : (trace + spread) / 2;
  const unique = preset === 'full' || penalty > 0;
  const determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
  const optimum = preset === 'duplicate'
    ? [5 / (6 + penalty), 5 / (6 + penalty)]
    : [(normalResponse[0] * gram[1][1] - normalResponse[1] * gram[0][1]) / determinant, (normalResponse[1] * gram[0][0] - normalResponse[0] * gram[1][0]) / determinant];
  // At zero penalty the duplicate fixture returns the minimum-norm member
  // of its line of minimizers, without claiming that member is unique.
  const objectiveAt = weights => rows.reduce((sum, row, index) => sum + (dot(row, weights) - response[index]) ** 2, 0) + penalty * squaredNorm(weights);
  const gradientAt = weights => hessian.map((row, index) => dot(row, weights) - 2 * normalResponse[index]);
  const optimumValue = objectiveAt(optimum);
  const step = stepFactor / largestCurvature;
  const path = [];
  let weights = [-1, 2];
  for (let iteration = 0; iteration <= iterations; iteration += 1) {
    const gradient = gradientAt(weights);
    const value = objectiveAt(weights);
    path.push({
      iteration,
      weights: weights.slice(),
      gradient,
      value,
      error: Math.max(0, value - optimumValue)
    });
    weights = weights.map((value, index) => value - step * gradient[index]);
  }
  const angle = 0.5 * Math.atan2(2 * hessian[0][1], hessian[0][0] - hessian[1][1]);
  const highDirection = [Math.cos(angle), Math.sin(angle)];
  const lowDirection = [-Math.sin(angle), Math.cos(angle)];
  const current = path.at(-1);
  return {
    preset,
    rows,
    response,
    penalty,
    stepFactor,
    iterations,
    gram,
    hessian,
    smallestCurvature,
    largestCurvature,
    unique,
    condition: unique ? largestCurvature / smallestCurvature : null,
    optimum,
    optimumValue,
    step,
    path,
    current,
    highDirection,
    lowDirection,
    lowerBound: unique ? current.value - squaredNorm(current.gradient) / (2 * smallestCurvature) : null
  };
}
export function ridgeContourPoints(state, excess, count = 97) {
  bounded(excess, 0.01, 100, "Contour excess");
  if (!Number.isInteger(count) || count < 8 || count > 200) throw new RangeError("Contour resolution must be between 8 and 200.");
  if (!state.unique) return [];
  const highRadius = Math.sqrt(2 * excess / state.largestCurvature);
  const lowRadius = Math.sqrt(2 * excess / state.smallestCurvature);
  if (![highRadius, lowRadius].every(Number.isFinite)) throw new RangeError('Contour radius exceeds the finite plotting range.');
  return Array.from({
    length: count
  }, (_, index) => {
    const angle = 2 * Math.PI * index / (count - 1);
    return state.optimum.map((value, dimension) => value + highRadius * Math.cos(angle) * state.highDirection[dimension] + lowRadius * Math.sin(angle) * state.lowDirection[dimension]);
  });
}
export function softThresholdState(input, penalty, candidate) {
  bounded(input, -4, 4, 'Input');
  bounded(penalty, 0, 4, 'Penalty');
  bounded(candidate, -4, 4, 'Candidate');
  const optimum = Math.sign(input) * Math.max(0, Math.abs(input) - penalty);
  const objectiveAt = value => (value - input) ** 2 / 2 + penalty * Math.abs(value);
  const subgradient = candidate === 0 ? [-input - penalty, -input + penalty] : [candidate - input + penalty * Math.sign(candidate), candidate - input + penalty * Math.sign(candidate)];
  return {
    input,
    penalty,
    candidate,
    optimum,
    value: objectiveAt(candidate),
    optimumValue: objectiveAt(optimum),
    subgradient,
    stationary: subgradient[0] <= 0 && subgradient[1] >= 0,
    samples: Array.from({
      length: 101
    }, (_, index) => {
      const value = -5 + index / 10;
      return [value, objectiveAt(value)];
    })
  };
}
