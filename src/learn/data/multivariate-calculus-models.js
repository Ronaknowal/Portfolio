export const formatCalculusNumber = value => {
  if (value === 0) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 0.0001) return value.toExponential(3);
  return String(Number(value.toFixed(5)));
};
function bounded(value, minimum, maximum, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be a finite number from ${minimum} to ${maximum}.`);
  }
}
export function bowl(point) {
  return point[0] ** 2 + 2 * point[1] ** 2;
}
export function localChangeState(x, y, angle, step) {
  bounded(x, -2, 2, 'Base x');
  bounded(y, -2, 2, 'Base y');
  bounded(angle, 0, 360, 'Direction angle');
  bounded(step, -0.5, 0.5, 'Signed step');
  const radians = angle * Math.PI / 180;
  const direction = [Math.cos(radians), Math.sin(radians)];
  const gradient = [2 * x, 4 * y];
  const gradientNorm = Math.hypot(...gradient);
  const rate = gradient.reduce((sum, value, index) => sum + value * direction[index], 0);
  const base = [x, y];
  const next = base.map((value, index) => value + step * direction[index]);
  const initial = bowl(base);
  const predicted = initial + step * rate;
  const actual = bowl(next);
  const curvature = 2 * direction[0] ** 2 + 4 * direction[1] ** 2;
  const slices = Array.from({
    length: 81
  }, (_, index) => {
    const distance = (index - 40) / 80;
    return {
      distance,
      actual: bowl(base.map((value, axis) => value + distance * direction[axis])),
      linear: initial + distance * rate
    };
  });
  return {
    base,
    next,
    direction,
    gradient,
    gradientNorm,
    rate,
    step,
    initial,
    predicted,
    actual,
    predictedChange: step * rate,
    actualChange: actual - initial,
    remainder: actual - predicted,
    curvature,
    slices
  };
}
export function approachFunction(x, y) {
  const denominator = x ** 4 + y ** 2;
  return denominator === 0 ? 0 : x * x * y / denominator;
}
export function approachState(kind, coefficient) {
  if (!['axis', 'line', 'parabola'].includes(kind)) throw new Error('Choose an axis, line or parabola.');
  bounded(coefficient, -2, 2, 'Path coefficient');
  const point = x => [x, kind === 'axis' ? 0 : coefficient * x ** (kind === 'parabola' ? 2 : 1)];
  const samples = Array.from({
    length: 7
  }, (_, index) => {
    const [x, y] = point(10 ** -index);
    return {
      x,
      y,
      value: approachFunction(x, y),
      distance: Math.hypot(x, y)
    };
  });
  const path = Array.from({
    length: 61
  }, (_, index) => point(index / 60));
  return {
    kind,
    coefficient,
    samples,
    path,
    limit: kind === 'parabola' ? coefficient / (1 + coefficient ** 2) : 0
  };
}
export function circleMotionState(angle) {
  bounded(angle, 0, 360, 'Circle angle');
  const radians = angle * Math.PI / 180;
  const point = [Math.cos(radians), Math.sin(radians)];
  const tangent = [-point[1], point[0]];
  const gradient = [2, 1];
  const value = 2 * point[0] + point[1];
  const rate = 2 * tangent[0] + tangent[1];
  const samples = Array.from({
    length: 121
  }, (_, index) => {
    const theta = index * Math.PI / 60;
    return {
      angle: index * 3,
      value: 2 * Math.cos(theta) + Math.sin(theta)
    };
  });
  return {
    angle,
    point,
    tangent,
    gradient,
    value,
    rate,
    samples,
    tangentGradient: tangent.map(component => rate * component),
    maximumAngle: Math.atan2(1, 2) * 180 / Math.PI
  };
}
export const curvaturePresets = {
  bowl: {
    title: 'Quadratic minimum',
    formula: 'x² + 2y²',
    hessian: [[2, 0], [0, 4]],
    value: (x, y) => x * x + 2 * y * y,
    verdict: 'Positive definite: a strict local minimum.',
    eigenvalues: [2, 4]
  },
  tilted: {
    title: 'Tilted quadratic minimum',
    formula: 'x² + 2xy + 2y²',
    hessian: [[2, 2], [2, 4]],
    value: (x, y) => x * x + 2 * x * y + 2 * y * y,
    verdict: 'Positive definite: a strict local minimum; the principal directions are rotated.',
    eigenvalues: [3 - Math.sqrt(5), 3 + Math.sqrt(5)]
  },
  maximum: {
    title: 'Quadratic maximum',
    formula: '−x² − 2y²',
    hessian: [[-2, 0], [0, -4]],
    value: (x, y) => -x * x - 2 * y * y,
    verdict: 'Negative definite: a strict local maximum.',
    eigenvalues: [-4, -2]
  },
  saddle: {
    title: 'Quadratic saddle',
    formula: 'x² − y²',
    hessian: [[2, 0], [0, -2]],
    value: (x, y) => x * x - y * y,
    verdict: 'Indefinite: both signs of curvature establish a saddle.',
    eigenvalues: [-2, 2]
  },
  flatMinimum: {
    title: 'Flat quartic minimum',
    formula: 'x⁴ + y⁴',
    hessian: [[0, 0], [0, 0]],
    value: (x, y) => x ** 4 + y ** 4,
    verdict: 'Hessian inconclusive. The exact nonnegative function establishes a strict minimum.',
    eigenvalues: [0, 0]
  },
  flatSaddle: {
    title: 'Flat quartic saddle',
    formula: 'x⁴ − y⁴',
    hessian: [[0, 0], [0, 0]],
    value: (x, y) => x ** 4 - y ** 4,
    verdict: 'Hessian inconclusive. Opposite signs on the coordinate axes establish a saddle.',
    eigenvalues: [0, 0]
  }
};
export function curvatureState(preset, angle) {
  if (!Object.hasOwn(curvaturePresets, preset)) throw new Error('Choose a listed curvature example.');
  bounded(angle, 0, 360, 'Slice angle');
  const model = curvaturePresets[preset];
  const direction = [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)];
  const hessianDirection = model.hessian.map(row => row.reduce((sum, entry, index) => sum + entry * direction[index], 0));
  const curvature = direction.reduce((sum, entry, index) => sum + entry * hessianDirection[index], 0);
  const samples = Array.from({
    length: 81
  }, (_, index) => {
    const t = (index - 40) / 40;
    return {
      t,
      actual: model.value(t * direction[0], t * direction[1]),
      quadratic: 0.5 * curvature * t * t
    };
  });
  return {
    preset,
    angle,
    direction,
    curvature,
    hessian: model.hessian,
    eigenvalues: model.eigenvalues,
    verdict: model.verdict,
    formula: model.formula,
    samples
  };
}
export function descentState(rate, steps) {
  bounded(rate, 0, 0.75, 'Learning rate');
  if (!Number.isInteger(steps) || steps < 0 || steps > 12) throw new Error('Steps must be an integer from 0 to 12.');
  const factors = [1 - 2 * rate, 1 - 4 * rate];
  const history = [{
    step: 0,
    point: [3, -2],
    loss: 17
  }];
  for (let index = 1; index <= steps; index += 1) {
    const previous = history.at(-1).point;
    const point = previous.map((value, axis) => value * factors[axis]);
    history.push({
      step: index,
      point,
      loss: bowl(point)
    });
  }
  const current = history.at(-1);
  const gradient = [2 * current.point[0], 4 * current.point[1]];
  return {
    rate,
    steps,
    factors,
    history,
    current,
    gradient,
    convergesFromEveryPoint: rate > 0 && rate < 0.5,
    behavior: rate === 0 ? 'No update: both coordinate factors equal 1.' : rate < 0.5 ? 'Both factor magnitudes are below 1: this quadratic converges from every starting point.' : rate === 0.5 ? 'The y factor is −1: y oscillates without shrinking. This start does not converge to the minimum.' : 'The y factor magnitude exceeds 1: the nonzero y coordinate grows while changing sign.'
  };
}
