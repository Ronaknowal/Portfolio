// Bounded finite teaching models. These do not replace a general SVM solver.
export const SVM_GEOMETRY_POINTS = Object.freeze([{
  id: 'A',
  x: [-2, -1],
  y: -1
}, {
  id: 'B',
  x: [-1, 0],
  y: -1
}, {
  id: 'C',
  x: [-2, 1],
  y: -1
}, {
  id: 'D',
  x: [1, 0],
  y: 1
}, {
  id: 'E',
  x: [2, -1],
  y: 1
}, {
  id: 'F',
  x: [2, 1],
  y: 1
}].map(point => Object.freeze({
  ...point,
  x: Object.freeze(point.x)
})));
export const SVM_XOR_POINTS = Object.freeze([[-1, -1], [-1, 1], [1, -1], [1, 1]].map(point => Object.freeze(point)));
export const SVM_XOR_LABELS = Object.freeze([1, -1, -1, 1]);
function bounded(value, lower, upper, name) {
  if (!Number.isFinite(value) || value < lower || value > upper) {
    throw new RangeError(`${name} must be finite and between ${lower} and ${upper}.`);
  }
  return value;
}
function vector(values, length, lower, upper, name) {
  if (!Array.isArray(values) || values.length !== length) throw new TypeError(`${name} has the wrong shape.`);
  return Array.from(values, value => bounded(value, lower, upper, name));
}
export function svmDot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}
export function marginGeometry(angleDegrees = 0, offset = 0, scale = 1) {
  bounded(angleDegrees, -45, 45, 'Angle');
  bounded(offset, -.75, .75, 'Boundary offset');
  bounded(scale, .5, 3, 'Score scale');
  const angle = angleDegrees * Math.PI / 180;
  const normal = [Math.cos(angle), Math.sin(angle)];
  const w = normal.map(value => scale * value);
  const b = -scale * offset;
  const rows = SVM_GEOMETRY_POINTS.map(point => {
    const signedDistance = svmDot(normal, point.x) - offset;
    return {
      ...point,
      score: scale * signedDistance,
      signedDistance,
      labelDistance: point.y * signedDistance,
      projection: point.x.map((value, index) => value - signedDistance * normal[index])
    };
  });
  const nearest = rows.reduce((best, row) => row.labelDistance < best.labelDistance ? row : best);
  return {
    normal,
    w,
    b,
    norm: scale,
    offset,
    rows,
    nearest,
    separating: nearest.labelDistance > 0,
    geometricMargin: nearest.labelDistance,
    canonicalFeasible: rows.every(row => row.y * row.score >= 1 - 1e-12),
    scoreCorridorWidth: 2 / scale
  };
}
export function supportMotion(moved = 2) {
  bounded(moved, .25, 3, 'Moved positive point');
  const nearestPositive = Math.min(1, moved);
  const separation = 1 + nearestPositive;
  const w = 2 / separation;
  const b = (1 - nearestPositive) / separation;
  const alpha = 2 / separation ** 2;
  const positiveIndex = moved < 1 ? 3 : 2;
  const rows = [-2, -1, 1, moved].map((x, index) => {
    const y = index < 2 ? -1 : 1;
    return {
      id: ['A', 'B', 'C', 'D'][index],
      x,
      y,
      alpha: index === 1 || index === positiveIndex ? alpha : 0,
      oldMargin: y * x,
      margin: y * (w * x + b)
    };
  });
  return {
    moved,
    nearestPositive,
    w,
    b,
    boundary: -b / w,
    margin: 1 / w,
    oldFeasible: moved >= 1,
    rows,
    primal: .5 * w ** 2,
    dual: 2 * alpha - .5 * w ** 2
  };
}
export function softPairState(c = .25, conflicting = false, biasFraction = 0) {
  bounded(c, .05, 4, 'C');
  if (typeof conflicting !== 'boolean') throw new TypeError('Conflicting fixture must be boolean.');
  bounded(biasFraction, -1, 1, 'Bias fraction');
  const gap = conflicting ? 0 : 1;
  const alpha = conflicting ? c : Math.min(c, .5);
  const w = 2 * gap * alpha;
  const biasLimit = Math.max(0, 1 - gap * w);
  const b = biasFraction * biasLimit;
  const rows = [-1, 1].map(y => {
    const x = y * gap;
    const score = w * x + b;
    const margin = y * score;
    return {
      x,
      y,
      score,
      margin,
      hinge: Math.max(0, 1 - margin),
      alpha
    };
  });
  const normCost = .5 * w ** 2;
  const lossCost = c * rows.reduce((sum, row) => sum + row.hinge, 0);
  return {
    c,
    conflicting,
    w,
    b,
    alpha,
    biasLimit,
    rows,
    normCost,
    lossCost,
    primal: normCost + lossCost,
    dual: 2 * alpha - normCost,
    geometricMargin: w === 0 ? null : Math.min(...rows.map(row => row.margin)) / Math.abs(w)
  };
}
export function svmKernel(left, right, kind = 'rbf', gamma = .5) {
  vector(left, 2, -6, 6, 'Kernel input');
  vector(right, 2, -6, 6, 'Kernel input');
  bounded(gamma, .05, 4, 'Gamma');
  if (kind === 'linear') return svmDot(left, right);
  if (kind === 'poly') return svmDot(left, right) ** 2;
  if (kind !== 'rbf') throw new RangeError('Choose linear, poly or rbf.');
  return Math.exp(-gamma * left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0));
}
export function xorKernelState(kind = 'rbf', c = 1, gamma = .5, query = [.5, .5]) {
  bounded(c, .05, 4, 'C');
  vector(query, 2, -2, 2, 'Query');
  bounded(gamma, .05, 4, 'Gamma');
  if (!['linear', 'poly', 'rbf'].includes(kind)) throw new RangeError('Unknown kernel.');
  const eigenvalue = kind === 'linear' ? 0 : kind === 'poly' ? 8 : (-Math.expm1(-4 * gamma)) ** 2;
  const alpha = eigenvalue === 0 ? c : Math.min(c, 1 / eigenvalue);
  const gram = SVM_XOR_POINTS.map(left => SVM_XOR_POINTS.map(right => svmKernel(left, right, kind, gamma)));
  const contributions = SVM_XOR_POINTS.map((point, index) => {
    const kernel = svmKernel(point, query, kind, gamma);
    return {
      point,
      label: SVM_XOR_LABELS[index],
      kernel,
      alpha,
      value: alpha * SVM_XOR_LABELS[index] * kernel
    };
  });
  const score = contributions.reduce((sum, row) => sum + row.value, 0);
  const normSquared = 4 * eigenvalue * alpha ** 2;
  const margins = gram.map(row => alpha * svmDot(row, SVM_XOR_LABELS)).map((value, index) => value * SVM_XOR_LABELS[index]);
  const primal = .5 * normSquared + c * margins.reduce((sum, value) => sum + Math.max(0, 1 - value), 0);
  return {
    kind,
    c,
    gamma,
    query: [...query],
    gram,
    alpha,
    eigenvalue,
    contributions,
    score,
    prediction: score >= 0 ? 1 : -1,
    normSquared,
    margins,
    primal,
    dual: 4 * alpha - .5 * normSquared,
    lifted: SVM_XOR_POINTS.map(point => [point[0] ** 2, Math.SQRT2 * point[0] * point[1], point[1] ** 2])
  };
}
export function xorScore(point, kind, c, gamma) {
  bounded(c, .05, 4, 'C');
  bounded(gamma, .05, 4, 'Gamma');
  vector(point, 2, -2, 2, 'Query');
  const eigenvalue = kind === 'linear' ? 0 : kind === 'poly' ? 8 : (-Math.expm1(-4 * gamma)) ** 2;
  const alpha = eigenvalue === 0 ? c : Math.min(c, 1 / eigenvalue);
  return SVM_XOR_POINTS.reduce((sum, support, index) => sum + alpha * SVM_XOR_LABELS[index] * svmKernel(support, point, kind, gamma), 0);
}
export function svmPairFixture(duplicate = false) {
  return duplicate ? {
    points: [[0, 0], [0, 0]],
    labels: [-1, 1],
    alpha: [0, 0],
    c: 1.2
  } : {
    points: [[-1, 0], [.5, 1], [1, -.5]],
    labels: [-1, 1, 1],
    alpha: [.2, .1, .1],
    c: 1.2
  };
}
export function svmPairStep(points, labels, alpha, c, i, j) {
  bounded(c, .05, 4, 'C');
  if (!Array.isArray(points) || points.length < 2 || points.length > 8) throw new RangeError('Use two to eight points.');
  const count = points.length;
  const x = Array.from(points, point => vector(point, 2, -6, 6, 'Point'));
  const y = vector(labels, count, -1, 1, 'Labels');
  if (y.some(label => label !== 1 && label !== -1) || !y.includes(-1) || !y.includes(1)) throw new TypeError('Use both -1 and +1 labels.');
  const a = vector(alpha, count, 0, c, 'Alpha');
  if (Math.abs(svmDot(a, y)) > 1e-9) throw new RangeError('The starting alpha vector must satisfy label balance.');
  if (!Number.isInteger(i) || !Number.isInteger(j) || i < 0 || j < 0 || i >= count || j >= count || i === j) throw new RangeError('Select two distinct point indices.');
  const gram = x.map(left => x.map(right => svmDot(left, right)));
  const weights = a.map((value, index) => value * y[index]);
  const scores = gram.map(row => svmDot(row, weights));
  const gradient = scores.map((value, index) => 1 - y[index] * value);
  const sign = y[i] * y[j];
  const lower = Math.max(-a[j], sign === 1 ? a[i] - c : -a[i]);
  const upper = Math.min(c - a[j], sign === 1 ? a[i] : c - a[i]);
  const q = x[i].reduce((sum, value, dimension) => sum + (value - x[j][dimension]) ** 2, 0);
  const g = gradient[j] - sign * gradient[i];
  const improvement = delta => g * delta - .5 * q * delta ** 2;
  const bestDelta = q > 0 ? Math.max(lower, Math.min(upper, g / q)) : g === 0 ? 0 : improvement(upper) > improvement(lower) ? upper : lower;
  const after = [...a];
  after[i] = Math.max(0, Math.min(c, a[i] - sign * bestDelta));
  after[j] = Math.max(0, Math.min(c, a[j] + bestDelta));
  const dual = values => {
    const signed = values.map((value, index) => value * y[index]);
    return values.reduce((sum, value) => sum + value, 0) - .5 * signed.reduce((sum, value, row) => sum + value * svmDot(gram[row], signed), 0);
  };
  return {
    points: x,
    labels: y,
    alpha: a,
    c,
    i,
    j,
    sign,
    lower,
    upper,
    q,
    g,
    bestDelta,
    after,
    beforeDual: dual(a),
    afterDual: dual(after),
    improvement: improvement(bestDelta),
    balance: svmDot(after, y),
    curve: Array.from({
      length: 65
    }, (_, index) => {
      const delta = lower + (upper - lower) * index / 64;
      return {
        delta,
        alphaJ: a[j] + delta,
        gain: improvement(delta)
      };
    })
  };
}
export function svrTubeState(amplitude = 2, epsilon = .5, c = .5) {
  bounded(amplitude, 1, 3, 'Target amplitude');
  bounded(epsilon, 0, 2, 'Epsilon');
  bounded(c, .05, 4, 'C');
  const slope = Math.min(2 * c, Math.max(amplitude - epsilon, 0));
  const rows = [-1, 0, 1].map(x => {
    const target = amplitude * x;
    const prediction = slope * x;
    const residual = target - prediction;
    return {
      x,
      target,
      prediction,
      residual,
      excess: Math.max(0, Math.abs(residual) - epsilon)
    };
  });
  const normCost = .5 * slope ** 2;
  const lossCost = c * rows.reduce((sum, row) => sum + row.excess, 0);
  return {
    amplitude,
    epsilon,
    c,
    slope,
    bias: 0,
    rows,
    normCost,
    lossCost,
    objective: normCost + lossCost
  };
}
export function spectrumCounts(sequence, size = 2) {
  if (typeof sequence !== 'string' || !/^[AC]{0,12}$/.test(sequence)) throw new RangeError('Use at most twelve A/C characters.');
  if (!Number.isInteger(size) || size < 1 || size > 3) throw new RangeError('Use a word length of one to three.');
  const counts = {};
  const windows = [];
  for (let index = 0; index + size <= sequence.length; index += 1) {
    const word = sequence.slice(index, index + size);
    counts[word] = (counts[word] ?? 0) + 1;
    windows.push({
      index,
      word
    });
  }
  return {
    sequence,
    size,
    counts,
    windows
  };
}
