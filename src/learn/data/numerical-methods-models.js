// Topic-owned deterministic teaching models. Bounds below concern the stated
// mathematical functions; they do not enclose arbitrary floating-point errors.
export function formatNumerical(value, digits = 5) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 1e6) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}
function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite.`);
}
function interval(lower, upper) {
  finite(lower, 'Lower endpoint');
  finite(upper, 'Upper endpoint');
  if (!(lower < upper) || !Number.isFinite(upper - lower)) throw new RangeError('Use an increasing interval with finite width.');
}
function count(value, name, maximum = 16384) {
  if (!Number.isInteger(value) || value < 1 || value > maximum) throw new RangeError(`${name} must be an integer from 1 to ${maximum}.`);
}
const opposite = (left, right) => left < 0 && right > 0 || left > 0 && right < 0;
export const rootProblems = {
  square: {
    label: 'Square root of 2',
    f: x => x * x - 2,
    derivative: x => 2 * x,
    lower: 0,
    upper: 2,
    start: 1.8,
    domain: [-.2, 2.2],
    range: [-2.5, 3]
  },
  cycle: {
    label: 'A Newton cycle',
    f: x => x ** 3 - 2 * x + 2,
    derivative: x => 3 * x * x - 2,
    lower: -2,
    upper: 0,
    start: 0,
    domain: [-2.2, 1.2],
    range: [-5, 4]
  },
  repeated: {
    label: 'A touching root',
    f: x => (x - 1) ** 2,
    derivative: x => 2 * (x - 1),
    lower: 0,
    upper: 2,
    start: 1.8,
    domain: [0, 2],
    range: [-.2, 1.1]
  }
};
export function bracketTrace(f, lower, upper, {
  tolerance = 1e-6,
  maximumSteps = 40
} = {}) {
  interval(lower, upper);
  finite(tolerance, 'Tolerance');
  if (tolerance <= 0) throw new RangeError('Tolerance must be positive.');
  count(maximumSteps, 'Maximum steps', 1000);
  let leftValue = f(lower);
  let rightValue = f(upper);
  const steps = [];
  const result = status => ({
    status,
    lower,
    upper,
    steps,
    midpoint: lower + (upper - lower) / 2
  });
  if (![leftValue, rightValue].every(Number.isFinite)) return result('nonfinite evaluation');
  if (leftValue === 0 || rightValue === 0) return {
    ...result('endpoint root'),
    root: leftValue === 0 ? lower : upper
  };
  if (!opposite(leftValue, rightValue)) return result('no opposite endpoint signs');
  for (let iteration = 0; iteration <= maximumSteps; iteration++) {
    const midpoint = lower + (upper - lower) / 2;
    const value = f(midpoint);
    const state = {
      iteration,
      lower,
      upper,
      leftValue,
      rightValue,
      midpoint,
      value,
      width: upper - lower,
      radius: (upper - lower) / 2
    };
    steps.push(state);
    if (!Number.isFinite(value)) return result('nonfinite evaluation');
    if (value === 0) return {
      ...result('computed zero'),
      root: midpoint
    };
    if (state.radius <= tolerance) return result('bracket tolerance reached');
    if (midpoint === lower || midpoint === upper) return result('arithmetic stagnation');
    if (iteration === maximumSteps) return result('step limit reached');
    if (opposite(leftValue, value)) {
      upper = midpoint;
      rightValue = value;
    } else {
      lower = midpoint;
      leftValue = value;
    }
  }
  return result('step limit reached');
}
export function newtonTrace(problem, start, {
  safeguarded = false,
  maximumSteps = 10,
  tolerance = 1e-10
} = {}) {
  finite(start, 'Starting point');
  finite(tolerance, 'Tolerance');
  if (tolerance <= 0) throw new RangeError('Tolerance must be positive.');
  count(maximumSteps, 'Maximum steps', 1000);
  const {
    f,
    derivative
  } = problem;
  let {
    lower,
    upper
  } = problem;
  interval(lower, upper);
  let leftValue = f(lower);
  const rightValue = f(upper);
  if (safeguarded && (!Number.isFinite(leftValue) || !Number.isFinite(rightValue) || !opposite(leftValue, rightValue))) return {
    steps: [],
    status: 'safeguard needs opposite signs'
  };
  let current = start;
  const steps = [];
  for (let iteration = 0; iteration < maximumSteps; iteration++) {
    const value = f(current);
    const slope = derivative(current);
    if (!Number.isFinite(value)) return {
      steps,
      status: 'nonfinite evaluation'
    };
    if (value === 0) return {
      steps,
      current,
      status: 'computed zero'
    };
    if (Math.abs(value) <= tolerance) return {
      steps,
      current,
      status: 'residual tolerance reached'
    };
    const proposal = slope === 0 ? null : current - value / slope;
    const width = upper - lower;
    const accepted = Number.isFinite(proposal) && (!safeguarded || proposal >= lower + .1 * width && proposal <= upper - .1 * width);
    if (!safeguarded && !accepted) return {
      steps,
      current,
      status: slope === 0 ? 'zero derivative' : 'nonfinite proposal'
    };
    const next = accepted ? proposal : lower + width / 2;
    const nextValue = f(next);
    steps.push({
      iteration,
      current,
      value,
      slope,
      proposal,
      next,
      nextValue,
      lower,
      upper,
      accepted,
      reason: accepted ? 'tangent step' : 'bisection safeguard'
    });
    if (!Number.isFinite(nextValue)) return {
      steps,
      current,
      status: 'nonfinite evaluation'
    };
    if (next === current && nextValue !== 0) return {
      steps,
      current,
      status: 'arithmetic stagnation'
    };
    if (safeguarded && nextValue !== 0) {
      if (opposite(leftValue, nextValue)) upper = next;else {
        lower = next;
        leftValue = nextValue;
      }
    }
    current = next;
  }
  return {
    steps,
    current,
    status: 'step limit reached'
  };
}
export function derivativeEstimate(x, h, stencil = 'central', offset = 0) {
  [x, h, offset].forEach(value => finite(value, 'Input'));
  if (h <= 0 || !['forward', 'central', 'second', 'boundary'].includes(stencil)) throw new RangeError('Use positive h and a known stencil.');
  const terms = {
    forward: [[0, -1], [1, 1]],
    central: [[-1, -.5], [1, .5]],
    second: [[-1, 1], [0, -2], [1, 1]],
    boundary: [[0, -1.5], [1, 2], [2, -.5]]
  }[stencil];
  const nodes = terms.map(([shift, weight]) => ({
    x: x + shift * h,
    weight,
    value: offset + Math.sin(x + shift * h)
  }));
  if (nodes.some(node => !Number.isFinite(node.x) || !Number.isFinite(node.value))) return {
    nodes,
    status: 'nonfinite evaluation',
    estimate: null,
    error: null
  };
  if (new Set(nodes.map(node => node.x)).size !== nodes.length) return {
    nodes,
    status: 'sample coordinates coincide',
    estimate: null,
    error: null
  };
  const estimate = nodes.reduce((sum, node) => sum + node.weight * node.value, 0) / h ** (stencil === 'second' ? 2 : 1);
  const reference = stencil === 'second' ? -Math.sin(x) : Math.cos(x);
  return {
    nodes,
    estimate,
    reference,
    error: Math.abs(estimate - reference),
    status: Number.isFinite(estimate) ? 'computed' : 'nonfinite arithmetic'
  };
}
export const integralProblems = {
  polynomial: {
    label: 'x² on [0, 1]',
    f: x => x * x,
    integral: 1 / 3,
    maximum: 1
  },
  sine: {
    label: 'sin(πx) on [0, 1]',
    f: x => Math.sin(Math.PI * x),
    integral: 2 / Math.PI,
    maximum: 1
  },
  peak: {
    label: 'A resolved narrow peak',
    f: x => 1 / (1 + 400 * (x - .35) ** 2),
    integral: (Math.atan(13) + Math.atan(7)) / 20,
    maximum: 1
  },
  blind: {
    label: 'A sampling blind spot',
    f: x => (x * (x - .25) * (x - .5) * (x - .75) * (x - 1)) ** 2,
    integral: 5 / 1419264,
    maximum: .000015
  }
};
export function compositeQuadrature(f, lower, upper, panels = 4, method = 'trapezoid') {
  interval(lower, upper);
  count(panels, 'Panels');
  if (!['trapezoid', 'simpson'].includes(method) || method === 'simpson' && panels % 2) throw new RangeError('Simpson needs an even number of panels; choose a known method.');
  const h = (upper - lower) / panels;
  const nodes = Array.from({
    length: panels + 1
  }, (_, index) => {
    const x = index === panels ? upper : lower + index * h;
    const weight = method === 'trapezoid' ? index === 0 || index === panels ? .5 : 1 : index === 0 || index === panels ? 1 : index % 2 ? 4 : 2;
    const value = f(x);
    finite(value, 'Function evaluation');
    return {
      x,
      value,
      weight
    };
  });
  if (new Set(nodes.map(node => node.x)).size !== nodes.length) throw new RangeError('Panel coordinates coincide in this arithmetic.');
  const estimate = nodes.reduce((sum, node) => sum + node.value * node.weight, 0) * h / (method === 'simpson' ? 3 : 1);
  finite(estimate, 'Integral arithmetic');
  return {
    estimate,
    nodes,
    h,
    panels,
    method
  };
}
export function sampledTrapezoid(xs, ys) {
  if (xs.length !== ys.length || xs.length < 2) throw new RangeError('Supply matching samples with at least two points.');
  let integral = 0;
  const contributions = [];
  xs.forEach((x, index) => {
    finite(x, 'Coordinate');
    finite(ys[index], 'Value');
  });
  for (let index = 1; index < xs.length; index++) {
    interval(xs[index - 1], xs[index]);
    const contribution = (xs[index] - xs[index - 1]) * (ys[index - 1] / 2 + ys[index] / 2);
    finite(contribution, 'Area');
    contributions.push(contribution);
    integral += contribution;
  }
  finite(integral, 'Area total');
  return {
    integral,
    contributions
  };
}
export function adaptiveSimpson(f, lower, upper, {
  tolerance = 1e-5,
  maximumDepth = 12,
  maximumEvaluations = 4097
} = {}) {
  interval(lower, upper);
  finite(tolerance, 'Tolerance');
  if (tolerance <= 0) throw new RangeError('Tolerance must be positive.');
  count(maximumDepth, 'Maximum depth', 20);
  count(maximumEvaluations, 'Maximum evaluations', 65537);
  const cache = new Map();
  const leaves = [];
  let failure = null;
  function evaluate(x) {
    if (cache.has(x)) return cache.get(x);
    if (cache.size >= maximumEvaluations) throw new Error('evaluation limit reached');
    const value = f(x);
    if (!Number.isFinite(value)) throw new Error('nonfinite evaluation');
    cache.set(x, value);
    return value;
  }
  const panel = (a, b, fa, fm, fb) => (b - a) / 6 * (fa + 4 * fm + fb);
  function recurse(a, b, budget, depth) {
    const midpoint = a + (b - a) / 2;
    const leftMidpoint = a + (midpoint - a) / 2;
    const rightMidpoint = midpoint + (b - midpoint) / 2;
    if (new Set([a, leftMidpoint, midpoint, rightMidpoint, b]).size !== 5) throw new Error('arithmetic stagnation');
    const fa = evaluate(a),
      fm = evaluate(midpoint),
      fb = evaluate(b);
    const coarse = panel(a, b, fa, fm, fb);
    const refined = panel(a, midpoint, fa, evaluate(leftMidpoint), fm) + panel(midpoint, b, fm, evaluate(rightMidpoint), fb);
    const estimatedError = Math.abs(refined - coarse) / 15;
    const value = refined + (refined - coarse) / 15;
    if (![estimatedError, value].every(Number.isFinite)) throw new Error('nonfinite arithmetic');
    if (estimatedError <= budget || depth >= maximumDepth) {
      const accepted = estimatedError <= budget;
      leaves.push({
        lower: a,
        upper: b,
        budget,
        estimatedError,
        value,
        depth,
        accepted
      });
      if (!accepted) failure = 'depth limit reached';
      return value;
    }
    return recurse(a, midpoint, budget / 2, depth + 1) + recurse(midpoint, b, budget / 2, depth + 1);
  }
  let estimate = null;
  try {
    estimate = recurse(lower, upper, tolerance, 0);
  } catch (error) {
    failure = error.message;
  }
  return {
    estimate,
    estimatedError: estimate === null ? null : leaves.reduce((sum, leaf) => sum + leaf.estimatedError, 0),
    status: failure || 'estimated tolerance reached',
    leaves,
    samples: [...cache].sort((a, b) => a[0] - b[0]).map(([x, value]) => ({
      x,
      value
    }))
  };
}
export const pumpRate = time => time * Math.exp(-time);
export const pumpVolume = time => -Math.expm1(-time) - time * Math.exp(-time);
export function pumpCalibration(target = .8, panels = 32, candidate = 3) {
  if (![target, candidate].every(Number.isFinite) || target <= pumpVolume(2) || target >= pumpVolume(4) || candidate < 2 || candidate > 4) throw new RangeError('Use a target strictly between V(2) and V(4), and a time in [2, 4].');
  const quadrature = compositeQuadrature(pumpRate, 0, candidate, panels);
  const integralBound = candidate ** 3 / (6 * panels ** 2);
  const slopeLowerBound = 4 * Math.exp(-4);
  const residual = quadrature.estimate - target;
  const timeBound = (Math.abs(residual) + integralBound) / slopeLowerBound;
  return {
    ...quadrature,
    target,
    candidate,
    integralBound,
    slopeLowerBound,
    residual,
    timeBound,
    volume: pumpVolume(candidate),
    sign: residual > integralBound ? 'above target' : residual < -integralBound ? 'below target' : 'unresolved by this bound'
  };
}
