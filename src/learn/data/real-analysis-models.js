// Bounded analytical teaching models. Returned samples illustrate a formula;
// the separately returned bounds/witnesses supply the mathematical claim.
function finiteInRange(value, name, minimum, maximum) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
  return value;
}
function integerInRange(value, name, minimum, maximum) {
  finiteInRange(value, name, minimum, maximum);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
function choice(value, name, allowed) {
  if (!allowed.includes(value)) throw new RangeError(`Unknown ${name}.`);
  return value;
}
export function sequenceTail(numerator, denominator, proposedIndex) {
  integerInRange(numerator, 'Tolerance numerator', 1, 1000);
  integerInRange(denominator, 'Tolerance denominator', 1, 1000);
  integerInRange(proposedIndex, 'Proposed index', 1, 2000);
  const minimumIndex = Math.max(1, Math.floor(denominator / numerator));
  return {
    tolerance: numerator / denominator,
    minimumIndex,
    proposedIndex,
    firstError: 1 / (proposedIndex + 1),
    // These integer products are exact within this declared domain.
    certifiesTail: (proposedIndex + 1) * numerator > denominator,
    boundaryEquality: (proposedIndex + 1) * numerator === denominator,
    values: Array.from({
      length: 40
    }, (_, index) => {
      const n = index + 1;
      return {
        n,
        value: 3 + (-1) ** n / (n + 1),
        error: 1 / (n + 1)
      };
    })
  };
}
export function dyadicBracket(target, steps) {
  choice(target, 'Squared target', [2, 3, 5]);
  integerInRange(steps, 'Bisection steps', 0, 24);
  let left = target === 5 ? 2 : 1;
  let right = left + 1;
  const history = [{
    step: 0,
    left,
    right
  }];
  for (let step = 1; step <= steps; step += 1) {
    const midpoint = (left + right) / 2;
    if (midpoint * midpoint < target) left = midpoint;else right = midpoint;
    history.push({
      step,
      left,
      right
    });
  }
  const midpoint = (left + right) / 2;
  return {
    target,
    steps,
    left,
    right,
    midpoint,
    leftSquared: left * left,
    rightSquared: right * right,
    width: right - left,
    midpointErrorBound: (right - left) / 2,
    history
  };
}
export function cauchyBlock(n, family) {
  integerInRange(n, 'Block start', 1, 512);
  choice(family, 'Series family', ['harmonic', 'telescoping']);
  const terms = Array.from({
    length: n
  }, (_, index) => {
    const k = n + index + 1;
    return {
      k,
      value: family === 'harmonic' ? 1 / k : 1 / (k * (k + 1))
    };
  });
  return {
    n,
    family,
    terms,
    singleIncrement: family === 'harmonic' ? 1 / (n + 1) : 1 / ((n + 1) * (n + 2)),
    blockTotal: terms.reduce((total, term) => total + term.value, 0),
    analyticalBound: family === 'harmonic' ? 0.5 : 1 / (n + 1),
    exactTelescopingTotal: family === 'telescoping' ? 1 / (n + 1) - 1 / (2 * n + 1) : null
  };
}
export function powerFamily(n, domain, rightEndpoint, fixedPoint) {
  integerInRange(n, 'Power index', 1, 256);
  choice(domain, 'Domain', ['closed-unit', 'open-unit', 'compact-subinterval']);
  finiteInRange(rightEndpoint, 'Compact right endpoint', 0, 0.99);
  const end = domain === 'compact-subinterval' ? rightEndpoint : 1;
  finiteInRange(fixedPoint, 'Fixed point', 0, end);
  if (domain === 'open-unit' && fixedPoint === 1) {
    throw new RangeError('The point 1 is excluded from [0,1).');
  }
  const limitValue = domain === 'closed-unit' && fixedPoint === 1 ? 1 : 0;
  const witness = Math.exp(-Math.LN2 / n);
  const compact = domain === 'compact-subinterval';
  return {
    n,
    domain,
    end,
    fixedPoint,
    fixedValue: fixedPoint ** n,
    limitValue,
    fixedError: Math.abs(fixedPoint ** n - limitValue),
    supremumError: compact ? end ** n : 1,
    supremumAttained: compact,
    witness,
    witnessInDomain: witness <= end,
    witnessValue: 0.5,
    points: Array.from({
      length: 129
    }, (_, index) => {
      const x = end * index / 128;
      return {
        x,
        y: x ** n
      };
    })
  };
}
export function triangleValue(x, n, scaling) {
  finiteInRange(x, 'Triangle input', 0, 1);
  integerInRange(n, 'Triangle index', 2, 4096);
  choice(scaling, 'Triangle scaling', ['unit-height', 'unit-area', 'shrinking-height']);
  const height = scaling === 'unit-area' ? n : scaling === 'shrinking-height' ? 1 / n : 1;
  const local = n * x;
  // Branches retain exact zero outside the support instead of cancellation.
  if (local <= 0 || local >= 2) return 0;
  return height * (local <= 1 ? local : 2 - local);
}
export function triangleFamily(n, scaling, gridIntervals = 20, fixedPoint = 0.25) {
  integerInRange(n, 'Triangle index', 2, 4096);
  integerInRange(gridIntervals, 'Grid intervals', 2, 128);
  finiteInRange(fixedPoint, 'Fixed point', 0, 1);
  choice(scaling, 'Triangle scaling', ['unit-height', 'unit-area', 'shrinking-height']);
  const height = scaling === 'unit-area' ? n : scaling === 'shrinking-height' ? 1 / n : 1;
  const samples = Array.from({
    length: gridIntervals + 1
  }, (_, index) => {
    const x = index / gridIntervals;
    return {
      x,
      y: triangleValue(x, n, scaling)
    };
  });
  return {
    n,
    scaling,
    height,
    peakLocation: 1 / n,
    supportEnd: 2 / n,
    supremumError: height,
    integral: height / n,
    squaredL2Error: 2 * height * height / (3 * n),
    fixedPoint,
    fixedValue: triangleValue(fixedPoint, n, scaling),
    samples,
    sampledMaximum: Math.max(...samples.map(point => point.y)),
    corners: [{
      x: 0,
      y: 0
    }, {
      x: 1 / n,
      y: height
    }, {
      x: 2 / n,
      y: 0
    }, {
      x: 1,
      y: 0
    }]
  };
}
export function derivativeFamily(n, amplitudePower, point = 0) {
  integerInRange(n, 'Oscillation index', 1, 48);
  choice(amplitudePower, 'Amplitude power', [1, 2]);
  finiteInRange(point, 'Inspection point', -Math.PI, Math.PI);
  return {
    n,
    amplitudePower,
    point,
    value: Math.sin(n * point) / n ** amplitudePower,
    derivative: Math.cos(n * point) / n ** (amplitudePower - 1),
    functionBound: 1 / n ** amplitudePower,
    derivativeBound: 1 / n ** (amplitudePower - 1)
  };
}
export function inverseSquareSeries(n, x) {
  integerInRange(n, 'Series terms', 1, 1024);
  finiteInRange(x, 'Series input', -1, 1);
  let sum = 0;
  let derivativeSum = 0;
  let power = 1;
  for (let k = 1; k <= n; k += 1) {
    derivativeSum += power / k;
    power *= x;
    sum += power / (k * k);
  }
  return {
    n,
    x,
    sum,
    derivativeSum,
    uniformTailBound: 1 / n
  };
}
export function bernsteinWeights(n, x) {
  integerInRange(n, 'Polynomial degree', 1, 256);
  finiteInRange(x, 'Polynomial input', 0, 1);
  // Start from the larger endpoint factor, which is at least 0.5^256
  // in this bounded model. Recurrence computes all weights in O(n),
  // without factorials or division by a possibly zero smaller factor.
  const weights = Array(n + 1).fill(0);
  if (x <= 0.5) {
    weights[0] = (1 - x) ** n;
    const ratio = x / (1 - x);
    for (let k = 0; k < n; k += 1) {
      weights[k + 1] = weights[k] * ((n - k) / (k + 1)) * ratio;
    }
  } else {
    weights[n] = x ** n;
    const ratio = (1 - x) / x;
    for (let k = n; k > 0; k -= 1) {
      weights[k - 1] = weights[k] * (k / (n - k + 1)) * ratio;
    }
  }
  return weights;
}
export function bernsteinApproximation(n, x, corner = 0.3, slope = 1) {
  finiteInRange(corner, 'Corner location', 0, 1);
  finiteInRange(slope, 'Lipschitz slope', 0, 4);
  const weights = bernsteinWeights(n, x);
  const nodes = weights.map((weight, k) => ({
    k,
    location: k / n,
    weight,
    value: slope * Math.abs(k / n - corner)
  }));
  const approximation = nodes.reduce((total, node) => total + node.weight * node.value, 0);
  const target = slope * Math.abs(x - corner);
  const integral = nodes.reduce((total, node) => total + node.value, 0) / (n + 1);
  const targetIntegral = slope * (corner * corner + (1 - corner) ** 2) / 2;
  // Take square roots before division so a representable positive bound is
  // not rounded to zero by an underflowing intermediate variance.
  const pointwiseBound = slope * Math.sqrt(x) * Math.sqrt(1 - x) / Math.sqrt(n);
  const uniformBound = slope / (2 * Math.sqrt(n));
  if (slope > 0 && (uniformBound === 0 || x > 0 && x < 1 && pointwiseBound === 0)) {
    throw new RangeError('The positive error bound is too small to represent.');
  }
  return {
    n,
    x,
    corner,
    slope,
    nodes,
    approximation,
    target,
    error: Math.abs(approximation - target),
    pointwiseBound,
    uniformBound,
    weightSum: weights.reduce((total, weight) => total + weight, 0),
    weightedMean: nodes.reduce((total, node) => total + node.location * node.weight, 0),
    integral,
    targetIntegral,
    integralError: Math.abs(integral - targetIntegral)
  };
}
export function typewriterInterval(block, position, observer) {
  integerInRange(block, 'Dyadic block', 0, 12);
  const intervals = 2 ** block;
  integerInRange(position, 'Within-block position', 0, intervals - 1);
  finiteInRange(observer, 'Observer', 0, 1);
  if (observer === 1) throw new RangeError('The observer belongs to [0,1).');
  const left = position / intervals;
  const right = (position + 1) / intervals;
  return {
    block,
    position,
    n: intervals + position,
    intervals,
    left,
    right,
    observer,
    observed: observer >= left && observer < right ? 1 : 0,
    probability: 1 / intervals,
    observerVisitPosition: Math.floor(observer * intervals)
  };
}
