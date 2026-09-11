// Bounded analytic PDE investigations. Profiles are evaluations of stated formulas,
// never a finite-difference time simulation. Physical and normalized coordinates differ.
const PI = Math.PI;
const HEAT_CUTOFF = 63;
function bounded(value, name, minimum, maximum) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
  return value;
}
function choice(value, options, name) {
  if (!options.includes(value)) throw new RangeError(`Choose a valid ${name}.`);
  return value;
}
function integer(value, name, minimum, maximum) {
  bounded(value, name, minimum, maximum);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
function points(start, end, intervals, evaluate) {
  return Array.from({
    length: intervals + 1
  }, (_, index) => {
    const x = start + (end - start) * index / intervals;
    return {
      x,
      ...evaluate(x)
    };
  });
}

// Compensated finite accumulation improves arithmetic; it is not a rigorous error bound.
function sum(values) {
  let total = 0;
  let correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = total + adjusted;
    correction = next - total - adjusted;
    total = next;
  }
  return total;
}
export function controlVolume(time = .25, left = .25, right = .75) {
  bounded(time, 'Time', 0, 1);
  bounded(left, 'Left position', 0, 1);
  bounded(right, 'Right position', 0, 1);
  if (left >= right) throw new RangeError('Choose an interval with left < right.');
  const field = x => 1 + x + 2 * x * x + time * (3 - x);
  const flux = x => -(1 + x) * (1 + 4 * x - time);
  const source = x => time - 2 - 9 * x;
  const width = right - left;
  // Factor differences before evaluation so even a very thin legal slice avoids
  // subtracting nearly equal squared endpoints or endpoint fluxes.
  const sourceIntegral = width * (time - 2 - 4.5 * (right + left));
  const accumulation = width * (3 - (right + left) / 2);
  const netInflow = width * (5 - time + 4 * (right + left));
  return {
    time,
    left,
    right,
    leftFlux: flux(left),
    rightFlux: flux(right),
    sourceIntegral,
    accumulation,
    balance: netInflow + sourceIntegral,
    profile: points(0, 1, 64, x => ({
      value: field(x),
      flux: flux(x),
      source: source(x)
    }))
  };
}
export function transportValue(x, time, curvature = 2) {
  bounded(x, 'Position', 0, 1);
  bounded(time, 'Time', 0, 1);
  choice(curvature, [0, 2], 'inflow curvature');
  const fromInitial = x >= time;
  const dataTime = fromInitial ? 0 : time - x;
  const dataPosition = fromInitial ? x - time : 0;
  const value = fromInitial ? 1 + x - time : 1 - dataTime + curvature * dataTime ** 2;
  return {
    x,
    time,
    value,
    fromInitial,
    dataTime,
    dataPosition
  };
}
export function transportState(time = .35, x = .6, curvature = 2) {
  const selected = transportValue(x, time, curvature);
  return {
    ...selected,
    curvature,
    mass: 1.5 - time + curvature * time ** 3 / 3,
    massDerivative: -1 + curvature * time ** 2,
    incoming: 1 - time + curvature * time ** 2,
    outgoing: 2 - time,
    profile: points(0, 1, 128, position => transportValue(position, time, curvature))
  };
}
export function heatCoefficient(index) {
  integer(index, 'Harmonic index', 1, 255);
  return index % 2 === 0 ? 0 : -8 / (PI * index * (index ** 2 - 4));
}
export function boundaryMode(boundary = 'dirichlet', index = 1) {
  choice(boundary, ['dirichlet', 'neumann', 'mixed'], 'boundary');
  integer(index, 'Mode index', boundary === 'dirichlet' ? 1 : 0, 8);
  const frequency = boundary === 'mixed' ? index + .5 : index;
  const eigenvalue = (frequency * PI) ** 2;
  const isCosine = boundary === 'neumann';
  return {
    boundary,
    index,
    frequency,
    eigenvalue,
    profile: points(0, 1, 128, x => ({
      value: isCosine ? Math.cos(frequency * PI * x) : Math.sin(frequency * PI * x)
    }))
  };
}
function validateHeatTime(theta) {
  bounded(theta, 'Normalized heat time', 0, .5);
  if (theta !== 0 && theta < .002) throw new RangeError('Choose the exact initial state or positive time at least 0.002.');
}
export function heatValue(x, theta, boundary = 'dirichlet') {
  bounded(x, 'Normalized position', 0, 1);
  validateHeatTime(theta);
  choice(boundary, ['dirichlet', 'neumann'], 'heat boundary');
  if (theta === 0) return x === 0 || x === 1 ? 0 : Math.sin(PI * x) ** 2;
  if (boundary === 'neumann') return .5 - .5 * Math.exp(-4 * PI ** 2 * theta) * Math.cos(2 * PI * x);
  // Homogeneous endpoint values are exact parts of the data, not rounded sin(n*pi).
  if (x === 0 || x === 1) return 0;
  return sum(Array.from({
    length: 32
  }, (_, k) => {
    const n = 2 * k + 1;
    return heatCoefficient(n) * Math.exp(-n * n * PI ** 2 * theta) * Math.sin(n * PI * x);
  }));
}
export function heatBoundaryState(theta = .05) {
  validateHeatTime(theta);
  const modes = Array.from({
    length: 32
  }, (_, k) => {
    const n = 2 * k + 1;
    return {
      n,
      coefficient: heatCoefficient(n),
      amplitude: heatCoefficient(n) * Math.exp(-n * n * PI ** 2 * theta)
    };
  });
  const meanDirichlet = theta === 0 ? .5 : sum(modes.map(mode => 2 * mode.amplitude / (mode.n * PI)));
  const squareDirichlet = theta === 0 ? .375 : sum(modes.map(mode => mode.amplitude ** 2 / 2));
  const eachOutflow = theta === 0 ? 0 : sum(modes.map(mode => mode.amplitude * mode.n * PI));
  return {
    theta,
    cutoff: HEAT_CUTOFF,
    dirichlet: {
      mean: meanDirichlet,
      squaredNorm: squareDirichlet,
      leftOutflow: eachOutflow,
      rightOutflow: eachOutflow
    },
    neumann: {
      mean: .5,
      squaredNorm: .25 + Math.exp(-8 * PI ** 2 * theta) / 8,
      leftOutflow: 0,
      rightOutflow: 0
    },
    // Natural log of a positive analytic truncation bound. Floating error is separate.
    logTailBound: theta === 0 ? null : Math.log(36 / (5 * PI * HEAT_CUTOFF ** 2)) - (HEAT_CUTOFF + 1) ** 2 * PI ** 2 * theta,
    modes,
    profile: points(0, 1, 128, x => ({
      initial: Math.sin(PI * x) ** 2,
      dirichlet: heatValue(x, theta),
      neumann: heatValue(x, theta, 'neumann')
    }))
  };
}
export function heatKernel(x, time, diffusivity = 1) {
  bounded(x, 'Kernel position', -20, 20);
  bounded(time, 'Kernel time', .01, 2);
  bounded(diffusivity, 'Diffusivity', .1, 2);
  return Math.exp(-x * x / (4 * diffusivity * time)) / Math.sqrt(4 * PI * diffusivity * time);
}
export function compactBump(x) {
  bounded(x, 'Bump position', -10, 10);
  return Math.abs(x) < 1 ? (1 - x * x) ** 3 : 0;
}
function shiftedBump(center, offset) {
  const rightDistance = (1 - center) - offset;
  const leftDistance = (1 + center) + offset;
  return rightDistance > 0 && leftDistance > 0 ? (rightDistance * leftDistance) ** 3 : 0;
}
function bumpWindowIntegral(center, halfWidth) {
  const low = Math.max(-halfWidth, -1 - center);
  const high = Math.min(halfWidth, 1 - center);
  if (high <= low) return 0;
  const width = high - low;
  const rightAtLow = Math.max(0, (1 - center) - low);
  const rightAtHigh = Math.max(0, (1 - center) - high);
  const leftAtLow = Math.max(0, (1 + center) + low);
  const leftAtHigh = Math.max(0, (1 + center) + high);
  const chooseThree = [1, 3, 3, 1];
  const chooseSix = [1, 6, 15, 20, 15, 6, 1];
  // Expand both positive linear endpoint-distance factors in Bernstein form.
  // Integral_0^1 s^r(1-s)^(6-r) ds = 1 / (7 choose(6,r)).
  // All 16 terms are nonnegative; no nearly equal primitive values subtract.
  let average = 0;
  for (let first = 0; first <= 3; first += 1) {
    for (let second = 0; second <= 3; second += 1) {
      average += chooseThree[first] * chooseThree[second] / (7 * chooseSix[first + second])
        * rightAtLow ** (3 - first) * rightAtHigh ** first
        * leftAtLow ** (3 - second) * leftAtHigh ** second;
    }
  }
  return width * average;
}
export function waveValue(x, time, velocityFactor = .5) {
  bounded(x, 'Wave position', -3, 3);
  bounded(time, 'Wave time', 0, 1);
  choice(velocityFactor, [0, .5], 'initial velocity');
  const leftFoot = x - time;
  const rightFoot = x + time;
  const rightMoving = shiftedBump(x, -time) / 2;
  const leftMoving = shiftedBump(x, time) / 2;
  const velocityContribution = velocityFactor * bumpWindowIntegral(x, time) / 2;
  return {
    x,
    time,
    leftFoot,
    rightFoot,
    rightMoving,
    leftMoving,
    velocityContribution,
    total: rightMoving + leftMoving + velocityContribution
  };
}
export function waveState(time = .6, x = .4, velocityFactor = .5) {
  const selected = waveValue(x, time, velocityFactor);
  return {
    ...selected,
    velocityFactor,
    profile: points(-3, 3, 192, position => waveValue(position, time, velocityFactor))
  };
}
export function standingWave(time = .25) {
  bounded(time, 'Standing-wave time', 0, 2);
  const first = Math.cos(PI * time);
  const second = .25 * Math.sin(2 * PI * time);
  const kinetic = PI ** 2 * Math.sin(PI * time) ** 2 / 4 + PI ** 2 * Math.cos(2 * PI * time) ** 2 / 16;
  const strain = PI ** 2 * Math.cos(PI * time) ** 2 / 4 + PI ** 2 * Math.sin(2 * PI * time) ** 2 / 16;
  return {
    time,
    first,
    second,
    kinetic,
    strain,
    total: kinetic + strain,
    profile: points(0, 1, 128, x => ({
      value: first * Math.sin(PI * x) + second * Math.sin(2 * PI * x)
    }))
  };
}
export const POISSON_SOURCES = {
  uniform: {
    a: 2,
    b: 0,
    label: 'Uniform source f(x)=2'
  },
  balanced: {
    a: -1,
    b: 2,
    label: 'Balanced source f(x)=−1+2x'
  },
  sloped: {
    a: 3,
    b: -2,
    label: 'Sloped source f(x)=3−2x'
  }
};
export function poissonState(sourceName = 'uniform', boundary = 'neumann', left = 1, right = 1, mean = 0) {
  choice(sourceName, Object.keys(POISSON_SOURCES), 'Poisson source');
  choice(boundary, ['dirichlet', 'neumann'], 'Poisson boundary');
  // Exact small integers retain exact compatibility; no unexplained numeric tolerance.
  integer(left, 'Left boundary datum', -2, 2);
  integer(right, 'Right boundary datum', -2, 2);
  integer(mean, 'Selected mean', -1, 1);
  const {
    a,
    b,
    label
  } = POISSON_SOURCES[sourceName];
  const sourceIntegral = a + b / 2;
  const compatible = boundary === 'dirichlet' || left + right === sourceIntegral;
  if (!compatible) return {
    sourceName,
    boundary,
    left,
    right,
    mean,
    a,
    b,
    label,
    sourceIntegral,
    compatible: false,
    profile: [],
    mismatch: left + right - sourceIntegral
  };
  const slope = boundary === 'neumann' ? left : right - left + a / 2 + b / 6;
  const constant = boundary === 'neumann' ? mean - left / 2 + a / 6 + b / 24 : left;
  const value = x => constant + slope * x - a * x * x / 2 - b * x ** 3 / 6;
  const selectedMean = constant + slope / 2 - a / 6 - b / 24;
  return {
    sourceName,
    boundary,
    left,
    right,
    mean,
    a,
    b,
    label,
    sourceIntegral,
    compatible,
    slope,
    constant,
    selectedMean,
    leftOutflow: slope,
    rightOutflow: a + b / 2 - slope,
    profile: points(0, 1, 96, x => ({
      value: value(x),
      flux: -slope + a * x + b * x * x / 2,
      source: a + b * x
    }))
  };
}
export function harmonicValue(x, y, frequency = 1) {
  bounded(x, 'Horizontal position', 0, 1);
  bounded(y, 'Depth position', 0, 1);
  choice(frequency, [1, 3, 5], 'boundary frequency');
  if (x === 0 || x === 1 || y === 0) return 0;
  return Math.sin(frequency * PI * x) * Math.sinh(frequency * PI * y) / Math.sinh(frequency * PI);
}
export function harmonicState(frequency = 1) {
  choice(frequency, [1, 3, 5], 'boundary frequency');
  const rows = 24;
  const columns = 24;
  const cells = Array.from({
    length: rows * columns
  }, (_, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = (column + .5) / columns;
    const y = (row + .5) / rows;
    return {
      row,
      column,
      x,
      y,
      value: harmonicValue(x, y, frequency)
    };
  });
  return {
    frequency,
    rows,
    columns,
    cells,
    midpointRatio: Math.sinh(frequency * PI / 2) / Math.sinh(frequency * PI)
  };
}
export function pointSourceState(location = 1 / 3) {
  bounded(location, 'Source position', .1, .9);
  return {
    location,
    leftSlope: 1 - location,
    rightSlope: -location,
    slopeJump: -1,
    peak: location * (1 - location),
    energyIntegral: location * (1 - location),
    profile: [{
      x: 0,
      value: 0
    }, {
      x: location,
      value: location * (1 - location)
    }, {
      x: 1,
      value: 0
    }]
  };
}
export function burgersState(left = 2, right = 0, time = .5, showExpansion = false) {
  integer(left, 'Left state', -2, 3);
  integer(right, 'Right state', -2, 3);
  bounded(time, 'Shock time', 0, 1);
  if (typeof showExpansion !== 'boolean') throw new RangeError('Expansion display must be true or false.');
  const speed = (left + right) / 2;
  const isShock = left > right;
  const constant = left === right;
  const entropyProduction = (right ** 3 - left ** 3) / 3 - speed * (right ** 2 - left ** 2) / 2;
  const leftX = -3;
  const rightX = 4;
  let profile;
  if (constant) profile = [{
    x: leftX,
    value: left
  }, {
    x: rightX,
    value: right
  }];else if (isShock || time === 0 || showExpansion) {
    const jump = time * speed;
    // Duplicate x coordinates draw the actual jump, not a smoothed connection.
    profile = [{
      x: leftX,
      value: left
    }, {
      x: jump,
      value: left
    }, {
      x: jump,
      value: right
    }, {
      x: rightX,
      value: right
    }];
  } else {
    profile = [{
      x: leftX,
      value: left
    }, {
      x: time * left,
      value: left
    }, {
      x: time * right,
      value: right
    }, {
      x: rightX,
      value: right
    }];
  }
  return {
    left,
    right,
    time,
    speed,
    isShock,
    constant,
    showExpansion,
    entropyProduction,
    profile,
    admissible: !showExpansion || left >= right || time === 0,
    leftFront: time * left,
    rightFront: time * right,
    jumpPosition: time * speed
  };
}
export function inverseHeatState(index = 6, time = .03) {
  integer(index, 'Inverse mode', 1, 12);
  bounded(time, 'Observation time', .002, .05);
  const logAttenuation = -(index ** 2) * PI ** 2 * time;
  const attenuation = Math.exp(logAttenuation);
  return {
    index,
    time,
    logAttenuation,
    logGain: -logAttenuation,
    attenuation,
    gain: Math.exp(-logAttenuation),
    initialL2: 1 / Math.sqrt(2),
    finalL2: attenuation / Math.sqrt(2),
    profile: points(0, 1, 192, x => ({
      initial: Math.sin(index * PI * x),
      final: attenuation * Math.sin(index * PI * x)
    }))
  };
}
export function periodicDepth(depth, phase = 0) {
  bounded(depth, 'Depth in penetration units', 0, 4);
  bounded(phase, 'Surface phase', 0, 2 * PI);
  return {
    depth,
    phase,
    envelope: Math.exp(-depth),
    value: Math.exp(-depth) * Math.cos(phase - depth),
    lag: depth
  };
}
export function forcedRod(time = 0, amplitude = .2, tolerance = .05, length = 1) {
  bounded(time, 'Rod time', 0, 20);
  bounded(amplitude, 'Initial mode amplitude', 0, 1);
  bounded(tolerance, 'Uniform tolerance', .001, .2);
  bounded(length, 'Rod length', .25, 2);
  const diffusivity = .25;
  const rate = diffusivity * PI ** 2 / length ** 2;
  const excess = amplitude * Math.exp(-rate * time);
  const settlingTime = amplitude <= tolerance ? 0 : Math.log(amplitude / tolerance) / rate;
  // Keep source=1 and k=.5 fixed: the steady temperature scales as L².
  const mean = length ** 2 / 6 + 2 * excess / PI;
  const eachOutflow = length / 2 + .5 * excess * PI / length;
  return {
    time,
    amplitude,
    tolerance,
    length,
    diffusivity,
    rate,
    excess,
    settlingTime,
    mean,
    totalSource: length,
    eachOutflow,
    accumulation: -4 * length * rate * excess / PI,
    profile: points(0, length, 96, x => {
      const xi = x / length;
      const steady = length ** 2 * xi * (1 - xi);
      return {
        steady,
        value: steady + excess * Math.sin(PI * xi)
      };
    })
  };
}
