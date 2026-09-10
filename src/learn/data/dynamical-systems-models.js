// Bounded teaching models. These return numerical examples, never a certificate
// that an arbitrary observed system is stable, chaotic, or accurately modeled.
function boundedNumber(value, minimum, maximum, name) {
  if (typeof value !== "number" || !Number.isFinite(value) ||
      value < minimum || value > maximum) {
    throw new RangeError(name + " is outside this investigation's supported range.");
  }
  return value;
}

function boundedInteger(value, minimum, maximum, name) {
  boundedNumber(value, minimum, maximum, name);
  if (!Number.isInteger(value)) throw new RangeError(name + " must be an integer.");
  return value;
}

function initialVector(value, size, maximum = 2) {
  if (!Array.isArray(value) || value.length !== size) {
    throw new RangeError("The initial state has the wrong dimension.");
  }
  return value.map(component => boundedNumber(component, -maximum, maximum, "Initial component"));
}

function scalarRk4(field, value, step) {
  const first = field(value);
  const second = field(value + step * first / 2);
  const third = field(value + step * second / 2);
  const fourth = field(value + step * third);
  return value + step * (first + 2 * second + 2 * third + fourth) / 6;
}

export function coolingTrace({ decay = 0.5, step = 0.5, steps = 12, initial = 8 } = {}) {
  boundedNumber(decay, 0, 2, "Decay rate");
  boundedNumber(step, 0.05, 2, "Step");
  boundedInteger(steps, 0, 60, "Steps");
  boundedNumber(initial, -100, 100, "Initial excess temperature");
  if (initial !== 0 && Math.abs(initial) < 1e-8) {
    throw new RangeError("A nonzero initial temperature must have magnitude at least 1e-8.");
  }
  const multiplier = 1 - decay * step;
  return {
    multiplier,
    exactMultiplier: Math.exp(-decay * step),
    attractingEuler: Math.abs(multiplier) < 1,
    rows: Array.from({ length: steps + 1 }, (_, index) => ({
      time: index * step,
      exact: initial * Math.exp(-decay * index * step),
      euler: initial * multiplier ** index,
    })),
  };
}

export const CUBIC_FOLD = 2 / (3 * Math.sqrt(3));

function scalarDefinition(kind, parameter) {
  if (kind === "pitchfork") {
    boundedNumber(parameter, -1, 1, "Pitchfork parameter");
    return {
      field: value => parameter * value - value ** 3,
      potential: value => value ** 4 / 4 - parameter * value * value / 2,
      roots: parameter > 0 ? [-Math.sqrt(parameter), 0, Math.sqrt(parameter)] : [0],
    };
  }
  if (kind !== "tilted") throw new RangeError("Unknown scalar rule.");
  boundedNumber(parameter, -0.6, 0.6, "Tilt");
  let roots;
  if (parameter === 0) {
    roots = [-1, 0, 1];
  } else if (Math.abs(parameter) === CUBIC_FOLD) {
    const sign = Math.sign(parameter);
    roots = [-sign / Math.sqrt(3), 2 * sign / Math.sqrt(3)].sort((left, right) => left - right);
  } else if (Math.abs(parameter) < CUBIC_FOLD) {
    const angle = Math.acos(parameter / CUBIC_FOLD);
    roots = [0, 1, 2].map(index =>
      2 / Math.sqrt(3) * Math.cos((angle + 2 * Math.PI * index) / 3),
    ).sort((left, right) => left - right);
  } else {
    const radical = Math.sqrt(parameter * parameter / 4 - 1 / 27);
    roots = [Math.cbrt(parameter / 2 + radical) + Math.cbrt(parameter / 2 - radical)];
  }
  return {
    field: value => parameter + value - value ** 3,
    potential: value => value ** 4 / 4 - value * value / 2 - parameter * value,
    roots,
  };
}

export function scalarFlowState({
  kind = "pitchfork", parameter = 1, initial = 0.2, duration = 8, steps = 400,
} = {}) {
  boundedNumber(initial, -2, 2, "Initial state");
  boundedNumber(duration, 0, 12, "Duration");
  boundedInteger(steps, 1, 600, "Steps");
  if (duration / steps > 0.025) throw new RangeError("Use a step of at most 0.025.");
  const { field, potential, roots } = scalarDefinition(kind, parameter);
  const equilibria = roots.map(value => {
    let stability;
    if (kind === "pitchfork") {
      stability = parameter <= 0 || value !== 0 ? "attracting" : "repelling";
    } else if (Math.abs(parameter) === CUBIC_FOLD &&
               Math.abs(value) < 1) {
      stability = "attracting from one side";
    } else {
      stability = 1 - 3 * value * value < 0 ? "attracting" : "repelling";
    }
    return {
      value,
      slope: kind === "pitchfork" ? parameter - 3 * value * value : 1 - 3 * value * value,
      stability,
    };
  });
  const rows = [{ time: 0, value: initial, potential: potential(initial) }];
  const step = duration / steps;
  let value = initial;
  for (let index = 1; index <= steps; index += 1) {
    value = scalarRk4(field, value, step);
    if (!Number.isFinite(value) || Math.abs(value) > 3) {
      throw new RangeError("The scalar numerical path left its supported region.");
    }
    rows.push({ time: index * step, value, potential: potential(value) });
  }
  return {
    kind, parameter, equilibria, rows, step,
    curve: Array.from({ length: 161 }, (_, index) => {
      const point = -2 + index / 40;
      return { value: point, rate: field(point), potential: potential(point) };
    }),
  };
}

export function planarField(mode, state, parameter = 0.4) {
  const [horizontal, vertical] = initialVector(state, 2, 1000);
  boundedNumber(parameter, -1, 1, "Planar parameter");
  switch (mode) {
    case "center": return [-vertical, horizontal];
    case "spiral": return [-0.4 * horizontal - vertical, horizontal - 0.4 * vertical];
    case "saddle": return [0.4 * horizontal, -0.5 * vertical];
    case "transient": return [-horizontal + 6 * vertical, -2 * vertical];
    case "hopf": {
      const radiusSquared = horizontal * horizontal + vertical * vertical;
      return [(parameter - radiusSquared) * horizontal - vertical,
        horizontal + (parameter - radiusSquared) * vertical];
    }
    default: throw new RangeError("Unknown planar rule.");
  }
}

export function planarTrace({
  mode = "hopf", parameter = 0.4, initial = [0.2, 0], duration = 12, steps = 480,
} = {}) {
  const [horizontal, vertical] = initialVector(initial, 2);
  boundedNumber(parameter, -1, 1, "Planar parameter");
  boundedNumber(duration, 0, 20, "Duration");
  boundedInteger(steps, 1, 800, "Steps");
  planarField(mode, initial, parameter);
  const initialRadiusSquared = horizontal * horizontal + vertical * vertical;
  const rows = Array.from({ length: steps + 1 }, (_, index) => {
    const time = duration * index / steps;
    let position;
    if (mode === "saddle") {
      position = [horizontal * Math.exp(0.4 * time), vertical * Math.exp(-0.5 * time)];
    } else if (mode === "transient") {
      position = [
        horizontal * Math.exp(-time) + 6 * vertical * Math.exp(-time) * (-Math.expm1(-time)),
        vertical * Math.exp(-2 * time),
      ];
    } else {
      let scale = 1;
      if (mode === "spiral") scale = Math.exp(-0.4 * time);
      if (mode === "hopf") {
        scale = parameter === 0 ?
          1 / Math.sqrt(1 + 2 * initialRadiusSquared * time) :
          Math.sqrt(Math.exp(2 * parameter * time) /
            (1 + initialRadiusSquared * Math.expm1(2 * parameter * time) / parameter));
      }
      position = [
        scale * (horizontal * Math.cos(time) - vertical * Math.sin(time)),
        scale * (horizontal * Math.sin(time) + vertical * Math.cos(time)),
      ];
    }
    return { time, position, radius: Math.hypot(...position) };
  });
  return {
    mode, parameter, rows,
    cycleRadius: mode === "hopf" && parameter > 0 ? Math.sqrt(parameter) : null,
    radialReturnMultiplier: mode === "hopf" && parameter > 0 ? Math.exp(-4 * Math.PI * parameter) : null,
  };
}

const logisticStep = (value, growth) => growth * value * (1 - value);

export function logisticTrace({ growth = 2.5, initial = 0.2, steps = 24 } = {}) {
  boundedNumber(growth, 0, 4, "Growth parameter");
  boundedNumber(initial, 0, 1, "Initial population");
  boundedInteger(steps, 0, 5000, "Iterations");
  const values = [initial];
  for (let index = 0; index < steps; index += 1) {
    values.push(logisticStep(values[index], growth));
  }
  const fixedPoints = [{ value: 0, multiplier: growth }];
  if (growth > 1) fixedPoints.push({ value: 1 - 1 / growth, multiplier: 2 - growth });
  const twoCycle = growth > 3 ? {
    values: [-1, 1].map(sign =>
      (growth + 1 + sign * Math.sqrt((growth - 3) * (growth + 1))) / (2 * growth)),
    multiplier: 4 + 2 * growth - growth * growth,
  } : null;
  return { growth, values, fixedPoints, twoCycle };
}

export function logisticSensitivity({
  growth = 3.9, initial = 0.2, difference = 1e-6, steps = 70,
} = {}) {
  boundedNumber(difference, -0.01, 0.01, "Initial difference");
  if (difference !== 0 && Math.abs(difference) < 1e-12) {
    throw new RangeError("A nonzero difference must have magnitude at least 1e-12.");
  }
  boundedInteger(steps, 0, 160, "Sensitivity iterations");
  const reference = logisticTrace({ growth, initial, steps }).values;
  const other = logisticTrace({ growth, initial: initial + difference, steps }).values;
  if (difference !== 0 && reference[0] === other[0]) {
    throw new RangeError("The initial states cannot be distinguished in this arithmetic.");
  }
  let logGain = 0;
  let tangentVanished = false;
  const actualInitialDifference = other[0] - reference[0];
  const rows = reference.map((value, index) => {
    if (index > 0) {
      const multiplier = growth * (1 - 2 * reference[index - 1]);
      if (multiplier === 0) tangentVanished = true;
      if (!tangentVanished) logGain += Math.log(Math.abs(multiplier));
    }
    const separation = Math.abs(other[index] - value);
    return {
      iteration: index, reference: value, other: other[index], separation,
      logSeparation: separation === 0 ? null : Math.log(separation),
      logTangentSeparation: tangentVanished || difference === 0 ? null :
        Math.log(Math.abs(actualInitialDifference)) + logGain,
      logGain: tangentVanished ? null : logGain,
      tangentVanished,
    };
  });
  return { rows, actualInitialDifference };
}

export function logisticLyapunovEstimate({ growth = 3.9, initial = 0.217, burn = 1000, samples = 2000 } = {}) {
  boundedInteger(burn, 0, 5000, "Discarded iterations");
  boundedInteger(samples, 1, 5000, "Averaged iterations");
  boundedNumber(growth, 0, 4, "Growth parameter");
  boundedNumber(initial, 0, 1, "Initial population");
  let value = initial;
  for (let index = 0; index < burn; index += 1) value = logisticStep(value, growth);
  let logSum = 0;
  let zeroDerivatives = 0;
  for (let index = 0; index < samples; index += 1) {
    const magnitude = Math.abs(growth * (1 - 2 * value));
    if (magnitude === 0) zeroDerivatives += 1;
    else logSum += Math.log(magnitude);
    value = logisticStep(value, growth);
  }
  return { estimate: zeroDerivatives > 0 ? null : logSum / samples, zeroDerivatives, burn, samples };
}

export function bifurcationAtlas({ minimum = 2.5, maximum = 4, columns = 151, burn = 1000, retained = 48, initial = 0.217 } = {}) {
  boundedNumber(minimum, 0, 4, "Minimum growth");
  boundedNumber(maximum, 0, 4, "Maximum growth");
  if (minimum >= maximum) throw new RangeError("The growth interval must have positive width.");
  boundedInteger(columns, 2, 241, "Parameter samples");
  boundedInteger(burn, 0, 1500, "Discarded iterations");
  boundedInteger(retained, 1, 64, "Retained iterations");
  boundedNumber(initial, 0, 1, "Initial population");
  return {
    minimum, maximum, columns, burn, retained, initial,
    rows: Array.from({ length: columns }, (_, index) => {
      const growth = minimum + (maximum - minimum) * index / (columns - 1);
      let value = initial;
      for (let count = 0; count < burn; count += 1) value = logisticStep(value, growth);
      const values = [];
      for (let count = 0; count < retained; count += 1) {
        value = logisticStep(value, growth);
        values.push(value);
      }
      return { growth, values };
    }),
  };
}

export function tentStep(value) {
  boundedNumber(value, 0, 1, "Tent coordinate");
  return value <= 0.5 ? 2 * value : 2 * (1 - value);
}

export function logisticCoordinate(value) {
  boundedNumber(value, 0, 1, "Tent coordinate");
  return Math.sin(Math.PI * value / 2) ** 2;
}

export function logisticInvariantCdf(value) {
  boundedNumber(value, 0, 1, "Logistic coordinate");
  return 2 / Math.PI * Math.asin(Math.sqrt(value));
}

export function tentCylinder(word) {
  if (typeof word !== "string" || !/^[LR]{1,10}$/.test(word)) {
    throw new RangeError("A cylinder word needs one to ten L/R symbols.");
  }
  let slope = 1;
  let intercept = 0;
  for (const branch of word) {
    if (branch === "R") intercept += slope;
    slope *= branch === "L" ? 0.5 : -0.5;
  }
  return {
    word, slope, intercept,
    interval: [intercept, intercept + slope].sort((left, right) => left - right),
    periodicPoint: intercept / (1 - slope),
  };
}

export function quantizedTentTrace({ bits = 8, numerator = 51, steps = 12 } = {}) {
  boundedInteger(bits, 1, 20, "Fraction bits");
  const denominator = 2 ** bits;
  boundedInteger(numerator, 0, denominator, "Initial numerator");
  boundedInteger(steps, 0, 40, "Quantized iterations");
  const rows = [];
  let current = numerator;
  for (let index = 0; index <= steps; index += 1) {
    rows.push({ iteration: index, numerator: current, denominator, value: current / denominator });
    current = Math.min(2 * current, 2 * (denominator - current));
  }
  return { bits, denominator, rows };
}

export function oscillatorTrace({ step = 0.2, steps = 100, initial = [1, 0] } = {}) {
  boundedNumber(step, 0.01, 2.5, "Oscillator step");
  boundedInteger(steps, 0, 1000, "Oscillator steps");
  if (steps * Math.log1p(step + step * step) > 300) {
    throw new RangeError("This step/count combination exceeds the supported arithmetic range.");
  }
  const [initialPosition, initialMomentum] = initialVector(initial, 2);
  let euler = [initialPosition, initialMomentum];
  let symplectic = [...euler];
  const rows = [];
  const energy = state => (state[0] ** 2 + state[1] ** 2) / 2;
  const modifiedEnergy = state => (state[0] ** 2 + state[1] ** 2 - step * state[0] * state[1]) / 2;
  for (let index = 0; index <= steps; index += 1) {
    const time = index * step;
    const exact = [
      initialPosition * Math.cos(time) + initialMomentum * Math.sin(time),
      initialMomentum * Math.cos(time) - initialPosition * Math.sin(time),
    ];
    rows.push({
      time, exact, euler: [...euler], symplectic: [...symplectic],
      exactEnergy: energy(exact), eulerEnergy: energy(euler),
      symplecticEnergy: energy(symplectic), modifiedEnergy: modifiedEnergy(symplectic),
    });
    euler = [euler[0] + step * euler[1], euler[1] - step * euler[0]];
    const nextMomentum = symplectic[1] - step * symplectic[0];
    symplectic = [symplectic[0] + step * nextMomentum, nextMomentum];
  }
  return { step, steps, rows, symplecticStrictlyStable: step < 2 };
}

export function lorenzField(state, rho = 28) {
  const [first, second, third] = initialVector(state, 3, 1000);
  boundedNumber(rho, 0.5, 28, "Lorenz rho");
  return [10 * (second - first), first * (rho - third) - second, first * second - 8 * third / 3];
}

export function lorenzTrace({ rho = 28, initial = [1, 1, 1], duration = 24, steps = 4800 } = {}) {
  boundedNumber(rho, 0.5, 28, "Lorenz rho");
  boundedNumber(duration, 0, 30, "Lorenz duration");
  boundedInteger(steps, 1, 12000, "Lorenz steps");
  const step = duration / steps;
  if (step > 0.01) throw new RangeError("Use a Lorenz step of at most 0.01.");
  let state = initialVector(initial, 3, 20);
  const rows = [{ time: 0, state: [...state] }];
  const advance = (position, slope, scale) => position.map((value, index) => value + scale * slope[index]);
  for (let index = 1; index <= steps; index += 1) {
    const first = lorenzField(state, rho);
    const second = lorenzField(advance(state, first, step / 2), rho);
    const third = lorenzField(advance(state, second, step / 2), rho);
    const fourth = lorenzField(advance(state, third, step), rho);
    state = state.map((value, coordinate) =>
      value + step * (first[coordinate] + 2 * second[coordinate] + 2 * third[coordinate] + fourth[coordinate]) / 6);
    initialVector(state, 3, 1000);
    rows.push({ time: index * step, state: [...state] });
  }
  const equilibria = [[0, 0, 0]];
  if (rho > 1) {
    const root = Math.sqrt(8 * (rho - 1) / 3);
    equilibria.push([root, root, rho - 1], [-root, -root, rho - 1]);
  }
  return { rho, step, rows, equilibria, divergence: -10 - 1 - 8 / 3 };
}
