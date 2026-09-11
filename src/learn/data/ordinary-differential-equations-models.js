// Bounded deterministic teaching models. These are not general-purpose ODE solvers.
export function odeNumber(value, name, minimum = -1e6, maximum = 1e6) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
  return value;
}
export function formatOdeNumber(value, digits = 4) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 1e6) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}
export function coolingState(time, initial = 40, rate = 0.2, equilibrium = 0) {
  odeNumber(time, 'time', 0, 1000);
  odeNumber(initial, 'initial state');
  odeNumber(rate, 'rate', 0, 100);
  odeNumber(equilibrium, 'equilibrium');
  const elapsedRate = rate * time;
  if (time > 0 && rate > 0 && elapsedRate === 0 && initial !== equilibrium) {
    throw new RangeError('The rate-time product falls below the supported arithmetic range.');
  }
  const state = time === 0 || rate === 0 || initial === equilibrium
    ? initial
    : initial * Math.exp(-elapsedRate) + equilibrium * -Math.expm1(-elapsedRate);
  return {
    time,
    state,
    derivative: -rate * (state - equilibrium)
  };
}
export function logisticState(time, initial, rate = 1, capacity = 10) {
  odeNumber(time, 'time', 0, 1000);
  odeNumber(initial, 'initial population', 0, 1e6);
  odeNumber(rate, 'growth rate', 0, 100);
  odeNumber(capacity, 'capacity', 1e-6, 1e6);
  const decay = Math.exp(-rate * time);
  let state;
  if (initial === 0 || time === 0 || rate === 0 || initial === capacity) state = initial;else if (initial < capacity) {
    const logOdds = Math.log(capacity - initial) - Math.log(initial) - rate * time;
    const logDenominator = logOdds > 0 ? logOdds + Math.log1p(Math.exp(-logOdds)) : Math.log1p(Math.exp(logOdds));
    state = Math.exp(Math.log(capacity) - logDenominator);
  } else state = capacity / (capacity / initial * decay - Math.expm1(-rate * time));
  return {
    time,
    state,
    derivative: rate * state * (1 - state / capacity)
  };
}
export function waitingState(time, departure) {
  odeNumber(time, 'time', 0, 10);
  odeNumber(departure, 'departure', 0, 10);
  const elapsed = Math.max(0, time - departure);
  return {
    time,
    state: elapsed ** 2,
    derivative: 2 * elapsed
  };
}
export function blowupState(time, initial = 1) {
  odeNumber(time, 'time', -100, 100);
  odeNumber(initial, 'initial state', 0, 100);
  const denominator = 1 - initial * time;
  if (denominator <= 0) return {
    time,
    state: null,
    status: 'outside the solution interval through t = 0'
  };
  return {
    time,
    state: initial / denominator,
    status: 'inside the solution interval'
  };
}
function vector2(values, name) {
  if (!Array.isArray(values) || values.length !== 2) throw new TypeError(`${name} must have two entries.`);
  for (let index = 0; index < 2; index += 1) {
    if (!Object.hasOwn(values, index)) throw new TypeError(`${name} must contain its own two entries.`);
    odeNumber(values[index], `${name}[${index}]`);
  }
  return values;
}
function matrix2(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 2) throw new TypeError('The matrix must be 2 × 2.');
  for (let row = 0; row < 2; row += 1) {
    if (!Object.hasOwn(matrix, row)) throw new TypeError('The matrix must contain its own two rows.');
    vector2(matrix[row], 'matrix row');
    for (let column = 0; column < 2; column += 1) odeNumber(matrix[row][column], 'matrix entry', -20, 20);
  }
  return matrix;
}
export function applyOdeMatrix(matrix, vector) {
  matrix2(matrix);
  vector2(vector, 'vector');
  return matrix.map(row => row[0] * vector[0] + row[1] * vector[1]);
}
function sineRatio(value) {
  return Math.abs(value) < 1e-4 ? 1 - value ** 2 / 6 + value ** 4 / 120 : Math.sin(value) / value;
}
function hyperbolicSineRatio(value) {
  return Math.abs(value) < 1e-4 ? 1 + value ** 2 / 6 + value ** 4 / 120 : Math.sinh(value) / value;
}
export function exponential2(matrix, time) {
  matrix2(matrix);
  odeNumber(time, 'time', -20, 20);
  const halfTrace = (matrix[0][0] + matrix[1][1]) / 2;
  const centered = [[matrix[0][0] - halfTrace, matrix[0][1]], [matrix[1][0], matrix[1][1] - halfTrace]];
  const discriminant = centered[0][0] ** 2 + centered[0][1] * centered[1][0];
  const argument = Math.sqrt(Math.abs(discriminant)) * time;
  const scalar = Math.exp(halfTrace * time);
  const cosine = discriminant >= 0 ? Math.cosh(argument) : Math.cos(argument);
  const ratio = time * (discriminant >= 0 ? hyperbolicSineRatio(argument) : sineRatio(argument));
  const result = centered.map((row, rowIndex) => row.map((entry, columnIndex) => scalar * ((rowIndex === columnIndex ? cosine : 0) + ratio * entry)));
  if (result.some(row => row.some(value => !Number.isFinite(value)))) throw new RangeError('This exponential exceeds the model arithmetic range.');
  return result;
}
export function linearSystemState(matrix, initial, time) {
  vector2(initial, 'initial state');
  const transition = exponential2(matrix, time);
  const state = transition.map(row => row[0] * initial[0] + row[1] * initial[1]);
  if (state.some(value => !Number.isFinite(value))) throw new RangeError('The state exceeds the model arithmetic range.');
  return {
    time,
    transition,
    state,
    derivative: applyOdeMatrix(matrix, state)
  };
}
export function oscillatorState(time, damping = 2, initialPosition = 1, initialVelocity = 0) {
  odeNumber(damping, 'damping', 0, 10);
  const result = linearSystemState([[0, 1], [-4, -damping]], [initialPosition, initialVelocity], time);
  const [position, velocity] = result.state;
  return {
    ...result,
    position,
    velocity,
    energy: (velocity ** 2 + 4 * position ** 2) / 2,
    energyRate: -damping * velocity ** 2
  };
}
export const odeModePresets = {
  node: {
    label: 'Two decaying modes',
    matrix: [[-1, 0], [0, -2]]
  },
  saddle: {
    label: 'A hidden growing mode',
    matrix: [[1, 0], [0, -1]]
  },
  rotation: {
    label: 'Rotation without decay',
    matrix: [[0, -1], [1, 0]]
  },
  spiral: {
    label: 'A decaying spiral',
    matrix: [[-0.3, -1], [1, -0.3]]
  },
  jordan: {
    label: 'Repeated eigenvalue with shear',
    matrix: [[-1, 3], [0, -1]]
  },
  nilpotent: {
    label: 'Zero eigenvalues with drift',
    matrix: [[0, 1], [0, 0]]
  }
};
export function forcingTimeline(time, initial = 40, firstPower = 20, secondPower = 0, switchTime = 3) {
  odeNumber(time, 'time', 0, 20);
  odeNumber(initial, 'initial state', 0, 100);
  odeNumber(firstPower, 'first power', 0, 100);
  odeNumber(secondPower, 'second power', 0, 100);
  odeNumber(switchTime, 'switch time', 0, 20);
  const firstDuration = Math.min(time, switchTime);
  const afterSwitch = Math.max(0, time - switchTime);
  const initialContribution = initial * Math.exp(-0.2 * time);
  const firstContribution = firstPower / 2 * -Math.expm1(-0.2 * firstDuration) * Math.exp(-0.2 * afterSwitch);
  const secondContribution = secondPower / 2 * -Math.expm1(-0.2 * afterSwitch);
  return {
    time,
    initialContribution,
    firstContribution,
    secondContribution,
    state: initialContribution + firstContribution + secondContribution,
    power: time < switchTime ? firstPower : secondPower
  };
}
export function shearSchedule(initial = [1, 0], first = 'upper') {
  vector2(initial, 'initial state');
  if (!['upper', 'lower'].includes(first)) throw new RangeError('Choose upper or lower first.');
  const upper = [[1, 1], [0, 1]];
  const lower = [[1, 0], [1, 1]];
  const intermediate = applyOdeMatrix(first === 'upper' ? upper : lower, initial);
  const final = applyOdeMatrix(first === 'upper' ? lower : upper, intermediate);
  return {
    initial,
    intermediate,
    final,
    first
  };
}
export function odeStep(method, derivative, time, state, step) {
  if (!['euler', 'midpoint', 'rk4'].includes(method)) throw new RangeError('Unknown integration method.');
  for (const [name, value] of Object.entries({
    time,
    state,
    step
  })) odeNumber(value, name);
  if (step <= 0 || time + step === time) throw new RangeError('Step must advance time.');
  const stages = [];
  const evaluate = (stageTime, stageState) => {
    const slope = derivative(stageTime, stageState);
    if (![stageState, slope].every(Number.isFinite)) throw new RangeError('A stage became nonfinite.');
    stages.push({
      time: stageTime,
      state: stageState,
      slope
    });
    return slope;
  };
  const first = evaluate(time, state);
  let average = first;
  if (method === 'midpoint') average = evaluate(time + step / 2, state + step * first / 2);
  if (method === 'rk4') {
    const second = evaluate(time + step / 2, state + step * first / 2);
    const third = evaluate(time + step / 2, state + step * second / 2);
    const fourth = evaluate(time + step, state + step * third);
    average = (first + 2 * second + 2 * third + fourth) / 6;
  }
  const nextState = state + step * average;
  if (!Number.isFinite(nextState)) throw new RangeError('The accepted state became nonfinite.');
  return {
    time: time + step,
    state: nextState,
    stages,
    average,
    step
  };
}
export function integrateScalar({
  method = 'euler',
  derivative,
  initial,
  endTime,
  step,
  maximumSteps = 1000
}) {
  if (!['euler', 'midpoint', 'rk4'].includes(method) || typeof derivative !== 'function') throw new TypeError('Choose a supported method and callable derivative.');
  odeNumber(initial, 'initial state');
  odeNumber(endTime, 'end time', 0, 100);
  odeNumber(step, 'step', Number.MIN_VALUE, 100);
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1 || maximumSteps > 10000) throw new RangeError('Step budget must be an integer from 1 to 10000.');
  const history = [{
    time: 0,
    state: initial
  }];
  let status = 'reached the requested horizon';
  while (history.at(-1).time < endTime) {
    if (history.length - 1 >= maximumSteps) {
      status = 'step budget exhausted before the horizon';
      break;
    }
    const previous = history.at(-1);
    const actualStep = Math.min(step, endTime - previous.time);
    try {
      const next = odeStep(method, derivative, previous.time, previous.state, actualStep);
      history.push(next);
    } catch (error) {
      status = `stopped: ${error.message}`;
      break;
    }
  }
  return {
    history,
    status,
    steps: history.length - 1,
    endTime: history.at(-1).time
  };
}
export function numericalCooling(method, step, rate = 0.2, endTime = 5, initial = 40) {
  odeNumber(rate, 'rate', 0, 100);
  const result = integrateScalar({
    method,
    derivative: (_time, state) => -rate * state,
    initial,
    endTime,
    step
  });
  const exact = initial * Math.exp(-rate * result.endTime);
  return {
    ...result,
    exact,
    signedError: result.history.at(-1).state - exact
  };
}
export function coolingAmplification(rate, step) {
  odeNumber(rate, 'rate', 0, 100);
  odeNumber(step, 'step', 0, 100);
  const product = rate * step;
  return {
    product,
    exact: Math.exp(-product),
    explicit: 1 - product,
    implicit: 1 / (1 + product),
    explicitStrictDecay: product > 0 && product < 2,
    explicitNonnegative: product <= 1
  };
}
export function bvpFamily(endpoint, target, slope = 1) {
  if (!['half-pi', 'pi'].includes(endpoint)) throw new RangeError('Choose an exact symbolic endpoint.');
  odeNumber(target, 'target', -5, 5);
  odeNumber(slope, 'initial slope', -5, 5);
  if (endpoint === 'pi') return {
    classification: target === 0 ? 'infinitely many solutions' : 'no solution',
    slope,
    boundaryValue: 0,
    target,
    endTime: Math.PI
  };
  return {
    classification: 'one solution',
    slope: target,
    boundaryValue: target,
    target,
    endTime: Math.PI / 2
  };
}
