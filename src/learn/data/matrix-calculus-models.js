function finitePair(value, label) {
  if (!Array.isArray(value) || value.length !== 2 || !value.every(Number.isFinite)) {
    throw new Error(label + ' must contain two finite numbers.');
  }
}
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const multiply = (matrix, vector) => matrix.map(row => dot(row, vector));
export function polynomialValue(input) {
  finitePair(input, 'Input');
  const [first, second] = input;
  return [first * first + second, first * second];
}
export function polynomialJacobian(input) {
  finitePair(input, 'Input');
  return [[2 * input[0], 1], [input[1], input[0]]];
}
export function localApproximation(input, direction, step) {
  finitePair(input, 'Input');
  finitePair(direction, 'Direction');
  if (!Number.isFinite(step)) throw new Error('The step must be finite.');
  const jacobian = polynomialJacobian(input);
  const base = polynomialValue(input);
  const shiftedInput = input.map((value, index) => value + step * direction[index]);
  const shifted = polynomialValue(shiftedInput);
  const rate = multiply(jacobian, direction);
  const predicted = rate.map(value => value * step);
  const actual = shifted.map((value, index) => value - base[index]);
  const error = actual.map((value, index) => value - predicted[index]);
  return {
    input,
    direction,
    step,
    jacobian,
    base,
    shiftedInput,
    shifted,
    rate,
    predicted,
    actual,
    error
  };
}
export const chainMatrix = [[1, 2], [-1, 1]];
export function chainDerivatives(input, direction, outputWeights) {
  finitePair(input, 'Input');
  finitePair(direction, 'Direction');
  finitePair(outputWeights, 'Output weights');
  const intermediate = multiply(chainMatrix, input);
  const squared = intermediate.map(value => value * value);
  const value = dot(outputWeights, squared);
  const intermediateTangent = multiply(chainMatrix, direction);
  const squaredTangent = intermediate.map((value, index) => 2 * value * intermediateTangent[index]);
  const lossTangent = dot(outputWeights, squaredTangent);
  const intermediateGradient = intermediate.map((value, index) => 2 * value * outputWeights[index]);
  const contributions = input.map((_, column) => intermediateGradient.map((gradient, row) => gradient * chainMatrix[row][column]));
  const inputGradient = contributions.map(terms => terms.reduce((sum, value) => sum + value, 0));
  const squaredJacobian = chainMatrix.map((row, index) => row.map(value => 2 * intermediate[index] * value));
  return {
    input,
    direction,
    outputWeights,
    intermediate,
    squared,
    value,
    intermediateTangent,
    squaredTangent,
    lossTangent,
    intermediateGradient,
    contributions,
    inputGradient,
    squaredJacobian
  };
}
export const affineFixtures = {
  ordinary: {
    label: 'Two different observations',
    inputs: [[1, 2], [3, -1]],
    targets: [[1, 0], [0, 2]]
  },
  repeated: {
    label: 'Duplicate the first observation',
    inputs: [[1, 2], [1, 2]],
    targets: [[1, 0], [1, 0]]
  },
  zero: {
    label: 'Zero inputs: only the bias moves outputs',
    inputs: [[0, 0], [0, 0]],
    targets: [[1, 0], [0, 2]]
  }
};
export const affineWeights = [[1, -1], [2, 1]];
export const affineBias = [0, 1];
export function affineGradientState(fixtureKey = 'ordinary', reduction = 'mean') {
  const fixture = affineFixtures[fixtureKey];
  if (!fixture) throw new Error('Unknown observation fixture.');
  if (!['sum', 'mean'].includes(reduction)) throw new Error('Choose sum or mean over observations.');
  const {
    inputs,
    targets
  } = fixture;
  const divisor = reduction === 'mean' ? inputs.length : 1;
  const outputs = inputs.map(row => affineBias.map((bias, output) => bias + row.reduce((sum, value, feature) => sum + value * affineWeights[feature][output], 0)));
  const residuals = outputs.map((row, observation) => row.map((value, output) => value - targets[observation][output]));
  const incoming = residuals.map(row => row.map(value => value / divisor));
  const loss = residuals.flat().reduce((sum, value) => sum + value * value, 0) / (2 * divisor);
  const weightContributions = affineWeights.map((row, feature) => row.map((_, output) => inputs.map((input, observation) => ({
    observation,
    input: input[feature],
    incoming: incoming[observation][output],
    product: input[feature] * incoming[observation][output]
  }))));
  const weightGradient = weightContributions.map(row => row.map(terms => terms.reduce((sum, term) => sum + term.product, 0)));
  const biasGradient = affineBias.map((_, output) => incoming.reduce((sum, row) => sum + row[output], 0));
  const inputGradient = incoming.map(row => affineWeights.map(weights => dot(row, weights)));
  return {
    fixtureKey,
    reduction,
    inputs,
    targets,
    divisor,
    outputs,
    residuals,
    incoming,
    loss,
    weightContributions,
    weightGradient,
    biasGradient,
    inputGradient
  };
}
export const differenceSteps = [1, 0.1, 0.01, 0.0001, 0.000001, 0.00000001, 1e-10, 1e-12, 1e-14, 1e-16];
export function differenceCheck(kind, point, step) {
  if (!['cubic', 'absolute'].includes(kind)) throw new Error('Unknown function.');
  if (!Number.isFinite(point) || !Number.isFinite(step) || step <= 0) throw new Error('A finite point and positive finite step are required.');
  const evaluate = value => kind === 'cubic' ? value * value * value : Math.abs(value);
  const exact = kind === 'cubic' ? 3 * point * point : point === 0 ? null : Math.sign(point);
  const center = evaluate(point);
  const plus = evaluate(point + step);
  const minus = evaluate(point - step);
  const forward = (plus - center) / step;
  const backward = (center - minus) / step;
  const central = (plus - minus) / (2 * step);
  return {
    kind,
    point,
    step,
    center,
    plus,
    minus,
    exact,
    forward,
    backward,
    central,
    forwardError: exact === null ? null : Math.abs(forward - exact),
    centralError: exact === null ? null : Math.abs(central - exact),
    roundedInput: point + step === point || point - step === point
  };
}
export function calculusNumber(value) {
  if (value === null) return 'undefined';
  if (value === 0 || Object.is(value, -0)) return '0';
  if (Math.abs(value) < 0.00001 || Math.abs(value) >= 100000) return value.toExponential(3);
  return String(Number(value.toFixed(6)));
}
