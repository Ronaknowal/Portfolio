// Finite teaching models only: matrices are at most 4 by 4 and traces at most 25 states.
function bounded(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
  return value;
}
function integer(value, minimum, maximum, name) {
  bounded(value, minimum, maximum, name);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const transpose = matrix => matrix[0].map((_, column) => matrix.map(row => row[column]));
const multiply = (left, right) => left.map(row => transpose(right).map(column => dot(row, column)));
const matrixVector = (matrix, vector) => matrix.map(row => dot(row, vector));
const identity = size => Array.from({
  length: size
}, (_, row) => Array.from({
  length: size
}, (_, column) => Number(row === column)));
const add = (left, right) => left.map((row, i) => row.map((value, j) => value + right[i][j]));
const scale = (matrix, factor) => matrix.map(row => row.map(value => factor * value));
const outer = (left, right = left) => left.map(value => right.map(other => value * other));
const norm = vector => Math.hypot(...vector);
const frobenius = matrix => norm(matrix.flat());
const rotate = degrees => {
  const angle = degrees * Math.PI / 180;
  return [[Math.cos(angle), -Math.sin(angle)], [Math.sin(angle), Math.cos(angle)]];
};
function solve(matrix, vector) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-14) throw new RangeError('The teaching system is numerically singular.');
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    for (let row = column + 1; row < size; row += 1) {
      const factor = augmented[row][column] / augmented[column][column];
      for (let index = column; index <= size; index += 1) augmented[row][index] -= factor * augmented[column][index];
    }
  }
  const result = Array(size).fill(0);
  for (let row = size - 1; row >= 0; row -= 1) {
    let residual = augmented[row][size];
    for (let column = row + 1; column < size; column += 1) residual -= augmented[row][column] * result[column];
    result[row] = residual / augmented[row][row];
  }
  return result;
}
function symmetricPower(matrix, exponent) {
  const [a, b, d] = [matrix[0][0], matrix[0][1], matrix[1][1]];
  const angle = 0.5 * Math.atan2(2 * b, a - d);
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const radius = Math.hypot(a - d, 2 * b);
  const high = (a + d + radius) / 2;
  // det/high avoids cancellation in (trace-radius)/2 for an ill-conditioned SPD matrix.
  const low = (a * d - b * b) / high;
  if (!(low > 0) || !Number.isFinite(high)) throw new RangeError('A positive definite matrix is required for the inverse root.');
  const eigenvectors = [[cosine, -sine], [sine, cosine]];
  const powered = multiply(multiply(eigenvectors, [[high ** exponent, 0], [0, low ** exponent]]), transpose(eigenvectors));
  return {
    matrix: powered,
    eigenvalues: [high, low],
    eigenvectors
  };
}
export function curvatureTrace(curvature = 200, angle = 0, rateFraction = 1.6, updates = 12) {
  bounded(curvature, 2, 200, 'Steep curvature');
  bounded(angle, 0, 90, 'Rotation');
  bounded(rateFraction, 0.1, 1.9, 'Rate times steep curvature');
  integer(updates, 0, 24, 'Updates');
  const rotation = rotate(angle);
  const hessian = multiply(multiply(rotation, [[curvature, 0], [0, 2]]), transpose(rotation));
  const rate = rateFraction / curvature;
  const objective = point => dot(point, matrixVector(hessian, point)) / 2;
  let point = [1, 1];
  const frames = [{
    point: [...point],
    objective: objective(point)
  }];
  for (let step = 0; step < updates; step += 1) {
    const gradient = matrixVector(hessian, point);
    point = point.map((value, index) => value - rate * gradient[index]);
    frames.push({
      point: [...point],
      objective: objective(point)
    });
  }
  const initialGradient = matrixVector(hessian, [1, 1]);
  const newtonDirection = solve(hessian, initialGradient).map(value => -value);
  return {
    curvature,
    angle,
    rate,
    rateFraction,
    hessian,
    rotation,
    frames,
    initialGradient,
    newtonDirection,
    conditionNumber: curvature / 2
  };
}
export function safeguardedNewtonState(x = 0.2, damping = 1.2) {
  bounded(x, -1.4, 1.4, 'Initial x');
  bounded(damping, 0, 3, 'Damping');
  const point = [x, 0.5];
  const objective = ([first, second]) => (first * first - 1) ** 2 / 4 + second * second / 2;
  const gradient = [x * (x * x - 1), 0.5];
  const diagonal = [3 * x * x - 1, 1];
  const shifted = diagonal.map(value => value + damping);
  const singular = shifted.some(value => Math.abs(value) < 1e-12);
  const positiveDefinite = shifted.every(value => value > 0);
  const direction = singular ? null : gradient.map((value, index) => -value / shifted[index]);
  const slope = direction ? dot(gradient, direction) : null;
  const initialValue = objective(point);
  const evaluate = multiplier => objective(point.map((value, index) => value + multiplier * direction[index]));
  const attempts = [];
  let accepted = null;
  if (direction && slope < 0) {
    for (let backtrack = 0; backtrack < 20; backtrack += 1) {
      const multiplier = 2 ** -backtrack;
      const value = evaluate(multiplier);
      const threshold = initialValue + 0.0001 * multiplier * slope;
      const passes = value <= threshold;
      attempts.push({
        multiplier,
        value,
        threshold,
        passes
      });
      if (passes) {
        accepted = attempts.at(-1);
        break;
      }
    }
  }
  const samples = direction ? Array.from({
    length: 81
  }, (_, index) => {
    const multiplier = index / 80;
    return {
      multiplier,
      actual: evaluate(multiplier),
      model: initialValue + multiplier * slope + multiplier ** 2 * dot(direction, direction.map((value, i) => diagonal[i] * value)) / 2
    };
  }) : [];
  return {
    point,
    gradient,
    diagonal,
    shifted,
    damping,
    direction,
    slope,
    initialValue,
    positiveDefinite,
    singular,
    attempts,
    accepted,
    samples
  };
}
export function lbfgsHistoryTrace(memory = 2, useScaledIdentity = true, reverseLastPair = false) {
  integer(memory, 0, 3, 'History length');
  const hessian = [[4, 1], [1, 2]];
  const steps = [[1, 0], [0, 1], [1, -1]];
  const allPairs = steps.map((step, index) => ({
    name: `pair ${index + 1}`,
    step,
    change: matrixVector(hessian, step)
  }));
  if (reverseLastPair) allPairs[2].change = allPairs[2].change.map(value => -value);
  const checkedPairs = allPairs.map(pair => ({
    ...pair,
    curvature: dot(pair.step, pair.change),
    accepted: dot(pair.step, pair.change) > 1e-12
  }));
  const pairs = memory ? checkedPairs.filter(pair => pair.accepted).slice(-memory) : [];
  const latest = pairs.at(-1);
  const gamma = latest && useScaledIdentity ? latest.curvature / dot(latest.change, latest.change) : 1;
  const gradient = [3, 1];
  let vector = [...gradient];
  const frames = [{
    operation: 'Start with q = current gradient',
    vector: [...vector],
    pair: null,
    scalar: null
  }];
  const coefficients = [];
  for (let index = pairs.length - 1; index >= 0; index -= 1) {
    const pair = pairs[index];
    const alpha = dot(pair.step, vector) / pair.curvature;
    coefficients[index] = alpha;
    vector = vector.map((value, i) => value - alpha * pair.change[i]);
    frames.push({
      operation: 'Newest to oldest: q ← q − αy',
      vector: [...vector],
      pair: pair.name,
      scalar: alpha
    });
  }
  vector = vector.map(value => gamma * value);
  frames.push({
    operation: 'Apply the starting inverse: r ← γq',
    vector: [...vector],
    pair: null,
    scalar: gamma
  });
  for (let index = 0; index < pairs.length; index += 1) {
    const pair = pairs[index];
    const beta = dot(pair.change, vector) / pair.curvature;
    vector = vector.map((value, i) => value + pair.step[i] * (coefficients[index] - beta));
    frames.push({
      operation: 'Oldest to newest: r ← r + s(α − β)',
      vector: [...vector],
      pair: pair.name,
      scalar: beta
    });
  }
  let inverseApproximation = scale(identity(2), gamma);
  for (const pair of pairs) {
    const correction = add(identity(2), scale(outer(pair.step, pair.change), -1 / pair.curvature));
    inverseApproximation = add(multiply(multiply(correction, inverseApproximation), transpose(correction)), scale(outer(pair.step), 1 / pair.curvature));
  }
  return {
    memory,
    gradient,
    checkedPairs,
    pairs,
    gamma,
    frames,
    inverseApproximation,
    transformed: vector,
    direction: vector.map(value => -value),
    denseProduct: matrixVector(inverseApproximation, gradient),
    exactNewtonDirection: solve(hessian, gradient).map(value => -value)
  };
}
export function bernoulliGeometry(probability = 0.2, target = 0.8, fraction = 0.25) {
  bounded(probability, 0.05, 0.95, 'Initial probability');
  bounded(target, 0.1, 0.9, 'Target fraction');
  bounded(fraction, 0.01, 1, 'Step fraction');
  const logit = Math.log(probability / (1 - probability));
  const probabilityGradient = (probability - target) / (probability * (1 - probability));
  const probabilityFisher = 1 / (probability * (1 - probability));
  const logitFisher = probability * (1 - probability);
  const probabilityDirection = target - probability;
  const logitDirection = (target - probability) / logitFisher;
  const directNext = probability + fraction * probabilityDirection;
  const logitNext = 1 / (1 + Math.exp(-(logit + fraction * logitDirection)));
  const kl = next => probability * Math.log(probability / next) + (1 - probability) * Math.log((1 - probability) / (1 - next));
  const loss = next => -target * Math.log(next) - (1 - target) * Math.log(1 - next);
  const empiricalFisher = target / probability ** 2 + (1 - target) / (1 - probability) ** 2;
  const samples = Array.from({
    length: 51
  }, (_, index) => {
    const step = fraction * index / 50;
    const next = probability + step * probabilityDirection;
    return {
      step,
      exact: kl(next),
      local: 0.5 * probabilityFisher * (next - probability) ** 2
    };
  });
  return {
    probability,
    target,
    fraction,
    logit,
    probabilityGradient,
    probabilityFisher,
    logitFisher,
    empiricalFisher,
    probabilityDirection,
    logitDirection,
    mappedLogitDirection: logitFisher * logitDirection,
    directNext,
    logitNext,
    directKl: kl(directNext),
    logitKl: kl(logitNext),
    initialLoss: loss(probability),
    directLoss: loss(directNext),
    logitLoss: loss(logitNext),
    samples
  };
}
function kronecker(left, right) {
  return left.flatMap(row => right.map(innerRow => row.flatMap(value => innerRow.map(entry => value * entry))));
}
export function kfacFactorState(strength = 1, damping = 0.1) {
  bounded(strength, 0, 2, 'Layer strength');
  bounded(damping, 0.01, 1, 'Diagonal damping');
  const inputs = [[1, -1], [1, 2]];
  const labels = [[1, 0], [0, 1]];
  const weights = scale([[0.8, -0.5], [-0.4, 0.6]], strength);
  const cases = inputs.map((input, index) => {
    const probabilities = matrixVector(weights, input).map(value => 1 / (1 + Math.exp(-value)));
    const scoreCovariance = probabilities.map((value, row) => probabilities.map((_, column) => row === column ? value * (1 - value) : 0));
    return {
      input,
      probabilities,
      inputOuter: outer(input),
      scoreCovariance,
      gradient: outer(probabilities.map((value, row) => value - labels[index][row]), input)
    };
  });
  const mean = matrices => scale(matrices.reduce(add), 1 / matrices.length);
  const inputFactor = mean(cases.map(item => item.inputOuter));
  const outputFactor = mean(cases.map(item => item.scoreCovariance));
  const exactFisher = mean(cases.map(item => kronecker(item.inputOuter, item.scoreCovariance)));
  const factoredFisher = kronecker(inputFactor, outputFactor);
  const difference = add(factoredFisher, scale(exactFisher, -1));
  const gradient = mean(cases.map(item => item.gradient));
  const flatGradient = transpose(gradient).flat();
  const damp = matrix => add(matrix, scale(identity(matrix.length), damping));
  const exactDirection = solve(damp(exactFisher), flatGradient).map(value => -value);
  const factoredDirection = solve(damp(factoredFisher), flatGradient).map(value => -value);
  const rootDamping = Math.sqrt(damping);
  const factorDampedFisher = kronecker(add(inputFactor, scale(identity(2), rootDamping)), add(outputFactor, scale(identity(2), rootDamping)));
  const factorDampedDirection = solve(factorDampedFisher, flatGradient).map(value => -value);
  return {
    strength,
    damping,
    weights,
    cases,
    inputFactor,
    outputFactor,
    gradient,
    flatGradient,
    exactFisher,
    factoredFisher,
    difference,
    approximationError: frobenius(difference),
    exactDirection,
    factoredDirection,
    factorDampedFisher,
    factorDampedDirection,
    factorDampingDifference: frobenius(add(factorDampedFisher, scale(damp(factoredFisher), -1)))
  };
}
export function shampooMatrixState(updates = 1, epsilon = 0.1, rotationDegrees = 0) {
  integer(updates, 1, 3, 'Gradient frames');
  bounded(epsilon, 0.0001, 1, 'Root regularization');
  bounded(rotationDegrees, 0, 90, 'Row basis rotation');
  const originalGradients = [[[1, 2], [0, 1]], [[2, 0], [1, -1]], [[0.5, -1], [2, 0.5]]];
  const rotation = rotate(rotationDegrees);
  const gradients = originalGradients.map(gradient => multiply(rotation, gradient));
  let left = scale(identity(2), epsilon);
  let right = scale(identity(2), epsilon);
  const frames = [];
  for (const gradient of gradients.slice(0, updates)) {
    const leftContribution = multiply(gradient, transpose(gradient));
    const rightContribution = multiply(transpose(gradient), gradient);
    left = add(left, leftContribution);
    right = add(right, rightContribution);
    const leftRoot = symmetricPower(left, -0.25);
    const rightRoot = symmetricPower(right, -0.25);
    const direction = multiply(multiply(leftRoot.matrix, gradient), rightRoot.matrix);
    frames.push({
      gradient,
      leftContribution,
      rightContribution,
      left,
      right,
      leftRoot,
      rightRoot,
      direction
    });
  }
  return {
    updates,
    epsilon,
    rotationDegrees,
    rotation,
    originalGradients,
    frames,
    current: frames.at(-1)
  };
}
