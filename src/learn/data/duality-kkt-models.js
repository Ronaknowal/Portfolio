function bounded(value, minimum, maximum, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
  return value;
}
function frozen(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(frozen);
    Object.freeze(value);
  }
  return value;
}
export function formatDualityNumber(value) {
  if (value === null) return 'not a certificate';
  if (value === Infinity) return '+∞';
  if (value === -Infinity) return '−∞';
  if (!Number.isFinite(value)) return 'undefined';
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10000) return value.toExponential(3);
  return String(Number(value.toFixed(6)));
}
export function projectionCertificateState(budget = 5, candidate = [2, 3], multiplier = 1) {
  bounded(budget, -2, 10, 'Budget');
  bounded(multiplier, -2, 12, 'Multiplier');
  if (!Array.isArray(candidate) || candidate.length !== 2) throw new TypeError('Use two candidate coordinates.');
  candidate.forEach(value => bounded(value, -3, 8, 'Candidate coordinate'));
  const target = [3, 4];
  const shift = Math.max(0, (7 - budget) / 2);
  const optimum = target.map(value => value - shift);
  const optimalMultiplier = 2 * shift;
  const optimalValue = 2 * shift ** 2;
  const minimizer = target.map(value => value - multiplier / 2);
  const objective = candidate.reduce((sum, value, index) => sum + (value - target[index]) ** 2, 0);
  const constraint = candidate[0] + candidate[1] - budget;
  const slack = -constraint;
  const lagrangian = objective + multiplier * constraint;
  const dualValue = multiplier * (7 - budget) - multiplier ** 2 / 2;
  const minimizationGap = candidate.reduce((sum, value, index) => sum + (value - minimizer[index]) ** 2, 0);
  const complementarityGap = multiplier * slack;
  const primalFeasible = constraint <= 0;
  const dualFeasible = multiplier >= 0;
  const stationarity = candidate.map((value, index) => 2 * (value - target[index]) + multiplier);
  return frozen({
    budget,
    candidate: [...candidate],
    multiplier,
    target,
    optimum,
    optimalMultiplier,
    optimalValue,
    minimizer,
    objective,
    constraint,
    slack,
    lagrangian,
    dualValue,
    minimizationGap,
    complementarityGap,
    primalFeasible,
    dualFeasible,
    stationarity,
    formalDifference: objective - dualValue,
    certifiedGap: primalFeasible && dualFeasible ? objective - dualValue : null
  });
}
export function scalarKktState(center = -1, candidate = 0, multiplier = 2) {
  bounded(center, -2, 2, 'Quadratic center');
  bounded(candidate, -2, 4, 'Candidate');
  bounded(multiplier, -2, 6, 'Multiplier');
  const optimum = Math.max(center, 0);
  const optimalMultiplier = 2 * Math.max(-center, 0);
  const constraint = -candidate;
  const objective = (candidate - center) ** 2;
  const dualValue = -multiplier * center - multiplier ** 2 / 4;
  const stationarity = 2 * (candidate - center) - multiplier;
  const complementaryProduct = -multiplier * candidate;
  const conditions = {
    primal: candidate >= 0,
    dual: multiplier >= 0,
    stationarity: stationarity === 0,
    complementarity: complementaryProduct === 0
  };
  return frozen({
    center,
    candidate,
    multiplier,
    optimum,
    optimalMultiplier,
    constraint,
    objective,
    dualValue,
    stationarity,
    complementaryProduct,
    gradient: 2 * (candidate - center),
    constraintContribution: -multiplier,
    active: candidate === 0,
    conditions,
    allConditions: Object.values(conditions).every(Boolean),
    optimalValue: (optimum - center) ** 2
  });
}
export function sensitivityState(mode = 'quadratic', base = 5, change = 0.5, chosenPrice = 0.5) {
  if (!['quadratic', 'kink'].includes(mode)) throw new TypeError('Choose quadratic or kink.');
  bounded(change, -2, 2, 'Right-hand-side change');
  bounded(chosenPrice, 0, 1, 'Kink price');
  bounded(base, mode === 'quadratic' ? 0 : -2, mode === 'quadratic' ? 10 : 2, 'Base right-hand side');
  const value = mode === 'quadratic' ? argument => Math.max(0, 7 - argument) ** 2 / 2 : argument => Math.max(0, -argument);
  const price = mode === 'quadratic' ? Math.max(0, 7 - base) : base < 0 ? 1 : base > 0 ? 0 : chosenPrice;
  const originalValue = value(base);
  const changedValue = value(base + change);
  const supportingValue = originalValue - price * change;
  const domain = mode === 'quadratic' ? [-2, 12] : [-4, 4];
  const curve = Array.from({
    length: 113
  }, (_, index) => {
    const argument = domain[0] + (domain[1] - domain[0]) * index / 112;
    return [argument, value(argument)];
  });
  const differentiable = mode === 'quadratic' || base !== 0;
  return frozen({
    mode,
    base,
    change,
    chosenPrice,
    price,
    originalValue,
    changedValue,
    supportingValue,
    actualChange: changedValue - originalValue,
    linearChange: -price * change,
    supportingGap: changedValue - supportingValue,
    differentiable,
    domain,
    curve,
    derivative: differentiable ? -price : null,
    leftDerivative: mode === 'kink' && base === 0 ? -1 : -price,
    rightDerivative: mode === 'kink' && base === 0 ? 0 : -price
  });
}
export function resourceAllocationAtPrice(price, budget) {
  bounded(price, 0, 120, 'Resource price');
  bounded(budget, 0, 8, 'Resource budget');
  const allocation = [Math.max(0, 3 - price / 2), Math.max(0, 4 - price / 4)];
  const sum = allocation[0] + allocation[1];
  const objective = (allocation[0] - 3) ** 2 + 2 * (allocation[1] - 4) ** 2;
  const violation = sum - budget;
  // This priority repair is a feasible upper-bound witness, not the optimal projection.
  const repaired = [Math.min(allocation[0], budget), 0];
  const remainingBudget = budget - repaired[0];
  repaired[1] = Math.min(allocation[1], remainingBudget);
  const repairedSlack = remainingBudget - repaired[1];
  const repairedObjective = (repaired[0] - 3) ** 2 + 2 * (repaired[1] - 4) ** 2;
  const dualValue = objective + price * violation;
  // Evaluate the equivalent nonnegative gap without subtracting nearly equal costs.
  // At a clamped zero allocation, L's one-sided derivative need not vanish.
  const weights = [1, 2];
  const targets = [3, 4];
  const localMinimizationGap = allocation.reduce((total, value, index) => {
    const displacement = repaired[index] - value;
    const boundaryDerivative = value === 0 ? Math.max(0, price - 2 * weights[index] * targets[index]) : 0;
    return total + weights[index] * displacement ** 2 + boundaryDerivative * displacement;
  }, 0);
  const certificateGap = localMinimizationGap + price * repairedSlack;
  return frozen({
    price,
    budget,
    allocation,
    sum,
    objective,
    violation,
    primalFeasible: violation <= 0,
    dualValue,
    repaired,
    repairedObjective,
    repairedSlack,
    localMinimizationGap,
    subtractedGap: repairedObjective - dualValue,
    certificateGap
  });
}
export function resourceDualAscentState(budget = 5, rate = 1, initialPrice = 0, updates = 12) {
  bounded(budget, 0, 8, 'Resource budget');
  bounded(rate, 0, 4, 'Price step size');
  bounded(initialPrice, 0, 20, 'Initial price');
  if (!Number.isInteger(updates) || updates < 0 || updates > 30) throw new RangeError('Use zero through thirty price updates.');
  const optimumPrice = budget >= 7 ? 0 : budget >= 2.5 ? 4 * (7 - budget) / 3 : 16 - 4 * budget;
  const optimum = resourceAllocationAtPrice(optimumPrice, budget);
  const frames = [];
  let price = initialPrice;
  for (let step = 0; step <= updates; step++) {
    const state = resourceAllocationAtPrice(price, budget);
    const nextPrice = Math.max(0, price + rate * state.violation);
    frames.push({
      step,
      ...state,
      nextPrice
    });
    price = nextPrice;
  }
  return frozen({
    budget,
    rate,
    initialPrice,
    updates,
    frames,
    optimumPrice,
    optimum,
    smoothGradientBound: 0.75,
    sufficientStepUpper: 8 / 3
  });
}
export function dualDomainState(multiplier = 1) {
  bounded(multiplier, -2, 4, 'Linear-example multiplier');
  return frozen({
    multiplier,
    slope: 1 - multiplier,
    signAllowed: multiplier >= 0,
    dualValue: multiplier === 1 ? 1 : -Infinity,
    finite: multiplier === 1
  });
}
export function degenerateDualState(multiplier = 1) {
  bounded(multiplier, 0, 1000000, 'Degenerate-example multiplier');
  return frozen({
    multiplier,
    primalValue: 0,
    dualValue: multiplier > 0 ? -1 / (4 * multiplier) : -Infinity,
    minimizer: multiplier > 0 ? -1 / (2 * multiplier) : null,
    primalAttained: true,
    dualAttained: false
  });
}
