// Reproducible bounded teaching models, independent of their SVG renderers.
export function seededRandom(seed) {
  if (!Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) throw new RangeError('Seed must be an integer from 1 through 4294967295.');
  let state = seed >>> 0;
  return {
    uniform() {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return ((state >>> 0) + .5) / 4294967296;
    },
    normal() {
      return Math.sqrt(-2 * Math.log(this.uniform())) * Math.cos(2 * Math.PI * this.uniform());
    }
  };
}
function integer(value, low, high, label) {
  if (!Number.isInteger(value) || value < low || value > high) throw new RangeError(`${label} must be an integer from ${low} to ${high}.`);
}
function finite(value, low, high, label) {
  if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`${label} must be finite and between ${low} and ${high}.`);
}
const mean = values => values.reduce((total, value) => total + value, 0) / values.length;
function sampleVariance(values) {
  if (values.length < 2) return null;
  const center = mean(values);
  return values.reduce((total, value) => total + (value - center) ** 2, 0) / (values.length - 1);
}
export function independentEstimate({
  seed = 7,
  evaluations = 100,
  antithetic = false
} = {}) {
  integer(evaluations, 2, 2000, 'Evaluation budget');
  if (antithetic && evaluations % 2) throw new RangeError('Pairing requires an even evaluation budget.');
  const random = seededRandom(seed);
  const groups = [];
  const rows = [];
  const groupCount = antithetic ? evaluations / 2 : evaluations;
  for (let index = 0; index < groupCount; index += 1) {
    const x = random.uniform();
    const inputs = antithetic ? [x, 1 - x] : [x];
    const contributions = inputs.map(value => value * value);
    groups.push(mean(contributions));
    rows.push(Object.freeze({
      index,
      inputs: Object.freeze(inputs),
      contributions: Object.freeze(contributions),
      mean: mean(groups),
      evaluations: (index + 1) * inputs.length
    }));
  }
  return Object.freeze({
    rows: Object.freeze(rows),
    estimate: mean(groups),
    mcse: groups.length < 2 ? null : Math.sqrt(sampleVariance(groups) / groups.length),
    exactMean: 1 / 3,
    exactVariance: antithetic ? 1 / (90 * evaluations) : 4 / (45 * evaluations),
    independentGroups: groups.length
  });
}
export const ASYMMETRIC_PROPOSAL = Object.freeze([[.1, .8, .1], [.2, .1, .7], [.6, .3, .1]].map(Object.freeze));
export const SYMMETRIC_PROPOSAL = Object.freeze([[.2, .4, .4], [.4, .2, .4], [.4, .4, .2]].map(Object.freeze));
export function finiteMetropolis(weights = [2, 5, 3], proposal = ASYMMETRIC_PROPOSAL, correctProposal = true) {
  if (weights.length !== 3 || weights.some(value => !Number.isFinite(value) || value <= 0)) throw new RangeError('Exactly three positive finite weights required.');
  if (proposal.length !== 3 || proposal.some(row => row.length !== 3 || row.some(value => !Number.isFinite(value) || value < 0) || Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) > 1e-12)) throw new RangeError('Proposal rows must contain three nonnegative probabilities summing to one.');
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total)) throw new RangeError('The total target weight must be finite.');
  const target = weights.map(value => value / total);
  const acceptance = weights.map((from, i) => weights.map((to, j) => proposal[i][j] === 0 ? 0 : Math.min(1, to / from * (correctProposal ? proposal[j][i] / proposal[i][j] : 1))));
  const transition = acceptance.map((row, i) => {
    const next = row.map((value, j) => i === j ? 0 : value * proposal[i][j]);
    next[i] = 1 - next.reduce((sum, value) => sum + value, 0);
    return Object.freeze(next);
  });
  const afterOne = target.map((_, j) => target.reduce((sum, value, i) => sum + value * transition[i][j], 0));
  const flow = transition.map((row, i) => Object.freeze(row.map(value => target[i] * value)));
  return Object.freeze({
    target: Object.freeze(target),
    acceptance: Object.freeze(acceptance.map(Object.freeze)),
    transition: Object.freeze(transition),
    flow: Object.freeze(flow),
    afterOne: Object.freeze(afterOne)
  });
}
export function betaLogTarget(theta, alpha = 10, beta = 4) {
  if (!(theta > 0 && theta < 1)) return -Infinity;
  return (alpha - 1) * Math.log(theta) + (beta - 1) * Math.log1p(-theta);
}
export function metropolisTrace({
  seed = 7,
  scale = .12,
  iterations = 200,
  initial = .5
} = {}) {
  finite(scale, .001, 1, 'Proposal scale');
  integer(iterations, 1, 5000, 'Iterations');
  finite(initial, .001, .999, 'Initial probability');
  const random = seededRandom(seed);
  let state = initial;
  let accepted = 0;
  const rows = [];
  for (let index = 0; index < iterations; index += 1) {
    const before = state;
    const proposal = state + scale * random.normal();
    const logRatio = betaLogTarget(proposal) - betaLogTarget(state);
    const logAcceptance = Math.min(0, logRatio);
    const uniform = random.uniform();
    const accept = Math.log(uniform) < logAcceptance;
    if (accept) {
      state = proposal;
      accepted += 1;
    }
    rows.push(Object.freeze({
      index,
      before,
      proposal,
      logRatio,
      logAcceptance,
      uniform,
      accept,
      state
    }));
  }
  return Object.freeze({
    rows: Object.freeze(rows),
    accepted,
    acceptanceRate: accepted / iterations
  });
}
export function harmonicEnergy(position, momentum, sigma = 1) {
  return (position / sigma) ** 2 / 2 + momentum * momentum / 2;
}
export function leapfrog(position, momentum, step, sigma = 1) {
  const halfMomentum = momentum - step * position / (2 * sigma * sigma);
  const nextPosition = position + step * halfMomentum;
  const nextMomentum = halfMomentum - step * nextPosition / (2 * sigma * sigma);
  return Object.freeze({
    position: nextPosition,
    momentum: nextMomentum,
    halfMomentum
  });
}
export function exactHarmonic(position, momentum, time, sigma = 1) {
  return Object.freeze({
    position: position * Math.cos(time / sigma) + sigma * momentum * Math.sin(time / sigma),
    momentum: momentum * Math.cos(time / sigma) - position / sigma * Math.sin(time / sigma)
  });
}
export function hamiltonianTrajectory({
  position = 1,
  momentum = .7,
  step = .2,
  steps = 12,
  sigma = 1
} = {}) {
  finite(position, -3, 3, 'Initial position');
  finite(momentum, -3, 3, 'Initial momentum');
  finite(step, .01, 2, 'Step size');
  finite(sigma, .2, 2, 'Target standard deviation');
  integer(steps, 1, 64, 'Leapfrog steps');
  const initialEnergy = harmonicEnergy(position, momentum, sigma);
  const rows = [Object.freeze({
    index: 0,
    position,
    momentum,
    energy: initialEnergy,
    energyError: 0
  })];
  let current = {
    position,
    momentum
  };
  let divergent = false;
  for (let index = 1; index <= steps; index += 1) {
    const next = leapfrog(current.position, current.momentum, step, sigma);
    const energy = harmonicEnergy(next.position, next.momentum, sigma);
    const energyError = energy - initialEnergy;
    if (!Number.isFinite(energy) || Math.abs(energyError) > 1000) {
      divergent = true;
      break;
    }
    rows.push(Object.freeze({
      index,
      ...next,
      energy,
      energyError
    }));
    current = next;
  }
  const logAcceptance = divergent ? -Infinity : Math.min(0, initialEnergy - harmonicEnergy(current.position, current.momentum, sigma));
  return Object.freeze({
    rows: Object.freeze(rows),
    initialEnergy,
    divergent,
    logAcceptance,
    acceptance: Math.exp(logAcceptance),
    requestedSteps: steps
  });
}
function notTurning(left, right) {
  const displacement = right.position - left.position;
  return displacement * left.momentum >= 0 && displacement * right.momentum >= 0;
}
// Original slice-based Algorithm2: materialize candidates to expose eligibility.
// Stop a bad subtree early; never add its candidates to the outer pool.
export function nutsExpansion({
  seed = 7,
  position = 1,
  step = .25,
  maxDepth = 5,
  momentum: suppliedMomentum
} = {}) {
  finite(position, -3, 3, 'Initial position');
  finite(step, .01, 2, 'Step size');
  integer(maxDepth, 1, 7, 'Maximum tree depth');
  const random = seededRandom(seed);
  const momentum = suppliedMomentum ?? random.normal();
  finite(momentum, -20, 20, 'Initial momentum');
  const initial = Object.freeze({
    position,
    momentum,
    time: 0
  });
  const initialEnergy = harmonicEnergy(position, momentum);
  const logSlice = -initialEnergy + Math.log(random.uniform());
  const explored = [];
  const snapshots = [];
  let divergent = false;
  function build(start, direction, depth) {
    if (depth === 0) {
      const next = leapfrog(start.position, start.momentum, direction * step);
      const state = Object.freeze({
        position: next.position,
        momentum: next.momentum,
        time: start.time + direction
      });
      const joint = -harmonicEnergy(state.position, state.momentum);
      const validEnergy = Number.isFinite(joint) && joint > logSlice - 1000;
      const onSlice = validEnergy && logSlice <= joint;
      if (!validEnergy) divergent = true;
      explored.push(Object.freeze({
        ...state,
        onSlice,
        validEnergy
      }));
      return {
        left: state,
        right: state,
        candidates: onSlice ? [state] : [],
        keepBuilding: validEnergy
      };
    }
    const first = build(start, direction, depth - 1);
    if (!first.keepBuilding) return first;
    const second = build(direction < 0 ? first.left : first.right, direction, depth - 1);
    const left = direction < 0 ? second.left : first.left;
    const right = direction < 0 ? first.right : second.right;
    return {
      left,
      right,
      candidates: first.candidates.concat(second.candidates),
      keepBuilding: second.keepBuilding && notTurning(left, right)
    };
  }
  let left = initial;
  let right = initial;
  let candidates = [initial];
  let keepBuilding = true;
  for (let depth = 0; depth < maxDepth && keepBuilding; depth += 1) {
    const direction = random.uniform() < .5 ? -1 : 1;
    const beforeCount = explored.length;
    const subtree = build(direction < 0 ? left : right, direction, depth);
    if (direction < 0) left = subtree.left;else right = subtree.right;
    if (subtree.keepBuilding) candidates = candidates.concat(subtree.candidates);
    const wholeTreeTurns = !notTurning(left, right);
    keepBuilding = subtree.keepBuilding && !wholeTreeTurns;
    snapshots.push(Object.freeze({
      depth: depth + 1,
      direction,
      left,
      right,
      explored: Object.freeze(explored.slice()),
      newStates: explored.length - beforeCount,
      subtreeAccepted: subtree.keepBuilding,
      candidates: Object.freeze(candidates.slice()),
      stopReason: divergent ? 'Energy guard' : !subtree.keepBuilding ? 'Internal subtree turn' : wholeTreeTurns ? 'Whole-tree turn' : depth + 1 === maxDepth ? 'Depth cap' : 'Keep doubling'
    }));
  }
  const selected = candidates[Math.floor(random.uniform() * candidates.length)];
  return Object.freeze({
    initial,
    initialEnergy,
    logSlice,
    snapshots: Object.freeze(snapshots),
    selected,
    divergent,
    candidateCount: candidates.length
  });
}
export function stationaryMeanVariance(draws, correlation, thinning = 1) {
  integer(draws, 1, 2000, 'Transition budget');
  finite(correlation, -.99, .99, 'Lag-one correlation');
  integer(thinning, 1, draws, 'Thinning interval');
  const retained = Math.floor(draws / thinning);
  const rho = correlation ** thinning;
  let inflation = 1;
  for (let lag = 1; lag < retained; lag += 1) inflation += 2 * (1 - lag / retained) * rho ** lag;
  const variance = .25 * inflation / retained;
  return Object.freeze({
    retained,
    variance,
    mcse: Math.sqrt(variance),
    equivalentIID: .25 / variance,
    retainedCorrelation: rho
  });
}
export function twoStateTrace({
  seed = 7,
  correlation = .8,
  draws = 100
} = {}) {
  finite(correlation, -.99, .99, 'Lag-one correlation');
  integer(draws, 2, 1000, 'Draw count');
  const random = seededRandom(seed);
  let current = random.uniform() < .5 ? 0 : 1; // Stationary initial distribution.
  const states = [];
  for (let i = 0; i < draws; i += 1) {
    if (random.uniform() < (1 - correlation) / 2) current = 1 - current;
    states.push(current);
  }
  return Object.freeze(states);
}
