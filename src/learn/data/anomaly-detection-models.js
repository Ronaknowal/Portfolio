/** Pure teaching models for the anomaly-detection lesson.
 *
 * The isolation and LOF calculations are exact: coordinates are entered with at
 * most two decimals, so every quantity is a rational number and is carried as
 * one. That keeps 35/24 and 841/660 on screen as themselves rather than as
 * rounded decimals, and it makes ties genuine ties. The kernel boundary and the
 * alert-population counts are ordinary floating-point arithmetic, which is what
 * their exponentials and rates need.
 *
 * These are bounded fixtures for small hand-checkable examples, not a detection
 * library. Input limits are declared and enforced, and every function rejects
 * what it cannot answer exactly rather than returning an approximation quietly.
 */

const absolute = value => (value < 0n ? -value : value);
function greatestCommonDivisor(left, right) {
  let a = absolute(left);
  let b = absolute(right);
  while (b) [a, b] = [b, a % b];
  return a;
}

/** An exact rational number. Denominators stay positive and terms stay reduced. */
export class Exact {
  constructor(numerator, denominator = 1n) {
    if (denominator === 0n) throw new RangeError('An exact value cannot have a zero denominator.');
    const sign = denominator < 0n ? -1n : 1n;
    const divisor = greatestCommonDivisor(numerator, denominator) || 1n;
    this.numerator = sign * numerator / divisor;
    this.denominator = sign * denominator / divisor;
  }
  /** Build from a number with at most `decimals` decimal places. */
  static from(value, decimals = 2) {
    if (!Number.isFinite(value)) throw new RangeError('Use a finite value.');
    const scale = 10 ** decimals;
    const scaled = Math.round(value * scale);
    if (Math.abs(scaled / scale - value) > 1e-9) throw new RangeError(`Use at most ${decimals} decimal places.`);
    return new Exact(BigInt(scaled), BigInt(scale));
  }
  add(other) { return new Exact(this.numerator * other.denominator + other.numerator * this.denominator, this.denominator * other.denominator); }
  subtract(other) { return new Exact(this.numerator * other.denominator - other.numerator * this.denominator, this.denominator * other.denominator); }
  multiply(other) { return new Exact(this.numerator * other.numerator, this.denominator * other.denominator); }
  divide(other) {
    if (other.numerator === 0n) throw new RangeError('Division by an exact zero is undefined here.');
    return new Exact(this.numerator * other.denominator, this.denominator * other.numerator);
  }
  negate() { return new Exact(-this.numerator, this.denominator); }
  absolute() { return new Exact(absolute(this.numerator), this.denominator); }
  compare(other) {
    const difference = this.numerator * other.denominator - other.numerator * this.denominator;
    return difference === 0n ? 0 : difference > 0n ? 1 : -1;
  }
  equals(other) { return this.compare(other) === 0; }
  isZero() { return this.numerator === 0n; }
  toNumber() { return Number(this.numerator) / Number(this.denominator); }
  /** "17/6", or "3" when the denominator is one. */
  toString() { return this.denominator === 1n ? String(this.numerator) : `${this.numerator}/${this.denominator}`; }
}
export const exact = (numerator, denominator = 1) => new Exact(BigInt(numerator), BigInt(denominator));
const ZERO = exact(0);
const ONE = exact(1);

/** Average remaining path length in a terminal node holding m reference rows:
 * c(m) = 2H(m-1) - 2(m-1)/m, with c(0) = c(1) = 0. This normalizes a search
 * path; it is not a probability. */
export function correction(size) {
  if (!Number.isInteger(size) || size < 0 || size > 4096) throw new RangeError('Use a node size from 0 to 4096.');
  if (size <= 1) return ZERO;
  let harmonic = ZERO;
  for (let term = 1; term < size; term += 1) harmonic = harmonic.add(exact(1, term));
  return harmonic.multiply(exact(2)).subtract(exact(2 * (size - 1), size));
}

export const isolationLimits = { points: { minimum: 2, maximum: 6 }, coordinate: { minimum: -20, maximum: 30 }, decimals: 2, depthCap: { minimum: 1, maximum: 5 } };

function checkCoordinates(values, limits = isolationLimits) {
  if (!Array.isArray(values) || values.length < limits.points.minimum || values.length > limits.points.maximum) {
    throw new RangeError(`Use ${limits.points.minimum} to ${limits.points.maximum} positions.`);
  }
  return values.map(value => {
    if (!Number.isFinite(value) || value < limits.coordinate.minimum || value > limits.coordinate.maximum) {
      throw new RangeError(`Keep every position between ${limits.coordinate.minimum} and ${limits.coordinate.maximum}.`);
    }
    return Exact.from(value, limits.decimals);
  });
}

/** The gaps a first cut can fall in, with the exact probability of each and the
 * two groups it would produce. A cut landing exactly on a value has probability
 * zero in this continuous construction. */
export function firstCutIntervals(values) {
  const points = checkCoordinates(values);
  const order = points.map((value, id) => ({ id, value })).sort((left, right) => left.value.compare(right.value) || left.id - right.id);
  const span = order.at(-1).value.subtract(order[0].value);
  if (span.isZero()) return { span, intervals: [], constant: true, order };
  const intervals = [];
  for (let index = 0; index + 1 < order.length; index += 1) {
    const width = order[index + 1].value.subtract(order[index].value);
    if (width.isZero()) continue;
    intervals.push({
      from: order[index].value,
      to: order[index + 1].value,
      width,
      probability: width.divide(span),
      left: order.slice(0, index + 1).map(entry => entry.id),
      right: order.slice(index + 1).map(entry => entry.id),
    });
  }
  return { span, intervals, constant: false, order };
}

/** Expected corrected path length for every position, integrating exactly over
 * all one-dimensional cut sequences down to the depth cap.
 *
 * A node is always a contiguous run of the sorted positions, because a cut is a
 * threshold. Each node is solved once and reused. */
export function isolationExpectations(values, depthCap = 3) {
  const points = checkCoordinates(values);
  if (!Number.isInteger(depthCap) || depthCap < isolationLimits.depthCap.minimum || depthCap > isolationLimits.depthCap.maximum) {
    throw new RangeError('Use a depth cap from 1 to 5.');
  }
  const order = points.map((value, id) => ({ id, value })).sort((left, right) => left.value.compare(right.value) || left.id - right.id);
  const sorted = order.map(entry => entry.value);
  const solved = new Map();

  /** Expected corrected path of every member of sorted[from, to) at this depth. */
  function solve(from, to, depth) {
    const key = `${from},${to},${depth}`;
    if (solved.has(key)) return solved.get(key);
    const size = to - from;
    const span = sorted[to - 1].subtract(sorted[from]);
    const terminal = size <= 1 || depth === depthCap || span.isZero();
    let result;
    if (terminal) {
      const value = exact(depth).add(correction(size));
      result = new Array(size).fill(value);
    } else {
      result = new Array(size).fill(ZERO);
      for (let cut = from; cut + 1 < to; cut += 1) {
        const width = sorted[cut + 1].subtract(sorted[cut]);
        if (width.isZero()) continue;
        const probability = width.divide(span);
        const left = solve(from, cut + 1, depth + 1);
        const right = solve(cut + 1, to, depth + 1);
        for (let index = 0; index < size; index += 1) {
          const child = from + index <= cut ? left[index] : right[from + index - cut - 1];
          result[index] = result[index].add(probability.multiply(child));
        }
      }
    }
    solved.set(key, result);
    return result;
  }

  const normalizer = correction(points.length);
  const lengths = solve(0, sorted.length, 0);
  const rows = order.map((entry, index) => {
    const mean = lengths[index];
    return {
      id: entry.id,
      value: entry.value,
      meanPath: mean,
      score: normalizer.isZero() ? null : 2 ** -(mean.toNumber() / normalizer.toNumber()),
    };
  });
  const byId = [...rows].sort((left, right) => left.id - right.id);
  const constant = sorted.at(-1).subtract(sorted[0]).isZero();
  const best = constant ? null : byId.reduce((leader, row) => (row.score > leader.score ? row : leader), byId[0]);
  const tied = constant || byId.filter(row => Math.abs(row.score - best.score) < 1e-12).length > 1;
  return { rows: byId, normalizer, depthCap, constant, shortest: constant ? null : best.id, tied };
}

/** Follow one chosen sequence of cuts for one query, reporting the node it sits
 * in at each depth. Cut positions are exact, so a cut that misses the node's
 * span is reported rather than silently ignored. */
export function isolationPath(values, cuts, queryId, depthCap = 3) {
  const points = checkCoordinates(values);
  if (!Number.isInteger(queryId) || queryId < 0 || queryId >= points.length) throw new RangeError('Select one of the positions.');
  let members = points.map((value, id) => ({ id, value })).sort((left, right) => left.value.compare(right.value) || left.id - right.id);
  const steps = [];
  for (let depth = 0; depth < depthCap; depth += 1) {
    const low = members[0].value;
    const high = members.at(-1).value;
    if (members.length <= 1 || high.subtract(low).isZero()) break;
    const cut = cuts[depth] === undefined ? null : Exact.from(cuts[depth], isolationLimits.decimals);
    if (cut === null || cut.compare(low) <= 0 || cut.compare(high) > 0) {
      steps.push({ depth, members: members.map(entry => entry.id), cut, usable: false });
      break;
    }
    const left = members.filter(entry => entry.value.compare(cut) < 0);
    const right = members.filter(entry => entry.value.compare(cut) >= 0);
    const side = left.some(entry => entry.id === queryId) ? 'left' : 'right';
    steps.push({ depth, members: members.map(entry => entry.id), cut, usable: true, side, leftIds: left.map(e => e.id), rightIds: right.map(e => e.id) });
    members = side === 'left' ? left : right;
  }
  const size = members.length;
  const depth = steps.filter(step => step.usable).length;
  const leafCorrection = correction(size);
  const pathLength = exact(depth).add(leafCorrection);
  const normalizer = correction(points.length);
  return {
    steps,
    leaf: { members: members.map(entry => entry.id), size, depth, correction: leafCorrection },
    pathLength,
    normalizer,
    score: normalizer.isZero() ? null : 2 ** -(pathLength.toNumber() / normalizer.toNumber()),
  };
}

export const lofLimits = { references: { minimum: 3, maximum: 8 }, coordinate: { minimum: -5, maximum: 33 }, decimals: 2, neighbours: { minimum: 1, maximum: 5 } };

/** Local Outlier Factor on a line, with exactly k other rows as neighbours and
 * stable row-order tie breaking. `query` scores a new observation against the
 * frozen references; omit it to read the reference rows' own factors. */
export function lofState(references, k = 2, query = null) {
  if (!Array.isArray(references) || references.length < lofLimits.references.minimum || references.length > lofLimits.references.maximum) {
    throw new RangeError(`Use ${lofLimits.references.minimum} to ${lofLimits.references.maximum} reference rows.`);
  }
  const points = references.map(value => {
    if (!Number.isFinite(value) || value < lofLimits.coordinate.minimum || value > lofLimits.coordinate.maximum) {
      throw new RangeError(`Keep every reference between ${lofLimits.coordinate.minimum} and ${lofLimits.coordinate.maximum}.`);
    }
    return Exact.from(value, lofLimits.decimals);
  });
  if (points.some((value, index) => points.some((other, position) => position !== index && other.equals(value)))) {
    throw new RangeError('This exact calculation needs distinct reference coordinates: repeated rows give a zero reach, and the reciprocal of zero is not a finite density.');
  }
  if (!Number.isInteger(k) || k < lofLimits.neighbours.minimum || k > Math.min(lofLimits.neighbours.maximum, points.length - 1)) {
    throw new RangeError(`Use k between 1 and ${Math.min(lofLimits.neighbours.maximum, points.length - 1)}: k counts other rows.`);
  }

  const distance = (left, right) => left.subtract(right).absolute();
  /** The k nearest other rows, excluding `excludeId` by identity rather than by coordinate. */
  const select = (from, excludeId) => points
    .map((value, id) => ({ id, value, distance: distance(from, value) }))
    .filter(entry => entry.id !== excludeId)
    .sort((left, right) => left.distance.compare(right.distance) || left.id - right.id)
    .slice(0, k);

  const neighbours = points.map((value, id) => select(value, id));
  const radii = neighbours.map(chosen => chosen.at(-1).distance);
  const reaches = neighbours.map(chosen => chosen.map(entry => {
    const floor = radii[entry.id];
    return { ...entry, radius: floor, reach: entry.distance.compare(floor) >= 0 ? entry.distance : floor };
  }));
  const densities = reaches.map(chosen => {
    const mean = chosen.reduce((sum, entry) => sum.add(entry.reach), ZERO).divide(exact(chosen.length));
    if (mean.isZero()) throw new RangeError('A zero mean reach has no finite density.');
    return ONE.divide(mean);
  });
  const factors = reaches.map((chosen, id) => chosen
    .reduce((sum, entry) => sum.add(densities[entry.id].divide(densities[id])), ZERO)
    .divide(exact(chosen.length)));
  const rows = points.map((value, id) => ({
    id, value, neighbours: reaches[id], radius: radii[id], density: densities[id], factor: factors[id],
  }));

  if (query === null) return { k, rows, query: null };
  if (!Number.isFinite(query) || query < lofLimits.coordinate.minimum || query > lofLimits.coordinate.maximum) {
    throw new RangeError(`Keep the query between ${lofLimits.coordinate.minimum} and ${lofLimits.coordinate.maximum}.`);
  }
  // A new query has no reference identity, so a reference sharing its coordinate
  // is an ordinary zero-distance neighbour.
  const queryValue = Exact.from(query, lofLimits.decimals);
  const chosen = select(queryValue, null).map(entry => {
    const floor = radii[entry.id];
    return { ...entry, radius: floor, reach: entry.distance.compare(floor) >= 0 ? entry.distance : floor };
  });
  const mean = chosen.reduce((sum, entry) => sum.add(entry.reach), ZERO).divide(exact(chosen.length));
  if (mean.isZero()) {
    return { k, rows, query: { value: queryValue, neighbours: chosen, density: null, factor: null, undefinedDensity: true } };
  }
  const density = ONE.divide(mean);
  const factor = chosen.reduce((sum, entry) => sum.add(densities[entry.id].divide(density)), ZERO).divide(exact(chosen.length));
  return { k, rows, query: { value: queryValue, neighbours: chosen, density, factor, undefinedDensity: false } };
}

/** Compare the two fitting modes at one coordinate: the reference row's own
 * factor, which excludes its identity, against a new query at the same place,
 * which does not. */
export function lofModeComparison(references, k, coordinate) {
  const state = lofState(references, k, coordinate);
  const matching = state.rows.find(row => row.value.equals(Exact.from(coordinate, lofLimits.decimals))) ?? null;
  return {
    coordinate: Exact.from(coordinate, lofLimits.decimals),
    trainingRow: matching,
    queryRow: state.query,
    agree: matching !== null && state.query.factor !== null && matching.factor.equals(state.query.factor),
    references: state.rows,
    k,
  };
}

export const kernelLimits = { anchor: { minimum: 0.5, maximum: 3 }, gamma: { minimum: 0.05, maximum: 2 }, query: { minimum: -4, maximum: 4 } };

/** The exact symmetric two-reference One-Class SVM at nu = 1/2.
 *
 * Symmetry gives both weights 1/2 and puts both references on the boundary, so
 * rho is fixed by the anchors and gamma alone. This is that closed solution,
 * not a general solver with the coefficients held wrongly fixed. */
export function kernelBoundary(anchor = 1, gamma = 1, query = 0) {
  for (const [value, limits, name] of [[anchor, kernelLimits.anchor, 'anchor'], [gamma, kernelLimits.gamma, 'gamma'], [query, kernelLimits.query, 'query']]) {
    if (!Number.isFinite(value) || value < limits.minimum || value > limits.maximum) {
      throw new RangeError(`Keep ${name} between ${limits.minimum} and ${limits.maximum}.`);
    }
  }
  const similarity = x => [Math.exp(-gamma * (x + anchor) ** 2), Math.exp(-gamma * (x - anchor) ** 2)];
  const rho = (1 + Math.exp(-4 * gamma * anchor * anchor)) / 2;
  const decide = x => {
    const [left, right] = similarity(x);
    return { contributions: [left / 2, right / 2], sum: (left + right) / 2, decision: (left + right) / 2 - rho };
  };
  // The region the model accepts is where the decision is positive. It need not
  // be one interval: at a large gamma the two bumps separate. Find the maximal
  // positive runs on a dense scan and bisect each edge down to the zero.
  const samples = 4096;
  const low = kernelLimits.query.minimum;
  const high = kernelLimits.query.maximum;
  const sampleAt = index => low + (high - low) * index / samples;
  const edge = (outside, inside) => {
    let a = outside;
    let b = inside;
    for (let iteration = 0; iteration < 60; iteration += 1) {
      const middle = (a + b) / 2;
      if (decide(middle).decision > 0) b = middle;
      else a = middle;
    }
    return (a + b) / 2;
  };
  const positiveIntervals = [];
  let start = null;
  for (let index = 0; index <= samples; index += 1) {
    const positive = decide(sampleAt(index)).decision > 0;
    if (positive && start === null) start = index;
    if (!positive && start !== null) {
      positiveIntervals.push([start === 0 ? low : edge(sampleAt(start - 1), sampleAt(start)), edge(sampleAt(index), sampleAt(index - 1))]);
      start = null;
    }
  }
  if (start !== null) positiveIntervals.push([start === 0 ? low : edge(sampleAt(start - 1), sampleAt(start)), high]);
  const crossings = positiveIntervals.flat().filter(x => x > low && x < high);
  const tolerance = 1e-12;
  const at = decide(query);
  return {
    anchor, gamma, query, rho,
    ...at,
    classification: Math.abs(at.decision) <= tolerance ? 'boundary' : at.decision > 0 ? 'inside' : 'outside',
    tolerance,
    crossings,
    positiveIntervals,
    atMidpoint: decide(0).decision,
    atAnchor: decide(anchor).decision,
    curve: Array.from({ length: 241 }, (_, index) => {
      const x = low + (high - low) * index / 240;
      const point = decide(x);
      return { x, left: point.contributions[0], right: point.contributions[1], sum: point.sum, decision: point.decision };
    }),
  };
}

export const alarmLimits = { population: { minimum: 100, maximum: 1000000 }, budget: { minimum: 0, maximum: 1000000 } };

/** Expected alert counts for a population with a declared prevalence and two
 * conditional rates. Counts are expectations, so they can be fractional; a
 * fraction of a record is not a record. */
export function alarmCounts(population = 100000, prevalence = 0.001, sensitivity = 0.8, falsePositiveRate = 0.01, budget = 200) {
  if (!Number.isInteger(population) || population < alarmLimits.population.minimum || population > alarmLimits.population.maximum) {
    throw new RangeError(`Use a population between ${alarmLimits.population.minimum} and ${alarmLimits.population.maximum}.`);
  }
  for (const [value, name] of [[prevalence, 'prevalence'], [sensitivity, 'sensitivity'], [falsePositiveRate, 'false-positive rate']]) {
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError(`Give the ${name} as a fraction between 0 and 1.`);
  }
  if (!Number.isInteger(budget) || budget < alarmLimits.budget.minimum || budget > alarmLimits.budget.maximum) {
    throw new RangeError('Use a whole review budget.');
  }
  const faults = population * prevalence;
  const nonFaults = population - faults;
  const trueAlerts = faults * sensitivity;
  const falseAlerts = nonFaults * falsePositiveRate;
  const total = trueAlerts + falseAlerts;
  return {
    population, prevalence, sensitivity, falsePositiveRate, budget,
    faults, nonFaults, trueAlerts, falseAlerts, total,
    precision: total > 0 ? trueAlerts / total : null,
    missedFaults: faults - trueAlerts,
    withinBudget: total <= budget,
    budgetShortfall: Math.max(0, total - budget),
  };
}

/** NumPy's higher-interpolation quantile picks this position in the sorted
 * calibration scores. */
export const quantileIndex = (quantile, size) => {
  if (!Number.isFinite(quantile) || quantile < 0 || quantile > 1) throw new RangeError('Use a quantile between 0 and 1.');
  return Math.ceil(quantile * (size - 1));
};

/** Unpack the four window hits carried in one integer. */
export const windowHits = packed => [0, 1, 2, 3].map(index => Boolean((packed >> index) & 1));

/** The timestamp of a feature row, derived from the first row and the verified
 * five-minute spacing. */
export function rowTimestamp(start, index, stepMinutes = 5) {
  const base = new Date(`${start.replace(' ', 'T')}Z`);
  return new Date(base.getTime() + index * stepMinutes * 60000);
}
export const formatTimestamp = date => date.toISOString().replace('T', ' ').slice(0, 16);
