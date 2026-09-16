/** Pure models for the AutoML & neural architecture search lesson.
 *
 * Everything a figure or investigation draws is computed here, so that a drawn
 * rung, band, frontier, shaded area, arrow or bar is an asserted mathematical
 * claim rather than a shape chosen inside a component.
 *
 * Three categories of quantity live here and are never mixed:
 *
 *   1. **Constructed** exact calculations: the conditional grammar's counts and
 *      sampling measures, expected improvement under a stated Gaussian
 *      surrogate, the nine declared fidelity curves, network parameter blocks,
 *      the operation mixture and its gradient, the scalar bilevel derivatives,
 *      the activation-code kernel, the portfolio matrix and the hypothetical
 *      deployment points. These expose mechanisms. None is a measurement.
 *   2. **Observed** replay: the functions that read the recorded banknote study
 *      from automl-data.js select, order and count over values that were fitted
 *      natively. They never fit anything, and they never reach an inspection
 *      outcome from a search prefix.
 *   3. **Hypothetical deployment** numbers: the five latency/accuracy pairs of
 *      the Pareto task are invented and are labelled so wherever they appear.
 *
 * Every entry point refuses input it cannot honour rather than substituting a
 * silent default: a non-finite number, an out-of-range setting, an empty branch
 * or a duplicated option all raise a RangeError.
 */

import { candidates as recordedCandidates, replayOrder } from './automl-data.js';

/* ------------------------------------------------------------------ guards */

export const limits = {
  /** I1: the conditional grammar a learner may edit. */
  penalty: { minimum: 0.001, maximum: 100 },
  depth: { minimum: 1, maximum: 12 },
  neighbors: { minimum: 1, maximum: 25 },
  width: { minimum: 1, maximum: 32 },
  maximumLayers: 2,
  maximumBranchOptions: 8,
  /** I2: surrogate coordinates, not observed classification error. */
  surrogateMean: { minimum: -1, maximum: 2 },
  surrogateDeviation: { minimum: 0, maximum: 1 },
  incumbent: { minimum: -1, maximum: 2 },
  /** I3: constructed loss curves on a fixed resource ladder. */
  loss: { minimum: 0, maximum: 1 },
  /** I4: the operation mixture. */
  input: { minimum: -5, maximum: 5 },
  target: { minimum: -5, maximum: 5 },
  logit: { minimum: -8, maximum: 8 },
  stepSize: { minimum: 0, maximum: 1 },
  /** I6: hypothetical deployment measurements. */
  latency: { minimum: 0.1, maximum: 30 },
  accuracy: { minimum: 0, maximum: 1 },
  cap: { minimum: 0.1, maximum: 30 },
  points: { minimum: 2, maximum: 8 },
  /** Shared numerical conventions. */
  tolerance: 1e-12,
  displayTolerance: 1e-5,
  densitySamples: 481,
};

/** The spinner step of every editable control, in one place.
 *
 * The components render these and the verifier checks the prose's values
 * against them, so "the value the prose asks a learner to type lands on its
 * control's step" is one fact rather than two transcriptions that can agree
 * with each other while both disagreeing with the page. */
export const controlSteps = {
  penalty: 0.01,
  incumbent: 0.05,
  surrogateMean: 0.05,
  surrogateDeviation: 0.01,
  loss: 0.01,
  input: 0.5,
  target: 0.5,
  logit: 0.1,
  stepSize: 0.1,
  latency: 0.5,
  accuracy: 0.01,
  cap: 0.5,
  budget: 1,
};

export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number.`);
  }
  return value;
}
export function checkRange(value, range, name) {
  checkFinite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
}
export function checkWhole(value, range, name) {
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be a whole number.`);
  return checkRange(value, range, name);
}
const sum = values => values.reduce((total, value) => total + value, 0);

/* =========================================================== §2 · I1 grammar */

/** The declared space of the observed study, and the shape every edit keeps.
 *
 * A branch is a family with one list per dimension that is simultaneously
 * active inside it. The count of a branch is the product of its dimension
 * sizes; the count of the space is the sum over branches. Multiplying every
 * option in the whole form together instead would count configurations that do
 * not exist, which is the misconception this investigation exists to remove.
 */
export const declaredSpace = {
  logistic: { scaling: ['raw', 'standard'], penalty: [0.1, 1] },
  tree: { depth: [2, 5] },
  neighbors: { scaling: ['standard'], count: [3, 9] },
  mlp: { scaling: ['standard'], activation: ['tanh'], widths: [[8], [16], [8, 8]] },
};

export const familyOrder = ['logistic', 'tree', 'neighbors', 'mlp'];
export const familyLabels = {
  logistic: 'Logistic regression',
  tree: 'Decision tree',
  neighbors: 'Nearest neighbors',
  mlp: 'Small neural network',
};
export const dimensionLabels = {
  scaling: 'preprocessing', penalty: 'C', depth: 'maximum depth',
  count: 'neighbors', activation: 'activation', widths: 'hidden widths',
};
/** Dimensions the grammar fixes: they are shown, but they are not choices. */
export const fixedDimensions = { neighbors: ['scaling'], mlp: ['scaling', 'activation'] };

const optionKey = value => (Array.isArray(value) ? value.join('x') : String(value));

export function describeOption(dimension, value) {
  if (dimension === 'widths') return `(${value.join(', ')})`;
  if (dimension === 'penalty') return `C = ${value}`;
  return String(value);
}

/** Reject a branch that cannot produce a configuration, and refuse duplicates
 * rather than counting a repeated value as a second algorithm. */
export function normalizeSpace(space) {
  const result = {};
  for (const family of familyOrder) {
    const branch = space[family];
    if (!branch) throw new RangeError(`The grammar needs its ${familyLabels[family]} branch.`);
    result[family] = {};
    for (const [dimension, values] of Object.entries(branch)) {
      if (!Array.isArray(values) || values.length === 0) {
        throw new RangeError(`${familyLabels[family]} needs at least one ${dimensionLabels[dimension]}.`);
      }
      if (values.length > limits.maximumBranchOptions) {
        throw new RangeError(`Keep ${dimensionLabels[dimension]} to ${limits.maximumBranchOptions} values or fewer.`);
      }
      const seen = new Set();
      for (const value of values) {
        checkOption(family, dimension, value);
        const key = optionKey(value);
        if (seen.has(key)) {
          throw new RangeError(`${describeOption(dimension, value)} is already a permitted ${dimensionLabels[dimension]}.`);
        }
        seen.add(key);
      }
      result[family][dimension] = values.map(value => (Array.isArray(value) ? [...value] : value));
    }
  }
  return result;
}

export function checkOption(family, dimension, value) {
  if (dimension === 'scaling') {
    if (!['raw', 'standard'].includes(value)) throw new RangeError('Preprocessing is raw or standard scaling.');
    return value;
  }
  if (dimension === 'activation') {
    if (value !== 'tanh') throw new RangeError('This grammar fixes the activation at tanh.');
    return value;
  }
  if (dimension === 'penalty') return checkRange(value, limits.penalty, 'C');
  if (dimension === 'depth') return checkWhole(value, limits.depth, 'a tree depth');
  if (dimension === 'count') return checkWhole(value, limits.neighbors, 'a neighbor count');
  if (dimension === 'widths') {
    if (!Array.isArray(value) || value.length < 1 || value.length > limits.maximumLayers) {
      throw new RangeError(`A width pattern holds one or ${limits.maximumLayers} hidden layers.`);
    }
    value.forEach(width => checkWhole(width, limits.width, 'a hidden width'));
    return value;
  }
  throw new RangeError(`${dimension} is not a dimension of this grammar.`);
}

/** Counts per branch and for the space, with the expression that produced them. */
export function countConfigurations(space) {
  const normalized = normalizeSpace(space);
  const branches = familyOrder.map(family => {
    const dimensions = Object.entries(normalized[family]).map(([dimension, values]) => ({
      dimension, label: dimensionLabels[dimension], size: values.length,
      fixed: (fixedDimensions[family] ?? []).includes(dimension),
      options: values.map(value => describeOption(dimension, value)),
    }));
    const count = dimensions.reduce((product, entry) => product * entry.size, 1);
    return {
      family, label: familyLabels[family], dimensions, count,
      expression: dimensions.map(entry => entry.size).join(' × '),
    };
  });
  const total = sum(branches.map(branch => branch.count));
  return {
    branches, total,
    expression: `${branches.map(branch => (branch.dimensions.length > 1 ? `(${branch.expression})` : branch.expression)).join(' + ')} = ${total}`,
  };
}

/** Enumerate the space, so a count is never only a formula. */
export function enumerateConfigurations(space) {
  const normalized = normalizeSpace(space);
  const rows = [];
  for (const family of familyOrder) {
    const entries = Object.entries(normalized[family]);
    const build = (index, chosen) => {
      if (index === entries.length) {
        rows.push({ family, settings: { ...chosen } });
        return;
      }
      const [dimension, values] = entries[index];
      for (const value of values) build(index + 1, { ...chosen, [dimension]: value });
    };
    build(0, {});
  }
  return rows;
}

/** Add one permitted value to one branch, and report what the space gained. */
export function addOption(space, family, dimension, value) {
  const before = countConfigurations(space);
  const branch = space[family];
  if (!branch || !branch[dimension]) {
    throw new RangeError(`${familyLabels[family]} has no ${dimension} to extend.`);
  }
  const next = {
    ...space,
    [family]: { ...branch, [dimension]: [...branch[dimension], value] },
  };
  const after = countConfigurations(next);
  return {
    space: normalizeSpace(next), before: before.total, after: after.total,
    increment: after.total - before.total,
    /** The increment is the branch's count divided by the extended dimension's
     * old size: the new value pairs with every other active setting. */
    pairedWith: before.branches.find(entry => entry.family === family).count / branch[dimension].length,
  };
}

export function removeOption(space, family, dimension, index) {
  const values = space[family]?.[dimension];
  if (!values) throw new RangeError(`${familyLabels[family]} has no ${dimension} to shorten.`);
  if (values.length <= 1) {
    throw new RangeError(`${familyLabels[family]} needs at least one ${dimensionLabels[dimension]}.`);
  }
  const next = { ...space, [family]: { ...space[family], [dimension]: values.filter((_, position) => position !== index) } };
  const before = countConfigurations(space).total;
  const after = countConfigurations(next).total;
  return { space: normalizeSpace(next), before, after, increment: after - before };
}

/** The recipe a selection actually describes. A retained value for a family
 * that is not selected is a draft: it appears nowhere in the active recipe and
 * changes neither the fitted procedure nor the count. */
export function activeRecipe(space, selection) {
  const normalized = normalizeSpace(space);
  const { family } = selection;
  if (!familyOrder.includes(family)) throw new RangeError(`${family} is not a family of this grammar.`);
  const chosen = selection.choices?.[family] ?? {};
  const settings = [];
  for (const [dimension, values] of Object.entries(normalized[family])) {
    const index = Math.min(chosen[dimension] ?? 0, values.length - 1);
    settings.push({
      dimension, label: dimensionLabels[dimension], value: values[index],
      shown: describeOption(dimension, values[index]),
      fixed: (fixedDimensions[family] ?? []).includes(dimension),
    });
  }
  return {
    family, label: familyLabels[family], settings,
    /** A stable serialization of the active configuration alone. */
    serialized: `${family}{${settings.map(entry => `${entry.dimension}=${optionKey(entry.value)}`).join(',')}}`,
    inactiveFamilies: familyOrder.filter(other => other !== family),
  };
}

/** Two sampling rules over the same space give different family probabilities.
 * Neither is unbiased without naming the reference measure. */
export function samplingMeasure(space, rule) {
  const counted = countConfigurations(space);
  if (!['family-uniform', 'configuration-uniform'].includes(rule)) {
    throw new RangeError('The sampling rule is family-uniform or configuration-uniform.');
  }
  const bands = counted.branches.map(branch => ({
    family: branch.family, label: branch.label, count: branch.count,
    probability: rule === 'family-uniform' ? 1 / counted.branches.length : branch.count / counted.total,
    fraction: rule === 'family-uniform'
      ? `1/${counted.branches.length}`
      : `${branch.count}/${counted.total}`,
  }));
  return { rule, bands, total: sum(bands.map(band => band.probability)) };
}

/** Log-uniform sampling gives each decade equal probability; sampling the value
 * uniformly gives the largest decade almost all of it. */
export function decadeWeights(lowExponent, highExponent) {
  checkFinite(lowExponent, 'the low exponent');
  checkFinite(highExponent, 'the high exponent');
  if (!(highExponent > lowExponent)) throw new RangeError('The exponent range must be increasing.');
  const span = 10 ** highExponent - 10 ** lowExponent;
  const decades = [];
  for (let exponent = lowExponent; exponent < highExponent; exponent += 1) {
    decades.push({
      from: 10 ** exponent, to: 10 ** (exponent + 1),
      logUniform: 1 / (highExponent - lowExponent),
      valueUniform: (10 ** (exponent + 1) - 10 ** exponent) / span,
    });
  }
  return decades;
}

/* ============================================= §2 · I2 expected improvement */

/** Laplace continued fraction for the normal tail (NIST DLMF 7.9.1,
 * https://dlmf.nist.gov/7.9.E1 after z=a/sqrt(2)). For a>=2,128 terms
 * avoid cancellation in both Phi(-a) and phi(a)-a*Phi(-a). */
function normalTailCorrection(a) {
  let correction = 0;
  for (let n = 128; n >= 1; n -= 1) correction = n / (a + correction);
  return correction;
}

/** Phi, with a direct small tail rather than subtracting erf from one. */
export function normalCdf(value) {
  checkFinite(value, 'the standardized distance');
  // Abramowitz & Stegun 7.1.26 is not accurate enough for a displayed oracle;
  // this is the standard erf series/continued-fraction split, good to ~1e-15.
  if (Math.abs(value) >= 2) {
    const a = Math.abs(value);
    const tail = normalPdf(a) / (a + normalTailCorrection(a));
    return value < 0 ? tail : 1 - tail;
  }
  return 0.5 * (1 + erf(value / Math.SQRT2));
}
export function normalPdf(value) {
  checkFinite(value, 'the standardized distance');
  return Math.exp(-0.5 * value * value) / Math.sqrt(2 * Math.PI);
}
/** erf through its all-positive-terms series
 *
 *     erf(z) = (2/√π) e^{−z²} Σ_{n≥0} 2^n z^{2n+1} / (1·3·5···(2n+1)),
 *
 * whose terms never alternate, so it suffers none of the cancellation that
 * makes the ordinary Maclaurin series lose digits past |z| ≈ 3. Beyond |z| = 6
 * the complement is under 3 × 10⁻¹⁷ and erf is 1 to double precision. */
export function erf(x) {
  checkFinite(x, 'the error-function argument');
  const sign = Math.sign(x);
  const z = Math.abs(x);
  if (z >= 6) return sign;
  let term = z;
  let total = z;
  for (let n = 1; n < 400; n += 1) {
    term *= (2 * z * z) / (2 * n + 1);
    total += term;
    if (term < total * 1e-18) break;
  }
  return sign * (2 / Math.sqrt(Math.PI)) * Math.exp(-z * z) * total;
}

/** EI = E[max(b − F, 0)] for a Gaussian predictive F. A zero deviation is a
 * point mass, not a bell curve with an epsilon substituted for it. */
export function expectedImprovement(best, mean, deviation) {
  checkFinite(best, 'the incumbent loss');
  checkFinite(mean, 'a predicted mean');
  checkFinite(deviation, 'the predicted standard deviation');
  if (deviation < 0) throw new RangeError('A predicted standard deviation cannot be negative.');
  if (deviation === 0) return Math.max(best - mean, 0);
  const z = (best - mean) / deviation;
  if (z <= -2) {
    const a = -z;
    const correction = normalTailCorrection(a);
    return deviation * normalPdf(a) * correction / (a + correction);
  }
  return (best - mean) * normalCdf(z) + deviation * normalPdf(z);
}

/** The whole acquisition comparison, with ties reported as ties. */
export function acquisitionTable(best, candidates) {
  checkRange(best, limits.incumbent, 'the incumbent loss');
  if (!Array.isArray(candidates) || candidates.length < 2 || candidates.length > 5) {
    throw new RangeError('Compare between two and five surrogate predictions.');
  }
  const rows = candidates.map(candidate => {
    checkRange(candidate.mean, limits.surrogateMean, `${candidate.id}'s predicted mean`);
    checkRange(candidate.deviation, limits.surrogateDeviation, `${candidate.id}'s predicted deviation`);
    const value = expectedImprovement(best, candidate.mean, candidate.deviation);
    return {
      id: candidate.id, mean: candidate.mean, deviation: candidate.deviation, value,
      deterministic: candidate.deviation === 0,
      probabilityOfImprovement: candidate.deviation === 0
        ? (best > candidate.mean ? 1 : 0)
        : normalCdf((best - candidate.mean) / candidate.deviation),
    };
  });
  const highest = Math.max(...rows.map(row => row.value));
  const leaders = rows.filter(row => Math.abs(row.value - highest) <= limits.tolerance * Math.max(1, highest));
  return {
    best, rows, highest,
    winner: leaders.length === 1 ? leaders[0].id : null,
    tied: leaders.length > 1 ? leaders.map(row => row.id) : [],
    /** Lower predicted mean and higher acquisition can disagree; that is the point. */
    lowestMean: rows.reduce((low, row) => (row.mean < low.mean ? row : low)).id,
  };
}

/** One loss axis wide enough for every predictive distribution and the
 * incumbent, so the three panels can be compared and the drawn area is not a
 * clipped view of the analytic integral. */
export function acquisitionDomain(best, candidates) {
  const edges = candidates.flatMap(candidate => [
    candidate.mean - 5 * candidate.deviation, candidate.mean + 5 * candidate.deviation,
  ]).concat([best]);
  const low = Math.min(...edges);
  const high = Math.max(...edges);
  const pad = Math.max((high - low) * 0.08, 0.02);
  return [low - pad, high + pad];
}

/** The two drawn curves: the predictive density, and the density weighted by the
 * improvement it would buy. Integrating the second gives EI, which is what the
 * shaded area must mean. Shading probability mass alone would draw a different
 * quantity. */
export function improvementDensity(best, mean, deviation, domain, samples = limits.densitySamples) {
  const grid = Array.from({ length: samples }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / (samples - 1));
  if (deviation > 0) {
    for (let index = 0; index <= 240; index += 1) grid.push(mean + deviation * (-8 + index / 15));
    const z = (best - mean) / deviation;
    const root = z > 0 ? -2 / (z + Math.hypot(z, 2)) : (z - Math.hypot(z, 2)) / 2;
    grid.push(mean, best, mean + deviation * root);
  }
  const points = [...new Set(grid.filter(loss => loss >= domain[0] && loss <= domain[1]))]
    .sort((a, b) => a - b).map(loss => {
      const density = deviation === 0 ? 0 : normalPdf((loss - mean) / deviation) / deviation;
      return { loss, density, weighted: loss < best ? (best - loss) * density : 0 };
    });
  const step = (domain[1] - domain[0]) / (samples - 1);
  // Nonuniform trapezoids: local sigma sampling resolves narrow candidates.
  let area = 0;
  for (let index = 1; index < points.length; index += 1) {
    area += (points[index].weighted + points[index - 1].weighted) / 2 * (points[index].loss - points[index - 1].loss);
  }
  return {
    points, domain, step, area,
    exact: expectedImprovement(best, mean, deviation),
    pointMass: deviation === 0 ? { loss: mean, improvement: Math.max(best - mean, 0) } : null,
    truncated: deviation > 0 && (mean - 5 * deviation < domain[0] || mean + 5 * deviation > domain[1]),
  };
}

/* =================================================== §3 · I3 successive halving */

export const declaredCurves = [
  { id: 'A', losses: [0.10, 0.10, 0.10] },
  { id: 'B', losses: [0.12, 0.09, 0.08] },
  { id: 'C', losses: [0.13, 0.08, 0.07] },
  { id: 'D', losses: [0.14, 0.07, 0.02] },
  { id: 'E', losses: [0.20, 0.15, 0.09] },
  { id: 'F', losses: [0.25, 0.20, 0.10] },
  { id: 'G', losses: [0.30, 0.25, 0.20] },
  { id: 'H', losses: [0.35, 0.30, 0.25] },
  { id: 'I', losses: [0.40, 0.30, 0.20] },
];
export const declaredResource = [1, 3, 9];

/** One rung at a time, ranking only the losses that were actually purchased.
 * A candidate's later values never influence the cut that removed it. */
export function halvingSchedule(curves = declaredCurves, resource = declaredResource, keepFraction = 3) {
  if (!Array.isArray(curves) || curves.length < keepFraction) {
    throw new RangeError(`Run at least ${keepFraction} candidates.`);
  }
  curves.forEach(curve => {
    if (curve.losses.length !== resource.length) throw new RangeError('Every curve needs one loss per rung.');
    curve.losses.forEach((loss, rung) => checkRange(loss, limits.loss, `${curve.id}'s loss at rung ${rung + 1}`));
  });
  let alive = curves.map((_, index) => index);
  const rungs = [];
  for (let rung = 0; rung < resource.length; rung += 1) {
    const ranked = [...alive].sort((a, b) => (curves[a].losses[rung] - curves[b].losses[rung]) || (a - b));
    // Never let a rung empty the field: with three or four candidates on a
    // three-rung ladder, floor(n/3) reaches 0 and the next rung dereferences
    // `curves[undefined]`. The guard above admits those counts, so the body
    // must handle them rather than throwing a bare TypeError.
    const keep = rung === resource.length - 1 ? 1 : Math.max(1, Math.floor(alive.length / keepFraction));
    const survivors = ranked.slice(0, keep);
    const cutoff = curves[ranked[Math.max(keep - 1, 0)]].losses[rung];
    rungs.push({
      rung, resource: resource[rung], started: [...alive],
      ranked: ranked.map(index => ({ id: curves[index].id, loss: curves[index].losses[rung] })),
      survivors: survivors.map(index => curves[index].id),
      eliminated: ranked.slice(keep).map(index => ({
        id: curves[index].id, loss: curves[index].losses[rung], cutoff,
        reason: curves[index].losses[rung] === cutoff
          ? `${curves[index].losses[rung]} ties the cutoff; the declared identifier order breaks the tie`
          : `${curves[index].losses[rung]} is worse than the cutoff ${cutoff} at ${resource[rung]} resource units`,
      })),
      /** Only the purchased column. Nothing here can see a later value. */
      purchasedColumn: rung,
    });
    alive = survivors;
    if (rung === resource.length - 1) break;
  }
  const selected = curves[alive[0]];
  const counterfactualIndex = curves.reduce(
    (bestIndex, curve, index) => (curve.losses.at(-1) < curves[bestIndex].losses.at(-1) ? index : bestIndex), 0);
  const sizes = rungs.map(entry => entry.started.length);
  const restart = sum(sizes.map((size, rung) => size * resource[rung]));
  const resume = sizes[0] * resource[0]
    + sum(sizes.slice(1).map((size, index) => size * (resource[index + 1] - resource[index])));
  return {
    rungs, resource, keepFraction,
    selectedId: selected.id, selectedFinalLoss: selected.losses.at(-1),
    counterfactualId: curves[counterfactualIndex].id,
    counterfactualFinalLoss: curves[counterfactualIndex].losses.at(-1),
    missedBetter: curves[counterfactualIndex].losses.at(-1) < selected.losses.at(-1),
    work: {
      restart, resume, allFull: curves.length * resource.at(-1),
      restartExpression: sizes.map((size, rung) => `${size}(${resource[rung]})`).join(' + '),
      resumeExpression: [`${sizes[0]}(${resource[0]})`, ...sizes.slice(1).map(
        (size, index) => `${size}(${resource[index + 1]} − ${resource[index]})`)].join(' + '),
      allFullExpression: `${curves.length}(${resource.at(-1)})`,
    },
    /** Purchased traces stop where a candidate was eliminated; unpurchased
     * values are returned separately so a drawing cannot confuse the two. */
    traces: curves.map((curve, index) => {
      const lastRung = rungs.reduce(
        (found, entry) => (entry.started.includes(index) ? entry.rung : found), -1);
      return {
        id: curve.id,
        purchased: curve.losses.slice(0, lastRung + 1).map((loss, rung) => [resource[rung], loss]),
        unpurchased: curve.losses.slice(lastRung).map((loss, offset) => [resource[lastRung + offset], loss]),
        survived: curve.id === selected.id,
      };
    }),
  };
}

/** Hyperband's second loop. Allocation uses the paper's floor/ceil rounding, so
 * the brackets do not evaluate the same candidate set and their work does not
 * add up to one comparable full-resource campaign. */
export function hyperbandBrackets(maximumResource = 9, eta = 3) {
  checkWhole(maximumResource, { minimum: 1, maximum: 81 }, 'the maximum resource');
  checkWhole(eta, { minimum: 2, maximum: 5 }, 'the elimination factor');
  const rounds = Math.floor(Math.log(maximumResource) / Math.log(eta));
  const brackets = [];
  for (let s = rounds; s >= 0; s -= 1) {
    const startCount = Math.ceil(((rounds + 1) / (s + 1)) * eta ** s);
    const startResource = maximumResource / eta ** s;
    const stages = [];
    for (let i = 0; i <= s; i += 1) {
      stages.push({
        candidates: Math.floor(startCount / eta ** i),
        resource: startResource * eta ** i,
      });
    }
    brackets.push({
      bracket: s, startCount, startResource, stages,
      restartWork: sum(stages.map(stage => stage.candidates * stage.resource)),
    });
  }
  return {
    brackets, maximumResource, eta,
    totalRestartWork: sum(brackets.map(bracket => bracket.restartWork)),
  };
}

/* ======================================================= §4 · F2 architecture */

/** Parameter blocks of a fully connected stack, biases counted explicitly. */
export function networkBlocks(inputs, widths, outputs = 1) {
  checkWhole(inputs, { minimum: 1, maximum: 64 }, 'the input width');
  checkWhole(outputs, { minimum: 1, maximum: 8 }, 'the output width');
  if (!Array.isArray(widths) || widths.length < 1 || widths.length > 4) {
    throw new RangeError('Use between one and four hidden layers.');
  }
  widths.forEach(width => checkWhole(width, limits.width, 'a hidden width'));
  const sizes = [inputs, ...widths, outputs];
  const blocks = sizes.slice(0, -1).map((from, index) => {
    const to = sizes[index + 1];
    const last = index === sizes.length - 2;
    return {
      index, from, to, weights: from * to, biases: to, total: from * to + to,
      shape: `${from} × ${to}`,
      equation: `(${from} + 1) × ${to} = ${from * to + to}`,
      activation: last ? 'sigmoid' : 'tanh',
      role: last ? 'output affine block' : `hidden affine block ${index + 1}`,
    };
  });
  return {
    inputs, widths, outputs, sizes, blocks,
    total: sum(blocks.map(block => block.total)),
    expression: `${blocks.map(block => block.equation.split(' = ')[0]).join(' + ')} = ${sum(blocks.map(block => block.total))}`,
    shapePath: sizes.join(' → '),
  };
}

/** The single-hidden-layer shortcut the manuscript states for four inputs. */
export function singleLayerCount(width, inputs = 4) {
  return networkBlocks(inputs, [width]).total;
}

/* ================================================== §4 · F1 the evidence loop */

/** The stages and edges of the evidence diagram. Kept here so that "no arrow
 * carries an inspection score back to the proposer" is an assertion rather than
 * a reading of the drawing. */
export const evidenceFlow = {
  stages: [
    { id: 'proposer', label: 'Configuration proposer', role: 'selection' },
    { id: 'fit', label: 'Fit complete pipeline', role: 'selection', nested: 'fit scaler + model on fitting rows' },
    { id: 'score', label: 'Score validation predictions', role: 'selection' },
    { id: 'record', label: 'Search record', role: 'selection' },
    { id: 'refit', label: 'Refit on all development rows', role: 'commitment' },
    { id: 'inspect', label: 'Inspection comparison', role: 'assessment' },
    { id: 'reserved', label: 'Reserved rows', role: 'untouched' },
  ],
  edges: [
    { from: 'proposer', to: 'fit', carries: 'a candidate configuration' },
    { from: 'fit', to: 'score', carries: 'a fitted pipeline' },
    { from: 'score', to: 'record', carries: 'a validation accuracy' },
    { from: 'record', to: 'proposer', carries: 'the evidence that can change the next proposal' },
    { from: 'record', to: 'refit', carries: 'the fixed selected configuration' },
    { from: 'refit', to: 'inspect', carries: 'two frozen fitted procedures' },
  ],
  rowSources: [
    { id: 'fitting', label: 'fitting rows', to: 'fit' },
    { id: 'validation', label: 'validation rows', to: 'score' },
    { id: 'inspection', label: 'inspection rows', to: 'inspect' },
  ],
  bracket: { from: 'proposer', to: 'record', label: 'selection' },
};

/** Structural facts the figure must keep true. */
export function evidenceBoundary(flow = evidenceFlow) {
  const ids = new Set(flow.stages.map(stage => stage.id));
  flow.edges.forEach(edge => {
    if (!ids.has(edge.from) || !ids.has(edge.to)) throw new RangeError('An arrow leaves the declared stages.');
  });
  const into = id => flow.edges.filter(edge => edge.to === id);
  const outOf = id => flow.edges.filter(edge => edge.from === id);
  return {
    loop: ['proposer', 'fit', 'score', 'record'],
    feedbackEdge: flow.edges.find(edge => edge.from === 'record' && edge.to === 'proposer'),
    inspectionFeedsNothing: outOf('inspect').length === 0,
    reservedIsIsolated: outOf('reserved').length === 0 && into('reserved').length === 0,
    validationRowsNeverFit: !flow.rowSources.some(source => source.id === 'validation' && source.to === 'fit'),
    scalerRefitPerFold: flow.stages.find(stage => stage.id === 'fit').nested,
    selectionStages: flow.stages.filter(stage => stage.role === 'selection').map(stage => stage.id),
  };
}

/* ================================================= §5 · F3/I5 observed replay */

/** Plotted geometry for the observed candidate comparison: one mark per fold
 * accuracy and a separate mark for the arithmetic mean that owns selection. */
export function candidateSeries(records = recordedCandidates) {
  const rows = records.map((record, index) => ({
    index, id: record.id, label: record.label, family: record.family,
    parameterCount: record.parameterCount,
    folds: record.foldAccuracy.map((accuracy, fold) => ({
      fold, accuracy, correct: record.foldCorrect[fold], rows: record.foldRows[fold],
    })),
    mean: record.meanFoldAccuracy,
    pooled: record.pooledOofAccuracy,
    pooledCorrect: record.pooledCorrect,
    pooledRows: record.pooledRows,
    errorRows: record.outOfFoldErrorRows,
  }));
  const values = rows.flatMap(row => [...row.folds.map(fold => fold.accuracy), row.mean]);
  const bestMean = Math.max(...rows.map(row => row.mean));
  const beneath = rows.filter(row => row.mean < bestMean).map(row => row.mean);
  const runnerUpMean = beneath.length ? Math.max(...beneath) : null;
  /** The candidates that tie for second on the selection criterion.
   *
   * This is not the same set as "everyone who misses the hardest row": several
   * much weaker candidates also miss that row, among many others. Conflating
   * the two put seven names behind the word "three" on the page once. */
  const tiedRunnersUp = runnerUpMean === null
    ? []
    : rows.filter(row => Math.abs(row.mean - runnerUpMean) <= limits.tolerance);
  const sharedByTied = tiedRunnersUp.length
    ? tiedRunnersUp[0].errorRows.filter(row => tiedRunnersUp.every(entry => entry.errorRows.includes(row)))
    : [];
  return {
    rows,
    /** A nonzero axis start, reported so the caption can say so. */
    domain: [Math.min(...values), Math.max(...values)],
    bestMean,
    runnerUpMean,
    tiedRunnersUp,
    /** The mistakes every tied candidate makes, and nothing else. */
    sharedByTied,
    /** Whether the tie is fully explained by one shared mistake apiece. */
    tieExplainedByOneMistake: tiedRunnersUp.length > 1 && sharedByTied.length === 1
      && tiedRunnersUp.every(row => row.errorRows.length === 1),
    /** The mean of fold accuracies and pooled out-of-fold accuracy are two
     * different rules; unequal fold sizes make them differ. */
    meanAndPooledAgree: rows.every(row => Math.abs(row.mean - row.pooled) <= limits.tolerance),
  };
}

/** Which candidates share a mistake, which is what makes the tie tangible. */
export function sharedErrors(records = recordedCandidates) {
  const byRow = new Map();
  records.forEach(record => record.outOfFoldErrorRows.forEach(row => {
    byRow.set(row, [...(byRow.get(row) ?? []), record.id]);
  }));
  return [...byRow.entries()]
    .map(([row, ids]) => ({ row, ids }))
    .sort((a, b) => b.ids.length - a.ids.length || a.row - b.row);
}

/** Replay a finite table under a budget. The recommendation uses the registry
 * tie rule, which is independent of the order the candidates arrive in. */
export function replayPrefix({ records = recordedCandidates, order = replayOrder, enabled, budget }) {
  const available = (enabled ?? records.map((_, index) => index));
  if (!Array.isArray(available) || available.length === 0) {
    throw new RangeError('Keep at least one candidate enabled.');
  }
  available.forEach(index => checkWhole(index, { minimum: 0, maximum: records.length - 1 }, 'a candidate index'));
  const sequence = order.filter(index => available.includes(index));
  if (new Set(sequence).size !== sequence.length) throw new RangeError('A candidate cannot appear twice in the order.');
  checkWhole(budget, { minimum: 1, maximum: sequence.length }, 'the budget');
  const revealed = sequence.slice(0, budget);
  const steps = revealed.map((index, position) => {
    const seen = revealed.slice(0, position + 1);
    const best = Math.max(...seen.map(candidate => records[candidate].meanFoldAccuracy));
    const recommended = Math.min(...seen.filter(
      candidate => records[candidate].meanFoldAccuracy === best));
    const previous = position === 0 ? null : (() => {
      const earlier = revealed.slice(0, position);
      const earlierBest = Math.max(...earlier.map(candidate => records[candidate].meanFoldAccuracy));
      return {
        best: earlierBest,
        recommended: Math.min(...earlier.filter(candidate => records[candidate].meanFoldAccuracy === earlierBest)),
      };
    })();
    return {
      position: position + 1, index, id: records[index].id, label: records[index].label,
      score: records[index].meanFoldAccuracy,
      best, recommended, recommendedId: records[recommended].id,
      scoreImproved: previous ? best > previous.best : true,
      recommendationChanged: previous ? recommended !== previous.recommended : true,
      /** A flat maximum with a changed recommendation is exactly the case the
       * prose asks the learner to predict. */
      tieDecided: previous ? best === previous.best && recommended !== previous.recommended : false,
      fits: (position + 1) * 3,
    };
  });
  const last = steps.at(-1);
  return {
    order: sequence, revealed, steps,
    best: last.best, recommended: last.recommended, recommendedId: last.recommendedId,
    fits: last.fits,
    tieRule: 'lowest original registry index among candidates sharing the greatest mean fold accuracy',
    /** The replay owns no inspection outcome. */
    inspectionAvailable: false,
  };
}

/** Does moving one more candidate change the score, the recommendation, both or
 * neither? Graded against the committed budget, never against live state. */
export function replayComparison(before, after) {
  return {
    before, after,
    scoreImproved: after.best > before.best + limits.tolerance,
    scoreChanged: Math.abs(after.best - before.best) > limits.tolerance,
    recommendationChanged: after.recommended !== before.recommended,
    outcome: Math.abs(after.best - before.best) > limits.tolerance
      ? (after.recommended !== before.recommended ? 'both' : 'score')
      : (after.recommended !== before.recommended ? 'recommendation' : 'neither'),
    extraFits: after.fits - before.fits,
  };
}

/* ================================================== §6 · portfolio and Pareto */

export const declaredPortfolio = {
  ids: ['A', 'B', 'C'],
  oldTaskLosses: [[0.10, 0.50], [0.50, 0.10], [0.25, 0.25]],
  newTaskLosses: [0.40, 0.40, 0.20],
};

/** A portfolio covers complementary strengths; the best single default need not
 * belong to the best pair, and a new task can reverse the advantage. */
export function portfolioAnalysis(record = declaredPortfolio) {
  const { ids, oldTaskLosses, newTaskLosses } = record;
  const means = oldTaskLosses.map(row => sum(row) / row.length);
  const singleIndex = means.indexOf(Math.min(...means));
  const pairs = [];
  for (let a = 0; a < ids.length; a += 1) {
    for (let b = a + 1; b < ids.length; b += 1) {
      const perTask = oldTaskLosses[a].map((_, task) => Math.min(oldTaskLosses[a][task], oldTaskLosses[b][task]));
      pairs.push({ members: [ids[a], ids[b]], indices: [a, b], perTask, mean: sum(perTask) / perTask.length });
    }
  }
  const bestPair = pairs.reduce((low, pair) => (pair.mean < low.mean ? pair : low));
  return {
    ids, oldTaskLosses, means, singleId: ids[singleIndex], singleMean: means[singleIndex],
    pairs, bestPair,
    newTaskLosses,
    newTaskPortfolioBest: Math.min(...bestPair.indices.map(index => newTaskLosses[index])),
    newTaskOverallBest: Math.min(...newTaskLosses),
    newTaskOverallBestId: ids[newTaskLosses.indexOf(Math.min(...newTaskLosses))],
    portfolioMissesNewBest: Math.min(...bestPair.indices.map(index => newTaskLosses[index])) > Math.min(...newTaskLosses),
  };
}

export const declaredPareto = [
  { id: 'A', latency: 2, accuracy: 0.90 },
  { id: 'B', latency: 4, accuracy: 0.94 },
  { id: 'C', latency: 8, accuracy: 0.96 },
  { id: 'D', latency: 5, accuracy: 0.93 },
  { id: 'E', latency: 3, accuracy: 0.89 },
];

/** Dominance, the frontier, feasibility under a hard cap, and the chosen point.
 * Dominated means: some other point is no worse in both objectives and strictly
 * better in at least one. Identical pairs therefore do not dominate each other.
 *
 * These latency numbers are hypothetical throughout; nothing here is measured.
 */
export function paretoAnalysis(points = declaredPareto, cap = 5) {
  if (!Array.isArray(points) || points.length < limits.points.minimum || points.length > limits.points.maximum) {
    throw new RangeError(`Compare between ${limits.points.minimum} and ${limits.points.maximum} candidates.`);
  }
  points.forEach(point => {
    checkRange(point.latency, limits.latency, `${point.id}'s latency`);
    checkRange(point.accuracy, limits.accuracy, `${point.id}'s accuracy`);
  });
  if (new Set(points.map(point => point.id)).size !== points.length) {
    throw new RangeError('Every candidate needs its own name.');
  }
  checkRange(cap, limits.cap, 'the latency cap');
  const rows = points.map((point, index) => {
    const witness = points.find((other, position) => position !== index
      && other.latency <= point.latency && other.accuracy >= point.accuracy
      && (other.latency < point.latency || other.accuracy > point.accuracy));
    return {
      ...point, index,
      dominated: Boolean(witness),
      dominatedBy: witness ? witness.id : null,
      feasible: point.latency <= cap,
    };
  });
  const frontier = rows.filter(row => !row.dominated);
  const feasible = rows.filter(row => row.feasible);
  const selected = feasible.length === 0 ? null : feasible.reduce((best, row) => {
    if (row.accuracy > best.accuracy) return row;
    if (row.accuracy < best.accuracy) return best;
    if (row.latency < best.latency) return row;
    if (row.latency > best.latency) return best;
    return row.index < best.index ? row : best;
  });
  /** Feasible candidates that share the greatest accuracy. When more than one
   * does, the choice is settled by the declared tie rule and not by accuracy,
   * and the readout must say so: claiming the winner "has the greatest
   * accuracy" is false the moment a second candidate has the same. */
  const tiedOnAccuracy = selected
    ? feasible.filter(row => Math.abs(row.accuracy - selected.accuracy) <= limits.tolerance)
    : [];
  const tiedIds = tiedOnAccuracy.map(row => row.id);
  const brokenBy = tiedOnAccuracy.length < 2 ? null
    : (tiedOnAccuracy.every(row => Math.abs(row.latency - selected.latency) <= limits.tolerance)
      ? 'the earlier name' : 'the lower latency');
  return {
    cap, rows,
    frontierIds: frontier.map(row => row.id),
    feasibleIds: feasible.map(row => row.id),
    selectedId: selected ? selected.id : null,
    /** No feasible candidate is a real answer, not a nearest miss. */
    infeasible: feasible.length === 0,
    tiedOnAccuracyIds: tiedIds,
    selectionTied: tiedOnAccuracy.length > 1,
    tieBrokenBy: brokenBy,
    reason: !selected
      ? `no candidate meets a ${cap} ms cap, so the nearest miss is not a permitted answer`
      : (tiedOnAccuracy.length > 1
        ? `${tiedIds.join(' and ')} tie on accuracy at ${selected.accuracy} among the candidates at or under ${cap} ms, and the declared rule takes ${brokenBy}, which is ${selected.id}`
        : `${selected.id} has the greatest accuracy among the candidates at or under ${cap} ms`),
    selectionRule: 'greatest accuracy among feasible candidates; a tie takes the lower latency, then the earlier name',
  };
}

/* ================================================== §7 · I4 operation mixture */

export const declaredOperations = [
  { id: 'zero', label: 'zero', formula: 'o(x) = 0', evaluate: () => 0 },
  { id: 'identity', label: 'identity', formula: 'o(x) = x', evaluate: x => x },
  { id: 'negation', label: 'negation', formula: 'o(x) = −x', evaluate: x => -x },
];

/** The manuscript's worked mixture. Exported so the lesson body can quote it
 * without recomputing a logarithm in a module that shadows the global `Math`
 * with the page's math-rendering component. */
export const declaredMixtureInputs = {
  active: ['zero', 'identity', 'negation'],
  logits: [Math.log(2), 0, 0],
  x: 2, target: 1, step: 0.4,
};

/** Softmax with the maximum subtracted, so a large logit cannot overflow. */
export function softmaxWeights(logits) {
  if (!Array.isArray(logits) || logits.length < 2) throw new RangeError('Mix at least two operations.');
  logits.forEach((logit, index) => checkFinite(logit, `logit ${index + 1}`));
  const peak = Math.max(...logits);
  const weights = logits.map(logit => Math.exp(logit - peak));
  const total = sum(weights);
  return weights.map(weight => weight / total);
}

/** The mixed edge, its half-squared loss, and the exact architecture gradient
 * ∂L/∂α_i = (ō − t) p_i (o_i − ō). The logits are architecture variables; they
 * are not class probabilities. */
export function mixtureState({ operations = declaredOperations, active, logits, x, target, step = 0.4 }) {
  const chosen = (active ?? operations.map(operation => operation.id));
  const used = operations.filter(operation => chosen.includes(operation.id));
  if (used.length < 2) throw new RangeError('Keep at least two operations active.');
  checkRange(x, limits.input, 'the input');
  checkRange(target, limits.target, 'the target');
  checkRange(step, limits.stepSize, 'the step size');
  const usedLogits = used.map(operation => logits[operations.indexOf(operation)]);
  // The control bounds apply to entered logits, not the result of a gradient step.
  usedLogits.forEach((logit, index) => checkRange(logit, limits.logit, `logit ${index + 1}`));
  const probabilities = softmaxWeights(usedLogits);
  const outputs = used.map(operation => operation.evaluate(x));
  const mixed = sum(probabilities.map((probability, index) => probability * outputs[index]));
  const residual = mixed - target;
  const loss = 0.5 * residual * residual;
  const gradient = probabilities.map((probability, index) => residual * probability * (outputs[index] - mixed));
  const updatedLogits = usedLogits.map((logit, index) => logit - step * gradient[index]);
  const updatedProbabilities = softmaxWeights(updatedLogits);
  const updatedOutput = sum(updatedProbabilities.map((probability, index) => probability * outputs[index]));
  const zeroGradient = gradient.every(value => value === 0);
  return {
    operations: used.map((operation, index) => ({
      id: operation.id, label: operation.label, formula: operation.formula,
      logit: usedLogits[index], probability: probabilities[index], output: outputs[index],
      gradient: gradient[index], updatedLogit: updatedLogits[index],
      updatedProbability: updatedProbabilities[index],
    })),
    x, target, step, probabilities, outputs, mixed, residual, loss, gradient,
    updatedLogits, updatedOutput,
    updatedLoss: 0.5 * (updatedOutput - target) ** 2,
    movesTowardTarget: Math.abs(updatedOutput - target) < Math.abs(residual),
    /** What a step actually does, decided by whether the output moved — not by
     * enumerating the reasons it might not have.
     *
     * There are two independent ways for nothing to happen: the gradient
     * vanishes, or the step length is zero. A ternary that tests only the first
     * and then falls through to "away" grades a correct "nowhere at all" as
     * wrong, which is what it did. Asking the displayed quantity whether it
     * moved covers both, and any third reason nobody has thought of yet. */
    stepMoved: Math.abs(updatedOutput - mixed) > limits.tolerance,
    stepOutcome: Math.abs(updatedOutput - mixed) <= limits.tolerance ? 'stay'
      : (Math.abs(updatedOutput - target) < Math.abs(residual) ? 'toward' : 'away'),
    stepUnchangedReason: Math.abs(updatedOutput - mixed) > limits.tolerance ? null
      : (zeroGradient
        ? 'the architecture gradient is exactly zero, so there is no direction to move in'
        : step === 0 ? 'the step length is zero, so the logits are left exactly where they were'
          : 'the computed output change is within 10⁻¹²; the gradient and step may both be nonzero'),
    /** Every output identical leaves a zero gradient whatever the target is. */
    degenerate: outputs.every(value => value === outputs[0]),
    /** The gradient also vanishes when the mixture already sits on the target,
     * however different the operations are. Both cases mean "no logit is
     * favoured", and a learner who says so is right: `descendId` is then null
     * rather than an arbitrary argmin over a vector of zeros. */
    zeroGradient,
    zeroGradientReason: !zeroGradient ? null
      : (outputs.every(value => value === outputs[0])
        ? 'every active operation returns the same value at this input'
        : 'the mixture already sits exactly on the target, so the residual is zero'),
    descendId: zeroGradient ? null : used[gradient.indexOf(Math.min(...gradient))].id,
  };
}

/** Committing to one operation replaces the mixture with a different function.
 * Retraining is out of scope here, which is exactly why the gap is visible. */
export function commitOperation(state, id) {
  const chosen = state.operations.find(operation => operation.id === id);
  if (!chosen) throw new RangeError(`${id} is not one of the active operations.`);
  const loss = 0.5 * (chosen.output - state.target) ** 2;
  const best = state.operations.reduce((low, operation) => (
    0.5 * (operation.output - state.target) ** 2 < 0.5 * (low.output - state.target) ** 2 ? operation : low));
  return {
    id, output: chosen.output, loss,
    mixtureLoss: state.loss, difference: loss - state.loss,
    worseThanMixture: loss > state.loss + limits.tolerance,
    alternatives: state.operations.map(operation => ({
      id: operation.id, output: operation.output,
      loss: 0.5 * (operation.output - state.target) ** 2,
    })),
    bestDiscreteId: best.id,
    argmaxId: state.operations.reduce((high, operation) => (operation.probability > high.probability ? operation : high)).id,
  };
}

/** The exact mixed function over an input range, for the optional plot. It is a
 * constructed function, not a fitted decision boundary. */
export function mixtureCurve({ operations = declaredOperations, active, logits, target, domain = [-5, 5], samples = 81 }) {
  const points = [];
  for (let index = 0; index < samples; index += 1) {
    const x = domain[0] + ((domain[1] - domain[0]) * index) / (samples - 1);
    const state = mixtureState({ operations, active, logits, x, target, step: 0 });
    points.push({ x, mixed: state.mixed });
  }
  return { points, domain };
}

/* ====================================================== §7 · F4 bilevel lanes */

/** Three derivatives of three different functions of the same small problem.
 * L_train = ½(w − α)², L_val = ½(w − c)². */
export function bilevelDerivatives({ w, alpha, xi, valTarget = 1 }) {
  checkRange(w, { minimum: -5, maximum: 5 }, 'the current weight');
  checkRange(alpha, { minimum: -5, maximum: 5 }, 'the architecture variable');
  checkRange(xi, { minimum: 0, maximum: 1 }, 'the inner step size');
  checkFinite(valTarget, 'the validation target');
  const trainingGradient = w - alpha;
  const stepped = w - xi * trainingGradient;
  return {
    w, alpha, xi, valTarget, trainingGradient, stepped,
    lanes: [
      {
        id: 'direct', weightSymbol: 'w', weightValue: w,
        held: 'w is held fixed by convention, so α never enters the validation expression',
        // Short enough to sit between the two nodes it connects: the sentence
        // explaining the convention belongs in `held`, not in the drawing.
        dependency: 0, dependencyLabel: '∂w/∂α = 0',
        outer: 0,
        note: 'α does not appear in ½(w − ' + valTarget + ')² at fixed w',
      },
      {
        id: 'oneStep', weightSymbol: "w'", weightValue: stepped,
        held: 'the current w is held fixed inside the one-step formula',
        dependency: xi, dependencyLabel: "∂w'/∂α = ξ",
        outer: (stepped - valTarget) * xi,
        // Printed at the resolution the lane's own box and table use; the raw
        // binary value is 0.020000000000000004 and appeared nowhere else.
        note: `w' = w − ξ(w − α) = ${Number(stepped.toFixed(9))}`,
      },
      {
        id: 'exact', weightSymbol: 'w*', weightValue: alpha,
        held: 'the inner problem is solved exactly, so w* moves with α',
        dependency: 1, dependencyLabel: '∂w*/∂α = 1',
        outer: alpha - valTarget,
        note: 'w* = α minimizes ½(w − α)²',
      },
    ],
    /** At a stationary training point the current and one-step weights are
     * equal and the outer derivative is still not zero. */
    stationary: Math.abs(trainingGradient) <= limits.tolerance,
    allDistinct: new Set([0, (stepped - valTarget) * xi, alpha - valTarget].map(value => value.toFixed(12))).size === 3,
  };
}

/* ================================================= §7 · F5 weight provenance */

/** `steps` is the full wording, for the prose and the text equivalent.
 * `stepLabels` is what the drawing can actually fit: a box that truncates every
 * one of its labels to an ellipsis has stopped being a diagram. The model
 * verifier checks each short label against the box width it will be given. */
export const provenanceLanes = [
  {
    id: 'independent', label: 'Independently trained candidate',
    steps: ['fresh initialization', 'train this candidate', 'validation predictions'],
    stepLabels: ['fresh init', 'train it', 'validate'],
    stateReused: 'none', measures: 'a validation metric for weights fitted for this architecture alone',
  },
  {
    id: 'shared', label: 'Subgraph of a shared-weight supernetwork',
    steps: ['train supernetwork on sampled paths', 'select subgraph', 'validation predictions'],
    stepLabels: ['train supernet', 'pick subgraph', 'validate'],
    stateReused: 'weights trained through every path that shares them',
    measures: 'a validation metric under coupled weights, which can rank candidates differently',
  },
  {
    id: 'proxy', label: 'Untrained activation proxy',
    steps: ['random initialization', 'forward an actual input batch', 'activation codes', 'kernel score'],
    stepLabels: ['untrained', 'one batch', 'codes', 'score'],
    stateReused: 'no trained weights at all',
    measures: 'a score, not an accuracy',
  },
];

/** The box width each lane's steps are drawn in, matching the component. */
export function provenanceBoxWidth(stepCount, total = 336, gap = 14) {
  return (total - (stepCount - 1) * gap) / stepCount;
}

/** Hamming distance: the number of positions at which two codes differ. */
export function hammingDistance(left, right) {
  if (typeof left !== 'string' || typeof right !== 'string' || left.length !== right.length) {
    throw new RangeError('Compare two activation codes of the same length.');
  }
  if (!/^[01]+$/.test(left) || !/^[01]+$/.test(right)) {
    throw new RangeError('An activation code records one bit per unit.');
  }
  return [...left].filter((bit, index) => bit !== right[index]).length;
}

/** K_ij = N_A − d_H(c_i, c_j). Identical codes give a singular kernel, whose
 * log determinant has no finite real value; nothing is regularized here. */
export function activationKernel(codes) {
  if (!Array.isArray(codes) || codes.length !== 2) throw new RangeError('This inset compares exactly two codes.');
  const width = codes[0].length;
  const matrix = codes.map(row => codes.map(column => width - hammingDistance(row, column)));
  const determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  return {
    codes, width, matrix, determinant,
    distance: hammingDistance(codes[0], codes[1]),
    singular: determinant === 0,
    logDeterminant: determinant > 0 ? Math.log(determinant) : -Infinity,
    logDeterminantLabel: determinant > 0 ? null : 'no finite real value; the extended value is −∞',
  };
}
