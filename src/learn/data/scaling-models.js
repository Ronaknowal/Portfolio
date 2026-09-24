/** Pure models for the feature-preparation lesson.
 *
 * The organising idea of this topic is that a transformation is *fitted* on one
 * set of rows and then *applied*, frozen, to another. So every fitting function
 * here returns an explicit fitted object — a median list, a centre and a scale,
 * a category vocabulary — and every applying function takes one of those objects
 * plus a record. Nothing recomputes a statistic from the row it is transforming,
 * because that is exactly the mistake the lesson is about.
 *
 * Validation refuses bad input rather than substituting a plausible number. A
 * missing measurement is `null`, never 0; a zero divisor is an error, not a
 * silent 1; an empty donor set is reported as "cannot estimate", not as zero.
 */

const isMissing = value => value === null || value === undefined
  || (typeof value === 'number' && Number.isNaN(value));

const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${name} must be a finite number.`);
  return value;
};
const within = (value, range, name) => {
  finite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
};
const positive = (value, name) => {
  if (!(finite(value, name) > 0)) throw new RangeError(`${name} must be greater than zero.`);
  return value;
};
const observedNumbers = (values, name) => {
  if (!Array.isArray(values)) throw new RangeError(`${name} must be a list.`);
  const observed = values.filter(value => !isMissing(value));
  observed.forEach(value => finite(value, `a value in ${name}`));
  return observed;
};

export const limits = {
  measurement: { minimum: -1e6, maximum: 1e6 },
  divisor: { minimum: 1e-6, maximum: 1e6 },
  billLength: { minimum: 20, maximum: 80 },
  billDepth: { minimum: 8, maximum: 30 },
  flipper: { minimum: 150, maximum: 260 },
  mass: { minimum: 2000, maximum: 7000 },
  donorCell: { minimum: -1000, maximum: 1000 },
  neighbours: { minimum: 1, maximum: 3 },
  smoothing: { minimum: 0, maximum: 20 },
  lambda: { minimum: -2, maximum: 3 },
};

export const scalerKinds = ['standard', 'minmax', 'robust'];

/** sklearn's `_handle_zeros_in_scale`: a constant training column is divided by
 * one rather than by zero. Its training values become zero after centring; a
 * later value need not. */
const safeScale = scale => (Math.abs(scale) < 10 * Number.EPSILON ? { scale: 1, constant: true } : { scale, constant: false });

/** The linear percentile convention: index p/100 × (n − 1), interpolated. */
export function percentile(values, probability) {
  const sorted = observedNumbers(values, 'the column').slice().sort((a, b) => a - b);
  if (!sorted.length) throw new RangeError('A percentile needs at least one observed value.');
  within(probability, { minimum: 0, maximum: 100 }, 'the percentile');
  const position = (probability / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export const median = values => percentile(values, 50);

/** Fit one column's ruler. The returned object is the whole transformation:
 * nothing else about the training column is remembered or needed. */
export function fitScaler(kind, values) {
  const observed = observedNumbers(values, 'the training column');
  if (!observed.length) throw new RangeError('Fit a scaler on at least one observed training value.');
  if (kind === 'standard') {
    const center = observed.reduce((sum, value) => sum + value, 0) / observed.length;
    const variance = observed.reduce((sum, value) => sum + (value - center) ** 2, 0) / observed.length;
    const { scale, constant } = safeScale(Math.sqrt(variance));
    return { kind, center, scale, constant, n: observed.length, centerName: 'mean', scaleName: 'standard deviation' };
  }
  if (kind === 'minmax') {
    const low = Math.min(...observed);
    const high = Math.max(...observed);
    const { scale, constant } = safeScale(high - low);
    return { kind, center: low, scale, constant, low, high, n: observed.length, centerName: 'minimum', scaleName: 'range' };
  }
  if (kind === 'robust') {
    const center = percentile(observed, 50);
    const first = percentile(observed, 25);
    const third = percentile(observed, 75);
    const { scale, constant } = safeScale(third - first);
    return { kind, center, scale, constant, first, third, n: observed.length, centerName: 'median', scaleName: 'interquartile range' };
  }
  throw new RangeError('Use standard, minmax or robust.');
}

/** Apply a fitted ruler to any later value, including one outside the training
 * range. Nothing is clipped: the distance outside the range is information. */
export function applyScaler(fitted, value) {
  if (!fitted || !scalerKinds.includes(fitted.kind)) throw new RangeError('Apply a fitted scaler object.');
  finite(value, 'the value being transformed');
  return (value - fitted.center) / fitted.scale;
}

/** The whole fixture table of section 2, from one training column. */
export function scalerComparison(trainingValues, laterValues = []) {
  const fits = Object.fromEntries(scalerKinds.map(kind => [kind, fitScaler(kind, trainingValues)]));
  const describe = (value, training) => ({
    value,
    training,
    ...Object.fromEntries(scalerKinds.map(kind => [kind, applyScaler(fits[kind], value)])),
  });
  return {
    fits,
    rows: [
      ...observedNumbers(trainingValues, 'the training column').map(value => describe(value, true)),
      ...laterValues.map(value => describe(value, false)),
    ],
  };
}

/** Squared distance under per-feature divisors: sum of (difference/divisor)².
 * Returns the per-feature contributions, because the contributions are the
 * teaching point and the total is only their sum. */
export function scaledSquaredDistance(from, to, divisors) {
  if (!Array.isArray(from) || !Array.isArray(to) || from.length !== to.length) {
    throw new RangeError('Compare two points with the same number of coordinates.');
  }
  if (!Array.isArray(divisors) || divisors.length !== from.length) {
    throw new RangeError('Give one divisor per coordinate.');
  }
  const contributions = from.map((value, index) => {
    const difference = finite(to[index], 'a coordinate') - finite(value, 'a coordinate');
    const divisor = positive(divisors[index], 'a divisor');
    return { difference, divisor, contribution: (difference / divisor) ** 2 };
  });
  return { contributions, total: contributions.reduce((sum, part) => sum + part.contribution, 0) };
}

/** Which candidate a distance rule selects, with an explicit tie state. The
 * comparison uses unrounded totals and a declared tolerance; no classifier
 * tie-break is smuggled in. */
export function nearestCandidate(query, candidates, divisors, tolerance = 1e-12) {
  const scored = candidates.map(candidate => ({
    name: candidate.name,
    point: candidate.point,
    ...scaledSquaredDistance(query, candidate.point, divisors),
  }));
  if (scored.length !== 2) throw new RangeError('This comparison takes exactly two candidates.');
  const [first, second] = scored;
  const difference = first.total - second.total;
  return {
    scored,
    winner: Math.abs(difference) <= tolerance ? 'tie' : difference < 0 ? first.name : second.name,
    difference,
  };
}

/** L2 row normalization. A zero vector has no direction; it stays zero. */
export function l2Normalize(row) {
  const values = row.map(value => finite(value, 'a coordinate'));
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  return { norm, values: norm === 0 ? values.slice() : values.map(value => value / norm) };
}

/** Fit a category vocabulary. Order is the fitted order and is part of the
 * transformation: the coordinates mean nothing without it. */
export function fitOneHot(values, { fillMissing = null } = {}) {
  if (!Array.isArray(values)) throw new RangeError('Fit a vocabulary on a list of values.');
  const seen = new Set();
  values.forEach(value => {
    const resolved = isMissing(value) ? fillMissing : value;
    if (isMissing(resolved)) throw new RangeError('A missing category needs a declared fill value before fitting.');
    if (typeof resolved !== 'string') throw new RangeError('Categories must be strings.');
    seen.add(resolved);
  });
  if (!seen.size) throw new RangeError('Fit a vocabulary on at least one value.');
  return { categories: [...seen].sort(), fillMissing };
}

/** Apply a fitted vocabulary. Three states are kept apart: a known value, a
 * missing value that the fitted fill category absorbs, and an unknown value that
 * the declared policy represents as zeros across the block. */
export function encodeCategory(fitted, value, { handleUnknown = 'ignore', drop = null } = {}) {
  if (!fitted || !Array.isArray(fitted.categories)) throw new RangeError('Apply a fitted vocabulary object.');
  const resolved = isMissing(value) ? fitted.fillMissing : value;
  const state = isMissing(value)
    ? (fitted.categories.includes(fitted.fillMissing) ? 'missing' : 'unknown')
    : (fitted.categories.includes(value) ? 'known' : 'unknown');
  if (state === 'unknown' && handleUnknown !== 'ignore') {
    throw new RangeError(`The fitted vocabulary has no category "${value}".`);
  }
  const kept = drop === null ? fitted.categories : fitted.categories.filter(name => name !== drop);
  if (drop !== null && !fitted.categories.includes(drop)) throw new RangeError('Only a fitted category can be dropped.');
  return { state, resolved: state === 'unknown' ? null : resolved, names: kept, vector: kept.map(name => (name === resolved && state !== 'unknown' ? 1 : 0)) };
}

/** Every pairwise Euclidean distance in a one-hot block, so the chosen geometry
 * can be checked rather than imagined. */
export function categoryDistances(fitted, { drop = null, includeUnknown = true } = {}) {
  const points = fitted.categories.map(name => ({ name, vector: encodeCategory(fitted, name, { drop }).vector }));
  if (includeUnknown) points.push({ name: 'unknown input', vector: encodeCategory(fitted, ' absent', { drop }).vector });
  const pairs = [];
  for (let index = 0; index < points.length; index += 1) {
    for (let other = index + 1; other < points.length; other += 1) {
      const squared = points[index].vector.reduce((sum, value, position) => sum + (value - points[other].vector[position]) ** 2, 0);
      pairs.push({ from: points[index].name, to: points[other].name, squared, distance: Math.sqrt(squared) });
    }
  }
  return { points, pairs };
}

/** Fit one median per numeric column. Rows are the training rows; nothing else
 * is consulted. */
export function fitMedianImputer(rows) {
  if (!Array.isArray(rows) || !rows.length) throw new RangeError('Fit an imputer on at least one row.');
  const width = rows[0].length;
  return {
    medians: Array.from({ length: width }, (_, column) => {
      const observed = observedNumbers(rows.map(row => row[column]), `column ${column}`);
      if (!observed.length) throw new RangeError(`Column ${column} has no observed training value to take a median from.`);
      return percentile(observed, 50);
    }),
  };
}

export function applyMedianImputer(fitted, row) {
  if (!fitted || !Array.isArray(fitted.medians)) throw new RangeError('Apply a fitted imputer object.');
  if (row.length !== fitted.medians.length) throw new RangeError('The row does not match the fitted columns.');
  return row.map((value, column) => (isMissing(value)
    ? { value: fitted.medians[column], imputed: true, original: null }
    : { value: finite(value, 'a measurement'), imputed: false, original: value }));
}

/** The nan-aware squared distance scikit-learn's KNNImputer uses: squared
 * differences on jointly observed coordinates, multiplied by m/q. With no
 * overlap the distance is undefined — it is not zero. */
export function overlapDistance(query, donor, featureCount = query.length) {
  if (query.length !== donor.length) throw new RangeError('Compare rows with the same columns.');
  const shared = [];
  let sum = 0;
  query.forEach((value, index) => {
    if (isMissing(value) || isMissing(donor[index])) return;
    const difference = finite(donor[index], 'a donor cell') - finite(value, 'a query cell');
    shared.push({ index, difference, squared: difference * difference });
    sum += difference * difference;
  });
  const q = shared.length;
  return {
    shared, q, m: featureCount, raw: sum,
    factor: q === 0 ? null : featureCount / q,
    squared: q === 0 ? null : (featureCount / q) * sum,
    defined: q > 0,
  };
}

/** Nearest-neighbour imputation of one target column, with every donor's
 * eligibility shown. Ties are broken by source order, which is stated rather
 * than silent. The all-missing target column is reported, never filled. */
export function knnImpute({ donors, donorNames, query, target, neighbours = 2 }) {
  if (!Array.isArray(donors) || !donors.length) throw new RangeError('Give at least one donor row.');
  const featureCount = query.length;
  if (!Number.isInteger(target) || target < 0 || target >= featureCount) throw new RangeError('Choose a target column inside the table.');
  if (!isMissing(query[target])) throw new RangeError('The target cell of the query must be absent.');
  if (!Number.isInteger(neighbours) || neighbours < 1) throw new RangeError('Use a whole neighbour count of at least one.');
  const rows = donors.map((donor, index) => {
    const distance = overlapDistance(query, donor, featureCount);
    return {
      name: donorNames?.[index] ?? `D${index + 1}`,
      index, donor,
      ...distance,
      hasTarget: !isMissing(donor[target]),
      targetValue: isMissing(donor[target]) ? null : donor[target],
      eligible: distance.defined && !isMissing(donor[target]),
    };
  });
  const available = rows.filter(row => row.hasTarget);
  if (!available.length) {
    return { rows, selected: [], estimate: null, mode: 'no-target-column', neighbours };
  }
  const eligible = rows.filter(row => row.eligible)
    .sort((a, b) => (a.squared - b.squared) || (a.index - b.index));
  if (!eligible.length) {
    const mean = available.reduce((sum, row) => sum + row.targetValue, 0) / available.length;
    return { rows, selected: [], estimate: mean, mode: 'fallback-mean', neighbours };
  }
  const selected = eligible.slice(0, neighbours);
  return {
    rows, selected,
    estimate: selected.reduce((sum, row) => sum + row.targetValue, 0) / selected.length,
    mode: 'neighbours', neighbours,
  };
}

/** Fit the whole mixed-column preparation on training rows. The returned object
 * is everything the transform is allowed to know. */
export function fitPreparation({ numericRows, categoryValues, scaler = 'standard', fillMissing = 'not_recorded' }) {
  const imputer = fitMedianImputer(numericRows);
  const filled = numericRows.map(row => applyMedianImputer(imputer, row).map(cell => cell.value));
  const scalers = imputer.medians.map((_, column) => (scaler === 'raw'
    ? { kind: 'raw', center: 0, scale: 1, constant: false, centerName: 'nothing subtracted', scaleName: 'nothing divided' }
    : fitScaler(scaler, filled.map(row => row[column]))));
  return {
    scaler,
    imputer,
    scalers,
    vocabulary: fitOneHot(categoryValues, { fillMissing }),
    trainingRows: numericRows.length,
  };
}

/** Apply a fitted preparation to one record, keeping every intermediate step so
 * the trace can be displayed. The fitted object never changes here. */
export function transformRecord(fitted, record, { handleUnknown = 'ignore' } = {}) {
  if (!fitted?.imputer || !fitted?.vocabulary) throw new RangeError('Apply a fitted preparation object.');
  const imputed = applyMedianImputer(fitted.imputer, record.numeric);
  const numeric = imputed.map((cell, column) => {
    const ruler = fitted.scalers[column];
    const centred = cell.value - ruler.center;
    return {
      column,
      original: cell.original,
      wasMissing: cell.imputed,
      filled: cell.value,
      median: fitted.imputer.medians[column],
      center: ruler.center,
      scale: ruler.scale,
      centered: centred,
      output: ruler.kind === 'raw' ? cell.value : centred / ruler.scale,
    };
  });
  const category = encodeCategory(fitted.vocabulary, record.category, { handleUnknown });
  return {
    numeric,
    category,
    coordinates: [...numeric.map(cell => cell.output), ...category.vector],
  };
}

/** Smoothed target encoding of one category: the shrinkage toward a prior that
 * an unsmoothed single-row category would otherwise hand the model directly. */
export function smoothedEncoding(sum, count, prior, smoothing) {
  finite(sum, 'the target sum');
  if (!Number.isInteger(count) || count < 0) throw new RangeError('Use a whole donor count.');
  within(smoothing, limits.smoothing, 'the smoothing');
  finite(prior, 'the prior');
  const denominator = count + smoothing;
  if (denominator <= 0) return { numerator: prior, denominator: 1, value: prior, usedFallback: true };
  return { numerator: sum + smoothing * prior, denominator, value: (sum + smoothing * prior) / denominator, usedFallback: false };
}

/** Cross-fitted target encoding. For each internal fold, the sums, counts *and
 * the prior* come from the other folds only, which is what keeps a row's own
 * target out of its own encoded value even through the smoothing term. */
export function crossFitEncoding({ categories, target, folds, smoothing = 2 }) {
  const n = categories.length;
  if (target.length !== n || folds.length !== n) throw new RangeError('Categories, targets and folds must line up.');
  within(smoothing, limits.smoothing, 'the smoothing');
  target.forEach(value => {
    if (value !== 0 && value !== 1) throw new RangeError('This teaching example uses binary targets 0 or 1.');
  });
  const foldIds = [...new Set(folds)].sort((a, b) => a - b);
  if (foldIds.length < 2) throw new RangeError('Cross-fitting needs at least two non-empty internal folds.');
  foldIds.forEach(id => {
    if (folds.filter(fold => fold !== id).length === 0) throw new RangeError('Every fold needs donors outside it.');
  });
  const priors = {};
  const rows = new Array(n);
  foldIds.forEach(held => {
    const donors = [];
    folds.forEach((fold, index) => { if (fold !== held) donors.push(index); });
    const prior = donors.reduce((sum, index) => sum + target[index], 0) / donors.length;
    priors[held] = prior;
    folds.forEach((fold, row) => {
      if (fold !== held) return;
      const matching = donors.filter(index => categories[index] === categories[row]);
      const sum = matching.reduce((total, index) => total + target[index], 0);
      const encoded = smoothedEncoding(sum, matching.length, prior, smoothing);
      rows[row] = {
        row, fold: held, category: categories[row], target: target[row],
        donors, matching, sum, count: matching.length, prior,
        numerator: encoded.numerator, denominator: encoded.denominator,
        value: encoded.usedFallback ? prior : encoded.value,
        // With no smoothing and no same-category donor the manuscript's formula
        // has a zero denominator; the declared policy is the fold's own prior.
        usedPriorFallback: encoded.usedFallback,
      };
    });
  });
  return { rows, priors, foldIds, smoothing, encoded: rows.map(row => row.value) };
}

/** Signed feature hashing over a fixed constructed map, so a collision can be
 * seen rather than asserted. */
export function signedHash(counts, map, buckets) {
  if (!Number.isInteger(buckets) || buckets < 1) throw new RangeError('Use a positive bucket count.');
  const vector = new Array(buckets).fill(0);
  const detail = Object.entries(counts).map(([token, count]) => {
    const entry = map[token];
    if (!entry) throw new RangeError(`The constructed map has no entry for "${token}".`);
    finite(count, 'a count');
    vector[entry.bucket] += entry.sign * count;
    return { token, count, ...entry, contribution: entry.sign * count };
  });
  return { vector, detail };
}

/** Box–Cox, defined only for strictly positive input. */
export function boxCox(x, lambda) {
  positive(x, 'a Box–Cox input');
  within(lambda, limits.lambda, 'lambda');
  return lambda === 0 ? Math.log(x) : (x ** lambda - 1) / lambda;
}

/** Yeo–Johnson, with the branch it actually took. The negative side uses the
 * exponent 2 − λ, with a logarithm at λ = 2; it is not a square root. */
export function yeoJohnson(x, lambda) {
  finite(x, 'a Yeo–Johnson input');
  within(lambda, limits.lambda, 'lambda');
  if (x >= 0) {
    return lambda === 0
      ? { value: Math.log1p(x), branch: 'x ≥ 0, λ = 0: log(x + 1)' }
      : { value: ((x + 1) ** lambda - 1) / lambda, branch: 'x ≥ 0, λ ≠ 0: ((x + 1)^λ − 1)/λ' };
  }
  return lambda === 2
    ? { value: -Math.log1p(-x), branch: 'x < 0, λ = 2: −log(1 − x)' }
    : { value: -(((1 - x) ** (2 - lambda) - 1) / (2 - lambda)), branch: 'x < 0, λ ≠ 2: −((1 − x)^(2 − λ) − 1)/(2 − λ)' };
}

/** The explicit rank convention of section 7: (r − 1)/(n − 1) on the sorted
 * training values. This explains the idea; it does not reproduce every
 * interpolation, tie and endpoint rule of a fitted QuantileTransformer. */
export function rankCoordinates(values) {
  const observed = observedNumbers(values, 'the training column').slice().sort((a, b) => a - b);
  if (observed.length < 2) throw new RangeError('A rank coordinate needs at least two training values.');
  return observed.map((value, index) => ({
    value,
    rank: index + 1,
    coordinate: index / (observed.length - 1),
    log: value > 0 ? Math.log(value) : null,
  }));
}

/** Degrees to a point on the unit circle, so 359° and 1° can be near. */
export function circularCoordinates(degrees) {
  finite(degrees, 'an angle');
  const radians = (degrees * Math.PI) / 180;
  return { degrees, radians, sin: Math.sin(radians), cos: Math.cos(radians) };
}

/** Rubin's pooling rule for m completed-data analyses. */
export function poolEstimates(estimates, variances) {
  if (!Array.isArray(estimates) || estimates.length < 2) throw new RangeError('Pool at least two completed-data analyses.');
  if (!Array.isArray(variances) || variances.length !== estimates.length) throw new RangeError('Give one variance per estimate.');
  variances.forEach(value => { if (finite(value, 'a within-analysis variance') < 0) throw new RangeError('A variance cannot be negative.'); });
  const m = estimates.length;
  const mean = estimates.reduce((sum, value) => sum + finite(value, 'an estimate'), 0) / m;
  const within = variances.reduce((sum, value) => sum + value, 0) / m;
  const between = estimates.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (m - 1);
  const total = within + (1 + 1 / m) * between;
  return { m, mean, within, between, correction: (1 + 1 / m) * between, total, standardError: Math.sqrt(total) };
}

export { isMissing, within as checkRange, finite as checkFinite, positive as checkPositive };
