// Bounded, deterministic population models. None of these charts simulates data.
function bounded(value, low, high, name) {
  if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`${name} must lie in [${low}, ${high}].`);
  return value;
}
function integer(value, low, high, name) {
  bounded(value, low, high, name);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
function frozen(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(frozen);
    Object.freeze(value);
  }
  return value;
}
function denseArray(values, name) {
  if (!Array.isArray(values) || !values.length || values.length > 64) throw new RangeError(`${name} needs one to 64 entries.`);
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new RangeError(`${name} must not contain missing entries.`);
  }
}
export function finiteMoments(values, masses) {
  denseArray(values, 'Values');
  denseArray(masses, 'Probability masses');
  if (values.length !== masses.length) throw new RangeError('Use matching values and probability masses.');
  values.forEach(x => bounded(x, -1000, 1000, 'Value'));
  masses.forEach(p => {
    bounded(p, 0, 1, 'Mass');
    if (p > 0 && p < 1e-12) throw new RangeError('Positive mass must be at least 1e-12 in this finite teaching model.');
  });
  const total = masses.reduce((a, b) => a + b, 0);
  if (Math.abs(total - 1) > 1e-12) throw new RangeError('Probability masses must sum to one.');
  const probabilities = masses.map(p => p / total);
  // Anchor at an actual positive-mass value: constant laws then retain exactly
  // zero variance instead of accumulating roundoff in a repeated constant sum.
  const anchor = values[probabilities.findIndex(p => p > 0)];
  const mean = anchor + values.reduce((sum, x, i) => sum + probabilities[i] * (x - anchor), 0);
  const variance = values.reduce((sum, x, i) => {
    const residual = x - mean;
    const contribution = probabilities[i] * residual ** 2;
    if (probabilities[i] > 0 && residual !== 0 && contribution === 0) {
      throw new RangeError('The squared deviation is below this model’s representable arithmetic range.');
    }
    return sum + contribution;
  }, 0);
  if (variance > 0 && variance < 1e-280) {
    throw new RangeError('Positive variance must be at least 1e-280 in this teaching model.');
  }
  const second = values.reduce((sum, x, i) => sum + probabilities[i] * x * x, 0);
  return frozen({
    mean,
    variance,
    second,
    sd: Math.sqrt(variance),
    probabilities
  });
}
function law(values, masses) {
  const support = [...new Set(values)].sort((a, b) => a - b);
  let cumulative = 0;
  return support.map(value => {
    const mass = values.reduce((sum, x, i) => sum + (x === value ? masses[i] : 0), 0);
    cumulative += mass;
    return {
      value,
      mass,
      cumulative
    };
  });
}
export function outcomeState(firstPercent = 50, secondPercent = 50, mapping = 'heads') {
  integer(firstPercent, 0, 100, 'First head percentage');
  integer(secondPercent, 0, 100, 'Second head percentage');
  if (!['heads', 'first', 'equal'].includes(mapping)) throw new RangeError('Unknown outcome rule.');
  const p = firstPercent / 100,
    q = secondPercent / 100;
  const outcomes = [[0, 0], [0, 1], [1, 0], [1, 1]].map(([a, b]) => ({
    label: (a ? 'H' : 'T') + (b ? 'H' : 'T'),
    value: mapping === 'heads' ? a + b : mapping === 'first' ? a : Number(a === b),
    mass: (a ? p : 1 - p) * (b ? q : 1 - q)
  }));
  const values = outcomes.map(x => x.value),
    masses = outcomes.map(x => x.mass);
  return frozen({
    outcomes,
    distribution: law(values, masses),
    ...finiteMoments(values, masses)
  });
}
export function meanLossState(candidate = 0, preset = 'asymmetric') {
  bounded(candidate, -3, 4, 'Prediction');
  if (!['asymmetric', 'symmetric', 'constant'].includes(preset)) throw new RangeError('Unknown distribution.');
  const values = preset === 'asymmetric' ? [-2, 0, 3] : preset === 'symmetric' ? [-1, 1] : [2];
  const masses = preset === 'asymmetric' ? [.25, .5, .25] : preset === 'symmetric' ? [.5, .5] : [1];
  const moments = finiteMoments(values, masses);
  const pieces = values.map((value, i) => ({
    value,
    mass: masses[i],
    residual: value - candidate,
    contribution: masses[i] * (value - candidate) ** 2
  }));
  return frozen({
    ...moments,
    candidate,
    pieces,
    loss: pieces.reduce((sum, x) => sum + x.contribution, 0),
    excess: (candidate - moments.mean) ** 2
  });
}
export function pairedMoments(rows) {
  denseArray(rows, 'Paired rows');
  if (rows.some(row => !row || !Number.isFinite(row.x) || !Number.isFinite(row.y))) throw new RangeError('Paired rows need finite x and y.');
  const masses = rows.map(row => row.mass);
  const x = finiteMoments(rows.map(row => row.x), masses),
    y = finiteMoments(rows.map(row => row.y), masses);
  const covariance = rows.reduce((sum, row, i) => sum + x.probabilities[i] * (row.x - x.mean) * (row.y - y.mean), 0);
  const correlation = x.variance > 0 && y.variance > 0 ? covariance / (x.sd * y.sd) : null;
  return frozen({
    x,
    y,
    covariance,
    correlation
  });
}
export function jointState(preset = 'matching', scale = 1, shift = 0) {
  if (!['independent', 'matching', 'opposite', 'nonlinear'].includes(preset)) throw new RangeError('Unknown joint law.');
  integer(scale, -2, 2, 'Scale');
  integer(shift, -3, 3, 'Shift');
  const rows = preset === 'independent' ? [-1, 0, 1].flatMap(x => [-1, 0, 1].map(y => ({
    x,
    y,
    mass: 1 / 9
  }))) : [-1, 0, 1].map(x => ({
    x,
    y: preset === 'matching' ? x : preset === 'opposite' ? -x : x * x,
    mass: 1 / 3
  }));
  rows.forEach(row => {
    row.y = scale * row.y + shift;
  });
  const moments = pairedMoments(rows);
  const xs = [...new Set(rows.map(row => row.x))].sort((a, b) => a - b);
  const ys = [...new Set(rows.map(row => row.y))].sort((a, b) => a - b);
  const cells = xs.flatMap(x => ys.map(y => {
    const mass = rows.filter(row => row.x === x && row.y === y).reduce((sum, row) => sum + row.mass, 0);
    const px = rows.filter(row => row.x === x).reduce((sum, row) => sum + row.mass, 0);
    const py = rows.filter(row => row.y === y).reduce((sum, row) => sum + row.mass, 0);
    return {
      x,
      y,
      mass,
      px,
      py,
      product: px * py,
      contribution: mass * (x - moments.x.mean) * (y - moments.y.mean)
    };
  }));
  return frozen({
    ...moments,
    rows,
    xs,
    ys,
    cells,
    independent: cells.every(cell => Math.abs(cell.mass - cell.product) < 1e-12)
  });
}
export function sharedNoiseState(common = 2, local = 1, a = .5, b = .5) {
  bounded(common, 0, 3, 'Common amplitude');
  bounded(local, 0, 2, 'Local amplitude');
  bounded(a, -1, 1, 'A coefficient');
  bounded(b, -1, 1, 'B coefficient');
  const rows = [-1, 1].flatMap(s => [-1, 1].flatMap(e => [-1, 1].map(f => ({
    x: 10 + common * s + local * e,
    y: 20 + common * s + local * f,
    mass: 1 / 8,
    s,
    e,
    f
  }))));
  const moments = pairedMoments(rows);
  const values = rows.map(row => a * row.x + b * row.y);
  const combined = finiteMoments(values, rows.map(row => row.mass));
  return frozen({
    ...moments,
    rows,
    a,
    b,
    common,
    local,
    combined,
    distribution: law(values, rows.map(row => row.mass)),
    components: [((a + b) * common) ** 2, (a * local) ** 2, (b * local) ** 2]
  });
}
export function conditionalState(positivePercent = 50) {
  integer(positivePercent, 0, 100, 'Positive group percentage');
  const p = positivePercent / 100;
  const rows = [-2, 2].flatMap(g => [-1, 1].map(u => ({
    g,
    u,
    x: g + u,
    y: g - u,
    mass: (g === 2 ? p : 1 - p) / 2
  })));
  const moments = pairedMoments(rows);
  const groups = [-2, 2].map(g => {
    const selected = rows.filter(row => row.g === g),
      mass = selected.reduce((sum, row) => sum + row.mass, 0);
    return {
      g,
      mass,
      moments: mass > 0 ? pairedMoments(selected.map(row => ({
        ...row,
        mass: row.mass / mass
      }))) : null
    };
  });
  const withinVariance = groups.reduce((sum, group) => sum + (group.moments ? group.mass * group.moments.y.variance : 0), 0);
  const betweenVariance = groups.reduce((sum, group) => sum + (group.moments ? group.mass * (group.moments.y.mean - moments.y.mean) ** 2 : 0), 0);
  const withinCovariance = groups.reduce((sum, group) => sum + (group.moments ? group.mass * group.moments.covariance : 0), 0);
  const betweenCovariance = groups.reduce((sum, group) => sum + (group.moments ? group.mass * (group.moments.x.mean - moments.x.mean) * (group.moments.y.mean - moments.y.mean) : 0), 0);
  return frozen({
    ...moments,
    rows,
    groups,
    withinVariance,
    betweenVariance,
    withinCovariance,
    betweenCovariance
  });
}
export function squaredUniformState(lowerPercent = 25, upperPercent = 81) {
  integer(lowerPercent, 0, 100, 'Lower endpoint percentage');
  integer(upperPercent, 0, 100, 'Upper endpoint percentage');
  if (lowerPercent > upperPercent) throw new RangeError('Lower endpoint must not exceed upper endpoint.');
  const lower = lowerPercent / 100,
    upper = upperPercent / 100;
  const rootLower = Math.sqrt(lower),
    rootUpper = Math.sqrt(upper);
  const mass = rootUpper - rootLower;
  return frozen({
    lower,
    upper,
    rootLower,
    rootUpper,
    mass,
    preimages: [[-rootUpper, -rootLower], [rootLower, rootUpper]],
    mean: 1 / 3,
    variance: 4 / 45
  });
}
export function sampleMeanState(n = 8, percent = 50) {
  integer(n, 1, 16, 'Number of readings');
  integer(percent, 0, 100, 'Success percentage');
  const p = percent / 100;
  // Polynomial multiplication of ((1-p) + p*z)^n, not a sampled histogram.
  let masses = [1];
  for (let i = 0; i < n; i += 1) {
    const next = Array(masses.length + 1).fill(0);
    masses.forEach((mass, k) => {
      next[k] += mass * (1 - p);
      next[k + 1] += mass * p;
    });
    masses = next;
  }
  // Binomial tails can be smaller than finiteMoments' public mass contract.
  // Their range here is finite and representable (minimum .01^16 = 1e-32).
  const independent = masses.map((mass, k) => ({
    value: k / n,
    mass
  }));
  const copied = [{
    value: 0,
    mass: 1 - p
  }, {
    value: 1,
    mass: p
  }];
  const exactVariance = independent.reduce((sum, row) => sum + row.mass * (row.value - p) ** 2, 0);
  return frozen({
    n,
    p,
    independent,
    copied,
    mean: p,
    independentVariance: exactVariance,
    copiedVariance: p * (1 - p)
  });
}
