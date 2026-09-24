/** Pure models for the non-negative matrix factorization lesson.
 *
 * Observations in rows, components in H rows, activations in W columns:
 * X is n x d, W is n x k, H is k x d and the reconstruction is W H. Everything
 * the page states is computed here from the stated inputs; no figure
 * interpolates a fixture and no lab looks an answer up.
 *
 * Nonnegativity guarantees additive reconstruction and nothing else. A
 * component is not a physical part, a topic or a source until separate evidence
 * says so, and an activation is not a probability unless a stated normalization
 * gives it that meaning.
 *
 * Every entry point validates its input and throws a RangeError rather than
 * silently substituting a value: a lab that quietly repaired a bad matrix would
 * teach a reconstruction the learner never asked for.
 */

export const limits = {
  amount: { minimum: 0, maximum: 4, step: 0.25 },
  patternCell: { minimum: 0, maximum: 2, step: 0.25 },
  measurement: { minimum: 0, maximum: 6, step: 0.25 },
  initialFactor: { minimum: 0.1, maximum: 4, step: 0.1 },
  sweeps: { minimum: 0, maximum: 40 },
};

const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${name} must be a finite number.`);
  return value;
};
export const checkRange = (value, range, name) => {
  finite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
};
/** A value on the declared lattice, so an investigation grades an input it
 * actually supports rather than rounding one it does not. */
export const onLattice = (value, range, name) => {
  checkRange(value, range, name);
  const steps = value / range.step;
  if (Math.abs(steps - Math.round(steps)) > 1e-9) {
    throw new RangeError(`Use steps of ${range.step} for ${name}.`);
  }
  return value;
};

/** A rectangular, finite, nonnegative matrix. Returns a defensive copy. */
export function checkMatrix(matrix, name = 'a matrix', { nonnegative = true, positive = false } = {}) {
  if (!Array.isArray(matrix) || matrix.length === 0) throw new RangeError(`${name} needs at least one row.`);
  const width = Array.isArray(matrix[0]) ? matrix[0].length : -1;
  if (width <= 0) throw new RangeError(`${name} needs at least one column.`);
  return matrix.map((row, index) => {
    if (!Array.isArray(row) || row.length !== width) throw new RangeError(`${name} must be rectangular; row ${index + 1} differs.`);
    return row.map((value, column) => {
      finite(value, `${name} entry (${index + 1}, ${column + 1})`);
      if (positive && !(value > 0)) throw new RangeError(`${name} entry (${index + 1}, ${column + 1}) must be positive.`);
      if (nonnegative && value < 0) throw new RangeError(`${name} entry (${index + 1}, ${column + 1}) cannot be negative.`);
      return value;
    });
  });
}
export const shape = matrix => [matrix.length, matrix[0].length];
export const transpose = matrix => matrix[0].map((_, column) => matrix.map(row => row[column]));

/** Ordinary matrix multiplication, with the shared dimension checked. */
export function multiply(left, right, names = ['the left matrix', 'the right matrix']) {
  const a = checkMatrix(left, names[0], { nonnegative: false });
  const b = checkMatrix(right, names[1], { nonnegative: false });
  if (a[0].length !== b.length) {
    throw new RangeError(`Shapes do not meet: ${shape(a).join('x')} times ${shape(b).join('x')}.`);
  }
  return a.map(row => b[0].map((_, column) => row.reduce((sum, value, index) => sum + value * b[index][column], 0)));
}

/** The reconstruction, plus the per-component contributions that build it. */
export function reconstruct(W, H) {
  const activations = checkMatrix(W, 'the activation matrix W');
  const patterns = checkMatrix(H, 'the pattern matrix H');
  if (activations[0].length !== patterns.length) {
    throw new RangeError(`W has ${activations[0].length} components but H has ${patterns.length} rows.`);
  }
  return multiply(activations, patterns);
}

/** One reconstructed cell, term by term: the summands the figure highlights. */
export function cellTerms(W, H, row, column) {
  const activations = checkMatrix(W, 'the activation matrix W');
  const patterns = checkMatrix(H, 'the pattern matrix H');
  if (!Number.isInteger(row) || row < 0 || row >= activations.length) throw new RangeError('Choose an observation row that exists.');
  if (!Number.isInteger(column) || column < 0 || column >= patterns[0].length) throw new RangeError('Choose a feature column that exists.');
  const terms = activations[row].map((amount, component) => ({
    component, amount, patternCell: patterns[component][column], product: amount * patterns[component][column],
  }));
  return { terms, total: terms.reduce((sum, term) => sum + term.product, 0) };
}

/** Each component's weighted contribution row for one observation:
 * contribution[r] = W[i, r] * H[r, :]. They add to the reconstruction. */
export function contributionRows(W, H, row) {
  const activations = checkMatrix(W, 'the activation matrix W');
  const patterns = checkMatrix(H, 'the pattern matrix H');
  if (!Number.isInteger(row) || row < 0 || row >= activations.length) throw new RangeError('Choose an observation row that exists.');
  if (activations[0].length !== patterns.length) throw new RangeError('W and H disagree about the component count.');
  return activations[row].map((amount, component) => patterns[component].map(value => amount * value));
}

/** Half the squared Frobenius norm of the residual: the objective the lesson
 * minimises. The one-half cancels a factor of two when differentiating. */
export function halfSquaredLoss(X, W, H) {
  const data = checkMatrix(X, 'the data matrix X');
  const fitted = reconstruct(W, H);
  if (data.length !== fitted.length || data[0].length !== fitted[0].length) {
    throw new RangeError(`The reconstruction is ${shape(fitted).join('x')} but X is ${shape(data).join('x')}.`);
  }
  return data.reduce((sum, row, index) =>
    sum + row.reduce((rowSum, value, column) => rowSum + (value - fitted[index][column]) ** 2, 0), 0) / 2;
}

/** The signed residual: positive means missing reconstructed mass. */
export function residual(X, fitted) {
  const data = checkMatrix(X, 'the data matrix X', { nonnegative: false });
  const approximation = checkMatrix(fitted, 'the reconstruction', { nonnegative: false });
  if (data.length !== approximation.length || data[0].length !== approximation[0].length) {
    throw new RangeError('The residual needs two matrices of the same shape.');
  }
  return data.map((row, index) => row.map((value, column) => value - approximation[index][column]));
}

/** Mean squared residual over every entry, in the units of X squared. */
export function meanSquaredError(observed, fitted) {
  const signed = residual([observed], [fitted])[0];
  return signed.reduce((sum, value) => sum + value * value, 0) / signed.length;
}

// --------------------------------------------------------------- the update
/** The gradient of the half squared loss with respect to H:
 * (W^T W) H  -  W^T X. Its two terms are the update's denominator and
 * numerator, which is why the ratio moves the objective downhill. */
export function gradientH(X, W, H) {
  const numerator = multiply(transpose(checkMatrix(W, 'W')), checkMatrix(X, 'X'));
  const denominator = multiply(multiply(transpose(W), W), checkMatrix(H, 'H'));
  return denominator.map((row, index) => row.map((value, column) => value - numerator[index][column]));
}
export function gradientW(X, W, H) {
  const numerator = multiply(checkMatrix(X, 'X'), transpose(checkMatrix(H, 'H')));
  const denominator = multiply(checkMatrix(W, 'W'), multiply(H, transpose(H)));
  return denominator.map((row, index) => row.map((value, column) => value - numerator[index][column]));
}

/** One multiplicative H phase, with every cell's numerator and denominator kept
 * so the worksheet can show the ratio that moved it. A zero denominator is
 * refused: the displayed rule has no epsilon and 0/0 is not evidence. */
export function updateH(X, W, H) {
  const data = checkMatrix(X, 'the data matrix X');
  const activations = checkMatrix(W, 'the activation matrix W');
  const patterns = checkMatrix(H, 'the pattern matrix H');
  const numerator = multiply(transpose(activations), data);
  const denominator = multiply(multiply(transpose(activations), activations), patterns);
  const next = patterns.map((row, component) => row.map((value, column) => {
    if (!(denominator[component][column] > 0)) {
      throw new RangeError('This update needs a positive denominator; the displayed rule adds no epsilon.');
    }
    return value * numerator[component][column] / denominator[component][column];
  }));
  return { numerator, denominator, next, ratio: numerator.map((row, r) => row.map((value, c) => value / denominator[r][c])) };
}

/** One multiplicative W phase, using the H the previous phase produced. */
export function updateW(X, W, H) {
  const data = checkMatrix(X, 'the data matrix X');
  const activations = checkMatrix(W, 'the activation matrix W');
  const patterns = checkMatrix(H, 'the pattern matrix H');
  const numerator = multiply(data, transpose(patterns));
  const denominator = multiply(activations, multiply(patterns, transpose(patterns)));
  const next = activations.map((row, observation) => row.map((value, component) => {
    if (!(denominator[observation][component] > 0)) {
      throw new RangeError('This update needs a positive denominator; the displayed rule adds no epsilon.');
    }
    return value * numerator[observation][component] / denominator[observation][component];
  }));
  return { numerator, denominator, next, ratio: numerator.map((row, r) => row.map((value, c) => value / denominator[r][c])) };
}

/** A complete alternating sweep: H first with the old W, then W with the new H. */
export function sweep(X, W, H) {
  const hPhase = updateH(X, W, H);
  const wPhase = updateW(X, W, hPhase.next);
  return {
    hPhase, wPhase, H: hPhase.next, W: wPhase.next,
    lossBefore: halfSquaredLoss(X, W, H),
    lossAfterH: halfSquaredLoss(X, W, hPhase.next),
    loss: halfSquaredLoss(X, wPhase.next, hPhase.next),
  };
}

/** Repeat sweeps, keeping iteration 0 as the starting state so the trace reads
 * as a history. The objective cannot increase along it. */
export function sweepTrace(X, W, H, count = 40) {
  checkRange(count, { minimum: 0, maximum: limits.sweeps.maximum }, 'the sweep count');
  const history = [{ iteration: 0, W: checkMatrix(W, 'W'), H: checkMatrix(H, 'H'), loss: halfSquaredLoss(X, W, H) }];
  let current = { W, H };
  for (let iteration = 1; iteration <= count; iteration += 1) {
    const step = sweep(X, current.W, current.H);
    current = { W: step.W, H: step.H };
    history.push({ iteration, W: step.W, H: step.H, loss: step.loss });
  }
  return history;
}

/** Whether one selected H cell will grow, shrink or stay put on the next H
 * phase, computed from the committed inputs rather than the rendered state. */
export function hCellDirection(X, W, H, component, column, tolerance = 1e-12) {
  const { numerator, denominator } = updateH(X, W, H);
  const ratio = numerator[component][column] / denominator[component][column];
  const current = H[component][column];
  if (current === 0) return 'unchanged';
  if (Math.abs(ratio - 1) <= tolerance) return 'unchanged';
  return ratio > 1 ? 'grows' : 'shrinks';
}

/** The nonnegative first-order conditions. A positive variable needs a zero
 * gradient; a zero variable must not have a negative gradient pointing into a
 * feasible decrease. A zero gradient everywhere is not the boundary test. */
export function stationarityReport(X, W, H) {
  const gradient = gradientH(X, W, H);
  const violations = [];
  gradient.forEach((row, component) => row.forEach((value, column) => {
    const entry = H[component][column];
    if (entry === 0 && value < -1e-12) violations.push({ component, column, entry, gradient: value, kind: 'zero with a feasible descent direction' });
    if (entry > 0 && Math.abs(value) > 1e-9) violations.push({ component, column, entry, gradient: value, kind: 'positive with a nonzero gradient' });
  }));
  return { gradient, violations, stationary: violations.length === 0 };
}

// -------------------------------------------------------------- normalization
/** Divide each pattern by its own total and multiply the matching activation
 * column by the same number. The product, and therefore every reconstruction,
 * is unchanged. A zero pattern contributes nothing and is reported, not divided
 * by zero. */
export function normalizePatterns(W, H) {
  const activations = checkMatrix(W, 'the activation matrix W');
  const patterns = checkMatrix(H, 'the pattern matrix H');
  if (activations[0].length !== patterns.length) throw new RangeError('W and H disagree about the component count.');
  const sums = patterns.map(row => row.reduce((total, value) => total + value, 0));
  const zeroPatterns = sums.map((sum, index) => (sum > 0 ? -1 : index)).filter(index => index >= 0);
  return {
    sums, zeroPatterns,
    H: patterns.map((row, index) => (sums[index] > 0 ? row.map(value => value / sums[index]) : row.slice())),
    W: activations.map(row => row.map((value, index) => value * sums[index])),
  };
}

/** One observation's normalized mixture: the activations after normalization,
 * their total reconstructed mass, and the proportions that total implies. */
export function mixtureProportions(W, H, row) {
  const normalized = normalizePatterns(W, H);
  const amounts = normalized.W[row];
  const total = amounts.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) throw new RangeError('An all-zero reconstruction has no mixture proportions.');
  return { amounts, total, proportions: amounts.map(value => value / total), patterns: normalized.H, sums: normalized.sums };
}

// ---------------------------------------------------------------- the losses
export const frobeniusHalf = (x, y) => 0.5 * (finite(x, 'the observation') - finite(y, 'the reconstruction')) ** 2;
/** Generalized Kullback-Leibler, with the convention 0 log(0/y) = 0. A positive
 * observation against a zero reconstruction is infinite. */
export function generalizedKL(x, y) {
  if (finite(x, 'the observation') < 0 || finite(y, 'the reconstruction') < 0) throw new RangeError('Both entries must be nonnegative.');
  if (x === 0) return y;
  if (y === 0) return Infinity;
  return x * Math.log(x / y) - x + y;
}
/** Itakura-Saito, defined here for strictly positive entries. */
export function itakuraSaito(x, y) {
  if (!(finite(x, 'the observation') > 0) || !(finite(y, 'the reconstruction') > 0)) {
    throw new RangeError('Itakura-Saito needs strictly positive entries.');
  }
  return x / y - Math.log(x / y) - 1;
}
export const lossesAt = (x, y) => ({ frobeniusHalf: frobeniusHalf(x, y), kl: generalizedKL(x, y), itakuraSaito: itakuraSaito(x, y) });
/** How each loss responds when observation and reconstruction are both scaled
 * by c: squared error by c squared, KL by c, Itakura-Saito not at all. */
export function scaleLosses(x, y, c) {
  if (!(finite(c, 'the gain') > 0)) throw new RangeError('Use a positive common gain.');
  const base = lossesAt(x, y);
  const scaled = lossesAt(c * x, c * y);
  return { base, scaled, factors: { frobeniusHalf: c * c, kl: c, itakuraSaito: 1 } };
}

// --------------------------------------------------------------- the geometry
/** Is a point a nonnegative combination of two rays in the plane? The figure's
 * containment claim is arithmetic, not a drawn impression. */
export function coneCoordinates(rays, point) {
  const [[a, c], [b, d]] = [[rays[0][0], rays[0][1]], [rays[1][0], rays[1][1]]];
  const determinant = a * d - b * c;
  if (Math.abs(determinant) < 1e-12) throw new RangeError('These two rays are parallel, so they span no two-dimensional cone.');
  const [x, y] = point;
  const first = (x * d - y * b) / determinant;
  const second = (y * a - x * c) / determinant;
  return { coefficients: [first, second], inside: first >= -1e-12 && second >= -1e-12 };
}

/** Exact integer rank by fraction-free Bareiss elimination: the nonnegative-rank
 * contrast must not depend on a floating-point tolerance. */
export function integerRank(matrix) {
  const rows = checkMatrix(matrix, 'the matrix', { nonnegative: false }).map(row => row.map(value => {
    if (!Number.isInteger(value)) throw new RangeError('This exact rank needs integer entries.');
    return value;
  }));
  const height = rows.length;
  const width = rows[0].length;
  let rank = 0;
  let previous = 1;
  for (let column = 0; column < width && rank < height; column += 1) {
    let pivot = -1;
    for (let row = rank; row < height; row += 1) if (rows[row][column] !== 0) { pivot = row; break; }
    if (pivot < 0) continue;
    [rows[rank], rows[pivot]] = [rows[pivot], rows[rank]];
    for (let row = rank + 1; row < height; row += 1) {
      for (let position = column + 1; position < width; position += 1) {
        rows[row][position] = (rows[row][position] * rows[rank][column] - rows[rank][position] * rows[row][column]) / previous;
      }
      rows[row][column] = 0;
    }
    previous = rows[rank][column];
    rank += 1;
  }
  return rank;
}

/** The exact determinant of a small square integer matrix, by cofactor
 * expansion. Used to show a nonsingular 3x3 minor beside the rank claim, so the
 * ordinary rank is demonstrated rather than asserted. */
export function integerDeterminant(matrix) {
  const rows = checkMatrix(matrix, 'the matrix', { nonnegative: false }).map(row => row.map(value => {
    if (!Number.isInteger(value)) throw new RangeError('This exact determinant needs integer entries.');
    return value;
  }));
  if (rows.length !== rows[0].length) throw new RangeError('A determinant needs a square matrix.');
  if (rows.length > 6) throw new RangeError('This exact routine is for small matrices only.');
  if (rows.length === 1) return rows[0][0];
  return rows[0].reduce((total, value, column) => {
    if (value === 0) return total;
    const minor = rows.slice(1).map(row => row.filter((_, index) => index !== column));
    return total + (column % 2 === 0 ? 1 : -1) * value * integerDeterminant(minor);
  }, 0);
}

/** The first square submatrix of the given size, taken from the first rows, that
 * is nonsingular. Returns its row and column indices and its determinant. */
export function nonsingularMinor(matrix, size) {
  const rows = checkMatrix(matrix, 'the matrix', { nonnegative: false });
  const combinations = (total, take, start = 0) => {
    if (take === 0) return [[]];
    const out = [];
    for (let index = start; index <= total - take; index += 1) {
      for (const rest of combinations(total, take - 1, index + 1)) out.push([index, ...rest]);
    }
    return out;
  };
  for (const rowSet of combinations(rows.length, size)) {
    for (const columnSet of combinations(rows[0].length, size)) {
      const minor = rowSet.map(row => columnSet.map(column => rows[row][column]));
      const determinant = integerDeterminant(minor);
      if (determinant !== 0) return { rows: rowSet, columns: columnSet, minor, determinant };
    }
  }
  throw new RangeError(`No nonsingular ${size} by ${size} submatrix exists.`);
}

/** For two positive cells of a binary matrix, the rectangle spanning them and
 * whichever of its crossed corners is zero. A nonnegative rank-one contribution
 * has rectangular positive support, so a crossed zero forbids covering both. */
export function crossedZeros(matrix, first, second) {
  const rows = checkMatrix(matrix, 'the support matrix');
  const [r1, c1] = first;
  const [r2, c2] = second;
  for (const [row, column] of [first, second]) {
    if (!(rows[row][column] > 0)) throw new RangeError('Both selected cells must be positive.');
  }
  return [[r1, c2], [r2, c1]].filter(([row, column]) => rows[row][column] === 0);
}

/** Every unordered pair of marked cells with its crossed zeros. */
export function supportPairs(matrix, marked) {
  const pairs = [];
  for (let a = 0; a < marked.length; a += 1) {
    for (let b = a + 1; b < marked.length; b += 1) {
      pairs.push({ a, b, cells: [marked[a], marked[b]], crossed: crossedZeros(matrix, marked[a], marked[b]) });
    }
  }
  return pairs;
}

/** Rows normalized to sum to one, and whether each lies in the convex hull of
 * the chosen anchors. Separability is an extra assumption, not a free result. */
export function simplexPosition(rows, anchors) {
  const normalized = checkMatrix(rows, 'the observations').map(row => {
    const total = row.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) throw new RangeError('A zero row has no normalized position.');
    return row.map(value => value / total);
  });
  const anchorRows = anchors.map(index => normalized[index]);
  return normalized.map(row => {
    const weights = coneCoordinates([[anchorRows[0][0], anchorRows[0][1]], [anchorRows[1][0], anchorRows[1][1]]], row);
    return { row, weights: weights.coefficients, inside: weights.inside };
  });
}

// ------------------------------------------------------ the real-image lab
/** A contribution mask over a fitted row: what is reconstructed, what the
 * residual becomes, and how the squared error moves pixel by pixel. The fitted
 * dictionary and activations are held exactly as they came out of the fit. */
export function maskedReconstruction(activations, dictionary, mask) {
  const coefficients = activations.map(value => finite(value, 'an activation'));
  const patterns = checkMatrix(dictionary, 'the dictionary');
  if (coefficients.length !== patterns.length) throw new RangeError('One activation per dictionary row.');
  if (!Array.isArray(mask) || mask.length !== coefficients.length) throw new RangeError('The mask needs one flag per component.');
  return patterns[0].map((_, feature) => coefficients.reduce(
    (sum, amount, component) => sum + (mask[component] ? amount * patterns[component][feature] : 0), 0));
}

/** Everything the held-out image investigation reports for one mask. */
export function maskedReport(observed, activations, dictionary, mask) {
  const full = maskedReconstruction(activations, dictionary, activations.map(() => true));
  const kept = maskedReconstruction(activations, dictionary, mask);
  const fullResidual = observed.map((value, index) => value - full[index]);
  const keptResidual = observed.map((value, index) => value - kept[index]);
  const squaredChange = observed.map((value, index) => (value - kept[index]) ** 2 - (value - full[index]) ** 2);
  return {
    full, kept, fullResidual, keptResidual, squaredChange,
    fullMse: meanSquaredError(observed, full),
    maskedMse: meanSquaredError(observed, kept),
    improvedPixels: squaredChange.filter(value => value < -1e-15).length,
    worsenedPixels: squaredChange.filter(value => value > 1e-15).length,
    unchangedPixels: squaredChange.filter(value => Math.abs(value) <= 1e-15).length,
    contributionTotals: activations.map((amount, component) => amount * dictionary[component].reduce((sum, value) => sum + value, 0)),
  };
}

/** Two contribution masks on the same fitted row, compared pixel by pixel.
 * The total and the individual pixels answer different questions: a pixel the
 * removed component was overpredicting can improve while the total gets worse. */
export function maskComparison(observed, activations, dictionary, oldMask, newMask, tolerance = 1e-8) {
  const before = maskedReconstruction(activations, dictionary, oldMask);
  const after = maskedReconstruction(activations, dictionary, newMask);
  const squaredChange = observed.map((value, index) => (value - after[index]) ** 2 - (value - before[index]) ** 2);
  const beforeMse = meanSquaredError(observed, before);
  const afterMse = meanSquaredError(observed, after);
  const difference = afterMse - beforeMse;
  return {
    before, after, squaredChange,
    beforeResidual: observed.map((value, index) => value - before[index]),
    afterResidual: observed.map((value, index) => value - after[index]),
    beforeMse, afterMse, difference,
    direction: Math.abs(difference) <= tolerance ? 'unchanged' : difference > 0 ? 'rises' : 'falls',
    improvedPixels: squaredChange.filter(value => value < -1e-15).length,
    worsenedPixels: squaredChange.filter(value => value > 1e-15).length,
    unchangedPixels: squaredChange.filter(value => Math.abs(value) <= 1e-15).length,
    pixelDirection: index => {
      if (!Number.isInteger(index) || index < 0 || index >= observed.length) throw new RangeError('Choose a pixel that exists.');
      const change = squaredChange[index];
      return Math.abs(change) <= 1e-15 ? 'unchanged' : change > 0 ? 'rises' : 'falls';
    },
  };
}

/** The direction the total row error moves when a mask changes, graded from the
 * committed masks. At a row NNLS optimum, removing a nonzero coefficient with
 * the others fixed cannot reduce the total squared error; the tolerance is the
 * solver's accuracy, not a licence to round a real difference away. */
export function maskDirection(observed, activations, dictionary, oldMask, newMask, tolerance = 1e-8) {
  const comparison = maskComparison(observed, activations, dictionary, oldMask, newMask, tolerance);
  return { before: comparison.beforeMse, after: comparison.afterMse, difference: comparison.difference, direction: comparison.direction };
}

/** Grey levels for an image-shaped row, on a stated scale. Nothing is clipped:
 * a reconstruction above the scale maximum would be a visible defect, not a
 * value to hide. */
export function greyLevels(values, maximum) {
  if (!(finite(maximum, 'the display maximum') > 0)) throw new RangeError('Use a positive display maximum.');
  return values.map(value => {
    finite(value, 'an image value');
    return { value, level: value / maximum, above: value > maximum };
  });
}

/** A symmetric scale for a signed residual image, centred on zero. */
export function divergingScale(values) {
  const extent = Math.max(...values.map(value => Math.abs(finite(value, 'a residual'))), 0);
  return { extent, level: value => (extent === 0 ? 0 : value / extent) };
}

// ----------------------------------------------------------------- fixtures
export const fixtures = {
  X: [[2, 1, 3], [1, 2, 3], [3, 3, 6]],
  W1: [[2, 1], [1, 2], [3, 3]],
  H1: [[1, 0, 1], [0, 1, 1]],
  W2: [[1.5, 0.5], [0.5, 1.5], [2, 2]],
  H2: [[1.25, 0.25, 1.5], [0.25, 1.25, 1.5]],
  startW: [[1, 0.5], [0.5, 1], [1, 1]],
  startH: [[1, 0.2, 0.8], [0.2, 1, 0.8]],
  observed: [2, 1, 3],
  approximated: [1.5, 1, 2.5],
  support: [[0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]],
  supportMarks: [[0, 2], [1, 3], [2, 0], [3, 1]],
  supportNames: ['A', 'B', 'C', 'D'],
  /** Fixed W, X and H where the first H entry is locked at zero by any rule that
   * leaves zeros alone, even though its gradient is negative there. */
  zeroLock: { W: [[1]], X: [[2, 1]], H: [[0, 1]], nnls: [2, 1] },
  anchors: { rows: [[1, 0], [0, 1], [0.25, 0.75], [0.6, 0.4]], anchorIndices: [0, 1] },
  words: { vocabulary: ['orbit', 'rocket', 'goal', 'team'], H: [[3, 2, 0, 0], [0, 0, 1, 4]], w: [2, 1] },
  spectrum: { patterns: [[0.2, 0.6, 0.4], [0.8, 0.3, 0.1]], amounts: [0.3, 0.7] },
  practice: {
    one: { H: [[2, 0, 1], [0, 1, 2]], w: [1, 3] },
    two: { W: [[1], [2]], X: [[2, 1], [4, 3]], H: [[1, 1]] },
    six: { target: 2, endpoints: [[1, 2], [4, 0.5]] },
    eight: { W: [[1]], X: [[3, 1]], H: [[0, 2]] },
  },
};

/** The scalar bilinear counterexample: two zero-loss points whose midpoint has
 * positive loss, which directly refutes joint convexity. */
export function bilinearMidpoint(target, first, second) {
  const loss = ([w, h]) => 0.5 * (target - w * h) ** 2;
  const midpoint = [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2];
  return {
    first: { point: first, loss: loss(first) },
    second: { point: second, loss: loss(second) },
    midpoint: { point: midpoint, product: midpoint[0] * midpoint[1], loss: loss(midpoint) },
    average: (loss(first) + loss(second)) / 2,
    convexityRefuted: loss(midpoint) > (loss(first) + loss(second)) / 2 + 1e-15,
  };
}

/** Operation counts per alternating sweep, stated as counts and never as time. */
export function sweepCost(n, d, k, storedNonzeros = null) {
  for (const [value, name] of [[n, 'the observation count'], [d, 'the feature count'], [k, 'the component count']]) {
    if (!Number.isInteger(value) || value < 1) throw new RangeError(`${name} must be a positive whole number.`);
  }
  const dense = n * d * k + (n + d) * k * k;
  const sparse = storedNonzeros === null ? null : storedNonzeros * k + (n + d) * k * k;
  return { dense, sparse, factorStorage: k * (n + d), note: 'Operation and storage counts, not measured seconds.' };
}
