/** Pure teaching models for the PCA lesson. Two-dimensional fits use the exact
 * symmetric 2×2 eigenproblem; Wine calculations reuse natively fitted means,
 * scales and directions from pca-wine-data.js and recompute scores,
 * reconstructions and losses in the browser so they can be checked against the
 * recorded native values. Sample variances divide by n − 1; StandardScaler
 * scales divide by n. These are bounded fixtures, not a numerical library.
 */
import { wineRows, wineFullFit, wineSplit } from './pca-wine-data.js';

export const fourPoints = Object.freeze([[1, 1], [2, 0], [4, 4], [5, 3]].map(point => Object.freeze(point)));
export const pointLimit = 12;
/** The projection workbench editor accepts coordinates within ±10; the label
 * investigation's widest rectangle reaches ±12, so the shared guard is ±20. */
export const editorBound = 10;
export const coordinateBound = 20;

function checkPoints(points, minimum = 1) {
  if (!Array.isArray(points) || points.length < minimum || points.length > pointLimit) throw new RangeError(`Use ${minimum} to ${pointLimit} points.`);
  for (const point of points) {
    if (!Array.isArray(point) || point.length !== 2 || point.some(value => !Number.isFinite(value) || Math.abs(value) > coordinateBound)) {
      throw new RangeError(`Each point needs two finite coordinates within ±${coordinateBound}.`);
    }
  }
}
export function meanOf(points) {
  checkPoints(points);
  return [0, 1].map(axis => points.reduce((sum, point) => sum + point[axis], 0) / points.length);
}
export function centerPoints(points) {
  const mean = meanOf(points);
  return { mean, centered: points.map(point => [point[0] - mean[0], point[1] - mean[1]]) };
}
export function unitDirection(angleDegrees) {
  if (!Number.isFinite(angleDegrees)) throw new RangeError('Use a finite angle in degrees.');
  const radians = angleDegrees * Math.PI / 180;
  return [Math.cos(radians), Math.sin(radians)];
}
const dot = (left, right) => left[0] * right[0] + left[1] * right[1];
const squaredLength = vector => dot(vector, vector);
/** Project every point onto the line through the mean with the given angle.
 * Returns scores, projections in original coordinates, residual vectors and the
 * conserved accounting total = retained + sse. */
export function projectAtAngle(points, angleDegrees) {
  const { mean, centered } = centerPoints(points);
  const direction = unitDirection(angleDegrees);
  const scores = centered.map(vector => dot(vector, direction));
  const projections = scores.map(score => [mean[0] + score * direction[0], mean[1] + score * direction[1]]);
  const residuals = points.map((point, index) => [point[0] - projections[index][0], point[1] - projections[index][1]]);
  const total = centered.reduce((sum, vector) => sum + squaredLength(vector), 0);
  const retained = scores.reduce((sum, score) => sum + score * score, 0);
  const sse = residuals.reduce((sum, vector) => sum + squaredLength(vector), 0);
  const denominator = Math.max(points.length - 1, 1);
  return { mean, centered, direction, angleDegrees, scores, projections, residuals, total, retained, sse, scoreVariance: retained / denominator, retainedFraction: total > 0 ? retained / total : null };
}
/** Exact principal directions of a two-dimensional cloud from its sample
 * covariance. Signs make the largest-magnitude coefficient positive. `tie`
 * reports equal eigenvalues, where every direction is equally good. */
export function principalDirections(points) {
  return principalDirectionsFromCentered(centerPoints(points));
}
/** The rectangle validates its own parameter ranges before forming coordinates,
 * which can reach ±40. Keep the free-point editor's stricter guard separate. */
function principalDirectionsFromCentered({ mean, centered }) {
  const denominator = Math.max(centered.length - 1, 1);
  const a = centered.reduce((sum, vector) => sum + vector[0] * vector[0], 0) / denominator;
  const b = centered.reduce((sum, vector) => sum + vector[0] * vector[1], 0) / denominator;
  const c = centered.reduce((sum, vector) => sum + vector[1] * vector[1], 0) / denominator;
  const total = a + c;
  const half = Math.hypot(a - c, 2 * b) / 2;
  const eigenvalues = [(a + c) / 2 + half, (a + c) / 2 - half];
  const tie = half <= 1e-12 * Math.max(1, Math.abs(total));
  const angle = tie ? 0 : Math.atan2(2 * b, a - c) / 2;
  const orient = vector => {
    const index = Math.abs(vector[0]) >= Math.abs(vector[1]) - 1e-12 ? 0 : 1;
    return vector[index] < 0 ? [-vector[0], -vector[1]] : vector;
  };
  const first = orient([Math.cos(angle), Math.sin(angle)]);
  const second = orient([-Math.sin(angle), Math.cos(angle)]);
  return {
    mean, covariance: [[a, b], [b, c]], eigenvalues, directions: [first, second],
    angles: [Math.atan2(first[1], first[0]) * 180 / Math.PI, Math.atan2(second[1], second[0]) * 180 / Math.PI],
    total, fractions: total > 0 ? eigenvalues.map(value => value / total) : [null, null], tie, degenerate: total <= 1e-12
  };
}
/** Rectangle corners (±a, ±m·b) in raw units or after standardization (ddof 0),
 * with the leading axis under the resulting metric. */
export function rectangleMetric(a = 2, b = 1, multiplier = 1, standardized = false) {
  for (const [value, low, high, name] of [[a, 0.25, 4, 'width a'], [b, 0.25, 4, 'height b'], [multiplier, 0.25, 10, 'unit multiplier']]) {
    if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`Use ${name} between ${low} and ${high}.`);
  }
  const raw = [[-a, -multiplier * b], [-a, multiplier * b], [a, -multiplier * b], [a, multiplier * b]];
  const scales = standardized ? [a, multiplier * b] : [1, 1];
  const points = raw.map(point => [point[0] / scales[0], point[1] / scales[1]]);
  const fit = principalDirectionsFromCentered({ mean: [0, 0], centered: points });
  const variances = [fit.covariance[0][0], fit.covariance[1][1]];
  // A learner typing the tie multiplier a/b to a few decimals must see a tie, not
  // an arbitrary winner decided by the last floating-point digit.
  const tie = fit.tie || Math.abs(variances[0] - variances[1]) <= 1e-4 * (variances[0] + variances[1]);
  const leadingAxis = tie ? 'tie' : variances[0] > variances[1] ? 'first' : 'second';
  return { a, b, multiplier, standardized, raw, points, scales, variances, fractions: variances.map(value => value / (variances[0] + variances[1])), leadingAxis, fit };
}
/** Four constructed observations (±a, ±b) with a binary label from the sign of
 * one coordinate. Reports which observations share a retained coordinate while
 * carrying different labels. Requires a > b so PC1 is the first axis. */
export function labelCollisions(a = 10, b = 1, labelAxis = 'y', kept = 'pc1') {
  if (!Number.isFinite(a) || !Number.isFinite(b) || a < 2 || a > 12 || b < 0.25 || b > 1.5) throw new RangeError('Use a in [2, 12] and b in [0.25, 1.5].');
  if (!['x', 'y'].includes(labelAxis) || !['pc1', 'pc2', 'both'].includes(kept)) throw new RangeError('Choose a label axis and retained components.');
  const points = [[-a, -b], [-a, b], [a, -b], [a, b]];
  const labels = points.map(point => (labelAxis === 'x' ? point[0] : point[1]) > 0 ? 'B' : 'A');
  const fit = principalDirections(points);
  const retainedIndices = kept === 'both' ? [0, 1] : [kept === 'pc1' ? 0 : 1];
  const retainedFraction = retainedIndices.reduce((sum, index) => sum + fit.fractions[index], 0);
  const coordinates = points.map(point => retainedIndices.map(index => Number((point[0] * fit.directions[index][0] + point[1] * fit.directions[index][1]).toFixed(12)) + 0));
  const groups = new Map();
  coordinates.forEach((coordinate, index) => {
    const key = coordinate.join(',');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(index);
  });
  const collisions = [...groups.entries()].filter(([, members]) => new Set(members.map(index => labels[index])).size > 1).map(([key, members]) => ({ coordinate: key.split(',').map(Number), members }));
  return { a, b, labelAxis, kept, points, labels, fit, retainedIndices, retainedFraction, coordinates, distinctLocations: groups.size, collisions, distinguishable: collisions.length === 0 };
}
const standardizeRow = (row, mean, scale) => row.map((value, feature) => (value - mean[feature]) / scale[feature]);
const project = (standardizedRow, components, k) => components.slice(0, k).map(direction => direction.reduce((sum, weight, feature) => sum + weight * standardizedRow[feature], 0));
const reconstruct = (scores, components, k) => standardizedRow => standardizedRow.map((_, feature) => components.slice(0, k).reduce((sum, direction, index) => sum + scores[index] * direction[feature], 0));
/** Standardized scores of all 178 wines on the first k full-collection components. */
export function wineFullScores(k = 2) {
  if (!Number.isInteger(k) || k < 1 || k > 13) throw new RangeError('Keep 1 to 13 components.');
  return wineRows.map(row => project(standardizeRow(row, wineFullFit.mean, wineFullFit.scale), wineFullFit.components, k));
}
/** Reconstruct the 45 validation wines from k training components and return
 * each k's mean squared error as a fraction of the training-mean baseline. */
export function wineValidationCurve() {
  const rows = wineSplit.validation.map(index => standardizeRow(wineRows[index], wineSplit.mean, wineSplit.scale));
  const baseline = rows.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value * value, 0), 0) / (rows.length * 13);
  const ratios = [];
  for (let k = 0; k <= 13; k += 1) {
    const error = rows.reduce((sum, row) => {
      const scores = project(row, wineSplit.components, k);
      const rebuilt = reconstruct(scores, wineSplit.components, k)(row);
      return sum + row.reduce((inner, value, feature) => inner + (value - rebuilt[feature]) ** 2, 0);
    }, 0) / (rows.length * 13);
    ratios.push(error / baseline);
  }
  return { baselineMse: baseline, ratios };
}
export function smallestComponentCount(ratios, budget) {
  if (!Number.isFinite(budget) || budget < 0.01 || budget > 0.6) throw new RangeError('Use a budget fraction between 0.01 and 0.60.');
  const index = ratios.findIndex(ratio => ratio <= budget);
  return { k: index, precedingRatio: index > 0 ? ratios[index - 1] : null, ratio: ratios[index] };
}
/** One validation wine reconstructed from k training components, in
 * standardized and original units, with per-feature residuals. */
export function wineReconstruction(validationPosition, k) {
  if (!Number.isInteger(validationPosition) || validationPosition < 0 || validationPosition >= wineSplit.validation.length) throw new RangeError('Choose one of the 45 validation rows.');
  if (!Number.isInteger(k) || k < 0 || k > 13) throw new RangeError('Keep 0 to 13 components.');
  const rowIndex = wineSplit.validation[validationPosition];
  const original = wineRows[rowIndex];
  const standardized = standardizeRow(original, wineSplit.mean, wineSplit.scale);
  const scores = project(standardized, wineSplit.components, k);
  const rebuiltStandardized = reconstruct(scores, wineSplit.components, k)(standardized);
  const rebuiltOriginal = rebuiltStandardized.map((value, feature) => value * wineSplit.scale[feature] + wineSplit.mean[feature]);
  return { rowIndex, original, standardized, scores, rebuiltStandardized, rebuiltOriginal, residuals: standardized.map((value, feature) => value - rebuiltStandardized[feature]), squaredError: standardized.reduce((sum, value, feature) => sum + (value - rebuiltStandardized[feature]) ** 2, 0) };
}
/** Storage accounting for k scores per row plus the decoder (directions and mean). */
export function storageScalars(n, d, k) {
  if (![n, d, k].every(value => Number.isInteger(value) && value >= 0) || k > d) throw new RangeError('Use nonnegative integers with k ≤ d.');
  return { original: n * d, compressed: n * k + d * k + d };
}
