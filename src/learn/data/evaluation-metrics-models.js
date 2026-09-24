// Small exact teaching models. Null means an absent denominator, never a hidden epsilon.
export const METRIC_ROWS = [0.95, 0.8, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1].map((score, i) => ({ id: 'ABCDEFGH'[i], y: [1, 0, 1, 1, 0, 1, 0, 0][i], score }));
export const RESIDUAL_ROWS = [1, 2, 3, 4, 10].map((y, i) => ({ id: `T${i + 1}`, y, a: [1, 2, 3, 4, 4][i], b: [-1, 0, 1, 2, 8][i] }));
export const DOCUMENT_ROWS = [0, 3, 1, 2, 0].map((grade, i) => ({ id: `D${i + 1}`, grade }));
const sum = values => values.reduce((a, b) => a + b, 0);
const ratio = (a, b) => b === 0 ? null : a / b;
export const displayMetric = value => value === null ? 'undefined' : value === Infinity ? '∞ (unbounded)' : Number(value.toFixed(6)).toString();
export const direction = (before, after) => before === null || after === null ? 'undefined' : Math.abs(after - before) < 1e-10 ? 'same' : after > before ? 'up' : 'down';
export function validateBinaryRows(rows) {
  if (!Array.isArray(rows) || rows.length < 1 || rows.some(r => ![0, 1].includes(r.y) || !Number.isFinite(r.score))) throw new Error('Use binary labels and finite scores.');
}
export function binaryMetrics(rows, threshold, beta = 1, fpCost = 1, fnCost = 3) {
  validateBinaryRows(rows);
  if (Number.isNaN(threshold) || !Number.isFinite(beta) || beta <= 0 || ![fpCost, fnCost].every(c => Number.isFinite(c) && c >= 0)) throw new Error('Use a threshold, positive beta and nonnegative costs.');
  const cells = { tp: [], fp: [], fn: [], tn: [] };
  for (const row of rows) cells[row.score >= threshold ? (row.y ? 'tp' : 'fp') : (row.y ? 'fn' : 'tn')].push(row.id);
  const { tp, fp, fn, tn } = Object.fromEntries(Object.entries(cells).map(([key, ids]) => [key, ids.length]));
  const precision = ratio(tp, tp + fp), recall = ratio(tp, tp + fn), specificity = ratio(tn, tn + fp);
  const mccDen = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return { cells, tp, fp, fn, tn, precision, recall, specificity, fpr: ratio(fp, fp + tn), accuracy: (tp + tn) / rows.length, f1: ratio(2 * tp, 2 * tp + fp + fn), fbeta: ratio((1 + beta ** 2) * tp, (1 + beta ** 2) * tp + fp + beta ** 2 * fn), cost: fpCost * fp + fnCost * fn, mcc: ratio(tp * tn - fp * fn, mccDen), balancedAccuracy: recall === null || specificity === null ? null : (recall + specificity) / 2 };
}
export function rankingMetrics(rows) {
  validateBinaryRows(rows);
  const sorted = [...rows].sort((a, b) => b.score - a.score);
  const groups = [];
  for (const row of sorted) {
    if (!groups.length || groups.at(-1).score !== row.score) groups.push({ score: row.score, rows: [] });
    groups.at(-1).rows.push(row);
  }
  const points = [{ threshold: Infinity, ...binaryMetrics(rows, Infinity), ids: [] }, ...groups.map(group => ({ threshold: group.score, ...binaryMetrics(rows, group.score), ids: group.rows.map(r => r.id) }))];
  const positives = rows.filter(r => r.y), negatives = rows.filter(r => !r.y);
  const pairs = positives.map(p => negatives.map(n => ({ positive: p, negative: n, credit: p.score > n.score ? 1 : p.score === n.score ? 0.5 : 0 })));
  const auc = ratio(sum(pairs.flat().map(p => p.credit)), positives.length * negatives.length);
  const ap = positives.length ? sum(points.slice(1).map((p, i) => (p.recall - points[i].recall) * p.precision)) : null;
  const prTrapezoid = positives.length ? sum(points.slice(1).map((p, i) => (p.recall - points[i].recall) * (p.precision + (points[i].precision ?? 1)) / 2)) : null;
  return { groups, points, pairs, positives, negatives, auc, ap, prTrapezoid };
}
export function probabilityMetrics(rows) {
  validateBinaryRows(rows);
  if (rows.some(r => r.score < 0 || r.score > 1)) throw new Error('Probability forecasts must be in [0, 1].');
  const contributions = rows.map(r => ({ ...r, log: -Math.log(r.y ? r.score : 1 - r.score), brier: (r.score - r.y) ** 2 }));
  return { contributions, log: sum(contributions.map(r => r.log)) / rows.length, brier: sum(contributions.map(r => r.brier)) / rows.length };
}
export function regressionMetrics(rows, key = 'a', unit = 1) {
  if (!rows.length || rows.some(r => !Number.isFinite(r.y) || !Number.isFinite(r[key])) || !Number.isFinite(unit) || unit <= 0) throw new Error('Use nonempty finite targets, predictions and a positive unit scale.');
  const mean = sum(rows.map(r => r.y * unit)) / rows.length;
  const contributions = rows.map(r => ({ id: r.id, y: r.y * unit, prediction: r[key] * unit, residual: (r.y - r[key]) * unit, square: ((r.y - r[key]) * unit) ** 2 }));
  const absolutes = contributions.map(r => Math.abs(r.residual)).sort((a, b) => a - b);
  const sse = sum(contributions.map(r => r.square)), sst = sum(contributions.map(r => (r.y - mean) ** 2));
  const mse = sse / rows.length;
  return { contributions, mean, sse, sst, mae: sum(absolutes) / rows.length, mse, rmse: Math.sqrt(mse), medae: (absolutes[Math.floor((rows.length - 1) / 2)] + absolutes[Math.ceil((rows.length - 1) / 2)]) / 2, r2: rows.length < 2 || sst === 0 ? null : 1 - sse / sst, mape: rows.some(r => r.y === 0) ? null : sum(contributions.map(r => Math.abs(r.residual / r.y))) / rows.length * 100 };
}
export function retrievalMetrics(rows, cutoff = 3, gain = 'exponential') {
  if (!rows.length || !Number.isInteger(cutoff) || cutoff < 1 || cutoff > rows.length || rows.some(r => !Number.isInteger(r.grade) || r.grade < 0 || r.grade > 5) || !['exponential', 'linear'].includes(gain)) throw new Error('Use integer grades 0–5 and a cutoff within the list.');
  const gained = grade => gain === 'exponential' ? 2 ** grade - 1 : grade;
  const shelf = list => list.map((r, i) => ({ ...r, rank: i + 1, gain: gained(r.grade), discount: Math.log2(i + 2), contribution: i < cutoff ? gained(r.grade) / Math.log2(i + 2) : 0 }));
  const actual = shelf(rows), ideal = shelf([...rows].sort((a, b) => b.grade - a.grade));
  const relevant = rows.filter(r => r.grade > 0).length, found = rows.slice(0, cutoff).filter(r => r.grade > 0).length;
  const first = rows.findIndex(r => r.grade > 0);
  let cumulative = 0;
  const precisionCredits = rows.map((r, i) => { if (r.grade > 0) cumulative++; return r.grade > 0 ? cumulative / (i + 1) : 0; });
  const dcg = sum(actual.map(r => r.contribution)), idcg = sum(ideal.map(r => r.contribution));
  return { actual, ideal, precision: found / cutoff, recall: ratio(found, relevant), rr: first === -1 ? 0 : 1 / (first + 1), ap: ratio(sum(precisionCredits), relevant), precisionCredits, dcg, idcg, ndcg: ratio(dcg, idcg) };
}
