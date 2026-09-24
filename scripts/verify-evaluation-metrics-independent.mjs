// Reviewer-owned numerical checks: rank sums, positive credit, and raw retained predictions.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { binaryMetrics, rankingMetrics, probabilityMetrics, regressionMetrics, retrievalMetrics } from '../src/learn/data/evaluation-metrics-models.js';

const draft = 'docs/teaching/drafts/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae';
const assets = 'public/learn-assets/evaluation-metrics';
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const receipt = { reviewer: 'semi_supervised_implementation', passed: false, numericAssertions: 0, groups: [], source: {} };
function near(actual, expected, message) {
  receipt.numericAssertions++;
  if (expected === null || !Number.isFinite(expected)) assert.equal(actual, expected, message);
  else assert.ok(Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${message}: ${actual} != ${expected}`);
}
function counts(rows, threshold) {
  // Explicit bit-pattern tally, independent of the production cell-selection expression.
  const bins = [0, 0, 0, 0];
  for (const row of rows) bins[2 * row.y + Number(row.score >= threshold)]++;
  return { tn: bins[0], fp: bins[1], fn: bins[2], tp: bins[3] };
}
function rankedReference(rows) {
  const positives = rows.filter(row => row.y === 1);
  const negatives = rows.length - positives.length;
  // Mann–Whitney rank-sum formula, with average ranks for equal scores.
  const rankSum = positives.reduce((total, row) => total + 1 + rows.filter(other => other.score < row.score).length + (rows.filter(other => other.score === row.score).length - 1) / 2, 0);
  const auc = positives.length && negatives ? (rankSum - positives.length * (positives.length + 1) / 2) / (positives.length * negatives) : null;
  // Each positive gets precision at its entire score block's ending rank.
  const ap = positives.length ? positives.reduce((total, row) => total + rows.filter(other => other.score >= row.score && other.y === 1).length / rows.filter(other => other.score >= row.score).length, 0) / positives.length : null;
  return { auc, ap };
}

// Deterministic varied populations: one-class states, score ties, negative scores and reversed input order.
for (let seed = 0; seed < 240; seed++) {
  const n = 4 + seed % 9;
  const rows = Array.from({ length: n }, (_, i) => ({ id: `r${i}`, y: seed % 17 === 0 ? 0 : seed % 19 === 0 ? 1 : Number(((seed * 17 + i * 11) % 13) < 5), score: ((seed * 7 + i * i + 3 * i) % 7) - 3 }));
  const expected = rankedReference(rows), actual = rankingMetrics(rows);
  near(actual.auc, expected.auc, `rank-sum AUC seed ${seed}`);
  near(actual.ap, expected.ap, `positive-credit AP seed ${seed}`);
  const reversed = rankingMetrics([...rows].reverse());
  near(reversed.auc, expected.auc, `input order AUC ${seed}`);
  near(reversed.ap, expected.ap, `input order AP ${seed}`);
  assert.equal(actual.points.length, new Set(rows.map(r => r.score)).size + 1);
  for (const threshold of [-Infinity, -2.5, 0, 2, Infinity]) {
    const expectedCounts = counts(rows, threshold), value = binaryMetrics(rows, threshold, 2, 0, 5);
    for (const key of ['tp', 'fp', 'fn', 'tn']) near(value[key], expectedCounts[key], `counts ${seed}/${threshold}/${key}`);
    near(value.cost, 5 * expectedCounts.fn, `zero false-positive cost ${seed}/${threshold}`);
    near(value.fbeta, value.tp + value.fp + value.fn ? 5 * value.tp / (5 * value.tp + value.fp + 4 * value.fn) : null, 'F2 count expression');
  }
}
receipt.groups.push('240 varied binary populations: rank-sum AUC, positive-credit AP, permutations, full tie groups and 1,200 threshold/cost/F2 cases');

const probabilityCases = [
  [{ id: 'a', y: 0, score: 0 }, { id: 'b', y: 1, score: 1 }],
  [{ id: 'a', y: 0, score: 1 }, { id: 'b', y: 1, score: 0 }],
  [{ id: 'a', y: 0, score: 0.49 }, { id: 'b', y: 1, score: 0.51 }],
];
for (const rows of probabilityCases) {
  const actual = probabilityMetrics(rows);
  near(actual.log, rows.reduce((sum, row) => sum - Math.log(row.y === 1 ? row.score : 1 - row.score), 0) / rows.length, 'probability boundary log');
  near(actual.brier, rows.reduce((sum, row) => sum + (row.score - row.y) ** 2, 0) / rows.length, 'probability boundary Brier');
}
assert.throws(() => probabilityMetrics([{ id: 'invalid', y: 0, score: -0.1 }]));
receipt.groups.push('Probability exact zero/one boundaries retain zero or infinite mathematical loss, plus finite forecast and invalid domain');

for (let seed = 0; seed < 60; seed++) {
  const rows = Array.from({ length: 3 + seed % 8 }, (_, i) => ({ id: `t${i}`, y: seed % 10 === 0 ? 4 : ((i * i + seed) % 19) - 5, a: seed % 10 === 0 ? 4 : (i * 3 + seed) % 14 - 5, b: 3 }));
  const result = regressionMetrics(rows), seconds = regressionMetrics(rows, 'a', 60);
  const differences = rows.map(row => row.y - row.a);
  const squared = differences.reduce((sum, difference) => sum + difference * difference, 0);
  // Pair-difference identity SST = sum_{i<j}(yi-yj)^2/n avoids using the implementation's centered mean route.
  let pairSquared = 0;
  for (let i = 0; i < rows.length; i++) for (let j = i + 1; j < rows.length; j++) pairSquared += (rows[i].y - rows[j].y) ** 2;
  const sst = pairSquared / rows.length;
  near(result.sst, sst, `pairwise SST ${seed}`);
  near(result.r2, sst ? 1 - squared / sst : null, `R2 pair-reference ${seed}`);
  near(result.rmse ** 2, squared / rows.length, `RMSE ${seed}`);
  near(seconds.mae, 60 * result.mae, `MAE unit scale ${seed}`);
  near(seconds.mse, 3600 * result.mse, `MSE unit scale ${seed}`);
  near(seconds.r2, result.r2, `R2 unit invariance ${seed}`);
}
receipt.groups.push('60 residual datasets: independent pair-difference SST/R2 identity, constant perfect targets and unit changes');

function permutations(values) {
  if (!values.length) return [[]];
  return values.flatMap((value, i) => permutations(values.filter((_, j) => i !== j)).map(rest => [value, ...rest]));
}
const permutationsOfIds = permutations([0, 1, 2, 3, 4]);
for (const order of permutationsOfIds) for (const gain of ['linear', 'exponential']) for (const cutoff of [1, 3, 5]) {
  const rows = order.map(i => ({ id: `d${i}`, grade: [0, 3, 1, 2, 0][i] }));
  const r = retrievalMetrics(rows, cutoff, gain);
  const weights = rows.map(row => gain === 'linear' ? row.grade : 2 ** row.grade - 1);
  const discount = index => Math.log(2) / Math.log(index + 2);
  const dcg = weights.slice(0, cutoff).reduce((sum, value, i) => sum + value * discount(i), 0);
  const idcg = [...weights].sort((a, b) => b - a).slice(0, cutoff).reduce((sum, value, i) => sum + value * discount(i), 0);
  near(r.dcg, dcg, 'retrieval natural-log DCG'); near(r.ndcg, dcg / idcg, 'retrieval same-candidate NDCG');
  const relevantRanks = rows.map((row, i) => row.grade > 0 ? i + 1 : null).filter(Boolean);
  near(r.ap, relevantRanks.reduce((sum, rank, i) => sum + (i + 1) / rank, 0) / relevantRanks.length, 'retrieval relevant-rank AP');
  near(r.rr, 1 / relevantRanks[0], 'retrieval first success');
}
const zero = retrievalMetrics([0, 1, 2].map(id => ({ id, grade: 0 })), 2);
for (const key of ['ndcg', 'ap', 'recall']) near(zero[key], null, `no-relevance ${key}`);
near(zero.rr, 0, 'declared no-relevance RR convention');
receipt.groups.push('720 retrieval cases across all 120 stable-ID permutations, both gain conventions, three cutoffs; no-relevance null');

// Native evidence integrity: exact source/data bytes and display-output binding, not a second fitting run.
const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/evaluation-metrics-native.json', 'utf8'));
const measured = JSON.parse(fs.readFileSync(`${assets}/banknote-evaluation-results.json`, 'utf8'));
for (const file of ['metrics-calculations.py', 'banknote-evaluation.py', 'banknote-subset.csv']) {
  assert.equal(digest(`${assets}/${file}`), digest(`${draft}/${file}`), `exact packet bytes ${file}`);
  assert.equal(native.inputHashes[file], digest(`${assets}/${file}`), `native evidence bytes ${file}`);
}
const { evaluationMetricsExamples: examples } = await import('../src/learn/data/evaluation-metrics-examples.js');
for (const [key, filename] of [['calculations', 'metrics-calculations.py'], ['evaluation', 'banknote-evaluation.py']]) {
  assert.equal(examples[key].code.replaceAll('\r\n', '\n'), fs.readFileSync(`${assets}/${filename}`, 'utf8').replaceAll('\r\n', '\n'));
  assert.equal(examples[key].expected, native.stdout[filename]);
}
const testRows = measured.test_labels.map((y, i) => ({ id: `test${i}`, y, score: measured.test_scores[i] }));
const developmentRows = measured.development_labels.map((y, i) => ({ id: `dev${i}`, y, score: measured.development_scores[i] }));
const costs = measured.development_thresholds.map(row => {
  const count = counts(developmentRows, row.threshold === 'above_maximum' ? Infinity : row.threshold);
  for (const key of ['tp', 'fp', 'fn', 'tn']) near(count[key], row[key], `retained development ${key}`);
  const cost = count.fp + 5 * count.fn; near(cost, row.cost_fp1_fn5, 'retained development cost'); return cost;
});
assert.deepEqual(measured.selected_development_record, measured.development_thresholds[costs.indexOf(Math.min(...costs))]);
for (const [name, threshold] of [['fixed_0.5', 0.5], ['development_selected', measured.selected_development_record.threshold]]) {
  const report = measured.reports[name], count = counts(testRows, threshold);
  for (const key of ['tp', 'fp', 'fn', 'tn']) near(count[key], report[key], `${name} held-out ${key}`);
  near(count.fp + 5 * count.fn, report.cost_fp1_fn5, `${name} held-out cost`);
  near(rankedReference(testRows).auc, report.roc_auc, `${name} held-out rank-sum AUC`);
  near(rankedReference(testRows).ap, report.ap, `${name} held-out positive-credit AP`);
}
receipt.groups.push('Exact packet/download/native stdout integrity; all 81 development candidate counts/costs, highest-threshold tie rule, frozen held-out counts/costs/AUC/AP');

for (const file of ['src/learn/data/evaluation-metrics-models.js', 'src/learn/data/evaluation-metrics-examples.js', 'src/learn/data/evaluation-metrics-data.json', 'scripts/verify-evaluation-metrics-independent.mjs', `${assets}/metrics-calculations.py`, `${assets}/banknote-evaluation.py`, `${assets}/banknote-subset.csv`, `${assets}/banknote-evaluation-results.json`, 'docs/teaching/evidence/evaluation-metrics-native.json']) receipt.source[file] = digest(file);
receipt.passed = true;
fs.writeFileSync('docs/teaching/evidence/evaluation-metrics-independent.json', `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({ passed: receipt.passed, numericAssertions: receipt.numericAssertions, groups: receipt.groups }));
