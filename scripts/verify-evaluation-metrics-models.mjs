import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import katex from 'katex';
import { METRIC_ROWS, RESIDUAL_ROWS, DOCUMENT_ROWS, binaryMetrics, rankingMetrics, probabilityMetrics, regressionMetrics, retrievalMetrics, direction } from '../src/learn/data/evaluation-metrics-models.js';
import { evaluationMetricsExamples } from '../src/learn/data/evaluation-metrics-examples.js';
const fixture = JSON.parse(fs.readFileSync('public/learn-assets/evaluation-metrics/metric-fixtures.json', 'utf8'));
const close = (actual, expected) => expected === null ? assert.equal(actual, null) : assert.ok(Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${actual} versus ${expected}`);
const groups = [];
function check(name, fn) { fn(); groups.push(name); }
function compareCurve(rows, reference) {
  const result = rankingMetrics(rows);
  close(result.auc, reference.roc_auc); close(result.ap, reference.ap);
  assert.deepEqual(result.points.map(p => p.tp), reference.tp);
  assert.deepEqual(result.points.map(p => p.fp), reference.fp);
  for (const [i, point] of result.points.entries()) {
    close(point.recall, reference.recall?.[i] ?? null); close(point.fpr, reference.fpr?.[i] ?? null);
    close(point.precision ?? 1, reference.precision_for_plot[i]);
  }
}
check('Threshold counts/rates/costs match independently executed Python fixtures', () => {
  for (const [threshold, expected] of Object.entries(fixture.core.thresholds)) {
    const actual = binaryMetrics(METRIC_ROWS, Number(threshold));
    for (const [key, value] of Object.entries({ ...expected.counts, ...expected.rates })) close(actual[key], value);
    close(actual.cost, expected.cost_fp1_fn3);
    assert.equal(Object.values(actual.cells).flat().length, METRIC_ROWS.length);
  }
  assert.deepEqual(binaryMetrics(METRIC_ROWS, .7), binaryMetrics(METRIC_ROWS, .75));
  assert.equal(binaryMetrics(METRIC_ROWS.map(r => ({ ...r, score: r.score + 5 })), Infinity).precision, null);
});
check('Tied thresholds, no-class domains, constant-score null and changed C match Python', () => {
  compareCurve(METRIC_ROWS, fixture.core.curve);
  compareCurve(METRIC_ROWS.map(r => ({ ...r, score: .5 })), fixture.nulls.constant_scores);
  compareCurve(METRIC_ROWS.map(r => ({ ...r, y: 0 })), fixture.nulls.no_positives);
  compareCurve(METRIC_ROWS.map(r => ({ ...r, y: 1 })), fixture.nulls.no_negatives);
  compareCurve(METRIC_ROWS.map(r => r.id === 'C' ? { ...r, score: .45 } : r), fixture.contrasts.edited_C_score045);
  close(rankingMetrics(METRIC_ROWS).prTrapezoid, fixture.core.trapezoidal_pr_area);
});
check('Independent pair-credit enumeration agrees with ROC area over 40 changed datasets', () => {
  for (let seed = 1; seed <= 40; seed++) {
    const rows = Array.from({ length: 4 + seed % 13 }, (_, i) => ({ id: String(i), y: (i + seed) % 2, score: ((i * seed + 3) % 7) - 3 }));
    const model = rankingMetrics(rows);
    let rocArea = 0;
    for (let i = 1; i < model.points.length; i++) rocArea += (model.points[i].fpr - model.points[i - 1].fpr) * (model.points[i].recall + model.points[i - 1].recall) / 2;
    close(model.auc, rocArea);
    close(model.auc, rankingMetrics([...rows].reverse()).auc);
    close(model.ap, rankingMetrics([...rows].reverse()).ap);
  }
});
check('Squared probabilities preserve ranking while probability loss changes; exact boundary infinities', () => {
  const squared = METRIC_ROWS.map(r => ({ ...r, score: r.score ** 2 }));
  close(rankingMetrics(squared).auc, rankingMetrics(METRIC_ROWS).auc);
  close(rankingMetrics(squared).ap, rankingMetrics(METRIC_ROWS).ap);
  close(probabilityMetrics(METRIC_ROWS).log, .5775405542);
  close(probabilityMetrics(squared).log, .6673345247);
  close(probabilityMetrics(METRIC_ROWS).brier, .2040625);
  close(probabilityMetrics(squared).brier, .23132578125);
  assert.equal(probabilityMetrics([{ y: 1, score: 0 }]).log, Infinity);
  assert.equal(probabilityMetrics([{ y: 1, score: 1 }]).log, 0);
});
function compareResidual(rows, key, reference, unit = 1) {
  const actual = regressionMetrics(rows, key, unit);
  for (const field of ['mae', 'mse', 'rmse', 'sse', 'sst', 'r2']) close(actual[field], reference[field]);
  close(actual.medae, reference.median_absolute_error); close(actual.mean, reference.evaluation_mean);
  actual.contributions.forEach((r, i) => { close(r.residual, reference.residual[i]); close(r.square, reference.squared[i]); });
}
check('Residuals, unit conversions, fitted/evaluation references, perfect and constant cases match Python', () => {
  compareResidual(RESIDUAL_ROWS, 'a', fixture.regression.A);
  compareResidual(RESIDUAL_ROWS, 'b', fixture.regression.B);
  compareResidual(RESIDUAL_ROWS, 'a', fixture.contrasts.A_seconds, 60);
  compareResidual(RESIDUAL_ROWS.map(r => ({ ...r, a: 3 })), 'a', fixture.regression.training_mean3);
  compareResidual(RESIDUAL_ROWS.map(r => ({ ...r, a: 4 })), 'a', fixture.regression.evaluation_mean);
  compareResidual(RESIDUAL_ROWS.map(r => ({ ...r, a: r.y })), 'a', fixture.regression.perfect);
  compareResidual(RESIDUAL_ROWS.map(r => ({ ...r, y: 4, a: 4 })), 'a', fixture.regression.constant_target);
  compareResidual(RESIDUAL_ROWS.map((r, i) => i === 4 ? { ...r, a: 8 } : r), 'a', fixture.contrasts.A_last_prediction8);
});
function compareRetrieval(rows, reference) {
  const actual = retrievalMetrics(rows);
  for (const [key, expected] of Object.entries({ precision: reference.precision_at_k, recall: reference.recall_at_k, rr: reference.reciprocal_rank, ap: reference.ap, dcg: reference.dcg, idcg: reference.ideal_dcg, ndcg: reference.ndcg })) close(actual[key], expected);
}
check('Actual/ideal retrieval shelves, zero-grade undefineds, swap/null and both gain conventions match Python', () => {
  compareRetrieval(DOCUMENT_ROWS, fixture.retrieval.original);
  compareRetrieval([DOCUMENT_ROWS[1], DOCUMENT_ROWS[0], ...DOCUMENT_ROWS.slice(2)], fixture.retrieval.changed_order);
  compareRetrieval([DOCUMENT_ROWS[4], ...DOCUMENT_ROWS.slice(1, 4), DOCUMENT_ROWS[0]], fixture.retrieval.original);
  compareRetrieval(DOCUMENT_ROWS.map(r => ({ ...r, grade: 0 })), fixture.retrieval.all_zero);
  close(retrievalMetrics(DOCUMENT_ROWS, 3, 'linear').ndcg, fixture.retrieval.sklearn_linear_gain);
  close(retrievalMetrics(DOCUMENT_ROWS).ndcg, fixture.retrieval.sklearn_exponential_gain);
  for (let cutoff = 1; cutoff <= 5; cutoff++) close(retrievalMetrics([...DOCUMENT_ROWS].sort((a, b) => b.grade - a.grade), cutoff).ndcg, 1);
});
check('Undefined and invalid inputs remain distinct; changed numeric judgments are tolerance-aware', () => {
  assert.throws(() => binaryMetrics([{ y: 2, score: 1 }], .5));
  assert.throws(() => probabilityMetrics([{ y: 0, score: -1 }]));
  assert.throws(() => retrievalMetrics([{ id: 'x', grade: 1.5 }], 1));
  assert.throws(() => regressionMetrics([{ y: '', a: 2 }]));
  assert.equal(direction(1, 1 + 1e-8), 'up'); assert.equal(direction(1, 1 + 1e-12), 'same');
  assert.equal(binaryMetrics([{ y: 0, score: 0 }], .5).f1, null);
  assert.equal(regressionMetrics([{ y: 0, a: 1 }]).mape, null);
});
let mathExpressions = 0;
check('Every actual JSX math string parses; all nine figures/four labs and eight worked practices retained', () => {
  const text = fs.readFileSync('src/learn/data/topics/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae.jsx', 'utf8');
  const tree = parse(text, { sourceType: 'module', plugins: ['jsx'] });
  function visit(node) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'JSXElement' && node.openingElement.name.name === 'MathBlock') {
      const value = node.children.find(child => child.type === 'JSXExpressionContainer').expression.value;
      assert.ok(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value));
      katex.renderToString(value, { throwOnError: true }); mathExpressions++;
    }
    for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
  }
  visit(tree); assert.equal(mathExpressions, 14);
  assert.equal((text.match(/<details><summary>\{"Hint"\}/g) || []).length, 8);
  assert.ok(!text.includes('[Inline figure:'));
  for (const [key, file] of [['calculations', 'metrics-calculations.py'], ['evaluation', 'banknote-evaluation.py']]) assert.equal(evaluationMetricsExamples[key].code, fs.readFileSync(`public/learn-assets/evaluation-metrics/${file}`, 'utf8').replaceAll('\r\n', '\n'));
});
const files = ['src/learn/data/evaluation-metrics-models.js', 'src/learn/data/topics/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae.jsx', 'scripts/verify-evaluation-metrics-models.mjs'];
const report = { status: 'passed', checks: groups, mathExpressions, independentChangedRankings: 40, files: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/evaluation-metrics-models.json', JSON.stringify(report, null, 2)+'\n');
console.log(JSON.stringify({ status: 'passed', groups: groups.length, mathExpressions }));

