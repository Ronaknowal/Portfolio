// Complementary invariance checks for the finite Survival lesson models.
// Run after the author freezes the source; reuse its separate numerical oracles.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as survival from '../src/learn/data/survival-models.js';

const close = (actual, expected, context) => {
  assert.ok(Math.abs(actual - expected) <= 2e-11 * Math.max(1, Math.abs(expected)), `${context}: ${actual} != ${expected}`);
};
const cases = [];
const base = survival.kaplanMeier(survival.pumpTimes, survival.pumpEvents);
const baseArea = survival.restrictedMean(base, 9).value;
for (const scale of [0.25, 24, 100]) {
  const table = survival.kaplanMeier(survival.pumpTimes.map(time => scale * time), survival.pumpEvents);
  for (let index = 0; index < base.rows.length; index += 1) {
    const actual = table.rows[index];
    const expected = base.rows[index];
    close(actual.time, expected.time * scale, 'time units');
    for (const key of ['risk', 'failures', 'censored', 'survival', 'greenwood', 'nelsonAalen']) close(actual[key], expected[key], key);
    if (expected.interval) actual.interval.forEach((value, bound) => close(value, expected.interval[bound], 'pointwise limits'));
  }
  close(table.median, base.median * scale, 'median units');
  close(survival.restrictedMean(table, 9 * scale).value, baseArea * scale, 'restricted-mean units');
  for (const time of [0, 3.99, 4, 4.01, 9]) close(survival.survivalAt(table, time * scale), survival.survivalAt(base, time), 'same step under unit conversion');
  cases.push({ kind: 'time-unit conversion preserves risk/probabilities and scales medians/areas', scale });
}

for (const ties of ['efron', 'breslow']) {
  const reference = survival.coxRiskSets({ beta: 0.4, ties });
  for (const [scale, offset] of [[1.5, 0.7], [-1, 2], [0.5, -1]]) {
    const transformed = survival.coxRiskSets({ beta: 0.4 / scale, ties, features: survival.coxFeatures.map(value => scale * value + offset) });
    close(transformed.logLikelihood, reference.logLikelihood, 'Cox score offset cancels from risk sets');
    close(transformed.score, reference.score * scale, 'Cox score covector');
    close(transformed.information, reference.information * scale ** 2, 'Cox information coordinates');
    transformed.rows.forEach((row, index) => row.weights.forEach((item, position) => close(item.probability, reference.rows[index].weights[position].probability, 'risk probability')));
    cases.push({ kind: 'affine feature coordinates preserve the fitted relative-hazard law', ties, scale, offset });
  }
  const constant = survival.coxRiskSets({ beta: 2, ties, features: survival.coxFeatures.map(() => 3) });
  close(constant.score, 0, 'constant feature has no Cox score');
  close(constant.information, 0, 'constant feature has no Cox information');
  cases.push({ kind: 'constant feature has no identifiable relative effect', ties });
}

for (const groups of [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0]]) {
  const first = survival.logrankTable(survival.pumpTimes, survival.pumpEvents, groups);
  const renamed = survival.logrankTable(survival.pumpTimes, survival.pumpEvents, groups.map(value => 1 - value));
  close(renamed.difference, -first.difference, 'renaming groups reverses O-E');
  close(renamed.variance, first.variance, 'group-name-independent variance');
  close(renamed.statistic, first.statistic, 'same log-rank statistic');
  cases.push({ kind: 'group-label exchange preserves comparison evidence', groups });
}

for (const scores of [[3, 2, 2, 0], [0, 0, 0, 0], [-2, 3, -1, 0]]) {
  const first = survival.concordancePairs([1, 2, 2, 4], [true, false, true, true], scores);
  const reversed = survival.concordancePairs([1, 2, 2, 4], [true, false, true, true], scores.map(value => -value));
  assert.equal(reversed.comparable, first.comparable);
  assert.equal(reversed.concordant, first.discordant);
  assert.equal(reversed.tied, first.tied);
  close(reversed.value, 1 - first.value, 'reversing risk ordering complements concordance');
  cases.push({ kind: 'risk reversal complements concordance including score/time ties', scores });
}

for (const prediction of [0.1, 0.6, 0.9]) {
  for (const lateCensorProbability of [0.2, 0.5, 1]) {
    const result = survival.censorWeightedBrier({ prediction, lateCensorProbability });
    close(result.weighted - 0.24, (prediction - 0.6) ** 2, 'proper-score excess risk survives independent censoring');
  }
  cases.push({ kind: 'Brier excess risk equals squared forecast error under changed follow-up', prediction });
}

const times = [1, 2, 3, 4, 5, 6];
for (const statuses of [[2, 1, 0, 2, 1, 0], [1, 2, 2, 0, 0, 1], [0, 1, 0, 0, 2, 0]]) {
  const first = survival.competingIncidence(times, statuses);
  const renamed = survival.competingIncidence(times, statuses.map(value => value === 0 ? 0 : 3 - value));
  first.rows.forEach((row, index) => {
    close(renamed.rows[index].first, row.second, 'cause exchange');
    close(renamed.rows[index].second, row.first, 'cause exchange');
    close(renamed.rows[index].survival, row.survival, 'event-free state independent of cause names');
    close(row.first + row.second + row.survival, 1, 'population mass conservation');
  });
  cases.push({ kind: 'cause-name exchange preserves probability flow', statuses });
}

const source = 'src/learn/data/survival-models.js';
const directory = 'scratch/survival-independent';
fs.mkdirSync(directory, { recursive: true });
fs.writeFileSync(`${directory}/complementary-checks.json`, JSON.stringify({
  checkedAt: new Date().toISOString(), source,
  sha256: createHash('sha256').update(fs.readFileSync(source)).digest('hex'),
  scope: 'Complementary algebraic/interpretation invariants; author library and program executions are separate.', cases,
}, null, 2) + '\n');
console.log(`PASS: ${cases.length} complementary Survival invariance cases.`);
