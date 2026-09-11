import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { normalizeNaiveBayesScores, tokenEvidenceState, gaussianLogDensity, gaussianGeometryState, reliabilityState } from '../src/learn/data/naive-bayes-models.js';

const cases = [];
const close = (first, second, tolerance = 1e-11) => assert(Math.abs(first - second) <= tolerance);
for (const shift of [-800, -20, 0, 20, 800]) {
  const actual = normalizeNaiveBayesScores([-2, 0, 3].map(value => value + shift));
  const denominator = Math.exp(-2) + 1 + Math.exp(3);
  actual.probabilities.forEach((probability, index) => close(probability, [Math.exp(-2), 1, Math.exp(3)][index] / denominator));
  cases.push({ kind: 'common score-shift invariance', shift });
}
for (const alpha of [.25, 1, 7]) {
  const first = tokenEvidenceState('free meeting free win', alpha, .31).final;
  const second = tokenEvidenceState('win free free meeting', alpha, .31).final;
  first.probabilities.forEach((probability, index) => close(probability, second.probabilities[index]));
  cases.push({ kind: 'bag-of-words permutation invariance', alpha });
}
for (const value of [-2, 0, 1.5]) {
  const variance = .8;
  const original = gaussianLogDensity(value, .3, variance);
  const converted = gaussianLogDensity(value / 1000, .3 / 1000, variance / 1e6);
  close(converted - original, Math.log(1000));
  cases.push({ kind: 'density unit Jacobian', value });
}
for (const point of [[-4,4],[-1,1],[0,0],[3,-3]]) {
  const state = gaussianGeometryState('equal',point);
  close(state.probabilities[0],.5);
  close(state.probabilities[1],.5);
  assert.equal(state.decision,0);
  cases.push({kind:'equal-variance boundary symmetry',point});
}
for (const compression of [false,true]) {
  const reference = reliabilityState(2,compression);
  for (const count of [3,4,5,6]) {
    const current = reliabilityState(count,compression);
    close(current.brier,reference.brier);
    close(current.logLoss,reference.logLoss);
    assert.equal(current.bins.reduce((sum,bin)=>sum+bin.count,0),12);
  }
  cases.push({kind:'binning cannot change proper scores',compression});
}
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/naive-bayes-author-review.json'));
for (const row of author.production) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(row.path)).digest('hex'),row.sha256);
fs.mkdirSync('scratch/naive-bayes-independent',{recursive:true});
fs.writeFileSync('scratch/naive-bayes-independent/complementary-checks.json',JSON.stringify({checkedAt:new Date().toISOString(),cases,production:author.production,scope:'Complementary invariance/source identity checks; reuses author full native/browser evidence.'},null,2)+'\n');
console.log('Naive Bayes complementary invariance and frozen-source checks passed.');
