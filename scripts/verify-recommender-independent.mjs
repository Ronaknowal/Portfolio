import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as m from '../src/learn/data/recommender-models.js';
import { recommenderExamples as examples } from '../src/learn/data/recommender-examples.js';

const counts = {};
const close = (a, b) => assert(Math.abs(a - b) < 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)), `${a} != ${b}`);
const rotate = ([a, b]) => [(.6 * a - .8 * b), (.8 * a + .6 * b)];
for (const penalty of [0, .3, 2]) {
  for (const rate of [0, .15]) {
    const options = { user: [.3, -.7], positive: [-.2, .5], negative: [.9, .4], penalty, rate };
    const original = m.bprPairStep(options);
    const changed = m.bprPairStep({ ...options, user: rotate(options.user), positive: rotate(options.positive), negative: rotate(options.negative) });
    close(original.loss, changed.loss);
    close(original.nextGap, changed.nextGap);
    for (const key of ['user', 'positive', 'negative']) rotate(original.after[key]).forEach((value, i) => close(value, changed.after[key][i]));
  }
}
counts['BPR old-state orthogonal equivariance'] = 6;
const emptyUser = [[null, null, null], [0, 2, 4], [1, null, 5]];
close(m.neighborhoodPrediction({ matrix: emptyUser, user: 0, item: 1 }).prediction, 12 / 5);
close(m.neighborhoodPrediction({ matrix: [[null, null], [null, null]], user: 0, item: 1, prior: 2.75 }).prediction, 2.75);
counts['JS empty-user and empty-catalogue fallback'] = 2;
for (const opts of [{ order: [3, 1], grades: [2, 0, 1, 3], cutoff: 5 }, { order: [], grades: [1, 0, 3], cutoff: 2 }, { order: [1], grades: [0, 0], cutoff: 4 }]) {
  const state = m.evaluateRecommendationList(opts);
  assert(state.fillRate <= 1);
  assert(state.recall === null || state.recall <= state.bestCandidateRecall);
  assert(state.ndcg === null || state.ndcg <= 1 + 1e-12);
}
counts['short and empty slate boundaries'] = 3;
const native = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-X', 'utf8', 'scripts/verify-recommender-independent.py'], { input: JSON.stringify(examples), encoding: 'utf8', maxBuffer: 5e6 });
assert.equal(native.status, 0, native.stderr + native.stdout);
const names = ['src/learn/data/topics/recommender-systems-collaborative-filtering-matrix-factorization.jsx', 'src/learn/data/recommender-models.js', 'src/learn/data/recommender-examples.js', 'src/learn/components/lesson-labs/RecommenderLabs.jsx', 'src/learn/components/lesson-labs/RecommenderFigures.jsx', 'src/learn/components/lesson-labs/recommender-labs.css', 'src/learn/data/curriculum/blueprints/recommender-systems-collaborative-filtering-matrix-factorization.js'];
const result = { checkedAt: new Date().toISOString(), counts, native: JSON.parse(native.stdout), sources: names.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') })) };
if (!process.argv.includes('--initial')) assert(result.native.fallbackCases.every(row => row.passed && row.warnings.length === 0), 'The native advertised fallback must return its declared finite value without warnings.');
fs.mkdirSync('scratch/recommender-independent', { recursive: true });
const destination = process.argv.includes('--initial') ? 'initial-results.json' : 'numerical-results.json';
fs.writeFileSync(`scratch/recommender-independent/${destination}`, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ counts, native: result.native }));
