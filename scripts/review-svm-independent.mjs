// Bounded complementary invariances, independent of the author's QP/library suite.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { marginGeometry, svmKernel, svmPairFixture, svmPairStep, svrTubeState, spectrumCounts } from '../src/learn/data/support-vector-machines-models.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
const cases = [];
for (const [angle, offset] of [[-31, -.4], [17, .25], [43, .7]]) {
  const original = marginGeometry(angle, offset, .5);
  const scaled = marginGeometry(angle, offset, 3);
  original.rows.forEach((row, index) => {
    close(scaled.rows[index].score, 6 * row.score);
    close(scaled.rows[index].signedDistance, row.signedDistance);
    row.projection.forEach((value, coordinate) => close(scaled.rows[index].projection[coordinate], value));
  });
  cases.push({ kind: 'positive score scaling preserves distances and projections', angle, offset });
}
for (const [left, right] of [[[1, -2], [-1, 1]], [[0, .3], [.4, -.8]], [[2, 2], [-2, -1]]]) {
  const rotated = ([x, y]) => [(x - y) / Math.SQRT2, (x + y) / Math.SQRT2];
  close(svmKernel(rotated(left), rotated(right), 'rbf', .7), svmKernel(left, right, 'rbf', .7));
  close(svmKernel(left.map(x => 2 * x), right.map(x => 2 * x), 'rbf', .175), svmKernel(left, right, 'rbf', .7));
  cases.push({ kind: 'RBF rotation and isotropic units transformation', left, right });
}
const fixture = svmPairFixture();
for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) {
  const forward = svmPairStep(fixture.points, fixture.labels, fixture.alpha, fixture.c, i, j);
  const reverse = svmPairStep(fixture.points, fixture.labels, fixture.alpha, fixture.c, j, i);
  const shifted = svmPairStep(fixture.points.map(([x, y]) => [x + 2, y - 1]), fixture.labels, fixture.alpha, fixture.c, i, j);
  forward.after.forEach((value, index) => {
    close(reverse.after[index], value);
    close(shifted.after[index], value);
  });
  close(reverse.afterDual, forward.afterDual);
  close(shifted.afterDual, forward.afterDual);
  cases.push({ kind: 'pair reparameterization and feature translation preserve the feasible update', i, j });
}
const flat = svmPairStep([[0, 0], [1, 0], [1, 0]], [-1, 1, 1], [.6, .2, .4], 1, 1, 2);
assert.equal(flat.q, 0);
assert.equal(flat.g, 0);
assert.equal(flat.bestDelta, 0);
assert.deepEqual(flat.after, flat.alpha);
cases.push({ kind: 'flat zero-gain duplicate pair follows the stated leave-unchanged policy' });

for (const [amplitude, epsilon, c] of [[3, .8, .7], [2, 1.2, 1], [3, .2, 2]]) {
  const original = svrTubeState(amplitude, epsilon, c);
  const transformed = svrTubeState(amplitude / 2, epsilon / 2, c / 2);
  close(transformed.slope, original.slope / 2);
  close(transformed.objective, original.objective / 4);
  cases.push({ kind: 'SVR target-unit change preserves the objective up to scale', amplitude, epsilon, c });
}
for (const sequence of ['ACACCA', 'AAAA', '']) {
  for (const size of [1, 2, 3]) {
    const forward = spectrumCounts(sequence, size);
    const reverse = spectrumCounts([...sequence].reverse().join(''), size);
    assert.equal(forward.windows.length, Math.max(0, sequence.length - size + 1));
    for (const [word, count] of Object.entries(forward.counts)) assert.equal(reverse.counts[[...word].reverse().join('')], count);
  }
  cases.push({ kind: 'reversing sequence reindexes spectrum counts without inventing windows', sequence });
}
const source = 'src/learn/data/support-vector-machines-models.js';
fs.mkdirSync('scratch/svm-independent', { recursive: true });
fs.writeFileSync('scratch/svm-independent/complementary-checks.json', JSON.stringify({ checkedAt: new Date().toISOString(), source, sha256: createHash('sha256').update(fs.readFileSync(source)).digest('hex'), cases }, null, 2) + '\n');
console.log(`PASS: ${cases.length} complementary SVM cases.`);
