import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { transform } from 'esbuild';

const checks = {
  'anomaly-outlier-detection-isolation-forest-one-class-svm-lof': () => {
    assert.ok(Math.abs(2 ** -.5 - .7071067811865476) < 1e-14);
    assert.equal(2 ** -1, .5);
    assert.equal(2 ** -2, .25);
    const cap = 1 / (8 * .5);
    assert.equal(cap * 3, .75);
    assert.equal(cap * 4, 1);
    assert.equal((.2 / 10) / (.1 / 10), .2 / .1);
    return ['Normalized path-score values', 'Support-weight cap and incomplete budget', 'LOF common-scale cancellation'];
  },
  'gaussian-mixture-models-gmm-em-algorithm': () => {
    const joint = [.3, .1], total = joint.reduce((a, b) => a + b);
    const posterior = joint.map(value => value / total);
    const bound = q => q.reduce((sum, share, i) => sum + share * Math.log(joint[i] / share), 0);
    assert.ok(Math.abs(bound(posterior) - Math.log(total)) < 1e-14);
    assert.equal(bound([.5, .5]).toFixed(4), '-1.0601');
    assert.ok(bound([.5, .5]) < Math.log(total));
    assert.equal(.5 * (1 + (-2) ** 2) + .5 * (1 + 2 ** 2), 5);
    assert.equal((1 + 1) / 4, .5);
    assert.equal(1 - .5 ** 2 + (.5 - 2) ** 2, 3);
    return ['Posterior-tight and arbitrary Jensen bounds', 'Mixture versus independent-average variance', 'Conditional within/between variance'];
  },
  't-sne-umap-manifold-learning': () => {
    const affinities = (distances, sigma) => {
      const values = distances.map(d => Math.exp(-d * d / (2 * sigma * sigma)));
      const total = values.reduce((a, b) => a + b);
      return values.map(value => value / total);
    };
    assert.deepEqual(affinities([1, 2, 3], 1), affinities([10, 20, 30], 10));
    const q = ys => {
      const weights = ys.flatMap((y, i) => ys.map((z, j) => i === j ? 0 : 1 / (1 + (y - z) ** 2)));
      return weights[1] / weights.reduce((a, b) => a + b);
    };
    assert.ok(Math.abs(q([0, 1, 3]) - .3125) < 1e-14);
    assert.ok(Math.abs(q([0, 1, 2]) - 5 / 24) < 1e-14);
    assert.ok(-Math.log(.8) < -Math.log(.2));
    assert.ok(-Math.log(1 - .8) > -Math.log(1 - .2));
    return ['Local affinity scale cancellation', 'Global ordered-pair denominator at both layouts', 'Attractive and repulsive endpoint losses'];
  },
  'independent-component-analysis-ica': () => {
    const source = [-1, 1];
    const diagonal = source.flatMap(a => source.map(b => (a + b) / Math.sqrt(2)));
    const moment = (values, order) => values.reduce((sum, value) => sum + value ** order, 0) / values.length;
    assert.equal(moment(source, 2), 1);
    assert.ok(Math.abs(moment(diagonal, 2) - 1) < 1e-14);
    assert.equal(moment(source, 4) - 3, -2);
    assert.ok(Math.abs(moment(diagonal, 4) - 3 + 1) < 1e-14);
    const w = [Math.cos(.3), Math.sin(.3)];
    const points = source.flatMap(a => source.map(b => [a, b]));
    const linearUpdate = w.map((wi, i) => points.reduce((sum, z) => sum + z[i] * (z[0] * w[0] + z[1] * w[1]), 0) / 4 - wi);
    assert.ok(linearUpdate.every(value => Math.abs(value) < 1e-14));
    assert.equal(.5 * 2, 1);
    return ['Binary and diagonal projection moments', 'Whitened linear-contrast cancellation', 'Jacobian volume and probability preservation'];
  },
  'non-negative-matrix-factorization-nmf': () => {
    const loss = h => h * h - 3 * h + 2.5;
    const bound = h => .5 - (h - 1) + 1.5 * (h - 1) ** 2;
    for (let i = 0; i <= 80; i++) {
      const h = i / 40;
      assert.ok(Math.abs(bound(h) - loss(h) - .5 * (h - 1) ** 2) < 1e-14);
    }
    assert.equal(loss(1), bound(1));
    assert.ok(Math.abs(loss(4 / 3) - 5 / 18) < 1e-14);
    assert.ok(Math.abs(bound(4 / 3) - 1 / 3) < 1e-14);
    const patterns = [[1, 0], [1, 1]];
    const row = [1, 1];
    const dots = patterns.map(pattern => pattern.reduce((sum, value, i) => sum + value * row[i], 0));
    assert.deepEqual(dots, [1, 2]);
    assert.deepEqual(row.map((_, i) => dots.reduce((sum, value, k) => sum + value * patterns[k][i], 0)), [3, 2]);
    assert.equal((2 * 1 + 4 * 2) / (1 + 4), 2);
    assert.equal((10 + 6 * 3) / (5 + 9), 2);
    return ['Touching upper bound and coordinate update', 'Nonorthogonal dictionary transform counterexample', 'Streaming sufficient-statistic update'];
  },
  'feature-scaling-encoding-imputation': () => {
    const angle = Math.PI / 18;
    const chord = Math.hypot(Math.cos(angle) - Math.cos(-angle), Math.sin(angle) - Math.sin(-angle));
    assert.ok(Math.abs(chord - 2 * Math.sin(angle)) < 1e-14);
    assert.equal(chord.toFixed(4), '0.3473');
    assert.ok(Math.abs(Math.sin(angle) - Math.sin(Math.PI - angle)) < 1e-14);
    assert.ok(Math.cos(angle) > 0 && Math.cos(Math.PI - angle) < 0);
    assert.equal((1 + 2 * .5) / (1 + 2), 2 / 3);
    assert.ok(100 / 102 > 1 / 3);
    assert.equal((0 ** 2 + 2 ** 2) / 2, 2);
    assert.equal(((0 + 2) / 2) ** 2, 1);
    return ['Cyclic seam distance and sine-only collision', 'Target-encoding shrinkage weights', 'Pooling and nonlinear analysis do not commute'];
  },
  'cross-validation-hyperparameter-tuning': () => {
    const averageVariance = (variance, correlation, folds) => variance * (1 + (folds - 1) * correlation) / folds;
    assert.equal(averageVariance(1, 0, 4), .25);
    assert.equal(averageVariance(1, .5, 4), .625);
    assert.equal(.5 + .5 / 4, .625);
    assert.equal(averageVariance(1, 1, 4), 1);
    const ratio = (l, g) => 1 / (.25 + .75 * g / l);
    assert.ok(Math.abs(ratio(.4, .1) - 16 / 7) < 1e-14);
    assert.ok(Math.abs(ratio(.2, .4) - 4 / 7) < 1e-14);
    assert.equal(ratio(.4, .1) / ratio(.2, .4), 4);
    const sampled = [0, 0, 2, 2], distinct = new Set(sampled);
    assert.equal(distinct.size, 2);
    assert.deepEqual([0, 1, 2, 3].filter(id => !distinct.has(id)), [1, 3]);
    return ['Shared and independent averaging variance', 'TPE acquisition density-ratio comparison', 'Bootstrap sampled versus distinct and out-of-bag identities'];
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply the exact topic IDs to review.');
for (const topicId of selected) {
  assert.ok(checks[topicId], 'No scoped independent calculation registered for ' + topicId);
  const sourceFiles = [
    'src/learn/data/topics/' + topicId + '.jsx',
    'docs/teaching/concept-intuition/' + topicId + '/review.md',
    'scripts/verify-representation-concept-intuition.mjs',
  ];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  await transform(await readFile(sourceFiles[0], 'utf8'), { loader: 'jsx', jsx: 'automatic', target: 'es2022' });
  const passed = checks[topicId]();
  const record = {
    topicId, reviewedAt: new Date().toISOString(), sourceFiles, sourceHashes,
    checks: [{ name: 'Complete lesson reading and local concept map', status: 'author-reviewed', evidence: sourceFiles[1] },
      { name: 'JSX syntax transform', status: 'passed' },
      ...passed.map(name => ({ name, status: 'passed' }))],
    unchangedEvidence: 'Existing numerical models, native programs, datasets and fit campaigns retained; no claim of a fresh native campaign.',
    independentReview: 'pending', browserReview: 'pending root inspection', integration: 'pending',
  };
  await writeFile('docs/teaching/concept-intuition/' + topicId + '/author-checks.json', JSON.stringify(record, null, 2) + '\n');
  process.stdout.write(topicId + ': JSX and ' + passed.length + ' independent arithmetic groups passed\n');
}
