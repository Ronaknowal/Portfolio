import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { transform } from 'esbuild';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, actual + ' differs from ' + expected);
const checks = {
  'regularization-l1-l2-elastic-net-dropout': () => {
    assert.equal(10 - 8 + 3, 5);
    const z = [3, 4], strength = 4, length = Math.hypot(...z);
    const optimum = z.map(value => (1 - strength / length) * value);
    near(optimum[0], .6); near(optimum[1], .8);
    const cost = w => w.reduce((sum, value, i) => sum + .5 * (value - z[i]) ** 2, 0) + strength * Math.hypot(...w);
    near(cost(optimum), 12);
    assert.ok(cost([0, 0]) > cost(optimum));
    for (let angle = 0; angle < 2 * Math.PI; angle += .1) {
      assert.ok(cost([Math.cos(angle), Math.sin(angle)]) >= cost(optimum) - 1e-12);
    }
    assert.deepEqual(z.map(value => Math.max(value - strength, 0)), [0, 0]);
    assert.equal(Math.max(1 - 5 / length, 0), 0);
    near((2 / (20 * 3)) / (2 / (10 * 3)), .5);
    return ['Partial residual adds back current contribution', 'Group optimum versus coordinate threshold and zero boundary', 'Fixed-prior penalty halves when independent n doubles'];
  },
  'feature-selection-importance-shap-permutation-mutual-info': () => {
    const gini = counts => 1 - counts.reduce((sum, count) => sum + (count / counts.reduce((a, b) => a + b)) ** 2, 0);
    const gain = gini([2, 2]) - .75 * gini([2, 1]) - .25 * gini([0, 1]);
    near(gain, 1 / 6); near(.5 * gain, 1 / 12);
    const orders = ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA'];
    const predecessors = new Map();
    for (const order of orders) {
      const beforeA = [...order.slice(0, order.indexOf('A'))].sort().join('');
      predecessors.set(beforeA, (predecessors.get(beforeA) ?? 0) + 1 / 6);
    }
    near(predecessors.get(''), 1 / 3); near(predecessors.get('B'), 1 / 6);
    near(predecessors.get('C'), 1 / 6); near(predecessors.get('BC'), 1 / 3);
    near(orders.filter(order => 'AB'.includes(order.at(-1))).length / orders.length, 2 / 3);
    near(['GC', 'CG'].filter(order => order.at(-1) === 'G').length / 2, .5);
    return ['Node and globally weighted Gini gain', 'Predecessor-set weights enumerated from six orders', 'Grouped versus summed Shapley arrival counts'];
  },
  'bias-variance-tradeoff-learning-curves': () => {
    const s = [.8, .2];
    near(s.reduce((a, b) => a + b), 1);
    near(s.reduce((sum, value) => sum + value * value, 0), .68);
    const worlds = [-1, 1].flatMap(a => [-1, 1].map(b => [a, b]));
    let train = 0, fresh = 0, fitVariance = 0;
    for (const y of worlds) {
      const fit = y.map((value, i) => s[i] * value);
      train += y.reduce((sum, value, i) => sum + (fit[i] - value) ** 2, 0) / 8;
      fitVariance += fit.reduce((sum, value) => sum + value ** 2, 0) / 8;
      for (const target of worlds) fresh += target.reduce((sum, value, i) => sum + (fit[i] - value) ** 2, 0) / 32;
    }
    near(train, .34); near(fitVariance, .34); near(fresh, 1.34); near(fresh - train, 1);
    const noise = .1;
    assert.equal(noise / .01, 10); assert.equal(noise / 1, .1);
    return ['Smoother trace and squared-amplitude distinction', 'Exact independent-noise enumeration of train/fresh risk', 'Small singular value amplifies a fixed noise coordinate'];
  },
  'imbalanced-learning-smote-cost-sensitive-learning': () => {
    const minority = [0, 1, 10], majority = [.5, 5, 20, 30, 40];
    const distances = majority.map(candidate => minority.map(value => Math.abs(candidate - value)));
    const nearest = distances.map(values => Math.min(...values));
    const farthest = distances.map(values => Math.max(...values));
    assert.deepEqual(nearest, [.5, 4, 10, 20, 30]);
    assert.deepEqual(farthest, [9.5, 5, 20, 30, 40]);
    assert.equal(majority[nearest.indexOf(Math.min(...nearest))], .5);
    assert.equal(majority[farthest.indexOf(Math.min(...farthest))], 5);
    assert.deepEqual([.2, .8].map(value => 10 * value / (.2 + .8)), [2, 8]);
    const likelihoodRatio = 3, sourcePrior = .5, destinationPrior = .1;
    const sourceOdds = likelihoodRatio * sourcePrior / (1 - sourcePrior);
    const correctedOdds = sourceOdds * (destinationPrior / (1 - destinationPrior)) / (sourcePrior / (1 - sourcePrior));
    near(correctedOdds / (1 + correctedOdds), .25);
    return ['All-candidate NearMiss nearest/farthest outcomes', 'Normalized ADASYN allocation', 'Prior-shift odds preserve the likelihood multiplier'];
  },
  'automl-neural-architecture-search-nas': () => {
    assert.equal(12 * 8 + 8, 104); assert.equal(8 + 12, 20);
    const mixed = .5 * 1 + .5 * -1;
    assert.equal(.5 * mixed ** 2, 0); assert.equal(.5 * 1 ** 2, .5); assert.equal(.5 * (-1) ** 2, .5);
    const probability = .3, sharedOutput = 2;
    assert.equal(probability * (sharedOutput - (probability * sharedOutput + (1 - probability) * sharedOutput)), 0);
    const codes = [[1, 1, 0], [1, 0, 1]];
    const augmented = codes.map(code => [...code, ...code.map(bit => 1 - bit)]);
    const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
    const gram = augmented.map(a => augmented.map(b => dot(a, b)));
    assert.deepEqual(gram, [[3, 1], [1, 3]]);
    assert.equal(gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0], 8);
    assert.equal(Math.log(3 * 3 - 3 * 3), -Infinity);
    return ['Executable shape and affine parameter count', 'Mixture relaxation gap and equal-output derivative', 'NASWOT Gram determinant and singular boundary'];
  },
  'hidden-markov-models-hmm': () => {
    const source = [.4, .35, .25], transition = [[0, .5, .5], [1, 0, 0], [1, 0, 0]];
    const destination = source.map((_, j) => source.reduce((sum, value, i) => sum + value * transition[i][j], 0));
    near(source[0] * destination[0], .24);
    assert.equal(source[0] * transition[0][0], 0);
    const density = sigma => 1 / (sigma * Math.sqrt(2 * Math.PI));
    const posterior = (a, b) => a / (a + b);
    near(posterior(density(1), density(2)), 2 / 3);
    near(posterior(density(100), density(200)), 2 / 3);
    near(density(100) / density(1), .01);
    const worlds = [0, 1].flatMap(a => [0, 1].map(b => ({ a, b, prior: .25, supported: a + b === 1 })));
    const evidence = worlds.filter(world => world.supported).reduce((sum, world) => sum + world.prior, 0);
    const joint = worlds.map(world => world.supported ? world.prior / evidence : 0);
    assert.deepEqual(joint, [0, .5, .5, 0]);
    near(joint[2] + joint[3], .5); near(joint[1] + joint[3], .5);
    assert.notEqual(joint[3], (joint[2] + joint[3]) * (joint[1] + joint[3]));
    return ['Constrained adjacent-state joint versus marginal product', 'Gaussian posterior invariance under unit conversion', 'Shared-meter evidence induces posterior dependence'];
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply exact topic IDs.');
for (const topicId of selected) {
  assert.ok(checks[topicId], 'Unknown scoped topic: ' + topicId);
  const sourceFiles = [
    'src/learn/data/topics/' + topicId + '.jsx',
    'docs/teaching/concept-intuition/' + topicId + '/review.md',
    'scripts/verify-model-selection-sequence-intuition.mjs',
  ];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  await transform(await readFile(sourceFiles[0], 'utf8'), { loader: 'jsx', jsx: 'automatic', target: 'es2022' });
  const passed = checks[topicId]();
  await writeFile('docs/teaching/concept-intuition/' + topicId + '/author-checks.json', JSON.stringify({
    topicId, reviewedAt: new Date().toISOString(), sourceFiles, sourceHashes,
    checks: [{ name: 'Complete lesson reading and local concept map', status: 'author-reviewed', evidence: sourceFiles[1] },
      { name: 'JSX syntax transform', status: 'passed' }, ...passed.map(name => ({ name, status: 'passed' }))],
    unchangedEvidence: 'Existing models, native programs, datasets and run outputs retained; no fresh native-fit campaign claimed.',
    independentReview: 'pending', browserReview: 'pending root inspection', integration: 'pending',
  }, null, 2) + '\n');
  process.stdout.write(topicId + ': JSX and ' + passed.length + ' scoped arithmetic groups passed\n');
}
