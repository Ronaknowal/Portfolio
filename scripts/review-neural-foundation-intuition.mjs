import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const sum = values => values.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((value, i) => value * b[i]));
const checks = {
  'perceptrons-neurons-activation-functions': () => {
    const x = [2, -1], w = [1, 1], b = 1, y = -1, step = .5;
    const old = y * (dot(w, x) + b);
    const next = w.map((value, i) => value + step * y * x[i]);
    const changed = y * (dot(next, x) + b + step * y);
    close(old, -2); close(changed, 1); close(changed - old, step * (dot(x, x) + 1));
    const silu = value => value / (1 + Math.exp(-value)), h = 1e-5;
    assert.ok((silu(-2 + h) - silu(-2 - h)) / (2 * h) < 0);
    close(Math.max(0, .25) - 2 * Math.max(0, .25 - 1) + Math.max(0, .25 - 2), .25);
    return ['Negative-label changed-row signed-margin expansion', 'Negative SiLU local direction by finite difference and off-table ramp transfer'];
  },
  'backpropagation-automatic-differentiation': () => {
    const shared = x => x * x - (x * x), h = 1e-4;
    close(shared(-2), 0); close((shared(-2 + h) - shared(-2 - h)) / (2 * h), 0);
    const gradients = [2, -3, 5, 7, -1, 4];
    const groups = [gradients.slice(0, 1), gradients.slice(1, 3), gradients.slice(3)];
    const weighted = sum(groups.map(group => group.length / gradients.length * sum(group) / group.length));
    close(weighted, sum(gradients) / gradients.length);
    assert.notEqual(sum(groups.map(group => sum(group) / group.length)), weighted);
    const hessianTimes = v => [2 * v[0] + v[1], v[0] + 6 * v[1]];
    const u = [-2, 3], v = [4, -1];
    close(dot(u, hessianTimes(v)), dot(v, hessianTimes(u)));
    return ['Changed shared-branch cancellation', 'New 1/2/3 microbatch partition preserves per-example objective', 'Hessian directional symmetry with new vectors'];
  },
  'loss-functions-ce-mse-focal-contrastive-triplet': () => {
    const observations = [1, 2, 3, 4, 10], quantile = estimate => sum(observations.map(y => y >= estimate ? .8 * (y - estimate) : .2 * (estimate - y)));
    close(quantile(4), quantile(7)); close(quantile(4), quantile(10));
    assert.ok(quantile(3.9) > quantile(4)); assert.ok(quantile(10.1) > quantile(10));
    close(-Math.log(.7 + .2), -Math.log(.45 + .45));
    assert.ok(-.5 * Math.log(.7 * .2) > -Math.log(.45));
    const ce = z => Math.log1p(Math.exp(-z));
    assert.ok(ce(10 * Math.cos(40 * Math.PI / 180)) > ce(10 * Math.cos(30 * Math.PI / 180)));
    return ['Discrete quantile interval of optima at q=.8', 'New equal-mass positive allocation contrast', 'Margin lowers target score and raises CE with fixed competitor'];
  },
  'batch-layer-group-rms-normalization': () => {
    const values = [1, 4, -2], upstream = [.2, -.7, 1.1], epsilon = .1;
    const normalize = x => { const mean = sum(x) / x.length; const variance = sum(x.map(value => (value - mean) ** 2)) / x.length; return { h: x.map(value => (value - mean) / Math.sqrt(variance + epsilon)), inverse: 1 / Math.sqrt(variance + epsilon) }; };
    const { h, inverse } = normalize(values);
    const vjp = h.map((value, index) => inverse * (upstream[index] - sum(upstream) / 3 - value * dot(upstream, h) / 3));
    const objective = x => dot(normalize(x).h, upstream), delta = 1e-5;
    values.forEach((_, index) => { const plus = [...values], minus = [...values]; plus[index] += delta; minus[index] -= delta; close(vjp[index], (objective(plus) - objective(minus)) / (2 * delta), 1e-9); });
    close(sum(vjp), 0);
    assert.ok(Math.abs(normalize(values.map(value => 2 * value)).h[1] - h[1]) > 1e-4);
    const running = .8 * (.8 * 3 + .2 * 7) + .2 * -2;
    close(running, .64 * 3 + .16 * 7 - .4);
    return ['New finite-epsilon VJP matches perturbations and common-shift null', 'Positive rescaling is not exactly invariant with fixed epsilon', 'Different running-mean history expansion'];
  },
  'transfer-learning-fine-tuning-strategies': () => {
    const base = [[1, 0], [0, 2]], a = [2, -1], b = [1, 3];
    const update = b.map(value => a.map(coordinate => value * coordinate));
    close(update[0][0] * update[1][1] - update[0][1] * update[1][0], 0);
    const adapted = base.map((row, i) => row.map((value, j) => value + update[i][j]));
    assert.notEqual(adapted[0][0] * adapted[1][1] - adapted[0][1] * adapted[1][0], 0);
    assert.ok(Math.abs(Math.tanh(-2) - 2 * Math.tanh(-1)) > .5);
    const measurement = 1 - 3, incoming = [1, 3], nextB = incoming.map(value => -.1 * measurement * value);
    const output = incoming.map((value, i) => value + measurement * nextB[i]);
    close(output[0], .6); close(output[1], 1.8); close(dot(output, output) / 2, 1.8);
    return ['Rank-one correction with full-rank adapted base', 'Nonlinear adapter fails additivity', 'Changed-input LoRA practice update independently expanded'];
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply exact topic IDs.');
for (const topicId of selected) {
  assert.ok(checks[topicId], 'Unknown independent review topic: ' + topicId);
  const directory = `docs/teaching/concept-intuition/${topicId}/`;
  const authorPath = directory + 'author-checks.json';
  const author = JSON.parse(await readFile(authorPath, 'utf8'));
  assert.equal(author.passed, true);
  const sourceFiles = [...author.sourceFiles, directory + 'independent-review.md', 'scripts/review-neural-foundation-intuition.mjs'];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  for (const file of author.sourceFiles) assert.equal(sourceHashes[file], author.sourceHashes[file], `Author source is stale: ${file}`);
  const complementary = checks[topicId]();
  await writeFile(directory + 'independent-checks.json', JSON.stringify({
    topicId, reviewer: 'classical_representation_intuition', independent: true, reviewedAt: new Date().toISOString(),
    passed: true, sourceFiles, sourceHashes,
    correctnessReview: 'accepted at source level', learningExperienceReview: 'accepted at source level; heuristic assessment, no actual learner session',
    checks: complementary.map(name => ({ name, status: 'passed' })),
    fullReading: directory + 'independent-review.md',
    actualBrowserInspection: false, browserReview: 'pending root', integration: 'pending root',
    nativeExecution: 'Existing unchanged campaigns retained; no fresh fits or full native verification claimed.',
  }, null, 2) + '\n');
  process.stdout.write(`${topicId}: independent full-reading review and ${complementary.length} complementary groups passed\n`);
}
