// Independent source review and complementary calculations. Rendering is separate.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const sum = values => values.reduce((a, b) => a + b, 0);
const sigmoid = value => 1 / (1 + Math.exp(-value));
const checks = {
  'bayesian-networks-causal-graphical-models': () => {
    // Different potentials from the lesson: separator values must stay distinct.
    const phi = [[1, 4], [3, 2]], psi = [[2, 5], [3, 1]];
    const message = [0, 1].map(b => phi[0][b] + phi[1][b]);
    const masses = [0, 1].map(c => sum(message.map((m, b) => m * psi[b][c])));
    const brute = [0, 1].map(c => sum([0, 1].flatMap(a => [0, 1].map(b => phi[a][b] * psi[b][c]))));
    assert.deepEqual(masses, [26, 26]); assert.deepEqual(masses, brute);
    // Common evidence: A=B xor E, prior B/E independent, then condition on A=1.
    const mass = (b, e) => (b ? .2 : .8) * (e ? .3 : .7) * +(b !== e);
    const partition = sum([0, 1].flatMap(b => [0, 1].map(e => mass(b, e))));
    close(mass(1, 0) / partition, .14 / .38);
    close(mass(1, 1) / (mass(0, 1) + mass(1, 1)), 0);
    return ['Changed separator factors agree with exhaustive joint enumeration', 'Changed collider fixture creates dependence after observing common effect'];
  },
  'conditional-random-fields-crf': () => {
    const factor = 2.75, mass = [3, 6 * factor, 1, 2], z = sum(mass);
    close((mass[0] + mass[1]) / z, (3 + 6 * factor) / (6 + 6 * factor));
    const conditionalAB = (mass[1] + mass[3]) / z * (3 * factor) / (3 * factor + 1);
    close(conditionalAB, mass[1] / z);
    assert.ok(Math.abs((mass[0] + mass[1]) / z * (mass[1] + mass[3]) / z - conditionalAB) > .01);
    const loss = w => Math.log(6 + 6 * Math.exp(w)) - Math.log(3 + 6 * Math.exp(w));
    const w = Math.log(factor), h = 1e-5;
    close((loss(w + h) - loss(w - h)) / (2 * h), 6 * factor / z - 6 * factor / (3 + 6 * factor), 1e-9);
    return ['Off-table partial label likelihood', 'Changed backward conditional recovers joint but independent marginals do not', 'Partial-label gradient by finite differences'];
  },
  'gaussian-processes-gp': () => {
    const c = Math.exp(-2 / 9), a = 1.25;
    close((a + c) * (a - c), a * a - c * c);
    const solve = [(a + c) / (a * a - c * c), -(a + c) / (a * a - c * c)];
    close(solve[0] - solve[1], 2 / (a - c));
    // Changed inducing location: conditional-prior residual vanishes there only.
    const inducing = .5, inputs = [-1, .5, 2];
    const residual = inputs.map(x => 1 - Math.exp(-((x - inducing) ** 2)));
    close(residual[1], 0); close(residual[0], residual[2]);
    assert.ok(residual[0] > 0);
    const periodic = d => Math.exp(-2 * Math.sin(Math.PI * d) ** 2);
    close(periodic(3), 1); close(periodic(3) * Math.exp(-9 / 8), Math.exp(-9 / 8));
    return ['Direct inverse and eigenbasis residual cost agree', 'Shifted inducing point retains unexplained variance away from itself', 'Off-table recurrence and persistence covariance'];
  },
  'semi-supervised-learning-label-propagation-self-training-co-training': () => {
    const weights = [2, 3], degrees = [2, 5, 3], values = [2, -1, .5];
    const normalized = values.map((value, i) => value / Math.sqrt(degrees[i]));
    const edgeEnergy = sum(weights.map((weight, i) => weight * (normalized[i] - normalized[i + 1]) ** 2));
    const quadratic = sum(values.map(value => value * value)) - 2 * sum(weights.map((weight, i) => weight * normalized[i] * normalized[i + 1]));
    close(edgeEnergy, quadratic);
    const nullVector = degrees.map(Math.sqrt);
    close(sum(weights.map((weight, i) => weight * (nullVector[i] / Math.sqrt(degrees[i]) - nullVector[i + 1] / Math.sqrt(degrees[i + 1])) ** 2)), 0);
    const strongLogit = -.7, h = 1e-5, loss = z => -Math.log(sigmoid(z));
    close((loss(strongLogit + h) - loss(strongLogit - h)) / (2 * h), sigmoid(strongLogit) - 1, 1e-9);
    close((loss(strongLogit) + 0 + 0) / 3, loss(strongLogit) / 3);
    return ['Unequal-degree normalized energy equals matrix quadratic', 'Degree square-root null direction with changed graph', 'Changed strong-view gradient and three-item full-batch denominator'];
  },
  'active-learning': () => {
    const own = [4, 1], targetCovariance = [.2, .8];
    assert.ok(1 - targetCovariance[0] ** 2 / own[0] - targetCovariance[1] ** 2 / own[1] > 0);
    const firstReduction = targetCovariance[0] ** 2 / (own[0] + .25);
    const secondReduction = targetCovariance[1] ** 2 / (own[1] + .25);
    assert.ok(firstReduction < secondReduction && own[0] > own[1]);
    const gradient = z => [...z.map(value => -.3 * value), ...z.map(value => .3 * value)];
    const vectors = [[-1, 2], [3, 1]], embedded = vectors.map(gradient);
    close(sum(embedded[0].map((value, i) => (value - embedded[1][i]) ** 2)), .18 * sum(vectors[0].map((value, i) => (value - vectors[1][i]) ** 2)));
    const probability = [.15, .25, .1, .5], losses = [1, 0, .5, 1];
    close(sum(losses.map((loss, i) => probability[i] * loss / (4 * probability[i]))), sum(losses) / 4);
    return ['Three-variable covariance validity and target-specific acquisition reversal', 'New probability/features preserve gradient-distance scaling', 'Changed nonuniform query probabilities recover fixed-model uniform risk'];
  },
  'evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae': () => {
    const outcomes = [0, 1, 2, 7, 15];
    const squared = c => sum(outcomes.map(y => (y - c) ** 2));
    const absolute = c => sum(outcomes.map(y => Math.abs(y - c)));
    const quantile = c => sum(outcomes.map(y => y >= c ? .9 * (y - c) : .1 * (c - y)));
    assert.ok(squared(5) < squared(4.99) && squared(5) < squared(5.01));
    assert.ok(absolute(2) < absolute(1.99) && absolute(2) < absolute(2.01));
    assert.ok(quantile(15) < quantile(14.99) && quantile(15) < quantile(15.01));
    const y = [0, 1, 1, 0, 0, 1], prediction = [0, 0, 1, 0, 1, 1];
    const mean = values => sum(values) / values.length;
    const correlation = mean(y.map((value, i) => (value - mean(y)) * (prediction[i] - mean(prediction)))) / Math.sqrt(mean(y.map(value => (value - mean(y)) ** 2)) * mean(prediction.map(value => (value - mean(prediction)) ** 2)));
    close(correlation, (2 * 2 - 1 * 1) / Math.sqrt(3 * 3 * 3 * 3));
    const perClass = [16 / 21, .5, 0];
    close(sum(perClass.map((f1, i) => f1 * [10, 4, 2][i])) / 16, 101 / 168);
    return ['Changed empirical mean/median/quantile minima', 'MCC equals correlation on changed binary rows', 'Aggregation reconstructed from per-class support'];
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply exact topic IDs.');
for (const topicId of selected) {
  assert.ok(checks[topicId], `Unknown topic ${topicId}`);
  const directory = `docs/teaching/concept-intuition/${topicId}/`;
  const author = JSON.parse(await readFile(directory + 'author-checks.json', 'utf8'));
  assert.ok(author.checks.length > 0 && author.checks.every(check => check.passed === true || check.status === 'passed'));
  const sourceFiles = [...author.sourceFiles, directory + 'independent-review.md', 'scripts/review-probabilistic-learning-intuition.mjs'];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  for (const file of author.sourceFiles) assert.equal(sourceHashes[file], author.sourceHashes[file], `Stale author source: ${file}`);
  const result = checks[topicId]();
  await writeFile(directory + 'independent-checks.json', JSON.stringify({ topicId, reviewer: 'classical_representation_intuition', independent: true, reviewedAt: new Date().toISOString(), passed: true, sourceFiles, sourceHashes, checks: result.map(name => ({ name, status: 'passed' })), correctnessReview: 'accepted at source level', learningExperienceReview: 'accepted at source level; heuristic review, not an actual learner trial', fullReading: directory + 'independent-review.md', actualBrowserInspection: false, browserReview: 'root records separately', nativeExecution: 'Unchanged historical campaigns retained; no new model-fitting claim.' }, null, 2) + '\n');
  process.stdout.write(`${topicId}: ${result.length} independent groups passed\n`);
}
