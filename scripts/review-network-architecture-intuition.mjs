// Independent complementary arithmetic and source binding; browser review is separate.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const sum = values => values.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((value, i) => value * b[i]));
const sigmoid = value => 1 / (1 + Math.exp(-value));
const checks = {
  'weight-initialization-xavier-kaiming-p': () => {
    const input = [2, -3], rate = .02;
    const gradient = input.map(value => -value);
    close(dot(input, gradient.map(value => -rate * value)), .26);
    const changes = [-1, 1].flatMap(a => [-1, 1].map(b => rate * dot(input, [a, b])));
    close(Math.sqrt(sum(changes.map(value => value ** 2)) / changes.length), rate * Math.sqrt(13));
    // A nonzero mean is why variance alone does not give the next weighted-sum scale.
    const signal = [1, 3], mean = sum(signal) / 2;
    close(sum(signal.map(value => value ** 2)) / 2, 5);
    close(sum(signal.map(value => (value - mean) ** 2)) / 2, 1);
    const centered = [-2, 2];
    close(sum(centered.map(value => (value / 2) ** 2)) / 2, 1);
    // Tall orthonormal columns preserve input norm, not every output cotangent.
    const matrix = [[1, 0], [0, 1], [0, 0]], x = [3, -4], output = matrix.map(row => dot(row, x));
    close(dot(output, output), dot(x, x));
    assert.deepEqual([0, 1].map(j => sum(matrix.map((row, i) => row[j] * [0, 0, 2][i]))), [0, 0]);
    return ['Changed non-unit inputs distinguish aligned updates from independent random update RMS', 'Nonzero-mean second moment and changed LSUV rescaling', 'Rectangular orthogonality preserves only the appropriate subspace'];
  },
  'residual-connections-skip-connections': () => {
    const gate = x => sigmoid(2 * x), y = x => gate(x) * x * x + (1 - gate(x)) * x;
    const x = .7, t = gate(x), derivative = (1 - t) + t * 2 * x + 2 * t * (1 - t) * (x * x - x), h = 1e-5;
    close((y(x + h) - y(x - h)) / (2 * h), derivative);
    assert.ok(Math.abs(derivative - ((1 - t) + t * 2 * x)) > .05);
    const memory = k => 36 / k + k;
    assert.equal(memory(6), 12);
    for (const k of [1, 2, 3, 4, 9, 12, 18, 36]) assert.ok(memory(k) > memory(6));
    const p = [[1, 2], [-1, 0]], branch = [[.5, 0], [0, -.25]], upstream = [3, -2];
    const objective = v => dot(upstream, p.map((row, i) => dot(row, v) + dot(branch[i], v)));
    const at = [1.2, -.4];
    for (let j = 0; j < 2; j++) {
      const plus = [...at], minus = [...at]; plus[j] += h; minus[j] -= h;
      close((objective(plus) - objective(minus)) / (2 * h), sum(upstream.map((value, i) => value * (p[i][j] + branch[i][j]))));
    }
    return ['Nonlinear input-dependent highway gate finite difference includes the gate derivative', 'Changed checkpoint-memory balance at 36 layers', 'Changed projection skip plus branch obeys the transposed-Jacobian pullback'];
  },
  'dropout-droppath-stochastic-depth': () => {
    const mean = [.2, -.4], variance = [.09, .16], x = [2, -1], other = [1, 3];
    close(dot(mean, x), .8); close(dot(variance, x.map(value => value ** 2)), .52);
    close(sum(variance.map((value, j) => value * x[j] * other[j])), -.3);
    // Independent local draws have zero cross-example covariance; the shared-weight
    // construction above need not, although each marginal variance can match.
    const predictions = [-2, 1, 4], noiseVariance = [.25, .5, .75], average = sum(predictions) / 3;
    close(sum(predictions.map((value, i) => noiseVariance[i] + (value - average) ** 2)) / 3, 6.5);
    const features = [2, 3], matrix = [[1, 2], [-2, 1]], q = .5;
    const activationMasked = matrix.map(row => dot(row, [features[0] / q, 0]));
    const connectionMasked = [matrix[0][0] * features[0] / q, matrix[1][1] * features[1] / q];
    assert.deepEqual(activationMasked, [4, -8]); assert.deepEqual(connectionMasked, [4, 6]);
    const a = .3, b = .8, f = 2, g = -1;
    close(a * f + (1 - a) * g, -.1);
    assert.notEqual(b, a); close(b + (1 - b), 1);
    return ['Changed Gaussian inputs preserve marginals while exposing nonzero shared-weight covariance', 'Changed mixture total variance does not divide model variance by draw count', 'Changed DropConnect and activation masks have distinct effective matrices', 'Shake-Shake forward and backward coefficients remain separate'];
  },
  'convolution-pooling-receptive-fields': () => {
    const n = 7, m = 4, values = [2, -1, 0, 5, 3, 4, -2], upstream = [1, -2, .5, 3];
    const matrix = Array.from({ length: m }, (_, i) => {
      const start = Math.floor(i * n / m), end = Math.floor(((i + 1) * n + m - 1) / m);
      return Array.from({ length: n }, (_, j) => +(j >= start && j < end) / (end - start));
    });
    const output = matrix.map(row => dot(row, values));
    const pullback = values.map((_, j) => sum(matrix.map((row, i) => upstream[i] * row[j])));
    close(dot(output, upstream), dot(values, pullback));
    assert.ok(matrix[0][1] > 0 && matrix[1][1] > 0);
    for (const gamma of [-3, 0, 2]) {
      const w = [2, -.5], patch = [-1, 4], bias = .7, mu = -.2, variance = 3.99, eps = .01, beta = 1.2;
      const alpha = gamma / Math.sqrt(variance + eps);
      close(gamma * (dot(w, patch) + bias - mu) / Math.sqrt(variance + eps) + beta, dot(w.map(value => alpha * value), patch) + beta + alpha * (bias - mu));
    }
    const kernel = [2, -1], input = [3, 1, -2], cotangent = [-1, 4];
    const forward = [dot(kernel, input.slice(0, 2)), dot(kernel, input.slice(1, 3))];
    const reverse = [2 * cotangent[0], -cotangent[0] + 2 * cotangent[1], -cotangent[1]];
    close(dot(forward, cotangent), dot(input, reverse));
    assert.notDeepEqual(reverse, input);
    return ['Changed adaptive-pooling matrix has overlapping bins and satisfies its adjoint identity', 'Evaluation-BN folding with negative, zero and positive channel scales', 'Changed convolution transpose satisfies adjoint identity without claiming inversion'];
  },
  'landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet': () => {
    const width = u => 24 * 2 ** Math.round(Math.log2(u / 24));
    assert.equal(width(33), 24); assert.equal(width(35), 48);
    assert.equal(width(67), 48); assert.equal(width(69), 96);
    const initial = 6, growth = 4, layers = 3;
    assert.deepEqual(Array.from({ length: layers }, (_, i) => 9 * (initial + i * growth) * growth), [216, 360, 504]);
    const parameter = [6, 8], gradient = [3, 4], threshold = .02;
    const scale = threshold * Math.hypot(...parameter) / Math.hypot(...gradient);
    close(Math.hypot(...gradient.map(value => value * scale)), .2);
    close(.1 * .2 / Math.hypot(...parameter), .002);
    const coefficients = [2, -1], target = 3, at = [1, -2], h = 1e-5;
    const loss = values => (dot(coefficients, values) - target) ** 2;
    for (let j = 0; j < 2; j++) {
      const plus = [...at], minus = [...at]; plus[j] += h; minus[j] -= h;
      close((loss(plus) - loss(minus)) / (2 * h), 2 * (dot(coefficients, at) - target) * coefficients[j]);
    }
    close(dot(coefficients, [1, 2]), 0);
    return ['Changed RegNet base crosses log-space rather than arithmetic midpoint boundaries', 'Changed dense-growth fixture exposes increasing layer cost', 'Changed AGC threshold gives the stated SGD relative-step bound', 'Frozen nontrivial feature gradient matches finite differences and has a null direction'];
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply exact topic IDs.');
for (const topicId of selected) {
  assert.ok(checks[topicId], `Unknown topic: ${topicId}`);
  const directory = `docs/teaching/concept-intuition/${topicId}/`;
  const author = JSON.parse(await readFile(directory + 'author-checks.json', 'utf8'));
  assert.ok(author.checks.length && author.checks.every(check => check.passed === true || check.status === 'passed'));
  const sourceFiles = [...author.sourceFiles, directory + 'independent-review.md', 'scripts/review-network-architecture-intuition.mjs'];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  for (const file of author.sourceFiles) assert.equal(sourceHashes[file], author.sourceHashes[file], `Stale author source: ${file}`);
  const result = checks[topicId]();
  await writeFile(directory + 'independent-checks.json', JSON.stringify({ topicId, reviewer: 'classical_representation_intuition', independent: true, reviewedAt: new Date().toISOString(), passed: true, sourceFiles, sourceHashes, checks: result.map(name => ({ name, status: 'passed' })), correctnessReview: 'accepted at source level', learningExperienceReview: 'accepted at source level; heuristic review, not an actual learner trial', fullReading: directory + 'independent-review.md', actualBrowserInspection: false, browserReview: 'root records separately', nativeExecution: 'Unchanged historical campaigns retained; no new fitting or library-execution claim.' }, null, 2) + '\n');
  process.stdout.write(`${topicId}: ${result.length} independent groups passed\n`);
}
