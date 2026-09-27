// Complementary independent checks for the convolution-to-sequence teaching group.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const sum = values => values.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((value, i) => value * b[i]));
const close = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const sigmoid = x => 1 / (1 + Math.exp(-x));
const checks = {
  'depthwise-separable-dilated-convolutions': () => {
    const previous = [-2, -1, 0, 1, 2];
    const support = d => [...new Set([-d, 0, d].flatMap(offset => previous.map(x => offset + x)))].sort((a, b) => a - b);
    assert.deepEqual(support(5), Array.from({ length: 15 }, (_, i) => i - 7));
    assert.deepEqual(Array.from({ length: 17 }, (_, i) => i - 8).filter(x => !support(6).includes(x)), [-3, 3]);
    const h = 11, padding = 2, k = 4, dilation = 2, stride = 3;
    const starts = Array.from({ length: h + 2 * padding }, (_, i) => i).filter(i => i % stride === 0 && i + (k - 1) * dilation < h + 2 * padding);
    assert.equal(starts.length, Math.floor((h + 2 * padding - dilation * (k - 1) - 1) / stride) + 1);
    assert.deepEqual(starts, [0, 3, 6]);
    for (const x of [-7, -.1, 0, 2, 9]) close(Math.max(0, x) - Math.max(0, -x), x);
    const original = patch => [5 * patch[0], 2 * patch[1]], approximate = patch => [5 * patch[0], 0];
    assert.deepEqual(original([4, 0]), approximate([4, 0]));
    assert.deepEqual(original([0, 7]).map((value, i) => value - approximate([0, 7])[i]), [0, 14]);
    return ['Different reachable interval verifies exact discrete join threshold and two holes', 'Changed stencil enumeration reproduces the output-size formula', 'Signed expansion reconstructs before narrow clipping; rank error remains input-dependent'];
  },
  'convnext-modern-cnn-designs': () => {
    const normalize = values => {
      const mean = sum(values) / values.length, variance = sum(values.map(value => (value - mean) ** 2)) / values.length;
      return values.map(value => (value - mean) / Math.sqrt(variance + 1e-6));
    };
    const locations = [[0, 1, 5], [6, 3, 0]];
    const pooled = normalize(locations[0].map((value, i) => (value + locations[1][i]) / 2));
    const normalized = locations.map(normalize), other = normalized[0].map((value, i) => (value + normalized[1][i]) / 2);
    assert.ok(Math.hypot(...pooled.map((value, i) => value - other[i])) > .5);
    const channel = [1, 2], second = [2, 0], norm = Math.hypot(...channel), r = norm / ((norm + Math.hypot(...second)) / 2 + 1e-6);
    const loss = gamma => .5 * sum(channel.map(value => (value + gamma * value * r) ** 2));
    close((loss(1e-5) - loss(-1e-5)) / 2e-5, r * dot(channel, channel));
    const patch = [2, 6, 8];
    normalize(patch).forEach((value, i) => close(value, normalize(patch.map(x => x + 20))[i]));
    const largeGrid = 28 * 28, smallGrid = 7 * 7;
    close(largeGrid ** 2 / smallGrid ** 2, 256);
    return ['Changed three-channel head proves pooling and channel normalization do not commute', 'Changed GRN identity-start scale gradient checked by finite differences', 'Changed patch brightness invariance and global pair-count ratio'];
  },
  'capsule-networks': () => {
    const squash = vector => { const r = Math.hypot(...vector); return vector.map(value => value * r / (1 + r * r)); };
    const at = [1, 2, 2], radial = at.map(value => value / 3), tangent = [0, 1 / Math.sqrt(2), -1 / Math.sqrt(2)], h = 1e-5;
    for (const [direction, eigenvalue] of [[radial, .06], [tangent, .3]]) {
      const plus = squash(at.map((value, i) => value + h * direction[i]));
      const minus = squash(at.map((value, i) => value - h * direction[i]));
      plus.forEach((value, i) => close((value - minus[i]) / (2 * h), eigenvalue * direction[i]));
    }
    const value = -.7, full = x => sigmoid(2 * x) * x;
    close((full(value + h) - full(value - h)) / (2 * h), sigmoid(2 * value) + 2 * value * sigmoid(2 * value) * (1 - sigmoid(2 * value)));
    assert.ok(Math.abs(sigmoid(2 * value) - (full(value + h) - full(value - h)) / (2 * h)) > .1);
    close(sigmoid(-3 * Math.log(.25)), 64 / 65);
    const posteriorPrecision = 1 / 4 + 2 / 2;
    close(1 / posteriorPrecision, .8); close((3 + 3) / 2 / posteriorPrecision, 2.4);
    return ['Three-dimensional squash finite differences recover independent radial and tangent eigenvalues', 'Changed nonlinear coupling explicitly loses a derivative term under detachment', 'Changed Gaussian coding cost and nonunit prior/observation precision'];
  },
  'rnns-lstms-grus': () => {
    const a = [[0, 3], [0, 0]], b = [[0, 0], [4, 0]], apply = (matrix, vector) => matrix.map(row => dot(row, vector));
    assert.deepEqual(apply(a, apply(a, [0, 1])), [0, 0]);
    assert.deepEqual(apply(b, apply(a, [0, 1])), [0, 12]);
    const local = [[.2, .3], [.4, .8]], initial = [0, 1];
    close(apply(local, apply(local, initial))[1], .8 ** 2 + .4 * .3);
    const w = [[2, -1], [1, 3]], hidden = [-1, 2], reset = [.25, .75];
    const before = apply(w, hidden.map((value, i) => value * reset[i]));
    const after = apply(w, hidden).map((value, i) => value * reset[i]);
    assert.notDeepEqual(before, after);
    const d = 5, cell = 24, projected = 8;
    assert.equal(4 * cell * d + 4 * cell * projected + 8 * cell + projected * cell, 4 * cell * (d + projected + 2) + projected * cell);
    return ['Changed nilpotent matrices amplify under alternation', 'Changed two-state cell path sum includes hidden feedback; reset-placement contrast survives changed inputs', 'Projected recurrent parameter count reconstructed from individual matrix shapes'];
  },
  'sequence-to-sequence-encoder-decoder': () => {
    const reference = .7, replacementZero = .4;
    const joint = [[reference * replacementZero, (1 - reference) * replacementZero], [reference * (1 - replacementZero), (1 - reference) * (1 - replacementZero)]];
    for (const row of joint) close(row[0] / sum(row), reference);
    close(sum(joint.flat()), 1);
    const rewards = [-1, 2], expected = z => (1 - sigmoid(z)) * rewards[0] + sigmoid(z) * rewards[1], z = -.3, h = 1e-5;
    close((expected(z + h) - expected(z - h)) / (2 * h), 3 * sigmoid(z) * (1 - sigmoid(z)));
    const size = 4, direct = Array(size).fill(size), reversed = Array.from({ length: size }, (_, i) => 2 * i + 1);
    close(sum(direct) / size, sum(reversed) / size); assert.deepEqual(reversed, [1, 3, 5, 7]);
    const prefix = -1.4, finished = -.9, future = -1.5 / ((5 + 7) / 6);
    assert.ok(prefix < finished && future > finished);
    return ['Unequal replacement-prefix fixture learns the target marginal independent of shown prefix', 'Changed complete-answer reward derivative matches finite differences', 'Four-alignment reversal paths and changed length-normalized stopping counterexample'];
  },
};
const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply exact topic IDs.');
for (const topicId of selected) {
  assert.ok(checks[topicId], `Unknown topic ${topicId}`);
  const directory = `docs/teaching/concept-intuition/${topicId}/`;
  const author = JSON.parse(await readFile(directory + 'author-checks.json', 'utf8'));
  assert.equal(author.authorChecksPassed, true);
  assert.ok(author.actualChecks.length > 0);
  const sourceFiles = [...author.sourceFiles, directory + 'independent-review.md', 'scripts/review-convolution-sequence-intuition.mjs'];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  for (const file of author.sourceFiles) assert.equal(sourceHashes[file], author.sourceHashes[file], `Stale author source: ${file}`);
  const result = checks[topicId]();
  await writeFile(directory + 'independent-checks.json', JSON.stringify({ topicId, reviewer: 'classical_representation_intuition', independent: true, reviewedAt: new Date().toISOString(), passed: true, sourceFiles, sourceHashes, checks: result.map(name => ({ name, status: 'passed' })), correctnessReview: 'accepted at source level', learningExperienceReview: 'accepted at source level; heuristic review, not an actual learner trial', fullReading: directory + 'independent-review.md', actualBrowserInspection: false, browserReview: 'root records separately', nativeExecution: 'Unchanged scientific campaigns retained; no new native fit or library-execution claim.' }, null, 2) + '\n');
  process.stdout.write(`${topicId}: ${result.length} independent groups passed\n`);
}
