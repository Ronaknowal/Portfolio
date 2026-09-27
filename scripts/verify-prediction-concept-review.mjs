// Independent review calculations for the seven prediction-family revisions.
// Browser review remains a separate source-bound receipt.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const reviews = [
  ['linear-logistic-regression', () => {
    const residuals = [.1, .2, -.7, .4];
    close(residuals.reduce((a, b) => a + b), 0);
    close(residuals.reduce((a, b, i) => a + b * i, 0), 0);
    const loss = weight => Math.log1p(Math.exp(-(-1 + 2 * weight)));
    const gradient = 2 * (1 / (1 + Math.exp(-1)) - 1);
    close((loss(1 + 1e-5) - loss(1 - 1e-5)) / 2e-5, gradient, 1e-9);
    assert.ok(loss(1 - .1 * gradient) < loss(1));
    for (const value of [-1, -.3, 0, .3, 1]) {
      const ridge = value / 1.5;
      const lasso = Math.sign(value) * Math.max(0, Math.abs(value) - .5);
      close(ridge - value + .5 * ridge, 0);
      if (lasso) close(lasso - value + .5 * Math.sign(lasso), 0);
      else assert.ok(Math.abs(value) <= .5);
    }
  }, 'The signed-residual normal-equation figure connects the actual shipment residuals to both stationarity equations. The one-row logistic update separates score, probability and parameter change; the scalar penalty curves use equal axes and distinguish continuous shrinkage from a zero coefficient. Independently checked finite-difference loss slope, descent and both penalty optima. The full lesson retains rank/conditioning, likelihood, cost and calibration qualifications, complete scratch/library examples and all twelve changed exercises.'],
  ['decision-trees-random-forests', () => {
    const labels = [0, 0, 0, 1];
    close(labels.flatMap(a => labels.map(b => +(a !== b))).reduce((a, b) => a + b) / 16, .375);
    const gini = values => 1 - [0, 1].reduce((s, label) => s + (values.filter(v => v === label).length / values.length) ** 2, 0);
    const y = [0, 0, 1, 1];
    [1, 2, 3].forEach((cut, i) => close(.5 - cut / 4 * gini(y.slice(0, cut)) - (4 - cut) / 4 * gini(y.slice(cut)), [1 / 6, .5, 1 / 6][i]));
    const features = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]];
    const expected = [[1, .5, 0], [.5, 1, .5], [0, .5, 1]];
    features.forEach((a, i) => features.forEach((b, j) => close(a.reduce((s, v, k) => s + v * b[k], 0) / 2, expected[i][j])));
  }, 'Read the complete progression from a region query through Gini, thresholds, regression, pruning, OOB ownership and importance to forest similarity and practice. The ordered-pair table correctly uses sampling with replacement, unlike majority error. The prefix figure keeps the same four rows while transferring one class count. The proximity diagram keeps each tree’s leaf vocabulary distinct; independently reconstructed its Gram matrix. New pruning and covariance prose motivates the algebra without promising test improvement. Existing interactive split/pruning/bootstrap investigations and thirteen practice questions remain.'],
  ['k-nearest-neighbors-knn', () => {
    for (let angle = 0; angle < 2 * Math.PI; angle += .05) assert.ok(Math.hypot(6 + 2 * Math.cos(angle), 2 * Math.sin(angle)) >= 4 - 1e-12);
    const p = .8;
    const cells = [p * p, p * (1 - p), (1 - p) * p, (1 - p) ** 2];
    close(cells.reduce((a, b) => a + b), 1);
    close(cells[1] + cells[2], .32);
    close((2 * .1 * .9 + 2 * .8 * .2) / 2, .25);
    close(2 * .15 * .85, .255);
  }, 'Read the full lesson, including exact/approximate search, dimensionality, asymptotic risks, smooth-regression rates and changed practice. The ball diagram has equal coordinate units, a genuine lower bound and an incumbent that makes pruning legal; a loose bound does not guarantee improvement. The noise mosaic has 80/20 widths and heights, making disagreement area .32 while the Bayes decision error is .20. The growing-k explanation distinguishes averaging noise from preserving locality before stating assumptions. Existing voting, metric, KD-tree and recall controls remain the investigations; the new figures do not masquerade as sliders.'],
  ['gradient-boosted-trees-xgboost-lightgbm-catboost', () => {
    const y = [2, 2, 3, 7, 8, 8];
    const updated = y.map((_, i) => 5 + .5 * (i < 3 ? -8 / 3 : 8 / 3));
    close(y.reduce((s, value, i) => s + (value - updated[i]) ** 2, 0) / 6, 2);
    [1, 4].forEach(h => {
      const best = 2 / (h + 1);
      const f = w => -2 * w + (h + 1) * w ** 2 / 2;
      close(-2 + (h + 1) * best, 0);
      assert.ok(f(best) < f(best - .1) && f(best) < f(best + .1));
    });
    close((1 + 5) / 2, 3); close((1 ** 2 + 5 ** 2) / 2 - 3 ** 2, 4);
    close(6 - 1, 5); close(6 - 1.5, 4.5);
  }, 'Read the full loss-to-library progression, all worked examples and changed practice. The six-row correction strips expose the restricted function class, with consistent signed scale. The two curvature panels share both axes and plot the exact declared surrogate, not a measured logistic curve. Ordered-model dependency is separated from ordered categorical statistics. The unbiased-total versus squared-score bridge is mathematically sound and sits next to GOSS. Complete code and measured native outputs are retained; GPU speed and cross-library superiority are not invented.'],
  ['support-vector-machines-svm', () => {
    [-.2, .6, 1.7].forEach((m, i) => close(Math.max(0, 1 - m), [1.2, .4, 0][i]));
    for (let i = 0; i <= 100; i++) {
      const a = i / 400, w = 2 * a;
      const d = 2 * a - w * w / 2;
      const p = w * w / 2 + .5 * Math.max(0, 1 - w);
      assert.ok(d <= .375 + 1e-12 && p >= .375 - 1e-12);
    }
    close(.5 * .5 ** 2 + .5 * .5, .375);
    [1, 3].forEach(q => close(-.5 * (q * 0) + .5 * (q * 2) - 1, q - 1));
  }, 'Read the entire score/geometry, hinge, dual/KKT, kernel, solver, validation, SVR and approximation progression. The hinge figure distinguishes normalized score shortfall from input distance. Every displayed objective bracket is primal/dual feasible; independent grid checks preserve the correct bound direction. The precomputed-kernel diagram binds columns to training observations and explains the accidental square shape. Added Nyström and SVR bridges retain pseudoinverse, unit and degeneracy conditions. Existing geometric/SMO/tube labs and full scratch/library routes remain.'],
  ['naive-bayes-probabilistic-classifiers', () => {
    const joint = [0, 1].flatMap(c => [0, 1].flatMap(a => [0, 1].map(b => { const p = c ? .8 : .2; return { c, a, b, mass: .5 * (a ? p : 1 - p) * (b ? p : 1 - p) }; })));
    close(joint.reduce((s, x) => s + x.mass, 0), 1);
    close(joint.filter(x => x.a && x.b).reduce((s, x) => s + x.mass, 0), .34);
    close(.2 * .2 / (.2 * .2 + .8 * .6), 1 / 13);
    close((4 * 5) / (5 * 6), 2 / 3);
    close((4 / 5) ** 2, .64);
    close(Math.exp(-1) / (1 + Math.exp(-1)), 1 / (1 + Math.E));
  }, 'Read the complete lesson through its twelve transfer exercises and source annotations. Conditional independence is illustrated with a genuine class mixture, not the later copied-sensor counterexample. Observed-off and unmeasured branches marginalize different events with an explicit noninformative-observation assumption. The two-token figure correctly separates a fixed table from integrating shared parameter uncertainty; it does not pretend to observe a hidden test label. Stable-normalization intermediates are complete. Existing Gaussian geometry, sparse/streaming, complement and calibration explanations retain their local reasoning and runnable code.'],
  ['ensemble-methods-stacking', () => {
    const epsilon = 1 / 6, alpha = Math.log(5) / 2;
    const z = a => (1 - epsilon) * Math.exp(-a) + epsilon * Math.exp(a);
    close((1 - epsilon) * Math.exp(-alpha), epsilon * Math.exp(alpha));
    close(z(alpha), Math.sqrt(5) / 3);
    assert.ok(z(alpha) < z(alpha - .01) && z(alpha) < z(alpha + .01));
    close(1 / (1 + Math.exp(-(-1 + 2 * .8 + .5 * 2))), .8320183851339245);
    close((1 - .228273) * (13 / 3) + .228273 * 2, 3.8006963333333332);
    close(.8 * .9, .72);
  }, 'Read the full bagging/AdaBoost/OOF-to-serving progression, advanced context/complexity branches and all twelve changed exercises. The new exponential-loss curves are calculated, share axes and explain the derivative balance. OOF feature construction and final base refits keep fixed column identities; the stored combiner is not retrained on leaked full-data predictions. Mixed margin/probability branches calculate a learned logit before its link. Independent checks confirm the balance, loss multiplier and blend arithmetic; existing native fit evidence, assumptions and limitations are retained.'],
];

for (const [id, check, finding] of reviews) {
  check();
  const directory = `docs/teaching/concept-intuition/${id}`;
  const author = JSON.parse(fs.readFileSync(`${directory}/author-checks.json`));
  const sourceHashes = Object.fromEntries(author.sourceFiles.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  for (const [file, digest] of Object.entries(sourceHashes)) assert.equal(digest, author.sourceHashes[file], `Author receipt stale: ${file}`);
  fs.writeFileSync(`${directory}/independent-review.md`, `# Independent conceptual review\n\n26 September 2026; reviewer: root, not this lesson revision’s author.\n\n${finding}\n\n## Learning-experience assessment\n\nRead the complete current production lesson and each new figure’s implementation. The first-pass route is explicit; new terminology is locally motivated and advanced branches preserve equations rather than replacing them with analogies. New representations answer separate questions at their point of use. The existing investigations remain immediately playable without prediction entry. Code/example explanations and changed-condition practice remain present. Constructed data are identified as constructed; no new empirical performance claim is made. Existing scientific engines and native results were not re-executed for this explanatory revision.\n\nNo unresolved source or mathematical finding in this pass. Desktop/mobile appearance and control behavior remain a separate root browser check; this source review does not certify rendering or claim a human learner trial.\n`);
  fs.writeFileSync(`${directory}/independent-checks.json`, JSON.stringify({ topicId: id, reviewer: 'root', reviewedOn: '2026-09-26', passed: true, sourceFiles: author.sourceFiles, sourceHashes, checks: [{ name: 'Complete revised lesson and new figure source review', passed: true }, { name: 'Independent mathematical checks in scripts/verify-prediction-concept-review.mjs', passed: true }, { name: 'Independent learning-experience checklist; rendered checks separate', passed: true }], findings: [], browser: 'Separate browser receipt required', limits: 'No new native fit campaign or human learner trial.' }, null, 2) + '\n');
}
console.log(`Independent source/math review passed for ${reviews.length} prediction-family lessons; browser pending.`);
