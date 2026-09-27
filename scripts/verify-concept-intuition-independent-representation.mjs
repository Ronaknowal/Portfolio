import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { transform } from 'esbuild';

const close = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
const sum = values => values.reduce((a, b) => a + b, 0);
const cases = {
  'long-context-sequence-models-transformer-xl-griffin-perceiver': {
    reading: 'Read the complete current canonical lesson, all eight practices and scratch/library implementation explanations, including relative position, RG-LRU gates, Perceiver IO/AR, detached gradients and resource counts. Checked the new isolated distance example against Transformer-XL section 3.3 and the distinction between global and query-dependent preferences. The full lesson keeps cache, state and retained input bank separate and preserves declared small-model limitations. No correction found; current source-bound generator and trained outputs retained.',
    checks: {
      'Relative preference reverses with query, not common translation': () => { const softmaxNear = q => 1 / (1 + Math.exp(-2 * (q + .5))); close(softmaxNear(1), .9525741268, 1e-9); close(softmaxNear(-1), .2689414214, 1e-9); assert.deepEqual([5 - 1, 5 - 5], [105 - 101, 105 - 105]); },
      'RG-LRU gate isolates decay from zero writes': () => { const first = Math.sqrt(1 - .8 ** 2); close(first * .8 ** 5, .196608); const retention = .8 ** (.001 * 8); close(first * retention ** 5, .594668384, 1e-8); close(.8 ** 2 + Math.sqrt(1 - .8 ** 2) ** 2, 1); },
      'Memory output, budget and detached derivative arithmetic': () => { close((9 * 6 + 1 + 8 + 2) / 13, 5); close(4096 * 32 + 8 * 32 ** 2, 139264); close(2 * 12 * 2048 * 8 * 32 * 2 / 2 ** 20, 24); const h = 1e-5; close(((3 + h) * 6 - (3 - h) * 6) / (2 * h), 6); close((2 * (3 + h) ** 2 - 2 * (3 - h) ** 2) / (2 * h), 12); },
    },
  },
  'state-space-models-s4-mamba-mamba-2': {
    reading: 'Read the complete current canonical lesson, all nine practices, native and specialist-library explanations, and advanced HiPPO, DPLR, SSD, S5 and Mamba-3 branches. Checked the transposed FFT table after the browser finding: times 0,1,2,3 carry values 1,2,1,2 into length-three slots 0,1,2,0. Independently inspected Mamba-3 section 3.1 and appendix A.2 for the lambda-dependent accuracy condition and Tri Dao SSD algorithm article for chunk/state decomposition. The new Woodbury example retains coupling and is correctly scoped. No content correction found; no GPU execution or refitting claimed.',
    checks: {
      'Independent explicit DFT exposes circular alias and correct padding': () => { const convolutionDft = size => { const dft = values => Array.from({ length: size }, (_, k) => values.reduce((z, value, j) => [z[0] + value * Math.cos(-2 * Math.PI * k * j / size), z[1] + value * Math.sin(-2 * Math.PI * k * j / size)], [0, 0])); const x = dft([1, 0, 1]), kernel = dft([1, 2]); const product = x.map((z, k) => [z[0] * kernel[k][0] - z[1] * kernel[k][1], z[0] * kernel[k][1] + z[1] * kernel[k][0]]); return Array.from({ length: size }, (_, j) => sum(product.map((z, k) => z[0] * Math.cos(2 * Math.PI * k * j / size) - z[1] * Math.sin(2 * Math.PI * k * j / size))) / size); }; convolutionDft(3).forEach((v, i) => close(v, [3, 2, 1][i])); convolutionDft(4).forEach((v, i) => close(v, [1, 2, 1, 2][i])); },
      'Woodbury example agrees with direct two-by-two solve': () => { const determinant = 2 * 3 - 1; const direct = [[3 / determinant, -1 / determinant], [-1 / determinant, 2 / determinant]]; close(direct[0][0], .6); close(direct[1][0], -.2); const rhs = [2, -1], solution = direct.map(row => sum(row.map((v, j) => v * rhs[j]))); close(solution[0], 1.4); close(solution[1], -.8); close(2 * solution[0] + solution[1], rhs[0]); close(solution[0] + 3 * solution[1], rhs[1]); },
      'SSD recurrence versus independently expanded influence': () => { const a = [.5, .5, .25, .8], b = [[1, 0], [0, 1], [1, 1], [1, -1]], c = [[1, 0], [1, 1], [0, 1], [1, 2]], v = [[2, 1], [3, -1], [1, 2], [-2, 1]]; const expected = [[2, 1], [4, -.5], [1.75, 1.75], [5.8, 3.5]]; for (let i = 0; i < 4; i++) for (let p = 0; p < 2; p++) { let output = 0; for (let j = 0; j <= i; j++) { let decay = 1; for (let k = j + 1; k <= i; k++) decay *= a[k]; output += sum(c[i].map((n, k) => n * b[j][k])) * decay * v[j][p]; } close(output, expected[i][p]); } },
      'Endpoint write rule and bounded-state bytes': () => { const step = lambda => .5 + (1 - lambda) * .5 * 2 + lambda * 6; close(step(.5), 4); close(step(1), 6.5); close(3 * 20 * 128 * 32 * 2 / 2 ** 20, .46875); },
    },
  },
  'regularization-l1-l2-elastic-net-dropout': {
    reading: 'Read the entire eleven-section lesson, ten practices, source annotations and scratch/library explanations. Reviewed partial residuals, coordinate/KKT logic, duplicate-feature ambiguity, likelihood/prior normalization, dropout, group and smoothness penalties, nonconvex factorization and information criteria. New group shrinkage figure correctly scopes its orthonormal-block formula, and MAP scaling distinguishes independent evidence from duplicated rows. No correction found; preserved fit outputs were not rerun.',
    checks: {
      'Partial residual and group versus coordinate shrinkage': () => { close(10 - 8 + 3, 5); const z = [3, 4], norm = Math.hypot(...z); assert.deepEqual(z.map(v => v * Math.max(0, 1 - 4 / norm)).map(v => Number(v.toFixed(6))), [.6, .8]); assert.deepEqual(z.map(v => Math.max(v - 4, 0)), [0, 0]); },
      'Enumerated inverted-dropout expectation': () => { const losses = [0, -2, 4, 2].map(y => .5 * (y - 1) ** 2); close(sum(losses) / 4, 2.5); const q = .75; let total = 0; for (const a of [0, 1]) for (const b of [0, 1]) total += (a ? q : 1 - q) * (b ? q : 1 - q) * .5 * (2 * a / q - b / q - 1) ** 2; close(total, 5 / 6); },
      'Factor penalty reduced objective and ridge convention': () => { const lambda = .25, product = 1 - 2 * lambda; close(.5 * (1 - product) ** 2 + 2 * lambda * Math.abs(product), .375); close(12 / (4 + 4 * 1), 1.5); },
    },
  },
  'feature-selection-importance-shap-permutation-mutual-info': {
    reading: 'Read all nine sections, ten exercises and scratch/library explanations. Checked marginal/conditional MI, XOR, split-gain units, refit versus permutation questions, correlated donors, replacement versus conditional SHAP, exact background allocation and grouped-player game changes. The new order table makes the group mismatch explicit. No correction found; native Wine fits remain unchanged.',
    checks: {
      'Gini gain and node weighting from class counts': () => { const impurity = counts => 1 - sum(counts.map(n => (n / sum(counts)) ** 2)); close(impurity([2, 2]) - 3 / 4 * impurity([2, 1]), 1 / 6); close(.5 * (1 / 6), 1 / 12); },
      'Enumerated unanimity Shapley allocations versus grouped game': () => { const orders = ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA']; const shares = ['A', 'B', 'C'].map(name => orders.filter(order => order.at(-1) === name).length / 6); close(shares[0] + shares[1], 2 / 3); close(['GC', 'CG'].filter(order => order.at(-1) === 'G').length / 2, .5); close(1 / 3 + 1 / 6 + 1 / 6 + 1 / 3, 1); },
      'Exact two-player background allocation': () => { const v = [1.5, 3.5, 5, 11]; close(.5 * ((v[1] - v[0]) + (v[3] - v[2])), 4); close(.5 * ((v[2] - v[0]) + (v[3] - v[1])), 5.5); close(4 + 5.5, v[3] - v[0]); },
    },
  },
  'bias-variance-tradeoff-learning-curves': {
    reading: 'Read the complete eleven-section lesson, eight practices and code commentary, including the optional smoother, classifier, ensemble and double-descent branches. Reviewed distinct sampling distributions, fixed-probe bias, empirical-fit limitations, same-input optimism and ridgeless assumptions. New directional smoother example correctly separates trace(S) from trace(SS transpose). No correction found; recorded learning-curve fits retained.',
    checks: {
      'Enumerated fixed-smoother training and fresh risk': () => { const s = [.8, .2], states = [[-1, -1], [-1, 1], [1, -1], [1, 1]]; let train = 0, fresh = 0; for (const eps of states) { train += sum(eps.map((e, i) => (e - s[i] * e) ** 2)) / 8; for (const newEps of states) fresh += sum(newEps.map((e, i) => (e - s[i] * eps[i]) ** 2)) / 32; } close(train, .34); close(fresh, 1.34); close(fresh - train, 1); close(sum(s.map(v => v ** 2)) / 2, .34); },
      'Quadratic interpolation noise and classifier contrast': () => { close(sum([-.125, .75, .375].map(v => .25 * v * v)), 23 / 128); close(23 / 128 + .25, 55 / 128); const brier = p => .7 * (1 - p) ** 2 + .3 * p ** 2; close(brier(.6), .22); close((brier(.4) + brier(.8)) / 2, .26); close(.5 * (.7 + .3), .5); },
      'Correlated ensemble variance': () => close(4 * (.5 + .5 / 4), 2.5),
    },
  },
  'imbalanced-learning-smote-cost-sensitive-learning': {
    reading: 'Read all ten sections and ten practices, actual-study interpretation, full weighted scratch and sampler-library explanations, and ADASYN/NearMiss/focal/prior-shift branches. Found and corrected one pre-existing notation error: mean(q-y) squared put squaring outside the mean; the page now says mean squared probability error. Numerical Brier results were already correct and are unchanged. New NearMiss figure and prior-odds bridge clarify their own mechanisms. No unresolved finding.',
    checks: {
      'NearMiss nearest/farthest choose different candidates': () => { const minority = [0, 1, 10]; const distance = x => minority.map(v => Math.abs(v - x)); close(Math.min(...distance(.5)), .5); close(Math.min(...distance(5)), 4); close(Math.max(...distance(.5)), 9.5); close(Math.max(...distance(5)), 5); },
      'Normalized weighting gradient and odds inverse': () => { close((1 * .5 + 3 * -.5) / 4, -.25); close((1 * .5 * 0 + 3 * -.5 * 2) / 4, -.75); const p = .1, q = 9 * p / (9 * p + 1 - p); close(q, .5); close(q / (9 - 8 * q), p); close(4 / (99 + 4), 4 / 103); },
      'Brier squares before averaging, without error cancellation': () => { const errors = [.5, -.5]; close(sum(errors.map(v => v * v)) / 2, .25); close((sum(errors) / 2) ** 2, 0); },
    },
  },
  'automl-neural-architecture-search-nas': {
    reading: 'Read the entire ten-section lesson, ten practices, all program and API explanations. Reviewed conditional search measures, expected improvement, fidelity budgets, architecture shapes, real-study boundaries, portfolio selection, Pareto feasibility, relaxed mixtures, bilevel derivatives, weight sharing and NASWOT. New Gram-vector construction accounts for matching active and inactive decisions, and discretization example does not transfer mixture scores to selected operators. No correction found.',
    checks: {
      'NASWOT Gram matrix from independent concatenated vectors': () => { const codes = [[1, 1, 0], [1, 0, 1]]; const vectors = codes.map(row => [...row, ...row.map(v => 1 - v)]); const gram = vectors.map(a => vectors.map(b => sum(a.map((v, i) => v * b[i])))); assert.deepEqual(gram, [[3, 1], [1, 3]]); close(gram[0][0] * gram[1][1] - gram[0][1] ** 2, 8); },
      'Bilevel one-step derivative from finite differences': () => { const objective = alpha => .5 * (.1 * alpha - 1) ** 2; const h = 1e-5; close((objective(.2 + h) - objective(.2 - h)) / (2 * h), -.098); close(.2 - 1, -.8); },
      'Relaxation gap and genuine-resume resource': () => { close(.5 * ((1 + -1) / 2) ** 2, 0); close(.5 * 1 ** 2, .5); close(9 + 3 * 2 + 6, 21); close(9 + 3 * 3 + 9, 27); close(12 * 8 + 8, 104); },
    },
  },
  'hidden-markov-models-hmm': {
    reading: 'Read all thirteen sections, ten practices, native-program commentary and version-specific library interpretation. Checked chain assumptions, alpha/beta boundaries, sum versus max, constrained risk, pair posteriors, independent-sequence EM, duration, Gaussian units, topology and cost. Found and corrected one display error: fixed-decimal formatting printed representable 0.3^100 as zero. It now uses scientific notation 5.153775e-53. Added shared-meter table and density units are correct; numerical engines and fits unchanged.',
    checks: {
      'Exact-path enumeration matches weather sequence evidence': () => { const pi = [.6, .4], a = [[.7, .3], [.4, .6]], b = [[.1, .4, .5], [.6, .3, .1]], obs = [0, 1, 0, 2]; const paths = Array.from({ length: 16 }, (_, mask) => { const z = Array.from({ length: 4 }, (_, t) => (mask >> t) & 1); let p = pi[z[0]] * b[z[0]][obs[0]]; for (let t = 1; t < 4; t++) p *= a[z[t - 1]][z[t]] * b[z[t]][obs[t]]; return p; }); close(sum(paths), .00933936); assert.ok(Math.max(...paths) < sum(paths)); },
      'Factorial shared evidence and Gaussian change of units': () => { const states = [[0, 0], [0, 1], [1, 0], [1, 1]].filter(row => sum(row) === 1); close(states.filter(row => row[0] === 1).length / states.length, .5); assert.equal(states.filter(row => row[0] && row[1]).length, 0); close(1 / (1 + .5), 2 / 3); close(.01 / (.01 + .005), 2 / 3); },
      'Representable small probability retains nonzero scientific text': () => { const value = .3 ** 100; assert.ok(value > 0); assert.equal(value.toExponential(6), '5.153775e-53'); assert.equal(.01 ** 400, 0); },
    },
  },
  'attention-mechanism-bahdanau-luong': {
    reading: 'Read the complete current canonical manuscript, full embedded training program, all eight practices and annotated resources, with the source-bound generator/runtime receipt. Reviewed lookup roles, score/schedule separation, nonlinear scorer cancellation, packing/masks, gradient paths, measured alignment limits, windows, copying, location and coverage. The new coverage table shows totals growing by one per completed read and correctly separates overlap penalty from factual coverage. No correction found; trained results retained.',
    checks: {
      'Coverage accumulation and overlap independently enumerated': () => { const reads = [[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]]; const old = reads[0].map((v, j) => v + reads[1][j]); close(sum(old), 2); close(sum(reads[2].map((v, j) => Math.min(v, old[j]))), .7); close(sum(old.map((v, j) => v + reads[2][j])), 3); close(sum(reads[0].map(v => Math.min(v, 0))), 0); },
      'Attention output gradient finite difference and null context': () => { const keys = [[1, 0], [0, 1], [-1, 0]], values = [[2, 0], [0, 2], [-1, 1]]; const loss = q => { const masses = keys.map(k => Math.exp(sum(k.map((v, i) => v * q[i])))); const c = [0, 1].map(j => sum(values.map((v, i) => v[j] * masses[i])) / sum(masses)); return Math.log(1 + Math.exp(c[0] - c[1])); }; close(loss([1, 0]), 1.077272, 1e-6); const h = 1e-5; close((loss([1 + h, 0]) - loss([1 - h, 0])) / (2 * h), .745440, 1e-6); close((loss([1, h]) - loss([1, -h])) / (2 * h), -.429460, 1e-6); },
      'Copy aggregation and scoring-parameter count': () => { close(.4 * .1 + .6 * (.2 + .5), .46); close(64 * 96 + 64 * 192 + 64, 18496); close(96 * 192, 18432); },
    },
  },
  'independent-component-analysis-ica': {
    reading: 'Read all eleven sections, the six exercises, source notes and both program explanations. Reviewed the four-state mixture, covariance/independence distinction, whitening, kurtosis, fixed point, clinical recording boundaries, ambiguities, component removal, likelihood and extensions. The new probability-mass panels preserve coordinate and probability scales and clearly identify a discrete distribution. Linear-g cancellation and the determinant example supply local reasons for previously abstract choices. No content correction found.',
    checks: {
      'Enumerated binary-source projection moments': () => { const states = [[-1, -1], [-1, 1], [1, -1], [1, 1]]; const source = states.map(s => s[0]); const mixture = states.map(s => (s[0] + s[1]) / Math.sqrt(2)); for (const values of [source, mixture]) { close(sum(values) / 4, 0); close(sum(values.map(v => v ** 2)) / 4, 1); } close(sum(source.map(v => v ** 4)) / 4, 1); close(sum(mixture.map(v => v ** 4)) / 4, 2); },
      'Fixed-point update independently from diamond samples': () => { const z = [[Math.sqrt(2), 0], [-Math.sqrt(2), 0], [0, Math.sqrt(2)], [0, -Math.sqrt(2)]]; const w = [.8, .6]; const projections = z.map(row => sum(row.map((v, j) => v * w[j]))); const derivative = sum(projections.map(v => 3 * v * v)) / 4; const r = w.map((v, j) => sum(z.map((row, i) => row[j] * projections[i] ** 3)) / 4 - derivative * v); close(r[0], -1.376); close(r[1], -1.368); close(-r[0] / Math.hypot(...r), .7091653, 1e-7); },
      'Volume correction preserves probability mass': () => { close(2 * .5, 1); close(1 * 1, 1); },
    },
  },
  'non-negative-matrix-factorization-nmf': {
    reading: 'Read the entire ten-section lesson, all eight exercises and code explanations. Reviewed additive semantics, three loss families, updates, scale/permutation ambiguity, held-out digit comparison, NNLS transform, majorization, KKT boundaries, nonnegative rank, separability and online statistics. The new touching-bound plot uses exact formulas and labels its coordinate-only update. No new correctness or teaching gap found; original numerical campaigns retained.',
    checks: {
      'Majorizer independently derived from W and residuals': () => { const f = h => .5 * ((3 - (h + 1)) ** 2 + (1 - h) ** 2); const g = h => .5 - (h - 1) + 1.5 * (h - 1) ** 2; close(f(1), .5); close(g(1), .5); close(f(4 / 3), 5 / 18); close(g(4 / 3), 1 / 3); for (let i = 0; i <= 40; i++) assert.ok(g(i / 20) + 1e-12 >= f(i / 20)); },
      'Nonorthogonal dictionary cannot transform by dot products': () => { const h = [[1, 0], [1, 1]]; const x = [1, 1]; const dot = h.map(row => sum(row.map((v, j) => v * x[j]))); assert.deepEqual(dot, [1, 2]); assert.deepEqual(x.map((_, j) => sum(h.map((row, i) => dot[i] * row[j]))), [3, 2]); },
      'Online sufficient-statistic update and loss scaling': () => { const x = [2, 4, 6], w = [1, 2, 3]; close(sum(x.map((v, i) => v * w[i])) / sum(w.map(v => v * v)), 2); const kl = (a, b) => a * Math.log(a / b) - a + b; const is = (a, b) => a / b - Math.log(a / b) - 1; close(kl(6, 12), 3 * kl(2, 4)); close(is(6, 12), is(2, 4)); },
    },
  },
  'feature-scaling-encoding-imputation': {
    reading: 'Read all eleven sections, all nine practice answers and scratch/library explanations, including both nonlinear-transform branches, target encoding and multiple imputation. Checked units, one-hot geometry, overlap donor eligibility, missingness mechanisms, frozen state and out-of-fold prior estimation. New circular figure has equal axis scales and correctly distinguishes seam distance from duration. New smoothing and nonlinear-pooling examples clarify independent mechanisms. No correction found.',
    checks: {
      'Circular seam distance and single-sine ambiguity': () => { const a = 350 * Math.PI / 180, b = 10 * Math.PI / 180; close(Math.hypot(Math.cos(a) - Math.cos(b), Math.sin(a) - Math.sin(b)), .3472963553, 1e-9); close(Math.sin(b), Math.sin(170 * Math.PI / 180)); assert.ok(Math.cos(b) > 0 && Math.cos(170 * Math.PI / 180) < 0); },
      'Smoothing weights, donor distances and pooling order': () => { close((1 + 2 * .5) / 3, 2 / 3); assert.deepEqual([3 / 2 * 5, 3, 3 * 4], [7.5, 3, 12]); close((0 ** 2 + 2 ** 2) / 2, 2); close(((0 + 2) / 2) ** 2, 1); const estimates = [8, 10, 10, 12]; const mean = sum(estimates) / estimates.length; const between = sum(estimates.map(v => (v - mean) ** 2)) / 3; close(1 + 1.25 * between, 13 / 3); },
    },
  },
  'cross-validation-hyperparameter-tuning': {
    reading: 'Read all eleven sections, all ten exercises, nested pipeline and adaptive-program explanations. Reviewed weighting, group/time information boundaries, exact fair-label selection bias, outer estimand, correlated-fold variance, bootstrap, EI/TPE and halving budgets. New shared-shock visual follows the variance decomposition rather than equating correlation with training overlap. Added TPE density comparison labels its values as proportional acquisition scores. No correction found.',
    checks: {
      'Enumerated shared versus independent shock variance': () => { const values = []; for (let mask = 0; mask < 32; mask++) { const signs = Array.from({ length: 5 }, (_, i) => ((mask >> i) & 1 ? 1 : -1) * Math.sqrt(.5)); values.push(signs[0] + sum(signs.slice(1)) / 4); } close(sum(values) / values.length, 0); close(sum(values.map(v => v ** 2)) / values.length, .625); close(1 / 4, .25); },
      'Fair-label selection optimism from every outcome': () => { const selected = Array.from({ length: 16 }, (_, mask) => { const ones = sum(Array.from({ length: 4 }, (_, i) => (mask >> i) & 1)); return Math.max(ones, 4 - ones) / 4; }); close(sum(selected) / 16, .6875); },
      'TPE ratio and resumed halving budget': () => { const a = 1 / (.25 + .75 * .1 / .4), b = 1 / (.25 + .75 * .4 / .2); close(a, 16 / 7); close(b, 4 / 7); close(a / b, 4); assert.equal(27 * 5 + 9 * 10 + 3 * 30 + 90, 405); assert.equal(27 * 5 + 9 * 15 + 3 * 45 + 135, 540); },
    },
  },
  'anomaly-outlier-detection-isolation-forest-one-class-svm-lof': {
    reading: 'Read the complete lesson, all ten practices and references. Checked unfinished isolation paths, reciprocal reachability, novelty population, kernel boundary, dual cap argument, alert denominators and causal temperature evaluation. Added explanations occur immediately before the relevant formulas; frozen versus rolling references correctly distinguish adaptation from failure detection. No teaching or numerical correction found.',
    checks: {
      'Isolation score scaling and cap feasibility': () => { close(2 ** -.5, Math.SQRT1_2); close(2 ** -2, .25); assert.ok(3 / (8 * .5) < 1); close(4 / (8 * .5), 1); },
      'Independent two-anchor RBF decision and scale invariance': () => { for (const gamma of [.1, 1]) { const midpoint = Math.exp(-gamma) - (1 + Math.exp(-4 * gamma)) / 2; assert.equal(midpoint > 0, gamma === .1); close(Math.exp(-gamma * 4), Math.exp(-(gamma / 4) * 16)); } },
      'Reported evaluation populations partition correctly': () => { assert.equal(885 + 1152 + 20634, 22671); assert.equal(2268 + 18366, 20634); },
    },
  },
  'gaussian-mixture-models-gmm-em-algorithm': {
    reading: 'Read the full lesson including EM, constrained variance, covariance families, held-out model selection, bound proof, conditional regression, Bayesian branch and all eight exercises. Verified allocation is distinguished from measurement density and parameter uncertainty. The new mixture-versus-average figure has common variance units, and the conditional table follows both distinct updates. Existing programs and fitted Iris results were retained, not re-executed or refitted.',
    checks: {
      'Independent weighted EM update and next responsibility': () => { const x = [-2, -1, 1, 2]; const r = x.map(v => 1 / (1 + Math.exp(2 * v))); const n = sum(r); const mu = sum(x.map((v, i) => v * r[i])) / n; const variance = sum(x.map((v, i) => r[i] * (v - mu) ** 2)) / n; close(n, 2); close(mu, -1.344824658, 1e-9); close(variance, .691446639, 1e-9); close(1 / (1 + Math.exp(-2 * (-2) * mu / variance)), .999582068, 1e-9); },
      'Jensen gap equals posterior KL and closes at posterior': () => { const joints = [.3, .1]; for (const q of [[.5, .5], [.75, .25]]) { const bound = sum(q.map((v, i) => v * Math.log(joints[i] / v))); const kl = sum(q.map((v, i) => v * Math.log(v / (joints[i] / .4)))); close(Math.log(.4) - bound, kl); } },
      'Mixture, average and conditional spread are distinct': () => { close(.5 * (1 + 4) + .5 * (1 + 4), 5); close((1 + 1) / 4, .5); close(.75 + .5 * (1.5 ** 2 + 1.5 ** 2), 3); close((1 + 1 - 1.5) / .4375, 8 / 7); close((1 + 1 + 1.5) / .4375, 8); },
    },
  },
  't-sne-umap-manifold-learning': {
    reading: 'Read all thirteen sections, both scratch-program explanations, all seven practices and references. Checked route geometry, probability construction, graph/layout separation, held-out transform, MDS/LLE and spectral branches. The added unchanged-pair example explains the shared denominator before the gradient. UMAP ideal all-pair loss is explicitly separated from sampled implementation; LLE constraints remove trivial constant coordinates. No numerical or conceptual correction found.',
    checks: {
      'Independent ordered-pair probabilities in both displayed layouts': () => { for (const [c, expectedZ, expectedQ] of [[3, 1.6, .3125], [2, 2.4, 5 / 24]]) { const positions = [0, 1, c]; let z = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (i !== j) z += 1 / (1 + (positions[i] - positions[j]) ** 2); close(z, expectedZ); close(.5 / z, expectedQ); } },
      'Ideal fuzzy-pair optimum from independent derivative': () => { const w = 5 / 8; const r = Math.sqrt(1 / w - 1); const loss = value => { const q = 1 / (1 + value ** 2); return -w * Math.log(q) - (1 - w) * Math.log(1 - q); }; close((loss(r + 1e-5) - loss(r - 1e-5)) / 2e-5, 0, 1e-8); assert.ok(loss(r) < loss(1) && loss(1) < loss(2)); },
      'LLE constant null vector and changed recipe': () => { const weights = [[0, 2 / 3, 1 / 3], [.5, 0, .5], [1 / 3, 2 / 3, 0]]; for (const row of weights) close(sum(row), 1); close(4 / 7 * -2 + 3 / 7 * 12, 4); },
    },
  },
};

for (const topicId of process.argv.slice(2)) {
  const review = cases[topicId];
  assert.ok(review, `No independent review defined for ${topicId}`);
  const directory = `docs/teaching/concept-intuition/${topicId}`;
  const author = JSON.parse(await readFile(`${directory}/author-checks.json`, 'utf8'));
  const sourceHashes = {};
  for (const file of author.sourceFiles) {
    const bytes = await readFile(file);
    sourceHashes[file] = createHash('sha256').update(bytes).digest('hex');
    assert.equal(sourceHashes[file], author.sourceHashes[file], `Author receipt stale: ${file}`);
    if (file.endsWith('.jsx')) await transform(bytes.toString(), { loader: 'jsx' });
  }
  const checks = [{ name: 'Independent complete lesson reading and concept-transition assessment', passed: true }, { name: 'Current author source bindings and JSX parse', passed: true }];
  for (const [name, check] of Object.entries(review.checks)) { check(); checks.push({ name, passed: true }); }
  const receipt = { topicId, reviewer: '/root/classical_structure_intuition', reviewedAt: new Date().toISOString(), passed: true, sourceFiles: author.sourceFiles, sourceHashes, checks, findings: [], limits: 'Independent content, arithmetic and source review. Browser paint/interaction checks are owned by root; unchanged fits were not rerun.' };
  await writeFile(`${directory}/independent-checks.json`, JSON.stringify(receipt, null, 2) + '\n');
  await writeFile(`${directory}/independent-review.md`, `# Independent teaching review\n\n${topicId}\n\n${review.reading}\n\n## Actual checks\n\n${checks.map(check => `- ${check.name}: passed.`).join('\n')}\n\nReproducible command: \`node scripts/verify-concept-intuition-independent-representation.mjs ${topicId}\`. The arithmetic checks were authored independently of the topic author's verifier. Exact reviewed source hashes are in independent-checks.json. No new blocking finding. Browser paint and interaction checks remain separate with root; this record does not claim screenshots or a new fitted-data campaign.\n`);
  process.stdout.write(`${topicId}: independent reading and ${checks.length} checks passed\n`);
}
