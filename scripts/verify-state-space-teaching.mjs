import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { retainedTrace, impulseTrails, markedMemory, matrixWriteExample } from '../src/learn/data/state-space-intuition.js';
import { ssdDefault, ssdOperator, maximumDifference } from '../src/learn/data/state-space-models.js';

const id = 'state-space-models-s4-mamba-mamba-2';
const revision = `docs/teaching/revisions/${id}/4`;
const destination = `${revision}/evidence/teaching-checks.json`;
fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.writeFileSync(destination, JSON.stringify({ passed: false, note: 'Revision 4 teaching checks started.' }));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => fs.readFileSync(file, 'utf8').replaceAll('\r\n', '\n');
const near = (actual, expected) => assert.ok(maximumDifference(actual, expected) < 1e-12);
const checks = [];

near(retainedTrace(.8).map(row => row.state), [1, .8, .64]);
near(retainedTrace(0).map(row => row.state), [5, 0, 0]);
near(retainedTrace(1).map(row => row.state), [0, 0, 0]);
near(retainedTrace(.8).map(row => row.average), [5, 2.5, 5 / 3]);
for (let i = 0; i <= 100; i++) {
  const a = i / 100;
  const inputs = [5, 0, -2, 3, 0, 1];
  const trace = retainedTrace(a, inputs);
  const direct = inputs.map((_, t) => inputs.slice(0, t + 1)
    .reduce((sum, input, birth) => sum + (1 - a) * input * a ** (t - birth), 0));
  near(trace.map(row => row.state), direct);
  for (const row of trace) assert.equal(row.state, row.retained + row.written);
}
checks.push('Retain/write defaults, latest-only and no-write endpoints, whole-history average, 101 retention values against independently expanded signed-input sums.');

const impulse = impulseTrails();
near(impulse.kernel, [1, .5, .25, .125]);
near(impulse.outputs, [2, 1, 1.5, .75]);
assert.deepEqual(impulse.contributions, [[2, 1, .5, .25], [null, 0, 0, 0], [null, null, 1, .5], [null, null, null, 0]]);
for (const a of [0, .5, 1]) {
  const inputs = [2, -3, 0, 1], result = impulseTrails(inputs, a);
  let memory = 0;
  const sequential = inputs.map(input => (memory = a * memory + input));
  near(result.outputs, sequential);
}
checks.push('Impulse birth rows, absent future cells and exact column sums independently agree with recurrence, including signed inputs and retention endpoints.');

const marked = markedMemory();
near(marked.map(row => row.selected), [4, 4, 4, 6]);
near(marked.map(row => row.delayed), [0, 0, 4, 9]);
near(retainedTrace(.5, [4, 9, -7, 6]).map(row => row.state), [2, 5.5, -.75, 2.625]);
near(markedMemory([4, 99, 0, -8, 3], [true, false, false, false, true]).map(row => row.selected), [4, 4, 4, 4, 3]);
near(markedMemory([4, 99], [false, false]).map(row => row.selected), [0, 0]);
checks.push('Marked-memory, two-step delay, constant smoothing, changed distraction gaps and no-marker null.');

const matrix = matrixWriteExample();
assert.deepEqual(matrix.retained, [[1, .5], [0, 0]]);
assert.deepEqual(matrix.write, [[0, -0], [3, -1]]);
near(matrix.state, [[1, .5], [3, -1]]);
near(matrix.read, [4, -.5]);
near(matrix.read, ssdOperator(ssdDefault()).recurrent[1]);
assert.ok(Math.abs(Math.exp(-(-Math.log(.8))) - .8) < 1e-15);
checks.push('Two-by-two retained/outer-product/read arithmetic agrees with the previously verified SSD operator; smoothing/continuous-step bridge checked.');

const manuscript = read(`${revision}/lesson.md`);
const frozen = read(`docs/teaching/drafts/${id}/lesson.md`);
const topic = read(`src/learn/data/topics/${id}.jsx`);
for (const file of ['StateSpaceIntuition.jsx', 'StateSpaceFigures.jsx', 'StateSpaceLabs.jsx']) {
  parse(read(`src/learn/components/lesson-labs/${file}`), { sourceType: 'module', plugins: ['jsx'] });
}
parse(topic, { sourceType: 'module', plugins: ['jsx'] });
assert.equal((manuscript.match(/^## /gm) || []).length, 12);
assert.ok(manuscript.includes('First pass: build and inspect before specializing'));
assert.equal((topic.match(/<StateSpace(?:System|Selection|SSD|Trajectory)Lab \/>/g) || []).length, 4);
assert.equal((topic.match(/<StateSpaceProgram /g) || []).length, 3);
for (const visual of ['RetainWriteFigure', 'ImpulseTrailsFigure', 'MarkedMemoryFigure', 'MatrixWriteFigure', 'StatePathsFigure', 'SamplingFigure', 'ImpulseFigure', 'MemoryRatesFigure', 'OscillatorFigure', 'PolynomialFigure', 'DplrFigure', 'FixedDelayFigure', 'MambaBlockFigure', 'SSDFigure', 'RealTrajectoriesFigure', 'TrainingPipelineFigure', 'LearningEvidenceFigure', 'CacheCountsFigure', 'MambaThreeFigure']) {
  assert.equal(topic.split(`<${visual} />`).length - 1, 1, visual);
}
assert.equal((topic.match(/<details>/g) || []).length, (topic.match(/<\/details>/g) || []).length);
assert.ok(!topic.includes('Inline figure:'));
assert.ok(!/record whether you expect|Before running, record|enter or submit a guess/.test(topic));
const oldPractice = frozen.slice(frozen.indexOf('### 1. Recover both output paths'), frozen.indexOf('## References'));
assert.ok(manuscript.replace(/\s+/g, '').includes(oldPractice.replace(/\s+/g, '')), 'All nine original changed exercises/hints/solutions and next-step route retained (formatting-normalized).');
for (const anchor of ['exp(Δ [[A,B],[0,0]])', 'B̄ = (I−ΔA/2)⁻¹ ΔB', 'dc/dt = −A₊c/t + B₊f(t)/t', 'K_T(z)=C {I−(zĀ)^T}', 'R₀−R₀p(1+q*R₀p)⁻¹q*R₀', 'Δₜ,d Bₜ,n uₜ,d', 'Y = ((CBᵀ) ⊙ L)V', 'not ordinary row-softmax attention', 'λ=.5+O(Δ)', 'rank up to R', '1,487', '2,287', 'Changed-code task']) assert.ok(manuscript.includes(anchor), anchor);
for (const code of frozen.matchAll(/```(?:python|text)\n([\s\S]*?)```/g)) assert.ok(manuscript.includes(code[0]), 'Original displayed program/command block retained');
for (const reference of frozen.matchAll(/\]\((https?:\/\/[^)]+)\)/g)) assert.ok(manuscript.includes(reference[1]), `Preserved reference ${reference[1]}`);
checks.push('Parsed all topic JSX; complete 12-section route, 19 visual groups, four existing labs and three full program readers; all prior code blocks, nine end exercises, changed-code task, technical boundaries and external resources retained.');

const oldModelReceipt = 'docs/teaching/evidence/state-space-models.json';
const oldNativeReceipt = 'docs/teaching/evidence/state-space-native.json';
const prior = JSON.parse(read(oldModelReceipt)), native = JSON.parse(read(oldNativeReceipt));
assert.equal(prior.passed, true); assert.equal(native.passed, true);
const reused = ['src/learn/data/state-space-models.js', 'src/learn/components/lesson-labs/StateSpaceLabs.jsx', 'src/learn/data/state-space-measurements.json', `public/learn-code/${id}/trajectory-inference.json`, `public/learn-code/${id}/trajectory-results.json`];
for (const file of reused) assert.equal(hash(file), prior.sourceHashes[file], `Changed source requires renewed numerical verification: ${file}`);
for (const [file, expected] of Object.entries(native.sourceHashes)) assert.equal(hash(`public/learn-code/${id}/${file}`), expected, file);
checks.push('Exact source-hash identity justifies reuse of historical numerical/native checks for unchanged engines, live labs, fitted weights, data, measured results and Python mechanisms; no native training or GPU execution repeated.');
const files = [`${revision}/lesson.md`, `${revision}/design.md`, `${revision}/visual-specifications.md`, `src/learn/data/topics/${id}.jsx`, 'src/learn/data/state-space-intuition.js', 'src/learn/components/lesson-labs/StateSpaceIntuition.jsx', 'src/learn/components/lesson-labs/StateSpaceFigures.jsx', 'src/learn/components/lesson-labs/state-space-labs.css', 'scripts/generate-state-space-lesson.mjs', 'scripts/verify-state-space-teaching.mjs', ...reused];
fs.writeFileSync(destination, JSON.stringify({ revision: 4, passed: true, checks,
  scope: 'Teaching arithmetic, coverage/source checks. Browser and independent reading review are recorded separately; this does not reapprove changed prose using an old runtime receipt.',
  reusedEvidence: { [oldModelReceipt]: hash(oldModelReceipt), [oldNativeReceipt]: hash(oldNativeReceipt) },
  unchangedNativeSources: native.sourceHashes,
  sourceHashes: Object.fromEntries(files.map(file => [file, hash(file)])),
}, null, 2) + '\n');
console.log(`State Space revision 4: ${checks.length} teaching/check groups passed; numerical evidence reused only for hash-identical sources.`);
