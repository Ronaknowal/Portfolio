import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { parse } from '@babel/parser';
import * as model from '../src/learn/data/seq2seq-models.js';
const id = 'sequence-to-sequence-encoder-decoder';
const reportPath = 'docs/teaching/evidence/seq2seq-author.json';
const report = { passed: false, checkedAt: new Date().toISOString(), groups: [], sourceHashes: {} };
fs.writeFileSync(reportPath, JSON.stringify(report));
const group = (name, check) => { check(); report.groups.push(name); };
const close = (left, right, tolerance = 1e-9) => assert.ok(Math.abs(left - right) <= tolerance, `${left} != ${right}`);
const mechanical = JSON.parse(fs.readFileSync(`public/learn-code/${id}/mechanics-results.json`));
const { weights } = JSON.parse(fs.readFileSync(`public/learn-code/${id}/seed-one-inference.json`));
group('Actual shifted target tracks distinguish current target, previous prefix and ignored storage', () => {
  const base = model.shiftedTracks('cared'), edit = model.shiftedTracks('carts');
  assert.deepEqual(base.inputs, ['<bos>', ...'cared']);
  assert.deepEqual(base.targets, [...'cared', '<eos>']);
  assert.deepEqual(base.inputs.slice(0, 4), edit.inputs.slice(0, 4));
  assert.notEqual(base.inputs[4], edit.inputs[4]);
  assert.equal(model.shiftedTracks('cared', 3).valid, 6);
  assert.deepEqual(model.shiftedTracks('a', 2).targets, ['a', '<eos>', '<pad>', '<pad>']);
  assert.deepEqual(model.shiftedTracks('a', 2).inputs, ['<bos>', 'a', '<eos>', '<pad>']);
  assert.deepEqual(model.shiftedTracks('a', 2, true).inputs, ['a', '<eos>', '<pad>', '<pad>']);
});
group('Scalar joint derivatives match saved autograd and numerical perturbations over changed parameters', () => {
  for (const [inputs, key] of [[[.2, .8], 'scalar_worked'], [[-.3, .6], 'scalar_fresh'], [[-.3, .2], 'scalar_fresh_input_edit']]) {
    const value = model.scalarBridge(inputs), reference = mechanical[key];
    close(value.context, reference.context); close(value.loss, reference.mean_loss);
    close(value.gradient, reference.encoder_input_weight_gradient); close(value.nextLoss, reference.updated_mean_loss);
  }
  for (const input of [[-1, 1], [.3, -.8], [0, 0]]) for (const weight of [-1.5, -.2, .7, 1.5]) {
    const central = (model.scalarBridge(input, weight + 1e-5).loss - model.scalarBridge(input, weight - 1e-5).loss) / 2e-5;
    close(model.scalarBridge(input, weight).gradient, central, 1e-8);
    const detached = model.scalarBridge(input, weight, .1, true);
    assert.equal(detached.gradient, 0); close(detached.loss, detached.nextLoss);
  }
  close(model.scalarBridge([-.3, .6], .7, 0).loss, model.scalarBridge([-.3, .6], .7, 0).nextLoss);
});
group('Probability tree paths normalize and frontier agrees with independent saved histories', () => {
  for (const [a, ae, be, width, key] of [[.55, .6, .85, 1, 'tree_fresh_greedy'], [.55, .6, .85, 2, 'tree_fresh_beam2'], [.55, .9, .85, 2, 'tree_fresh_branch_edit']]) {
    const tree = model.probabilityTree(a, ae, be, width), reference = mechanical[key];
    assert.equal(tree.winner.path, reference.best[0]); close(tree.winner.probability, reference.best[1]);
    tree.history.forEach((row, index) => row.candidates.forEach((candidate, rank) => { assert.equal(candidate.path, reference.history[index][rank][0]); close(candidate.probability, reference.history[index][rank][1]); }));
    close(tree.leaves.reduce((sum, leaf) => sum + leaf.probability, 0), 1);
  }
  close(model.lengthScore(-1.2, 2, 0), -1.2); close(model.lengthScore(-1.2, 2, 1), -1.0285714285714285);
  assert.ok(model.lengthScore(-1.4, 5, 1) > model.lengthScore(-1.2, 2, 1));
});
group('Complete saved source/prefix/context traces match independent double-precision NumPy states and probabilities', () => {
  const lactateContext = model.encodeSequence(weights, 'lactate', 'past').context;
  const cases = [['worked_lactate_past', 'lactate', 'past', {}], ['fresh_emmove_past', 'emmove', 'past', {}], ['fresh_emmove_participle', 'emmove', 'participle', {}], ['fresh_source_edit_emmode', 'emmode', 'past', {}], ['fresh_forced_first_a', 'emmove', 'past', { prefix: 'a' }], ['fresh_zero_context', 'emmove', 'past', { context: Array(64).fill(0) }], ['fresh_context_from_lactate', 'emmove', 'past', { context: lactateContext }]];
  let maximum = 0;
  for (const [name, lemma, feature, options] of cases) {
    const trace = model.generatedTrace(weights, lemma, feature, options), expected = mechanical.traces[name];
    assert.equal(trace.word, expected.generated); assert.equal(trace.ended, expected.ended_with_eos);
    assert.deepEqual(trace.tokens, expected.output_ids);
    trace.states.forEach((state, index) => state.forEach((value, coordinate) => close(value, expected.encoder_states[index][coordinate], 1e-10)));
    trace.steps.forEach((step, index) => {
      close(step.probabilities.reduce((sum, probability) => sum + probability, 0), 1, 1e-12);
      [0, 1, 3, 4, 5].forEach(token => assert.equal(step.probabilities[token], 0));
      step.state.forEach((value, coordinate) => close(value, expected.decoder[index].hidden[coordinate], 1e-10));
      step.probabilities.forEach((value, token) => { maximum = Math.max(maximum, Math.abs(value - expected.decoder[index].probabilities[token])); close(value, expected.decoder[index].probabilities[token], 1e-10); });
    });
  }
  report.maximumManualProbabilityDifference = maximum;
  assert.throws(() => model.generatedTrace(weights, 'Bad', 'past'));
  assert.throws(() => model.generatedTrace(weights, 'a', 'past', { prefix: '!' }));
  assert.throws(() => model.generatedTrace(weights, 'a', 'past', { cap: 17 }));
});
group('Replay, context identity, forced-prefix temporal boundary and explicit cap nulls', () => {
  const base = model.generatedTrace(weights, 'emmove', 'past');
  const forced = model.generatedTrace(weights, 'emmove', 'past', { prefix: 'a' });
  assert.deepEqual(forced.steps[0].probabilities, base.steps[0].probabilities);
  assert.notDeepEqual(forced.steps[1].probabilities, base.steps[1].probabilities);
  assert.deepEqual(model.generatedTrace(weights, 'emmove', 'past', { prefix: base.word.slice(0, 2) }).tokens, base.tokens);
  assert.deepEqual(model.generatedTrace(weights, 'emmove', 'past', { context: base.context }).tokens, base.tokens);
  const capped = model.generatedTrace(weights, 'emmove', 'past', { cap: 3 });
  assert.equal(capped.word, 'emo'); assert.equal(capped.ended, false);
  close(capped.logProbability, mechanical.fresh_cap_3.log_probability, 3e-6);
});
group('Candidate-owned beams agree with 27 actual native width1/2/3 runs and preserve source model bytes', () => {
  const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/seq2seq-native.json'));
  const before = JSON.stringify(weights), started = performance.now();
  for (const fixture of native.beamFixtures) {
    const candidate = model.sequenceBeam(weights, fixture.query.lemma, fixture.query.feature, fixture.width).candidates[0];
    assert.deepEqual(candidate.tokens, fixture.tokens);
    close(candidate.logProbability, fixture.log_probability, 3e-5);
  }
  assert.equal(JSON.stringify(weights), before);
  report.nativeBeamComparisonMilliseconds = performance.now() - started;
});
group('Owned JSX parses and preserves all section/practice/math and deferred resource contracts', () => {
  for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/Seq2SeqLabs.jsx']) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  const body = fs.readFileSync(`src/learn/data/topics/${id}.jsx`, 'utf8');
  assert.equal((body.match(/<H2>/g) || []).length, 12);
  assert.ok((body.match(/<MathBlock>/g) || []).length >= 9);
  assert.ok((body.match(/<summary>Solution/g) || []).length >= 9);
  assert.ok(!body.includes('{"$"}'));
  assert.ok(!body.includes('until its improved page is published'));
  assert.ok(!body.includes('Complete runnable program placement'));
  const labs = fs.readFileSync('src/learn/components/lesson-labs/Seq2SeqLabs.jsx', 'utf8');
  assert.ok(labs.includes('AbortController'));
  assert.ok(!labs.includes('import weights'));
  assert.ok(!/commitGuess|predictionEntry|submitGuess/.test(labs));
});
for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/Seq2SeqLabs.jsx', 'src/learn/components/lesson-labs/seq2seq-labs.css', 'src/learn/data/seq2seq-models.js', 'src/learn/data/seq2seq-measurements.json', `src/learn/data/curriculum/blueprints/${id}.js`, 'scripts/generate-seq2seq-lesson.mjs', 'scripts/verify-seq2seq-models.mjs']) report.sourceHashes[file] = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
report.passed = true;
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, groups: report.groups.length, maximumManualProbabilityDifference: report.maximumManualProbabilityDifference, nativeBeamComparisonMilliseconds: report.nativeBeamComparisonMilliseconds }));
