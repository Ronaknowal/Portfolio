import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parse } from '@babel/parser';
import { gatedFeedforwardTrace, blockDefault, traceBlock, normalizationProbe, movementBlockForward, blockCosts } from '../src/learn/data/transformer-block-models.js';
import { dense, softmax, layerNorm, maxDifference, gelu, attention } from '../src/learn/data/sequence-tensor-operations.js';

const id = 'transformer-block-architecture';
const source = `docs/teaching/drafts/${id}`;
const output = `docs/teaching/deep-learning-completion/${id}`;
const recorded = JSON.parse(fs.readFileSync(`${source}/author-results.json`));
const models = JSON.parse(fs.readFileSync(`${source}/block-models.json`));
const checks = [];
function close(name, actual, expected, tolerance = 1e-10) {
  const error = maxDifference(actual, expected);
  assert.ok(Number.isFinite(error) && error <= tolerance, `${name}: ${error} > ${tolerance}`);
  checks.push({ name, passed: true, maxAbsoluteError: error, tolerance });
}
assert.deepEqual(dense([2, 3], [[1, 4], [-2, 1]], [1, 2]), [15, 1]);
close('Shift-stable softmax', softmax([1000, 1001]), softmax([0, 1]));
assert.throws(() => softmax([-Infinity, -Infinity]));
assert.throws(() => attention([[1]], [[1]], [[1]], () => false));
close('LayerNorm common-offset invariance', layerNorm([1, 2, 5, 8]), layerNorm([6, 7, 10, 13]));
close('GELU independent reference at ±1', [gelu(1), gelu(-1)], [.8413447460685429, -.15865525393145707], 2e-7);
const fixture = blockDefault();
const gate = [[1,0,0],[0,-1,0],[0,0,1],[0,0,0]];
const gated = gatedFeedforwardTrace([1,2,-1,-2],fixture.up,gate,fixture.down);
const silu = x => x / (1+Math.exp(-x));
close('SwiGLU gate projection reads the actual input',gated.gateLogits,[1,-2,-1]);
close('SwiGLU value projection',gated.value,[2,4,3]);
close('SwiGLU signed write and output',gated.output,[2*silu(1),4*silu(-2),1.5*silu(-1),-1.5*silu(-1)]);
close('SwiGLU zero-gate projection null',gatedFeedforwardTrace([1,2,-1,-2],fixture.up,gate.map(r=>r.map(()=>0)),fixture.down).output,[0,0,0,0]);
const mapping = { attentionInput: 'attention_input', weights: 'weights', update: 'attention_update', context: 'context', ffnInput: 'ffn_input', hidden: 'hidden', ffnUpdate: 'ffn_update', output: 'output' };
for (const [name, settings] of [['pre', fixture], ['post', { ...fixture, pre: false }], ['zero_pre', { ...fixture, branch: 0 }], ['zero_post', { ...fixture, pre: false, branch: 0 }], ['edited_pre', { ...fixture, inputs: [[1, 2, 5, 8], [5, 0, 2, 1]] }]]) {
  const actual = traceBlock(settings);
  for (const [key, referenceKey] of Object.entries(mapping)) close(`${name}: ${key}`, actual[key], recorded.fixtures.trace[name][referenceKey]);
}
for (const input of [[1, 2, 5, 8], [-.03, -.01, .01, .03], [2, 2, 2, 2], [0, 0, 0, 0]]) {
  for (const probe of [[1, 1, 1, 1], [1, -1, 0, 0], [2, -.5, .2, 4]]) {
    const actual = normalizationProbe(input, probe, 1e-5, 1e-6);
    close(`Analytic/finite-difference ${input}/${probe}`, actual.gradient, actual.finite, 1e-4);
  }
}
close('Exact copy entropy denominator', [8 / 18 * Math.log(10)], [recorded.fixtures.copy_loss_floor]);
assert.equal(blockCosts(512).maps, 1610612736);
assert.equal(blockCosts(4096).pairs, 17179869184);
for (const [placement, saved] of Object.entries(models)) {
  const model = { ...saved, preNorm: placement === 'pre-norm' };
  const original = movementBlockForward(model, saved.points, saved.times);
  close(`${placement} retained original logits`, original.logits, saved.original.logits, 1e-4);
  const cases = {
    paired_permutation: [[...saved.points].reverse(), [...saved.times].reverse(), []],
    reverse_coordinates_fixed_time: [[...saved.points].reverse(), saved.times, []],
    frame23_x_reflection: [saved.points.map((row, i) => i === 22 ? [1 - row[0], row[1]] : row), saved.times, []],
    masked_padding: [[...saved.points, ...Array.from({ length: 5 }, () => [.75, .75])], [...saved.times, 0, 0, 0, 0, 0], [...Array(45).fill(false), ...Array(5).fill(true)]],
    unmasked_padding: [[...saved.points, ...Array.from({ length: 5 }, () => [.75, .75])], [...saved.times, 0, 0, 0, 0, 0], []],
  };
  for (const [name, args] of Object.entries(cases)) close(`${placement} ${name}`, movementBlockForward(model, ...args).logits, saved[name].logits, 1e-4);
  const traceNames = { input: 'input', attentionInput: 'attention_input', update: 'attention_update', residual: 'attention_residual', context: 'context', ffnInput: 'feedforward_input', hidden: 'feedforward_hidden', ffnUpdate: 'feedforward_update', output: 'output', weights: 'attention_weights' };
  for (let layer = 0; layer < 2; layer++) for (const [name, referenceName] of Object.entries(traceNames)) close(`${placement} layer ${layer} ${name}`, original.traces[layer][name], saved.trace[layer][referenceName], 1e-4);
  assert.throws(() => movementBlockForward(model, saved.points, saved.times, saved.points.map(() => true)));
  assert.throws(() => movementBlockForward(model, [], []));
}
for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/TransformerBlockLabs.jsx', 'src/learn/components/lesson-labs/TransformerBlockDiagrams.jsx', 'src/learn/components/lesson-labs/TransformerBlockMechanisms.jsx']) {
  parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  checks.push({ name: `Babel JSX parse ${file}`, passed: true });
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(`${output}/model-checks.json`, JSON.stringify({ topicId: id, passed: true, checks, limits: ['Float64 JavaScript versus retained float32 PyTorch checkpoints; 1e-4 absolute bound.', 'No hardware speed claim. Browser geometry and interactions reviewed separately.'] }, null, 2));
console.log(`${checks.length} Transformer primitive, stage, retained-model and syntax checks passed.`);
