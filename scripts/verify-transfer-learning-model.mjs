import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import { TRANSFER_METHODS, finiteIn, loraFixture, loraForward, loraStep, batchNormFixture, batchNormForward, adapterBudget, selectTransferCandidate } from '../src/learn/data/transfer-learning-model.js';

const packet = 'docs/teaching/drafts/transfer-learning-fine-tuning-strategies';
const bodyPath = 'src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx';
const labPath = 'src/learn/components/lesson-labs/TransferLearningLabs.jsx';
const reportPath = 'docs/teaching/evidence/transfer-learning-author-model.json';
const read = path => fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, '');
const data = JSON.parse(read(`${packet}/calculated-inputs.json`));
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const checks = [];
const check = (name, test) => { test(); checks.push(name); };
const close = (actual, expected, tolerance = 1e-9) => {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); expected.forEach((value, i) => close(actual[i], value, tolerance)); }
  else assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected} (tolerance ${tolerance})`);
};
if (!process.argv.includes('--no-evidence')) fs.writeFileSync(reportPath, JSON.stringify({ status: 'running', checkedAt: new Date().toISOString(), topicId: 'transfer-learning-fine-tuning-strategies', limits: 'This attempt has not passed all assertions. A failed or interrupted run must not reuse an older success record.' }, null, 2) + '\n');
const body = read(bodyPath), labs = read(labPath);
const ast = parse(body, { sourceType: 'module', plugins: ['jsx'] });
parse(labs, { sourceType: 'module', plugins: ['jsx'] });
check('Both runtime JSX modules parse', () => assert.ok(ast.program.body.length));
check('All published downloads and experiment records preserve exact packet bytes', () => {
  for (const file of ['transfer-experiments.py', 'digits-400.csv', 'data-provenance.md']) assert.equal(hash(`src/learn/assets/transfer-learning/${file}`), hash(`${packet}/${file}`));
  assert.equal(hash('src/learn/data/transfer-learning-experiment.json'), hash(`${packet}/calculated-inputs.json`));
});
check('Dataset identity, 400 unique rows, pixel domain and ten balanced digits', () => {
  assert.equal(hash(`${packet}/digits-400.csv`), data.dataset_sha256);
  const rows = read(`${packet}/digits-400.csv`).trim().split(/\r?\n/).slice(1).map(line => line.split(',').map(Number));
  assert.equal(rows.length, 400); assert.equal(new Set(rows.map(row => row[0])).size, 400);
  rows.forEach(row => { assert.equal(row.length, 66); row.slice(1, 65).forEach(value => assert.ok(Number.isInteger(value) && value >= 0 && value <= 16)); });
  for (let digit = 0; digit < 10; digit++) assert.equal(rows.filter(row => row[65] === digit).length, 40);
  const ids = Object.values(data.splits_source_ids).flat(); assert.equal(ids.length, 400); assert.equal(new Set(ids).size, 400);
  assert.deepEqual([...ids].sort((a, b) => a - b), rows.map(row => row[0]).sort((a, b) => a - b));
  assert.deepEqual(Object.values(data.splits_source_ids).map(values => values.length), [150, 50, 40, 60, 100]);
  const lookup = new Map(rows.map(row => [row[0], row]));
  const specimens = JSON.parse(read('src/learn/data/transfer-learning-specimens.json'));
  for (const row of [...specimens.training, ...specimens.test]) { assert.deepEqual(row.pixels, lookup.get(row.source_id).slice(1, 65)); assert.equal(row.digit, lookup.get(row.source_id)[65]); }
  assert.deepEqual(specimens.test.map(row => row.source_id), data.splits_source_ids.target_test);
  specimens.training.forEach(row => assert.ok(data.splits_source_ids[row.digit < 5 ? 'source_train' : 'target_train'].includes(row.source_id)));
  specimens.test.forEach((row, i) => assert.equal(row.digit - 5, data.selection.test_labels[i]));
});
check('Complete declared candidate sets, trace endpoints and denominators', () => {
  assert.equal(data.runs.filter(row => row.method).length, 18);
  for (const seed of [1, 2, 3]) {
    assert.deepEqual(data.runs.filter(row => row.seed === seed && row.method).map(row => row.method), TRANSFER_METHODS);
    for (const row of data.runs.filter(row => row.seed === seed && row.method)) {
      assert.deepEqual(row.trace.map(point => point.step), [0, 1, 10, 100, 300]);
      assert.deepEqual(row.trace.at(-1).validation, row.validation);
      assert.deepEqual(row.trace.at(-1).train, row.train);
      row.trace.forEach(point => { assert.equal(point.train.count, 40); assert.equal(point.validation.count, 60); assert.ok(point.train.ce >= 0 && point.validation.ce >= 0); });
      assert.equal(row.source_before.count, 50); if (row.source_after) assert.equal(row.source_after.count, 50);
      if (row.method === 'probe') assert.deepEqual(row.source_before, row.source_after);
      if (row.method === 'scratch') assert.equal(row.source_after, null);
    }
  }
});
check('Actual selection, unfavorable test report and probability reconstruction', () => {
  assert.equal(selectTransferCandidate(data.runs).winner.method, 'scratch');
  assert.equal(data.selection.seed, 1); assert.equal(data.selection.method, 'scratch');
  assert.equal(data.selection.test.correct, 77); assert.equal(data.selection.test.count, 100);
  let correct = 0, loss = 0;
  data.selection.test_probabilities.forEach((row, i) => { close(row.reduce((a, b) => a + b, 0), 1, 2e-7); const label = data.selection.test_labels[i]; if (row.indexOf(Math.max(...row)) === label) correct++; loss -= Math.log(row[label]); });
  assert.equal(correct, 77); close(loss / 100, data.selection.test.ce, 1e-6);
  assert.ok(data.selection.test.ce > selectTransferCandidate(data.runs).winner.validation.ce);
  assert.equal(data.selection.state_roundtrip_max_probability_error, 0);
});
check('Budget decisions, no eligible candidates, stable null contrast and no new test metrics', () => {
  for (const [budget, winner] of [[0, null], [84, null], [85, 'probe'], [250, 'probe'], [400, 'lora2'], [500, 'lora2'], [5000, 'scratch']]) assert.equal(selectTransferCandidate(data.runs, budget).winner?.method ?? null, winner);
  assert.deepEqual(selectTransferCandidate(data.runs, 400).candidates, selectTransferCandidate(data.runs, 500).candidates);
  const tied = data.runs.map(row => row.method ? { ...row, validation: { ...row.validation, ce: 1 } } : row);
  assert.equal(selectTransferCandidate(tied).winner.method, 'scratch');
  assert.ok(data.runs.every(row => !Object.hasOwn(row, 'test')));
});
check('LoRA default, changed input, null input and both-zero factor fixtures match PyTorch', () => {
  const initial = loraStep(loraFixture());
  close(initial.before.loss, data.mechanisms.loss);
  close(initial.before.gradA, data.mechanisms.A_gradient);
  close(initial.before.gradB, data.mechanisms.B_gradient);
  close(initial.after.output, data.mechanisms.after_output[0]);
  close(initial.after.loss, data.mechanisms.after_loss);
  for (const [input, key] of [[[1, 3], 'changed_input'], [[1, 1], 'null_measurement']]) {
    const result = loraStep({ ...loraFixture(), x: input }); const fixture = data.mechanisms.input_contrasts[key];
    close(result.before.gradB, fixture.B_gradient); close(result.after.output, fixture.after_output[0]); close(result.after.loss, fixture.after_loss);
  }
  const zero = loraForward({ ...loraFixture(), A: [[0, 0]] }); close(zero.gradA, [[0, 0]]); close(zero.gradB, [[0], [0]]); assert.equal(zero.loss, 2.5);
});
check('Independent finite differences verify both factor gradients and upstream gradient at ranks 1 and 2', () => {
  const step = 1e-6;
  for (const rank of [1, 2]) {
    const state = { ...loraFixture(rank), x: [-1.2, 0.7], target: [0.6, -0.4], alpha: 1.7, B: [[0.2, -0.4].slice(0, rank), [0.5, 0.3].slice(0, rank)] };
    const current = loraForward(state);
    for (const key of ['A', 'B']) state[key].forEach((row, i) => row.forEach((_, j) => {
      const high = structuredClone(state), low = structuredClone(state); high[key][i][j] += step; low[key][i][j] -= step;
      close((loraForward(high).loss - loraForward(low).loss) / (2 * step), current[key === 'A' ? 'gradA' : 'gradB'][i][j], 2e-8);
    }));
    state.x.forEach((_, i) => { const high = structuredClone(state), low = structuredClone(state); high.x[i] += step; low.x[i] -= step; close((loraForward(high).loss - loraForward(low).loss) / (2 * step), current.inputGradient[i], 2e-8); });
    close(current.output, current.mergedOutput, 1e-12);
    const immutable = structuredClone(state), stepped = loraStep(state); assert.deepEqual(state, immutable);
    stepped.next.A.forEach((row, i) => row.forEach((value, j) => close(value, state.A[i][j] - state.rate * current.gradA[i][j])));
    stepped.next.B.forEach((row, i) => row.forEach((value, j) => close(value, state.B[i][j] - state.rate * current.gradB[i][j])));
  }
});
check('All bounded LoRA corners stay finite, merge correctly and reject unsafe applied steps', () => {
  for (const rank of [1, 2]) for (const a of [-5, 0, 5]) for (const b of [-5, 0, 5]) for (const alpha of [0, 4]) {
    const state = { ...loraFixture(rank), A: Array.from({ length: rank }, () => [a, -a]), B: [[...Array(rank)].map(() => b), [...Array(rank)].map(() => -b)], x: [-5, 5], target: [5, -5], alpha, rate: 0.5 };
    const result = loraStep(state); assert.ok(Number.isFinite(result.after.loss)); close(result.before.output, result.before.mergedOutput, 1e-10);
    assert.equal(result.withinEditorBounds, [...result.next.A.flat(), ...result.next.B.flat()].every(value => Math.abs(value) <= 5));
    if (alpha === 0) { close(result.before.correction, [0, 0]); close(result.next.A, state.A, 0); close(result.next.B, state.B, 0); }
  }
});
check('BatchNorm state: train/no_grad changes buffers; eval preserves them; graph and ownership are independent', () => {
  const state = batchNormFixture(), result = batchNormForward(state);
  close(result.nextMean, data.mechanisms.frozen_bn_training_no_grad.mean[0]); close(result.nextVariance, data.mechanisms.frozen_bn_training_no_grad.variance[0]);
  const evaluated = batchNormForward({ ...state, mean: result.nextMean, variance: result.nextVariance, training: false, input: [11, 13] });
  close(evaluated.nextMean, 0.2); close(evaluated.nextVariance, 1.1);
  for (const training of [false, true]) for (const recording of [false, true]) for (const trainable of [false, true]) for (const optimizer of [false, true]) {
    const r = batchNormForward({ ...state, training, recording, trainable, optimizer });
    assert.equal(r.canUpdate, recording && trainable && optimizer);
    assert.equal(r.gradGamma !== null, recording && trainable);
    assert.equal(r.inputGradient !== null, recording);
    close(r.nextMean, training ? 0.2 : 0); close(r.nextVariance, training ? 1.1 : 1);
    close(r.output, batchNormForward({ ...state, training }).output);
  }
  assert.deepEqual(state, batchNormFixture());
});
check('Independent finite differences verify BatchNorm affine/input derivatives in both modes, including a constant batch', () => {
  const step = 1e-6;
  const loss = state => batchNormForward(state).output.reduce((sum, value) => sum + value * value, 0) / 2;
  for (const training of [false, true]) for (const input of [[1, 3], [2, 2], [-20, 20]]) {
    const state = { ...batchNormFixture(), input, gamma: 0.6, beta: 0.3, mean: 0.2, variance: 1.1, training, recording: true, trainable: true, optimizer: true };
    const result = batchNormForward(state);
    for (const [key, gradient] of [['gamma', result.gradGamma], ['beta', result.gradBeta]]) close((loss({ ...state, [key]: state[key] + step }) - loss({ ...state, [key]: state[key] - step })) / (2 * step), gradient, 1e-6);
    input.forEach((_, i) => { const high = structuredClone(state), low = structuredClone(state); high.input[i] += step; low.input[i] -= step; close((loss(high) - loss(low)) / (2 * step), result.inputGradient[i], 1e-6); });
  }
});
check('Exact adapter accounting, head toggles, units and validation domains', () => {
  assert.deepEqual(adapterBudget(16, 4), { down: 68, up: 80, adapter: 148, head: 85, count: 233, bytes: 2796 });
  assert.equal(adapterBudget(16, 4, false).count, 148);
  assert.equal(adapterBudget(1024, 64).count, 137285);
  assert.equal(8 * (1024 + 256) * 12, 122880);
  for (const invalid of [NaN, Infinity, -Infinity, -1, 5001]) assert.equal(finiteIn(invalid, 0, 5000, true), false);
  assert.equal(finiteIn(0.123456, 0, 0.5), true); assert.equal(finiteIn(3.5, 1, 64, true), false);
});

function walk(node, visit) { if (!node || typeof node !== 'object') return; visit(node); for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(child => walk(child, visit)); else if (value && typeof value === 'object') walk(value, visit); }
function jsxText(node) {
  if (node.type === 'JSXText') return node.value;
  if (node.type === 'JSXExpressionContainer' && node.expression.type === 'StringLiteral') return node.expression.value;
  return (node.children || []).map(jsxText).join('');
}
check('Every prepared prose paragraph and equation is retained; six independent practice hint/solution pairs', () => {
  const prose = [], equations = [];
  walk(ast, node => { if (node.type === 'JSXElement' && node.openingElement.name.type === 'JSXIdentifier') {
    if (node.openingElement.name.name === 'Prose') prose.push(jsxText(node));
    if (node.openingElement.name.name === 'MathBlock') equations.push(jsxText(node));
  }});
  const manuscript = read(`${packet}/lesson.md`).replace(/\r/g, '');
  const blocks = manuscript.split(/\n\s*\n/);
  const plain = text => text.replace(/\[([^\]]+)\]\([^\s]+\)/g, '$1').replace(/\\\((.*?)\\\)/g, '$1').replace(/`([^`]+)`/g, '$1').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1').replace(/\n/g, ' ');
  for (const block of blocks) {
    if (!block.trim() || /^(#|\||<|```|\d+\. |- |\\\[)/.test(block)) continue;
    assert.ok(prose.includes(plain(block.trim())), `Missing paragraph: ${block.slice(0, 90)}`);
  }
  const math = [...manuscript.matchAll(/\\\[\n([\s\S]*?)\n\\\]/g)].map(match => match[1]); assert.deepEqual(equations, math);
  assert.equal((body.match(/<summary>Hint<\/summary>/g) || []).length, 6);
  assert.equal((body.match(/<summary>Worked solution<\/summary>/g) || []).length, 6);
  assert.ok(!/<details\s+open/.test(body));
  for (const component of ['TransferReuseFigure', 'TransferFreezeLab', 'TransferPartitionsFigure', 'TransferProgram', 'TransferLoraLab', 'TransferAdapterBudget', 'TransferEvidenceLab', 'TransferCheckpointLab', 'TransferScheduleFigure']) assert.equal((body.match(new RegExp(`<${component} \\/>`, 'g')) || []).length, 1);
  assert.ok(body.includes('weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals'));
});

const rerunIndex = process.argv.indexOf('--rerun-json');
if (rerunIndex !== -1) check('Full final CPU program rerun equals every saved result byte for byte', () => assert.equal(hash(process.argv[rerunIndex + 1]), hash(`${packet}/calculated-inputs.json`)));
const sourcePaths = [bodyPath, labPath, 'src/learn/data/transfer-learning-model.js', 'src/learn/data/transfer-learning-experiment.json', 'src/learn/data/transfer-learning-specimens.json', 'src/learn/components/lesson-labs/transfer-learning.css', 'src/learn/assets/transfer-learning/transfer-experiments.py', 'src/learn/assets/transfer-learning/digits-400.csv', 'src/learn/assets/transfer-learning/data-provenance.md'];
const report = { status: 'passed', checkedAt: new Date().toISOString(), topicId: 'transfer-learning-fine-tuning-strategies', checkGroups: checks, sourceHashes: Object.fromEntries(sourcePaths.map(path => [path, hash(path)])), limits: 'Author calculation/source checks only; independent review and final production browser integration remain separate.' };
if (!process.argv.includes('--no-evidence')) fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ status: report.status, checkGroups: checks.length, rerunCompared: rerunIndex !== -1, evidence: process.argv.includes('--no-evidence') ? null : reportPath }));
