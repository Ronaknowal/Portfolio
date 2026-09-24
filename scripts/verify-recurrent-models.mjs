import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
import { scalarRecurrence, lstmAccounting, retentionPath, resetPlacement, recurrentSequence, boundaryExperiment, paddingExperiment, maxDifference, streamOwnership } from '../src/learn/data/recurrent-models.js';

const root = 'public/learn-assets/rnns-lstms-grus/';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const native = read('docs/teaching/evidence/recurrent-native.json');
assert.equal(native.passed, true);
for (const [path, hash] of Object.entries(native.sourceHashes)) assert.equal(createHash('sha256').update(fs.readFileSync(path)).digest('hex'), hash, path);
const data = read(root + 'pen-models.json'), evidence = read(root + 'recurrent-evidence.json');
const fixtures = read('docs/teaching/drafts/rnns-lstms-grus/fresh-investigation-fixtures.json');
const close = (a, b, tolerance = 1e-9) => assert.ok(maxDifference(a, b) < tolerance, `Difference ${maxDifference(a, b)} exceeds ${tolerance}`);
for (const row of native.pen) {
  const output = recurrentSequence(row.kind, row.points, data.weights[row.kind]);
  close(output.map(row => row.hidden), row.hidden);
  close(output.map(row => row.probabilities), row.probabilities);
}
for (const row of native.boundaries) {
  const result = boundaryExperiment(evidence.boundaries.sequence[0], evidence.boundaries.weights, row.boundary, row.mode);
  close(result.current.map(row => row.hidden), row.hidden);
  close(result.gradients, row.gradient, 1e-8);
  close([result.loss], [row.loss]);
}
for (const row of native.padding) {
  const result = paddingExperiment(evidence.boundaries.sequence[0], evidence.boundaries.padding.True.weights, row.length, row.padding);
  close(result.forward.map(row => row.hidden), row.forward);
  close(result.backward.map(row => row.hidden), row.backward);
  close(result.packedForward.map(row => row.hidden), row.validForward);
  close(result.packedBackward.map(row => row.hidden), row.validBackward);
  assert.equal(result.forwardEditError, 0);
}
const b = fixtures.B.default;
const scalar = scalarRecurrence({ inputs: b.inputs, initial: b.h0, inputWeight: b.parameters[0], recurrentWeight: b.parameters[1], bias: b.parameters[2], target: b.target, rate: b.rate });
close(scalar.states, b.states);
close(Object.values(scalar.gradient), b.gradient_wx_wh_b);
close(Object.values(scalar.updated), b.updated_parameters);
const cell = lstmAccounting({ cell: -.4, forget: .85, input: .3, candidate: .6, output: .7 });
close([cell.cell, cell.hidden], [-.16, -.11105395300726311]);
const reset = resetPlacement([-.5, 1.5], [.7, .3], [[.4, -1.2], [1.1, .5]], [.2, -.1]);
close(reset.before, [-.48, -.26]); close(reset.after, [-1.26, .03]);
const retention = retentionPath(.65, 80, 25, 0);
close([retention.forget, retention.halfLife], [.9946296855, 128.7232441], 1e-7);
close(retention.plain, retention.changed);
for (const boundary of [1, 2, 3, 4]) {
  const ownership = streamOwnership(evidence.boundaries.sequence[0], evidence.boundaries.weights, boundary, false);
  assert.equal(ownership.difference, 0);
}
for (const file of ['src/learn/data/topics/rnns-lstms-grus.jsx', 'src/learn/components/lesson-labs/RecurrentLabs.jsx']) parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const record = { passed: true, groups: ['27 native learned trajectories', '12 native forward/backward boundary states', '20 native bidirectional padding cases', 'Scalar forward/BPTT/update fixture', 'LSTM signed accounting and retention', 'GRU reset placement', 'Correct stream ownership', 'JSX syntax'], sourceHashes: Object.fromEntries(['src/learn/data/recurrent-models.js', 'src/learn/components/lesson-labs/RecurrentLabs.jsx', 'src/learn/components/lesson-labs/recurrent-labs.css', 'src/learn/data/topics/rnns-lstms-grus.jsx'].map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/recurrent-author.json', JSON.stringify(record, null, 2) + '\n');
console.log('PASS: eight recurrent model/author groups.');
