import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { poissonProblem, piecewiseLinearValue } from '../src/learn/data/numerical-pde-models.js';
import { numericalPdeExamples as examples } from '../src/learn/data/numerical-pde-examples.js';

const directory = 'scratch/numerical-pde-endpoint-amendment';
fs.mkdirSync(directory, { recursive: true });
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const before = read('docs/teaching/evidence/numerical-pdes-endpoint-before.json');
assert(before.files.every(file => hash(file.archive) === file.sha256));
const archived = source => before.files.find(file => file.source === source).archive;
const oldModule = async source => import('data:text/javascript;base64,' + fs.readFileSync(archived(source)).toString('base64'));
const oldModel = await oldModule('src/learn/data/numerical-pde-models.js');
const oldExamples = (await oldModule('src/learn/data/numerical-pde-examples.js')).numericalPdeExamples;
const originalPacket = read(archived('docs/teaching/evidence/numerical-pdes-author-review.json'));
const sourceHashes = Object.fromEntries(Object.keys(originalPacket.sourceHashes).map(file => [file, hash(file)]));
const reproducer = { intervals: 3, length: .7, profile: 'linear', scale: 1.7, left: -2.3, right: 4.1, method: 'jacobi', iterations: 9, tolerance: .03 };
let originalError;
try { oldModel.poissonProblem(reproducer); } catch (error) { originalError = String(error); }
assert(originalError, 'Archived source must reproduce the reported failure.');
const repaired = poissonProblem(reproducer);
assert.equal(repaired.nodes.at(-1), .7);
assert.equal(repaired.curve.at(-1).x, .7);
assert.equal(repaired.curve.at(-1).numerical, 4.1);
let cases = 0, previousFailures = 0, preservedInteriorCases = 0;
const fixtures = [];
for (const intervals of [3, 7, 11, 17, 31]) {
  for (const length of [.3, .7, 1.1, 1.7, 2.3]) {
    for (const profile of ['quartic', 'quadratic', 'linear']) {
      for (const method of ['direct', 'jacobi']) {
        const input = { ...reproducer, intervals, length, profile, method };
        const actual = poissonProblem(input);
        assert.equal(actual.nodes[0], 0);
        assert.equal(actual.nodes.at(-1), length);
        assert.equal(actual.curve[0].x, 0);
        assert.equal(actual.curve.at(-1).x, length);
        assert.equal(actual.curve[0].numerical, input.left);
        assert.equal(actual.curve.at(-1).numerical, input.right);
        assert.equal(piecewiseLinearValue(actual.nodes, actual.values, length), input.right);
        assert(actual.nodes.every((x, index) => index === 0 || x > actual.nodes[index - 1]));
        // Interior arithmetic and mathematical certificates are unchanged.
        assert.deepEqual(actual.nodes.slice(1, -1), Array.from({length: intervals - 1}, (_, j) => (j + 1) * length / intervals));
        let previous;
        try { previous = oldModel.poissonProblem(input); } catch { previousFailures += 1; }
        if (previous) {
          for (const key of ['values', 'source', 'rhs', 'residuals', 'direct', 'certificate']) assert.deepEqual(actual[key], previous[key]);
          preservedInteriorCases += 1;
        }
        fixtures.push({ input, nodes: actual.nodes, values: actual.values });
        cases += 1;
      }
    }
  }
}
assert.throws(() => piecewiseLinearValue([0, .25, .5], [0, 1, 2], .7));
const changedPrograms = [];
assert.deepEqual(Object.keys(examples), Object.keys(oldExamples));
for (const [key, example] of Object.entries(examples)) {
  assert.equal(example.expected, oldExamples[key].expected, key + ' stdout preserved');
  if (example.code !== oldExamples[key].code) {
    assert.equal(example.code.replace('    # The prescribed domain endpoints are exact inputs, not accumulated products.\n    nodes[0], nodes[-1] = 0.0, length\n', ''), oldExamples[key].code);
    assert.deepEqual({ ...example, code: oldExamples[key].code }, oldExamples[key]);
    changedPrograms.push(key);
  } else assert.deepEqual(example, oldExamples[key]);
}
assert.deepEqual(changedPrograms, ['poisson']);
const result = { checkedAt: new Date().toISOString(), passed: true, sourceHashes, originalError, reproducer, originalUnpinnedEndpoint: 3 * .7 / 3, finalEndpoint: repaired.nodes.at(-1), cases, previousFailures, preservedInteriorCases, changedPrograms, conservedStdoutPrograms: 16, unchangedExampleRecords: 15, strictOutsideDomainStillRejected: true };
fs.writeFileSync(path.join(directory, 'model-results.json'), JSON.stringify(result, null, 2) + '\n');
fs.writeFileSync(path.join(directory, 'native-fixtures.json'), JSON.stringify({ sourceHashes, fixtures, examples, oldExamples }, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
