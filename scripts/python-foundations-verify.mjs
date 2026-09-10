import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pythonFoundationsExamples as examples } from '../src/learn/data/python-foundations-examples.js';
import { pythonCoreExamples } from "../src/learn/data/python-core-examples.js";
import { referenceTrace, selectionTrace, functionTrace } from '../src/learn/data/python-foundations-model.js';

const root = path.resolve('scratch/python-foundations');
fs.mkdirSync(root, { recursive: true });
const work = fs.mkdtempSync(path.join(root, 'runtime-'));
const python = process.env.PYTHON_EXECUTABLE || 'python';
function run(code, filename = '-c') {
  const args = filename === '-c' ? ['-c', code] : [filename];
  const result = spawnSync(python, args, { cwd: work, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  return result.stdout.replace(/\r\n/g, '\n').trimEnd();
}
function json(code) { return JSON.parse(run(code)); }
const verified = [];
for (const [name, example] of Object.entries({ ...examples, ...Object.fromEntries(Object.entries(pythonCoreExamples).filter(([key]) => key.startsWith('basics') && key !== 'basicsProject')) })) {
  for (const [filename, code] of Object.entries(example.files || {})) fs.writeFileSync(path.join(work, filename), code);
  const filename = example.filename || `${name}.py`;
  fs.writeFileSync(path.join(work, filename), example.code);
  assert.equal(run('', filename), example.output, name);
  verified.push(name);
}
let comparisons = 0;
for (const copy of [false, true]) for (const action of ['append', 'rebind']) {
  const trace = referenceTrace(copy, action);
  const source = `import json\nscope = {}\nrows = []\nlines = ${JSON.stringify(trace.code.slice(0, 3))}\nfor line in lines:\n    exec(line, scope)\n    row = {name: list(scope[name]) for name in ['readings', 'backup'] if name in scope}\n    row['same'] = scope.get('readings') is scope.get('backup')\n    rows.append(row)\nprint(json.dumps(rows))`;
  const actual = json(source);
  for (let i = 0; i < actual.length; i++) {
    const state = trace.states[i + 1];
    for (const [name, object] of Object.entries(state.names)) assert.deepEqual(state.objects[object], actual[i][name]);
    assert.equal(state.names.readings === state.names.backup, actual[i].same);
    comparisons++;
  }
}
for (const threshold of [-5, 0, 20, 30, 40]) for (const readings of [[18, null, 25, 31, 0], [], [null], [-3, 0, -3, 7]]) {
  const trace = selectionTrace(threshold, readings);
  const input = JSON.stringify(readings).replaceAll('null', 'None');
  // Python comprehension is an independent implementation of the desired selection.
  const actual = json(`import json\nvalues = ${input}\nprint(json.dumps([[x for x in values[:n] if x is not None and x >= ${threshold}] for n in range(len(values) + 1)]))`);
  for (const state of trace.states) {
    const processed = ['append', 'reject', 'skip'].includes(state.phase) ? state.index + 1 : state.phase === 'done' ? readings.length : Math.max(0, state.index);
    assert.deepEqual(state.accepted, actual[processed]);
    comparisons++;
  }
}
assert.throws(() => selectionTrace(NaN), /finite/);
for (const input of [0, 20, 100]) for (const mode of ['return', 'print']) {
  const trace = functionTrace(input, mode);
  const source = trace.code.slice(0, 4).join('\n');
  const actual = json(`import json, io, contextlib\nscope = {}\noutput = io.StringIO()\nwith contextlib.redirect_stdout(output):\n    exec(${JSON.stringify(source)}, scope)\nprint(json.dumps({'result': scope['result'], 'output': output.getvalue()}))`);
  assert.equal(trace.states[4].result, actual.result);
  assert.equal(trace.states[4].output.map(x => `${x}.0\n`).join(''), actual.output);
  const final = run(trace.code.join('\n'));
  assert.equal(final, trace.states.at(-1).output.map(v => v === null ? 'None' : `${v}.0`).join('\n'));
  comparisons += 3;
}
const contract = json(`import json\nfrom readings import summarize\nrows = []\nfor raw in [['0', '24'], ['-6', '0', ' ', '12'], [''], ['bad'], ['nan'], ['inf'], ['-inf']]:\n    before = list(raw)\n    try:\n        outcome = summarize(raw)\n    except ValueError as error:\n        outcome = str(error)\n    assert raw == before\n    rows.append(outcome)\nprint(json.dumps(rows))`);
assert.deepEqual(contract, [{ count: 2, mean: 12 }, { count: 3, mean: 2 }, 'no readings', "invalid reading: 'bad'", "reading must be finite: 'nan'", "reading must be finite: 'inf'", "reading must be finite: '-inf'"]);
const winnerFunction = pythonCoreExamples.basicsPractice.code.split('\nprint(')[0];
assert.equal(run(winnerFunction + `\nprint(best_model({'slow': -4, 'steady': -1, 'risky': -3}))`), 'steady');
assert.equal(run(winnerFunction.replace('score > best_score', 'score >= best_score') + `\nprint(best_model({'first': -2, 'second': -2}))`), 'second');
const result = { version: spawnSync(python, ['--version'], { encoding: 'utf8' }).stdout.trim(), examples: verified, modelComparisons: comparisons, projectContractCases: contract.length, independentTransferCases: 2, scratchDirectory: work };
fs.writeFileSync(path.join(root, 'runtime-results.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
