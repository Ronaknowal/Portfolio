import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { loopWork, loopPatterns, parseSumValues, sumCallTrace, recurrenceLevels, recurrencePatterns } from '../src/learn/data/complexity-recursion-models.js';

let loopCases = 0;
for (let size = 0; size <= 32; size += 1) {
  for (const pattern of Object.keys(loopPatterns)) {
    const model = loopWork(pattern, size);
    const expected = pattern === 'square' ? size ** 2 : pattern === 'triangle' ? size * (size - 1) / 2 : size * (size > 1 ? Math.ceil(Math.log2(size)) : 0);
    assert.equal(model.total, Math.abs(expected));
    for (const row of model.rows) {
      for (let column = 0; column < size; column += 1) {
        const belongs = pattern === 'square' || (pattern === 'triangle' ? column < row.outer : column > 0 && Number.isInteger(Math.log2(column)));
        assert.equal(row.columns.includes(column), belongs);
      }
    }
    loopCases += 1;
  }
}
let recurrenceCases = 0;
for (const size of [1, 2, 4, 8, 16, 32]) {
  for (const pattern of Object.keys(recurrencePatterns)) {
    const model = recurrenceLevels(pattern, size);
    const expected = { chainConstant: size, chainLinear: size * (size + 1) / 2, halfConstant: Math.log2(size) + 1, halfLinear: 2 * size - 1, twoHalfLinear: size * (Math.log2(size) + 1) }[pattern];
    assert.equal(model.totalWork, expected);
    assert.equal(model.totalCalls, pattern === 'twoHalfLinear' ? 2 * size - 1 : model.peakFrames);
    assert.equal(model.peakFrames, pattern.startsWith('chain') ? size : Math.log2(size) + 1);
    assert.equal(model.levels.at(-1).size, 1);
    recurrenceCases += 1;
  }
}
let frameCases = 0;
const nativeTraces = [];
for (let length = 0; length <= 8; length += 1) {
  for (let variant = 0; variant < 31; variant += 1) {
    const values = Array.from({ length }, (_, index) => ((variant * 13 + index * 7) % 199) - 99);
    const trace = sumCallTrace(values);
    assert.equal(trace.at(-1).result, values.reduce((sum, value) => sum + value, 0));
    assert.equal(trace.at(-1).calls, length + 1);
    assert.equal(Math.max(...trace.map(state => state.frames.length)), length + 1);
    assert.equal(trace.length, 3 * length + 3);
    for (const state of trace) {
      assert.deepEqual(state.frames.map(frame => frame.index), Array.from({ length: state.frames.length }, (_, index) => index));
      for (const frame of state.frames) {
        if (frame.phase === 'return') assert.equal(frame.value, values.slice(frame.index).reduce((sum, value) => sum + value, 0));
      }
    }
    const copy = sumCallTrace(values);
    trace[0].frames[0].index = -1;
    assert.notEqual(trace[1].frames[0].index, -1);
    assert.equal(copy[0].frames[0].index, 0);
    nativeTraces.push({ values, states: copy.map(({ message, ...state }) => state) });
    frameCases += 1;
  }
}
assert.deepEqual(parseSumValues(''), []);
assert.deepEqual(parseSumValues(' -3, 0, 99 '), [-3, 0, 99]);
for (const bad of ['1,', '2.5', 'NaN', '100', '1,2,3,4,5,6,7,8,9']) assert.throws(() => parseSumValues(bad), RangeError);
for (const invalid of [-1, 33, 1.5, NaN]) assert.throws(() => loopWork('triangle', invalid), RangeError);
for (const invalid of [0, 3, 64, NaN]) assert.throws(() => recurrenceLevels('twoHalfLinear', invalid), RangeError);
assert.throws(() => loopWork('unknown', 4), RangeError);
assert.throws(() => recurrenceLevels('unknown', 4), RangeError);
assert.throws(() => sumCallTrace([Infinity]), RangeError);
mkdirSync('scratch/complexity-recursion-verification', { recursive: true });
writeFileSync('scratch/complexity-recursion-verification/model-traces.json', JSON.stringify(nativeTraces));
writeFileSync('scratch/complexity-recursion-verification/model-results.json', JSON.stringify({ loopCases, recurrenceCases, frameCases, invalidBoundaries: 'passed' }, null, 2));
console.log(`PASS: ${loopCases} loop shapes, ${recurrenceCases} recurrence configurations, ${frameCases} complete frame traces and invalid boundaries.`);
