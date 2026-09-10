import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { stringMatchingExamples } from '../src/learn/data/string-matching-examples.js';
import { kmpTrace, prefixFunction, prefixTrace, rollingTrace, streamTrace, symbols, zFunction } from '../src/learn/data/string-matching-models.js';

const directory = resolve('scratch/string-matching-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [key, example] of Object.entries(stringMatchingExamples)) {
  const file = resolve(directory, `${key}.py`);
  writeFileSync(file, example.code);
  const execution = spawnSync(python, ['-X', 'utf8', '-I', file], { encoding: 'utf8', env: environment, timeout: 20000 });
  assert.equal(execution.status, 0, `${key}: ${execution.stderr}`);
  assert.equal(execution.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), key);
}
writeFileSync(resolve(directory, 'examples.json'), JSON.stringify(stringMatchingExamples));

function words(alphabet, maximum) {
  const result = [''];
  let layer = [''];
  for (let length = 1; length <= maximum; length++) {
    layer = layer.flatMap(prefix => alphabet.map(letter => prefix + letter));
    result.push(...layer);
  }
  return result;
}
function positions(text, pattern) {
  const letters = symbols(text), needle = symbols(pattern);
  return Array.from({ length: Math.max(0, letters.length - needle.length + 1) }, (_, start) => start)
    .filter(start => needle.every((letter, index) => letters[start + index] === letter));
}
function borders(sequence) {
  return sequence.map((_, position) => {
    let best = 0;
    for (let length = 1; length <= position; length++) {
      if (sequence.slice(0, length).every((letter, index) => letter === sequence[position + 1 - length + index])) best = length;
    }
    return best;
  });
}
function directHash(sequence, base, modulus) {
  return Number(sequence.reduce((sum, letter, index) => sum + BigInt(letter.codePointAt(0)) * BigInt(base) ** BigInt(sequence.length - index - 1), 0n) % BigInt(modulus));
}
const counts = { nativePrograms: Object.keys(stringMatchingExamples).length, prefixPatterns: 0, prefixStates: 0, kmpCases: 0, kmpStates: 0, rollingCases: 0, rollingWindows: 0, streamPartitions: 0, zPatterns: 0, invalidCases: 0 };
for (const pattern of [...words(['a', 'b'], 8), '🙂a🙂a🙂', 'e\u0301e\u0301', 'aabaaab', 'aabaaac']) {
  const letters = symbols(pattern), expected = borders(letters);
  assert.deepEqual(prefixFunction(letters), expected);
  const trace = prefixTrace(pattern);
  assert.deepEqual(trace.table, expected);
  for (const state of trace.states) {
    state.table.forEach((value, index) => { if (value !== null) assert.equal(value, expected[index]); });
    if (letters.length) {
      const end = state.position + (['equal', 'commit', 'initial'].includes(state.kind) ? 1 : 0);
      assert.deepEqual(letters.slice(0, state.matched), letters.slice(end - state.matched, end));
    }
    counts.prefixStates++;
  }
  const expectedZ = letters.map((_, start) => {
    if (start === 0) return 0;
    let length = 0;
    while (start + length < letters.length && letters[length] === letters[start + length]) length++;
    return length;
  });
  assert.deepEqual(zFunction(letters), expectedZ);
  counts.zPatterns++;
  counts.prefixPatterns++;
}
const texts = [...words(['a', 'b'], 7), 'a🙂a🙂a', 'e\u0301é', 'adbaad'];
const patterns = [...words(['a', 'b'], 4), '🙂a', 'é', 'ba'];
for (const text of texts) {
  for (const pattern of patterns) {
    const expected = positions(text, pattern), trace = kmpTrace(text, pattern);
    assert.deepEqual(trace.matches, expected, `${text} / ${pattern}`);
    let previousConsumed = 0;
    for (const state of trace.states) {
      assert(state.consumed >= previousConsumed);
      assert.deepEqual(trace.letters.slice(state.consumed - state.matched, state.consumed), trace.needle.slice(0, state.matched));
      assert(state.matches.every(start => expected.includes(start)));
      if (state.kind === 'fallback') assert.equal(state.consumed, previousConsumed);
      assert(state.comparisons <= 2 * trace.letters.length);
      previousConsumed = state.consumed;
      counts.kmpStates++;
    }
    counts.kmpCases++;
    for (const [base, modulus] of [[3, 7], [5, 101], [31, 1009]]) {
      const rolling = rollingTrace(text, pattern, base, modulus);
      assert.deepEqual(rolling.matches, expected);
      assert.equal(rolling.target, directHash(symbols(pattern), base, modulus));
      for (const row of rolling.rows) {
        assert.equal(row.value, directHash(row.window, base, modulus));
        assert.equal(row.exact, expected.includes(row.start));
        if (row.update) assert.equal(row.update.next, directHash(symbols(text).slice(row.start + 1, row.start + 1 + symbols(pattern).length), base, modulus));
        counts.rollingWindows++;
      }
      counts.rollingCases++;
    }
  }
}
for (const text of words(['a', 'b'], 5)) {
  for (const pattern of words(['a', 'b'], 3)) {
    for (let mask = 0; mask < 2 ** Math.max(0, text.length - 1); mask++) {
      const chunks = [''];
      for (let index = 0; index < text.length; index++) {
        chunks[chunks.length - 1] += text[index];
        if (mask & 2 ** index) chunks.push('');
      }
      chunks.splice(1, 0, '');
      const trace = streamTrace(chunks, pattern);
      assert.deepEqual(trace.matches, positions(text, pattern));
      trace.states.forEach(state => {
        const consumed = chunks.slice(0, state.fed).join('');
        assert.deepEqual(state.matches, positions(consumed, pattern));
        assert.equal(state.consumed, consumed.length);
      });
      counts.streamPartitions++;
    }
  }
}
assert.deepEqual(streamTrace(['ab', 'a'], 'aba', true).matches, []);
assert.deepEqual(streamTrace(['ab', 'a'], 'aba').matches, [0]);
assert(rollingTrace('adbaad', 'ba').rows.some(row => row.candidate && !row.exact));
for (const action of [() => prefixTrace('a'.repeat(19)), () => kmpTrace('a'.repeat(41), 'a'), () => streamTrace(Array(13).fill(''), 'a'), () => rollingTrace('a', 'a', 31, 7), () => rollingTrace('a', 'a', 2.5, 7), () => rollingTrace('a', 'a', 3, Infinity), () => prefixTrace(null)]) {
  assert.throws(action);
  counts.invalidCases++;
}
const native = spawnSync(python, ['-I', 'scripts/verify-string-matching-native.py'], { encoding: 'utf8', env: environment, timeout: 120000 });
assert.equal(native.status, 0, native.stderr || native.stdout);
const result = { checkedAt: new Date().toISOString(), ...counts, native: JSON.parse(native.stdout), scope: 'Independent slicing, proper-border, direct BigInt polynomial and native enumeration oracles; no benchmark or universal interview guarantee.' };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
