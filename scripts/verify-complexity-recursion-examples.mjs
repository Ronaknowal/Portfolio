import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { complexityRecursionExamples } from '../src/learn/data/complexity-recursion-examples.js';
import './verify-complexity-recursion-models.mjs';

const directory = resolve('scratch/complexity-recursion-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
for (const [name, example] of Object.entries(complexityRecursionExamples)) {
  const filename = resolve(directory, `${name}.py`);
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-I', filename], { encoding: 'utf8', timeout: 20000, env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected, `${name}: actual output`);
}
writeFileSync(resolve(directory, 'examples.json'), JSON.stringify(complexityRecursionExamples));
const oracle = spawnSync(python, ['-I', resolve('scripts/verify-complexity-recursion-native.py'), directory], { encoding: 'utf8', timeout: 120000 });
assert.equal(oracle.status, 0, oracle.stderr || oracle.stdout);
console.log(`PASS: ${Object.keys(complexityRecursionExamples).length} exact displayed Python programs. ${oracle.stdout.trim()}`);
