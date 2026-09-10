import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { backtrackingDivideExamples } from '../src/learn/data/backtracking-divide-examples.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(repository, 'scratch/backtracking-divide-native-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [name, example] of Object.entries(backtrackingDivideExamples)) {
  const source = resolve(directory, `${name}.py`);
  writeFileSync(source, example.code);
  const result = spawnSync(python, ['-I', source], { encoding: 'utf8', env: environment, timeout: 20000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), name);
}
const manifest = resolve(directory, 'programs.json');
writeFileSync(manifest, JSON.stringify(backtrackingDivideExamples));
const result = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-backtracking-divide-native.py'), manifest], { encoding: 'utf8', env: environment, timeout: 120000 });
assert.equal(result.status, 0, result.stderr || result.stdout);
console.log(`Verified ${Object.keys(backtrackingDivideExamples).length} complete Python programs and exact stdout.`);
console.log(result.stdout.trim());
