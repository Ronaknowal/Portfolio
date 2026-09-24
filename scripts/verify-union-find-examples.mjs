import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { unionFindExamples } from '../src/learn/data/union-find-examples.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixtureDirectory = resolve(repository, 'scratch/union-find-native-verification');
mkdirSync(fixtureDirectory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [name, example] of Object.entries(unionFindExamples)) {
  const source = resolve(fixtureDirectory, `${name}.py`);
  writeFileSync(source, example.code, 'utf8');
  const result = spawnSync(python, ['-I', source], { encoding: 'utf8', env: environment, timeout: 20000 });
  assert.equal(result.error, undefined, `${name}: ${result.error?.message}`);
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), `${name}: displayed output`);
}
const manifest = resolve(fixtureDirectory, 'programs.json');
writeFileSync(manifest, JSON.stringify(unionFindExamples), 'utf8');
const result = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-union-find-native.py'), manifest], { encoding: 'utf8', env: environment, timeout: 120000 });
assert.equal(result.status, 0, result.stderr || result.stdout);
console.log(`Verified ${Object.keys(unionFindExamples).length} complete displayed Python programs and exact stdout.`);
console.log(result.stdout.trim());
