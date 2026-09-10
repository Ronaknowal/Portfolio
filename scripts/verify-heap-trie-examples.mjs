import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { heapTrieExamples } from '../src/learn/data/heap-trie-examples.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixtureDirectory = resolve(repository, 'scratch/heap-trie-native-verification');
mkdirSync(fixtureDirectory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
let programs = 0;
for (const [name, example] of Object.entries(heapTrieExamples)) {
  const source = resolve(fixtureDirectory, `${name}.py`);
  writeFileSync(source, example.code, 'utf8');
  const result = spawnSync(python, ['-I', source], { encoding: 'utf8', env: environment, timeout: 20000, cwd: fixtureDirectory });
  assert.equal(result.error, undefined, `${name}: ${result.error?.message}`);
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), `${name}: displayed output`);
  programs++;
}
const manifest = resolve(fixtureDirectory, 'programs.json');
writeFileSync(manifest, JSON.stringify(heapTrieExamples), 'utf8');
const oracle = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-heap-trie-native.py'), manifest], {
  encoding: 'utf8', env: environment, timeout: 120000, cwd: fixtureDirectory,
});
assert.equal(oracle.error, undefined, oracle.error?.message);
assert.equal(oracle.status, 0, oracle.stderr || oracle.stdout);
console.log(`Verified ${programs} complete displayed Python programs against their displayed stdout.`);
console.log(oracle.stdout.trim());
