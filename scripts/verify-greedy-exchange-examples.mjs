import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { greedyExchangeExamples } from '../src/learn/data/greedy-exchange-examples.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(repository, 'scratch/greedy-native-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
for (const [name, example] of Object.entries(greedyExchangeExamples)) {
  const filename = resolve(directory, `${name}.py`);
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-I', filename], { encoding: 'utf8', timeout: 20000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), name);
}
const manifest = resolve(directory, 'programs.json');
writeFileSync(manifest, JSON.stringify(greedyExchangeExamples));
const result = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-greedy-exchange-native.py'), manifest], { encoding: 'utf8', timeout: 120000 });
assert.equal(result.status, 0, result.stderr || result.stdout);
console.log(`Verified ${Object.keys(greedyExchangeExamples).length} complete Python programs and exact stdout.`);
console.log(result.stdout.trim());
