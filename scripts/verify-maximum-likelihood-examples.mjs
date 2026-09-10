import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { maximumLikelihoodExamples as examples } from '../src/learn/data/maximum-likelihood-examples.js';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = resolve(root, 'scratch/maximum-likelihood-native-verification');
mkdirSync(output, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(root, 'scratch/lesson-tools/Scripts/python.exe');
for (const [name, example] of Object.entries(examples)) {
  const filename = resolve(output, `${name}.py`);
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-I', filename], { encoding: 'utf8', timeout: 20000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), name);
}
const manifest = resolve(output, 'programs.json');
writeFileSync(manifest, JSON.stringify(examples));
const result = spawnSync(python, ['-I', resolve(root, 'scripts/verify-maximum-likelihood-native.py'), manifest], { encoding: 'utf8', timeout: 120000 });
assert.equal(result.status, 0, result.stderr || result.stdout);
console.log(`Verified ${Object.keys(examples).length} exact Python outputs. ${result.stdout.trim()}`);
