import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { variationalExamples as examples } from '../src/learn/data/variational-inference-examples.js';
const output = resolve('scratch/variational-inference-native-verification');
mkdirSync(output, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
for (const [name, example] of Object.entries(examples)) {
  const filename = resolve(output, `${name}.py`);
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-I', filename], { encoding: 'utf8', timeout: 30000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), name);
}
writeFileSync(resolve(output, 'programs.json'), JSON.stringify(examples));
const result = spawnSync(python, ['-I', resolve('scripts/verify-variational-inference-native.py'), output], { encoding: 'utf8', timeout: 120000 });
assert.equal(result.status, 0, result.stderr || result.stdout);
console.log(`Verified ${Object.keys(examples).length} exact Python outputs. ${result.stdout.trim()}`);
