import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { secondOrderMethodsExamples } from '../src/learn/data/second-order-methods-examples.js';

const destination = resolve('scratch/second-order-review/native');
mkdirSync(destination, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
const results = [];
for (const [name, example] of Object.entries(secondOrderMethodsExamples)) {
  const filename = resolve(destination, `${name}.py`);
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-X', 'utf8', '-I', filename], {
    encoding: 'utf8', timeout: 60000,
  });
  assert.equal(result.status, 0, `${name}: ${result.stderr || result.error}`);
  const stdout = result.stdout.replace(/\r\n/g, '\n').trim();
  assert.equal(stdout, example.expected.trim(), `${name}: displayed output differs`);
  results.push({ name, stdout });
}
writeFileSync(resolve(destination, 'verification.json'), JSON.stringify({
  checkedAt: new Date().toISOString(), status: 'passed', results,
}, null, 2) + '\n');
console.log(`Passed ${results.length} complete second-order programs and exact displayed outputs.`);
