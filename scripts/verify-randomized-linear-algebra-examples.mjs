import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { randomizedLinearAlgebraExamples } from '../src/learn/data/randomized-linear-algebra-examples.js';

const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
for (const [name, example] of Object.entries(randomizedLinearAlgebraExamples)) {
  const result = spawnSync(python, ['-I', '-c', example.code], { encoding: 'utf8', timeout: 15000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replaceAll('\r\n', '\n').trimEnd(), example.expected, name);
}
mkdirSync('scratch/randomized-linear-algebra-review', { recursive: true });
writeFileSync('scratch/randomized-linear-algebra-review/programs.json', JSON.stringify(randomizedLinearAlgebraExamples));
console.log(`${Object.keys(randomizedLinearAlgebraExamples).length} complete NumPy programs match their displayed stdout.`);
