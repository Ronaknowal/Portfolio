import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { graphTraversalExamples } from '../src/learn/data/graph-traversal-examples.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixtures = resolve(repository, 'scratch/graph-traversal-native-verification');
mkdirSync(fixtures, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [name, example] of Object.entries(graphTraversalExamples)) {
  const source = resolve(fixtures, `${name}.py`);
  writeFileSync(source, example.code, 'utf8');
  const result = spawnSync(python, ['-I', source], { encoding: 'utf8', env: environment, timeout: 20000, cwd: fixtures });
  assert.equal(result.error, undefined, `${name}: ${result.error?.message}`);
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), `${name}: displayed output`);
}
const manifest = resolve(fixtures, 'programs.json');
writeFileSync(manifest, JSON.stringify(graphTraversalExamples), 'utf8');
const oracle = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-graph-traversal-native.py'), manifest], {
  encoding: 'utf8', env: environment, timeout: 120000, cwd: fixtures,
});
assert.equal(oracle.error, undefined, oracle.error?.message);
assert.equal(oracle.status, 0, oracle.stderr || oracle.stdout);
console.log(`Verified ${Object.keys(graphTraversalExamples).length} complete displayed Python programs against their displayed stdout.`);
console.log(oracle.stdout.trim());
