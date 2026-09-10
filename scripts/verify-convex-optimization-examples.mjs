import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { convexOptimizationExamples } from '../src/learn/data/convex-optimization-examples.js';

const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const checks = [];
for (const [name, example] of Object.entries(convexOptimizationExamples)) {
  const result = spawnSync(python, ['-I', '-c', example.code], { encoding: 'utf8', timeout: 60000 });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replaceAll('\r\n', '\n').trimEnd(), example.expected, name);
  checks.push({ name, status: 'passed', stderr: result.stderr });
}
mkdirSync('scratch/convex-optimization-review', { recursive: true });
writeFileSync('scratch/convex-optimization-review/programs.json', JSON.stringify(convexOptimizationExamples));
writeFileSync('scratch/convex-optimization-review/example-results.json', JSON.stringify({ checkedAt: new Date().toISOString(), checks }, null, 2));
console.log(`${checks.length} complete NumPy/CVXPY programs match their displayed stdout.`);
