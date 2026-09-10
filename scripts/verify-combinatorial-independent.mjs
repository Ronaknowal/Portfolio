import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { assignmentTrace, scaledKnapsackState } from '../src/learn/data/combinatorial-optimization-models.js';
import { combinatorialOptimizationExamples } from '../src/learn/data/combinatorial-optimization-examples.js';

const directory = 'scratch/combinatorial-independent-review';
fs.mkdirSync(directory, { recursive: true });
const assignments = [];
for (const [workers, jobs] of [[2, 3], [3, 2], [3, 4], [4, 3], [4, 4]]) {
  for (let sample = 0; sample < 12; sample += 1) {
    const costs = Array.from({ length: workers }, (_, worker) =>
      Array.from({ length: jobs }, (_, job) => {
        const position = 17 * sample + 11 * worker + 7 * job + 3 * worker * job;
        return position % 7 === 0 ? null : (position % 9 - 4) * [1, 17, 250][sample % 3];
      }));
    for (let required = 0; required <= Math.min(workers, jobs); required += 1) {
      assignments.push(assignmentTrace({ costs, required }));
    }
  }
}
const scaling = [];
for (let sample = 0; sample < 18; sample += 1) {
  const items = Array.from({ length: 5 }, (_, index) => ({
    name: String(index),
    weight: 1 + (sample * 3 + index * 7) % 19,
    value: (sample * 11 + index * 13) % 31,
  }));
  for (const epsilonDenominator of [2, 7, 100]) {
    scaling.push(scaledKnapsackState({ items, capacity: sample, epsilonDenominator }));
  }
}
const files = [
  'src/learn/data/topics/combinatorial-optimization-approximation-algorithms.jsx',
  'src/learn/data/combinatorial-optimization-models.js',
  'src/learn/data/combinatorial-optimization-examples.js',
  'src/learn/components/lesson-labs/CombinatorialOptimizationLabs.jsx',
  'src/learn/components/lesson-labs/combinatorial-optimization-labs.css',
  'src/learn/data/curriculum/blueprints/combinatorial-optimization-approximation-algorithms.js',
];
const sourceHashes = files.map(file => ({ file, sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') }));
fs.writeFileSync(directory + '/fixtures.json', JSON.stringify({ assignments, scaling, examples: combinatorialOptimizationExamples, sourceHashes }));
const checked = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-combinatorial-independent.py'], { encoding: 'utf8' });
process.stdout.write(checked.stdout);
process.stderr.write(checked.stderr);
if (checked.status !== 0) process.exitCode = 1;
