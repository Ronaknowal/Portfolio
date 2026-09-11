import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

const metadata = [
  ['matrix', 'matrix_chain', 'Plan a matrix chain and recover its operation order', 'Which grouping of the fixed matrix chain uses the fewest ordinary scalar multiplications, and which intermediate products realize it?'],
  ['balloons', 'balloons', 'Choose the last balloon, then recover the first removal', 'What is the maximum reward when every balloon must be removed, and does replaying the original-index witness earn that exact reward?'],
  ['tree', 'tree_boundary', 'Pass a parent boundary condition through a tree', 'Which nonadjacent original nodes maximize total weight, and how does an externally selected parent change the root answer?'],
  ['digits', 'digit_prefix', 'Count legal digit completions under an inclusive bound', 'How many positive integers have no repeated decimal digit up to the bound, and how do two prefix counts give an inclusive range answer?'],
];
const examples = {};
for (const [id, filename, title, question] of metadata) {
  const code = fs.readFileSync(`scripts/fixtures/dp-state-families/${filename}.py`, 'utf8').replace(/\r\n/g, '\n').trimEnd();
  const execution = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-'], { input: code, encoding: 'utf8', timeout: 15000 });
  if (execution.status !== 0) throw new Error(execution.stderr);
  examples[id] = { title, question, code, expected: execution.stdout.replace(/\r\n/g, '\n').trimEnd() };
  console.log(`${id}: ${examples[id].expected}`);
}
fs.writeFileSync('src/learn/data/dp-state-families-examples.js', `// Complete independently executed Python programs; regenerate with the semantic example script.\nexport const dpStateFamiliesExamples = ${JSON.stringify(examples, null, 2)};\n`);
