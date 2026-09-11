import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { dynamicProgrammingExamples } from '../src/learn/data/dynamic-programming-examples.js';
import practice from '../src/learn/data/practice/dynamic-programming-states-transitions-optimization.js';

const root = process.cwd();
const directory = 'docs/teaching/archive/dynamic-programming-before-state-families';
const recordPath = 'docs/teaching/evidence/dp-state-families-original.json';
if (fs.existsSync(recordPath)) throw new Error('Original archive already exists; do not overwrite it.');
const files = [
  'src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx',
  'src/learn/data/dynamic-programming-models.js',
  'src/learn/data/dynamic-programming-examples.js',
  'src/learn/components/lesson-labs/DynamicProgrammingLabs.jsx',
  'src/learn/components/lesson-labs/dynamic-programming-labs.css',
  'src/learn/data/practice/dynamic-programming-states-transitions-optimization.js',
  'src/learn/data/curriculum/blueprints/dynamic-programming-states-transitions-optimization.js',
  'docs/teaching/DYNAMIC-PROGRAMMING-LESSON-DESIGN.md',
  'docs/teaching/DYNAMIC-PROGRAMMING-VERIFICATION.md',
  'docs/teaching/topic-notes/dynamic-programming-states-transitions-optimization.md',
  'scripts/verify-dynamic-programming.mjs',
  'scripts/verify-dynamic-programming-native.py',
  'scripts/review-dynamic-programming-lesson.cjs',
  'scripts/review-dynamic-programming-layout.cjs',
];
for (const folder of ['scratch/dynamic-programming-verification', 'scratch/dynamic-programming-lesson-review']) {
  for (const name of fs.readdirSync(folder)) {
    if (fs.statSync(path.join(folder, name)).isFile()) files.push(`${folder}/${name}`);
  }
}
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const archived = files.map(source => {
  const bytes = fs.readFileSync(source);
  const destination = `${directory}/${source}${source.endsWith('.png') ? '' : '.txt'}`;
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, bytes, { flag: 'wx' });
  if (!bytes.equals(fs.readFileSync(destination))) throw new Error(`Archive mismatch: ${source}`);
  return { source, archive: destination, sha256: hash(bytes), bytes: bytes.length };
});
const inventory = execFileSync(process.execPath, ['scripts/build-curriculum-inventory.mjs', '--topic', 'dynamic-programming-states-transitions-optimization'], { cwd: root, encoding: 'utf8' });
fs.writeFileSync(`${directory}/topic-plan.txt`, inventory, { flag: 'wx' });
const programs = Object.entries(dynamicProgrammingExamples).map(([id, example]) => ({
  id, title: example.title, codeSha256: hash(example.code), expected: example.expected,
}));
if (programs.length !== 16) throw new Error('Unexpected program baseline.');
const problems = practice.groups.flatMap(group => group.problems.map(problem => ({ group: group.id, ...problem })));
if (problems.length !== 12) throw new Error('Unexpected practice baseline.');
fs.writeFileSync(recordPath, JSON.stringify({
  archivedAt: new Date().toISOString(), scope: 'Pre-extension byte archive; historical author evidence is retained without promoting it to new verification.',
  topicId: practice.topicId, files: archived, programs, problems, inventory: `${directory}/topic-plan.txt`,
}, null, 2) + '\n', { flag: 'wx' });
console.log(`Archived ${archived.length} files, ${programs.length} complete programs and ${problems.length} practice placements.`);
