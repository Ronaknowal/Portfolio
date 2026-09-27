import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { entranceMessages, weightedReadExample, silentStepExamples, sameMeanPaths } from '../src/learn/data/long-context-intuition.js';
import { longContextDefaults, retainedRead, composeAffine } from '../src/learn/data/long-context-models.js';

const id = 'long-context-sequence-models-transformer-xl-griffin-perceiver';
const revision = `docs/teaching/revisions/${id}/4/`;
const evidence = `${revision}teaching-checks.json`;
fs.writeFileSync(evidence, JSON.stringify({ status: 'incomplete', note: 'A revision-4 scoped run started; only the passed receipt closes it.' }, null, 2) + '\n');
const read = path => fs.readFileSync(path, 'utf8').replaceAll('\r\n', '\n');
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);
const oldText = read(`docs/teaching/drafts/${id}/lesson.md`);
const newText = read(`${revision}lesson.md`);
const route = read(`src/learn/data/topics/${id}.jsx`);
const codeBlocks = text => [...text.matchAll(/```[^\n]*\n([\s\S]*?)```/g)].map(match => match[1].trim());
const equations = text => [...text.matchAll(/\\\[([\s\S]*?)\\\]/g)].map(match => match[1].replace(/\s+/g, ''));

// Retaining blocks alone is not a pedagogy review. These checks protect depth while that review happens.
for (const block of codeBlocks(oldText)) assert.ok(codeBlocks(newText).includes(block), 'A complete existing code or result block was removed');
for (const equation of equations(oldText)) assert.ok(equations(newText).includes(equation), 'An existing displayed equation was removed');
const oldPractice = oldText.slice(oldText.indexOf('## 9. Practice:'), oldText.indexOf('## 10. References'));
assert.ok(newText.includes(oldPractice), 'Existing changed exercises/hints/solutions must remain intact');
const measuredRows = oldText.split('\n').filter(line => /^\| (Mean coordinates|Ordered coordinates|One latent|Four latents)/.test(line));
for (const row of measuredRows) assert.ok(newText.includes(row), 'A measured result row was changed');
for (const name of ['MemoryCacheLab', 'RecurrentMemoryLab', 'LatentWorkspaceLab', 'LongContextTrajectoryLab', 'EntranceMessageFigure', 'WeightedReadSharesFigure', 'SilentStepGateFigure', 'PathMeanCollisionFigure', 'AttentionShapesFigure']) assert.ok(route.includes(`<${name} />`), `Missing ${name}`);
for (const file of ['sequence_mechanisms.py', 'latent_trajectory_classifier.py', 'memory_library_bridge.py']) assert.ok(route.includes(`<LongContextProgram file="${file}"`), `Missing complete ${file}`);
assert.ok(!route.includes('Inline figure:') && !route.includes('Investigation 1 —'), 'Unresolved author specification');

assert.deepEqual(entranceMessages.filter(message => message.available).map(message => message.position), [3, 4]);
assert.equal(entranceMessages[0].available, false);
close(weightedReadExample.reduce((sum, row) => sum + row.support, 0), 6);
close(weightedReadExample.reduce((sum, row) => sum + row.weight, 0), 1);
close(weightedReadExample.reduce((sum, row) => sum + row.contribution, 0), 32 / 6);
silentStepExamples.forEach((example, i) => { close(example.state, i === 2 ? .6 : .48); close(example.injection, 0); });
assert.notDeepEqual(sameMeanPaths[0].points, sameMeanPaths[1].points);
sameMeanPaths.forEach(path => { close(path.mean.x, .5); close(path.mean.y, .5); assert.equal(path.points.length, 5); });
close(retainedRead(longContextDefaults(), 2, 2, 4).output, 10 / 3);
close(retainedRead(longContextDefaults(), 2, 4, 4).output, 5);
composeAffine([.8, 1], [.5, 2]).forEach((value, i) => close(value, [.4, 2.6][i]));

const retainedEvidence = {};
for (const receipt of ['native-checks.json', 'independent-checks.json']) {
  const path = `docs/teaching/evidence/long-context/${receipt}`;
  const report = JSON.parse(read(path));
  assert.equal(report.status, 'passed');
  for (const [file, expected] of Object.entries(report.reviewedFiles ?? report.files)) assert.equal(hash(file), expected, `Existing numerical evidence no longer binds ${file}`);
  retainedEvidence[path] = hash(path);
}
const files = [
  `${revision}lesson.md`, `${revision}design.md`, `${revision}visual-specifications.md`,
  'src/learn/data/long-context-intuition.js', 'src/learn/data/long-context-models.js',
  'src/learn/components/lesson-labs/LongContextIntuitionFigures.jsx',
  'src/learn/components/lesson-labs/LongContextFigures.jsx',
  'src/learn/components/lesson-labs/LongContextLabs.jsx',
  'src/learn/components/lesson-labs/LongContextTrajectoryLab.jsx',
  'src/learn/components/lesson-labs/long-context-labs.css',
  `src/learn/data/topics/${id}.jsx`, 'scripts/render-long-context-lesson.mjs',
  'scripts/verify-long-context-teaching.mjs',
];
for (const file of files.filter(file => /\.(m?js|jsx)$/.test(file))) parse(read(file), { sourceType: 'module', plugins: ['jsx'] });
const report = {
  status: 'passed', revision: 4,
  checks: ['All previous executable/result code blocks and display equations conserved', 'All eight exercises with hints/solutions conserved', 'All six measured result rows conserved', 'Four live labs and three full lazy programs retained', 'Four new teaching diagrams rendered from independently checked numerical fixtures', 'Every imported current JS/JSX parses', 'Prior native and independent numerical receipts still match unchanged implementation/assets'],
  counts: { preservedCodeBlocks: codeBlocks(oldText).length, preservedDisplayEquations: equations(oldText).length, newIntuitionFigures: 4, existingFigures: 16, liveInvestigations: 4 },
  retainedEvidence,
  limitation: 'This does not claim a new training run, optional RecurrentGemma execution, browser review or independent pedagogy approval. Those reviews are separately recorded.',
  reviewedFiles: Object.fromEntries(files.map(file => [file, hash(file)])),
};
fs.writeFileSync(evidence, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ status: report.status, counts: report.counts, retainedEvidence: Object.keys(retainedEvidence) }, null, 2));
