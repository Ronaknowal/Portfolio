import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
import { sectionLabel, headingId, collectLessonSections } from '../src/learn/components/lesson-navigation.js';
import { filesMatch } from './lib/lesson-delivery.mjs';

const directory = 'docs/teaching/lesson-navigation';
const hash = value => createHash('sha256').update(value).digest('hex');
const baseline = JSON.parse(fs.readFileSync(`${directory}/baseline.json`, 'utf8'));
const manifest = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
const ast = source => parse(source, { sourceType: 'module', plugins: ['jsx'] });
const name = node => node?.openingElement?.name?.name;
const attribute = (node, key) => node.openingElement.attributes.find(item => item.name?.name === key)?.value?.value;
const emptyProse = ast('const p = <Prose></Prose>').program.body[0].declarations[0].init;

// Only the three audited navigation wrappers and explicit placement metadata are
// normalized away. All other JSX, programs, expressions and imports must match.
function semanticTree(value, filename) {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(item => semanticTree(item, filename)).filter(item => item !== undefined);
  if (value.type === 'JSXText' && !value.value.trim()) return undefined;
  if (value.type === 'JSXAttribute' && value.name?.name === 'opening') return undefined;
  if (value.type === 'JSXElement') {
    if (filename.endsWith('/backprop.jsx') && name(value) === 'nav' && attribute(value, 'className') === 'backprop-section-route') return undefined;
    const isTransfer = filename.endsWith('/transfer-learning-fine-tuning-strategies.jsx') && attribute(value, 'className') === 'transfer-downloads';
    const isResidual = filename.endsWith('/residual-connections-skip-connections.jsx') && attribute(value, 'className') === 'res-route';
    if (name(value) === 'aside' && (isTransfer || isResidual) && value.children.some(child => name(child) === 'nav')) {
      const paragraphs = value.children.filter(child => name(child) === 'p');
      assert.equal(paragraphs.length, 1, `Unexpected custom opening: ${filename}`);
      value = { ...emptyProse, children: paragraphs[0].children };
    }
  }
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['start', 'end', 'loc', 'extra', 'comments', 'leadingComments', 'trailingComments', 'innerComments', 'tokens'].includes(key))
    .map(([key, item]) => [key, semanticTree(item, filename)]));
}
const semantics = (source, filename) => hash(JSON.stringify(semanticTree(ast(source), filename)));
const subjects = Object.values(manifest).map(file => `src/learn/data/${file.replace(/^\.\//, '')}`);
assert.equal(subjects.length, 234, 'Reassess coverage after publication changes');
let changed = 0;
const conservation = subjects.map(file => {
  const before = baseline.files[file];
  assert.ok(before, `Missing pre-change baseline: ${file}`);
  const current = fs.readFileSync(file, 'utf8');
  const expected = before.semanticSha256 ?? semantics(before.source, file);
  assert.equal(semantics(current, file), expected, `Lesson substance changed: ${file}`);
  if (hash(fs.readFileSync(file)) !== before.sha256) changed++;
  return { file, before: before.sha256, after: hash(fs.readFileSync(file)), semanticSha256: expected };
});

for (const [input, expected] of [
  ['1. Start here', { number: '1', title: 'Start here' }],
  ['12) Build it', { number: '12', title: 'Build it' }],
  ['3 — Compare cases', { number: '3', title: 'Compare cases' }],
  ['2.1: Two paths', { number: '2.1', title: 'Two paths' }],
  ['References and other ways to learn', { number: null, title: 'References and other ways to learn' }],
  ['3D coordinates', { number: null, title: '3D coordinates' }],
]) assert.deepEqual(sectionLabel(input), expected);
assert.equal(headingId('1. A → B?'), '1-a-b');

// A minimal DOM contract fixture checks existing aliases, colliding headings,
// lab exclusion and repeat collection without introducing a browser dependency.
const headings = [
  { textContent: '1. Start here', id: 'existing-link' },
  { textContent: '2) Build it', id: '' },
  { textContent: '2) Build it', id: '' },
  { textContent: 'A nested lab title', id: '', nested: true },
  { textContent: 'References', id: '' },
];
const document = { getElementById: id => id === '2-build-it' ? { alias: true } : headings.find(item => item.id === id) };
for (const item of headings) Object.assign(item, { ownerDocument: document, closest: () => item.nested, classList: { add() {} } });
const article = { ownerDocument: document, querySelectorAll: () => headings };
const first = collectLessonSections(article);
assert.deepEqual(first.map(item => item.id), ['existing-link', '2-build-it-2', '2-build-it-3', 'references']);
assert.deepEqual(collectLessonSections(article), first, 'Repeated collection changed anchors');
assert.ok(headings.filter(item => !item.nested).every(item => item.tabIndex === -1));
const result = { passed: true, checkedAt: new Date().toISOString(), publishedLessons: subjects.length,
  presentationEditedLessons: changed, checks: ['All published lesson ASTs preserved except explicit opening placement and three audited navigation wrappers', 'Numbers and unnumbered titles', 'Stable anchors, aliases, collisions, lab exclusion and repeated collection'], conservation };
if (process.argv.includes('--record')) fs.writeFileSync(`${directory}/conservation-checks.json`, JSON.stringify(result, null, 2) + '\n');
const receiptPath = `${directory}/presentation-review.json`;
if (fs.existsSync(receiptPath)) {
  const receipt = JSON.parse(fs.readFileSync(receiptPath, 'utf8'));
  const ledger = JSON.parse(fs.readFileSync('docs/teaching/lesson-delivery-progress.json', 'utf8'));
  assert.equal(receipt.passed, true);
  assert.ok(filesMatch(process.cwd(), receipt.reviewedFiles), 'Presentation source changed since review');
  assert.ok(filesMatch(process.cwd(), receipt.evidenceHashes), 'Presentation evidence changed since review');
  assert.equal(receipt.preservedCurrentTopics.length, 81);
  for (const id of receipt.preservedCurrentTopics) {
    const row = ledger.topics[id];
    assert.ok(filesMatch(process.cwd(), row.content.files), `Stale content: ${id}`);
    assert.ok(filesMatch(process.cwd(), row.implementation.reviewedFiles), `Stale implementation: ${id}`);
  }
  // Undo only the documented file bindings in memory. Original phase dates,
  // revisions, scientific receipts and historical records must remain identical.
  for (const [id, expected] of Object.entries(receipt.rowHashesBefore)) {
    const row = structuredClone(ledger.topics[id]);
    const delta = receipt.checkpointDeltas[id];
    if (delta) {
      assert.deepEqual(row.implementation.presentationReviews, [receiptPath]);
      delete row.implementation.presentationReviews;
      for (const [phase, key, changes] of [['content', 'files', delta.contentFiles], ['implementation', 'reviewedFiles', delta.implementationFiles]]) {
        for (const [file, before] of Object.entries(changes)) {
          if (before === null) delete row[phase][key][file];
          else row[phase][key][file] = before;
        }
      }
    }
    assert.equal(hash(JSON.stringify(row)), expected, `Unreviewed ledger change: ${id}`);
  }
  result.currentCheckpointsPreserved = receipt.preservedCurrentTopics.length;
  result.historicalRowsUnchanged = receipt.preservedHistoricalRows;
}
console.log(JSON.stringify({ ...result, conservation: `${directory}/conservation-checks.json` }));
