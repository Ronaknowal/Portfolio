import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

// Read-only reconciliation of this bounded review; never refresh stale checkpoints.
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidenceDirectory = 'docs/teaching/evidence/';
const baseline = read(`${evidenceDirectory}lesson-usability-ledger-baseline.json`);
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const production = read(`${evidenceDirectory}lesson-usability-production-source.json`);
const manifestHash = hash('dist/.vite/manifest.json');
assert.equal(production.status, 'build-passed');
assert.equal(production.manifestHash, manifestHash);
for (const [file, expected] of Object.entries(production.sourceHashes)) {
  assert.equal(hash(file), expected, `Production source changed: ${file}`);
}

const changedTopics = Object.keys(baseline.topics).filter(id =>
  Object.entries(baseline.topics[id].implementation.reviewedFiles || {})
    .some(([file, expected]) => hash(file) !== expected));
assert.equal(changedTopics.length, 24, 'Review scope changed; investigate instead of rebasing');
assert.deepEqual(Object.keys(ledger.topics), Object.keys(baseline.topics));
let preservedTopics = 0;
for (const [id, row] of Object.entries(ledger.topics)) {
  const previous = baseline.topics[id];
  if (!changedTopics.includes(id)) {
    assert.deepEqual(row, previous, `Unrelated ledger row changed: ${id}`);
    preservedTopics++;
  } else {
    assert.equal(row.revision, previous.revision + 1, id);
    assert.equal(row.record, 'docs/teaching/LESSON-USABILITY-REVIEW.md');
    const { previousRevisions = [], ...checkpoint } = previous;
    assert.deepEqual(row.previousRevisions, [...previousRevisions, checkpoint]);
    assert.equal(row.content.status, 'complete');
    assert.equal(row.implementation.status, 'complete');
  }
  for (const [file, expected] of Object.entries(row.content.files || {})) {
    assert.equal(hash(file), expected, `Content identity: ${id}: ${file}`);
  }
  for (const [file, expected] of Object.entries(row.implementation.reviewedFiles || {})) {
    assert.equal(hash(file), expected, `Implementation identity: ${id}: ${file}`);
  }
}
assert.equal(Object.values(ledger.topics).filter(row => row.content.status === 'complete').length, 177);
assert.equal(Object.values(ledger.topics).filter(row => row.implementation.status === 'complete').length, 135);
assert.equal(Object.values(ledger.topics).filter(row => row.implementation.status === 'not-started').length, 42);
assert.ok(!ledger.topics['byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram'], 'UI triage must not promote the older BPE article');

const receipts = ['programming-dsa-control-review', 'math-control-affordances-browser',
  'classical-control-followup-browser', 'published-lesson-ui-repairs'];
for (const name of receipts) {
  const result = read(`${evidenceDirectory}${name}.json`);
  assert.ok(result.passed === true || result.status === 'passed', `${name} did not pass`);
  assert.equal(result.errors.length, 0, `${name} has errors`);
  if (result.manifestHash) assert.equal(result.manifestHash, manifestHash, name);
  for (const [file, expected] of Object.entries(result.sourceHashes)) assert.equal(hash(file), expected, `${name}: ${file}`);
}
const older = read(`${evidenceDirectory}published-lesson-control-triage.json`);
assert.equal(older.records.length, 288);
assert.equal(older.errors.length, 0);
assert.equal(older.records.filter(row => row.candidates.length || row.overflow.length || row.articleError).length, 0);
for (const [file, expected] of Object.entries(older.sourceHashes)) assert.equal(hash(file), expected, file);
const inventory = read('docs/curriculum/curriculum-inventory.json');
assert.equal(inventory.counts.contentComplete, 177);
assert.equal(inventory.counts.implementationComplete, 135);
assert.equal(inventory.counts.published, 231);
assert.equal(inventory.counts.uniqueTopics, 1460);
const result = { passed: true, checkedAt: new Date().toISOString(), manifestHash,
  changedTopics, preservedTopics, contentComplete: 177, implementationComplete: 135,
  preparedUnimplemented: 42, published: 231, reviewedProductionSources: Object.keys(production.sourceHashes).length,
  reusedOlderSurvey: { manifestHash: older.manifestHash, unchangedSources: older.sourceHashes,
    explanation: 'The final rebuild changed only two DSA sources; these older-route and shared source checks remain applicable.' },
  receipts: Object.fromEntries(receipts.map(name => [`${evidenceDirectory}${name}.json`, hash(`${evidenceDirectory}${name}.json`)])) };
fs.writeFileSync(`${evidenceDirectory}lesson-usability-reconciliation.json`, JSON.stringify(result, null, 2) + '\n');
console.log(`PASS: ${changedTopics.length} reviewed revisions, ${preservedTopics} unchanged rows, 177 content / 135 implementations, 42 prepared packets preserved.`);
