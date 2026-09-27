// Open source-bound content checkpoints for explicitly named author-ready topics.
// This never closes implementation: independent and rendered review are separate.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateDeliveryLedger } from './lib/lesson-delivery.mjs';

const ledgerPath = 'docs/teaching/lesson-delivery-progress.json';
const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/concept-intuition-baseline.json'));
const ledger = JSON.parse(fs.readFileSync(ledgerPath));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const ids = process.argv.slice(2);
assert.ok(ids.length, 'Pass explicit stable topic IDs.');
assert.equal(new Set(ids).size, ids.length, 'Duplicate IDs');
for (const id of ids) {
  const original = baseline.originalEntries[id];
  assert.ok(original, `Outside authorized review scope: ${id}`);
  const record = `docs/teaching/concept-intuition/${id}/review.md`;
  const checkPath = `docs/teaching/concept-intuition/${id}/author-checks.json`;
  const checks = JSON.parse(fs.readFileSync(checkPath));
  assert.equal(checks.topicId, id);
  const authorPassed = checks.authorChecksPassed === true && checks.actualChecks?.length > 0;
  const listedChecksPassed = checks.passed !== false && checks.checks?.length > 0 && checks.checks.every(check => check.passed === true || ['passed', 'author-reviewed'].includes(check.status));
  assert.ok(authorPassed || listedChecksPassed, `Author checks absent or failed: ${id}`);
  assert.ok(checks.sourceFiles?.length && checks.sourceHashes, `Missing source checkpoint: ${id}`);
  for (const file of checks.sourceFiles) assert.equal(hash(file), checks.sourceHashes[file], `Unreviewed changes: ${file}`);
  const previous = ledger.topics[id];
  assert.ok(previous.revision === original.revision || (previous.revision === original.revision + 1 && previous.record === record), `Unexpected concurrent ledger revision: ${id}`);
  const manuscripts = checks.sourceFiles.filter(file => file.startsWith('docs/') && file.endsWith('/lesson.md'));
  assert.ok(checks.canonicalManuscript || manuscripts.length <= 1, `Ambiguous canonical manuscript: ${id}`);
  const manuscript = checks.canonicalManuscript ?? manuscripts[0] ?? checks.sourceFiles.find(file => file.startsWith('src/learn/data/topics/') && file.endsWith('.jsx'));
  assert.ok(manuscript, `No production manuscript: ${id}`);
  assert.ok(checks.sourceFiles.includes(manuscript), `Canonical manuscript is not source-bound: ${id}`);
  const retainedInputs = Object.keys(original.content.files ?? {}).filter(file => file.startsWith('src/') || file.startsWith('public/'));
  for (const file of retainedInputs.filter(file => !checks.sourceFiles.includes(file))) {
    assert.equal(hash(file), baseline.actualFileHashes[file], `Changed dependency needs author review: ${file}`);
  }
  const files = [...new Set([manuscript, record, checkPath, ...retainedInputs, ...checks.sourceFiles])];
  const { previousRevisions = [], ...historical } = original;
  ledger.topics[id] = {
    revision: original.revision + 1, deliveryMode: 'full', record,
    content: { status: 'complete', completedAt: new Date().toISOString(), manuscript, visualSpecifications: record, files: Object.fromEntries(files.map(file => [file, hash(file)])) },
    implementation: { status: 'in-progress' },
    nextAction: 'Complete independent concept/correctness review, scoped rendered verification and integration; preserve existing native evidence for unchanged source.',
    previousRevisions: [...previousRevisions, historical],
  };
}
for (const [id, row] of Object.entries(ledger.topics)) {
  if (!baseline.topicIds.includes(id)) assert.equal(createHash('sha256').update(JSON.stringify(row)).digest('hex'), baseline.rowHashes[id], `Unrelated row changed: ${id}`);
}
validateDeliveryLedger(ledger);
ledger.updatedOn = '2026-09-26';
const temporary = ledgerPath + '.tmp';
fs.writeFileSync(temporary, JSON.stringify(ledger, null, 2) + '\n');
fs.renameSync(temporary, ledgerPath);
console.log(JSON.stringify({ contentCheckpointed: ids, implementation: 'in-progress; not certified complete' }));
