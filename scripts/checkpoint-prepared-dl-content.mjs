// Reconcile an explicitly reviewed prepared packet; never certify implementation here.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateDeliveryLedger } from './lib/lesson-delivery.mjs';

const root = 'docs/teaching/deep-learning-completion';
const ledgerPath = 'docs/teaching/lesson-delivery-progress.json';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const rowHash = row => createHash('sha256').update(JSON.stringify(row)).digest('hex');
const baseline = read(`${root}/baseline.json`);
const ledger = read(ledgerPath);
const ids = process.argv.slice(2);
assert.ok(ids.length, 'Pass explicit stable topic IDs.');
assert.equal(new Set(ids).size, ids.length, 'Duplicate IDs.');

for (const id of ids) {
  assert.ok(baseline.topicIds.includes(id), `Outside current request: ${id}`);
  const original = baseline.originalEntries[id];
  const receiptPath = `${root}/${id}/content-checks.json`;
  const receipt = read(receiptPath);
  assert.equal(receipt.topicId, id);
  assert.equal(receipt.passed, true, `Content assessment remains open: ${id}`);
  assert.ok(receipt.checks?.length && receipt.checks.every(check => check.passed === true));
  assert.ok(receipt.reconciliation?.length, `Reconciliation assessment missing: ${id}`);
  assert.ok(receipt.sourceFiles?.length);
  const files = new Set(receipt.sourceFiles);
  for (const required of Object.keys(original.content.files)) {
    assert.ok(files.has(required), `Prepared dependency omitted without a retained reconciliation: ${required}`);
  }
  for (const file of files) assert.equal(hash(file), receipt.sourceHashes[file], `Stale content assessment: ${file}`);
  assert.ok(files.has(receipt.manuscript), 'Manuscript must be assessed.');
  assert.ok(files.has(receipt.visualSpecifications), 'Visual specifications must be assessed.');
  const row = ledger.topics[id];
  assert.equal(row.revision, original.revision, 'Continue the prepared revision instead of discarding its history.');
  assert.notEqual(row.implementation.status, 'complete', 'Do not reopen a finished implementation.');
  row.content = {
    status: 'complete', completedAt: new Date().toISOString(),
    manuscript: receipt.manuscript, visualSpecifications: receipt.visualSpecifications,
    files: Object.fromEntries([...files, receiptPath].map(file => [file, hash(file)])),
  };
  row.implementation = { status: 'in-progress' };
  row.nextAction = 'Implement the reconciled packet, complete author and independent checks, then actual browser and shared integration review. Current implementation is not yet certified.';
}

for (const [id, row] of Object.entries(ledger.topics)) {
  if (!baseline.topicIds.includes(id)) assert.equal(rowHash(row), baseline.rowHashes[id], `Unrelated ledger row changed: ${id}`);
}
validateDeliveryLedger(ledger);
ledger.updatedOn = new Date().toISOString().slice(0, 10);
fs.writeFileSync(`${ledgerPath}.tmp`, JSON.stringify(ledger, null, 2) + '\n');
fs.renameSync(`${ledgerPath}.tmp`, ledgerPath);
console.log(JSON.stringify({ contentCheckpointed: ids, implementation: 'in-progress' }));
