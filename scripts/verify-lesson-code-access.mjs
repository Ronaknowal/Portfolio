import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { filesMatch } from './lib/lesson-delivery.mjs';

const directory = 'docs/teaching/lesson-code-access';
const receiptPath = process.argv[2] || `${directory}/presentation-review.json`;
const hash = value => createHash('sha256').update(value).digest('hex');
const receipt = JSON.parse(fs.readFileSync(receiptPath));
const ledger = JSON.parse(fs.readFileSync('docs/teaching/lesson-delivery-progress.json'));
assert.equal(receipt.passed, true);
assert.ok(filesMatch(process.cwd(), receipt.reviewedFiles), 'Reviewed code-access source changed');
assert.ok(filesMatch(process.cwd(), receipt.evidenceHashes), 'Reviewed evidence changed');
assert.equal(hash(fs.readFileSync(receipt.previousReview)), receipt.previousReviewSha256, 'Prior presentation receipt changed');
assert.equal(hash(fs.readFileSync('src/learn/data/lesson-manifest.json')), receipt.manifestSha256, 'Published mappings changed');
assert.ok(receipt.preservedCurrentTopics.length > 0);
for (const id of receipt.preservedCurrentTopics) {
  assert.ok(filesMatch(process.cwd(), ledger.topics[id].content.files), `Stale content: ${id}`);
  assert.ok(filesMatch(process.cwd(), ledger.topics[id].implementation.reviewedFiles), `Stale implementation: ${id}`);
}
for (const [id, expected] of Object.entries(receipt.rowHashesBefore)) {
  const row = structuredClone(ledger.topics[id]), delta = receipt.checkpointDeltas[id];
  if (delta) {
    assert.equal(row.implementation.presentationReviews.pop(), receiptPath);
    for (const [phase, key, changes] of [['content', 'files', delta.contentFiles], ['implementation', 'reviewedFiles', delta.implementationFiles]]) {
      for (const [file, before] of Object.entries(changes)) {
        assert.ok(file in receipt.reviewedFiles, `Unreviewed checkpoint refresh: ${file}`);
        if (before === null) delete row[phase][key][file];
        else row[phase][key][file] = before;
      }
    }
  }
  assert.equal(hash(JSON.stringify(row)), expected, `Unreviewed ledger change: ${id}`);
}
assert.equal(Object.keys(receipt.rowHashesBefore).length, Object.keys(ledger.topics).length, 'Ledger topics changed');
console.log(JSON.stringify({ passed: true, currentTopicsPreserved: receipt.preservedCurrentTopics.length, historicalRowsUnchanged: Object.keys(ledger.topics).length - receipt.preservedCurrentTopics.length, reviewedFiles: Object.keys(receipt.reviewedFiles).length, scope: receipt.scope }));
