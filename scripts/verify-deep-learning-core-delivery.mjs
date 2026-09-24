import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const digest = value => createHash('sha256').update(value).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const baseline = read('docs/teaching/evidence/deep-learning-core-baseline.json');
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const receipts = ['perceptron-browser', 'backprop-browser', 'loss-normalization-browser-production', 'transfer-learning-production-browser', 'deep-learning-core-production-integration'];
const report = { status: 'running', checkedAt: new Date().toISOString(), topics: [], receipts: [], unchangedOtherRows: 0 };
const output = 'docs/teaching/evidence/deep-learning-core-reconciliation.json';
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
save();
try {
  assert.equal(hash('src/learn/data/lesson-manifest.json'), baseline.manifestHash);
  assert.equal(Object.keys(ledger.topics).length, Object.keys(baseline.allRowHashes).length);
  for (const [id, before] of Object.entries(baseline.allRowHashes)) {
    const row = ledger.topics[id];
    if (!baseline.ids.includes(id)) {
      assert.equal(digest(JSON.stringify(row)), before, `Unrelated row changed: ${id}`);
      report.unchangedOtherRows++;
      continue;
    }
    assert.equal(row.revision, baseline.rows[id].revision);
    assert.equal(row.deliveryMode, 'content-first');
    assert.deepEqual(row.previousRevisions, baseline.rows[id].previousRevisions);
    assert.equal(row.content.status, 'complete'); assert.equal(row.implementation.status, 'complete');
    for (const [file, expected] of Object.entries({ ...row.content.files, ...row.implementation.reviewedFiles })) assert.equal(hash(file), expected, `${id}: stale ${file}`);
    report.topics.push({ id, revision: row.revision, content: 'complete', implementation: 'complete', reviewedFiles: Object.keys(row.implementation.reviewedFiles).length });
  }
  for (const name of receipts) {
    const file = `docs/teaching/evidence/${name}.json`, receipt = read(file);
    assert.ok(receipt.status === 'passed' || receipt.passed === true || name === 'backprop-browser' && receipt.assertions === 94 && receipt.errors.length === 0, `Receipt failed: ${name}`);
    for (const [file, expected] of Object.entries(receipt.sourceHashes || {})) assert.equal(hash(file), expected, `Stale browser source: ${file}`);
    if (receipt.manifestHash) assert.equal(hash('dist/.vite/manifest.json'), receipt.manifestHash);
    report.receipts.push({ file, sha256: hash(file) });
  }
  report.manifestHash = hash('dist/.vite/manifest.json');
  report.contentComplete = Object.values(ledger.topics).filter(row => row.content.status === 'complete').length;
  report.implementationComplete = Object.values(ledger.topics).filter(row => row.implementation.status === 'complete').length;
  report.preparedRemaining = report.contentComplete - report.implementationComplete;
  assert.equal(report.contentComplete, 177); assert.equal(report.implementationComplete, 140); assert.equal(report.preparedRemaining, 37);
  report.status = 'passed';
} catch (error) { report.status = 'failed'; report.failure = error.stack; process.exitCode = 1; }
save(); console.log(JSON.stringify(report));
