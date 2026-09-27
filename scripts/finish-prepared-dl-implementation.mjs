// Close only explicitly selected prepared revisions after all six stages.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { filesMatch, validateDeliveryLedger } from './lib/lesson-delivery.mjs';

const directory = 'docs/teaching/deep-learning-completion';
const ledgerPath = 'docs/teaching/lesson-delivery-progress.json';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const rowHash = row => createHash('sha256').update(JSON.stringify(row)).digest('hex');
const baseline = read(`${directory}/baseline.json`), ledger = read(ledgerPath);
const ids = process.argv.slice(2);
assert.ok(ids.length, 'Pass explicit stable IDs.');
assert.equal(new Set(ids).size, ids.length);
const manifest = read('src/learn/data/lesson-manifest.json');
for (const [id, mapping] of Object.entries(baseline.manifest)) assert.deepEqual(manifest[id], mapping, `Existing publication mapping changed: ${id}`);
const additions = Object.keys(manifest).filter(id => !Object.hasOwn(baseline.manifest, id));
assert.ok(additions.every(id => ['mini-batches-training-loops-gradient-accumulation', 'neural-training-diagnostics-reproducible-experiments'].includes(id)), 'Unexpected new publication');

for (const id of ids) {
  assert.ok(baseline.topicIds.includes(id), `Outside request: ${id}`);
  const row = ledger.topics[id], topic = `${directory}/${id}`;
  assert.notEqual(row.implementation.status, 'complete', `Already closed; do not rewrite its evidence: ${id}`);
  assert.equal(row.revision, baseline.originalEntries[id].revision, 'Continue the prepared revision.');
  assert.equal(row.content.status, 'complete');
  assert.ok(filesMatch(process.cwd(), row.content.files), `Stale content: ${id}`);
  assert.ok(Object.hasOwn(manifest, id), `No published body: ${id}`);
  const authorPath = `${topic}/implementation-checks.json`, author = read(authorPath);
  assert.equal(author.passed, true, `Author checks open: ${id}`);
  assert.ok(filesMatch(process.cwd(), author.sourceHashes), `Stale author evidence: ${id}`);
  if (author.evidenceHashes) assert.ok(filesMatch(process.cwd(), author.evidenceHashes), `Stale author evidence attachment: ${id}`);
  const independentPath = `${topic}/independent-checks.json`, independent = read(independentPath);
  const browserPath = `${topic}/browser-checks.json`, browser = read(browserPath);
  for (const [stage, review] of [['independent', independent], ['browser', browser]]) {
    assert.equal(review.passed, true, `${stage} review open: ${id}`);
    assert.ok(review.checks?.length, `${stage} checks missing: ${id}`);
    assert.ok((review.findings ?? []).every(finding => finding.status === 'closed'), `${stage} findings open: ${id}`);
    assert.ok(filesMatch(process.cwd(), review.sourceHashes), `Stale ${stage} review: ${id}`);
  }
  const independentNarrative = `${topic}/independent-review.md`;
  assert.ok(fs.readFileSync(independentNarrative, 'utf8').length > 500);
  assert.ok(browser.buildReceipt?.startsWith(`${directory}/builds/`), 'Browser must name its immutable build receipt.');
  const build = read(browser.buildReceipt);
  assert.equal(build.passed, true);
  assert.ok(build.coveredTopicIds.includes(id), `Build did not cover: ${id}`);
  for (const file of author.sourceFiles) {
    if (file.startsWith('src/') || file.startsWith('public/')) {
      assert.equal(build.sourceHashes[file], hash(file), `Runtime changed since rendered build: ${file}`);
      assert.equal(browser.sourceHashes[file], hash(file), `Browser identity omitted: ${file}`);
    }
    if (file.startsWith('src/') || file.endsWith('/lesson.md')) assert.equal(independent.sourceHashes[file], hash(file), `Independent identity omitted: ${file}`);
  }
  // Freeze per-topic integration identity; do not depend on the mutable build pointer.
  const integrationPath = `${topic}/integration-checks.json`;
  const integration = { passed: true, checkedAt: new Date().toISOString(), topicId: id,
    buildReceipt: browser.buildReceipt, buildReceiptHash: hash(browser.buildReceipt),
    publicationMapping: manifest[id], preservedPublicationCount: Object.keys(baseline.manifest).length,
    scope: 'Current topic runtime built and rendered; original mappings preserved. Whole-increment conservation is also checked at final handoff.' };
  fs.writeFileSync(integrationPath, JSON.stringify(integration, null, 2) + '\n');
  const reviewedFiles = [...new Set([...Object.keys(row.content.files), ...author.sourceFiles,
    ...Object.keys(independent.sourceHashes), ...Object.keys(browser.sourceHashes),
    authorPath, independentPath, independentNarrative, browserPath, browser.buildReceipt, integrationPath])];
  row.implementation = { status: 'complete', completedAt: new Date().toISOString(),
    reviewedFiles: Object.fromEntries(reviewedFiles.map(file => [file, hash(file)])),
    evidence: { author: authorPath, independent: independentPath, browser: browserPath, integration: integrationPath } };
  row.record = independentNarrative;
  row.nextAction = 'Prepared revision implemented with independent, browser and build review. User acceptance remains separate. Preserve current sources and proceed only within the authorized module scope.';
}
for (const [id, row] of Object.entries(ledger.topics)) {
  if (!baseline.topicIds.includes(id)) assert.equal(rowHash(row), baseline.rowHashes[id], `Unrelated ledger row changed: ${id}`);
}
validateDeliveryLedger(ledger);
ledger.updatedOn = new Date().toISOString().slice(0, 10);
fs.writeFileSync(`${ledgerPath}.tmp`, JSON.stringify(ledger, null, 2) + '\n');
fs.renameSync(`${ledgerPath}.tmp`, ledgerPath);
console.log(JSON.stringify({ implementationCompleted: ids, preservedUnrelatedRows: Object.keys(ledger.topics).length - baseline.topicIds.length }));
