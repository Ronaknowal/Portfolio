// Close only explicitly named, fully reviewed lesson revisions. No recertification
// by publication status or by a stale historical allowlist is permitted.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateDeliveryLedger, filesMatch } from './lib/lesson-delivery.mjs';

const directory = 'docs/teaching/concept-intuition';
const ledgerPath = 'docs/teaching/lesson-delivery-progress.json';
const read = file => JSON.parse(fs.readFileSync(file));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const baseline = read('docs/teaching/evidence/concept-intuition-baseline.json');
const ledger = read(ledgerPath);
const integrationPath = `${directory}/integration-checks.json`;
const integration = read(integrationPath);
assert.equal(integration.passed, true, 'Integration checks remain open');
const ids = process.argv.slice(2);
assert.ok(ids.length, 'Pass explicit stable IDs');
assert.equal(new Set(ids).size, ids.length);
for (const id of ids) {
  assert.ok(baseline.topicIds.includes(id), `Outside scope: ${id}`);
  const row = ledger.topics[id];
  assert.equal(row.revision, baseline.originalEntries[id].revision + 1);
  assert.ok(filesMatch(process.cwd(), row.content.files), `Stale content: ${id}`);
  const author = read(`${directory}/${id}/author-checks.json`);
  const independentPath = `${directory}/${id}/independent-checks.json`;
  const independent = read(independentPath);
  assert.equal(independent.passed, true, `Independent review pending: ${id}`);
  assert.ok(independent.checks?.length, `No independent checks: ${id}`);
  assert.ok((independent.findings ?? []).every(finding => finding.status === 'closed'), `Open independent findings: ${id}`);
  assert.ok(filesMatch(process.cwd(), independent.sourceHashes), `Stale independent review: ${id}`);
  for (const file of author.sourceFiles.filter(file => file.startsWith('src/') || file.endsWith('/lesson.md'))) {
    assert.equal(independent.sourceHashes[file], hash(file), `Independent review does not cover ${file}`);
  }
  const browserPath = `${directory}/${id}/browser-checks.json`;
  const browser = read(browserPath);
  assert.equal(browser.passed, true, `Rendered review pending: ${id}`);
  assert.ok((browser.findings ?? []).every(finding => finding.status === 'closed'), `Open browser findings: ${id}`);
  assert.ok(filesMatch(process.cwd(), browser.sourceHashes), `Stale rendered review: ${id}`);
  for (const file of author.sourceFiles.filter(file => file.startsWith('src/'))) {
    assert.equal(browser.sourceHashes[file], hash(file), `Rendered source identity missing: ${file}`);
    assert.equal(integration.builtSourceHashes[file], hash(file), `Latest production build does not cover ${file}`);
  }
  const independentRecord = `${directory}/${id}/independent-review.md`;
  assert.ok(fs.readFileSync(independentRecord, 'utf8').length > 400, `Independent narrative missing: ${id}`);
  const reviewedFiles = [...new Set([...Object.keys(row.content.files), ...independent.sourceFiles, independentPath, independentRecord, browserPath, integrationPath])];
  row.implementation = {
    status: 'complete', completedAt: new Date().toISOString(),
    reviewedFiles: Object.fromEntries(reviewedFiles.map(file => [file, hash(file)])),
    evidence: { author: `${directory}/${id}/author-checks.json`, independent: independentPath, browser: browserPath, integration: integrationPath },
  };
  row.nextAction = 'Current concept-by-concept revision is implemented and reviewed. User acceptance remains separate; preserve unchanged numerical evidence and follow the next scoped request.';
}
for (const [id, row] of Object.entries(ledger.topics)) {
  if (!baseline.topicIds.includes(id)) assert.equal(createHash('sha256').update(JSON.stringify(row)).digest('hex'), baseline.rowHashes[id], `Unrelated row changed: ${id}`);
}
assert.deepEqual(read('src/learn/data/lesson-manifest.json'), baseline.manifest, 'Publication mappings changed');
validateDeliveryLedger(ledger);
ledger.updatedOn = '2026-09-26';
fs.writeFileSync(ledgerPath + '.tmp', JSON.stringify(ledger, null, 2) + '\n');
fs.renameSync(ledgerPath + '.tmp', ledgerPath);
console.log(JSON.stringify({ implementationCompleted: ids, unrelatedRowsPreserved: Object.keys(ledger.topics).length - baseline.topicIds.length }));
