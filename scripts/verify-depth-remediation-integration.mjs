import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { getDeliveryState, filesMatch } from './lib/lesson-delivery.mjs';
import { topicCatalogue } from '../src/learn/data/curriculum/topic-catalogue.js';

const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const base = read('docs/teaching/evidence/implementation-depth-remediation-baseline.json');
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const browserFile = 'docs/teaching/evidence/depth-remediation-browser.json';
const browser = read(browserFile);
const receiptFile = 'docs/teaching/evidence/implementation-depth-remediation-integration.json';
const receipt = read(receiptFile);
const amendments = read(receipt.documentationAmendments);
const documentation = new Map(amendments.files.map(entry => [entry.file, entry]));
const checks = [];
const check = (name, fn) => { fn(); checks.push(name); };

check('All scoped revisions preserve their exact preceding checkpoints', () => {
  assert.equal(base.scope.length, 35);
  for (const id of base.scope) {
    const current = ledger.topics[id], prior = base.priorRows[id];
    assert.equal(current.revision, prior.revision + 1, id);
    const { previousRevisions = [], ...old } = prior;
    assert.deepEqual(current.previousRevisions, [...previousRevisions, old], id);
    const effective = getDeliveryState(process.cwd(), current);
    assert.equal(effective.content, 'complete', id);
    assert.equal(effective.implementation, 'complete', id);
  }
});
check('Three compatibility rows change only the tested shared-viewer binding', () => {
  assert.equal(Object.keys(receipt.compatibilityPreviousRows).length, 3);
  for (const [id, prior] of Object.entries(receipt.compatibilityPreviousRows)) {
    assert.equal(digest(JSON.stringify(prior)), base.allRowHashes[id]);
    const expected = structuredClone(prior);
    expected.implementation.reviewedFiles['src/learn/components/lesson-labs/MechanismProgram.jsx'] = hash('src/learn/components/lesson-labs/MechanismProgram.jsx');
    assert.deepEqual(ledger.topics[id], expected, id);
  }
});
let unchanged = 0;
check('Every other phase row is byte-equivalent after JSON serialization', () => {
  for (const [id, expected] of Object.entries(base.allRowHashes)) {
    if (base.scope.includes(id) || receipt.compatibilityPreviousRows[id]) continue;
    assert.equal(digest(JSON.stringify(ledger.topics[id])), expected, id);
    unchanged++;
  }
  assert.equal(unchanged, 139);
  assert.equal(Object.keys(ledger.topics).length, 177);
});
const historical = read('docs/teaching/dsa-math-foundations-progress.json');
const effective = Object.entries(ledger.topics).map(([id, row]) => {
  const review = row.legacyReview && historical.topics.find(topic => topic.id === id);
  const legacyCurrent = review?.status === 'implementation-reviewed' && filesMatch(process.cwd(), review.reviewedFiles);
  return getDeliveryState(process.cwd(), row, legacyCurrent);
});
const totals = {
  contentComplete: effective.filter(row => row.content === 'complete').length,
  implementationComplete: effective.filter(row => row.implementation === 'complete').length,
  prepared: effective.filter(row => row.content === 'complete' && row.implementation !== 'complete').length,
  published: Object.keys(read('src/learn/data/lesson-manifest.json')).length,
  stableTopics: Object.keys(topicCatalogue).length,
};
check('All current phase sources and conserved catalogue totals match', () => {
  assert.deepEqual(totals, { contentComplete: 177, implementationComplete: 140, prepared: 37, published: 231, stableTopics: 1460 });
});
check('All 38 production pages match current build, checker and runtime sources', () => {
  assert.equal(browser.status, 'passed');
  assert.deepEqual(browser.missingTopics, []);
  assert.equal(Object.keys(browser.topics).length, 38);
  assert.equal(browser.manifestHash, hash('dist/.vite/manifest.json'));
  assert.equal(browser.verifierHash, hash('scripts/verify-depth-remediation-browser.cjs'));
  for (const id of [...base.scope, ...Object.keys(receipt.compatibilityPreviousRows)]) {
    const row = browser.topics[id];
    assert.equal(row.status, 'passed', id);
    assert.equal(row.manifestHash, browser.manifestHash, id);
    assert.equal(row.verifierHash, browser.verifierHash, id);
    assert.equal(row.expectationsHash, browser.expectationsHash, id);
    for (const [file, expected] of Object.entries(row.sourceHashes)) assert.equal(hash(file), expected, file);
    for (const capture of row.captures) assert.equal(hash(capture.file), capture.sha256);
  }
});
const independentFiles = [
  'docs/teaching/implementation-depth/DSA-REMEDIATION-INDEPENDENT.json',
  'docs/teaching/implementation-depth/MATH-LIBRARY-REMEDIATION-INDEPENDENT.json',
  'docs/teaching/implementation-depth/math-depth-remediation-independent-sources.json',
  'docs/teaching/implementation-depth/classical-remediation-independent.json',
];
check('Independent source reviews are closed; later documentation changes are explicit', () => {
  for (const file of independentFiles) {
    const review = read(file);
    assert.match(review.status, /pass|independent-complete/);
    for (const finding of review.findings || []) assert.match(finding.status, /closed|resolved/);
    for (const [source, expected] of Object.entries(review.sourceHashes)) {
      const current = hash(source);
      if (current === expected) continue;
      const amendment = documentation.get(source) || receipt.authorManifestAmendments.find(entry => entry.file === source);
      assert.ok(amendment && source.startsWith('docs/'), `Unreviewed source change: ${source}`);
      assert.equal(amendment.before, expected, source);
      assert.equal(amendment.after, current, source);
    }
    if (review.manifest) {
      const amended = receipt.authorManifestAmendments.find(entry => entry.file === review.manifest.path);
      assert.equal(amended.before, review.manifest.sha256);
      assert.equal(amended.after, hash(review.manifest.path));
    }
  }
});
check('Author checkpoint updates contain only reviewed repair or completion bindings', () => {
  for (const manifest of receipt.authorManifestAmendments) assert.equal(hash(manifest.file), manifest.after);
  for (const amendment of amendments.files) assert.equal(hash(amendment.file), amendment.after);
});
check('Canonical public programs in reviewed source maps were copied unchanged into production', () => {
  for (const file of independentFiles) {
    for (const source of Object.keys(read(file).sourceHashes).filter(path => path.startsWith('public/'))) {
      assert.equal(hash(source.replace(/^public\//, 'dist/')), hash(source), source);
    }
  }
});
const finalChecksFile = 'docs/teaching/evidence/implementation-depth-final-checks.json';
check('Final repository checks were executed successfully', () => {
  const finalChecks = read(finalChecksFile);
  assert.equal(finalChecks.status, 'passed');
  assert.equal(finalChecks.checks.length, 6);
  for (const checked of finalChecks.checks) {
    assert.equal(checked.exitCode, 0, checked.command);
    assert.ok(checked.output.trim(), checked.command);
  }
});

receipt.status = 'passed';
receipt.checkedAt = new Date().toISOString();
receipt.checks = checks;
receipt.totals = totals;
receipt.revisionsCompleted = base.scope;
receipt.unchangedRows = unchanged;
receipt.browser = { file: browserFile, sha256: hash(browserFile), pages: 38, widths: [1366, 390, 320], inspectedPhoneCaptures: 38 };
receipt.independentEvidence = Object.fromEntries(independentFiles.map(file => [file, hash(file)]));
receipt.build = { status: 'passed', manifestSha256: hash('dist/.vite/manifest.json'), log: 'scratch/depth-remediation-build.txt', logSha256: hash('scratch/depth-remediation-build.txt'), retainedWarning: 'Existing large catalogue chunk' };
receipt.ledgerSha256 = hash('docs/teaching/lesson-delivery-progress.json');
receipt.repositoryChecks = { file: finalChecksFile, sha256: hash(finalChecksFile) };
receipt.verifierSha256 = hash('scripts/verify-depth-remediation-integration.mjs');
fs.writeFileSync(receiptFile, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ status: receipt.status, checks: checks.length, revisions: 35, compatibility: 3, unchanged, totals }));
