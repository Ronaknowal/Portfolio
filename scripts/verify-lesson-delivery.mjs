import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { topicCatalogue } from '../src/learn/data/curriculum/topic-catalogue.js';
import { deliveryLedgerPath, validateDeliveryLedger, getDeliveryState, assertDeliveryRequest, getImplementationReviewOverride } from './lib/lesson-delivery.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ledger = JSON.parse(fs.readFileSync(path.join(root, deliveryLedgerPath), 'utf8'));
validateDeliveryLedger(ledger, new Set(Object.keys(topicCatalogue)));
for (const entry of Object.values(ledger.topics)) {
  assert.ok(fs.existsSync(path.join(root, entry.record)), `Missing handoff record: ${entry.record}`);
}

const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'lesson-delivery-'));
const filenames = ['lesson.md', 'visual-specifications.md', 'lesson.jsx'];
const hash = filename => createHash('sha256').update(fs.readFileSync(path.join(temporaryRoot, filename))).digest('hex');
const cases = [];
function check(name, callback) {
  callback();
  cases.push(name);
}
try {
  for (const filename of filenames) fs.writeFileSync(path.join(temporaryRoot, filename), `Fixture for ${filename}\n`);
  const entry = {
    revision: 1, deliveryMode: 'content-first', record: 'design.md',
    content: { status: 'in-progress' }, implementation: { status: 'not-started' },
  };
  const validate = candidate => validateDeliveryLedger({ schemaVersion: 1, topics: { sample: candidate } }, new Set(['sample']));
  check('Missing content prevents finish; full/content requests can begin', () => {
    const state = getDeliveryState(temporaryRoot);
    assert.throws(() => assertDeliveryRequest('finish', state), /complete, current content/);
    assert.doesNotThrow(() => assertDeliveryRequest('content', state));
    assert.doesNotThrow(() => assertDeliveryRequest('full', state));
    assert.throws(() => assertDeliveryRequest('typo', state), /--work/);
  });
  check('Incomplete content and premature implementation are rejected', () => {
    validate(entry);
    assert.throws(() => assertDeliveryRequest('finish', getDeliveryState(temporaryRoot, entry)), /complete, current content/);
    for (const status of ['in-progress', 'complete']) {
      assert.throws(() => validate({ ...entry, implementation: { status } }), /cannot start before content/);
    }
  });
  entry.content = { status: 'complete', manuscript: 'lesson.md', visualSpecifications: 'visual-specifications.md',
    files: Object.fromEntries(filenames.slice(0, 2).map(file => [file, hash(file)])) };
  check('A complete matching manuscript/specification checkpoint permits finish only', () => {
    assert.throws(() => validate({ ...entry, content: { status: 'complete', files: entry.content.files } }), /identify and hash its manuscript/);
    assert.throws(() => validate({ ...entry, content: { ...entry.content, visualSpecifications: 'unwritten-specs.md' } }), /identify and hash its visualSpecifications/);
    validate(entry);
    const state = getDeliveryState(temporaryRoot, entry);
    assert.equal(state.content, 'complete');
    assert.equal(state.implementation, 'not-started');
    assert.doesNotThrow(() => assertDeliveryRequest('finish', state));
    assert.equal(getImplementationReviewOverride(entry, state), null);
  });
  check('Changed or missing specification invalidates a previously complete handoff', () => {
    fs.appendFileSync(path.join(temporaryRoot, 'visual-specifications.md'), 'Changed mechanism\n');
    let state = getDeliveryState(temporaryRoot, entry);
    assert.equal(state.content, 'stale');
    assert.equal(state.recordedContent, 'complete');
    assert.throws(() => assertDeliveryRequest('finish', state), /complete, current content/);
    entry.content.files['visual-specifications.md'] = hash('visual-specifications.md');
    const missing = structuredClone(entry);
    missing.content.files['missing.md'] = '0'.repeat(64);
    assert.equal(getDeliveryState(temporaryRoot, missing).canFinish, false);
  });
  check('Implementation completion needs its own exact source checkpoint', () => {
    assert.throws(() => validate({ ...entry, implementation: { status: 'complete' } }), /file\/hash map/);
    entry.implementation = { status: 'complete', reviewedFiles: { 'lesson.jsx': hash('lesson.jsx') } };
    validate(entry);
    assert.equal(getDeliveryState(temporaryRoot, entry).implementation, 'complete');
    fs.appendFileSync(path.join(temporaryRoot, 'lesson.jsx'), 'Unreviewed change\n');
    const state = getDeliveryState(temporaryRoot, entry);
    assert.equal(state.content, 'complete');
    assert.equal(state.implementation, 'stale');
    assert.equal(state.recordedImplementation, 'complete');
    assert.equal(getImplementationReviewOverride(entry, state).teachingReview, 'individual-review-required');
    entry.implementation.reviewedFiles['lesson.jsx'] = hash('lesson.jsx');
    assert.equal(getImplementationReviewOverride(entry, getDeliveryState(temporaryRoot, entry)).teachingReview, 'implementation-reviewed-user-acceptance-pending');
  });
  check('Historical completion is preserved without blessing a changed revision', () => {
    const historical = { revision: 1, deliveryMode: 'full', record: 'old-review.md', legacyReview: true,
      content: { status: 'complete' }, implementation: { status: 'complete' } };
    validate(historical);
    assert.equal(getDeliveryState(temporaryRoot, historical, true).implementation, 'complete');
    const stale = getDeliveryState(temporaryRoot, historical, false);
    assert.equal(stale.implementation, 'stale');
    assert.equal(stale.recordedImplementation, 'complete');
    assert.equal(stale.canFinish, false);
    assert.throws(() => validate({ ...historical, implementation: { status: 'in-progress' } }), /historical revision/);
  });
  check('Malformed status, topic, hash and path cannot bypass the checkpoint', () => {
    assert.throws(() => validate({ ...entry, content: { status: 'approved' } }), /content status/);
    assert.throws(() => validateDeliveryLedger({ schemaVersion: 1, topics: { unknown: entry } }, new Set(['sample'])), /Unknown delivery topic/);
    for (const files of [{ '../outside.md': '0'.repeat(64) }, { 'C:\\outside.md': '0'.repeat(64) }, { 'lesson.md': 'not-a-hash' }, {}]) {
      assert.throws(() => validate({ ...entry, content: { ...entry.content, files } }));
    }
  });
  check('An active full rewrite cannot inherit an old allowlisted/user-approved status', () => {
    const rewriting = { ...entry, deliveryMode: 'full', implementation: { status: 'not-started' } };
    assert.equal(getImplementationReviewOverride(rewriting, getDeliveryState(temporaryRoot, rewriting)).teachingReview, 'individual-review-required');
  });
} finally {
  // Delete only the three files created here, then their empty directory.
  for (const filename of filenames) fs.unlinkSync(path.join(temporaryRoot, filename));
  fs.rmdirSync(temporaryRoot);
}
console.log(JSON.stringify({ status: 'passed', trackedTopics: Object.keys(ledger.topics).length, cases }, null, 2));
