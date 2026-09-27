import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { tracks } from '../src/learn/data/generated/navigation.js';
import { getDeliveryState, validateDeliveryLedger } from './lib/lesson-delivery.mjs';

// A revision-specific audit. Never rewrites the ledger or historical receipts.
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const baseline = read('docs/teaching/evidence/attention-memory-intuition-baseline.json');
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const publication = read('src/learn/data/lesson-manifest.json');
const evidenceDirectory = 'docs/teaching/evidence/attention-memory-intuition';
const destination = `${evidenceDirectory}/integration.json`;
const report = { passed: false, date: new Date().toISOString(), revision: 4, topics: [], preservedOtherRows: 0, receipts: [] };
const save = () => fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
save();

function checkBindings(bindings, description) {
  assert.ok(Object.keys(bindings || {}).length, `Empty source binding: ${description}`);
  for (const [file, expected] of Object.entries(bindings)) {
    assert.equal(hash(file), expected, `Stale ${description}: ${file}`);
  }
}

try {
  validateDeliveryLedger(ledger);
  assert.equal(baseline.topicIds.length, 3);
  assert.equal(Object.keys(ledger.topics).length, Object.keys(baseline.rowHashes).length);
  for (const [id, before] of Object.entries(baseline.rowHashes)) {
    const row = ledger.topics[id];
    if (!baseline.topicIds.includes(id)) {
      assert.equal(digest(JSON.stringify(row)), before, `Unrelated ledger row changed: ${id}`);
      report.preservedOtherRows++;
      continue;
    }
    const original = baseline.originalEntries[id];
    assert.equal(row.revision, original.revision + 1);
    assert.equal(row.deliveryMode, 'full');
    const { previousRevisions = [], ...originalCurrent } = original;
    assert.deepEqual(row.previousRevisions, [...previousRevisions, originalCurrent], `History changed: ${id}`);
    checkBindings(original.content.files, `unchanged original packet ${id}`);
    for (const [file, expected] of Object.entries(original.implementation.reviewedFiles)) {
      // Runtime UI is revised; old programs, data, records and receipts are not.
      if (file.startsWith('public/') || file.startsWith('docs/')) {
        assert.equal(hash(file), expected, `Historical artifact changed: ${file}`);
      }
    }
    const state = getDeliveryState(process.cwd(), row);
    assert.equal(state.content, 'complete', `Content checkpoint not current: ${id}`);
    assert.equal(state.implementation, 'complete', `Implementation checkpoint not current: ${id}`);
    report.topics.push({ id, revision: row.revision, content: state.content, implementation: state.implementation, reviewedFiles: Object.keys(row.implementation.reviewedFiles).length });
  }
  assert.equal(report.preservedOtherRows, 174);
  assert.deepEqual(publication, baseline.manifest, 'Publication mappings changed');
  const module = tracks.find(track => track.id === 'deep-learning-fundamentals');
  assert.ok(module);
  const ordered = module.sections.flatMap(section => section.topicIds);
  report.sequence = ordered.slice(14, 19);
  assert.deepEqual(report.sequence, ['sequence-to-sequence-encoder-decoder', ...baseline.topicIds, 'rwkv-linear-attention-models']);

  const packet = id => `docs/teaching/revisions/${id}/4`;
  const receipts = [
    `${packet(baseline.topicIds[0])}/author-checks.json`,
    `${packet(baseline.topicIds[1])}/teaching-checks.json`,
    `${packet(baseline.topicIds[1])}/evidence/independent-review.json`,
    `${packet(baseline.topicIds[2])}/evidence/teaching-checks.json`,
    `${packet(baseline.topicIds[2])}/evidence/independent-review.json`,
    `${evidenceDirectory}/browser-review.json`,
  ];
  for (const file of receipts) {
    const receipt = read(file);
    assert.ok(receipt.passed === true || ['passed', 'no-open-findings-within-scope'].includes(receipt.status), `Incomplete receipt: ${file}`);
    checkBindings(receipt.sourceHashes || receipt.reviewedFiles, file);
    report.receipts.push({ file, sha256: hash(file) });
  }
  const browser = read(`${evidenceDirectory}/browser-review.json`);
  assert.deepEqual(Object.keys(browser.topics), baseline.topicIds);
  assert.deepEqual(browser.widths, [1280, 760, 320]);
  checkBindings(browser.screenshotHashes, 'retained browser screenshots');
  const independent = `${packet(baseline.topicIds[0])}/independent-teaching-review.md`;
  const bindings = [...fs.readFileSync(independent, 'utf8').matchAll(/^\| `([^`]+)` \| `([a-f0-9]{64})` \|$/gm)];
  assert.equal(bindings.length, 9, 'Incomplete independent attention bindings');
  checkBindings(Object.fromEntries(bindings.map(([, file, expected]) => [file, expected])), independent);
  report.independentAttentionBindings = bindings.length;

  const built = read('dist/.vite/manifest.json');
  function closure(key, visited = new Set()) {
    if (visited.has(key)) return visited;
    assert.ok(built[key], `Missing built chunk: ${key}`);
    visited.add(key);
    for (const dependency of built[key].imports || []) closure(dependency, visited);
    return visited;
  }
  const readers = Object.keys(built).filter(key => /\/Reader\.jsx$/.test(key));
  assert.ok(readers.length);
  const entryClosures = ['index.html', ...readers].map(key => closure(key));
  report.chunks = baseline.topicIds.map(id => {
    const key = `src/learn/data/${publication[id].replace('./', '')}`;
    assert.ok(built[key], `Missing topic chunk: ${id}`);
    for (const entries of entryClosures) assert.ok(!entries.has(key), `Eager lesson import: ${id}`);
    const file = `dist/${built[key].file}`;
    const bytes = fs.readFileSync(file);
    return { id, file, bytes: bytes.length, gzipBytes: gzipSync(bytes).length, sha256: hash(file) };
  });
  for (const id of baseline.topicIds) {
    const directory = `public/learn-code/${id}`;
    const names = fs.readdirSync(directory);
    assert.ok(names.length > 0);
    for (const name of names) {
      assert.ok(!name.endsWith('.pyc') && name !== '__pycache__', 'Cache in public downloads');
      const source = `${directory}/${name}`;
      if (fs.statSync(source).isFile()) assert.equal(hash(`dist/learn-code/${id}/${name}`), hash(source), `Stale built download: ${source}`);
    }
  }
  report.buildManifestHash = hash('dist/.vite/manifest.json');
  report.recordedCounts = {
    contentComplete: Object.values(ledger.topics).filter(row => row.content.status === 'complete').length,
    implementationComplete: Object.values(ledger.topics).filter(row => row.implementation.status === 'complete').length,
    preparedRemaining: Object.values(ledger.topics).filter(row => row.content.status === 'complete' && row.implementation.status !== 'complete').length,
  };
  assert.deepEqual(report.recordedCounts, { contentComplete: 177, implementationComplete: 153, preparedRemaining: 24 });
  report.passed = true;
} catch (error) {
  report.failure = error.stack;
  process.exitCode = 1;
}
save();
console.log(JSON.stringify({ passed: report.passed, topics: report.topics, preservedOtherRows: report.preservedOtherRows, receipts: report.receipts.length, recordedCounts: report.recordedCounts, failure: report.failure }));
