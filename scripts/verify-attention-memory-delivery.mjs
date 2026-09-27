import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { tracks } from '../src/learn/data/generated/navigation.js';
import { getDeliveryState, validateDeliveryLedger } from './lib/lesson-delivery.mjs';

const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const baseline = read('docs/teaching/evidence/attention-sequence-baseline.json');
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const publication = read('src/learn/data/lesson-manifest.json');
const destination = 'docs/teaching/evidence/attention-memory-integration.json';
const report = { passed: false, date: new Date().toISOString(), topics: [], preservedOtherRows: 0, metadataRepairs: [], receipts: [] };
const save = () => fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
save();

// These two pre-existing catalogue validation errors received a bounded metadata
// repair. Reconstruct their original row to prove no other fields were changed.
const metadataRepairs = {
  'depthwise-separable-dilated-convolutions': '295a887422347d975467f94ef48f4326c8173b6d6225cd63ac4ceb663a6bd969',
  'convnext-modern-cnn-designs': 'ad10d31a918cf57a4db2f8205b811257aea1d2af21295d7dff80f3f05be17b3d',
};

try {
  validateDeliveryLedger(ledger);
  assert.equal(Object.keys(ledger.topics).length, Object.keys(baseline.rowHashes).length);
  for (const [id, before] of Object.entries(baseline.rowHashes)) {
    const row = ledger.topics[id];
    if (metadataRepairs[id]) {
      const file = `src/learn/data/curriculum/blueprints/${id}.js`;
      assert.equal(row.implementation.reviewedFiles[file], hash(file));
      const restored = structuredClone(row);
      restored.implementation.reviewedFiles[file] = metadataRepairs[id];
      assert.equal(digest(JSON.stringify(restored)), before, `Unexpected metadata-repair scope: ${id}`);
      report.metadataRepairs.push({ id, file, oldHash: metadataRepairs[id], currentHash: hash(file) });
      continue;
    }
    if (!baseline.topicIds.includes(id)) {
      assert.equal(digest(JSON.stringify(row)), before, `Unrelated ledger changed: ${id}`);
      report.preservedOtherRows++;
      continue;
    }
    const original = baseline.originalEntries[id];
    assert.equal(row.revision, original.revision);
    assert.equal(row.deliveryMode, original.deliveryMode);
    assert.deepEqual(row.content, original.content);
    assert.deepEqual(row.previousRevisions, original.previousRevisions);
    const state = getDeliveryState(process.cwd(), row);
    assert.equal(state.content, 'complete', `${id}: content not current`);
    assert.equal(state.implementation, 'complete', `${id}: implementation not current`);
    report.topics.push({ id, revision: row.revision, content: state.content, implementation: state.implementation, reviewedFiles: Object.keys(row.implementation.reviewedFiles).length });
  }

  for (const [id, filename] of Object.entries(baseline.manifest)) assert.equal(publication[id], filename, `Existing publication mapping changed: ${id}`);
  assert.deepEqual(Object.keys(publication).filter(id => !Object.hasOwn(baseline.manifest, id)), [baseline.topicIds[1]]);
  const module = tracks.find(track => track.id === 'deep-learning-fundamentals');
  assert.ok(module, 'Missing module');
  const ordered = module.sections.flatMap(section => section.topicIds);
  assert.deepEqual(ordered.slice(14, 19), ['sequence-to-sequence-encoder-decoder', ...baseline.topicIds, 'rwkv-linear-attention-models']);
  report.sequence = ordered.slice(14, 19);

  const receipts = [
    'recurrent-attention-models.json', 'recurrent-attention-native.json', 'recurrent-attention-browser.json',
    'long-context/model-checks.json', 'long-context/native-checks.json', 'long-context/independent-checks.json', 'long-context/browser-review.json',
    'state-space-models.json', 'state-space-native.json',
  ];
  for (const name of receipts) {
    const file = `docs/teaching/evidence/${name}`;
    const receipt = read(file);
    assert.ok(receipt.passed === true || receipt.status === 'passed', `Incomplete receipt: ${file}`);
    for (const field of ['reviewedFiles', 'sourceHashes', 'hashes', 'files']) {
      for (const [source, expected] of Object.entries(receipt[field] || {})) {
        const file = name === 'state-space-native.json' && !source.includes('/')
          ? `public/learn-code/state-space-models-s4-mamba-mamba-2/${source}` : source;
        assert.equal(hash(file), expected, `Stale ${name}: ${file}`);
      }
    }
    report.receipts.push({ file, sha256: hash(file) });
  }
  const independentReview = fs.readFileSync('docs/teaching/ATTENTION-INDEPENDENT-REVIEW.md', 'utf8');
  const identities = [...independentReview.matchAll(/^\| `([^`]+)` \| `([a-f0-9]{64})` \|$/gm)];
  assert.equal(identities.length, 17, 'Incomplete independent Attention source binding');
  for (const [, file, expected] of identities) assert.equal(hash(file), expected, `Independent Attention review is stale: ${file}`);
  report.independentAttentionBindings = identities.length;

  const built = read('dist/.vite/manifest.json');
  function staticClosure(key, visited = new Set()) {
    if (visited.has(key)) return visited;
    assert.ok(built[key], `Missing built chunk: ${key}`);
    visited.add(key);
    for (const dependency of built[key].imports || []) staticClosure(dependency, visited);
    return visited;
  }
  const entryClosures = ['index.html', ...Object.keys(built).filter(key => /\/Reader\.jsx$/.test(key))].map(key => staticClosure(key));
  report.chunks = baseline.topicIds.map(id => {
    const key = `src/learn/data/${publication[id].replace('./', '')}`;
    for (const closure of entryClosures) assert.ok(!closure.has(key), `${id}: lesson eagerly imported by app/reader`);
    const file = `dist/${built[key].file}`;
    const bytes = fs.readFileSync(file);
    return { id, file, bytes: bytes.length, gzipBytes: gzipSync(bytes).length, sha256: hash(file) };
  });
  for (const id of baseline.topicIds) {
    const sourceDir = `public/learn-code/${id}`;
    const names = fs.readdirSync(sourceDir);
    assert.ok(names.length > 0);
    for (const name of names) {
      assert.ok(!name.endsWith('.pyc') && name !== '__pycache__', 'Runtime cache in public assets');
      const source = `${sourceDir}/${name}`;
      if (!fs.statSync(source).isFile()) continue;
      assert.equal(hash(`dist/learn-code/${id}/${name}`), hash(source), `Build has stale download: ${source}`);
    }
    const provenance = `${sourceDir}/data-provenance.md`;
    for (const match of fs.readFileSync(provenance, 'utf8').matchAll(/\[[^\]]*\]\(([^\s)]+)\)/g)) {
      if (!/^(https?:|#)/.test(match[1])) assert.ok(fs.existsSync(`${sourceDir}/${match[1]}`), `Missing local provenance link: ${match[1]}`);
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
console.log(JSON.stringify({ passed: report.passed, topics: report.topics, preservedOtherRows: report.preservedOtherRows, metadataRepairs: report.metadataRepairs.length, receipts: report.receipts.length, recordedCounts: report.recordedCounts, failure: report.failure }));
