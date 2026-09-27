// Final conservation and evidence check for the scoped prepared-DL completion.
// Does not rewrite lesson checkpoints or recertify unrelated historical rows.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { filesMatch, validateDeliveryLedger } from './lib/lesson-delivery.mjs';
import { tracks, learningPaths } from './lib/authoring-curriculum.mjs';
import { topicCatalogue } from '../src/learn/data/curriculum/topic-catalogue.js';

const directory = 'docs/teaching/deep-learning-completion';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const digest = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const baseline = read(`${directory}/baseline.json`);
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const manifest = read('src/learn/data/lesson-manifest.json');
validateDeliveryLedger(ledger, new Set(Object.keys(topicCatalogue)));
assert.equal(baseline.topicIds.length, 24);
const module = tracks.find(track => track.id === 'deep-learning-fundamentals');
assert.ok(module);
assert.equal(module.topicIds.length, 42);
assert.deepEqual(module.topicIds.slice(18), baseline.topicIds, 'Module teaching order changed');
for (const id of module.topicIds.slice(0, 18)) {
  assert.equal(ledger.topics[id].implementation.status, 'complete', `Prior DL checkpoint changed: ${id}`);
  assert.ok(filesMatch(process.cwd(), ledger.topics[id].implementation.reviewedFiles),
    `Previously reviewed DL source changed: ${id}`);
}
assert.deepEqual(Object.keys(ledger.topics).sort(), Object.keys(baseline.rowHashes).sort(), 'Ledger lost or added a topic');
const selected = new Set(baseline.topicIds);
for (const [id, row] of Object.entries(ledger.topics)) {
  if (!selected.has(id)) assert.equal(digest(row), baseline.rowHashes[id], `Unrelated ledger row changed: ${id}`);
}
for (const [id, source] of Object.entries(baseline.manifest)) {
  assert.equal(manifest[id], source, `Existing publication owner changed: ${id}`);
}
const additions = Object.keys(manifest).filter(id => !Object.hasOwn(baseline.manifest, id)).sort();
assert.deepEqual(additions, ['mini-batches-training-loops-gradient-accumulation', 'neural-training-diagnostics-reproducible-experiments']);
const built = read('dist/.vite/manifest.json');
function staticImports(entry, found = new Set()) {
  if (found.has(entry)) return found;
  assert.ok(built[entry], `Missing build entry: ${entry}`);
  found.add(entry);
  for (const dependency of built[entry].imports ?? []) staticImports(dependency, found);
  return found;
}
const entryPoints = ['index.html', 'src/learn/LearnHub.jsx', 'src/learn/Reader.jsx'];
for (const entry of entryPoints) {
  assert.ok([...staticImports(entry)].every(key => !key.startsWith('src/learn/data/topics/')),
    `${entry} eagerly imports a lesson body`);
}
for (const id of baseline.topicIds) {
  const source = `src/learn/data/${manifest[id].replace(/^\.\//, '')}`;
  assert.ok(built[source]?.isDynamicEntry, `Prepared lesson is not loaded on demand: ${id}`);
}
const topics = baseline.topicIds.map(id => {
  const row = ledger.topics[id];
  assert.equal(row.content.status, 'complete', `${id}: content incomplete`);
  assert.equal(row.implementation.status, 'complete', `${id}: implementation incomplete`);
  assert.ok(filesMatch(process.cwd(), row.content.files), `${id}: stale content evidence`);
  assert.ok(filesMatch(process.cwd(), row.implementation.reviewedFiles), `${id}: stale implementation evidence`);
  assert.equal(row.revision, baseline.originalEntries[id].revision, `${id}: prepared revision replaced`);
  const evidence = row.implementation.evidence;
  for (const stage of ['author', 'independent', 'browser', 'integration']) {
    const receipt = read(evidence[stage]);
    assert.equal(receipt.passed, true, `${id}: ${stage} not passed`);
    assert.ok((receipt.findings ?? []).every(item => item.status === 'closed'), `${id}: open ${stage} finding`);
    if (receipt.evidenceHashes) assert.ok(filesMatch(process.cwd(), receipt.evidenceHashes), `${id}: stale ${stage} evidence attachment`);
  }
  return { id, revision: row.revision, evidence };
});
const result = {
  passed: true, checkedAt: new Date().toISOString(),
  scope: 'All 24 prepared DL revisions completed; unrelated historical evidence is preserved, not recertified.',
  catalogueTopics: Object.keys(topicCatalogue).length, modules: tracks.length, guidedPaths: learningPaths.length,
  dlTopics: module.topicIds.length, completedPreparedRevisions: topics.length,
  preservedPriorDlSourceCheckpoints: 18,
  preservedUnrelatedLedgerRows: Object.keys(ledger.topics).length - topics.length,
  preservedPublicationMappings: Object.keys(baseline.manifest).length,
  addedPublicationMappings: additions, publicationCount: Object.keys(manifest).length,
  lazyLoading: { checkedEntryPoints: entryPoints, individuallyDeferredLessonBodies: baseline.topicIds.length },
  recordedContentComplete: Object.values(ledger.topics).filter(row => row.content.status === 'complete').length,
  recordedImplementationComplete: Object.values(ledger.topics).filter(row => row.implementation.status === 'complete').length,
  currentContentAndImplementationCheckpoints: Object.values(ledger.topics).filter(row =>
    filesMatch(process.cwd(), row.content.files) && filesMatch(process.cwd(), row.implementation.reviewedFiles)).length,
  topics,
};
assert.equal(result.catalogueTopics, 1461);
assert.equal(result.modules, 29);
assert.equal(result.guidedPaths, 10);
fs.writeFileSync(`${directory}/integration-checks.json`, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ ...result, topics: topics.map(topic => topic.id) }));
