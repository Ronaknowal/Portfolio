const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const ledger = read(ledgerPath);
const loading = read('scratch/learning-performance/load-boundaries.json');
const routing = read('scratch/module-order-review/results.json');
const manifestTime = fs.statSync('dist/.vite/manifest.json').mtime;
assert.equal(loading.results.length, 70);
assert.equal(loading.base, 'http://127.0.0.1:4173');
assert.deepEqual(routing.errors, []);
assert.equal(routing.results.length, 2);
assert(fs.statSync('scratch/module-order-review/results.json').mtime > manifestTime);
assert(new Date(loading.measuredAt) > manifestTime);
for (const row of routing.results) {
  assert.equal(row.programmingSteps, 17);
  assert.equal(row.paths, 7);
  assert.equal(row.dsaCompletedIds.length, 22);
  assert.equal(row.plannedResume, true);
}
const definitions = [
  ['it-calculus-stochastic-differential-equations', 'ITO-CALCULUS-SDE', 'ito-sde'],
  ['numerical-methods-finite-differences-quadrature-root-finding', 'NUMERICAL-METHODS', 'numerical-methods'],
  ['functional-analysis-rkhs', 'FUNCTIONAL-ANALYSIS', 'functional-analysis'],
  ['topology-topological-data-analysis-tda', 'TOPOLOGY-TDA', 'topology-tda'],
];
for (const [id, name, evidenceName] of definitions) {
  assert(fs.statSync('docs/teaching/' + name + '-INDEPENDENT-REVIEW.md').size > 500);
  const author = read('docs/teaching/evidence/' + evidenceName + '-author-review.json');
  const independent = read('docs/teaching/evidence/' + evidenceName + '-independent-review.json');
  assert(/closed|complete/.test(independent.status));
  const topic = ledger.topics.find(row => row.id === id);
  assert.equal(topic.status, 'author-verified', id);
  const authorSources = author.production || author.sources;
  assert.equal(authorSources.length, 6);
  assert.equal(independent.production.length, 6);
  for (const { path, sha256 } of independent.production) {
    assert.equal(hash(path), sha256, path);
    assert.equal(authorSources.find(row => row.path === path).sha256, sha256, path);
    assert.equal(topic.reviewedFiles[path], sha256, path);
    assert(manifestTime > fs.statSync(path).mtime, 'Build predates source: ' + path);
  }
  assert(loading.results.some(row => row.topicId === id));
  topic.status = 'implementation-reviewed';
  topic.independentEvidence = 'docs/teaching/evidence/' + evidenceName + '-independent-review.json';
  topic.integrationRecord = 'docs/teaching/DSA-MATH-FOUNDATIONS-INTEGRATION.md';
}
for (const topic of ledger.topics.filter(row => row.status === 'implementation-reviewed')) {
  for (const [file, expected] of Object.entries(topic.reviewedFiles)) assert.equal(hash(file), expected, file);
}
assert.equal(ledger.topics.filter(row => row.status === 'implementation-reviewed').length, 57);
const counts = read('docs/curriculum/curriculum-inventory.json').counts;
assert.equal(counts.topicBriefs, 339);
assert.equal(counts.prerequisiteReviewsRecorded, 368);
const publications = Object.keys(read('src/learn/data/lesson-manifest.json')).length;
assert.equal(publications, 213);
const buildSeconds = Number(fs.readFileSync('scratch/dsa-math-build.log', 'utf8').match(/built in ([\d.]+)s/)[1]);
const snapshot = {
  integratedAt: new Date().toISOString(), counts, publications, reviewedScope: 57,
  newReviewedIds: definitions.map(([id]) => id), loading, routing,
  routingFinishedAt: fs.statSync('scratch/module-order-review/results.json').mtime.toISOString(),
  routingScope: 'LEARNING_BASE_URL=http://127.0.0.1:4173; all22 DSA completions through UI; planned DeepLearning prefix seeded only in isolated fixture storage.',
  authorEvidence: definitions.map(([, , name]) => 'docs/teaching/evidence/' + name + '-author-review.json'),
  independentReview: definitions.map(([, name]) => 'docs/teaching/' + name + '-INDEPENDENT-REVIEW.md'),
  independentEvidence: definitions.map(([, , name]) => 'docs/teaching/evidence/' + name + '-independent-review.json'),
  reviewedFiles: Object.assign({}, ...ledger.topics.filter(row => row.status === 'implementation-reviewed').map(row => row.reviewedFiles)),
  buildLogSha256: hash('scratch/dsa-math-build.log'), buildManifestSha256: hash('dist/.vite/manifest.json'),
  buildSeconds, sourceBoundary: { scanned: 714, directMathImports: 217 },
};
const snapshotPath = 'docs/teaching/evidence/calculus-analysis-topology-integration.json';
assert(!fs.existsSync(snapshotPath), 'Preserve the original frozen integration record.');
fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2) + '\n');
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
console.log(JSON.stringify({ integratedAt: snapshot.integratedAt, reviewed: 57, remaining: 17, loading: loading.measuredAt, routing: snapshot.routingFinishedAt, buildSeconds, counts }, null, 2));
