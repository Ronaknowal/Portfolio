const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');

const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const previousPath = 'docs/teaching/evidence/calculus-analysis-topology-integration.json';
const snapshotPath = 'docs/teaching/evidence/category-geometry-foundations-integration.json';
const logDirectory = 'scratch/category-geometry-integration';
const ledger = read(ledgerPath);
const previous = read(previousPath);
const loading = read('scratch/learning-performance/load-boundaries.json');
const routing = read('scratch/module-order-review/results.json');
const manifestTime = fs.statSync('dist/.vite/manifest.json').mtime;
const definitions = [
  ['category-theory-emerging-use-in-ml', 'CATEGORY-THEORY', 'category-theory'],
  ['differential-geometry-riemannian-manifolds', 'DIFFERENTIAL-GEOMETRY', 'differential-geometry'],
  ['algebra-functions-exponentials-logarithms', 'ALGEBRA-FUNCTIONS', 'algebra-functions'],
  ['sets-logic-relations-proof-techniques', 'SETS-LOGIC', 'sets-logic'],
  ['geometry-trigonometry-coordinate-reasoning', 'GEOMETRY-TRIGONOMETRY', 'geometry-trigonometry'],
];

assert(!fs.existsSync(snapshotPath), 'Preserve the existing frozen integration record.');
assert.equal(previous.reviewedScope, 57);
assert.equal(ledger.topics.filter(topic => topic.status === 'implementation-reviewed').length, 57);
for (const [file, digest] of Object.entries(previous.reviewedFiles)) assert.equal(hash(file), digest, file);
assert.equal(loading.results.length, 75);
assert.equal(loading.base, 'http://127.0.0.1:4173');
assert(new Date(loading.measuredAt) > manifestTime, 'Loading evidence must follow this build.');
assert(fs.statSync('scratch/module-order-review/results.json').mtime > manifestTime);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('DSA')).length, 22);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('reviewed mathematics')).length, 45);
assert.deepEqual(routing.errors, []);
assert.deepEqual(routing.results.map(row => row.width), [1440, 390]);
for (const row of routing.results) {
  assert.equal(row.programmingSteps, 17);
  assert.equal(row.paths, 7);
  assert.equal(row.dsaCompletedIds.length, 22);
  assert.equal(row.plannedResume, true);
  assert.deepEqual(row.newMathematicsIds, definitions.map(([id]) => id));
  assert.equal(row.nextMathematicsPlannedId, 'counting-combinatorics-mathematical-induction');
}

for (const [id, name, evidenceName] of definitions) {
  const authorPath = `docs/teaching/evidence/${evidenceName}-author-review.json`;
  const independentPath = `docs/teaching/evidence/${evidenceName}-independent-review.json`;
  const author = read(authorPath);
  const independent = read(independentPath);
  const authorSources = author.production || author.sources;
  const independentSources = independent.production || independent.sources;
  assert.equal(authorSources.length, 6);
  assert.equal(independentSources.length, 6);
  assert.match(independent.status, /closed|complete|passed/i);
  assert(fs.statSync(`docs/teaching/${name}-INDEPENDENT-REVIEW.md`).size > 500);
  const topic = ledger.topics.find(row => row.id === id);
  assert.equal(topic.status, 'author-verified');
  for (const { path, sha256 } of independentSources) {
    assert.equal(hash(path), sha256, path);
    assert.equal(authorSources.find(source => source.path === path).sha256, sha256, path);
    assert.equal(topic.reviewedFiles[path], sha256, path);
    assert(manifestTime > fs.statSync(path).mtime, `Build predates final source: ${path}`);
  }
  assert(loading.results.some(row => row.topicId === id));
  topic.status = 'implementation-reviewed';
  topic.authorEvidence = authorPath;
  topic.independentEvidence = independentPath;
  topic.independentReviewRecord = `docs/teaching/${name}-INDEPENDENT-REVIEW.md`;
  topic.integrationRecord = 'docs/teaching/DSA-MATH-FOUNDATIONS-INTEGRATION.md';
}

const reviewed = ledger.topics.filter(topic => topic.status === 'implementation-reviewed');
assert.equal(reviewed.length, 62);
for (const topic of reviewed) {
  for (const [file, digest] of Object.entries(topic.reviewedFiles)) assert.equal(hash(file), digest, file);
}
const counts = read('docs/curriculum/curriculum-inventory.json').counts;
assert.deepEqual(counts, { modules: 28, uniqueTopics: 1218, published: 216, topicBriefs: 341, prerequisiteReviewsRecorded: 370, guidedPaths: 7 });
const publications = Object.keys(read('src/learn/data/lesson-manifest.json')).length;
assert.equal(publications, counts.published);
const buildLog = fs.readFileSync(`${logDirectory}/build.log`, 'utf8');
const buildSeconds = Number(buildLog.match(/built in ([\d.]+)s/)[1]);
const sourceBoundary = read(`${logDirectory}/source-boundary.log`);
assert.equal(sourceBoundary.passed, true);
const snapshot = {
  integratedAt: new Date().toISOString(), reviewedScope: 62, remainingScope: 12,
  previousSnapshot: { path: previousPath, sha256: hash(previousPath), reviewedScope: 57, conservedReviewedFiles: Object.keys(previous.reviewedFiles).length },
  counts, publications, plannedOutlines: fs.readdirSync('src/learn/data/generated/outlines').filter(name => name.endsWith('.json')).length,
  newReviewedIds: definitions.map(([id]) => id), loading, routing,
  routingFinishedAt: fs.statSync('scratch/module-order-review/results.json').mtime.toISOString(),
  routingScope: 'Production 4173 at 1440/390: all 17 programming steps, seven paths, 22 actual DSA completion actions, five actual Math41–45 completions into planned Counting46, shared module/progress and planned resume. The separate Deep Learning published prefix is seeded only in isolated fixture storage.',
  authorEvidence: definitions.map(([, , name]) => `docs/teaching/evidence/${name}-author-review.json`),
  independentReview: definitions.map(([, name]) => `docs/teaching/${name}-INDEPENDENT-REVIEW.md`),
  independentEvidence: definitions.map(([, , name]) => `docs/teaching/evidence/${name}-independent-review.json`),
  reviewedFiles: Object.assign({}, ...reviewed.map(topic => topic.reviewedFiles)),
  artifactSources: ['src/learn/data/lesson-manifest.json', 'src/learn/data/generated/navigation.js', 'src/learn/data/generated/lesson-imports.js'].map(path => ({ path, sha256: hash(path) })),
  checks: ['curriculum', 'artifacts', 'source-boundary', 'runtime-organization', 'progress-before', 'inventory-before', 'build', 'loading', 'routes'].map(name => ({ path: `${logDirectory}/${name}.log`, sha256: hash(`${logDirectory}/${name}.log`) })),
  buildSeconds, buildManifestSha256: hash('dist/.vite/manifest.json'), buildManifestModifiedAt: manifestTime.toISOString(), sourceBoundary,
  warnings: ['Existing Bayesian Networks raw greater-than JSX warning.', 'Existing shared navigation chunk exceeds the 500 kB warning threshold; no threshold was raised.'],
  limits: ['Local build/loading/route observations do not imply universal performance or learner outcomes.', 'Unregistered drafts and the remaining twelve scoped mathematics lessons are not promoted.', 'Author/independent per-topic evidence remains separately attributed; integration does not claim those complete suites were rerun.', 'User acceptance and completion of the full 74-topic goal remain pending.'],
};
assert.equal(snapshot.plannedOutlines, 257);
fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2) + '\n');
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
console.log(JSON.stringify({ integratedAt: snapshot.integratedAt, reviewed: 62, remaining: 12, buildSeconds, loading: loading.measuredAt, routing: snapshot.routingFinishedAt, counts, conservedReviewedFiles: snapshot.previousSnapshot.conservedReviewedFiles }, null, 2));
