const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');

const read = path => JSON.parse(fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const previousPath = 'docs/teaching/evidence/category-geometry-foundations-integration.json';
const snapshotPath = 'docs/teaching/evidence/counting-measurement-foundations-integration.json';
const logDirectory = 'scratch/counting-measurement-integration';
const definitions = [
  ['counting-combinatorics-mathematical-induction', 'counting-combinatorics', 'COUNTING-COMBINATORICS', 7],
  ['single-variable-calculus-limits-derivatives-integrals', 'single-variable-calculus', 'SINGLE-VARIABLE-CALCULUS', 6],
  ['random-variables-expectation-covariance', 'random-variables', 'RANDOM-VARIABLES', 6],
  ['sampling-measurement-experimental-design', 'sampling-measurement', 'SAMPLING-MEASUREMENT', 6],
];
const finalSources = review => review.sources || review.production || review.finalProductionSources || review.productionSourcesUnchanged;
const ledger = read(ledgerPath);
const previous = read(previousPath);
const loading = read('scratch/learning-performance/load-boundaries.json');
const routing = read('scratch/module-order-review/results.json');
const manifestTime = fs.statSync('dist/.vite/manifest.json').mtime;
assert(!fs.existsSync(snapshotPath), 'Preserve the frozen integration record.');
assert.equal(previous.reviewedScope, 62);
assert.equal(ledger.topics.filter(topic => topic.status === 'implementation-reviewed').length, 62);
for (const [path, digest] of Object.entries(previous.reviewedFiles)) assert.equal(hash(path), digest, path);
assert.equal(loading.results.length, 79);
assert.equal(loading.base, 'http://127.0.0.1:4173');
assert(new Date(loading.measuredAt) > manifestTime, 'Loading evidence must follow this build.');
assert(fs.statSync('scratch/module-order-review/results.json').mtime > manifestTime);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('DSA')).length, 22);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('reviewed mathematics')).length, 49);
assert.deepEqual(routing.errors, []);
assert.deepEqual(routing.results.map(row => row.width), [1440, 390]);
for (const row of routing.results) {
  assert.equal(row.programmingSteps, 17);
  assert.equal(row.paths, 7);
  assert.equal(row.dsaCompletedIds.length, 22);
  assert.equal(row.plannedResume, true);
  assert.deepEqual(row.newMathematicsIds, definitions.map(([id]) => id));
  assert.deepEqual(row.additionalPublishedMathematicsIds, ['ordinary-differential-equations-linear-systems']);
  assert.equal(row.nextMathematicsPlannedId, 'complex-numbers-fourier-laplace-transforms');
}

const amendments = [];
for (const [id, stem, name, sourceCount] of definitions) {
  const authorPath = `docs/teaching/evidence/${stem}-author-review.json`;
  const independentPath = `docs/teaching/evidence/${stem}-independent-review.json`;
  const author = read(authorPath), independent = read(independentPath);
  const originalSources = finalSources(author), reviewedSources = finalSources(independent);
  assert.equal(originalSources.length, sourceCount);
  assert.equal(reviewedSources.length, sourceCount);
  assert.match(independent.status, /closed|complete|passed|independently-reviewed/i);
  assert.equal(hash(authorPath), independent.authorRecord.sha256, 'Original author packet must remain immutable.');
  assert(fs.statSync(`docs/teaching/${name}-INDEPENDENT-REVIEW.md`).size > 500);
  const topic = ledger.topics.find(row => row.id === id);
  assert.equal(topic.status, 'author-verified');
  for (const source of reviewedSources) {
    const original = originalSources.find(item => item.path === source.path);
    assert(original, source.path);
    assert.equal(hash(source.path), source.sha256, source.path);
    assert.equal(topic.reviewedFiles[source.path], source.sha256, source.path);
    assert(manifestTime > fs.statSync(source.path).mtime, `Build predates final source: ${source.path}`);
    if (original.sha256 !== source.sha256) {
      assert.equal(source.amended, true);
      assert.equal(source.authorSha256, original.sha256);
      const archive = independent.originalAuthorSources.find(item => item.path === source.path);
      assert.equal(archive.sha256, original.sha256);
      assert.equal(hash(archive.archive), original.sha256, archive.archive);
      assert(independent.findingsResolved.length > 0);
      amendments.push({ topicId: id, path: source.path, authorSha256: original.sha256, finalSha256: source.sha256, independentEvidence: independentPath });
    }
  }
  assert(loading.results.some(row => row.topicId === id));
  Object.assign(topic, {
    status: 'implementation-reviewed',
    integrationRecord: 'docs/teaching/DSA-MATH-FOUNDATIONS-INTEGRATION.md',
    reviewStage: 'Final author/independent evidence and production loading/navigation integrated; user acceptance remains separate.',
  });
}
assert.equal(amendments.length, 4);
const reviewed = ledger.topics.filter(topic => topic.status === 'implementation-reviewed');
assert.equal(reviewed.length, 66);
for (const topic of reviewed) for (const [path, digest] of Object.entries(topic.reviewedFiles)) assert.equal(hash(path), digest, path);
const counts = read('docs/curriculum/curriculum-inventory.json').counts;
assert.deepEqual(counts, { modules: 28, uniqueTopics: 1218, published: 221, topicBriefs: 341, prerequisiteReviewsRecorded: 370, guidedPaths: 7 });
assert.equal(Object.keys(read('src/learn/data/lesson-manifest.json')).length, counts.published);
const buildLog = fs.readFileSync(`${logDirectory}/build.log`, 'utf8');
const buildSeconds = Number(buildLog.match(/built in ([\d.]+)s/)[1]);
const sourceBoundary = read(`${logDirectory}/source-boundary.log`);
assert.equal(sourceBoundary.passed, true);
const snapshot = {
  integratedAt: new Date().toISOString(), reviewedScope: 66, remainingScope: 8,
  previousSnapshot: { path: previousPath, sha256: hash(previousPath), reviewedScope: 62, conservedReviewedFiles: Object.keys(previous.reviewedFiles).length },
  counts, publications: counts.published,
  plannedOutlines: fs.readdirSync('src/learn/data/generated/outlines').filter(name => name.endsWith('.json')).length,
  newReviewedIds: definitions.map(([id]) => id), loading, routing,
  routingFinishedAt: fs.statSync('scratch/module-order-review/results.json').mtime.toISOString(),
  routingScope: 'Production 4173 at 1440/390: all 17 programming steps, seven paths, 22 actual DSA completion actions, four actual Math46–49 completions, traversal through separately published ODE50 to planned Fourier/Laplace51, shared progress/context and planned resume. ODE50 is not promoted by this integration. The separate published-prefix planned-resume fixture uses isolated seeded storage.',
  authorEvidence: definitions.map(([, stem]) => `docs/teaching/evidence/${stem}-author-review.json`),
  independentReview: definitions.map(([, , name]) => `docs/teaching/${name}-INDEPENDENT-REVIEW.md`),
  independentEvidence: definitions.map(([, stem]) => `docs/teaching/evidence/${stem}-independent-review.json`),
  amendments,
  reviewedFiles: Object.assign({}, ...reviewed.map(topic => topic.reviewedFiles)),
  artifactSources: ['src/learn/data/lesson-manifest.json', 'src/learn/data/generated/navigation.js', 'src/learn/data/generated/lesson-imports.js'].map(path => ({ path, sha256: hash(path) })),
  checks: ['curriculum', 'artifacts', 'source-boundary', 'runtime-organization', 'progress-before', 'inventory-before', 'build', 'loading', 'routes'].map(name => ({ path: `${logDirectory}/${name}.log`, sha256: hash(`${logDirectory}/${name}.log`) })),
  buildSeconds, buildManifestSha256: hash('dist/.vite/manifest.json'), buildManifestModifiedAt: manifestTime.toISOString(), sourceBoundary,
  warnings: ['Existing Bayesian Networks raw greater-than JSX warning.', 'Existing shared navigation chunk exceeds the 500 kB warning threshold; no threshold was raised.'],
  limits: ['Local build/loading/route observations do not imply universal performance or learner outcomes.', 'The remaining eight scoped mathematics topics are not promoted by this snapshot.', 'Per-topic author and independent executions remain separately attributed; integration does not claim complete topic suites were rerun.', 'Original author sources amended during independent review remain archived and attributed.', 'User acceptance and completion of the full 74-topic goal remain pending.'],
};
assert.equal(snapshot.plannedOutlines, 252);
assert.equal(Object.keys(snapshot.reviewedFiles).length, 418);
fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2) + '\n');
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
console.log(JSON.stringify({ integratedAt: snapshot.integratedAt, reviewed: 66, remaining: 8, buildSeconds, loadingCases: loading.results.length, counts, conservedReviewedFiles: snapshot.previousSnapshot.conservedReviewedFiles }, null, 2));
