const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');

const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const sourcesOf = record => record.sources || record.productionSources;
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const previousPath = 'docs/teaching/evidence/counting-measurement-foundations-integration.json';
const snapshotPath = 'docs/teaching/evidence/analysis-decisions-foundations-integration.json';
const logDirectory = 'scratch/analysis-decisions-integration';
const definitions = [
  ['ordinary-differential-equations-linear-systems', 'ordinary-differential-equations', 'ORDINARY-DIFFERENTIAL-EQUATIONS', 7],
  ['complex-numbers-fourier-laplace-transforms', 'complex-transforms', 'COMPLEX-FOURIER-LAPLACE', 6],
  ['conditioning-stability-numerical-analysis', 'conditioning-stability', 'CONDITIONING-STABILITY', 6],
  ['decision-theory-risk-cost-sensitive-decisions', 'decision-theory', 'DECISION-THEORY', 7],
  ['real-analysis-sequences-modes-of-convergence', 'real-analysis', 'REAL-ANALYSIS', 6],
];
const ledger = read(ledgerPath), previous = read(previousPath);
const loading = read('scratch/learning-performance/load-boundaries.json');
const routing = read('scratch/module-order-review/results.json');
const manifestTime = fs.statSync('dist/.vite/manifest.json').mtime;
assert(!fs.existsSync(snapshotPath), 'Preserve the frozen integration record.');
assert.equal(previous.reviewedScope, 66);
assert.equal(ledger.topics.filter(topic => topic.status === 'implementation-reviewed').length, 66);
for (const [file, sha256] of Object.entries(previous.reviewedFiles)) assert.equal(hash(file), sha256, file);
assert.equal(loading.results.length, 84);
assert.equal(loading.base, 'http://127.0.0.1:4173');
assert(new Date(loading.measuredAt) > manifestTime);
assert(fs.statSync('scratch/module-order-review/results.json').mtime > manifestTime);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('DSA')).length, 22);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('reviewed mathematics')).length, 54);
assert.deepEqual(routing.errors, []);
assert.deepEqual(routing.results.map(row => row.width), [1440, 390]);
for (const row of routing.results) {
  assert.equal(row.programmingSteps, 17);
  assert.equal(row.paths, 7);
  assert.equal(row.dsaCompletedIds.length, 22);
  assert.equal(row.plannedResume, true);
  assert.deepEqual(row.newMathematicsIds, definitions.map(([id]) => id));
  assert.deepEqual(row.additionalPublishedMathematicsIds, ['abstract-algebra-groups-symmetry-actions']);
  assert.equal(row.nextMathematicsPlannedId, 'partial-differential-equations-conservation-boundary-conditions');
}

const amendments = [], authorEvidence = [];
for (const [id, stem, name, sourceCount] of definitions) {
  const authorPath = `docs/teaching/evidence/${stem}-author-review.json`;
  const independentPath = `docs/teaching/evidence/${stem}-independent-review.json`;
  const author = read(authorPath), independent = read(independentPath);
  const authorReference = independent.authorRecord || independent.authorFreeze || independent.authorPacket || independent.originalAuthorPacket;
  assert.equal(hash(authorPath), authorReference.sha256, authorPath);
  assert.match(independent.status, /closed|independently reviewed/i);
  assert(fs.statSync(`docs/teaching/${name}-INDEPENDENT-REVIEW.md`).size > 500);
  const originalAuthor = stem === 'complex-transforms' ? read(independent.originalAuthorFreeze.path) : author;
  if (stem === 'complex-transforms') assert.equal(hash(independent.originalAuthorFreeze.path), independent.originalAuthorFreeze.sha256);
  const originalSources = sourcesOf(originalAuthor), finalSources = sourcesOf(independent);
  assert.equal(originalSources.length, sourceCount);
  assert.equal(finalSources.length, sourceCount);
  let realArchive;
  if (stem === 'real-analysis') {
    assert.equal(hash(independent.originalAuthorSources.path), independent.originalAuthorSources.sha256);
    realArchive = read(independent.originalAuthorSources.path);
    for (const source of realArchive.sources) assert.equal(digest(Buffer.from(source.content, source.encoding)), source.sha256, source.path);
  }
  const topic = ledger.topics.find(row => row.id === id);
  assert.equal(topic.status, 'author-verified');
  for (const source of finalSources) {
    const original = originalSources.find(item => item.path === source.path);
    assert(original, source.path);
    assert.equal(hash(source.path), source.sha256, source.path);
    assert.equal(topic.reviewedFiles[source.path], source.sha256, source.path);
    assert(manifestTime > fs.statSync(source.path).mtime, `Build predates final source: ${source.path}`);
    if (original.sha256 !== source.sha256) {
      let archive;
      if (stem === 'ordinary-differential-equations') {
        archive = independent.originalAuthorSources.find(item => item.path === source.path);
        assert.equal(hash(archive.archive), original.sha256);
        assert.equal(source.authorSha256, original.sha256);
        assert.equal(source.amended, true);
      } else if (stem === 'complex-transforms') {
        const archivedPath = path.join(independent.originalAuthorFreeze.sourceArchive, source.path);
        assert.equal(hash(archivedPath), original.sha256);
        archive = { path: archivedPath, sha256: original.sha256 };
      } else if (stem === 'real-analysis') {
        const bytes = realArchive.sources.find(item => item.path === source.path);
        assert.equal(bytes.sha256, original.sha256);
        assert(independent.changedProductionFiles.includes(source.path));
        archive = independent.originalAuthorSources;
      } else assert.fail(`Undocumented amendment: ${source.path}`);
      amendments.push({ topicId: id, path: source.path, authorSha256: original.sha256, finalSha256: source.sha256, originalArchive: archive, independentEvidence: independentPath });
    }
  }
  authorEvidence.push({ path: authorPath, sha256: hash(authorPath), originalAuthorFreeze: stem === 'complex-transforms' ? independent.originalAuthorFreeze : undefined });
  assert(loading.results.some(row => row.topicId === id));
  Object.assign(topic, { status: 'implementation-reviewed', integrationRecord: 'docs/teaching/DSA-MATH-FOUNDATIONS-INTEGRATION.md', reviewStage: 'Final author/independent evidence and production loading/navigation integrated; user acceptance remains separate.' });
}
assert.equal(amendments.length, 6);
const reviewed = ledger.topics.filter(topic => topic.status === 'implementation-reviewed');
assert.equal(reviewed.length, 71);
for (const topic of reviewed) for (const [file, sha256] of Object.entries(topic.reviewedFiles)) assert.equal(hash(file), sha256, file);
const counts = read('docs/curriculum/curriculum-inventory.json').counts;
assert.deepEqual(counts, { modules: 28, uniqueTopics: 1218, published: 226, topicBriefs: 341, prerequisiteReviewsRecorded: 370, guidedPaths: 7 });
assert.equal(Object.keys(read('src/learn/data/lesson-manifest.json')).length, counts.published);
const buildLog = fs.readFileSync(`${logDirectory}/build.log`, 'utf8');
const buildSeconds = Number(buildLog.match(/built in ([\d.]+)s/)[1]);
const sourceBoundary = read(`${logDirectory}/source-boundary.log`);
assert.equal(sourceBoundary.passed, true);
const snapshot = {
  integratedAt: new Date().toISOString(), reviewedScope: 71, remainingScope: 3,
  previousSnapshot: { path: previousPath, sha256: hash(previousPath), reviewedScope: 66, conservedReviewedFiles: Object.keys(previous.reviewedFiles).length },
  counts, publications: counts.published,
  plannedOutlines: fs.readdirSync('src/learn/data/generated/outlines').filter(name => name.endsWith('.json')).length,
  newReviewedIds: definitions.map(([id]) => id), loading, routing,
  routingFinishedAt: fs.statSync('scratch/module-order-review/results.json').mtime.toISOString(),
  routingScope: 'Production 4173 at 1440/390: all 17 programming steps, seven paths, 22 actual DSA completion actions, five actual Math50–54 completions, traversal through separately published Algebra55 to planned PDE56, shared progress/context and planned resume. Algebra55 is not promoted by this integration. The separate published-prefix planned-resume fixture uses isolated seeded storage.',
  authorEvidence,
  independentReview: definitions.map(([, , name]) => `docs/teaching/${name}-INDEPENDENT-REVIEW.md`),
  independentEvidence: definitions.map(([, stem]) => ({ path: `docs/teaching/evidence/${stem}-independent-review.json`, sha256: hash(`docs/teaching/evidence/${stem}-independent-review.json`) })),
  amendments, reviewedFiles: Object.assign({}, ...reviewed.map(topic => topic.reviewedFiles)),
  artifactSources: ['src/learn/data/lesson-manifest.json', 'src/learn/data/generated/navigation.js', 'src/learn/data/generated/lesson-imports.js'].map(file => ({ path: file, sha256: hash(file) })),
  checks: ['curriculum', 'artifacts', 'source-boundary', 'runtime-organization', 'progress-before', 'inventory-before', 'build', 'loading', 'routes'].map(name => ({ path: `${logDirectory}/${name}.log`, sha256: hash(`${logDirectory}/${name}.log`) })),
  buildSeconds, buildManifestSha256: hash('dist/.vite/manifest.json'), buildManifestModifiedAt: manifestTime.toISOString(), sourceBoundary,
  warnings: ['Existing Bayesian Networks raw greater-than JSX warning.', 'Existing shared navigation chunk exceeds the 500 kB warning threshold; no threshold was raised.'],
  limits: ['Local build/loading/route observations do not imply universal performance or learner outcomes.', 'The remaining three scoped mathematics topics are not promoted by this snapshot.', 'Per-topic author and independent executions remain separately attributed; integration does not claim complete topic suites were rerun.', 'Original author sources amended during independent review remain archived and attributed.', 'User acceptance and completion of the full 74-topic goal remain pending.'],
};
assert.equal(snapshot.plannedOutlines, 247);
assert.equal(Object.keys(snapshot.reviewedFiles).length, 450);
fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2) + '\n');
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
console.log(JSON.stringify({ integratedAt: snapshot.integratedAt, reviewed: 71, remaining: 3, buildSeconds, loadingCases: loading.results.length, counts, conservedReviewedFiles: snapshot.previousSnapshot.conservedReviewedFiles }, null, 2));
