const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');

const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file.replaceAll('\\', '/')));
const bind = path => ({ path, sha256: hash(path) });
function sourcesOf(record) {
  const sources = record.sources || record.productionSources || record.sourceHashes;
  assert(sources, 'Review must identify exact production sources.');
  return Array.isArray(sources) ? sources : Object.entries(sources).map(([path, sha256]) => ({ path, sha256 }));
}

const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const previousPath = 'docs/teaching/evidence/analysis-decisions-foundations-integration.json';
const snapshotPath = 'docs/teaching/evidence/dsa-math-foundations-complete-integration.json';
const logDirectory = 'scratch/foundations-completion-integration';
assert(!fs.existsSync(snapshotPath), 'Preserve the frozen completion snapshot.');
const ledger = read(ledgerPath), previous = read(previousPath);
const manifest = read('src/learn/data/lesson-manifest.json');
const buildTime = fs.statSync('dist/.vite/manifest.json').mtime;
const loading = read('scratch/learning-performance/load-boundaries.json');
const routing = read('scratch/module-order-review/results.json');
const definitions = [
  ['abstract-algebra-groups-symmetry-actions', 'abstract-algebra', 'ABSTRACT-ALGEBRA', 6],
  ['partial-differential-equations-conservation-boundary-conditions', 'pde', 'PARTIAL-DIFFERENTIAL-EQUATIONS', 6],
  ['numerical-pdes-grids-finite-elements-stability', 'numerical-pdes', 'NUMERICAL-PDES', 6],
  ['dynamic-programming-states-transitions-optimization', 'dp-state-families', 'DP-STATE-FAMILIES', 7],
  ['arrays-strings-hash-maps', 'bitwise-foundations', 'BITWISE-FOUNDATIONS', 11],
  ['linked-lists-stacks-queues', 'linked-traversal-extension', 'LINKED-TRAVERSAL-MONOTONIC', 11],
];

assert.equal(previous.reviewedScope, 71);
assert.equal(loading.results.length, 87);
assert.equal(loading.base, 'http://127.0.0.1:4173');
assert(new Date(loading.measuredAt) > buildTime);
assert(fs.statSync('scratch/module-order-review/results.json').mtime > buildTime);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('DSA')).length, 22);
assert.equal(loading.results.filter(row => row.topicId && row.case.startsWith('reviewed mathematics')).length, 57);
assert.deepEqual(routing.errors, []);
assert.deepEqual(routing.results.map(row => row.width), [1440, 390]);
for (const row of routing.results) {
  assert.equal(row.programmingSteps, 17);
  assert.equal(row.paths, 7);
  assert.equal(row.dsaCompletedIds.length, 22);
  assert.equal(row.plannedResume, true);
  assert.deepEqual(row.newMathematicsIds, definitions.slice(0, 3).map(([id]) => id));
  assert.equal(row.mathematicsModuleBoundary, true);
  assert.notEqual(row.mathematicsSuccessor.moduleId, 'math-foundations');
}

const dpOriginalPath = 'docs/teaching/evidence/dp-state-families-original.json';
const dpOriginal = read(dpOriginalPath);
for (const source of dpOriginal.files) assert.equal(hash(source.archive), source.sha256, source.archive);
const previousSourceAmendments = [];
for (const [file, sha256] of Object.entries(previous.reviewedFiles)) {
  if (hash(file) === sha256) continue;
  const archive = dpOriginal.files.find(row => row.source === file);
  assert(archive, `Unexpected change to integrated source: ${file}`);
  assert.equal(archive.sha256, sha256);
  assert.equal(hash(archive.archive), sha256);
  previousSourceAmendments.push({ path: file, previousSha256: sha256, finalSha256: hash(file), originalArchive: bind(archive.archive), reason: 'Reviewed interval/tree/digit DP extension' });
}
assert.equal(previousSourceAmendments.length, 3);

const reviewEvidence = [], extensionReviews = [];
for (const [id, stem, name, sourceCount] of definitions) {
  const authorPath = `docs/teaching/evidence/${stem}-author-review.json`;
  const independentPath = `docs/teaching/evidence/${stem}-independent-review.json`;
  const independent = read(independentPath);
  assert.match(independent.status, /closed|passed|independently reviewed|independent review complete/i);
  const evidenceEntries = Array.isArray(independent.evidenceHashes) ? independent.evidenceHashes : Object.entries(independent.evidenceHashes || {}).map(([path, sha256]) => ({ path, sha256 }));
  const authorReference = [independent.authorRecord, independent.authorFreeze, independent.authorPacket, independent.originalAuthorPacket, ...evidenceEntries.filter(value => value.path === authorPath)].find(value => value && typeof value === 'object' && value.sha256);
  assert(authorReference, independentPath);
  assert.equal(hash(authorPath), authorReference.sha256, authorPath);
  const sources = sourcesOf(independent);
  assert.equal(sources.length, sourceCount);
  const verificationRecord = `docs/teaching/${name}-INDEPENDENT-REVIEW.md`;
  assert(fs.statSync(verificationRecord).size > 500);
  const reviewedFiles = Object.fromEntries(sources.map(source => {
    assert.equal(hash(source.path), source.sha256, source.path);
    assert(buildTime > fs.statSync(source.path).mtime, `Build predates final source: ${source.path}`);
    return [source.path, source.sha256];
  }));
  const topic = ledger.topics.find(row => row.id === id);
  if (topic) {
    assert.equal(topic.status, 'author-verified');
    for (const [file, sha256] of Object.entries(reviewedFiles)) assert.equal(topic.reviewedFiles[file], sha256, file);
    Object.assign(topic, { status: 'implementation-reviewed', integrationRecord: 'docs/teaching/DSA-MATH-FOUNDATIONS-INTEGRATION.md', reviewStage: 'Final independent source evidence and production integration complete; user acceptance remains separate.' });
  }
  if (id === 'dynamic-programming-states-transitions-optimization' || !topic) {
    extensionReviews.push({ topicId: id, verificationRecord, independentEvidence: bind(independentPath), reviewedFiles: topic ? topic.reviewedFiles : reviewedFiles });
  }
  assert(loading.results.some(row => row.topicId === id));
  reviewEvidence.push({ topicId: id, author: bind(authorPath), independent: bind(independentPath), narrative: bind(verificationRecord), reviewedFiles });
}

// Author corrections and the tiny obsolete Trees bridge retain their originals.
const algebraOriginal = read('docs/teaching/evidence/abstract-algebra-original-author-sources.json');
for (const source of algebraOriginal.sources) assert.equal(digest(Buffer.from(source.content, source.encoding)), source.sha256, source.path);
const pdeOriginal = read('docs/teaching/evidence/pde-independent-amendment-originals.json');
for (const source of pdeOriginal.archived) assert.equal(hash(source.archive), source.sha256, source.archive);
const numericalPdeOriginal = read('docs/teaching/evidence/numerical-pdes-endpoint-before.json');
for (const source of numericalPdeOriginal.files) assert.equal(hash(source.archive), source.sha256, source.archive);
const treeAmendmentPath = 'docs/teaching/evidence/trees-published-bridge-amendment.json';
const treeAmendment = read(treeAmendmentPath);
assert.equal(digest(Buffer.from(treeAmendment.before.content, treeAmendment.before.encoding)), treeAmendment.before.sha256);
assert.equal(hash(treeAmendment.path), treeAmendment.afterSha256);
assert(buildTime > fs.statSync(treeAmendment.path).mtime);

const reviewed = ledger.topics.filter(topic => topic.status === 'implementation-reviewed');
assert.equal(reviewed.length, 74);
const reviewedFiles = Object.assign({}, ...reviewed.map(topic => topic.reviewedFiles));
for (const [file, sha256] of Object.entries(reviewedFiles)) assert.equal(hash(file), sha256, file);
const counts = read('docs/curriculum/curriculum-inventory.json').counts;
assert.equal(counts.modules, 28);
assert.equal(counts.uniqueTopics, 1218);
assert.equal(counts.guidedPaths, 7);
assert.equal(counts.published, 228);
assert.equal(counts.topicBriefs, 341);
assert.equal(Object.keys(manifest).length, counts.published);
const sourceBoundary = read(`${logDirectory}/source-boundary.log`);
assert.equal(sourceBoundary.passed, true);
const buildLog = fs.readFileSync(`${logDirectory}/build.log`, 'utf8');
const buildSeconds = Number(buildLog.match(/built in ([\d.]+)s/)[1]);
const practiceCoverage = read('scratch/dsa-gap-documentation/practice-counts.json');
assert.deepEqual(practiceCoverage.totals, { topics: 22, placements: 164, distinct: 151, optionalPlacements: 29 });
for (const source of practiceCoverage.topics) assert.equal(hash(source.path), source.sha256, source.path);
const snapshot = {
  integratedAt: new Date().toISOString(), reviewedScope: 74, remainingScope: 0,
  previousSnapshot: { ...bind(previousPath), reviewedScope: 71, reviewedFileCount: Object.keys(previous.reviewedFiles).length, unchangedReviewedFileCount: Object.keys(previous.reviewedFiles).length - previousSourceAmendments.length },
  counts, publications: counts.published, practiceCoverage,
  practiceStandard: 'docs/teaching/DSA-PRACTICE-STANDARD.md',
  plannedOutlines: fs.readdirSync('src/learn/data/generated/outlines').filter(name => name.endsWith('.json')).length,
  newReviewedIds: definitions.slice(0, 3).map(([id]) => id), reviewEvidence, extensionReviews,
  previousSourceAmendments, reviewedFiles, treeBridgeAmendment: bind(treeAmendmentPath),
  originalEvidence: [dpOriginalPath, 'docs/teaching/evidence/abstract-algebra-original-author-sources.json', 'docs/teaching/evidence/pde-author-review-before-independent-amendment.json', 'docs/teaching/evidence/pde-independent-amendment-originals.json', 'docs/teaching/evidence/numerical-pdes-endpoint-before.json'].map(bind),
  loading, routing, routingFinishedAt: fs.statSync('scratch/module-order-review/results.json').mtime.toISOString(),
  openedIntegrationImages: ['header-1440.png', 'header-390.png', 'sidebar-390.png'].map(name => bind(`scratch/module-order-review/${name}`)),
  routingScope: 'Production 4173: 17 programming steps, seven path entries/counts, 22 actual DSA completions, three actual final mathematics completions into the next module, shared context/progress, and an isolated seeded planned-resume fixture at 1440/390.',
  artifactSources: ['src/learn/data/lesson-manifest.json', 'src/learn/data/generated/navigation.js', 'src/learn/data/generated/lesson-imports.js'].map(bind),
  checks: ['curriculum', 'artifacts', 'source-boundary', 'runtime-organization', 'progress-before', 'inventory-before', 'build', 'loading', 'routes'].map(name => bind(`${logDirectory}/${name}.log`)),
  buildSeconds, buildManifest: bind('dist/.vite/manifest.json'), buildManifestModifiedAt: buildTime.toISOString(), sourceBoundary,
  warnings: ['Existing unrelated Bayesian Networks raw greater-than JSX warning.', 'Existing shared navigation chunk exceeds the 500 kB warning threshold; no threshold was raised.'],
  limits: ['Each per-topic author and independent check retains its actual scope; integration does not claim every complete native/browser suite was repeated.', 'Local loading and route observations do not establish universal performance or learner outcomes.', 'Finite practice selections do not guarantee solving every interview question.', 'User acceptance remains separate; no deployment was requested.'],
};
assert.equal(snapshot.plannedOutlines, 245);
fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2) + '\n');
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
console.log(JSON.stringify({ integratedAt: snapshot.integratedAt, reviewed: 74, remaining: 0, sourceHashes: Object.keys(reviewedFiles).length, extensionReviews: extensionReviews.length, loadingCases: loading.results.length, counts, buildSeconds }, null, 2));
