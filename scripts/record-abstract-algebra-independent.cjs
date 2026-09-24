const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const path = require('node:path');
const root = 'scratch/abstract-algebra-independent';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const stamp = file => ({ path: file, sha256: hash(file), bytes: fs.statSync(file).size });
const authorPath = 'docs/teaching/evidence/abstract-algebra-author-review.json';
const independentPath = 'docs/teaching/evidence/abstract-algebra-independent-review.json';
assert(!fs.existsSync(independentPath));
const author = read(authorPath), payload = read(`${root}/payload.json`);
const native = read(`${root}/native-results.json`), browser = read(`${root}/browser-results.json`), graph = read(`${root}/graph-paint-results.json`);
assert.equal(hash(authorPath), payload.authorPacket.sha256);
assert.equal(hash(`${root}/author-baseline.json`), hash(authorPath));
for (const source of payload.sources) assert.equal(hash(source.path), source.sha256);
assert.deepEqual(native.sources, payload.sources); assert.deepEqual(graph.sources, payload.sources);
assert.equal(native.status, 'passed'); assert.equal(browser.status, 'passed'); assert.equal(graph.status, 'passed');
const originalSources = author.productionSources.map(source => {
  const archived = path.join(root, 'author-sources', source.path);
  assert.equal(hash(archived), source.sha256);
  return { ...source, encoding: 'base64', content: fs.readFileSync(archived).toString('base64') };
});
const archivePath = 'docs/teaching/evidence/abstract-algebra-original-author-sources.json';
assert(!fs.existsSync(archivePath));
fs.writeFileSync(archivePath, JSON.stringify({ originalAuthorFrozenAt: author.frozenAt, authorPacket: stamp(authorPath), sources: originalSources }, null, 2) + '\n');
const opened = ['changed-composition-320','changed-chiral-orbit-320','representative-failure-320','normal-representative-320','changed-map-raw-320','changed-map-averaged-320','projection-proof-320','changed-audit-answer-320','modular-preimages-320','actual-program-320','references-320','final-cayley-paint-1440','final-cayley-paint-320','final-action-paint-320'];
const record = {
  reviewedAt: new Date().toISOString(), topicId: author.topicId, status: 'Independent review closed; production integration and user acceptance remain separate.', reviewer: 'root',
  sources: payload.sources, originalAuthorFrozenAt: author.frozenAt, originalAuthorPacket: stamp(authorPath), originalAuthorSources: stamp(archivePath),
  sourceReview: 'Full body, group/coset/action/counting/quotient/representation/projection/ring proofs, all eight changed practice solutions and two checkpoints, all ten actual programs, model exports, final labs/CSS and brief read. Final lab/CSS display amendments were read after author freeze.',
  findingsResolved: [{ finding: 'Three reverse-cycle arrowheads terminate inside destination circles, hiding their direction in the actual Cayley diagram.', amendment: 'Move only those path endpoints from89+140i to105+140i, beyond the circles and marker tip; all other production sources unchanged.', source: 'src/learn/components/lesson-labs/AbstractAlgebraLabs.jsx', originalSha256: author.productionSources.find(source => source.path.endsWith('AbstractAlgebraLabs.jsx')).sha256 }],
  native: { evidence: stamp(`${root}/native-results.json`), result: native },
  browser: { evidence: stamp(`${root}/browser-results.json`), result: browser, scope: 'Independent three-width changed interactions and actual question/code/output/anchor/equation review precedes the static arrow amendment. It is not relabeled as a full post-amendment run.' },
  finalGraphClosure: { evidence: stamp(`${root}/graph-paint-results.json`), result: graph, scope: 'Post-amendment actual-font three-width graph geometry/local keyboard scrolling and replacement captures on all six final hashes; two corrected graph images and the unchanged action image actually opened.' },
  finalImagesActuallyOpened: opened.map(name => stamp(`${root}/${name}.png`)),
  imageAttribution: 'Eleven unaffected view captures were opened from the full independent pass. Three graph captures were opened after the arrow amendment. Earlier blank/stale captures from keyboard End are rejected; stable target bounds and local ArrowRight replaced them. The actual-program capture visibly shows the modular program; all ten programs were separately compared through the DOM.',
  primarySourceRevisited: [{ url: 'https://proceedings.mlr.press/v48/cohenc16.pdf', inspected: 'Introduction and section2, explicit potentially different input/output representations and invariance as identity output action. No universal accuracy inference.' }, { url: 'https://www.math.clemson.edu/~macaule/classes/m18_math4120/slides/math4120_lecture-5-01_h.pdf', inspected: 'Fresh tool fetch failed with cache miss; original author inspection remains separately attributed.' }],
  scripts: ['scripts/verify-abstract-algebra-independent.mjs','scripts/verify-abstract-algebra-independent.py','scripts/review-abstract-algebra-independent.cjs','scripts/review-abstract-algebra-graph-paint.cjs'].map(stamp),
  harnessCorrections: ['The initial exact sensor assertion compared SymPy Rational with Float structural values; converting serialized values to exact Rational corrected the oracle without changing production.', 'Keyboard End moved the long page and yielded stale/blank captures; the focused graph recapture asserts element position and local ArrowRight scrolling.'],
  limits: ['The complete finite projection operators and changed exact cases complement source-read proofs; they do not establish arbitrary-group implementation behavior or predictive accuracy.', 'Browser observations are local Edge at1440/390/320, not a screen-reader or learner study.', 'Original author evidence remains immutable; integrated reviewed status is not promoted by this record.'],
};
fs.writeFileSync(independentPath, JSON.stringify(record, null, 2) + '\n');
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json', ledger = read(ledgerPath);
const topic = ledger.topics.find(topic => topic.id === author.topicId);
Object.assign(topic, { status: 'author-verified', verificationRecord: 'docs/teaching/ABSTRACT-ALGEBRA-VERIFICATION.md', authorReviewRecord: authorPath, independentEvidence: independentPath, independentReviewRecord: 'docs/teaching/ABSTRACT-ALGEBRA-INDEPENDENT-REVIEW.md', reviewedFiles: Object.fromEntries(payload.sources.map(source => [source.path, source.sha256])), reviewedSourceSha256: payload.sources.find(source => source.path.includes('/topics/')).sha256, reviewStage: 'Independent review closed with one documented static-diagram amendment; final production integration remains pending.' });
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2)+'\n');
console.log(JSON.stringify({ reviewedAt: record.reviewedAt, finalSources: payload.sources.length, amendments: 1, actuallyOpenedImages: opened.length }, null, 2));
