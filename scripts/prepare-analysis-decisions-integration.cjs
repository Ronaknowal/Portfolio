const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const ledger = read(ledgerPath);
const definitions = [
  ['ordinary-differential-equations-linear-systems', 'ordinary-differential-equations', 'ORDINARY-DIFFERENTIAL-EQUATIONS'],
  ['complex-numbers-fourier-laplace-transforms', 'complex-transforms', 'COMPLEX-FOURIER-LAPLACE'],
  ['conditioning-stability-numerical-analysis', 'conditioning-stability', 'CONDITIONING-STABILITY'],
  ['decision-theory-risk-cost-sensitive-decisions', 'decision-theory', 'DECISION-THEORY'],
  ['real-analysis-sequences-modes-of-convergence', 'real-analysis', 'REAL-ANALYSIS']
];
assert.equal(ledger.topics.filter(row => row.status === 'implementation-reviewed').length, 66);
for (const [id, stem, title] of definitions) {
  const independentPath = `docs/teaching/evidence/${stem}-independent-review.json`;
  const independent = read(independentPath);
  const sources = independent.sources || independent.productionSources;
  assert.match(independent.status, /closed|independently reviewed/i);
  for (const source of sources) assert.equal(hash(source.path), source.sha256);
  Object.assign(ledger.topics.find(row => row.id === id), {
    status: 'author-verified', source: `./topics/${id}.jsx`,
    verificationRecord: `docs/teaching/${title}-VERIFICATION.md`,
    authorReviewRecord: `docs/teaching/evidence/${stem}-author-review.json`,
    independentEvidence: independentPath,
    independentReviewRecord: `docs/teaching/${title}-INDEPENDENT-REVIEW.md`,
    reviewedSourceSha256: hash(`src/learn/data/topics/${id}.jsx`),
    reviewedFiles: Object.fromEntries(sources.map(source => [source.path, source.sha256])),
    reviewStage: 'Independent review closed on final sources; original author evidence and documented amendments preserved; production integration pending.'
  });
}
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
const loadingPath = 'scripts/verify-learning-load-boundaries.cjs';
let loading = fs.readFileSync(loadingPath, 'utf8');
assert(!loading.includes('const mathematicsLessons = ["ordinary-differential-equations'));
loading = loading.replace('const mathematicsLessons = [', `const mathematicsLessons = [${definitions.map(([id]) => JSON.stringify(id)).join(', ')}, `);
fs.writeFileSync(loadingPath, loading);
const routePath = 'scripts/review-module-order.cjs';
let route = fs.readFileSync(routePath, 'utf8');
assert(route.includes('const reviewedMathematicsStart=45, reviewedMathematicsEnd=49;'));
route = route.replace('const reviewedMathematicsStart=45, reviewedMathematicsEnd=49;', 'const reviewedMathematicsStart=49, reviewedMathematicsEnd=54;')
  .replace('assert.equal(newMathematicsIds.length,4);', 'assert.equal(newMathematicsIds.length,5);')
  .replace(".innerText(),'4 completed');", ".innerText(),'5 completed');");
fs.writeFileSync(routePath, route);
console.log('Five closed independent reviews staged; frozen66 integration preserved.');
