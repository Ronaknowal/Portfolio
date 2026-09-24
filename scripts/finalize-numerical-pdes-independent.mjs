import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { numericalPdeExamples as currentExamples } from '../src/learn/data/numerical-pde-examples.js';

const root = 'scratch/numerical-pdes-independent-review';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const reference = path => ({ path, sha256: sha(path) });
const start = read(`${root}/start.json`);
const authorPath = 'docs/teaching/evidence/numerical-pdes-author-review.json';
const author = read(authorPath);
const native = read(`${root}/native-results.json`);
const browser = read(`${root}/browser/results.json`);
assert(native.passed && browser.passed);
for (const entry of native.sourceHashes) {
  assert.equal(sha(entry.path), entry.sha256);
  assert.equal(author.sourceHashes[entry.path], entry.sha256, 'Await amended author packet before closure.');
  assert.equal(browser.sourceHashes.find(source => source.path === entry.path).sha256, entry.sha256);
}
for (const entry of start.sourceHashes) assert.equal(sha(entry.archive), entry.sha256);
const originalAuthorPath = 'docs/teaching/archive/numerical-pdes-independent-start/author-review.json';
assert.equal(sha(originalAuthorPath), start.authorSha256);
const originalModel = start.sourceHashes.find(entry => entry.path.endsWith('/numerical-pde-models.js'));
const reversedModel = fs.readFileSync(originalModel.path, 'utf8')
  .replace('j === 0 ? 0 : j === intervals ? length : j * length / intervals', 'j * length / intervals')
  .replace('j === 0 ? 0 : j === 256 ? length : j * length / 256', 'j * length / 256');
assert.equal(reversedModel, fs.readFileSync(originalModel.archive, 'utf8'));
const originalExamples = start.sourceHashes.find(entry => entry.path.endsWith('/numerical-pde-examples.js'));
const oldExamples = (await import('data:text/javascript;base64,' + fs.readFileSync(originalExamples.archive).toString('base64'))).numericalPdeExamples;
assert.deepEqual(Object.keys(currentExamples), Object.keys(oldExamples));
for (const [key, example] of Object.entries(currentExamples)) {
  if (key === 'poisson') {
    const reversedCode = example.code.replace('    # The prescribed domain endpoints are exact inputs, not accumulated products.\n    nodes[0], nodes[-1] = 0.0, length\n', '');
    assert.deepEqual({ ...example, code: reversedCode }, oldExamples[key]);
  } else assert.deepEqual(example, oldExamples[key]);
  assert.equal(example.expected, oldExamples[key].expected);
}
const authorAmendmentPath = 'docs/teaching/evidence/numerical-pdes-endpoint-before.json';
const authorAmendment = read(authorAmendmentPath);
for (const file of authorAmendment.files) assert.equal(sha(file.archive), file.sha256);
const opened = [
  ['restriction-reading-320.png', 'Ordinary restriction/reconstruction flow; same dots do not imply the same between-node field.'],
  ['boundary-row-390.png', 'Right endpoint row contributions, nonzero boundary and numerical field.'],
  ['field-budget-320.png', 'Three separate error terms, summed certificate and arithmetic qualification.'],
  ['diffusion-overshoot-1440.png', 'Actual unstable amplitude stays inside the rescaled plot at a common time.'],
  ['material-profile-390.png', 'Reversed material contrast, continuous temperature and matching flux interpretation.'],
  ['left-translation-320.png', 'Periodic cell values and exactly translated reference at the same time.'],
  ['nonuniform-assembly-1440.png', 'Highlighted overlapping matrix entries and source load; distinct missed kink.'],
  ['point-field-390.png', 'Source between nodes; nodal agreement versus integrated field and energy errors.'],
  ['corner-indices-320.png', 'Upper-right interior index and two prescribed boundary neighbors without row wrap.'],
  ['triangle-gradients-320.png', 'Equal-scale stretched triangle and the inverse-length gradient table.'],
  ['coarse-norms-390.png', 'Energy decreases while the immediately preceding L2 norm increases.'],
  ['stationary-face-390.png', 'Shock state versus unique flux; keyboard focus and explicit nonunique trace.'],
  ['energy-proof-320.png', 'Ordinary Galerkin/Pythagorean proof and visible exact-integration/solve assumptions.'],
  ['reconstruction-proof-390.png', 'Local interpolation proof, Rolle argument and added between-node term.'],
  ['changed-report-question-320.png', 'Changed physical domain/source/boundary task with separate hint and solution.'],
  ['changed-report-output-1440.png', 'Complete finite-budget failure/success output and scoped readiness/resources.']
].map(([name, observation]) => ({ ...reference(`${root}/browser/${name}`), observation, actuallyOpened: true }));
const result = {
  reviewedAt: new Date().toISOString(),
  topicId: 'numerical-pdes-grids-finite-elements-stability',
  status: 'bounded independent review passed; production integration and user acceptance remain separate',
  reviewer: '/root/scientific_visual_improvements',
  authorFreeze: author.authorFrozenAt,
  authorPacket: reference(authorPath),
  sourceHashes: native.sourceHashes,
  preservedOriginalAuthorPacket: reference(originalAuthorPath),
  authorAmendmentArchive: reference(authorAmendmentPath),
  sourceConservation: { modelReversedToExactOriginal: true, unchangedExampleRecords: 15, endpointOnlyNativeProgramChange: 'poisson', conservedStdoutPrograms: 16, unchangedOtherProductionFiles: 4 },
  originalReviewStart: { ...reference(`${root}/start.json`), result: start },
  initialFindingEvidence: { ...reference(`${root}/initial-findings.json`), result: read(`${root}/initial-findings.json`) },
  findings: [{
    severity: 'material accepted-input correctness defect',
    problem: 'An allowed odd grid and decimal domain length rounded its generated endpoint below the requested curve endpoint, causing the exported Poisson model to throw.',
    exactReproducer: { intervals: 3, length: .7, profile: 'linear', scale: 1.7, left: -2.3, right: 4.1, method: 'jacobi', iterations: 9, tolerance: .03 },
    authorRepair: 'Pin both mesh and plot endpoints to the actual domain values. The displayed native Poisson helper pins its returned mesh endpoints as well. No interpolation domain tolerance was broadened.',
    resolution: 'Closed by changed odd-grid Green/continuous-extremum cases, explicit JS mesh/curve endpoint equality, 20 changed native endpoint/cubic cases and actual final source-matched browser programs.'
  }],
  productionEditsByReviewer: [],
  sourceRead: ['Entire 15-section lesson and 15 changed practice solutions', 'All 16 actual complete program strings, questions and stdout', 'Entire pure model, component and CSS source', 'Individual brief, complete assessed design, author verification and design-only independent assessment', 'Exact inventory and three incoming destination-note dispositions'],
  mathematicalReview: ['Negative-Laplacian sign, Taylor remainder and defect direction', 'Boundary elimination, discrete maximum principle and normalization-aware inverse bound', 'Convex reconstruction, local remainder and exact-versus-sampled error', 'Explicit monotonicity versus finite-grid Euclidean spectral condition', 'Method-specific implicit factors, energy identity and fixed-space time comparison', 'Outward Neumann/Robin signs, compatibility and series resistance', 'Periodic upwind contraction, CFL necessity and scalar entropy flux scope', 'Exact load functionals, nonuniform assembly and source-alignment kink', 'Galerkin energy orthogonality, 1D constant-coefficient nodal exactness and integrated errors', 'Unequal grid spacing, row-major boundaries and affine triangle gradients', 'Weighted Jacobi and norm-specific coarse-space correction'],
  native: { record: reference(`${root}/native-results.json`), result: native, scripts: [reference('scripts/verify-numerical-pdes-independent.mjs'), reference('scripts/verify-numerical-pdes-independent.py')] },
  browser: { record: reference(`${root}/browser/results.json`), result: browser, script: reference('scripts/review-numerical-pdes-independent.cjs') },
  openedImages: opened,
  primarySourceChecks: [
    { url: 'https://fncbook.com/upwind/', scope: 'Domain-of-dependence definitions, CFL necessity rather than sufficiency, upwind direction and inflow boundary.' },
    { url: 'https://fncbook.com/python/absstab-diffusion/', scope: 'Scalar absolute-stability and spatial-spectrum reference opened; lesson-specific Dirichlet and implicit factors reviewed by local derivation and complementary full-sine tests.' },
    { url: 'https://jschoeberl.github.io/iFEM/FEM/erroranalysis.html', scope: 'Affine chain-rule transformation, shape regularity, H2 interpolation/coercive approximation and regularity-dependent L2 extension. No transfer of general theorem assumptions into the weaker local lesson proof.' }
  ],
  limitations: ['Finite complementary cases support the source read; they do not prove every accepted floating input or every PDE theorem.', 'Whole-field extrema checks use 80-digit arithmetic and monotone-root bisection; the lesson supplies the analytical guarantee.', 'Only the 16 listed screenshots were actually opened, out of 60 reviewer captures.', 'The author owns its larger independent-from-implementation oracle and comprehensive browser suites; they are separately attributed.', 'No full video playback, physical-device or complete screen-reader audit, or observed beginner study was performed.', 'Production build, loading/routes, final shared ledger and user acceptance belong to separate integration.']
};
const destination = 'docs/teaching/evidence/numerical-pdes-independent-review.json';
assert(!fs.existsSync(destination), 'Preserve any previous independent closure instead of overwriting.');
fs.writeFileSync(destination, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ reviewedAt: result.reviewedAt, authorFreeze: result.authorFreeze, sourceCount: result.sourceHashes.length, openedImages: opened.length }));
