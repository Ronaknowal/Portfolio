const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const read = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const file = path => ({ path, sha256: hash(path), bytes: fs.statSync(path).size });
const directory = 'scratch/ordinary-differential-equations-independent';
const authorPath = 'docs/teaching/evidence/ordinary-differential-equations-author-review.json';
const author = read(authorPath);
const baseline = read(`${directory}/author-baseline.json`);
const native = read(`${directory}/results.json`);
const browser = read(`${directory}/browser-results.json`);
assert(native.passed && browser.passed);
const sources = baseline.sources.map(source => {
  assert.equal(hash(source.archive), source.sha256);
  assert.equal(author.sources.find(row => row.path === source.path).sha256, source.sha256);
  assert.equal(native.sources.find(row => row.path === source.path).sha256, hash(source.path));
  assert.equal(browser.sources.find(row => row.path === source.path).sha256, hash(source.path));
  return { ...file(source.path), authorSha256: source.sha256, amended: hash(source.path) !== source.sha256 };
});
assert.equal(sources.filter(source => source.amended).length, 2);
const destination = 'docs/teaching/evidence/ordinary-differential-equations-independent-review.json';
assert(!fs.existsSync(destination), 'Preserve frozen evidence.');
const packet = {
  reviewedAt: new Date().toISOString(),
  topicId: 'ordinary-differential-equations-linear-systems',
  reviewer: '/root',
  status: 'independent-review-closed; two final amendments verified; production integration and user acceptance separate',
  sources,
  authorRecord: file(authorPath),
  originalAuthorSources: baseline.sources,
  sourceReview: [
    'Full fourteen-section body, fourteen independent practice groups and thirteen actual Python programs.',
    'All models, eight investigations, four inline figures, scoped CSS, individual blueprint, assessed design and incoming ownership note.',
    'Thermal units; local existence and Picard uniqueness; maximal domains; scalar separation/integrating factors; exact and Bernoulli equations; matrix-series and fundamental-column arguments; Jordan and state-specific growth.',
    'Initial versus forced response, chronological products, resonance conditions, actual numerical stages, common-domain Euler error proof, stiffness/events, endpoint compatibility and ordinary-point series.'
  ],
  findingsResolved: [
    'Cooling now uses cancellation-resistant weighted terms and exact initial/equilibrium shortcuts; a positive rate-time product that underflows before a potentially amplified gain is rejected explicitly.',
    'Dense vector and matrix inputs require their own indexed values rather than accepting inherited entries.',
    'A zero-length stationary trajectory has a visible labelled point; the nilpotent first basis column is no longer visually absent.'
  ],
  native: { file: file(`${directory}/results.json`), result: native },
  browser: { file: file(`${directory}/browser-results.json`), result: browser },
  visuallyOpenedFinal: ['changed-summed-columns-320', 'changed-midpoint-probes-1440', 'inline-0-320'].map(name => file(`${directory}/${name}.png`)),
  earlierVisualReview: 'Thirteen first-run reviewer images were opened before the stationary-point amendment. Three final-run images were reopened, including the affected narrow matrix-column plot. The 39 saved final images are not all claimed as opened.',
  sourceRecheck: [
    { date: '2026-09-11', url: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html', scope: 'API tolerances, output-time sampling, event limitations and status; current page 1.18.0 versus executed installed 1.18.1.' },
    { date: '2026-09-11', url: 'https://www.jirka.org/diffyqs/html/forcedo_section.html', scope: 'Section 2.6.2 damped forcing, amplitude and resonance peak derivation. No unqualified claim that increasing damping always accelerates transient decay is imported.' }
  ],
  limitations: [
    'Finite numerical comparisons supplement separately reviewed arguments; they are not a solver certification or learner study.',
    'Actual browser checks cover the bounded controls at 1440, 390 and 320 pixels; source validation and high-precision checks extend selected numerical boundaries.',
    'The original author packet remains unchanged and its timestamps are not reassigned to amended sources.',
    'Production integration and user acceptance remain separate; no site deployment occurred.'
  ]
};
fs.writeFileSync(destination, JSON.stringify(packet, null, 2) + '\n');
console.log(JSON.stringify({ destination, reviewedAt: packet.reviewedAt, sourceCount: sources.length, amended: 2, finalImagesOpened: 3 }));
