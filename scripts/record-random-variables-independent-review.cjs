const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const file = path => ({path,sha256:hash(path),bytes:fs.statSync(path).size});
const read = path => JSON.parse(fs.readFileSync(path,'utf8'));
const directory = 'scratch/random-variables-independent';
const authorPath = 'docs/teaching/evidence/random-variables-author-review.json';
const author = read(authorPath);
const baseline = read(`${directory}/author-baseline.json`);
const native = read(`${directory}/results.json`);
const browser = read(`${directory}/browser-results.json`);
assert(native.passed && browser.passed);
const sources = baseline.sources.map(source=>{
  assert.equal(hash(source.archive),source.sha256,'Preserve original author source');
  assert.equal(author.sources.find(row=>row.path===source.path).sha256,source.sha256);
  assert.equal(browser.sources.find(row=>row.path===source.path).sha256,hash(source.path),'Browser must cover final source');
  return {...file(source.path),authorSha256:source.sha256,amended:hash(source.path)!==source.sha256};
});
assert.equal(sources.filter(row=>row.amended).length,3);
const opened = ['changed-outcome-law-390','changed-nonlinear-witness-320','changed-squared-loss-1440',
  'changed-noise-combination-390','changed-group-mixture-1440','changed-two-preimages-320',
  'changed-independent-versus-copies-390','changed-dice-practice-320','python-setup-320'];
const destination = 'docs/teaching/evidence/random-variables-independent-review.json';
assert(!fs.existsSync(destination),'Preserve a frozen independent packet.');
const packet = {
  reviewedAt:new Date().toISOString(),topicId:'random-variables-expectation-covariance',reviewer:'/root',
  status:'independent-review-closed; amended final sources verified; production integration and user acceptance separate',
  authorRecord:file(authorPath),originalAuthorSources:baseline.sources,
  sources,
  sourceReview:['Complete twelve-section body, all eleven changed practices and explained solutions',
    'All fourteen actual Python programs, pure model, seven investigations, scoped CSS, blueprint and design',
    'Probability-preserving grouping, finite moment hypotheses, conditional and covariance proofs, transformation branches, population/sample distinction and numerical arithmetic contracts'],
  findingsResolved:[
    'Sparse and inherited missing entries now reject before arithmetic; 109 complementary invalid inputs and exact conservation of 8488 admitted states checked.',
    'Static hasIntegratedGuide metadata removes the legacy duplicate guide; one integrated introduction verified in the actual page.',
    'Save/run instructions now precede the first optional Python program, with standard-library and execution boundaries.',
    'Probability plot label margin increased after actual-font bounds exposed a tight lower edge. Rounded accessible readouts are labelled as rounded.',
    'Author verification wording now describes the actual product of separately computed standard deviations; its original source freeze remains preserved.'
  ],
  native:{file:file(`${directory}/results.json`),result:native},
  browser:{file:file(`${directory}/browser-results.json`),result:browser},
  visuallyOpened:opened.map(name=>file(`${directory}/${name}.png`)),
  sourceRecheck:{date:'2026-09-11',url:'https://www2.stat.duke.edu/courses/Fall19/sta721/lectures/NormalTheory/multnorm.pdf',
    scope:'Selected parsed block-Gaussian construction and independence conclusion, with singular definition context; not the entire notes. PDF screenshot fetch failed and is not claimed as visually reviewed.'},
  limitations:['Finite complementary tests are not proofs of arbitrary probability laws; derivations and their hypotheses were separately read.',
    'The original author packet retains its earlier model/browser evidence; this packet binds only the final amendments and explicitly described complementary checks.',
    'Nine of 27 reviewer captures were actually opened. The earlier failing pre-margin image was inspected but is not final evidence.',
    'No real measurement data, full-video viewing, production deployment, observed beginner study or user acceptance is claimed.']
};
fs.writeFileSync(destination,JSON.stringify(packet,null,2)+'\n');
console.log(JSON.stringify({destination,reviewedAt:packet.reviewedAt,sources:sources.length,amended:sources.filter(row=>row.amended).length,opened:opened.length}));
