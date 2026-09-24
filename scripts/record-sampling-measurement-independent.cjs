const fs=require('node:fs');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const read=path=>JSON.parse(fs.readFileSync(path,'utf8'));
const hash=path=>crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const file=path=>({path,sha256:hash(path),bytes:fs.statSync(path).size});
const directory='scratch/sampling-measurement-independent';
const authorPath='docs/teaching/evidence/sampling-measurement-author-review.json';
const author=read(authorPath),baseline=read(`${directory}/author-baseline.json`);
const native=read(`${directory}/results.json`),browser=read(`${directory}/browser-results.json`);
assert(native.passed&&browser.passed);
const sources=baseline.sources.map(source=>{
  assert.equal(hash(source.archive),source.sha256);
  assert.equal(author.production.find(row=>row.path===source.path).sha256,source.sha256);
  assert.equal(native.sources.find(row=>row.path===source.path).sha256,hash(source.path));
  assert.equal(browser.sources.find(row=>row.path===source.path).sha256,hash(source.path));
  return {...file(source.path),authorSha256:source.sha256,amended:hash(source.path)!==source.sha256};
});
assert.equal(sources.filter(row=>row.amended).length,1);
const opened=['incomplete-frame-three-390','changed-inclusion-contributions-1440','changed-units-and-readings-320',
  'unlike-pairs-observation-390','changed-negative-interaction-320','inline-0-320','inline-1-390',
  'inline-2-320','inline-3-390','inline-4-320','student-weighted-protocol-1440'];
const destination='docs/teaching/evidence/sampling-measurement-independent-review.json';
assert(!fs.existsSync(destination),'Preserve frozen evidence.');
const packet={reviewedAt:new Date().toISOString(),topicId:'sampling-measurement-experimental-design',reviewer:'/root',
  status:'independent-review-closed; final model amendment verified; production integration and user acceptance separate',
  sources,authorRecord:file(authorPath),originalAuthorSources:baseline.sources,
  sourceReview:['Full ten-section body, eleven changed practice answers and twelve actual Python programs',
    'All pure models, five investigations, five inline figures, scoped CSS, individual blueprint, design and incoming note',
    'SRS indicator variance, HT/HH counting conventions, weighted targets, shared-error covariance, complete/blocked randomization, factorial and missing-outcome proofs and limits'],
  findingsResolved:['Dense own-index validation rejects sparse/inherited missing values instead of silently omitting their mass. Missing outcomes require explicit null.',
    'Accepted near-normalized mass sums are normalized; anchored offsets retain zero constant-law variance and the correct variance for adjacent values on a large baseline.',
    'Explicit underflow rejection prevents a nonzero contribution or squared coverage bias from being reported as exact zero.',
    'Inherited preset names reject through the stated contract.'],
  native:{file:file(`${directory}/results.json`),result:native},
  browser:{file:file(`${directory}/browser-results.json`),result:browser},
  visuallyOpened:opened.map(name=>file(`${directory}/${name}.png`)),
  sourceRecheck:{date:'2026-09-11',url:'https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-2-classification-components-uncertainty',scope:'Sections2.1–2.7: TypeA/B evaluation is distinct from random/systematic effects and error is distinct from uncertainty. Statistics Canada repeat fetch timed out; the author has a separate prior source record.'},
  limitations:['Finite exact checks and regression comparisons supplement separately read proofs; they are not a study of learning effectiveness.',
    'One browser harness assertion initially assumed every revealed table must overflow on a narrow viewport. It was corrected to test actual overflow on the longer observed table and wait for the real keyboard scroll.',
    'Eleven of 33 final reviewer images were actually opened. Original author images and timestamps remain separately attributed.',
    'No actual people were assigned, data collected, external messages sent or site deployed. User acceptance remains separate.']};
fs.writeFileSync(destination,JSON.stringify(packet,null,2)+'\n');
console.log(JSON.stringify({destination,reviewedAt:packet.reviewedAt,sourceCount:sources.length,opened:opened.length}));
