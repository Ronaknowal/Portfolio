const fs=require('node:fs');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const hash=path=>crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const read=path=>JSON.parse(fs.readFileSync(path,'utf8'));
const ledgerPath='docs/teaching/dsa-math-foundations-progress.json';
const ledger=read(ledgerPath);
const definitions=[
  ['counting-combinatorics-mathematical-induction','counting-combinatorics','COUNTING-COMBINATORICS'],
  ['single-variable-calculus-limits-derivatives-integrals','single-variable-calculus','SINGLE-VARIABLE-CALCULUS'],
  ['random-variables-expectation-covariance','random-variables','RANDOM-VARIABLES'],
  ['sampling-measurement-experimental-design','sampling-measurement','SAMPLING-MEASUREMENT']
];
for(const [id,stem,name] of definitions){
  const path=`docs/teaching/evidence/${stem}-independent-review.json`,review=read(path);
  const sources=review.sources||review.production||review.productionSourcesUnchanged||review.finalProductionSources;
  assert.equal(sources.length,id.startsWith('counting-')?7:6);
  for(const source of sources)assert.equal(hash(source.path),source.sha256,source.path);
  const topic=ledger.topics.find(row=>row.id===id);
  assert.equal(topic.status,'in-progress');
  Object.assign(topic,{status:'author-verified',source:`./topics/${id}.jsx`,verificationRecord:`docs/teaching/${name}-VERIFICATION.md`,
    authorEvidence:`docs/teaching/evidence/${stem}-author-review.json`,independentEvidence:path,
    independentReviewRecord:`docs/teaching/${name}-INDEPENDENT-REVIEW.md`,
    reviewedSourceSha256:hash(`src/learn/data/topics/${id}.jsx`),reviewedFiles:Object.fromEntries(sources.map(source=>[source.path,source.sha256])),
    reviewStage:'Independent review closed on these final sources; production integration pending. Original author packets and any amendments remain separately attributed.'});
}
ledger.topics.find(row=>row.modulePosition===50&&row.moduleId==='math-foundations').source='./topics/ordinary-differential-equations-linear-systems.jsx';
Object.assign(ledger.topics.find(row=>row.modulePosition===51&&row.moduleId==='math-foundations'),{status:'in-progress',designRecord:'docs/teaching/COMPLEX-FOURIER-LAPLACE-LESSON-DESIGN.md'});
fs.writeFileSync(ledgerPath,JSON.stringify(ledger,null,2)+'\n');
const loadPath='scripts/verify-learning-load-boundaries.cjs';
const text=fs.readFileSync(loadPath,'utf8');
assert(!text.includes("const mathematicsLessons = ['counting-combinatorics"));
fs.writeFileSync(loadPath,text.replace('const mathematicsLessons = [',`const mathematicsLessons = [${definitions.map(([id])=>JSON.stringify(id)).join(', ')}, `));
console.log('Four independently reviewed lessons staged for integration; Math51 design indexed separately.');
