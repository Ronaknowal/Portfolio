const fs=require('node:fs');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const hashes=require('./dp-state-families-source-hashes.cjs');
const read=path=>JSON.parse(fs.readFileSync(path,'utf8'));
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const packetPath='docs/teaching/evidence/dp-state-families-author-review.json';
assert(!fs.existsSync(packetPath),'Preserve an existing author packet before any amendment.');
const sourceHashes=hashes();
const native=read('docs/teaching/evidence/dp-state-families-native-verification.json');
const preservation=read('docs/teaching/evidence/dp-state-families-preservation.json');
const browser=read('docs/teaching/evidence/dp-state-families-browser.json');
const reading=read('docs/teaching/evidence/dp-state-families-reading.json');
const amendment=read('docs/teaching/evidence/dp-state-families-select-label-amendment.json');
const focused=read('docs/teaching/evidence/dp-state-families-select-label-browser.json');
assert.deepEqual(focused.sourceHashes,sourceHashes);
for(const source of sourceHashes){
 const earlier=browser.sourceHashes.find(item=>item.path===source.path);
 const readingSource=reading.sourceHashes.find(item=>item.path===source.path);
 assert.equal(earlier.sha256,readingSource.sha256);
 if(source.path===amendment.path){
  assert.equal(earlier.sha256,amendment.beforeSha256);
  assert.equal(source.sha256,amendment.afterSha256);
  const before=fs.readFileSync(amendment.archive,'utf8');
  const current=fs.readFileSync(source.path,'utf8');
  assert.equal(current.replace(amendment.after,amendment.before),before);
  assert.equal(hash(before),amendment.beforeSha256);
 }else assert.equal(source.sha256,earlier.sha256);
}
for(const [path,sha256] of Object.entries(native.sourceHashes))assert.equal(sourceHashes.find(source=>source.path===path).sha256,sha256);
assert.deepEqual(browser.errors,[]);assert.deepEqual(browser.failedRequests,[]);
for(const result of reading.results){assert.deepEqual(result.overflow,[]);assert.deepEqual(result.errors,[]);assert.deepEqual(result.failedRequests,[]);}
for(const result of focused.results){assert.deepEqual(result.errors,[]);assert.deepEqual(result.failedRequests,[]);}
const images=[
 'scratch/dp-state-families-reading/inline-0-320.png',
 'scratch/dp-state-families-reading/inline-1-320.png',
 'scratch/dp-state-families-reading/inline-2-320.png',
 'scratch/dp-state-families-reading/digit-changed-solution-320.png',
 'scratch/dp-state-families-reading/original-compression-formula-320.png',
 'scratch/dp-state-families-reading/tree-nonempty-solution-390.png',
 'scratch/dp-state-families-reading/balloon-signed-solution-1440.png',
 'scratch/dp-state-families-browser/tree-parent-selected-reading-390.png',
 'scratch/dp-state-families-browser/digit-tight-reading-390.png',
 'scratch/dp-state-families-browser/digit-loose-reading-320.png',
 'scratch/dp-state-families-browser/family-practice-reading-320.png',
 'scratch/dp-state-families-select-label/chosen-split-1440.png',
 'scratch/dp-state-families-select-label/chosen-split-390.png',
 'scratch/dp-state-families-select-label/chosen-split-320.png',
].map(path=>({path,sha256:hash(fs.readFileSync(path)),openedByAuthor:true}));
const record={
 topicId:'dynamic-programming-states-transitions-optimization',
 scope:'Bounded interval/tree/digit extension; original lesson preserved with one original prose spacing repair. Author verification only; parent owns independent review and integration.',
 frozenAt:new Date().toISOString(),sourceHashes,
 original:'docs/teaching/evidence/dp-state-families-original.json',
 preservation:'docs/teaching/evidence/dp-state-families-preservation.json',
 native:'docs/teaching/evidence/dp-state-families-native-verification.json',
 oldAlgorithmRegression:'docs/teaching/evidence/dp-state-families-original-regression.json',
 fullBrowser:'docs/teaching/evidence/dp-state-families-browser.json',
 expandedReading:'docs/teaching/evidence/dp-state-families-reading.json',
 finalLabelAmendment:'docs/teaching/evidence/dp-state-families-select-label-amendment.json',
 finalLabelBrowser:'docs/teaching/evidence/dp-state-families-select-label-browser.json',
 nativeChecks:native.checks,
 browserWidths:browser.results.map(result=>({width:result.width,originalStates:result.rewardStates+result.gridStates+result.sequenceStates+result.capacityStates+result.subsetStates,...result.families.counts,programs:result.examples,practice:result.practice,anchors:result.anchors,overflow:result.overflow})),
 disclosures:reading.results.map(({width,disclosures})=>({width,disclosures})),
 finalLabelStates:focused.results.map(result=>({width:result.width,states:result.values.length})),
 actuallyOpenedImages:images,
 limitations:['Finite independent enumeration and changed-input checks complement the written proofs; they are not a universal formal proof.','Full 639-state browser run and 50-disclosure reading check precede only the explicitly archived option-label wording amendment; final targeted candidate checks bind the final source.','No LeetCode submissions/editorial access or complete audiovisual playback claimed.','No production build or shared metadata mutation by this author.','Old source/evidence is preserved with its original date; this packet is not a retroactive rewrite of the previous review.'],
};
fs.writeFileSync(packetPath,JSON.stringify(record,null,2)+'\n');
console.log(JSON.stringify({frozenAt:record.frozenAt,sourceHashes,images:images.length,browser:record.browserWidths},null,2));
