import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const hash=file=>digest(fs.readFileSync(file));
const baseline=read('docs/teaching/evidence/deep-learning-sequence-baseline.json');
const ledger=read('docs/teaching/lesson-delivery-progress.json');
const report={passed:false,checkedAt:new Date().toISOString(),topics:[],unchangedOtherRows:0,receipts:[]};
const destination='docs/teaching/evidence/deep-learning-sequence-reconciliation.json';
const save=()=>fs.writeFileSync(destination,JSON.stringify(report,null,2)+'\n');
save();
try{
 assert.equal(hash('src/learn/data/lesson-manifest.json'),baseline.publication,'Publication mappings changed');
 assert.equal(Object.keys(ledger.topics).length,Object.keys(baseline.rows).length);
 for(const[id,before]of Object.entries(baseline.rows)){
  const row=ledger.topics[id];
  if(!baseline.scope.includes(id)){assert.equal(digest(JSON.stringify(row)),before,`Unrelated ledger row ${id}`);report.unchangedOtherRows++;continue;}
  const original=baseline.prepared[id];
  assert.equal(row.revision,original.revision);assert.equal(row.deliveryMode,original.deliveryMode);
  assert.deepEqual(row.previousRevisions,original.previousRevisions);
  assert.deepEqual(row.content,original.content,'Prepared content checkpoint changed');
  assert.equal(row.content.status,'complete');assert.equal(row.implementation.status,'complete');
  for(const[file,expected]of Object.entries({...row.content.files,...row.implementation.reviewedFiles}))assert.equal(hash(file),expected,`Stale ${id}: ${file}`);
  report.topics.push({id,revision:row.revision,content:'complete',implementation:'complete',reviewedFiles:Object.keys(row.implementation.reviewedFiles).length});
 }
 const receipts=[
  'depthwise-convolution-author.json','depthwise-convolution-native.json','depthwise-independent.json','depthwise-independent-native.json',
  'convnext-models.json','convnext-native.json','convnext-independent.json','convnext-independent-native.json',
  'capsule-author.json','capsule-native.json','capsule-independent.json','capsule-independent-native.json',
  'recurrent-author.json','recurrent-native.json','recurrent-independent-review.json','recurrent-independent-native.json','recurrent-independent-models.json',
  'seq2seq-author.json','seq2seq-native.json','seq2seq-independent.json','seq2seq-independent-native.json',
  'modern-convolution-browser/report.json','capsule-browser/report.json','recurrent-browser/report.json','seq2seq-browser/report.json','deep-learning-sequence-production-integration.json',
 ];
 for(const name of receipts){
  const file='docs/teaching/evidence/'+name,r=read(file);
  assert.ok(r.passed===true||r.accepted===true,`Failed receipt ${name}`);
  for(const field of ['reviewedFiles','sourceHashes','hashes','sources'])for(const[path,expected]of Object.entries(r[field]||{}))assert.equal(hash(path),expected,`${name}: stale ${path}`);
  report.receipts.push({path:file,sha256:hash(file)});
 }
 const integration=read('docs/teaching/evidence/deep-learning-sequence-production-integration.json');
 assert.equal(integration.routes.length,5);assert.equal(integration.recovery.length,2);
 assert.equal(integration.manifestHash,hash('dist/.vite/manifest.json'));
 report.contentComplete=Object.values(ledger.topics).filter(row=>row.content.status==='complete').length;
 report.implementationComplete=Object.values(ledger.topics).filter(row=>row.implementation.status==='complete').length;
 report.preparedRemaining=report.contentComplete-report.implementationComplete;
 assert.equal(report.contentComplete,177);assert.equal(report.implementationComplete,150);assert.equal(report.preparedRemaining,27);
 report.passed=true;
}catch(error){report.failure=error.stack;process.exitCode=1;}
save();console.log(JSON.stringify({passed:report.passed,topics:report.topics.length,unchangedOtherRows:report.unchangedOtherRows,receipts:report.receipts.length,contentComplete:report.contentComplete,implementationComplete:report.implementationComplete,preparedRemaining:report.preparedRemaining,failure:report.failure}));
