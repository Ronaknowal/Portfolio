import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const read = file => JSON.parse(fs.readFileSync(file,'utf8'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const baseline=read('docs/teaching/evidence/deep-learning-convolution-baseline.json');
const ledger=read('docs/teaching/lesson-delivery-progress.json');
const output='docs/teaching/evidence/deep-learning-cnn-reconciliation.json';
const report={passed:false, checkedAt:new Date().toISOString(), topics:[], unchangedOtherRows:0, receipts:[]};
const save=()=>fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
save();
try {
 assert.deepEqual(read('src/learn/data/lesson-manifest.json'),baseline.manifest);
 assert.equal(Object.keys(ledger.topics).length,Object.keys(baseline.ledgerRows).length);
 for(const [id,before] of Object.entries(baseline.ledgerRows)) {
  const row=ledger.topics[id];
  if(!baseline.ids.includes(id)) {assert.equal(digest(JSON.stringify(row)),before,`Unrelated entry ${id}`);report.unchangedOtherRows++;continue;}
  const original=baseline.selectedEntries[id];
  assert.equal(row.revision,original.revision); assert.equal(row.deliveryMode,original.deliveryMode);
  assert.deepEqual(row.previousRevisions,original.previousRevisions);
  assert.equal(row.content.status,'complete'); assert.equal(row.implementation.status,'complete');
  for(const [file,expected] of Object.entries({...row.content.files,...row.implementation.reviewedFiles})) assert.equal(hash(file),expected,`Stale ${id}: ${file}`);
  report.topics.push({id,revision:row.revision,content:'complete',implementation:'complete',reviewedFiles:Object.keys(row.implementation.reviewedFiles).length});
 }
 const receipts=[
  'weight-initialization-author.json','weight-initialization-native.json','initialization-independent-native.json','initialization-independent-review.json','weight-initialization-browser/report.json',
  'residual-connections-author.json','residual-connections-independent-native.json','residual-connections-independent-review.json','residual-connections/browser.json',
  'dropout-implementation/author.json','dropout-implementation/independent-native.json','dropout-implementation/independent-model.json','dropout-implementation/independent-review.json','dropout-implementation/browser.json',
  'convolution-author-checks.json','convolution-independent-native.json','convolution-independent-model.json','convolution-independent-review.json','convolution-browser/report.json',
  'landmark-architecture-author.json','landmark-architecture-native.json','landmark-architectures-independent-native.json','landmark-architectures-independent-review.json','landmark-architecture-browser/report.json',
  'deep-learning-cnn-production-integration.json',
 ];
 for(const name of receipts) {
  const path='docs/teaching/evidence/'+name, receipt=read(path);
  const dropoutAuthor=name==='dropout-implementation/author.json'&&receipt.status==='author-verified-awaiting-independent-review-and-browser';
  if(dropoutAuthor) {assert.equal(receipt.training.completeFits,27);assert.equal(receipt.training.maxAbsoluteDifference,0);assert.equal(receipt.library.executed,true);}
  assert.ok(receipt.passed===true||receipt.status==='passed'||receipt.status==='passed-source-and-native-review-browser-pending'||dropoutAuthor,`Failed receipt ${name}`);
  for(const field of ['reviewedFiles','sourceHashes','hashes','sources']) for(const [file,expected] of Object.entries(receipt[field]||{})) assert.equal(hash(file),expected,`${name}: stale ${file}`);
  report.receipts.push({path,sha256:hash(path)});
 }
 const integration=read('docs/teaching/evidence/deep-learning-cnn-production-integration.json');
 assert.equal(integration.routes.length,5); assert.equal(integration.recovery.length,2);
 assert.equal(integration.manifestHash,hash('dist/.vite/manifest.json'));
 report.contentComplete=Object.values(ledger.topics).filter(row=>row.content.status==='complete').length;
 report.implementationComplete=Object.values(ledger.topics).filter(row=>row.implementation.status==='complete').length;
 report.preparedRemaining=report.contentComplete-report.implementationComplete;
 assert.equal(report.contentComplete,177); assert.equal(report.implementationComplete,145); assert.equal(report.preparedRemaining,32);
 report.passed=true;
} catch(error) {report.failure=error.stack;process.exitCode=1;}
save(); console.log(JSON.stringify(report));
