import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
const id='ring-attention-sequence-parallelism';
const base=`docs/teaching/deep-learning-completion/${id}/`;
const draft=`docs/teaching/drafts/${id}/`;
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const receipt=JSON.parse(fs.readFileSync(base+'implementation-checks.json','utf8'));
for(const p of receipt.sourceFiles)assert.equal(hash(p),receipt.sourceHashes[p],p);
const deployed=receipt.sourceFiles.filter(p=>p.startsWith(`public/learn-assets/${id}/`));
assert.equal(deployed.length,9);
for(const p of deployed)assert.deepEqual(fs.readFileSync(p),fs.readFileSync(draft+p.split('/').at(-1)),p);
const rows=fs.readFileSync(draft+'movement_libras.data','utf8').trim().split(/\r?\n/).map(s=>s.split(',').map(Number));
const examples=JSON.parse(fs.readFileSync('src/learn/data/ring-attention-examples.json','utf8'));
for(const x of examples.real){
  // Published source IDs count from one, as in the canonical study.
  assert.equal(x.label,rows[x.source-1][90]);
  assert.deepEqual(x.points.flat(),rows[x.source-1].slice(0,90));
}
const body=fs.readFileSync(receipt.productionPath,'utf8');
assert.equal((body.match(/<H2>/g)||[]).length,15);
assert.equal((body.match(/<summary>Hint<\/summary>/g)||[]).length,8);
assert.equal((body.match(/<summary>Solution(?: and acceptance criteria)?<\/summary>/g)||[]).length,8);
assert.equal((body.match(/<RingProgram file=/g)||[]).length,4);
for(const p of ['ring-attention-reference.py','distributed_ring.py','attention-partition-study.py','systems-calculations.py'])assert.ok(body.includes(`file="${p}"`));
const result={passed:true,authorPaths:receipt.sourceFiles.length,deployedIdenticalFiles:deployed.length,realTrajectories:examples.real.length,checks:[
 {name:'Every final author source identity matches current bytes, including shared render dependencies and deployed assets',passed:true},
 {name:'All nine public downloads equal their canonical packet sources byte for byte',passed:true},
 {name:'Both compact real examples retain all 45 raw coordinate pairs and source labels exactly',passed:true},
 {name:'Generated article retains all 15 sections, eight independent hint/solution pairs and four complete program readers',passed:true}
],limits:['Source and data integrity; painted layout and live browser behavior are verified separately by root.']};
fs.writeFileSync(base+'independent-source-checks.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
