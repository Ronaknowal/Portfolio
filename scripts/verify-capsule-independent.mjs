import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { routeCapsules, loadCapsuleWeights, encodeCapsuleImage, diagonalCapsuleEM, emVotes } from '../src/learn/data/capsule-models.js';
const native=JSON.parse(fs.readFileSync('docs/teaching/evidence/capsule-independent-native.json'));
const hash=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
assert.equal(native.passed,true);
for(const[file,expected] of Object.entries(native.sourceHashes))assert.equal(hash(file),expected,file);
let maximumError=0;
const close=(a,b)=>{const x=a.flat(Infinity),y=b.flat(Infinity);assert.equal(x.length,y.length);x.forEach((v,i)=>{const error=Math.abs(v-y[i]);maximumError=Math.max(maximumError,error);assert.ok(error<1e-11);});};
for(const row of native.fixtures){const actual=routeCapsules(row.votes,row.rounds,row.temperature);for(let i=0;i<row.rounds;i++)for(const field of ['coupling','sums','outputs'])close(actual[i][field],row.trace[i][field]);}
const assets='public/learn-assets/capsule-networks/',bytes=fs.readFileSync(assets+'frozen-model.f32'),metadata=JSON.parse(fs.readFileSync(assets+'frozen-model.json'));
const state=loadCapsuleWeights(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),metadata);
for(const row of native.frozen){const result=encodeCapsuleImage(row.image,state);close(result.capsules,row.capsules);close(result.reconstruction,row.reconstruction);assert.equal(result.predicted,row.predicted);}
// The declared denominator floor must be described as a guard, not inferred evidence.
const sparse=diagonalCapsuleEM(emVotes,[0,0,1e-20]);assert.ok(sparse.every(row=>row.mass.every(mass=>mass<1e-12)));
const files=['src/learn/data/topics/capsule-networks.jsx','src/learn/data/capsule-models.js','src/learn/components/lesson-labs/CapsuleLabs.jsx','src/learn/components/lesson-labs/capsules.css','scripts/verify-capsule-independent.mjs'];
fs.writeFileSync('docs/teaching/evidence/capsule-independent.json',JSON.stringify({passed:true,reviewer:'root; separate from author initialization_implementation',groups:['36changed-shape/temperature/round cases against independent Torch contractions','Three new frozen native image interventions: constant, stripes and ramp','Negligible activation fixture exposes declared denominator-guard boundary'],maximumError,sourceHashes:Object.fromEntries(files.map(file=>[file,hash(file)])),limits:'Numerical source review; negligible-mass display correction and actual browser evidence are reviewed separately.'},null,2)+'\n');
console.log(JSON.stringify({passed:true,groups:3,maximumError}));
