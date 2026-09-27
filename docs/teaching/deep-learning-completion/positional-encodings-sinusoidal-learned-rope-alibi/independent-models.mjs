import fs from 'node:fs';
import assert from 'node:assert/strict';
import { positionMovementForward, cachePositionRead, positionCacheDefault, alibiSlopes, relativeBucket, extensionFrequencies, learnedPosition } from '../../../../src/learn/data/positional-encoding-models.js';
const id = 'positional-encodings-sinusoidal-learned-rope-alibi', root = `docs/teaching/deep-learning-completion/${id}`;
const native = JSON.parse(fs.readFileSync(`${root}/independent-native.json`));
let scalarComparisons = 0, maxModelError = 0, maxGradientError = 0;
const checks = [];
function close(left, right, tolerance = 2e-5) {
  if (Array.isArray(left)) { assert.equal(left.length, right.length); left.forEach((v, i) => close(v, right[i], tolerance)); }
  else { const error = Math.abs(left - right); assert.ok(error <= tolerance, `${left} != ${right}, difference ${error}`); maxModelError = Math.max(maxModelError, error); scalarComparisons++; }
}
for (const test of native.cases) {
  const model = JSON.parse(fs.readFileSync(`public/learn-code/${id}/movement-${test.mode}.json`));
  const result = positionMovementForward(model, test.points, test.ids, test.pads);
  for (const key of ['logits', 'probabilities', 'pooled']) close(result[key], test[key]);
  close(result.logits,test.browserGeluLogits,1e-11);
  close(result.heads.map(head => head.attention), test.attention);
  if (test.gradient) {
    for (const [row, coordinate] of [[0,0],[0,1],[5,0],[8,1],[10,0]]) {
      const epsilon = 1e-5, left = structuredClone(test.points), right = structuredClone(test.points);
      left[row][coordinate] -= epsilon; right[row][coordinate] += epsilon;
      const a = positionMovementForward(model,left,test.ids).logits, b = positionMovementForward(model,right,test.ids).logits;
      const derivative = a.reduce((sum,v,i) => sum + (b[i]-v)/(2*epsilon)*(-.7+1.8*i/14),0);
      const error = Math.abs(derivative - test.gradient[row][coordinate]);
      assert.ok(error < 5e-5, `${test.mode} input derivative error ${error}`);
      assert.ok(Math.abs(derivative-test.browserGeluGradient[row][coordinate]) < 1e-6, 'Finite differences match native autograd for the actual browser GELU approximation');
      maxGradientError = Math.max(maxGradientError,error); scalarComparisons++;
    }
    const permutation = [7,2,10,0,5,1,9,3,8,6,4];
    close(positionMovementForward(model,permutation.map(i=>test.points[i]),permutation.map(i=>test.ids[i])).logits,result.logits,1e-11);
    if (['none','rope','alibi'].includes(test.mode)) close(positionMovementForward(model,test.points,test.ids.map(v=>v+137)).logits,result.logits,1e-11);
    if (['none','alibi'].includes(test.mode)) close(positionMovementForward(model,[...test.points].reverse(),test.ids).logits,result.logits,1e-11);
  }
  checks.push(`${test.mode}/${test.name}: full output, pooled state and both attention maps vs independent native SDPA`);
}
// A separate complex-style pair implementation and explicit legal-row read.
const turn = (row,position,base) => row.map((_,i)=> {
  const j=i-i%2, angle=position/Math.pow(base,j/row.length);
  return i%2 ? row[j]*Math.sin(angle)+row[j+1]*Math.cos(angle) : row[j]*Math.cos(angle)-row[j+1]*Math.sin(angle);
});
const sumProduct = (a,b) => a.reduce((sum,x,i)=>sum+x*b[i],0);
for (const mode of ['rope','alibi']) {
  const setting = {...positionCacheDefault(), mode, query:[-.9,.2,1.1,.7], keys:[[1,.3,-.4,.1],[.2,-1,.6,.8],[-.3,.4,1.2,-.5],[.8,.7,-.1,.3]], values:[[2,-3],[-1,.5],[.6,.1],[-.2,1.3]], ids:[14,2,9,11], queryId:11,rotaryId:11,maskId:11,base:73,cacheBase:73,slope:.37};
  const q = mode==='rope' ? turn(setting.query,11,73):setting.query;
  const scores=setting.keys.map((key,i)=>setting.ids[i]>11 ? -Infinity : sumProduct(q,mode==='rope'?turn(key,setting.ids[i],73):key)/2-(mode==='alibi'?.37*(11-setting.ids[i]):0));
  const terms=scores.map(v=>Math.exp(v-Math.max(...scores))), total=terms.reduce((a,b)=>a+b,0), expected=terms.map(v=>v/total);
  const actual=cachePositionRead(setting); close(actual.weights,expected,1e-14); close(actual.output,[0,1].map(d=>sumProduct(expected,setting.values.map(v=>v[d]))),1e-14);
  const perm=[3,1,0,2];close(cachePositionRead({...setting,keys:perm.map(i=>setting.keys[i]),values:perm.map(i=>setting.values[i]),ids:perm.map(i=>setting.ids[i])}).output,actual.output,1e-14);
  const values=structuredClone(setting.values);values[0]=[999,-999];close(cachePositionRead({...setting,values}).output,actual.output,0);
  assert.equal(cachePositionRead({...setting,maskId:0}).output,null);
  checks.push(`${mode}: new irregular logical IDs, future-value isolation, physical permutation and all-masked null`);
}
for (const h of [1,3,5,7,12,33,127]) {
  const recurse=n=> { if ((n&(n-1))===0) {const start=Math.pow(2,-Math.pow(2,-(Math.log2(n)-3)));return Array.from({length:n},(_,i)=>start**(i+1));}const p=2**Math.floor(Math.log2(n));return [...recurse(p),...recurse(2*p).filter((_,i)=>i%2===0).slice(0,n-p)];};
  close(alibiSlopes(h),recurse(h),1e-14);
}
for(let offset=-256;offset<=256;offset++) {const distance=Math.abs(offset);let b=0;while(b<15 && (b<8 ? distance>=b+1 : distance>=8*16**((b+1-8)/8)))b++;assert.equal(relativeBucket(offset),b+(offset>0?16:0));scalarComparisons++;}
for(const width of [4,8,32,128]) { const e=extensionFrequencies(width,731,512,1);for(const key of ['pi','baseScaled','yarn'])close(e[key],e.original,1e-14);const stretched=extensionFrequencies(width,731,512,7);close(stretched.baseScaled[0],stretched.original[0],1e-14);close(stretched.baseScaled.at(-1)*7,stretched.original.at(-1),1e-14); }
assert.throws(()=>learnedPosition([[1],[2]],2));assert.throws(()=>learnedPosition([[1],[2]],.5));assert.throws(()=>learnedPosition([[1],[2]],-1));
for(const name of ['author-calculations.py','position_library_bridge.py','mechanism-calculations.py','data-provenance.md','movement_libras.data','movement_libras.names','author-results.json','position-models.json','mechanism-fixtures.json'])assert.deepEqual(fs.readFileSync(`public/learn-code/${id}/${name}`),fs.readFileSync(`docs/teaching/drafts/${id}/${name}`));
checks.push('Non-power-of-two original slope ordering; all signed T5 bucket boundaries -256..256; identity extension and stretched last-pair law; strict learned indices; all nine published source/data copies match canonical bytes');
const receipt={passed:true,scalarComparisons,maxModelError,maxGradientError,checks,nativeChecks:native.checks,limitations:['No duplicate fit campaign; inference/gradients and recorded validation selection checked.','Source and numerical review does not establish painted browser behavior.']};
fs.writeFileSync(`${root}/independent-numerical-checks.json`,JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({passed:true,scalarComparisons,maxModelError,maxGradientError,checks:checks.length+native.checks.length}));
