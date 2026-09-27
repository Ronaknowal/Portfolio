import assert from 'node:assert/strict';
import fs from 'node:fs';
import {rotaryConventionTrace,xposPairTrace} from '../../../../src/learn/data/position-convention-models.js';
let comparisons=0;
const near=(a,b)=>{assert.ok(Math.abs(a-b)<1e-12);comparisons++;};
for(const [q,k,m,n,b] of [[[-.3,.8,1.2,-.7],[.5,-1,.2,.9],11,3,73],[[0,0,0,0],[1,2,3,4],0,27,10000],[[2,-2,.7,.1],[-1,.2,.9,-.6],-17,210,31]]){
 for(const pairs of [0,1,2]){
  const r=rotaryConventionTrace(q,k,m,n,b,pairs);
  const rotate=(a,p,j)=>[a[2*j]*Math.cos(p/b**(j/2))-a[2*j+1]*Math.sin(p/b**(j/2)),a[2*j]*Math.sin(p/b**(j/2))+a[2*j+1]*Math.cos(p/b**(j/2))];
  for(let j=0;j<2;j++){
   const qr=rotate(q,m,j),kr=rotate(k,n,j);
   near(r.restored[2*j],qr[0]);near(r.restored[2*j+1],qr[1]);
   const expected=j<pairs?qr[0]*kr[0]+qr[1]*kr[1]:q[2*j]*k[2*j]+q[2*j+1]*k[2*j+1];
   near(r.contributions[j],expected);
  }
 }
}
for(const [m,n] of [[512,0],[700,188],[0,512],[1024,1024],[177,59]]){
 const r=xposPairTrace(m,n),factor=Math.exp(Math.log(2/7)*(m-n)/512);
 near(r.product,factor);near(r.scaledDot,factor*Math.cos(m-n));near(Math.hypot(...r.scaledQuery),r.queryAmplitude);near(Math.hypot(...r.scaledKey),r.keyAmplitude);
 near(xposPairTrace(m+83,n+83).scaledDot,r.scaledDot);
}
const result={passed:true,comparisons,checks:['Independent2×2 trigonometric mapping checks half-split inverse and every partial-rotation pair choice','XPos amplitude product/common-shift invariance versus exp-log and cosine relative-angle identities; vector norms intentionally change']};
fs.writeFileSync('docs/teaching/deep-learning-completion/positional-encodings-sinusoidal-learned-rope-alibi/independent-conventions.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
