import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { savedDigitLogits, channelGradient, channelForward, samplingSupport } from '../src/learn/data/depthwise-convolution-models.js';
const evidence='docs/teaching/evidence/depthwise-independent.json';
fs.writeFileSync(evidence,JSON.stringify({passed:false,status:'running'}));
const reference=JSON.parse(fs.readFileSync('docs/teaching/evidence/depthwise-independent-native.json'));
const records=JSON.parse(fs.readFileSync('public/learn-code/depthwise-separable-dilated-convolutions/digit-inference.json'));
assert.equal(reference.passed,true);
let maxError=0;
function compare(actual,expected,tolerance=1e-10){assert.equal(actual.flat(Infinity).length,expected.flat(Infinity).length);assert.ok(actual.flat(Infinity).length>0);actual.flat(Infinity).forEach((value,i)=>{const error=Math.abs(value-expected.flat(Infinity)[i]);assert.ok(Number.isFinite(error)&&error<tolerance,`${error}`);maxError=Math.max(maxError,error);});}
assert.throws(()=>compare([NaN],[0]));assert.throws(()=>compare([1],[1.01]));
for(const row of reference.fixtures){const run=records.runs.find(run=>run.dilation===row.dilation);compare(savedDigitLogits(run,row.image,row.rank),row.logits);}
for(const row of reference.gradients){const gradient=channelGradient(row.input,row.filters,row.mixing,row.target,row.rate);compare(channelForward(row.input,row.filters,row.mixing).output,row.outputs);compare(gradient.filterGradient,row.filter_gradient);compare(gradient.mixingGradient,row.mixing_gradient);compare([gradient.after],[row.after]);}
let schedules=0;
// Full four-layer supported domain. Polynomial coefficients count path multiplicity,
// while the browser stores only a deduplicated set of reachable positions.
for(let a=1;a<=9;a++)for(let b=1;b<=9;b++)for(let c=1;c<=9;c++)for(let d=1;d<=9;d++){
 let coefficients=[1];
 for(const rate of [a,b,c,d]){const next=Array(coefficients.length+2*rate).fill(0);coefficients.forEach((count,i)=>{next[i]+=count;next[i+rate]+=count;next[i+2*rate]+=count;});coefficients=next;}
 assert.equal(coefficients.reduce((sum,count)=>sum+count,0),81);
 const expected=coefficients.flatMap((count,i)=>count?[i-(a+b+c+d)]:[]);
 assert.deepEqual(samplingSupport([a,b,c,d]).sites,expected);schedules++;
}
const sources=['src/learn/data/depthwise-convolution-models.js','src/learn/components/lesson-labs/DepthwiseConvolutionLabs.jsx','src/learn/components/lesson-labs/depthwise-convolutions.css','src/learn/data/topics/depthwise-separable-dilated-convolutions.jsx','scripts/verify-depthwise-independent.py','scripts/verify-depthwise-independent.mjs'];
const report={passed:true,reviewer:'convnext_finish',freshNativeScoreValues:300,gradientStates:3,fourLayerSchedules:schedules,maxAbsoluteDifference:maxError,findings:[{issue:'Cost-bar flex/padding selector collision',status:'Closed: scoped selectors and measured proportional widths at1366/390/320; see modern-convolution-browser report'},{issue:'Small grid targets without full-size alternatives',status:'Closed: synchronized row/column selectors measured44px at all three widths; see browser report'}],checks:['Independent effective-kernel Torch forward matches browser grouped route on thirty new model/rank/image states','Fresh autograd gradients and simultaneous updates match hand model on signed arbitrary states','All6561four-layer schedules match polynomial support and81path total','Comparison guards reject nonfinite and incorrect fixtures'],sources:Object.fromEntries(sources.map(file=>[file,createHash('sha256').update(fs.readFileSync(file)).digest('hex')])),limits:'Numerical review is distinct from modern-convolution-browser evidence; require its final production source match. No training rerun.'};
fs.writeFileSync(evidence,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
