import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {parse} from '@babel/parser';
import katex from 'katex';
import {fittedConstant,regressionPenalty,tripletGeometry,candidateCompetition,focalTerm} from '../src/learn/data/loss-functions-models.js';
import {normalizeVector,normalizeTensor,normalizationGradient,batchNormalizationStep} from '../src/learn/data/normalization-models.js';
const records=[];
const near=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<t,`${a} versus ${b}`);
function check(name,fn){fn();records.push(name);}
check('Huber optimum is locally minimal on a changed signed seven-point sample',()=>{
 const values=[-1,-1,-1,0,1,1,20],fit=fittedConstant(values,'huber');near(fit,0);
 const objective=c=>values.reduce((sum,v)=>sum+regressionPenalty(c-v,'huber'),0);
 assert.ok(objective(fit)<objective(fit-.01));assert.ok(objective(fit)<objective(fit+.01));
});
check('Strict triplet boundaries and equal-distance row-order tie are retained',()=>{
 const result=tripletGeometry([[0,0],[1,0],[1,0],[Math.sqrt(2),0],[1.1,0]],1,true);
 assert.equal(result.candidates[0].kind,'hard');assert.equal(result.candidates[1].kind,'easy');assert.equal(result.selected,2);
 assert.equal(tripletGeometry([[0,0],[1,0],[1.1,0],[-1.1,0],[2,0]],1,true).selected,0);
});
check('Candidate CE preserves a common score shift, and duplicate-score loss approaches log 2',()=>{
 const a=candidateCompetition([.5,.5,-.3],.1),b=candidateCompetition([.7,.7,-.1],.1);
 near(a.loss,b.loss);a.probabilities.forEach((p,i)=>near(p,b.probabilities[i]));near(candidateCompetition([.8,.8,-.1],.05).loss,Math.log(2),1e-7);
});
check('Focal shared-bias cancellation at count 90 is distinct from weighting and focusing',()=>{
 near(90*focalTerm(.01,0,0).slope+focalTerm(.1,1,0).slope,0);
 assert.ok(90*focalTerm(.01,0,2).slope+focalTerm(.1,1,2).slope<0);
 assert.ok(Math.abs(focalTerm(.2,1,0,.25).slope-focalTerm(.2,1,0).slope)>.1);
});
check('Normalization input derivatives cancel in the shared-offset direction on changed features',()=>{
 const x=[1.2,-2.4,3.7],gamma=[.4,2,-.8],beta=[1,-1,.2],target=[0,.5,-1];
 const result=normalizationGradient(x,gamma,beta,target);near(result.inputGradient.reduce((a,b)=>a+b,0),0);
 const shifted=normalizationGradient(x.map(v=>v+13),gamma,beta,target);near(result.loss,shifted.loss);
 const h=1e-5;x.forEach((_,i)=>{const plus=x.map((v,j)=>v+(i===j?h:0)),minus=x.map((v,j)=>v-(i===j?h:0));near(result.inputGradient[i],(normalizationGradient(plus,gamma,beta,target).loss-normalizationGradient(minus,gamma,beta,target).loss)/(2*h),1e-8);});
});
check('GroupNorm change outside the selected channel group is an exact local null',()=>{
 const values=Array.from({length:16},(_,i)=>i+1),original=normalizeTensor(values,'group',2),changed=values.slice();changed[4]=40;
 const result=normalizeTensor(changed,'group',2);for(let i=0;i<4;i++)near(result[i],original[i]);assert.ok(Math.abs(result[4]-original[4])>.1);
});
check('Finite epsilon breaks exact positive-scale invariance but preserves common offsets',()=>{
 const x=[.0001,.0003],a=normalizeVector(x,1),b=normalizeVector(x.map(v=>10*v),1),c=normalizeVector(x.map(v=>v+3),1);
 assert.ok(Math.abs(a.output[0]-b.output[0])>.0001);a.output.forEach((v,i)=>near(v,c.output[i]));
});
check('BatchNorm evaluation is a buffer null and an affine change does not alter memory',()=>{
 const values=[-2,1,4,7],buffers={mean:2,variance:4};
 const evaluation=batchNormalizationStep(values,buffers,.25,false,1e-5,2,-1);assert.deepEqual(evaluation.next,buffers);
 const a=batchNormalizationStep(values,buffers,.25,true,1e-5,1,0),b=batchNormalizationStep(values,buffers,.25,true,1e-5,2,-1);assert.deepEqual(a.next,b.next);a.output.forEach((v,i)=>near(2*v-1,b.output[i]));
});
const files=['src/learn/data/topics/loss-functions-ce-mse-focal-contrastive-triplet.jsx','src/learn/data/topics/batch-layer-group-rms-normalization.jsx'];
const math={};
for(const file of files){let count=0,headings=0;const ast=parse(fs.readFileSync(file,'utf8'),{sourceType:'module',plugins:['jsx']});function visit(n){if(!n||typeof n!=='object')return;if(n.type==='JSXElement'){const name=n.openingElement.name.name;if(['InlineMath','MathBlock'].includes(name)){const value=n.children.find(c=>c.type==='JSXExpressionContainer')?.expression?.value;katex.renderToString(value,{throwOnError:true});count++;}if(name==='H2')headings++;}for(const value of Object.values(n)){if(Array.isArray(value))value.forEach(visit);else if(value&&typeof value==='object')visit(value);}}visit(ast);math[file]={mathExpressions:count,sections:headings};}
records.push('Both complete generated lessons parse, and all actual inline/block mathematics renders with strict KaTeX');
const sources=[...files,'src/learn/data/loss-functions-models.js','src/learn/data/normalization-models.js'];
const evidence={reviewer:'implement_dl_perceptrons (independent of root author)',status:'complementary probes passed; representation/input findings reported separately',checks:records,math,sourceHashes:Object.fromEntries(sources.map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]))};
fs.writeFileSync('docs/teaching/evidence/loss-normalization-independent-probes.json',JSON.stringify(evidence,null,2)+'\n');console.log(JSON.stringify(evidence,null,2));
