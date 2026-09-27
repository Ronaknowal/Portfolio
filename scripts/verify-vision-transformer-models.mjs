import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { composeInfluence, dinoDirection, featureCosines, frozenVision, gramGeometry, imagePatches, patchProjection, visionMACs, windowRead } from '../src/learn/data/vision-transformer-models.js';
const id = 'vision-transformers-vit-deit-swin-dinov2', folder = `docs/teaching/deep-learning-completion/${id}`;
const read = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const close = (a,b,t=1e-10) => { if (Array.isArray(b)) { assert.equal(a.length,b.length); b.forEach((v,i)=>close(a[i],v,t)); } else assert.ok(Math.abs(a-b)<=t, `${a} differs from ${b} by more than ${t}`); };
const error = (a,b) => Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-b.flat(Infinity)[i])));
const checks = [], check = (name,fn) => { fn(); checks.push({name,passed:true}); };
const asset=read(`public/learn-assets/${id}/plain-vit.json`), examples=read('src/learn/data/vision-transformer-examples.json'), fixtures=read(`${folder}/native-fixtures.json`);
let maximumLogitError=0, maximumFeatureError=0, maximumAttentionError=0, maximumPCAError=0;
check('All sixteen native original/pixel/permutation/fresh probes match logits, probabilities, all features and sixteen attention rows',()=>{
  for(const fixture of fixtures){
    const actual=frozenVision(asset.state,fixture.image,{order:fixture.order,movePositions:fixture.movePositions,pca:asset.pca});
    const rows=actual.attention.map(layer=>layer.map(head=>[head[0],head[6]]));
    maximumLogitError=Math.max(maximumLogitError,error(actual.logits,fixture.logits));
    maximumFeatureError=Math.max(maximumFeatureError,error(actual.features,fixture.features));
    maximumAttentionError=Math.max(maximumAttentionError,error(rows,fixture.attentionRows));
    close(actual.logits,fixture.logits,1e-4);close(actual.probabilities,fixture.probabilities,1e-5);close(actual.features,fixture.features,1e-4);close(rows,fixture.attentionRows,1e-5);
    if(fixture.condition==='original'){const expected=examples.examples[fixture.source-1];maximumPCAError=Math.max(maximumPCAError,error(actual.patchPCA,expected.patch_pca));close(actual.patchPCA,expected.patch_pca,1e-4);}
  }
});
check('Patch contributions, coefficient changes, complete unfold coordinates and nontrivial null',()=>{
  const weights=[[1,0,0,1],[0,1,-1,0]], biases=[.5,1];
  close(patchProjection([2,1,4,0],weights,biases).output,[2.5,-2]);close(patchProjection([2,1,2,0],weights,biases).output,[2.5,0]);close(patchProjection([1,1,4,1],weights,biases).output,[2.5,-2]);
  const image=Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>4*r+c));close(imagePatches(image).map(p=>p.pixels),[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]);
  close(patchProjection([1,4,2,3],weights,biases).output,[4.5,3]);
});
check('Spatial paths compose; boundary mask and relative bias have distinct measurable effects',()=>{
  const values=Array.from({length:36},(_,i)=>i===0?16:0), first=windowRead(values,6,6,2,0), second=windowRead(first.output,6,6,2,1);
  const influence=composeInfluence(second.weights,first.weights);close(second.output[14],1);close(influence[14][0],1/16);close(influence[14][35],0);
  close(windowRead(first.output,6,6,2,0).output[14],0);
  for(const matrix of [first.weights,second.weights,influence])for(const row of matrix)close(row.reduce((a,b)=>a+b,0),1);
  const corner=values.map((_,i)=>i===35?16:0), firstCorner=windowRead(corner,6,6,2,0);
  close(windowRead(firstCorner.output,6,6,2,1).output[0],0);close(windowRead(firstCorner.output,6,6,2,1,{wrap:true}).output[0],1);
  const signal=values.map((_,i)=>i===7?10:0), bias=Array.from({length:3},()=>[0,0,0]);
  close(windowRead(signal,6,6,2,1).output[14],2.5);bias[2][2]=Math.log(2);close(windowRead(signal,6,6,2,1,{bias}).output[14],4);
  const four=windowRead(Array(16).fill(0),4,4,2,0), shifted=windowRead(four.output,4,4,2,1);close(composeInfluence(shifted.weights,four.weights)[5],Array(16).fill(1/16));
});
check('Cross-view targets use stable probabilities, exact derivatives, offset null and collapsed stationary case',()=>{
  const teacher=[.2,-.1,.4], center=[.1,.1,0], student=[0,.2,-.2], base=dinoDirection(teacher,center,student,.25,.5), shifted=dinoDirection(teacher,center,student,.25,.5,3);
  close(base.target,shifted.target);assert.ok(base.gradient[0]>0);assert.ok(dinoDirection([.2,-.1,0],center,student,.25,.5).gradient[0]<0);
  for(let i=0;i<3;i++){const plus=[...student],minus=[...student];plus[i]+=1e-6;minus[i]-=1e-6;close((dinoDirection(teacher,center,plus,.25,.5).loss-dinoDirection(teacher,center,minus,.25,.5).loss)/2e-6,base.gradient[i],1e-8);}
  const collapse=dinoDirection([0,0,0],[0,0,0],[0,0,0],.25,.5);close(collapse.gradient,[0,0,0]);close(collapse.loss,Math.log(3));
  const stepped=student.map((v,i)=>v-.05*base.gradient[i]);assert.ok(dinoDirection(teacher,center,stepped,.25,.5).loss<base.loss);assert.throws(()=>dinoDirection(teacher,center,student,0,.5));
});
check('Gram matrix geometry and cosine matching retain explicit normalization',()=>{
  close(gramGeometry([90,180,270,360]).loss,0);close(gramGeometry([0,90,180,0]).loss,6);close(gramGeometry([0,0],[0,90]).loss,2);
  const source=frozenVision(asset.state,examples.examples[0].image), target=frozenVision(asset.state,examples.examples[1].image);
  close(featureCosines(source.features[6],target.features.slice(1)),examples.matching.all_cosines,1e-5);assert.equal(featureCosines([0,0],[[1,0]])[0],null);
});
check('Frozen model meaningful pixel contrast, input null, content permutation and joint invariance',()=>{
  const original=examples.examples[2].image, baseline=frozenVision(asset.state,original), edit=original.map(row=>[...row]);
  assert.notEqual(baseline.predicted,2);edit[2][3]=1-edit[2][3];const changed=frozenVision(asset.state,edit);assert.equal(changed.predicted,2);assert.ok(changed.probabilities[2]>.99);
  const unchanged=original.map(row=>[...row]);unchanged[2][4]=1-unchanged[2][4];close(frozenVision(asset.state,unchanged).logits,baseline.logits,0);
  const order=Array.from({length:16},(_,i)=>(i*5+3)%16);assert.ok(error(frozenVision(asset.state,original,{order}).logits,baseline.logits)>1e-3);close(frozenVision(asset.state,original,{order,movePositions:true}).logits,baseline.logits,1e-11);
});
check('Calculated operation counts and all seventeen native weighted-value products agree',()=>{
  assert.equal(visionMACs(224,224).sequence,197);assert.equal(visionMACs(384,384).sequence,577);assert.equal(visionMACs(224,224).pairs,2*197**2*768);
  const r=examples.clsRead;close(r.weights.reduce((a,b)=>a+b,0),1,2e-7);close(r.weights.map((v,i)=>v*r.values[i]),r.products,3e-7);close(r.products.reduce((a,b)=>a+b,0),r.output,3e-7);
});
check('Current JSX parses, imports exist, and every deployed learner file matches canonical bytes',()=>{
  for(const path of [`src/learn/data/topics/${id}.jsx`,'src/learn/components/lesson-labs/VisionTransformerFigures.jsx','src/learn/components/lesson-labs/VisionTransformerLabs.jsx'])parse(fs.readFileSync(path,'utf8'),{sourceType:'module',plugins:['jsx']});
  for(const file of fs.readdirSync(`public/learn-code/${id}`))assert.ok(fs.readFileSync(`public/learn-code/${id}/${file}`).equals(fs.readFileSync(`docs/teaching/drafts/${id}/${file}`)),`stale deployed ${file}`);
  assert.ok(fs.existsSync('src/learn/components/lesson-labs/vision-transformer.css'));
});
const result={passed:true,checks,nativeFixtures:fixtures.length,maximumLogitError,maximumFeatureError,maximumAttentionError,maximumPCAError};
fs.writeFileSync(`${folder}/model-checks.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
