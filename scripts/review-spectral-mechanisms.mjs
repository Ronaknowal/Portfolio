import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as model from '../src/learn/data/spectral-regularization-models.js';
const dir='docs/teaching/deep-learning-completion/spectral-normalization-gradient-penalty';
const fixtures=JSON.parse(fs.readFileSync(`${dir}/independent-fixtures.json`));
let comparisons=0,maximumError=0;
function close(a,b,label,tolerance=1e-9) {
  const aa=[a].flat(Infinity),bb=[b].flat(Infinity); assert.equal(aa.length,bb.length,label);
  aa.forEach((x,i)=>{assert.ok(Number.isFinite(x)&&Number.isFinite(bb[i]),label);const e=Math.abs(x-bb[i]);assert.ok(e<=tolerance,`${label}: ${e}`);maximumError=Math.max(e,maximumError);comparisons++;});
}
for(const c of fixtures.matrices){
  close(model.singular2(c.w).values,c.values,'full SVD');
  close(model.normalizationGradient(c.w,c.h).gradient,c.gradient,'autograd normalization quotient');
  for(const key of ['exact','cap','frobenius'])close(model.normalizeMatrix(c.w,key,c.target).effective,c[key],key);
  close(model.normalizeMatrix(c.w,'singular-cap',c.target).effective,c.capped,'singular cap');
  const gradient=model.normalizationGradient(c.w,c.h).gradient;
  close(model.dot(gradient.flat(),c.w.flat()),0,'radial derivative of scale invariant normalization');
  close(model.normalizeMatrix(model.scale(c.w,3.7)).effective,model.normalizeMatrix(c.w).effective,'positive scale null');
}
for(const c of fixtures.convolutions){const value=model.convolution(c.kernel,c.mode,c.x,c.stride);close(value.matrix,c.matrix,'literal windows');close(value.output,c.output,'window outputs');close(value.operatorNorm,c.sigma,'operator SVD');}
for(const c of fixtures.penalties){const v=model.linearPenalty(c.w,c.strength,c.rate,c.kind);close(v.penalty,c.loss,'penalty objective');close(v.gradient,c.gradient,'penalty autograd');close(v.updated,c.updated,'penalty step');}
for(const c of fixtures.forwards){
  const saved=JSON.parse(fs.readFileSync(`public/learn-code/spectral-normalization-gradient-penalty/${c.file}`)),layers=saved[`${c.kind}_layers`];
  const before=JSON.stringify(layers),v=model.frozenForward(c.point,layers,c.kind);
  close(v.value,c.output,'independent functional forward');close(v.jacobian,c.jacobian,'full input Jacobian');
  if(c.kind==='generator')close(model.frozenForward([...c.point].reverse(),model.swappedGenerator(layers),'generator').value,v.value,'exact coordinate rename');
  assert.equal(JSON.stringify(layers),before,'frozen input weights stay immutable');
}
for(const c of fixtures.library){let buffers=c.initial;for(const r of c.records){let actual=model.libraryAccess(c.w,buffers,r.training);close(actual.effective,r.effective,'real PyTorch effective weights');close(actual.u,r.u,'real PyTorch cached u');close(actual.v,r.v,'real PyTorch cached v');buffers=actual;}}
assert.equal(model.normalizationGradient([[1,0],[0,1]],[[1,2],[3,4]]),null,'repeated singular value explicitly unsupported');
assert.equal(model.normalizationGradient([[0,0],[0,0]],[[1,2],[3,4]]),null,'zero singular value explicitly unsupported');
assert.ok(model.powerTrace([[1,0],[0,0]],[0,1]).error,'rank deficient direction gets a useful error');
assert.equal(model.linearPenalty([0,0],2,.1).gradient,null,'target-one cusp is not assigned an invented derivative');
close(model.linearPenalty([0,0],2,.1,'zero').gradient,[0,0],'zero-centered smooth origin');
close(model.probePenalty(1,4,[-.5,0,.5]).penalty,0,'unsampled steep region');
close(model.probePenalty(1,4,[-.5,0,2]).penalty,32/3,'probe crosses kink');
for(let j=1;j<18;j++){
  const w=[[1+j/10,-.4],[.2,.8]],b=[.6,-.1],x=[1.7,j/20],r=model.linearMargin(w,b,x);
  close(r.boundary.map((v,i)=>v-x[i]),r.delta,'closest boundary displacement');
  close(model.dot(r.normal,r.boundary)+b[0]-b[1],0,'boundary reaches exact logit tie');
  assert.ok(r.jointRadius<=r.radius+1e-12,'sufficient radius does not exceed exact radius');
}
const receipt={passed:true,comparisons,maximumError,tolerance:1e-9,scope:'Independent NumPy SVD, functional PyTorch/autograd, actual parametrization modes, exact symmetries and boundary cases. All9 retained models; no refitting.'};
fs.writeFileSync(`${dir}/independent-model-checks.json`,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
