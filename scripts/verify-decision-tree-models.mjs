import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as model from '../src/learn/data/decision-tree-models.js';
const near=(a,b)=>assert(Math.abs(a-b)<1e-11,`${a} != ${b}`);
const root=model.splitCandidates(model.inspectionRows);
assert.deepEqual(root.map(row=>[row.feature,row.threshold,row.left.length,row.right.length]),[[0,1.5,2,6],[0,3,4,4],[0,4.5,6,2],[1,2,4,4]]);
root.forEach((row,index)=>near(row.gain,[1/96,1/32,3/32,1/32][index]));
assert.equal(model.growTree(model.inspectionRows,{minimumLeaf:3}).threshold,3);
const cases=[];
for(const criterion of ['gini','entropy']) for(let depth=0;depth<=6;depth++) for(let minimum=1;minimum<=4;minimum++) {
  const tree=model.growTree(model.inspectionRows,{criterion,maxDepth:depth,minimumLeaf:minimum});
  const regions=model.leafRegions(tree);
  for(let x=0;x<=6;x+=.25) for(let y=0;y<=4;y+=.25) {
    const prediction=model.treePrediction(tree,[x,y]);
    const region=regions.find(row=>row.path===prediction.leafPath);
    assert(region && x>=region.bounds[0] && x<=region.bounds[1] && y>=region.bounds[2] && y<=region.bounds[3]);
    near(prediction.probability,prediction.rows.filter(id=>model.inspectionRows.find(row=>row.id===id).label).length/prediction.rows.length);
  }
  cases.push({criterion,depth,minimum,tree,regions});
}
for(const allowZero of [false,true]) assert.deepEqual(model.xorRows.map(row=>model.treePrediction(model.growTree(model.xorRows,{allowZero,maxDepth:2}),row.features).label),allowZero?[0,1,1,0]:[0,0,0,0]);
for(const values of [[1,1+Number.EPSILON],[1+Number.EPSILON,1+2*Number.EPSILON],[-Number.MIN_VALUE,0],[0,Number.MIN_VALUE]]) {
  const rows=values.map((x,index)=>({id:String(index),features:[x,0],label:index}));
  const tree=model.growTree(rows); assert(tree.left); assert.deepEqual(rows.map(row=>model.treePrediction(tree,row.features).label),[0,1]);
}
const pruning=[];
for(const alpha of [0,.04,.1,15/128,.12,.2,.3]) {
  const report=model.pruningReport(alpha); assert.equal(report.candidates.length,6);
  assert.equal(report.best.leaves,alpha<15/128?5:1); near(report.best.objective,Math.min(5*alpha,15/32+alpha)); pruning.push(report);
}
const forests=[];
for(const featureSampling of [false,true]) for(const trees of [1,2,6,12]) for(let selectedRow=0;selectedRow<8;selectedRow++) {
  const report=model.forestReport({featureSampling,trees,selectedRow});
  const omitted=report.members.filter(row=>!row.sampleIndices.includes(selectedRow));
  assert.equal(report.oobCount,omitted.length); assert.equal(report.oobProbability===null,omitted.length===0);
  for(const member of report.members) near(member.prediction.probability,member.prediction.rows.filter(id=>model.inspectionRows.find(row=>row.id===id).label).length/member.prediction.count);
  if(omitted.length) near(report.oobProbability,omitted.reduce((sum,row)=>sum+row.prediction.probability/omitted.length,0));
  forests.push(report);
}
for(const b of [1,2,20,200]) for(const rho of [0,.25,.3,1]) near(model.ensembleVariance(b,rho),(b+b*(b-1)*rho)/(b*b));
assert.deepEqual(['copy','used','group'].map(mode=>model.permutationRows(mode).filter(row=>row.label===row.predicted).length),[4,0,0]);
for(const operation of [()=>model.growTree([]),()=>model.growTree(undefined,{maxDepth:-1}),()=>model.splitCandidates(model.inspectionRows,{minimumLeaf:0}),()=>model.impurity([2]),()=>model.treePrediction(model.growTree(),[NaN,0]),()=>model.forestReport({trees:0}),()=>model.pruningReport(-1),()=>model.ensembleVariance(0,.3),()=>model.permutationRows('invalid')]) assert.throws(operation);
fs.mkdirSync('scratch/decision-tree-native',{recursive:true});
fs.writeFileSync('scratch/decision-tree-native/model-payload.json',JSON.stringify({rows:model.inspectionRows,cases,pruning,forests}));
console.log('Tree models: 56 fitted variants, 23,800 query-region checks, exact candidates/pruning, 64 OOB reports, adjacent-float partitions and bounded inputs passed.');
