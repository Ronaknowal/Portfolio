import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {matrixChainPlan, balloonPlan, treeBoundaryPlan, treeBoundaryWitness, createDigitCounter} from '../src/learn/data/dp-state-families-models.js';
import {dpStateFamiliesExamples} from '../src/learn/data/dp-state-families-examples.js';
import {dynamicProgrammingExamples} from '../src/learn/data/dynamic-programming-examples.js';
import practice from '../src/learn/data/practice/dynamic-programming-states-transitions-optimization.js';

const require=createRequire(import.meta.url);
const sourceHashes=require('./dp-state-families-source-hashes.cjs')();
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const baseline=JSON.parse(fs.readFileSync('docs/teaching/evidence/dp-state-families-original.json'));
const retainedPaths=[
  'src/learn/data/dynamic-programming-models.js',
  'src/learn/data/dynamic-programming-examples.js',
  'src/learn/components/lesson-labs/DynamicProgrammingLabs.jsx',
  'src/learn/components/lesson-labs/dynamic-programming-labs.css',
];
const preserved=retainedPaths.map(path=>{
  const prior=baseline.files.find(file=>file.source===path);
  assert.equal(digest(fs.readFileSync(path)),prior.sha256);
  assert.equal(digest(fs.readFileSync(prior.archive)),prior.sha256);
  return {path,sha256:prior.sha256};
});
const oldPrograms=baseline.programs.map(prior=>{
  const now=dynamicProgrammingExamples[prior.id];
  assert.equal(digest(now.code),prior.codeSha256);
  assert.equal(now.expected,prior.expected);
  return {id:prior.id,...now};
});
assert.equal(oldPrograms.length,16);
const problems=practice.groups.flatMap(group=>group.problems.map(problem=>({group:group.id,...problem})));
baseline.problems.forEach(prior=>assert.deepEqual(problems.find(p=>p.number===prior.number&&p.group===prior.group),prior));
assert.equal(baseline.problems.length,12);
assert.equal(problems.length,15);
let seed=837291;
const random=max=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed%max;};
const matrices=Array.from({length:18},(_,index)=>{
  const dimensions=Array.from({length:2+index%5},()=>1+random(12));
  return {dimensions,plan:matrixChainPlan(dimensions)};
});
matrices.push({dimensions:[20,1,20,1,20,1,20],plan:matrixChainPlan([20,1,20,1,20,1,20])});
const balloons=Array.from({length:15},(_,index)=>{
  const values=Array.from({length:index%7},()=>random(7));
  return {values,plan:balloonPlan(values)};
});
function pruferTree(n){
  const sequence=Array.from({length:n-2},()=>random(n));
  const degree=Array(n).fill(1),edges=[];
  sequence.forEach(node=>degree[node]++);
  for(const node of sequence){
    const leaf=degree.findIndex(value=>value===1);
    edges.push([node,leaf]);degree[leaf]--;degree[node]--;
  }
  edges.push(degree.map((value,node)=>value===1?node:-1).filter(node=>node>=0));
  return edges;
}
const trees=Array.from({length:24},(_,index)=>{
  const count=4+index%6,weights=Array.from({length:count},()=>random(26)-8);
  const edges=pruferTree(count),root=random(count),plan=treeBoundaryPlan(weights,edges,root);
  const queries=weights.flatMap((_,node)=>[false,true].map(parent=>({node,parent,...treeBoundaryWitness(plan,node,parent)})));
  return {weights,edges,root,plan,queries};
});
const bounds=[0,101,1000,10203,500102,908070,999999,234561,700001];
const digits=bounds.map(bound=>{
  const counter=createDigitCounter(bound),length=String(bound).length,prefixes=new Set(['']);
  const numbers=[0,bound];
  for(let index=0;index<35;index++)numbers.push(random(bound+1));
  for(const number of numbers){
    const padded=String(number).padStart(length,'0');
    for(let end=0;end<=length;end++)prefixes.add(padded.slice(0,end));
  }
  const states=[],rejections=[];
  for(const prefix of prefixes){
    try{states.push(counter.inspect(prefix));}catch(error){rejections.push({prefix,message:error.message});}
  }
  return {bound,total:counter.total,states,rejections};
});
const destination='scratch/dp-state-families-independent';
fs.mkdirSync(destination,{recursive:true});
fs.writeFileSync(destination+'/model-fixtures.json',JSON.stringify({
  checkedAt:new Date().toISOString(),sourceHashes,preserved,oldPrograms,newPrograms:dpStateFamiliesExamples,
  preservedPractice:baseline.problems.length,practice:problems,matrices,balloons,trees,digits,
},null,2));
const originalPacket='docs/teaching/evidence/dp-state-families-author-review.json';
const preservedPacket=destination+'/author-packet-before-independent-amendment.json';
if(fs.existsSync(originalPacket)&&!fs.existsSync(preservedPacket))fs.copyFileSync(originalPacket,preservedPacket);
console.log(JSON.stringify({matrices:matrices.length,trees:trees.length,balloons:balloons.length,digitStates:digits.reduce((sum,caseValue)=>sum+caseValue.states.length,0),preservedPrograms:16,preservedPractice:12}));
