import assert from 'node:assert/strict';
import fs from 'node:fs';
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import {osExamples} from '../src/learn/data/os-foundations-examples.js';
import {arrayMapExamples} from '../src/learn/data/array-map-foundations-examples.js';
import {linkedExamples} from '../src/learn/data/linked-foundations-examples.js';
import {scheduleTrace,translationModel,sharingTrace} from '../src/learn/data/os-foundations-model.js';
import {arrayMovementTrace,textModel,hashTrace} from '../src/learn/data/array-map-foundations-model.js';
import {reverseTrace,bracketTrace,ringTrace,queueEvents} from '../src/learn/data/linked-foundations-model.js';
import {topicCatalogue} from '../src/learn/data/curriculum/topic-catalogue.js';
import {learningPaths,getLearningRoute} from './lib/authoring-curriculum.mjs';
import {tracks} from './lib/authoring-curriculum.mjs';
const dir='scratch/systems-three-review';fs.mkdirSync(dir,{recursive:true});
const examples={};
for(const [group,source,id] of [['os',osExamples,'os-processes-virtual-memory-isolation'],['arrays',arrayMapExamples,'arrays-strings-hash-maps'],['linked',linkedExamples,'linked-lists-stacks-queues']]) {
  const lesson=fs.readFileSync(`src/learn/data/topics/${id}.jsx`,'utf8');
  assert.deepEqual((await collectLessonExamples(`src/learn/data/topics/${id}.jsx`)).map(reference=>reference.key).sort(),Object.keys(source).sort());
  for(const [key,example]of Object.entries(source))examples[group+'.'+key]=example;
}
const schedules=[];
for(const quantum of [1,2])for(const io of [false,true]) {
  const trace=scheduleTrace(quantum,io),expected=quantum===1?(io?'ABABBA':'ABABAB'):(io?'AABBBA':'AABBAB');
  assert.equal(trace.at(-1).timeline.join(''),expected);
  for(const [i,s]of trace.entries()){
    assert.equal(s.time,i);assert.equal(s.timeline.length,i);
    assert.equal(new Set(s.ready).size,s.ready.length);
    assert.ok(Object.values(s.jobs).filter(j=>j.state==='running').length<=1);
    for(const id of ['A','B']){const j=s.jobs[id];assert.equal(j.state==='ready',s.ready.includes(id));assert.equal(j.value,j.code.slice(0,j.pc).filter(c=>c!=='I/O').reduce((n,c)=>n+Number(c),0));}
  }
  assert.equal(trace.at(-1).jobs.A.value,io?2:3);assert.equal(trace.at(-1).jobs.B.value,30);
  schedules.push({quantum,io,trace});
}
const translations=[];for(const process of ['A','B'])for(let address=0;address<64;address++)for(const access of ['read','write'])translations.push(translationModel(process,address,access));
const sharing=[false,true].map(shared=>({shared,trace:sharingTrace(shared)}));
for(const {shared,trace}of sharing){assert.equal(trace.at(-1).aValue,shared?9:7);assert.equal(trace.at(-1).bValue,9);for(const s of trace){assert.equal(s.aValue,s.frames[s.aFrame]);assert.equal(s.bValue,s.frames[s.bFrame]);}}
const arrays=[];for(const operation of ['front','insert','append','delete'])for(const full of [false,true])arrays.push({operation,full,trace:arrayMovementTrace(operation,full)});
const text=[];for(const kind of ['ascii','composed','decomposed','emoji'])for(const normalize of [false,true])text.push({kind,normalize,model:textModel(kind,normalize)});
const hashes=[];for(const key of [10,14,18,22])for(const capacity of [4,5])for(const operation of ['get','set'])hashes.push({key,capacity,operation,trace:hashTrace(key,capacity,operation)});
const reversals=[];
function walk(nodes,start){const ids=[];while(start!==null){assert.ok(!ids.includes(start),'cycle');ids.push(start);start=nodes.find(n=>n.id===start).next;}return ids;}
for(const size of [0,1,3])for(const broken of [false,true]){
  const trace=reverseTrace(size,broken),ids=['A','B','C'].slice(0,size),last=trace.at(-1);
  assert.deepEqual(walk(last.nodes,last.head),broken?ids.slice(0,1):ids.toReversed());
  if(!broken)for(const s of trace.filter(s=>['Initialize','Advance'].includes(s.phase))) {
    const prefix=walk(s.nodes,s.previous),suffix=walk(s.nodes,s.current);
    assert.deepEqual([...prefix].reverse().concat(suffix),ids);
    assert.equal(new Set([...prefix,...suffix]).size,size);
  }
  reversals.push({size,broken,trace});
}
const brackets=[];function strings(prefix,left){brackets.push({text:prefix,result:bracketTrace(prefix).at(-1).result});if(left)for(const c of '()[]')strings(prefix+c,left-1);}strings('',6);
const rings=[3,4].map(capacity=>({capacity,trace:ringTrace(capacity)}));
for(const {capacity,trace}of rings)for(const s of trace){assert.ok(s.size>=0&&s.size<=capacity);assert.equal(s.tail,(s.head+s.size)%capacity);assert.equal(s.logical.length,s.size);}
fs.writeFileSync(`${dir}/native-cases.json`,JSON.stringify({examples,schedules,translations,sharing,arrays,text,hashes,reversals,brackets,rings,queueEvents},null,2));
if(fs.existsSync(`${dir}/before.json`)) {
  const before=JSON.parse(fs.readFileSync(`${dir}/before.json`));
  assert.deepEqual(Object.keys(topicCatalogue).sort(),before.ids.slice().sort(),'topic IDs changed');
  const routes=learningPaths.map(p=>({id:p.id,topicIds:getLearningRoute(p).topicIds}));
  for(const old of before.paths){const current=routes.find(p=>p.id===old.id);for(const id of old.topicIds)assert.ok(current.topicIds.includes(id),`old path topic removed: ${old.id}/${id}`);}
  for(const old of before.modules)assert.deepEqual([...new Set(tracks.find(t=>t.id===old.id).topicIds)].sort(),[...new Set(old.topicIds)].sort(),`module membership ${old.id} changed`);
}
console.log(`PASS: nine model contracts; ${Object.keys(examples).length} displayed programs exported; ${brackets.length} bracket strings. Snapshot IDs/memberships retained when available; current reading-order/conservation checks live in verify-curriculum and verify-programming-module-conservation.`);
