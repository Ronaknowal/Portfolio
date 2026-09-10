import assert from 'node:assert/strict';
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import { initialRace } from "../src/learn/data/thread-coordination-models.js";
import { raceStep } from "../src/learn/data/thread-coordination-models.js";
import { initialLocks } from "../src/learn/data/thread-coordination-models.js";
import { lockStep } from "../src/learn/data/thread-coordination-models.js";
import { deadlocked } from "../src/learn/data/thread-coordination-models.js";
import { conditionTrace } from "../src/learn/data/thread-coordination-models.js";
import {threadExamples} from "../src/learn/data/thread-coordination-examples.js";

const output='scratch/thread-completion-review';
fs.mkdirSync(output,{recursive:true});
// Exhaust all choices of conceptual workers, merging equivalent machine states.
function explore(initial,step,terminal,key) {
 const queue=[initial],seen=new Set(),finished=[];
 while(queue.length) {
  const state=queue.shift(),signature=key(state);
  if(seen.has(signature))continue;
  seen.add(signature);
  if(terminal(state)){finished.push(state);continue;}
  for(const id of ['A','B'])queue.push(step(state,id));
 }
 return {seen:seen.size,finished};
}
const raceKey=s=>JSON.stringify([s.value,s.owner,s.workers]);
const unsafe=explore(initialRace(),raceStep,s=>Object.values(s.workers).every(w=>w.phase===3),raceKey);
assert.deepEqual([...new Set(unsafe.finished.map(s=>s.value))].sort(),[1,2]);
const safe=explore(initialRace(true),raceStep,s=>Object.values(s.workers).every(w=>w.phase===3),raceKey);
assert.ok(safe.finished.length && safe.finished.every(s=>s.value===2 && s.owner===null));
const lockKey=s=>JSON.stringify([s.owners,s.workers]);
const opposite=explore(initialLocks(),lockStep,s=>deadlocked(s)||Object.values(s.workers).every(w=>w.pc===3),lockKey);
assert.ok(opposite.finished.some(deadlocked));
const ordered=explore(initialLocks(true),lockStep,s=>deadlocked(s)||Object.values(s.workers).every(w=>w.pc===3),lockKey);
assert.ok(ordered.finished.length && ordered.finished.every(s=>!deadlocked(s)&&Object.values(s.owners).every(owner=>owner===null)));
for(const scenario of ['empty','item','stolen']) {
 const states=conditionTrace(scenario);
 assert.equal(states[1].owner,'none');
 assert.equal(states[1].consumer,'waiting');
 assert.equal(states.at(-1).queue.length,0);
 assert.equal(states.some(s=>s.consumer.startsWith('consumed')),scenario==='item');
}
fs.writeFileSync(`${output}/fixtures.json`,JSON.stringify({examples:threadExamples,counts:{unsafe:unsafe.seen,safe:safe.seen,opposite:opposite.seen,ordered:ordered.seen}},null,2));
const python=process.env.LESSON_PYTHON||'scratch/lesson-tools/Scripts/python.exe';
const run=spawnSync(python,['scripts/verify-thread-completion.py',`${output}/fixtures.json`],{encoding:'utf8',timeout:120000});
process.stdout.write(run.stdout||'');process.stderr.write(run.stderr||'');
assert.equal(run.status,0,run.error?.message||'Native thread verification failed');
console.log(`PASS model exploration: ${unsafe.seen+safe.seen+opposite.seen+ordered.seen} reachable states; unsafe update and deadlock witnessed; protected invariants hold for every reachable terminal state.`);
