import assert from 'node:assert/strict';
import fs from 'node:fs';
import {tracks} from './lib/authoring-curriculum.mjs';
import {learningPaths,getLearningRoute} from './lib/authoring-curriculum.mjs';
import {topicCatalogue} from '../src/learn/data/curriculum/topic-catalogue.js';
const before=JSON.parse(fs.readFileSync(new URL('../docs/curriculum/pre-programming-module-completion.json',import.meta.url),'utf8'));
assert.deepEqual(Object.keys(topicCatalogue).sort(),before.ids.slice().sort(),'Topic identity changed');
const changed=[];
for(const previous of before.modules) {
 const current=tracks.find(t=>t.id===previous.id);assert.ok(current,'Module removed');
 assert.deepEqual(current.topicIds.slice().sort(),previous.topicIds.slice().sort(),`Membership changed: ${previous.id}`);
 if(JSON.stringify(current.topicIds)!==JSON.stringify(previous.topicIds))changed.push(previous.id);
}
assert.deepEqual(changed,['programming-scientific-computing']);
const added={};
for(const previous of before.paths) {
 const current=getLearningRoute(learningPaths.find(p=>p.id===previous.id));
 for(const id of previous.topicIds)assert.ok(current.topicIds.includes(id),`Path topic removed: ${previous.id}/${id}`);
 added[previous.id]=current.topicIds.filter(id=>!previous.topicIds.includes(id));
}
assert.deepEqual(added,{
 'ml-foundations':[],
 'llm-engineer':['iterators-iterables-generators','decorators-context-managers'],
 'embodied-intelligence':[],
 'neural-engineer':[],
 'gpu-engineer':['object-oriented-programming-in-python','iterators-iterables-generators','decorators-context-managers'],
 'research-revision':[],
 'full-curriculum':[],
});
const programming=tracks.find(t=>t.id==='programming-scientific-computing');
assert.deepEqual(programming.topicIds,[
 'python-basics-types-control-flow-functions-modules','object-oriented-programming-in-python',
 'iterators-iterables-generators','decorators-context-managers','testing-debugging-dependency-management',
 'numpy-arrays-broadcasting-vectorization','scientific-file-formats-schemas-reliable-data-i-o',
 'sql-relational-data-transactions-for-ml','pandas-data-wrangling-joins-grouping','matplotlib-scientific-plotting',
 'reproducible-notebooks-experiment-structure','code-documentation-type-hints-api-design',
 'git-github-collaborative-version-control','linux-basics-filesystems-processes','bash-scripting-command-line-automation',
 'os-processes-virtual-memory-isolation','threads-concurrency-locks-deadlocks',
]);
const inventory=JSON.parse(fs.readFileSync(new URL('../docs/curriculum/curriculum-inventory.json',import.meta.url),'utf8'));
for(const id of programming.topicIds) {
 const entry=inventory.topics.find(t=>t.id===id);
 assert.equal(entry.publicationStatus,'published');assert.ok(entry.blueprint);
 assert.ok(['user-approved-reference','implementation-reviewed-user-acceptance-pending'].includes(entry.teachingReview));
}
console.log('PASS: all 1218 IDs and 28 module memberships conserved; no prior path topic lost. Only programming syllabus reordered; LLM adds two and GPU adds three reviewed Python prerequisites. All 17 programming lessons published with individual briefs and explicit review status.');
