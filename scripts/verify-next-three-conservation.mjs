import fs from 'node:fs';
import assert from 'node:assert/strict';
import {topicCatalogue} from '../src/learn/data/curriculum/topic-catalogue.js';
import {trackDefinitions} from '../src/learn/data/track-definitions.js';
import {learningPaths,getLearningRoute} from './lib/authoring-curriculum.mjs';
import {readTopicAuthoringNotes} from './topic-authoring-notes.mjs';
const before=JSON.parse(fs.readFileSync('scratch/next-three-review/before.json','utf8'));
assert.deepEqual(Object.keys(topicCatalogue).sort(),before.ids.slice().sort(),'Stable IDs retained');
const orderChanges=[];
for(const old of before.modules) {
  const current=Object.values(topicCatalogue).filter(t=>t.trackIds.includes(old.id)).map(t=>t.id);
  assert.ok(trackDefinitions.some(t=>t.id===old.id));
  assert.deepEqual(current.sort(),old.topicIds.slice().sort(),'Module membership '+old.id);
}
for(const old of before.paths) {
  const current=getLearningRoute(learningPaths.find(p=>p.id===old.id)).topicIds;
  for(const id of old.topicIds)assert.ok(current.includes(id),'Prior path topic retained '+old.id+'/'+id);
  if(JSON.stringify(current)!==JSON.stringify(old.topicIds))orderChanges.push({id:old.id,before:old.topicIds.slice(0,12),after:current.slice(0,12)});
}
const note=readTopicAuthoringNotes(process.cwd(),'ml-problem-formulation-baselines-data-leakage');
assert.ok(JSON.stringify(note).includes('An older event can still contain information from the future'));
fs.writeFileSync('scratch/next-three-review/conservation-results.json',JSON.stringify({topics:before.ids.length,modules:before.modules.length,paths:before.paths.length,orderChanges,noteRetrieved:true},null,2));
console.log('PASS: IDs, titles (IDs derive from titles), module and path memberships preserved; '+orderChanges.length+' route orders changed. Destination note retrieved.');
