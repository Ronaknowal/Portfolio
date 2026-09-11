const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const directory = 'scratch/random-variables-independent';
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/random-variables-author-review.json'));
const archive = path.join(directory, 'author-sources');
fs.mkdirSync(archive, { recursive:true });
const records=[];
for(const source of author.sources) {
  const bytes=fs.readFileSync(source.path);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),source.sha256);
  const target=path.join(archive,path.basename(source.path));
  if(fs.existsSync(target)) assert.deepEqual(fs.readFileSync(target),bytes); else fs.writeFileSync(target,bytes);
  records.push({...source,archive:target.replaceAll('\\','/')});
}
const record=path.join(directory,'author-baseline.json');
assert(!fs.existsSync(record),'Author baseline already exists; preserve it.');
fs.writeFileSync(record,JSON.stringify({at:new Date().toISOString(),sources:records,findings:[
  'finiteMoments([,1],[.5,.5]) returned NaN moments rather than rejecting a sparse array.',
  'finiteMoments([0,1],[,1]) returned NaN moments; pairedMoments([,{x:1,y:2,mass:1}]) accepted a missing row.',
  'Integrated LessonIntro lacks static hasIntegratedGuide metadata, so TopicContent inserts the legacy guide.',
  'The first optional Python program lacks a preceding save/run instruction; add a concise local setup.'
]},null,2));
console.log('Preserved all six author sources before independent amendments.');
