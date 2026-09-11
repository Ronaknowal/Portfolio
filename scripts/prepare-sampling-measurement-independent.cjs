const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/sampling-measurement-independent';
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/sampling-measurement-author-review.json','utf8'));
fs.mkdirSync(`${directory}/author-sources`,{recursive:true});
assert(!fs.existsSync(`${directory}/author-baseline.json`),'Do not overwrite the author baseline.');
const sources=author.production.map(source=>{
  const bytes=fs.readFileSync(source.path);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),source.sha256,source.path);
  const archive=`${directory}/author-sources/${path.basename(source.path)}`;
  fs.writeFileSync(archive,bytes,{flag:'wx'});
  return {...source,archive};
});
fs.writeFileSync(`${directory}/author-baseline.json`,JSON.stringify({at:new Date().toISOString(),sources,findings:[
  'A sparse finite distribution silently omits a value; sparse probabilities can return NaN.',
  'A tolerated non-unit mass sum produces a spurious variance for a constant 1e9-valued law.',
  'Reject inherited preset names explicitly and require explicit null for missing observations.'
]},null,2)+'\n');
console.log('Preserved all six Sampling author sources before independent amendments.');
