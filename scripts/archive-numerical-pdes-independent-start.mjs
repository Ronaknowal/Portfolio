import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const authorPath='docs/teaching/evidence/numerical-pdes-author-review.json';
const bytes=fs.readFileSync(authorPath);const author=JSON.parse(bytes);
const directory='docs/teaching/archive/numerical-pdes-independent-start';
assert(!fs.existsSync(directory));fs.mkdirSync(directory,{recursive:true});
const digest=value=>crypto.createHash('sha256').update(value).digest('hex');
const sourceHashes=[];
for(const [path,sha256] of Object.entries(author.sourceHashes)){
 const content=fs.readFileSync(path);assert.equal(digest(content),sha256);
 const archive=`${directory}/${path.replaceAll('/','__')}.txt`;
 fs.writeFileSync(archive,content);sourceHashes.push({path,sha256,archive});
}
fs.writeFileSync(`${directory}/author-review.json`,bytes);
fs.mkdirSync('scratch/numerical-pdes-independent-review',{recursive:true});
fs.writeFileSync('scratch/numerical-pdes-independent-review/start.json',JSON.stringify({startedAt:new Date().toISOString(),authorPath,authorSha256:digest(bytes),sourceHashes},null,2)+'\n');
console.log('Matched and archived all six author-frozen numerical PDE sources.');
