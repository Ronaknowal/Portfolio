import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
const path='src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx';
const packet='docs/teaching/evidence/dp-state-families-author-review.json';
const archive='docs/teaching/archive/dynamic-programming-before-state-families/author-frozen-body-before-hints.txt';
const previousPacket='docs/teaching/evidence/dp-state-families-author-review-before-hints.json';
assert(!fs.existsSync(archive)&&!fs.existsSync(previousPacket));
const original=fs.readFileSync(path,'utf8');
const originalPacket=fs.readFileSync(packet);
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');
assert.equal(hash(original),JSON.parse(originalPacket).sourceHashes.find(source=>source.path===path).sha256);
fs.writeFileSync(archive,original);fs.writeFileSync(previousPacket,originalPacket);
const cases=[
 ['Change the dimensions to [3,7,2,5]', 'Write the shape of each intermediate first, then the two scalar-multiplication totals. With equal dimensions, does changing the split change either multiplication cost?'],
 ['Why would reading the split tree root-first fail?', 'Replay the proposed first removal using its actual live neighbors. For the signed singleton, list the legal complete removal plans before choosing an initial maximum.'],
 ['Use a star with weights [4,3,3,3]', 'Compare taking the center with taking its leaves. After adding the new edge, test whether the separately optimal child selections can still be combined legally.'],
 ['Change a two-node chain to weights [−2,−5]', 'What choice does an all-zero answer represent? Name one extra distinction that separates an empty selection from a legal nonempty selection while combining children.'],
 ['For bound 213, explain why tight=false', 'Compare the prefixes 12 and 21 before placing the final digit. For the range count, group candidates by their first two actual digits and reject a repeated digit.'],
];
let updated=original;
for(const [start,hint] of cases){
 const marker=`<Checkpoint prompt="${start}`;
 const index=updated.indexOf(marker);assert(index>=0);
 const end=updated.indexOf('</Checkpoint>',index);assert(end>=0);
 const block=updated.slice(index,end+'</Checkpoint>'.length);
 const replacement=block.replace('<Checkpoint ',`<StateFamilyCheckpoint hint="${hint}" `).replace('</Checkpoint>','</StateFamilyCheckpoint>');
 updated=updated.slice(0,index)+replacement+updated.slice(end+'</Checkpoint>'.length);
}
const helper=`function StateFamilyCheckpoint({ prompt, hint, children }) {
  return <div className="lesson-check" data-dpf-checkpoint>
    <p><strong>Try it first.</strong> {prompt}</p>
    <details><summary>Optional hint</summary><Prose>{hint}</Prose></details>
    <details><summary>Show explanation</summary><div>{children}</div></details>
  </div>;
}

`;
assert.equal(updated.split('export default {').length,2);
updated=updated.replace('export default {',helper+'export default {');
fs.writeFileSync(path,updated);
fs.writeFileSync('docs/teaching/evidence/dp-state-families-hint-amendment.json',JSON.stringify({amendedAt:new Date().toISOString(),path,archive,previousPacket,beforeSha256:hash(original),afterSha256:hash(updated),changedCheckpoints:cases.map(([promptStart,hint])=>({promptStart,hint})),reason:'Independent reviewer found five new tasks lacked the separately disclosed hints promised by the approved design. The question now precedes an optional hint and a separate hidden explanation; original checkpoints and all numerical sources are unchanged.'},null,2)+'\n');
