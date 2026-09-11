// A bounded complementary pass: structural invariants and changed actual helper calls.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as trees from '../src/learn/data/decision-tree-models.js';
import * as knn from '../src/learn/data/knn-models.js';
import { decisionTreeExamples } from '../src/learn/data/decision-tree-examples.js';
import { knnExamples } from '../src/learn/data/knn-examples.js';
const directory = 'scratch/tree-knn-independent';
fs.mkdirSync(directory, { recursive: true });
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const counts = { affineTreeQueries: 0, pruningCrossovers: 0, bootstrapPrefixes: 0, geometricIdentities: 0, tinyCosineCases: 0, rejectedSparse: 0 };
const close = (a, b) => assert(Math.abs(a-b) <= 1e-11*Math.max(1,Math.abs(b)), `${a} != ${b}`);
const transformed = trees.inspectionRows.map(row => ({...row, features:[2*row.features[0]+8,4*row.features[1]-8]}));
for (const depth of [0,1,2,3]) {
  const original = trees.growTree(trees.inspectionRows,{maxDepth:depth});
  const changed = trees.growTree(transformed,{maxDepth:depth});
  for(const query of [[0,0],[1,1],[2.5,2],[4.5,3],[5,3],[8,6]]) {
    const first=trees.treePrediction(original,query),second=trees.treePrediction(changed,[2*query[0]+8,4*query[1]-8]);
    close(first.probability,second.probability); assert.equal(first.leafPath,second.leafPath);
    counts.affineTreeQueries++;
  }
}
for(const [alpha,leaves] of [[15/128-1e-8,5],[15/128,1],[15/128+1e-8,1]]) {
  const report=trees.pruningReport(alpha); assert.equal(report.best.leaves,leaves);
  for(const candidate of report.candidates) assert(report.best.objective<=candidate.objective+1e-13);
  counts.pruningCrossovers++;
}
for(const row of [0,3,7])for(const featureSampling of [true,false]) {
  const small=trees.forestReport({trees:3,selectedRow:row,featureSampling});
  const large=trees.forestReport({trees:9,selectedRow:row,featureSampling});
  assert.deepEqual(small.members,large.members.slice(0,3));
  const allowed=large.members.filter(member=>!member.sampleIndices.includes(row));
  assert.equal(large.oobCount,allowed.length);
  if(allowed.length) close(large.oobProbability,allowed.reduce((sum,m)=>sum+m.prediction.probability,0)/allowed.length);
  else assert.equal(large.oobProbability,null);
  counts.bootstrapPrefixes++;
}
for(const [x,z] of [[[1,2],[4,6]],[[-2,5],[3,-1]],[[1,.1],[-1,3]],[[3,4],[6,8]]]) {
  close(knn.metricDistance(x,z),knn.metricDistance([-x[1],x[0]],[-z[1],z[0]]));
  close(knn.metricDistance(x,z),Math.sqrt((x[0]-z[0])**2+(x[1]-z[1])**2));
  const xn=x.map(v=>v/Math.hypot(...x)),zn=z.map(v=>v/Math.hypot(...z));
  close(knn.metricDistance(x,z,'cosine'),.5*knn.metricDistance(xn,zn)**2);
  counts.geometricIdentities++;
}
for (const scale of [1e-200, 1e-160, 1e-150]) {
  close(knn.metricDistance([scale, 0], [scale, 0], 'cosine'), 0);
  close(knn.metricDistance([scale, 0], [scale, scale], 'cosine'), 1 - 1 / Math.sqrt(2));
  counts.tinyCosineCases += 2;
}
const rejected=[
  ()=>trees.impurity(Array(2)),
  ()=>trees.treePrediction(trees.growTree(),Array(2)),
  ()=>trees.growTree([{id:'broken',features:Array(2),label:0}]),
  ()=>trees.growTree(Array(2)),
  ()=>knn.metricDistance(Array(2),[1,2]),
  ()=>knn.neighborReport({query:Array(2)}),
  ()=>knn.neighborReport({candidates:Array(2),k:1}),
];
for(const run of rejected){assert.throws(run,RangeError);counts.rejectedSparse++;}
fs.writeFileSync(`${directory}/native-input.json`,JSON.stringify({tree:decisionTreeExamples.find(e=>e.id==='scratch-tree-forest'),knn:knnExamples.filter(e=>['scratch-estimator','kd-tree'].includes(e.id))}));
const run=spawnSync('scratch/lesson-tools/Scripts/python.exe',['scripts/verify-classical-tree-knn-independent.py'],{encoding:'utf8',timeout:30000});
process.stdout.write(run.stdout||'');process.stderr.write(run.stderr||'');assert.equal(run.status,0);
const sourceHashes={};
for(const packet of ['decision-tree','knn']) {
  const author=JSON.parse(fs.readFileSync(`docs/teaching/evidence/${packet}-author-review.json`));
  sourceHashes[packet]=author.sourceHashes.map(item=>({path:item.path,sha256:hash(item.path)}));
}
const result={checkedAt:new Date().toISOString(),counts,native:JSON.parse(fs.readFileSync(`${directory}/native-results.json`)),sourceHashes,scope:'Complementary structural/geometry and changed actual helper checks. Author complete program/browser evidence is reused, not rerun.'};
fs.writeFileSync(`${directory}/results.json`,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({checkedAt:result.checkedAt,counts}));
