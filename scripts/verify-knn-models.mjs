import fs from 'node:fs';
import assert from 'node:assert/strict';
import { neighborRows, neighborReport, metricDistance, localMean, unitsReport, kdSearch, candidateReport, volumeReport } from '../src/learn/data/knn-models.js';
const payload = { neighbors: [], regression: [], searches: [], units: [], candidates: [], volumes: [] };
for (const query of [[3.1,2.9],[0,0],[6,6],[1,2],[3,3],[4,1]]) {
  for (const k of [1,2,3,5,8]) for (const metric of ['euclidean','manhattan','maximum']) for (const weights of ['uniform','distance']) {
    const result = neighborReport({ query,k,metric,weights });
    assert.equal(result.neighbors.length,k);
    assert(Math.abs(result.votes.reduce((sum,row)=>sum+row.probability,0)-1)<1e-12);
    payload.neighbors.push({query,k,metric,weights,result});
  }
}
for (let query=-1;query<=8;query+=.5) for (const k of [1,2,3,6]) for (const weights of ['uniform','distance']) {
  const result=localMean(query,k,weights);
  assert(result.prediction>=0 && result.prediction<=25);
  payload.regression.push({query,k,weights,result});
}
assert(Number.isFinite(localMean(1e-320,3,'distance').prediction));
for(let x=0;x<=6;x+=.25) for(let y=0;y<=6;y+=.25) {
  const query=[x,y]; const result=kdSearch(query);
  const distance=Math.min(...neighborRows.map(row=>metricDistance(query,row.point)));
  assert(Math.abs(result.best.distance-distance)<1e-12);
  payload.searches.push({query,result});
}
payload.units=[false,true].map(scale=>({scale,result:unitsReport(scale)}));
payload.candidates=['all','lose-close-c','keep-close-c'].map(mode=>({mode,result:candidateReport(mode)}));
for(const dimension of [1,2,5,10,50,100])for(const fraction of [.01,.1,.5])payload.volumes.push({dimension,fraction,result:volumeReport(dimension,fraction)});
for(const run of [()=>neighborReport({k:0}),()=>neighborReport({k:9}),()=>neighborReport({query:[NaN,1]}),()=>metricDistance([0,0],[1,2],'cosine'),()=>localMean(9),()=>volumeReport(0)]) assert.throws(run,RangeError);
fs.mkdirSync('scratch/knn-native',{recursive:true});
fs.writeFileSync('scratch/knn-native/model-payload.json',JSON.stringify(payload));
console.log('KNN model invariants and bounded grids passed; independent-oracle payload saved.');
