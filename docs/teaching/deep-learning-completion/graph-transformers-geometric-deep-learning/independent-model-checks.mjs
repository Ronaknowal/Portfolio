import fs from 'node:fs';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import * as m from '../../../../src/learn/data/geometric-graph-models.js';
const folder=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(folder,'../../../..'),id='graph-transformers-geometric-deep-learning';
const data=JSON.parse(fs.readFileSync(path.join(folder,'independent-native-fixtures.json'),'utf8'));
let count=0;const groups={};let category='';
function near(a,b,tol=4e-11){if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((v,i)=>near(v,b[i],tol));return;}assert.ok(Number.isFinite(a)&&Number.isFinite(b),`${category}: finite ${a}/${b}`);const e=Math.abs(a-b);groups[category]??={comparisons:0,maxError:0};groups[category].comparisons++;groups[category].maxError=Math.max(groups[category].maxError,e);count++;assert.ok(e<=tol,`${category}: ${a} / ${b}, error ${e}, tolerance ${tol}`);}
const clone=x=>structuredClone(x),loss=(a,b)=>a.flat().reduce((s,v,i)=>s+v*b.flat()[i],0);
category='functional-native-full-network';
for(const f of data.fitted){
 const model=JSON.parse(fs.readFileSync(path.join(root,'public/learn-assets',id,`${f.kind}-11.json`))),r=m.inferGraphTransformer(model,f.features,f.adjacency);
 near(r.logits,f.logits);near(r.probabilities,f.probabilities);
 if(f.weights)r.trace.forEach((t,i)=>near(t.weights,f.weights[i]));
 const p=f.permutation;near(m.inferGraphTransformer(model,p.map(i=>f.features[i]),p.map(i=>p.map(j=>f.adjacency[i][j]))).logits,f.permutedLogits);
 category='network-input-gradient-central-difference';
 for(const [i,c] of [[0,0],[3,1],[11,2],[27,0],[33,2]]){const h=1e-5,a=clone(f.features),b=clone(f.features);a[i][c]+=h;b[i][c]-=h;const fd=(loss(m.inferGraphTransformer(model,a,f.adjacency).logits,f.probe)-loss(m.inferGraphTransformer(model,b,f.adjacency).logits,f.probe))/(2*h);near(fd,f.gradient[i][c],3e-7);}
 category='functional-native-full-network';
}
category='native-PyG-new-directed-and-empty-graphs';
for(const f of data.gps){
 near(m.gpsForward(f.features,f.edges,f.batch,f.state).output,f.output);
 const changed=f.features.map((r,i)=>r.map(v=>i>=5?v+1.4:v));near(m.gpsForward(changed,f.edges,f.batch,f.state).output,f.changed);
 category='native-PyG-input-gradient';
 for(let i=0;i<7;i++)for(let c=0;c<4;c++){const a=clone(f.features),b=clone(f.features),h=1e-5;a[i][c]+=h;b[i][c]-=h;near((loss(m.gpsForward(a,f.edges,f.batch,f.state).output,f.probe)-loss(m.gpsForward(b,f.edges,f.batch,f.state).output,f.probe))/(2*h),f.gradient[i][c],3e-8);}
 category='native-PyG-new-directed-and-empty-graphs';
}
category='all-64-four-node-graph-eigenspaces';
for(const f of data.eigen){const l=m.laplacian(f.a),r=m.eigenSymmetric(l);near(r.values,f.values);near(m.multiply(m.transpose(r.vectors),r.vectors),m.identity(4));near(m.multiply(l,r.vectors),r.vectors.map(row=>row.map((v,j)=>v*r.values[j])));for(const group of f.groups){const u=r.vectors.map(row=>group.indices.map(i=>row[i]));near(m.multiply(u,m.transpose(u)),group.projector);}}
category='native-energy-force-geometry';
for(const f of data.geometry){const r=m.energyForce(f.points);near(r.energy,f.energy);near(r.forces,f.forces);near(r.total,[0,0,0]);near(m.geometricUpdate(f.points,f.step).output,f.updated);near(m.geometryContract(f.points,f.q,f.t,f.step).error,0);const changed=m.energyForce(m.transformPoints(f.points,f.q,f.t));near(changed.energy,r.energy);near(changed.forces,r.forces.map(v=>m.matvec(f.q,v)));}
category='SciPy-erf-and-normal-CDF';for(const f of data.special){near(m.erf(f.x),f.erf,5e-15);near(m.gelu(f.x),f.gelu,2e-14);}
category='chirality-role-scale-and-orthogonal-parity';
const p=[[.2,-.4,.8],[1.4,.3,-.2],[-.6,.8,.1],[.7,-.9,1.9]],base=m.chirality(p);
for(const f of data.geometry){near(m.chirality(m.transformPoints(p,f.q,f.t)),m.determinant(f.q)*base);for(const scale of [-1.8,0,.25,2.4])near(m.chirality(p.map(r=>r.map(v=>v*scale))),base*scale**3);const changed=clone(p);[changed[1],changed[2]]=[changed[2],changed[1]];near(m.chirality(changed),-base);}
category='safe-exact-budget-boundary';
for(const nodes of [1,273,19999,20000])for(const heads of [1,7,16])for(const bytes of [2,4])for(const batch of [1,3,4]){const r=m.attentionBudget({nodes,heads,bytes,batch});assert.equal(BigInt(r.bytes),BigInt(nodes)**2n*BigInt(heads)*BigInt(bytes)*BigInt(batch));near(r.pairs,r.bytes/bytes,0);}
const out={passed:true,scalarComparisons:count,groups,nativeVersions:data.versions,limits:['New inputs and parameter state only; no repeated graph fitting or additional dataset claim.','Central differences verify selected full-network gradients and every new GPS input gradient; they are not an exhaustive parameter-gradient proof.','Browser review remains root-owned.']};
fs.writeFileSync(path.join(folder,'independent-numerical-checks.json'),JSON.stringify(out,null,2)+'\n');console.log(JSON.stringify(out,null,2));
