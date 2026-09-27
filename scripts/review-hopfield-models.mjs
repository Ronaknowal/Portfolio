import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';import * as m from '../src/learn/data/hopfield-memory-models.js';
const id='modern-hopfield-networks',dir=`docs/teaching/deep-learning-completion/${id}`,base=`public/learn-code/${id}`,read=p=>JSON.parse(fs.readFileSync(p)),clone=structuredClone,bank=read(`${base}/digit-bank.json`),fixtures=read(`${dir}/independent-native-fixtures.json`);
let checks=0,maxNative=0,maxInputGradient=0,maxProjectionGradient=0;
function near(a,b,tol=1e-10,label='value'){if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((v,i)=>near(v,b[i],tol,label));return;}checks++;assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tol,`${label}: ${a} versus ${b}`);}
const maxDiff=(a,b)=>Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-b.flat(Infinity)[i]))),sum=x=>x.reduce((a,b)=>a+b,0);
for(const r of fixtures.records){const model=read(`${base}/model-${r.model}.json`),prepared=m.prepareDigitBank(bank,model),actual=m.digitRead(r.raw,bank,model,prepared);for(const name of ['logClasses','weights','readPixels']){near(actual[name],r[name],2e-11,'new native '+name);maxNative=Math.max(maxNative,maxDiff(actual[name],r[name]));}
 const loss=(pixels,md=model,pr=prepared)=>m.dot(m.digitRead(pixels,bank,md,pr).logClasses,r.probe),h=1e-4;
 for(let j=0;j<64;j++){const up=[...r.raw],down=[...r.raw];up[j]+=h;down[j]-=h;const fd=(loss(up)-loss(down))/(2*h);maxInputGradient=Math.max(maxInputGradient,Math.abs(fd-r.inputGradient[j]));near(fd,r.inputGradient[j],2e-8,'new native input gradient');}
 if(model.projection){
  // Every matrix entry receives contributions through both query and memory.
  // Inspect a deterministic spread of entries plus the whole directional derivative.
  const indices=Array.from({length:32},(_,i)=>(i*71+29)%1024),direction=model.projection.map((row,i)=>row.map((_,j)=>Math.sin(i*64+j+1)/32));
  for(const index of indices){const row=Math.floor(index/64),col=index%64,up=clone(model),down=clone(model),eps=1e-5;up.projection[row][col]+=eps;down.projection[row][col]-=eps;const fd=(loss(r.raw,up,m.prepareDigitBank(bank,up))-loss(r.raw,down,m.prepareDigitBank(bank,down)))/(2*eps);maxProjectionGradient=Math.max(maxProjectionGradient,Math.abs(fd-r.projectionGradient[row][col]));near(fd,r.projectionGradient[row][col],3e-7,'shared projection entry');}
  const up=clone(model),down=clone(model),eps=1e-5;for(let i=0;i<16;i++)for(let j=0;j<64;j++){up.projection[i][j]+=eps*direction[i][j];down.projection[i][j]-=eps*direction[i][j];}
  near((loss(r.raw,up,m.prepareDigitBank(bank,up))-loss(r.raw,down,m.prepareDigitBank(bank,down)))/(2*eps),sum(direction.flat().map((v,i)=>v*r.projectionGradient.flat()[i])),2e-7,'whole shared projection directional derivative');
  const rotated=clone(model);rotated.projection=model.projection.map((row,i)=>model.projection[(i*5+3)%16].map(v=>v*(i%2?-1:1)));near(m.digitRead(r.raw,bank,rotated).logClasses,actual.logClasses,1e-10,'orthogonal embedding coordinates');
 }
 const reversed={...bank,memory:[...bank.memory].reverse()},duplicated={...bank,memory:[...bank.memory,...bank.memory]},relabeled={...bank,memory:bank.memory.map(x=>({...x,label:(x.label+3)%10}))};
 const rev=m.digitRead(r.raw,reversed,model),dup=m.digitRead(r.raw,duplicated,model),relabel=m.digitRead(r.raw,relabeled,model);
 near(rev.weights,[...actual.weights].reverse(),1e-12);near(rev.readPixels,actual.readPixels);near(rev.logClasses,actual.logClasses);near(dup.logClasses,actual.logClasses);near(dup.readPixels,actual.readPixels);near(dup.weights,actual.weights.concat(actual.weights).map(x=>x/2),1e-12);
 near(relabel.classes,Array.from({length:10},(_,i)=>actual.classes[(i+7)%10]));near(relabel.readPixels,actual.readPixels);near(m.digitRead(r.raw.map(x=>.37*x),bank,model,prepared).logClasses,actual.logClasses);
}
let seed=81401;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
for(let trial=0;trial<80;trial++){
 const n=1+trial%6,memories=Array.from({length:n},()=>[4*random()-2,4*random()-2]),q=[4*random()-2,4*random()-2],beta=.1+7.9*random(),a=m.continuousRead(memories,q,beta),angle=random()*6.28,c=Math.cos(angle),s=Math.sin(angle),rot=v=>[c*v[0]-s*v[1],s*v[0]+c*v[1]],b=m.continuousRead(memories.map(rot),rot(q),beta);
 near(b.weights,a.weights);near(b.read,rot(a.read));near(b.energy,a.energy);near(b.sensitivity,a.sensitivity,2e-10);
 // Scaling coordinates by c and beta by 1/c² preserves weights and scales E by c².
 const scale=1.7,scaled=m.continuousRead(memories.map(v=>v.map(x=>x*scale)),q.map(x=>x*scale),beta/scale**2);near(scaled.weights,a.weights);near(scaled.read,a.read.map(x=>x*scale));near(scaled.energy,a.energy*scale**2);
 const doubled=m.continuousRead([...memories,...memories],q,beta);near(doubled.read,a.read);near(doubled.energy,a.energy-Math.log(2)/beta);
 // Covariance must be PSD; its trace is the weighted squared spread.
 assert.ok(a.jacobian[0][0]>=-1e-11&&a.jacobian[1][1]>=-1e-11&&a.jacobian[0][0]*a.jacobian[1][1]-a.jacobian[0][1]**2>=-1e-9);checks++;
 near(a.jacobian[0][0]+a.jacobian[1][1],beta*sum(memories.map((x,i)=>a.weights[i]*m.distance(x,a.read)**2)),2e-10);
 // Signed-coordinate gauge and coordinate permutation preserve discrete dynamics.
 const patterns=Array.from({length:1+trial%3},()=>Array.from({length:4},()=>random()>.5?1:-1)),cue=Array.from({length:4},()=>random()>.5?1:-1),gauge=[1,-1,-1,1],perm=[2,0,3,1],transform=x=>perm.map(i=>x[i]*gauge[i]),order=[0,1,2,3],source=m.binaryRecall(patterns,cue,order),target=m.binaryRecall(patterns.map(transform),transform(cue),order.map(i=>perm.indexOf(i)));
 assert.equal(source.trace.length,target.trace.length);for(let i=0;i<source.trace.length;i++){near(target.trace[i].state,transform(source.trace[i].state),0);near(target.trace[i].energy,source.trace[i].energy,0);}
 const gap=.1+random()*3,count=2+trial%13,norm=1+random(),bound=m.marginBound(count,beta,gap,norm),logits=[0,...Array.from({length:count-1},()=>-beta*(gap+random()*3))],weights=m.softmax(logits);assert.ok(weights[0]>=bound.targetMass-1e-14);checks++;
}
const association=m.associate([{key:[1,0],value:[1,0]},{key:[0,1],value:[0,1]},{key:[-1,0],value:[0,1]}],[-.3,.4],2);near(association.output,[.11939846732982522,.8806015326701748],1e-10);
const author=read(`${dir}/implementation-checks.json`);for(const path of author.sourceFiles){assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex'),author.sourceHashes[path],`changed author source ${path}`);checks++;}
const report={passed:true,checks,freshNativeReads:6,inputGradientCoordinates:384,sharedProjectionFiniteDifferences:128,fullProjectionDirections:4,seed:81401,geometryAndGaugeCases:80,maxNativeError:maxNative,maxInputGradientError:maxInputGradient,maxProjectionEntryGradientError:maxProjectionGradient,sourceFilesVerified:author.sourceFiles.length,scope:'Complementary native queries/gradients, both shared projection paths, whole-bank duplication, semantic class/memory permutations, coordinate rotation/scaling, covariance PSD and binary gauge dynamics. Existing 502047 author comparisons reused; no fitting or browser checks.'};fs.writeFileSync(`${dir}/independent-model-checks.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
