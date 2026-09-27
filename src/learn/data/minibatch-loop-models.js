const sum=a=>a.reduce((s,v)=>s+v,0);
const mean=a=>sum(a)/a.length;
export function rowPartitions(rows,boundaries){const groups=[[]];rows.forEach((_,i)=>{groups.at(-1).push(i);if(i<rows.length-1&&boundaries[i])groups.push([]);});return groups;}
export function scalarLoopTrace(rows,groups,{rate=.1,momentum=0,policy='correct',initial=0}={}){
 let weight=initial,gradient=null,buffer=null,examples=0,backwards=0,updates=0;const events=[];
 const record=(operation,group=null,contribution=null,forward=null)=>events.push({operation,group,contribution,forward,weight,gradient,buffer,examples,backwards,updates});
 record('initial');record('clear');
 const step=()=>{buffer=momentum?momentum*(buffer??0)+(gradient??0):null;weight-=rate*(momentum?buffer:gradient??0);updates++;record('optimizer step');gradient=null;record('clear');};
 for(const [groupIndex,group] of groups.entries()){
  if(policy==='clear_each'&&groupIndex){gradient=null;record('clear before chunk',groupIndex);}
  const forward=group.map(i=>{const prediction=weight*rows[i].x,residual=prediction-rows[i].y;return {index:i,prediction,residual,loss:.5*residual**2,derivative:residual*rows[i].x};});examples+=group.length;record('forward + loss',groupIndex,null,forward);
  const contribution=sum(forward.map(r=>r.derivative))/rows.length;gradient=(gradient??0)+contribution;backwards++;record('backward',groupIndex,contribution,forward);
  if(policy==='step_each')step();
 }
 if(policy!=='step_each')step();
 return {events,weight,buffer,updates,backwards,examples,postLoss:mean(rows.map(r=>.5*(weight*r.x-r.y)**2))};
}
export function weightedBatchGradient(rows,mode='mass'){
 const groups=[...new Set(rows.map(r=>r.group))].sort((a,b)=>a-b).map(id=>{const indices=rows.map((r,i)=>r.group===id?i:-1).filter(i=>i>=0),mass=sum(indices.map(i=>rows[i].included?rows[i].weight:0)),numerator=sum(indices.map(i=>rows[i].included?-rows[i].weight*rows[i].x*rows[i].y:0));return {id,indices,mass,numerator};});
 const mass=sum(groups.map(g=>g.mass)),numerator=sum(groups.map(g=>g.numerator)),active=groups.filter(g=>g.mass>0),eligible=rows.filter(r=>r.included).length,reference=mass>0?numerator/mass:null;
 const denominator=mode==='mass'?mass:mode==='eligible'?eligible:mode==='slots'?rows.length:null,compared=mode==='chunks'?(active.length?mean(active.map(g=>g.numerator/g.mass)):null):(denominator>0?numerator/denominator:null);
 const details=rows.map(r=>{const q=r.included?r.weight:0,g=groups.find(g=>g.id===r.group);return {mass:q,derivative:-r.x*r.y,numerator:-q*r.x*r.y,referenceCoefficient:mass>0?q/mass:null,comparedCoefficient:mode==='chunks'?(active.length?(g.mass>0?q/(g.mass*active.length):0):null):(denominator>0?q/denominator:null)};});
 return {groups,details,mass,numerator,eligible,reference,compared,defined:reference!==null&&compared!==null,matches:reference!==null&&compared!==null&&Math.abs(reference-compared)<=1e-9};
}
export function normalizationBatchComparison(rows,theta=1,mode='local',frozen={mean:5,variance:10}){
 const x=rows.map(r=>r.x),fullMean=mean(x),fullVariance=mean(x.map(v=>(v-fullMean)**2)),groups=[...new Set(rows.map(r=>r.group))].sort((a,b)=>a-b).map(id=>({id,indices:rows.map((r,i)=>r.group===id?i:-1).filter(i=>i>=0)}));
 if(mode==='local'&&groups.some(g=>g.indices.length<2))return {valid:false,reason:`Group ${groups.find(g=>g.indices.length<2).id+1} has one row. Local training normalization needs at least two rows per physical group.`,fullMean,fullVariance,groups};
 const normalize=(v,m,vv)=>(v-m)/Math.sqrt(vv+1e-5),fullZ=x.map(v=>mode==='none'?v:normalize(v,mode==='frozen'?frozen.mean:fullMean,mode==='frozen'?frozen.variance:fullVariance)),localZ=Array(rows.length),states=[];let running=0;
 for(const g of groups){const values=g.indices.map(i=>x[i]),m=mean(values),variance=mean(values.map(v=>(v-m)**2));if(mode==='local')running=.9*running+.1*m;g.indices.forEach(i=>localZ[i]=mode==='none'?x[i]:normalize(x[i],mode==='frozen'?frozen.mean:m,mode==='frozen'?frozen.variance:variance));states.push({...g,mean:m,variance,running});}
 const summarize=z=>({z,predictions:z.map(v=>theta*v),residuals:z.map((v,i)=>theta*v-rows[i].y),rowGradients:z.map((v,i)=>(theta*v-rows[i].y)*v),loss:mean(z.map((v,i)=>.5*(theta*v-rows[i].y)**2)),gradient:mean(z.map((v,i)=>(theta*v-rows[i].y)*v))});
 return {valid:true,fullMean,fullVariance,groups:states,full:summarize(fullZ),local:summarize(localZ),fullRunningMean:mode==='local'?.1*fullMean:0,localRunningMean:running};
}
const dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
export function irisRowGradient(parameters,raw,target,center,scale){
 const x=raw.map((v,i)=>(v-center[i])/scale[i]),pre=parameters['0.weight'].map((r,i)=>dot(r,x)+parameters['0.bias'][i]),hidden=pre.map(Math.tanh),logits=parameters['2.weight'].map((r,i)=>dot(r,hidden)+parameters['2.bias'][i]),largest=Math.max(...logits),exp=logits.map(v=>Math.exp(v-largest)),normalizer=sum(exp),probabilities=exp.map(v=>v/normalizer),delta=probabilities.map((v,i)=>v-Number(i===target)),hiddenDelta=hidden.map((v,j)=>(1-v*v)*sum(delta.map((d,i)=>d*parameters['2.weight'][i][j])));
 const gradient={'0.weight':hiddenDelta.map(d=>x.map(v=>d*v)),'0.bias':hiddenDelta,'2.weight':delta.map(d=>hidden.map(v=>d*v)),'2.bias':delta};return {x,hidden,logits,probabilities,loss:Math.log(normalizer)+largest-logits[target],gradient};
}
const mapTree=(v,fn)=>Array.isArray(v)?v.map(x=>mapTree(x,fn)):fn(v);
const zipTree=(a,b,fn)=>Array.isArray(a)?a.map((v,i)=>zipTree(v,b[i],fn)):fn(a,b);
export function irisGroupAccumulation(snapshot,rows,microbatch=12,{rate=.05,momentum=.9}={}){
 const results=rows.map(row=>irisRowGradient(snapshot.parameters,row.features,row.target,snapshot.center,snapshot.scale)),keys=Object.keys(snapshot.parameters),empty=()=>Object.fromEntries(keys.map(k=>[k,mapTree(snapshot.parameters[k],()=>0)])),add=(dest,source)=>keys.forEach(k=>{dest[k]=zipTree(dest[k],source[k],(a,b)=>a+b);}),full=empty(),accumulated=empty(),chunks=[];
 results.forEach(r=>add(full,Object.fromEntries(keys.map(k=>[k,mapTree(r.gradient[k],v=>v/rows.length)]))));
 for(let start=0;start<rows.length;start+=microbatch){const partial=empty(),size=Math.min(microbatch,rows.length-start);for(const r of results.slice(start,start+size))add(partial,r.gradient);keys.forEach(k=>partial[k]=mapTree(partial[k],v=>v/rows.length));add(accumulated,partial);chunks.push({start,size,gradient:partial});}
 const parameters={},buffer={};for(const k of keys){buffer[k]=zipTree(snapshot.momentum?.[k]??mapTree(snapshot.parameters[k],()=>0),accumulated[k],(m,g)=>momentum*m+g);parameters[k]=zipTree(snapshot.parameters[k],buffer[k],(w,v)=>w-rate*v);}
 const gap=Math.max(...keys.flatMap(k=>full[k].flat(Infinity).map((v,i)=>Math.abs(v-accumulated[k].flat(Infinity)[i]))));return {rows:results,full,accumulated,chunks,parameters,buffer,gap,loss:mean(results.map(r=>r.loss))};
}
