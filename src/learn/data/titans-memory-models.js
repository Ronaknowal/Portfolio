// Scalar-output and two-coordinate teaching memories. Every call owns its state.
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const mapTree=(a,fn)=>Array.isArray(a)?a.map(v=>mapTree(v,fn)):fn(a);
const zipTree=(a,b,fn)=>Array.isArray(a)?a.map((v,i)=>zipTree(v,b[i],fn)):fn(a,b);
const flatten=a=>a.flat(Infinity);
export const zeroMemory=a=>mapTree(a,()=>0);
export function linearMemoryWrite(weight,momentum,key,value,{rate=.5,retention=0,decay=0}={}){
 const before=weight.map(row=>dot(row,key)),residual=before.map((v,i)=>v-value[i]);
 const gradient=residual.map(v=>key.map(k=>v*k)),nextMomentum=zipTree(momentum,gradient,(m,g)=>retention*m-rate*g),nextWeight=zipTree(weight,nextMomentum,(w,s)=>(1-decay)*w+s);
 return {before,residual,loss:.5*dot(residual,residual),gradient,momentum:nextMomentum,weight:nextWeight,update:zipTree(nextWeight,weight,(a,b)=>a-b),gradientNorm:Math.hypot(...flatten(gradient)),updateNorm:Math.hypot(...flatten(zipTree(nextWeight,weight,(a,b)=>a-b)))};
}
export function associativeMemoryTrace(cards,events,query,settings={},initial=[[0,0]],initialMomentum=[[0,0]]){
 let weight=structuredClone(initial),momentum=structuredClone(initialMomentum);const records=[];
 for(const cardIndex of events){const card=cards[cardIndex],oldWeight=weight,oldMomentum=momentum,r=linearMemoryWrite(weight,momentum,card.key,[card.value],settings);weight=r.weight;momentum=r.momentum;records.push({...r,cardIndex,oldWeight,oldMomentum,read:dot(weight[0],query)});}
 return {records,weight,momentum,read:dot(weight[0],query)};
}
export function gatedMemorySequence(tokens,{prefix=true,rate=.5,retention=.5,decay=0,state=null}={}){
 let weight=state?structuredClone(state.weight):[[0,0],[0,0]],momentum=state?structuredClone(state.momentum):[[0,0],[0,0]],recent=state?structuredClone(state.recent):[];const records=[];
 for(const token of tokens){recent=[...recent,[...token]].slice(-2);const context=[...(prefix?[[1,0]]:[]),...recent],scores=context.map(row=>dot(row,token)/Math.sqrt(2)),largest=Math.max(...scores),exp=scores.map(v=>Math.exp(v-largest)),sum=exp.reduce((a,b)=>a+b,0),attention=exp.map(v=>v/sum),short=[0,1].map(j=>context.reduce((s,row,i)=>s+row[j]*attention[i],0));
  const write=linearMemoryWrite(weight,momentum,token,token,{rate,retention,decay});weight=write.weight;momentum=write.momentum;const long=weight.map(row=>dot(row,token)),gate=long.map(Math.tanh),output=short.map((v,i)=>v*gate[i]);records.push({input:[...token],context,attention,scores,short,weight,momentum,long,gate,output,write});
 }
 return {records,state:{weight,momentum,recent},outputs:records.map(r=>r.output)};
}
export function chunkMemoryTrace(targets=[1,2],initial=0,rate=.5,chunkSize=2,anchored=false){
 let weight=initial,anchor=initial;return targets.map((target,i)=>{if(i%chunkSize===0)anchor=weight;const evaluation=anchored?anchor:weight,gradient=evaluation-target,before=weight;weight-=rate*gradient;return {target,before,anchor,evaluation,gradient,weight};});
}
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
export function neuralMemoryRead(parameters,key){
 const[W1,b1,W2,b2]=parameters,pre=W1.map((row,i)=>dot(row,key)+b1[i]),sig=pre.map(sigmoid),hidden=pre.map((v,i)=>v*sig[i]),output=dot(W2[0],hidden)+b2[0];
 const hiddenDerivative=pre.map((v,i)=>sig[i]*(1+v*(1-sig[i]))),sensitivity=W2[0].map((v,i)=>v*hiddenDerivative[i]),jacobian=[sensitivity.map(v=>key.map(k=>v*k)),sensitivity,[hidden],[1]];
 return {pre,hidden,output,jacobian};
}
export function neuralMemoryWrite(parameters,momentum,key,target,{rate=.005,retention=.5,decay=.0001}={}){
 const read=neuralMemoryRead(parameters,key),residual=read.output-target,gradient=mapTree(read.jacobian,v=>v*residual),nextMomentum=zipTree(momentum,gradient,(s,g)=>retention*s-rate*g),nextParameters=zipTree(parameters,nextMomentum,(p,s)=>(1-decay)*p+s),update=zipTree(nextParameters,parameters,(a,b)=>a-b);
 return {prediction:read.output,loss:.5*residual**2,residual,gradient,gradientNorm:Math.hypot(...flatten(gradient)),updateNorm:Math.hypot(...flatten(update)),parameters:nextParameters,momentum:nextMomentum,read};
}
export function neuralOuterRate(parameters,key,target,query,outerTarget,rate){
 const write=neuralMemoryWrite(parameters,zeroMemory(parameters),key,target,{rate,retention:0,decay:0}),read=neuralMemoryRead(write.parameters,query),residual=read.output-outerTarget,derivative=-residual*dot(flatten(read.jacobian),flatten(write.gradient));
 return {write,read,loss:.5*residual**2,derivative};
}
export function rentalMemoryKey(dates,counts,index,mean,scale){
 const weekday=(new Date(`${dates[index]}T00:00:00Z`).getUTCDay()+6)%7,phase=2*Math.PI*weekday/7,key=[...counts.slice(index-7,index).map(v=>(v-mean)/scale),Math.sin(phase),Math.cos(phase)],norm=Math.hypot(...key);return key.map(v=>v/norm);
}
export function rentalMemoryReplay(snapshot,dates,counts,{start=365,stop=counts.length,parameters=snapshot.initial_parameters,momentum=null,rate=.005,retention=.5,decay=.0001,mean,scale}={}){
 let weights=structuredClone(parameters),state=momentum?structuredClone(momentum):zeroMemory(weights);const trace=[];
 for(let index=start;index<stop;index++){const key=rentalMemoryKey(dates,counts,index,mean,scale),r=neuralMemoryWrite(weights,state,key,(counts[index]-mean)/scale,{rate,retention,decay}),frozen=neuralMemoryRead(snapshot.initial_parameters,key).output*scale+mean;trace.push({index,date:dates[index],observed:counts[index],prediction_z:r.prediction,adaptive:r.prediction*scale+mean,frozen,loss_before_write:r.loss,gradient_norm:r.gradientNorm,update_norm:r.updateNorm,key});weights=r.parameters;state=r.momentum;}
 return {trace,parameters:weights,momentum:state};
}
export function titansPayload({width=2048,hidden=512,layers=24,batch=4,kvHeads=8,headWidth=128,window=2048,length=131072,weightBytes=4,kvBytes=2}={}){
 const b=BigInt;const perLayer=2n*b(width)*b(hidden),fast=perLayer*b(layers)*b(batch)*b(weightBytes),momentum=fast,windowKV=2n*b(batch)*b(layers)*b(Math.min(window,length))*b(kvHeads)*b(headWidth)*b(kvBytes),fullKV=2n*b(batch)*b(layers)*b(length)*b(kvHeads)*b(headWidth)*b(kvBytes);return {parametersPerLayer:perLayer,fast,momentum,windowKV,fullKV,partialTotal:fast+momentum+windowKV};
}
