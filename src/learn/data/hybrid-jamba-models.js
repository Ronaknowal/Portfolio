import { dense, dot, softmax, maxDifference } from './sequence-tensor-operations.js';

const sum=values=>values.reduce((a,b)=>a+b,0);
const clone=value=>value===null?null:JSON.parse(JSON.stringify(value));
const silu=x=>x/(1+Math.exp(-x));
const softplus=x=>x>20?x:Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const norm=(x,gain)=>{const scale=1/Math.sqrt(sum(x.map(v=>v*v))/x.length+1e-6);return x.map((v,i)=>v*scale*gain[i]);};
export function hybridMemoryRead(values,keys,query,decay=.5,beta=Math.log(9)){
 if(!values.length||values.length!==keys.length)throw new Error('A memory needs at least one keyed value.');
 let numerator=0,normalizer=0;const states=values.map(value=>{numerator=decay*numerator+value;normalizer=decay*normalizer+1;return[numerator,normalizer,numerator/normalizer];});
 const weights=softmax(keys.map(key=>key===query?beta:0)),recurrenceWeights=values.map((_,i)=>decay**(values.length-1-i)/normalizer);
 return{states,weights,recurrenceWeights,attention:dot(weights,values),summary:states.at(-1)[2]};
}
export const jambaOriginalBudget={batch:1,layers:32,attentionLayers:4,width:4096,kvHeads:8,headWidth:128,stateSize:16,expand:2,convWidth:4,kvBytes:2,stateBytes:4,convBytes:2,length:262144,window:null};
export const jambaFreshBudget={...jambaOriginalBudget,batch:3,layers:12,attentionLayers:3,width:512,kvHeads:2,headWidth:64,stateSize:8,length:2048};
export function hybridCacheBytes(settings={}){
 const c={...jambaOriginalBudget,...settings};const b=BigInt(c.batch),a=BigInt(c.attentionLayers),m=BigInt(c.layers-c.attentionLayers),i=BigInt(c.width*c.expand),retained=c.window===null?c.length:Math.min(c.length,c.window);
 const kv=b*a*2n*BigInt(c.kvHeads)*BigInt(c.headWidth)*BigInt(retained)*BigInt(c.kvBytes),recurrent=b*m*i*BigInt(c.stateSize)*BigInt(c.stateBytes),convolution=b*m*i*BigInt(c.convWidth)*BigInt(c.convBytes),total=kv+recurrent+convolution;
 return{kv,recurrent,convolution,total,totalGiB:Number(total)/2**30,retained};
}
export function hybridAttentionOperations({batch=1,width=4096,kvHeads=8,headWidth=128,length=1024}={}){
 const b=BigInt(batch),d=BigInt(width),k=BigInt(kvHeads*headWidth),t=BigInt(length);
 return{projection:4n*b*t*d*(d+k),causalPairs:t*(t+1n)/2n,promptPairs:2n*b*d*t*(t+1n),decodePairs:4n*b*d*(t+1n)};
}
export function hybridRoute(logits,expertValues,k=2,renormalize=false){
 const probabilities=softmax(logits),order=probabilities.map((_,i)=>i).sort((a,b)=>probabilities[b]-probabilities[a]||a-b),selected=order.slice(0,k),selectedMass=sum(selected.map(i=>probabilities[i]));
 const weights=selected.map(i=>probabilities[i]/(renormalize?selectedMass:1)),contributions=weights.map((w,j)=>expertValues[selected[j]].map(v=>w*v));
 return{probabilities,selected,weights,selectedMass,contributions,output:expertValues[0].map((_,j)=>sum(contributions.map(v=>v[j]))),boundaryTie:k<order.length&&probabilities[order[k-1]]===probabilities[order[k]]};
}
export function hybridExpertParameters({width=4096,ffnWidth=14336,denseLayers=16,expertLayers=16,experts=16,selected=2}={}){
 const one=3n*BigInt(width)*BigInt(ffnWidth);return{one,stored:(BigInt(denseLayers)+BigInt(expertLayers*experts))*one,active:(BigInt(denseLayers)+BigInt(expertLayers*selected))*one,dense:BigInt(denseLayers+expertLayers)*one};
}

// Arrays are exact exported float32 parameters; browser arithmetic is float64.
// The caller owns its request identity/offset. Inputs and caches are never mutated.
export function hybridStrokeStream(snapshot,coordinates,{cache=null,offset=0}={}){
 const w=snapshot.state,linear=(x,name)=>dense(x,w[name+'.weight'],w[name+'.bias']);
 if(snapshot.pattern==='linear'){
  if(coordinates.length!==8)throw new Error('The flattened baseline requires all eight points.');
  const logits=linear(coordinates.flat().map(v=>v/50-1),'classifier');return{logits:[logits],probabilities:softmax(logits),cache:null,frames:[]};
 }
 const caches=cache?clone(cache):Array(snapshot.pattern.length).fill(null),logits=[],frames=[];
 for(let token=0;token<coordinates.length;token++){
  const position=(offset+token)/7,input=[...coordinates[token].map(v=>v/50-1),position,position*position];let h=linear(input,'embed');const layers=[];
  for(let index=0;index<snapshot.pattern.length;index++){
   const prefix=`layers.${index}`,m=prefix+'.mixer',kind=snapshot.pattern[index],before=[...h],normalized=norm(h,w[prefix+'.norm1.weight']);let mixed,trace;
   if(kind==='M'){
    const both=linear(normalized,m+'.input'),projected=both.slice(0,16),gate=both.slice(16),previous=caches[index]??{state:Array.from({length:16},()=>Array(4).fill(0)),history:Array.from({length:16},()=>[0,0])};
    const window=projected.map((v,i)=>[...previous.history[i],v]),u=window.map((v,i)=>silu(dot(v,w[m+'.convolution.weight'][i][0])+w[m+'.convolution.bias'][i]));
    const delta=linear(u,m+'.delta').map(softplus),write=norm(linear(u,m+'.write'),w[m+'.write_norm.weight']),read=norm(linear(u,m+'.read'),w[m+'.read_norm.weight']);
    const decay=delta.map((v,i)=>w[m+'.log_rates'][i].map(a=>Math.exp(-v*Math.exp(a)))),retained=decay.map((r,i)=>r.map((a,j)=>a*previous.state[i][j])),written=delta.map((v,i)=>write.map(b=>v*b*u[i])),state=retained.map((r,i)=>add(r,written[i]));
    const value=state.map((r,i)=>dot(r,read)+w[m+'.skip'][i]*u[i]),gated=value.map((v,i)=>v*silu(gate[i]));mixed=linear(gated,m+'.output');
    caches[index]={state,history:window.map(row=>row.slice(1))};trace={projected,gate,window,u,delta,write,read,decay,retained,written,state,value,gated};
   }else{
    const qkv=linear(normalized,m+'.qkv'),query=qkv.slice(0,16),key=qkv.slice(16,32),value=qkv.slice(32),keys=[...(caches[index]?.keys??[]),key],values=[...(caches[index]?.values??[]),value];
    const scores=keys.map(k=>dot(query,k)/4),weights=softmax(scores),read=values[0].map((_,j)=>dot(weights,values.map(v=>v[j])));mixed=linear(read,m+'.output');caches[index]={keys,values};trace={query,key,value,keys,values,scores,weights,read};
   }
   const intermediate=add(h,mixed),normalizedFfn=norm(intermediate,w[prefix+'.norm2.weight']),gateUp=linear(normalizedFfn,prefix+'.gate_up'),ffnHidden=gateUp.slice(0,32).map((v,i)=>silu(v)*gateUp[i+32]),ffn=linear(ffnHidden,prefix+'.down');h=add(intermediate,ffn);
   layers.push({kind,before,normalized,mixed,intermediate,ffn,output:[...h],...trace});
  }
  const normalized=norm(h,w['norm.weight']),output=linear(normalized,'classifier');logits.push(output);frames.push({position:offset+token,input,hidden:h,normalized,logits:output,layers});
 }
 return{logits,probabilities:logits.length?softmax(logits.at(-1)):null,cache:caches,frames};
}
export function hybridStrokeTrace(snapshot,coordinates,boundary=3,mode='carry',donorCoordinates=null){
 const full=hybridStrokeStream(snapshot,coordinates),prefix=hybridStrokeStream(snapshot,coordinates.slice(0,boundary)),snapshotCache=clone(prefix.cache),faultCache=mode==='cache-swap'&&donorCoordinates?hybridStrokeStream(snapshot,donorCoordinates.slice(0,boundary)).cache:clone(snapshotCache);
 for(let i=0;i<snapshot.pattern.length;i++){
  if((mode==='recurrent-reset'&&snapshot.pattern[i]==='M')||(mode==='kv-reset'&&snapshot.pattern[i]==='A'))faultCache[i]=null;
  if(mode==='convolution-reset'&&snapshot.pattern[i]==='M'&&faultCache[i])faultCache[i].history=faultCache[i].history.map(row=>row.map(()=>0));
 }
 const suffix=hybridStrokeStream(snapshot,coordinates.slice(boundary),{cache:faultCache,offset:mode==='position-reset'?0:boundary}),branch=[...prefix.logits,...suffix.logits],probabilities=softmax(branch.at(-1));
 return{full:full.logits,branch,fullProbabilities:full.probabilities,probabilities,fullPrediction:full.probabilities.indexOf(Math.max(...full.probabilities)),prediction:probabilities.indexOf(Math.max(...probabilities)),difference:maxDifference(full.logits,branch),positionDifferences:branch.map((row,i)=>maxDifference(row,full.logits[i])),cache:snapshotCache,branchCache:suffix.cache,frames:full.frames,branchFrames:[...prefix.frames,...suffix.frames]};
}
