// Bounded pure attention, ownership and explicitly hypothetical resource models.
export const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
const zeros=n=>Array(n).fill(0),clone=x=>structuredClone(x);
export const maxError=(a,b)=>Array.isArray(a)?Math.max(0,...a.map((v,i)=>maxError(v,b[i]))):Math.abs(a-b);
export function softmax(scores,allowed=scores.map(()=>true)){const valid=scores.filter((_,i)=>allowed[i]);if(!valid.length)return null;const maximum=Math.max(...valid),mass=scores.map((s,i)=>allowed[i]?Math.exp(s-maximum):0),total=sum(mass);return mass.map(v=>v/total);}
export function ownership(length,ranks,kind='contiguous'){
 if(!Number.isInteger(length)||!Number.isInteger(ranks)||ranks<1||ranks>length)throw Error('Each owner needs at least one position.');
 if(kind==='striped')return Array.from({length:ranks},(_,i)=>Array.from({length:Math.ceil((length-i)/ranks)},(_,j)=>i+j*ranks));
 if(kind==='zigzag'){if(length%(2*ranks))throw Error('Equal zigzag pieces require L divisible by 2P.');const c=length/(2*ranks);return Array.from({length:ranks},(_,i)=>[...Array.from({length:c},(_,j)=>i*c+j),...Array.from({length:c},(_,j)=>(2*ranks-i-1)*c+j)]);}
 let start=0;return Array.from({length:ranks},(_,i)=>{const size=Math.floor(length/ranks)+(i<length%ranks?1:0),ids=Array.from({length:size},(_,j)=>start+j);start+=size;return ids;});
}
export function validateOwnership(owners,length){const a=owners.flat();return owners.length>0&&owners.every(r=>r.length>0)&&a.length===length&&a.every(Number.isInteger)&&a.slice().sort((a,b)=>a-b).every((x,i)=>x===i);}
export const ownerOf=(owners,id)=>owners.findIndex(r=>r.includes(id));
export function scalarMerge(scores,values,blocks,allowed=scores.map(()=>true),offset=0){
 const actual=scores.map(v=>v+offset);let maximum=-Infinity,total=0,numerator=0,wrongTotal=0,wrongNumerator=0;const trace=[],local=[];
 for(const ids of blocks){const valid=ids.filter(i=>allowed[i]),blockMaximum=valid.length?Math.max(...valid.map(i=>actual[i])):-Infinity,newMaximum=Math.max(maximum,blockMaximum),safe=Number.isFinite(newMaximum)?newMaximum:0,alpha=Math.exp(maximum-safe),weights=ids.map(i=>allowed[i]?Math.exp(actual[i]-safe):0),addMass=sum(weights),products=weights.map((w,j)=>w*values[ids[j]]),addNumerator=sum(products),old={maximum,total,numerator};
  wrongTotal+=addMass;wrongNumerator+=addNumerator;total=alpha*total+addMass;numerator=alpha*numerator+addNumerator;maximum=newMaximum;
  const w=softmax(ids.map(i=>actual[i]),ids.map(i=>allowed[i]));if(w)local.push(dot(w,ids.map(i=>values[i])));
  trace.push({ids,valid,old,maximum,alpha,rescaledMass:alpha*old.total,rescaledNumerator:alpha*old.numerator,weights,products,addMass,addNumerator,total,numerator,output:total>0?numerator/total:0,validRow:total>0});
 }
 const weights=softmax(actual,allowed),output=weights?dot(weights,values):0;
 const independent=blocks.map(ids=>{const valid=ids.filter(i=>allowed[i]),m=valid.length?Math.max(...valid.map(i=>actual[i])):-Infinity,e=ids.map(i=>allowed[i]?Math.exp(actual[i]-(Number.isFinite(m)?m:0)):0);return{ids,maximum:m,total:sum(e),numerator:dot(e,ids.map(i=>values[i]))};});
 const common=Math.max(...independent.map(r=>r.maximum)),terms=independent.map(r=>({...r,alpha:Number.isFinite(common)?Math.exp(r.maximum-common):0}));
 return{trace,weights,output,valid:!!weights,naive:local.length?sum(local)/local.length:0,unscaled:wrongTotal?wrongNumerator/wrongTotal:0,parallel:{maximum:common,terms,total:sum(terms.map(r=>r.alpha*r.total)),numerator:sum(terms.map(r=>r.alpha*r.numerator))}};
}
export function recordMask(records,{causal=true,packed=true,empty=-1,localWrong=false,owners}={}){return records.map((q,i)=>records.map((k,j)=>i!==empty&&q.valid&&k.valid&&(!packed||q.document===k.document)&&(!causal||(localWrong?owners[ownerOf(owners,j)].indexOf(j)<=owners[ownerOf(owners,i)].indexOf(i):k.position<=q.position))));}
export function denseAttention(query,key,value,allowed){const scale=Math.sqrt(query[0][0].length),scores=query.map((head,h)=>head.map(q=>key[h].map(k=>dot(q,k)/scale))),weights=scores.map(head=>head.map((row,i)=>softmax(row,allowed[i])??zeros(row.length))),output=weights.map((head,h)=>head.map(w=>value[h][0].map((_,c)=>sum(w.map((a,j)=>a*value[h][j][c])))));return{scores,weights,output};}
export function ringAttention(query,key,value,owners,allowed,direction=1,selected={head:0,query:0}){
 const H=query.length,L=query[0].length,D=query[0][0].length,V=value[0][0].length,P=owners.length;
 if(!validateOwnership(owners,L)||![1,-1].includes(direction))throw Error('Need an exact nonempty owner partition and direction +1 or −1.');
 const output=Array.from({length:H},()=>Array.from({length:L},()=>zeros(V))),lse=Array.from({length:H},()=>Array(L).fill(-Infinity));const trace=[];
 for(let h=0;h<H;h++)for(let rank=0;rank<P;rank++)for(const i of owners[rank]){let m=-Infinity,total=0,u=zeros(V);
  for(let step=0;step<P;step++){const owner=(rank-direction*step+P*P)%P,ids=owners[owner],scores=ids.map(j=>allowed[i][j]?dot(query[h][i],key[h][j])/Math.sqrt(D):-Infinity),newMaximum=Math.max(m,...scores),safe=Number.isFinite(newMaximum)?newMaximum:0,alpha=Math.exp(m-safe),e=scores.map(s=>Math.exp(s-safe)),addMass=sum(e),addNumerator=zeros(V).map((_,c)=>sum(e.map((w,k)=>w*value[h][ids[k]][c]))),old={maximum:m,total,numerator:[...u]};
   total=alpha*total+addMass;u=u.map((v,c)=>alpha*v+addNumerator[c]);m=newMaximum;
   if(h===selected.head&&i===selected.query)trace.push({step,rank,owner,ids,scores,exponentials:e,alpha,old,maximum:m,addMass,addNumerator,total,numerator:[...u],output:u.map(v=>total?v/total:0),valid:total>0});
  }output[h][i]=u.map(v=>total?v/total:0);lse[h][i]=total?m+Math.log(total):-Infinity;
 }
 return{output,lse,trace};
}
export function attentionBackward(query,key,value,owners,allowed,upstream,selected={head:0,key:0}){
 const r=ringAttention(query,key,value,owners,allowed),H=query.length,L=query[0].length,D=query[0][0].length,V=value[0][0].length;
 const dQ=query.map(head=>head.map(row=>zeros(row.length))),dK=clone(dQ),dV=value.map(head=>head.map(row=>zeros(row.length))),localK=clone(dK),localV=clone(dV),partials=owners.map((ids,owner)=>({owner,ids,dK:zeros(D),dV:zeros(V)}));
 for(let h=0;h<H;h++)for(let i=0;i<L;i++){const correction=dot(upstream[h][i],r.output[h][i]);for(let j=0;j<L;j++)if(allowed[i][j]){const probability=Math.exp(dot(query[h][i],key[h][j])/Math.sqrt(D)-r.lse[h][i]),ds=probability*(dot(upstream[h][i],value[h][j])-correction),same=ownerOf(owners,i)===ownerOf(owners,j);
  for(let c=0;c<D;c++){dQ[h][i][c]+=ds*key[h][j][c]/Math.sqrt(D);const v=ds*query[h][i][c]/Math.sqrt(D);dK[h][j][c]+=v;if(same)localK[h][j][c]+=v;if(h===selected.head&&j===selected.key)partials[ownerOf(owners,i)].dK[c]+=v;}
  for(let c=0;c<V;c++){const v=probability*upstream[h][i][c];dV[h][j][c]+=v;if(same)localV[h][j][c]+=v;if(h===selected.head&&j===selected.key)partials[ownerOf(owners,i)].dV[c]+=v;}
 }}return{dQ,dK,dV,localK,localV,partials,forward:r};
}
export function workCounts(owners,tile=1,direction=1){const P=owners.length,pairs=owners.map(q=>owners.map(k=>sum(q.map(i=>k.filter(j=>j<=i).length)))),executed=owners.map(q=>owners.map(k=>{let count=0;for(let i=0;i<q.length;i+=tile)for(let j=0;j<k.length;j+=tile){const qs=q.slice(i,i+tile),ks=k.slice(j,j+tile);if(qs.some(a=>ks.some(b=>b<=a)))count+=qs.length*ks.length;}return count;})),rounds=Array.from({length:P},(_,step)=>owners.map((_,rank)=>({owner:(rank-direction*step+P*P)%P,useful:pairs[rank][(rank-direction*step+P*P)%P],executed:executed[rank][(rank-direction*step+P*P)%P]})));return{pairs,executed,rounds,totals:pairs.map(sum),usefulTotal:sum(pairs.map(sum)),critical:sum(rounds.map(row=>Math.max(...row.map(r=>r.executed)))),idealCritical:sum(rounds.map(row=>Math.max(...row.map(r=>r.useful))))};}
export function timeline(P,C,D){const compute=[],transfer=[];let start=0;for(let i=0;i<P;i++){compute.push({round:i,start,end:start+C,wait:i?Math.max(0,D-C):0});if(i<P-1)transfer.push({round:i,start,end:start+D});start+=Math.max(C,D);}return{compute,transfer,serial:P*C+(P-1)*D,overlap:C+(P-1)*Math.max(C,D),computeOnly:P*C};}
export function resourceCost({local=1024,ranks=4,B=1,Hq=8,Hkv=2,d=64,bytes=2,flops=100,bandwidth=50,latency=2,alias=false}={}){
 const c=BigInt(local),b=BigInt(B),h=BigInt(Hq),kv=BigInt(Hkv),w=BigInt(d),s=BigInt(bytes),P=BigInt(ranks),F=4n*b*c*c*h*w,payload=2n*b*c*kv*w*s,Q=b*c*h*w*s,inventory={Q,currentKV:payload,nextKV:ranks>1?payload:0n,FP32Numerator:b*c*h*w*4n,FP32RowStatistics:2n*b*c*h*4n,distinctOutput:alias?0n:Q,FP32ScoreTile:b*h*BigInt(Math.min(local,128))**2n*4n},C=Number(F)/(flops*1e12)*1e6,D=latency+Number(payload)/(bandwidth*1e9)*1e6;
 return{flops:F,payload,forwardBytes:(P-1n)*payload,inventory,totalBytes:Object.values(inventory).reduce((a,b)=>a+b,0n),C,D,...timeline(ranks,C,D),global:local*ranks};
}
export function reshard(length,heads,ranks,swap=false){if(length%ranks||heads%ranks)throw Error('Equal sequence/head boards need L and H divisible by P.');const tensor=Array.from({length},(_,i)=>Array.from({length:heads},(_,h)=>i*heads+h)),sequence=ownership(length,ranks).map(ids=>ids.map(i=>tensor[i])),boards=Array.from({length:ranks},(_,rank)=>tensor.map(row=>row.slice(rank*heads/ranks,(rank+1)*heads/ranks)));if(swap&&length>1)[boards[0][0][0],boards[0][1][0]]=[boards[0][1][0],boards[0][0][0]];const restored=tensor.map((_,i)=>boards.flatMap(b=>b[i])),mismatches=tensor.flatMap((r,i)=>r.flatMap((v,h)=>restored[i][h]===v?[]:[{token:i,head:h,expected:v,actual:restored[i][h]}]));return{tensor,sequence,boards,restored,mismatches,perRank:length*heads/ranks,nonlocal:length*heads*(ranks-1)/(ranks*ranks)};}
export function nextTargets(tokens,documents,owners){const correct=tokens.map((_,i)=>i+1<tokens.length&&documents[i]===documents[i+1]?tokens[i+1]:null),wrong=tokens.map(()=>null);owners.forEach(ids=>ids.forEach((id,k)=>{const next=ids[k+1];wrong[id]=next!==undefined&&documents[id]===documents[next]?tokens[next]:null;}));return{correct,wrong};}
const linear=(x,w,b)=>w.map((row,i)=>dot(x,row)+(b?.[i]??0));
export function movementInference(points,packet,owners=ownership(45,4),direction=1,selected={head:0,query:0}){const s=packet.state_dict,L=points.length,features=points.map(row=>linear(row.map(v=>2*v-1),s['stem.weight'],s['stem.bias']).map(Math.tanh)),project=name=>[0,1].map(h=>features.map(row=>linear(row,s[`mixer.${name}.weight`]).slice(h*12,h*12+12))),query=project('query'),key=project('key'),value=project('value'),allowed=Array.from({length:L},()=>Array(L).fill(true)),dense=denseAttention(query,key,value,allowed),ring=ringAttention(query,key,value,owners,allowed,direction,selected),finish=output=>{const mixed=features.map((row,i)=>linear(output.flatMap(head=>head[i]),s['mixer.output.weight']).map((v,c)=>v+row[c])),pooled=zeros(24).map((_,c)=>sum(mixed.map(row=>row[c]))/L),logits=linear(pooled,s['classifier.weight'],s['classifier.bias']);return{logits,probabilities:softmax(logits)};};return{features,query,key,value,dense,ring,original:finish(dense.output),partitioned:finish(ring.output),error:maxError(dense.output,ring.output)};}
