// Independent, deterministic numerical model. No React or frozen-weight imports.
export const dot = (a,b) => a.reduce((sum,x,i) => sum+x*b[i],0);
export function softmax(xs) { const maximum = Math.max(...xs), masses = xs.map(x => Math.exp(x-maximum)), sum=masses.reduce((a,b)=>a+b,0); return masses.map(x=>x/sum); }
export const top = (scores,k) => scores.map((value,id)=>({value,id})).sort((a,b)=>b.value-a.value || a.id-b.id).slice(0,k).map(x=>x.id);
export function route(scores, experts, k, selectedNormalization=true, direction=[1,1]) {
  const selected=top(scores,k), probabilities=softmax(scores), selectedWeights=softmax(selected.map(i=>scores[i]));
  const weights=scores.map((_,i)=>selected.includes(i) ? (selectedNormalization ? selectedWeights[selected.indexOf(i)] : probabilities[i]) : 0);
  const output=experts[0].map((_,j)=>experts.reduce((sum,e,i)=>sum+weights[i]*e[j],0));
  const loss=dot(output,direction), gradient=scores.map((_,i)=> selectedNormalization ? weights[i]*(dot(experts[i],direction)-loss) : probabilities[i]*((selected.includes(i)?dot(experts[i],direction):0)-loss));
  const sorted=top(scores,scores.length);
  return {selected,probabilities,weights,output,gradient,products:experts.map((e,i)=>e.map(x=>x*weights[i])),margin:k<scores.length?scores[sorted[k-1]]-scores[sorted[k]]:null};
}
export function dispatch(routes, capacity, {order=routes.map((_,i)=>i),values=[1,2,3,4],gates=routes.map(()=>[.5,.5]),renormalize=false}={}) {
  const used=values.map(()=>0), requested=values.map(()=>0), retained=[],dropped=[],output=routes.map(()=>0),mass=routes.map(()=>0);
  for(const token of order) for(let slot=0;slot<routes[token].length;slot++) { const expert=routes[token][slot];requested[expert]++; const record={token,slot,expert,weight:gates[token][slot]}; if(used[expert]<capacity){used[expert]++;retained.push(record);mass[token]+=record.weight;output[token]+=record.weight*values[expert];}else dropped.push(record); }
  if(renormalize) output.forEach((_,i)=>{if(mass[i]>0)output[i]/=mass[i];});
  return {used,requested,retained,dropped,output,mass,fullyDropped:routes.map((_,i)=>i).filter(i=>!retained.some(row=>row.token===i))};
}
export function balance(scores,k=1) {
  const probabilities=scores.map(softmax),n=scores[0].length,counts=Array(n).fill(0),mean=Array(n).fill(0);
  scores.forEach((row,t)=>{top(row,k).forEach(i=>counts[i]++); probabilities[t].forEach((p,i)=>mean[i]+=p/scores.length);});
  const fractions=counts.map(x=>x/(scores.length*k));
  const partitions=scores.map(row=>{const maximum=Math.max(...row);return maximum+Math.log(row.reduce((sum,x)=>sum+Math.exp(x-maximum),0));});
  return {probabilities,counts,mean,fractions,value:n*dot(fractions,mean),partitions,zLoss:dot(partitions,partitions)/scores.length};
}
export function costs({d=48,m=80,n=10,k=2,tokens=24,bytes=4,remote=1,shared=0}={}) {
  const expert=3*d*m; return {expert,stored:(n+shared)*expert,router:d*n,active:(k+shared)*expert,weightBytes:((n+shared)*expert+d*n)*bytes,payload:2*tokens*k*d*bytes*remote,assignments:tokens*k};
}
export function biasedSelection(affinity,bias,k=2){const selected=top(affinity.map((s,i)=>s+bias[i]),k),sum=selected.reduce((s,i)=>s+affinity[i],0);return{selected,weights:affinity.map((s,i)=>selected.includes(i)?s/sum:0)};}
export function choice(probabilities,k=1,capacity=1){return{token:probabilities.map(row=>top(row,k)),expert:probabilities[0].map((_,i)=>top(probabilities.map(row=>row[i]),capacity))};}
export function frozenMoE(state,pixels,{temperature=1,disabled=null}={}) {
  const linear=(x,name)=>state[`${name}.weight`].map((w,i)=>dot(w,x)+(state[`${name}.bias`]?.[i]??0));
  const norm=(x,name)=>{const mean=x.reduce((a,b)=>a+b,0)/x.length,variance=x.reduce((s,v)=>s+(v-mean)**2,0)/x.length;return x.map((v,i)=>(v-mean)/Math.sqrt(variance+1e-5)*state[`${name}.weight`][i]+state[`${name}.bias`][i]);};
  const patches=Array.from({length:16},(_,p)=>{const r=Math.floor(p/4)*2,c=p%4*2;return[pixels[r*8+c],pixels[r*8+c+1],pixels[(r+1)*8+c],pixels[(r+1)*8+c+1]];});
  let hidden=patches.map((patch,p)=>linear(patch,'project').map((v,i)=>v+state.position[p][i]));
  const qkv=hidden.map(x=>linear(norm(x,'norm_attention'),'qkv'));
  const attention=Array.from({length:2},(_,head)=>qkv.map(q=>softmax(qkv.map(k=>dot(q.slice(head*8,head*8+8),k.slice(16+head*8,24+head*8))/Math.sqrt(8)))));
  hidden=hidden.map((x,t)=>{const context=Array.from({length:16},(_,i)=>qkv.reduce((sum,v,s)=>sum+attention[Math.floor(i/8)][t][s]*v[32+i],0));return linear(context,'attention_output').map((v,i)=>v+x[i]);});
  const expertInputs=hidden.map(x=>norm(x,'norm_ffn')),scores=expertInputs.map(x=>linear(x,'router').map(s=>s/temperature)),probabilities=scores.map(softmax),selected=scores.map(row=>top(row,2)),weights=scores.map((row,i)=>softmax(selected[i].map(e=>row[e])));
  const expertOutputs=expertInputs.map((x,t)=>Array.from({length:4},(_,e)=>{if(!selected[t].includes(e)||e===disabled)return null;const gate=linear(x,`experts.${e}.gate`),value=linear(x,`experts.${e}.value`);return linear(gate.map((g,i)=>g/(1+Math.exp(-g))*value[i]),`experts.${e}.down`);}));
  const combined=expertInputs.map((_,t)=>Array.from({length:16},(_,i)=>selected[t].reduce((sum,e,slot)=>sum+(e===disabled?0:weights[t][slot]*expertOutputs[t][e][i]),0)));
  const final=hidden.map((x,t)=>norm(x.map((v,i)=>v+combined[t][i]),'norm_final'));
  const logits=linear(final[0].map((_,i)=>final.reduce((sum,x)=>sum+x[i]/16,0)),'classifier'),classProbabilities=softmax(logits),counts=Array(4).fill(0);
  selected.flat().forEach(e=>counts[e]++);
  return {patches,attention,expertInputs,scores,probabilities,selected,weights,expertOutputs,combined,logits,classProbabilities,counts,predicted:top(logits,1)[0],auxiliary:balance(scores,2).value};
}
