// C is key rows × value columns. Projected model queries are scaled once.
// Mechanism inputs already contain their intended query scale.
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
export const logSigmoid=x=>-Math.max(-x,0)-Math.log1p(Math.exp(-Math.abs(x)));
export const softmax=x=>{const top=Math.max(...x),p=x.map(v=>Math.exp(v-top)),sum=p.reduce((a,b)=>a+b,0);return p.map(v=>v/sum);};
const zeros=n=>Array(n).fill(0),matrix=(k,v)=>Array.from({length:k},()=>zeros(v)),clone=x=>structuredClone(x);
export const transposeRead=(cell,q)=>cell[0].map((_,j)=>cell.reduce((s,row,k)=>s+row[j]*q[k],0));
export function scalarScan(rows,output=.6,stabilized=true,offset=0){
 let cell=0,normalizer=0,logScale=0;const trace=[];
 for(let t=0;t<rows.length;t++){
  const row=rows[t],writeLog=row.writeLog+offset,forgetLog=Math.log(row.retention),newScale=stabilized?Math.max(forgetLog+logScale,writeLog):0,write=Math.exp(writeLog-newScale),retain=Math.exp(forgetLog+logScale-newScale);
  cell=retain*cell+write*row.value;normalizer=retain*normalizer+write;
  const weights=rows.slice(0,t+1).map((source,j)=>Math.exp(source.writeLog+offset+rows.slice(j+1,t+1).reduce((s,r)=>s+Math.log(r.retention),0)-newScale));
  trace.push({cell,normalizer,log_scale:newScale,write,retain,hidden:output*cell/normalizer,estimate:cell/normalizer,weights,contributions:weights.map((w,j)=>w*rows[j].value)});logScale=newScale;
 }
 return trace;
}
export function matrixScan(rows,{stabilized=true,wrongFloor=false,initial=null}={}){
 const kd=rows[0].key.length,vd=rows[0].value.length;let cell=initial?clone(initial.cell):matrix(kd,vd),normalizer=initial?[...initial.normalizer]:zeros(kd),logScale=initial?.log_scale||0;const trace=[];
 for(let t=0;t<rows.length;t++){
  const row=rows[t],forgetLog=Math.log(row.retention),newScale=stabilized?Math.max(forgetLog+logScale,row.writeLog):0,write=Math.exp(row.writeLog-newScale),retain=Math.exp(forgetLog+logScale-newScale);
  cell=cell.map((values,k)=>values.map((old,v)=>retain*old+write*row.key[k]*row.value[v]));normalizer=normalizer.map((old,k)=>retain*old+write*row.key[k]);
  const numerator=transposeRead(cell,row.query),mass=dot(normalizer,row.query),floor=wrongFloor?1:Math.exp(-newScale),denominator=Math.max(Math.abs(mass),floor);
  const weights=rows.slice(0,t+1).map((source,j)=>Math.exp(source.writeLog+rows.slice(j+1,t+1).reduce((s,r)=>s+Math.log(r.retention),0)-newScale)),coefficients=weights.map((w,j)=>w*dot(rows[j].key,row.query));
  trace.push({cell,normalizer,log_scale:newScale,numerator,mass,floor,denominator,read:numerator.map(v=>v/denominator),write,retain,weights,coefficients,contributions:coefficients.map((a,j)=>rows[j].value.map(v=>a*v))});logScale=newScale;
 }
 return trace;
}
export function denseMatrixRead(rows){
 return rows.map((row,t)=>{
  const logs=rows.slice(0,t+1).map((source,j)=>source.writeLog+rows.slice(j+1,t+1).reduce((s,r)=>s+Math.log(r.retention),0)),scale=Math.max(...logs),coefficients=rows.map((source,j)=>j<=t?Math.exp(logs[j]-scale)*dot(row.query,source.key):0),numerator=row.value.map((_,v)=>rows.reduce((s,r,j)=>s+coefficients[j]*r.value[v],0)),mass=coefficients.reduce((a,b)=>a+b,0),denominator=Math.max(Math.abs(mass),Math.exp(-scale));
  return{read:numerator.map(v=>v/denominator),coefficients,numerator,mass,denominator,log_scale:scale};
 });
}
export function chunkMatrixRead(rows,size,initial=null,resetAtBoundary=false){
 const kd=rows[0].key.length,vd=rows[0].value.length;let cell=initial?clone(initial.cell):matrix(kd,vd),normalizer=initial?[...initial.normalizer]:zeros(kd);const outputs=[],boundaries=[];
 for(let start=0;start<rows.length;start+=size){
  if(start>0&&resetAtBoundary){cell=matrix(kd,vd);normalizer=zeros(kd);}
  const incoming={cell:clone(cell),normalizer:[...normalizer]},end=Math.min(rows.length,start+size),local=rows.slice(start,end);
  for(let t=start;t<end;t++){
   const row=rows[t],retain=rows.slice(start,t+1).reduce((p,r)=>p*r.retention,1),incomingNumerator=transposeRead(cell,row.query).map(x=>x*retain),incomingMass=dot(normalizer,row.query)*retain;
   const coefficients=local.map((source,j)=>start+j<=t?Math.exp(source.writeLog)*rows.slice(start+j+1,t+1).reduce((p,r)=>p*r.retention,1)*dot(row.query,source.key):0),localNumerator=row.value.map((_,v)=>local.reduce((s,r,j)=>s+coefficients[j]*r.value[v],0)),localMass=coefficients.reduce((a,b)=>a+b,0),numerator=incomingNumerator.map((x,v)=>x+localNumerator[v]),mass=incomingMass+localMass,denominator=Math.max(Math.abs(mass),1);
   outputs.push({read:numerator.map(v=>v/denominator),incomingNumerator,incomingMass,localNumerator,localMass,numerator,mass,denominator,coefficients,start,end,incoming});
  }
  const retained=local.reduce((p,r)=>p*r.retention,1),weights=local.map((row,j)=>Math.exp(row.writeLog)*local.slice(j+1).reduce((p,r)=>p*r.retention,1));
  cell=cell.map((r,k)=>r.map((v,j)=>retained*v+local.reduce((s,row,i)=>s+weights[i]*row.key[k]*row.value[j],0)));normalizer=normalizer.map((v,k)=>retained*v+local.reduce((s,row,i)=>s+weights[i]*row.key[k],0));
  boundaries.push({start,end,incoming,outgoing:{cell:clone(cell),normalizer:[...normalizer]}});
 }
 return{outputs,boundaries,state:{cell,normalizer}};
}
export function gateLearning(theta,rate=.5,target=.7){const w=Math.exp(theta),prediction=(-.1+.8*w)/(.5+w),derivative=.5*w/(.5+w)**2,gradient=(prediction-target)*derivative;return{prediction,loss:.5*(prediction-target)**2,gradient,next:theta-rate*gradient};}
const linear=(x,p,name)=>p[name+'.weight'].map((row,i)=>dot(row,x)+(p[name+'.bias']?.[i]||0));
const rms=(x,weights)=>{const denominator=Math.sqrt(dot(x,x)/x.length+1e-6);return x.map((v,i)=>v/denominator*weights[i]);};
export function emptyReaderState(kind){return kind==='lstm'?[zeros(16),zeros(16)]:kind==='slstm'?[zeros(16),zeros(16),zeros(16),zeros(16)]:[matrix(8,16),zeros(8),0];}
export function readerForward(rawRows,model,{initial=null,resetAfter=-1,resetNormalizer=false}={}){
 const p=model.parameters,kind=model.kind,trace=[];let state=initial?clone(initial):emptyReaderState(kind);
 for(let t=0;t<rawRows.length;t++){
  if(t===resetAfter){if(resetNormalizer&&kind!=='lstm'){state=clone(state);state[kind==='slstm'?2:1]=zeros(kind==='slstm'?16:8);}else state=emptyReaderState(kind);}
  const input=rawRows[t].map(x=>x/16),embedded=linear(input,p,'input_projection'),normalized=rms(embedded,p['pre_norm.weight']),stages={input_projection:embedded,pre_norm:normalized};let mixed,cellDetails={};
  if(kind==='lstm'){
   const [h,c]=state,a=p['sequence_model.weight_ih_l0'].map((row,i)=>dot(row,normalized)+p['sequence_model.bias_ih_l0'][i]+dot(p['sequence_model.weight_hh_l0'][i],h)+p['sequence_model.bias_hh_l0'][i]),i=a.slice(0,16).map(sigmoid),f=a.slice(16,32).map(sigmoid),g=a.slice(32,48).map(Math.tanh),o=a.slice(48).map(sigmoid),nextC=c.map((v,j)=>f[j]*v+i[j]*g[j]);
   mixed=nextC.map((v,j)=>o[j]*Math.tanh(v));state=[mixed,nextC];cellDetails={inputGate:i,forgetGate:f,candidate:g,outputGate:o};
  }else if(kind==='slstm'){
   const [h,c,n,m]=state,inputGates=linear(normalized,p,'sequence_model.input_gates'),recurrent=linear(h,p,'sequence_model.recurrent_gates'),a=inputGates.map((v,i)=>v+recurrent[i]),writeLog=a.slice(0,16),forgetLog=a.slice(16,32).map(logSigmoid),outputGate=a.slice(32,48).map(sigmoid),candidate=a.slice(48).map(Math.tanh),scale=m.map((old,j)=>Math.max(forgetLog[j]+old,writeLog[j])),write=scale.map((s,j)=>Math.exp(writeLog[j]-s)),retain=scale.map((s,j)=>Math.exp(forgetLog[j]+m[j]-s)),nextC=c.map((v,j)=>retain[j]*v+write[j]*candidate[j]),nextN=n.map((v,j)=>retain[j]*v+write[j]);
   mixed=nextC.map((v,j)=>outputGate[j]*v/nextN[j]);state=[mixed,nextC,nextN,scale];cellDetails={inputGates,recurrent,writeLog,forgetLog,write,retain,candidate,outputGate};
  }else{
   const [c,n,m]=state,queries=linear(normalized,p,'sequence_model.queries'),q=queries.map(v=>v/Math.sqrt(8)),k=linear(normalized,p,'sequence_model.keys'),v=linear(normalized,p,'sequence_model.values'),gateRaw=linear(normalized,p,'sequence_model.gates'),a=gateRaw.map(x=>15*Math.tanh(x/15)),outputRaw=linear(normalized,p,'sequence_model.output_gate'),outputGate=outputRaw.map(sigmoid),forgetLog=logSigmoid(a[1]),scale=Math.max(forgetLog+m,a[0]),write=Math.exp(a[0]-scale),retain=Math.exp(forgetLog+m-scale),nextC=c.map((row,i)=>row.map((x,j)=>retain*x+write*k[i]*v[j])),nextN=n.map((x,i)=>retain*x+write*k[i]),numerator=transposeRead(nextC,q),mass=dot(nextN,q),floor=Math.exp(-scale),denominator=Math.max(Math.abs(mass),floor),read=numerator.map(x=>x/denominator),readNorm=rms(read,p['sequence_model.read_norm.weight']);
   mixed=readNorm.map((x,i)=>outputGate[i]*x);state=[nextC,nextN,scale];Object.assign(stages,{'sequence_model.queries':queries,'sequence_model.keys':k,'sequence_model.values':v,'sequence_model.gates':gateRaw,'sequence_model.output_gate':outputRaw,'sequence_model.read_norm':readNorm});cellDetails={query:q,key:k,value:v,writeLog:a[0],forgetLog,write,retain,numerator,mass,floor,denominator,read,readNorm,outputGate};
  }
  const residual=embedded.map((x,i)=>x+mixed[i]),postNorm=rms(residual,p['post_norm.weight']),expand=linear(postNorm,p,'expand'),gate=linear(postNorm,p,'gate'),gated=expand.map((x,i)=>x*sigmoid(x)*gate[i]),contract=linear(gated,p,'contract'),output=residual.map((x,i)=>x+contract[i]),logits=linear(output,p,'classifier');
  Object.assign(stages,{post_norm:postNorm,expand,gate,contract,classifier:logits});trace.push({logits,probabilities:softmax(logits),prediction:logits.indexOf(Math.max(...logits)),state:clone(state),stages,cellDetails,mixed,residual,output});
 }
 return{trace,state,logits:trace.map(t=>t.logits)};
}
export function recurrentStorage(layers,heads,keyWidth,valueWidth,bytes=4){const scale=BigInt(layers)*BigInt(heads)*BigInt(bytes),matrixBytes=scale*BigInt(keyWidth)*BigInt(valueWidth);return{matrixBytes,normalizerBytes:scale*BigInt(keyWidth),scaleBytes:scale,totalBytes:matrixBytes+scale*BigInt(keyWidth+1)};}
