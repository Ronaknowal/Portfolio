// Bounded diagnostic mechanisms. Recorded experimental evidence is supplied separately.
export function diagnosticScalar(rows,weight,rate){
 const details=rows.map(row=>{const prediction=weight*row.x,residual=prediction-row.y;return{...row,prediction,residual,loss:residual**2/2,contribution:residual*row.x};});
 const gradient=details.reduce((s,r)=>s+r.contribution,0)/rows.length,loss=details.reduce((s,r)=>s+r.loss,0)/rows.length,change=-rate*gradient;
 return{details,gradient,loss,change,correct:weight+change,omitted:weight,distinguishes:Math.abs(change)>1e-12};
}
export function diagnosticMode({values,mean,variance,training,gradEnabled,module='batchnorm'}){
 const batchMean=(values[0]+values[1])/2,populationVariance=(values[0]-values[1])**2/4,sampleVariance=2*populationVariance;
 const usedMean=training?batchMean:mean,usedVariance=training?populationVariance:variance;
 return{batchMean,populationVariance,sampleVariance,usedMean:module==='linear'?null:usedMean,usedVariance:module==='linear'?null:usedVariance,denominator:module==='linear'?null:Math.sqrt(usedVariance+1e-5),output:module==='linear'?values.map(v=>2*v+1):values.map(v=>(v-usedMean)/Math.sqrt(usedVariance+1e-5)),mean:module==='linear'?null:training?.9*mean+.1*batchMean:mean,variance:module==='linear'?null:training?.9*variance+.1*sampleVariance:variance,graph:gradEnabled};
}
export function diagnosticPairs(pairs,direction='smaller'){
 if(!pairs.length)return{pairs:[],mean:null,sd:null,wins:0,losses:0,ties:0,direction};
 const rows=pairs.map(p=>({...p,difference:p.b-p.a})),mean=rows.reduce((s,r)=>s+r.difference,0)/rows.length,sd=rows.length<2?null:Math.sqrt(rows.reduce((s,r)=>s+(r.difference-mean)**2,0)/(rows.length-1));
 const signed=rows.map(r=>r.difference*(direction==='larger'?1:-1));
 return{pairs:rows,mean,sd,wins:signed.filter(v=>v>1e-12).length,losses:signed.filter(v=>v< -1e-12).length,ties:signed.filter(v=>Math.abs(v)<=1e-12).length,direction};
}
export function diagnosticRestart({target,weight,rate,momentum,save,steps}){
 const step=(w,v,index)=>{const gradient=w-target,nextVelocity=momentum*v+gradient,nextWeight=w-rate*nextVelocity;return{index,weight:w,velocity:v,gradient,loss:gradient**2/2,nextVelocity,nextWeight};};
 const prefix=[];let w=weight,v=0;
 for(let i=0;i<save;i++){const row=step(w,v,i+1);prefix.push(row);w=row.nextWeight;v=row.nextVelocity;}
 const checkpoint={weight:w,velocity:v},full=[],reset=[];let fw=w,fv=v,rw=w,rv=0;
 for(let i=0;i<steps;i++){const a=step(fw,fv,save+i+1),b=step(rw,rv,save+i+1);full.push(a);reset.push(b);fw=a.nextWeight;fv=a.nextVelocity;rw=b.nextWeight;rv=b.nextVelocity;}
 let divergence=null;for(let i=0;i<steps&&!divergence;i++){for(const [field,stage]of[['gradient','forward gradient'],['nextVelocity','velocity update'],['nextWeight','parameter update']])if(Math.abs(full[i][field]-reset[i][field])>1e-12){divergence={step:save+i+1,stage};break;}}
 return{prefix,checkpoint,full,reset,divergence,finalFull:fw,finalReset:rw};
}
export function recordedDiagnosticPairs(data,seeds,update,metric,same=false){
 const column={validation_loss:4,validation_accuracy:5,training_accuracy:2}[metric];
 return seeds.map(seed=>{const pair=data.seeds[String(seed)],a=pair.clean[update][column],b=same?a:pair.shuffled_labels[update][column];return{id:`seed ${seed}`,a,b};});
}
export function diagnosticFirstObservable(reference,other){for(let i=0;i<reference.length;i++){const a=reference[i],b=other[i];if(a.training_positions.some((v,j)=>v!==b.training_positions[j]))return{update:a.update,quantity:'batch positions'};if(a.pre_update_loss!==b.pre_update_loss)return{update:a.update,quantity:'pre-update loss'};if(a.learning_rate_used!==b.learning_rate_used)return{update:a.update,quantity:'learning rate used'};}return null;}
