// Row-major, output-by-input matrices. Exact small geometry uses double precision;
// frozen float32 training results retain their declared comparison tolerance.
export const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
export const norm = a => Math.hypot(...a);
export const transpose = a => a[0].map((_,j)=>a.map(row=>row[j]));
export const matvec = (a,x) => a.map(row=>dot(row,x));
export const multiply = (a,b) => a.map(row=>transpose(b).map(col=>dot(row,col)));
export const scale = (a,c) => a.map(row=>row.map(x=>c*x));
export const difference = (a,b) => Math.max(...a.flat(Infinity).map((x,i)=>Math.abs(x-b.flat(Infinity)[i])));
const unit = x => norm(x)>1e-12 ? x.map(v=>v/norm(x)) : null;
export function singular2(w) {
  const [[a,b],[c,d]]=w, aa=a*a+c*c, bb=b*b+d*d, ab=a*b+c*d;
  const first=Math.sqrt(Math.max(0,(aa+bb+Math.hypot(aa-bb,2*ab))/2));
  const second=first>0?Math.abs(a*d-b*c)/first:0;
  const angle=.5*Math.atan2(2*ab,aa-bb),v=[Math.cos(angle),Math.sin(angle)];
  return {values:[first,second],v,u:first>1e-12?matvec(w,v).map(x=>x/first):null,frobenius:Math.hypot(a,b,c,d)};
}
export function powerTrace(w,initial,steps=1) {
  let u=unit(initial),records=[];const exact=singular2(w).values[0];
  if (!u) return {records,error:'Choose a nonzero starting direction.'};
  for(let i=0;i<steps;i++) {
    const wtU=matvec(transpose(w),u),v=unit(wtU);
    if(!v) return {records,error:'This direction gives no usable estimate: Wᵀu is zero.'};
    const wV=matvec(w,v),next=unit(wV);
    if(!next) return {records,error:'This direction gives no usable estimate: Wv is zero.'};
    const estimate=dot(next,wV);
    records.push({step:i+1,before:u,wtU,v,wV,u:next,estimate,trueNorm:exact/estimate});
    u=next;
  }
  return {records,error:null};
}
export function normalizeMatrix(w,method='exact',target=1,initial=[1,1],steps=1) {
  const svd=singular2(w),sigma=svd.values[0];let effective,denominator,trace=null;
  if(sigma<=1e-12) return {effective:scale(w,0),sigma,values:svd.values,denominator:0,trueNorm:0,error:null,zero:true};
  if(method==='power') {
    trace=powerTrace(w,initial,steps);
    if(trace.error) return {effective:null,sigma,values:svd.values,error:trace.error,trace};
    denominator=trace.records.at(-1).estimate/target;effective=scale(w,1/denominator);
  } else if(method==='entry') effective=w.map(row=>row.map(x=>Math.max(-target,Math.min(target,x))));
  else if(method==='singular-cap') {
    const v=svd.v,other=[-v[1],v[0]],basis=[v,other];
    effective=w.map(()=>[0,0]);
    basis.forEach((direction,k)=>{
      const factor=svd.values[k]>target?target/svd.values[k]:1,projected=matvec(w,direction);
      effective.forEach((row,i)=>row.forEach((_,j)=>effective[i][j]+=factor*projected[i]*direction[j]));
    });
  } else {
    denominator=method==='cap'?Math.max(1,sigma/target):method==='frobenius'?svd.frobenius/target:sigma/target;
    effective=scale(w,1/denominator);
  }
  return {effective,sigma,values:svd.values,denominator,trueNorm:singular2(effective).values[0],error:null,trace};
}
export function normalizationGradient(w,h) {
  const s=singular2(w);
  if(s.values[0]<=1e-12||Math.abs(s.values[0]-s.values[1])<=1e-10) return null;
  const effective=scale(w,1/s.values[0]),inner=dot(h.flat(),effective.flat());
  const detached=scale(h,1/s.values[0]);
  return {gradient:h.map((row,i)=>row.map((x,j)=>(x-inner*s.u[i]*s.v[j])/s.values[0])),detached,effective,inner,...s};
}
// Jacobi diagonalization of the symmetric Gram matrix, bounded to dimensions ≤6.
// This diagonalizes all directions; it is not a finite leading-vector power estimate.
export function smallOperatorNorm(a) {
  const gram=multiply(a,transpose(a)),n=gram.length;
  if(n>6) throw Error('This exact-small operator view supports at most six outputs.');
  for(let iteration=0;iteration<100*n*n;iteration++) {
    let p=0,q=0,big=0;
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(gram[i][j])>big){big=Math.abs(gram[i][j]);p=i;q=j;}
    if(big<1e-14) break;
    const angle=.5*Math.atan2(2*gram[p][q],gram[q][q]-gram[p][p]),c=Math.cos(angle),s=Math.sin(angle);
    const app=gram[p][p],aqq=gram[q][q],apq=gram[p][q];
    for(let k=0;k<n;k++)if(k!==p&&k!==q){const kp=gram[k][p],kq=gram[k][q];gram[k][p]=gram[p][k]=c*kp-s*kq;gram[k][q]=gram[q][k]=s*kp+c*kq;}
    gram[p][p]=c*c*app-2*s*c*apq+s*s*aqq;gram[q][q]=s*s*app+2*s*c*apq+c*c*aqq;gram[p][q]=gram[q][p]=0;
  }
  return Math.sqrt(Math.max(0,...gram.map((row,i)=>row[i])));
}
export function convolution(kernel,mode,input,stride=mode==='disjoint'?2:1) {
  const n=input.length,rows=mode==='circular'?Math.ceil(n/stride):Math.floor((n-2)/stride)+1;
  if(n<3||n>6||![1,2].includes(stride)) throw Error('Use 3–6 inputs and stride one or two.');
  const matrix=Array.from({length:rows},(_,i)=>{
    const row=Array(n).fill(0),start=stride*i;
    row[start]=kernel[0];row[(start+1)%n]=kernel[1];return row;
  });
  const kernelNorm=norm(kernel),operatorNorm=smallOperatorNorm(matrix);
  return {matrix,kernelNorm,operatorNorm,normalizedNorm:kernelNorm>1e-12?operatorNorm/kernelNorm:0,output:matvec(matrix,input.slice(0,n)),zero:kernelNorm<=1e-12};
}
export function penaltyValue(r,kind='target-one',strength=1) {
  return strength*(kind==='zero'?r*r:kind==='one-sided'?Math.max(0,r-1)**2:(r-1)**2);
}
export function linearPenalty(w,strength=2,rate=.1,kind='target-one') {
  const r=norm(w),factor=strength===0?0:kind==='zero'?2*strength:kind==='one-sided'&&r<=1?0:r>1e-12?2*strength*(r-1)/r:null;
  const gradient=factor===null?null:w.map(v=>factor*v),updated=gradient?w.map((v,i)=>v-rate*gradient[i]):null;
  return {norm:r,penalty:penaltyValue(r,kind,strength),gradient,updated,updatedNorm:updated?norm(updated):null,updatedPenalty:updated?penaltyValue(norm(updated),kind,strength):null};
}
export const kinkValue=(x,k,extra)=>x+extra*Math.max(0,x-k);
export function probePenalty(knot,extra,probes,kind='target-one',strength=2) {
  const rows=probes.map(x=>{const slope=Math.abs(x-knot)<1e-12&&extra!==0?null:x>knot?1+extra:1;return {x,value:kinkValue(x,knot,extra),slope,contribution:slope===null?null:penaltyValue(Math.abs(slope),kind,strength)};});
  return {rows,penalty:rows.some(x=>x.slope===null)?null:rows.reduce((s,x)=>s+x.contribution,0)/rows.length,globalBound:Math.max(1,Math.abs(1+extra))};
}
export function linearMargin(w,b,x) {
  const logits=matvec(w,x).map((v,i)=>v+b[i]),gap=logits[0]-logits[1],normal=w[0].map((v,i)=>v-w[1][i]),pair=norm(normal),joint=singular2(w).values[0];
  const radius=gap>0?(pair>1e-12?gap/pair:Infinity):0;
  const delta=pair>1e-12?normal.map(v=>-gap*v/(pair*pair)):null;
  return {logits,gap,normal,pair,joint,radius,jointRadius:gap>0?(joint>1e-12?gap/(Math.SQRT2*joint):Infinity):0,delta,boundary:delta?x.map((v,i)=>v+delta[i]):null};
}
export function frozenForward(point,layers,kind='generator') {
  let value=[...point],jacobian=[[1,0],[0,1]],trace=[],atKink=false;
  layers.forEach((layer,index)=>{
    const pre=matvec(layer.weight,value).map((v,i)=>v+layer.bias[i]);
    let derivative=pre.map(()=>1);value=pre;
    if(index<layers.length-1){
      atKink ||= pre.some(x=>Math.abs(x)<1e-12);
      derivative=pre.map(x=>kind==='generator'?(x>0?1:0):(x>0?1:.2));
      value=pre.map(x=>kind==='generator'?Math.max(0,x):(x>=0?x:.2*x));
    } else if(kind==='generator'){
      value=pre.map(x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x)));
      derivative=value.map(x=>x*(1-x));
    }
    jacobian=multiply(layer.weight,jacobian).map((row,i)=>row.map(x=>x*derivative[i]));
    trace.push({pre,value});
  });
  return {value,jacobian,trace,atKink};
}
export function swappedGenerator(layers) {
  return layers.map((layer,i)=>({...layer,weight:layer.weight.map(row=>i===0?[row[1],row[0]]:[...row])}));
}
// PyTorch's stored-buffer order updates u from cached v, then v from updated u.
// This differs from the manuscript's v-first trace only in the chosen initial state.
export function libraryAccess(w,buffers,training=true) {
  let u=[...buffers.u],v=[...buffers.v];
  if(training){u=unit(matvec(w,v));if(!u)return {error:'The cached direction cannot estimate this weight.'};v=unit(matvec(transpose(w),u));if(!v)return {error:'The cached direction cannot estimate this weight.'};}
  const sigma=dot(u,matvec(w,v));
  if(Math.abs(sigma)<=1e-12)return {error:'The effective denominator is zero; choose a usable weight and cache.'};
  return {u,v,sigma,effective:scale(w,1/sigma),error:null};
}
