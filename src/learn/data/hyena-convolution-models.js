import {dense,layerNorm,softmax} from './sequence-tensor-operations.js';
// Convolution arrays are time-major. Dense weights use output-by-input storage.
export const alphabet='ACGTDNRS';
export const classes=['EI','IE','N'];
export const dot=(a,b)=>a.reduce((sum,value,i)=>sum+value*b[i],0);
export const nextPowerOfTwo=n=>2**Math.ceil(Math.log2(n));
export const directConvolution=(values,kernel)=>values.map((_,t)=>values.reduce((sum,value,j)=>sum+(j<=t&&t-j<kernel.length?value*kernel[t-j]:0),0));
export function fullConvolution(values,kernel){const result=Array(values.length+kernel.length-1).fill(0);values.forEach((value,j)=>kernel.forEach((h,r)=>result[j+r]+=value*h));return result;}
export function circularConvolution(values,kernel){if(kernel.length>values.length)throw Error('Circular demonstration requires kernel length at most signal length.');return values.map((_,t)=>kernel.reduce((sum,h,r)=>sum+h*values[(t-r+values.length)%values.length],0));}
export const toeplitz=(kernel,length)=>Array.from({length},(_,t)=>Array.from({length},(_,j)=>j<=t&&t-j<kernel.length?kernel[t-j]:0));
export function fft(real,imaginary=null,inverse=false){
 const n=real.length;if(n<1||(n&(n-1))!==0)throw Error('Radix-two FFT needs a positive power-of-two length.');
 const re=[...real],im=imaginary?[...imaginary]:Array(n).fill(0);
 for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
 for(let size=2;size<=n;size*=2){const angle=(inverse?2:-2)*Math.PI/size,wr=Math.cos(angle),wi=Math.sin(angle);for(let start=0;start<n;start+=size){let ar=1,ai=0;for(let j=0;j<size/2;j++){const left=start+j,right=left+size/2,tr=ar*re[right]-ai*im[right],ti=ar*im[right]+ai*re[right];re[right]=re[left]-tr;im[right]=im[left]-ti;re[left]+=tr;im[left]+=ti;[ar,ai]=[ar*wr-ai*wi,ar*wi+ai*wr];}}}
 if(inverse)for(let i=0;i<n;i++){re[i]/=n;im[i]/=n;}return{real:re,imaginary:im};
}
const padded=(values,size)=>Array.from({length:size},(_,i)=>values[i]??0);
const multiplySpectra=(a,b)=>({real:a.real.map((x,i)=>x*b.real[i]-a.imaginary[i]*b.imaginary[i]),imaginary:a.real.map((x,i)=>x*b.imaginary[i]+a.imaginary[i]*b.real[i])});
export function fftConvolution(values,kernel){const size=nextPowerOfTwo(values.length+kernel.length-1),input=padded(values,size),filter=padded(kernel,size),inputSpectrum=fft(input),filterSpectrum=fft(filter),product=multiplySpectra(inputSpectrum,filterSpectrum),inverse=fft(product.real,product.imaginary,true);return{size,input,filter,inputSpectrum,filterSpectrum,product,full:inverse.real.slice(0,values.length+kernel.length-1),output:inverse.real.slice(0,values.length)};}
export function blockedConvolution(values,kernel,blockSize){
 const size=nextPowerOfTwo(blockSize+kernel.length-1),filterSpectrum=fft(padded(kernel,size)),full=Array(values.length+kernel.length-1).fill(0),blocks=[];
 for(let start=0;start<values.length;start+=blockSize){const input=values.slice(start,start+blockSize),spectrum=fft(padded(input,size)),product=multiplySpectra(spectrum,filterSpectrum),convolved=fft(product.real,product.imaginary,true).real.slice(0,input.length+kernel.length-1);convolved.forEach((v,j)=>full[start+j]+=v);blocks.push({start,input,padded:padded(input,size),spectrum,product,convolved,after:[...full]});}
 return{size,filterSpectrum,blocks,full,output:full.slice(0,values.length)};
}
export function gatedConvolution(values,kernel,query,key,preceding=null){const n=values.length,first=preceding?directConvolution(values,preceding):[...values],transmitted=first.map((v,i)=>v*key[i]),filtered=directConvolution(transmitted,kernel),output=filtered.map((v,t)=>v*query[t]),matrix=Array.from({length:n},(_,t)=>Array.from({length:n},(_,j)=>j>t?0:preceding?Array.from({length:t-j+1},(_,i)=>{const m=j+i;return query[t]*(kernel[t-m]??0)*key[m]*(preceding[m-j]??0);}).reduce((a,b)=>a+b,0):query[t]*(kernel[t-j]??0)*key[j]));return{first,transmitted,filtered,output,matrix,contributions:matrix.map(row=>row.map((c,j)=>c*values[j]))};}
export function gatePaths(kernel,query,key,preceding,t,j){if(j>t)return[];return Array.from({length:t-j+1},(_,i)=>{const m=j+i,factors=[query[t],kernel[t-m]??0,key[m],preceding[m-j]??0];return{m,factors,coefficient:factors.reduce((a,b)=>a*b,1)};});}
export function modalScan(values,residues,poles,{initial=null,resetAt=-1}={}){let state=initial?[...initial]:poles.map(()=>0);return values.map((input,t)=>{if(t===resetAt)state=poles.map(()=>0);const before=[...state];state=state.map((v,i)=>poles[i]*v+input);const contributions=state.map((v,i)=>residues[i]*v);return{before,input,state:[...state],contributions,output:contributions.reduce((a,b)=>a+b,0)};});}
export const modalFilter=(residues,poles,length)=>Array.from({length},(_,r)=>residues.reduce((sum,R,i)=>sum+R*poles[i]**r,0));
export function finiteTruncation(values,kernel,retained){const approximation=kernel.map((h,r)=>r<retained?h:0),full=directConvolution(values,kernel),output=directConvolution(values,approximation),omitted=kernel.reduce((sum,h,r)=>sum+(r>=retained?Math.abs(h):0),0),bound=Math.max(...values.map(Math.abs))*omitted;return{approximation,full,output,omitted,bound,errors:output.map((v,i)=>Math.abs(v-full[i]))};}
// erf from convergent regularized-gamma series/continued fraction (a=1/2).
// Native SciPy checks cover the switch and tails; this avoids a GELU variant change.
export function erfPrecise(value){
 if(value===0)return 0;const sign=Math.sign(value),x=value*value,a=.5,logGamma=.5723649429247001;if(x>745)return sign;
 const factor=Math.exp(-x+a*Math.log(x)-logGamma);
 if(x<1.5){let ap=a,term=1/a,sum=term;for(let n=1;n<=200;n++){ap++;term*=x/ap;sum+=term;if(Math.abs(term)<Math.abs(sum)*1e-16)break;}return sign*sum*factor;}
 let b=x+1-a,c=1e300,d=1/b,h=d;for(let n=1;n<=200;n++){const an=-n*(n-a);b+=2;d=an*d+b;if(Math.abs(d)<1e-300)d=1e-300;c=b+an/c;if(Math.abs(c)<1e-300)c=1e-300;d=1/d;const change=d*c;h*=change;if(Math.abs(change-1)<2e-16)break;}return sign*(1-factor*h);
}
export const geluExact=value=>.5*value*(1+erfPrecise(value/Math.SQRT2));
const softplus=value=>Math.max(value,0)+Math.log1p(Math.exp(-Math.abs(value)));
const linear=(values,p,name)=>dense(values,p[name+'.weight'],p[name+'.bias']);
export function generateFilter(model,block,length=60){const p=model.parameters,base=`blocks.${block}.filter`,positions=p[base+'.positions'].slice(0,length),first=positions.map(row=>linear(row,p,base+'.first')),raw=first.map(row=>linear(row.map(Math.sin),p,base+'.last')),window=p[base+'.time'].slice(0,length).map(time=>p[base+'.decay'].map(a=>Math.exp(-time*softplus(a)))),kernel=raw.map((row,t)=>row.map((v,c)=>v*window[t][c]));return{positions,first,raw,window,kernel};}
export const prepareHyenaModel=model=>model.kind==='linear'?null:[generateFilter(model,0),generateFilter(model,1)];
export function spliceForward(sequence,model,{kernelLimit=null,gatesOff=false,preparedFilters=null}={}){
 const tokens=typeof sequence==='string'?[...sequence].map(char=>alphabet.indexOf(char)):sequence;
 if(tokens.length<1||tokens.length>60||tokens.some(t=>!Number.isInteger(t)||t<0||t>=alphabet.length))throw Error('A supported1–60-symbol sequence is required.');
 const p=model.parameters,stages={},blocks=[];
 if(model.kind==='linear'){if(tokens.length!==60)throw Error('The positional linear model requires all60 input positions.');const oneHot=tokens.flatMap(token=>Array.from({length:8},(_,i)=>i===token?1:0)),logits=linear(oneHot,p,'head');return{logits,probabilities:softmax(logits),prediction:logits.indexOf(Math.max(...logits)),stages:{head:logits},blocks,hidden:null};}
 let hidden=tokens.map(token=>[...p['embedding.weight'][token]]);stages.embedding=hidden;
 for(let block=0;block<2;block++){
  const base=`blocks.${block}`,normalized=hidden.map(row=>layerNorm(row,p[base+'.norm.weight'],p[base+'.norm.bias'],1e-5)),projected=normalized.map(row=>linear(row,p,base+'.project'));
  // Conv1d is cross-correlation: tap2 is current, tap1 previous, tap0 two back.
  const short=projected.map((_,t)=>Array.from({length:48},(_,c)=>p[base+'.short.bias'][c]+p[base+'.short.weight'][c][0].reduce((sum,w,j)=>sum+(t+j-2>=0?w*projected[t+j-2][c]:0),0))),query=short.map(row=>row.slice(0,16)),key=short.map(row=>row.slice(16,32)),value=short.map(row=>row.slice(32));
  if(model.kind==='ungated'||gatesOff){query.forEach(row=>row.fill(1));key.forEach(row=>row.fill(1));}
  const filter=preparedFilters?.[block]||generateFilter(model,block),kernel=filter.kernel.slice(0,tokens.length).map((row,r)=>row.map(v=>kernelLimit!==null&&r>=kernelLimit?0:v)),transmitted=value.map((row,t)=>row.map((v,c)=>v*key[t][c])),convolved=transmitted.map((_,t)=>Array.from({length:16},(_,c)=>{let sum=0;for(let j=0;j<=t;j++)sum+=kernel[t-j][c]*transmitted[j][c];return sum;})),mixed=convolved.map((row,t)=>row.map((v,c)=>query[t][c]*(v+p[base+'.skip'][c]*transmitted[t][c]))),projectedOutput=mixed.map(row=>linear(row,p,base+'.output')),residual=hidden.map((row,t)=>row.map((v,c)=>v+projectedOutput[t][c])),feedNorm=residual.map(row=>layerNorm(row,p[base+'.feed_norm.weight'],p[base+'.feed_norm.bias'],1e-5)),feedFirst=feedNorm.map(row=>linear(row,p,base+'.feed.0')),activated=feedFirst.map(row=>row.map(geluExact)),feedOutput=activated.map(row=>linear(row,p,base+'.feed.2'));
  hidden=residual.map((row,t)=>row.map((v,c)=>v+feedOutput[t][c]));
  Object.assign(stages,{[base+'.norm']:normalized,[base+'.project']:projected,[base+'.short']:short,[base+'.filter.first']:filter.first.slice(0,tokens.length),[base+'.filter.last']:filter.raw.slice(0,tokens.length),[base+'.filter']:filter.kernel.slice(0,tokens.length),[base+'.output']:projectedOutput,[base+'.feed_norm']:feedNorm,[base+'.feed.0']:feedFirst,[base+'.feed.1']:activated,[base+'.feed.2']:feedOutput,[base]:hidden});
  blocks.push({normalized,projected,short,query,key,value,filter,kernel,transmitted,convolved,mixed,residual,feedNorm,feedFirst,activated,feedOutput,hidden});
 }
 const logits=linear(hidden.at(-1),p,'head');stages.head=logits;return{logits,probabilities:softmax(logits),prediction:logits.indexOf(Math.max(...logits)),stages,blocks,hidden};
}
