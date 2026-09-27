import { dense, softmax } from './sequence-tensor-operations.js';
const addScaled = (z, direction, scale) => z.map((v, i) => v + scale * direction[i]);
export const matrixVector = (matrix, vector) => matrix.map(row => row.reduce((sum, v, i) => sum + v * vector[i], 0));

export function odeStep(field, time, state, step, method = 'rk4') {
  const k1 = field(time, state);
  if (method === 'euler') return { next: addScaled(state, k1, step), stages: [{ time, state, derivative: k1 }], nfe: 1 };
  if (method !== 'rk4') throw new Error('Choose Euler or classical RK4.');
  const z2 = addScaled(state, k1, step / 2), k2 = field(time + step / 2, z2);
  const z3 = addScaled(state, k2, step / 2), k3 = field(time + step / 2, z3);
  const z4 = addScaled(state, k3, step), k4 = field(time + step, z4);
  return { next: state.map((v, i) => v + step * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6), stages: [{time,state,derivative:k1},{time:time+step/2,state:z2,derivative:k2},{time:time+step/2,state:z3,derivative:k3},{time:time+step,state:z4,derivative:k4}], nfe: 4 };
}
export function fixedOde(field, initial, endpoint = 1, steps = 16, method = 'rk4') {
  if (!Number.isInteger(steps) || steps < 1 || steps > 1024 || !Number.isFinite(endpoint)) throw new Error('Use a finite endpoint and 1–1024 steps.');
  const trace = [[...initial]], stages = [], h = endpoint / steps;
  for (let i = 0; i < steps; i++) { const step = odeStep(field, i*h, trace.at(-1), h, method); trace.push(step.next); stages.push(step.stages); }
  return { trace, stages, nfe: steps * (method === 'euler' ? 1 : 4), endpoint: trace.at(-1) };
}
export function matrixExponential2(matrix, time = 1) {
  const [[a,b],[c,d]] = matrix, center=(a+d)/2, half=(a-d)/2, delta=half*half+b*c, factor=Math.exp(center*time);
  let cosine, sinc;
  if (Math.abs(delta*time*time)<1e-8) { const x=delta*time*time; cosine=1+x/2+x*x/24; sinc=time*(1+x/6+x*x/120); }
  else if(delta>0) {const r=Math.sqrt(delta);cosine=Math.cosh(r*time);sinc=Math.sinh(r*time)/r;}
  else {const r=Math.sqrt(-delta);cosine=Math.cos(r*time);sinc=Math.sin(r*time)/r;}
  return [[factor*(cosine+sinc*half),factor*sinc*b],[factor*sinc*c,factor*(cosine-sinc*half)]];
}
export function linearOdeComparison(matrix, initial, endpoint, steps, method) {
  const result=fixedOde((_,z)=>matrixVector(matrix,z),initial,endpoint,steps,method);
  const exact=matrixVector(matrixExponential2(matrix,endpoint),initial);
  const exactTrace=Array.from({length:129},(_,i)=>matrixVector(matrixExponential2(matrix,endpoint*i/128),initial));
  const residuals=result.trace.map((row,i)=>{const reference=matrixVector(matrixExponential2(matrix,endpoint*i/steps),initial);return row.map((v,j)=>v-reference[j]);});
  return {...result,exact,exactTrace,residuals,error:Math.hypot(...result.endpoint.map((v,i)=>v-exact[i])),radius:Math.hypot(...result.endpoint)};
}
export function adaptiveHeun(rates, initial, endpoint=.4, tolerance=.01, initialStep=.1) {
  let time=0, state=[...initial], step=initialStep, accepted=0, rejected=0;
  const attempts=[], trace=[{time,state:[...state]}];
  while(time<endpoint) {
    if(attempts.length>=10000 || step<1e-14) throw new Error('The tolerance or initial step needs too much work for this bounded investigation.');
    step=Math.min(step,endpoint-time);
    const first=state.map((v,i)=>rates[i]*v), euler=addScaled(state,first,step), second=euler.map((v,i)=>rates[i]*v), heun=state.map((v,i)=>v+step*(first[i]+second[i])/2);
    const ratios=state.map((v,i)=>(heun[i]-euler[i])/(.01*tolerance+tolerance*Math.max(Math.abs(v),Math.abs(heun[i]))));
    const ratio=Math.sqrt(ratios.reduce((s,v)=>s+v*v,0)/state.length), allow=ratio<=1;
    attempts.push({time,step,before:[...state],euler,heun,ratio,accepted:allow});
    if(allow) {time+=step;state=heun;accepted++;trace.push({time,state:[...state]});}else rejected++;
    step*=ratio===0 ? 5 : Math.min(5,Math.max(.1,.9/Math.sqrt(ratio)));
  }
  const exact=initial.map((v,i)=>v*Math.exp(rates[i]*endpoint));
  return {attempts,trace,accepted,rejected,nfe:2*attempts.length,endpoint:state,exact,error:Math.hypot(...state.map((v,i)=>v-exact[i]))};
}
export function scalarOdeGradient(rate=-.7, initial=1.2, target=.4, endpoint=1.3, steps=4, method='euler') {
  // Integrating the tangent dz/dtheta differentiates this exact fixed-step program,
  // including the zero amplification case without division by that amplification.
  const tangent=fixedOde((_,[z,s])=>[rate*z,z+rate*s],[initial,0],endpoint,steps,method);
  const [prediction,sensitivity]=tangent.endpoint, gradient=(prediction-target)*sensitivity;
  const exactPrediction=initial*Math.exp(rate*endpoint), exactGradient=(exactPrediction-target)*endpoint*exactPrediction;
  const epsilon=1e-6, evaluate=r=> {const z=fixedOde((_,s)=>[r*s[0]],[initial],endpoint,steps,method).endpoint[0];return .5*(z-target)**2;};
  const backward=fixedOde((_,[z,a])=>[rate*z,-rate*a,-a*z],[prediction,prediction-target,0],-endpoint,steps,'rk4');
  return {prediction,sensitivity,gradient,centralDifference:(evaluate(rate+epsilon)-evaluate(rate-epsilon))/(2*epsilon),exactPrediction,exactGradient,continuousAccumulator:backward.endpoint[2],reconstructedInitial:backward.endpoint[0],trace:tangent.trace,backwardTrace:backward.trace};
}
export function liftedOde(points, endpoint, threshold) {
  return {final:points.map(x=>[x,endpoint*x*x]),predicted:points.map(x=>Number(endpoint*x*x>threshold)),tracks:points.map(x=>Array.from({length:33},(_,i)=>[x,endpoint*i/32*x*x]))};
}
export function observationOde(times, values, query, omit=-1) {
  if(times.length!==values.length || times.some((t,i)=>!Number.isFinite(t)||t<0||(i>0&&t<=times[i-1])))throw new Error('Observation times must increase strictly.');
  let state=0,last=0;const events=[], segments=[];
  const decay=(from,to,value)=>Array.from({length:25},(_,i)=>{const t=from+(to-from)*i/24;return[t,value*Math.exp(-.5*(t-from))];});
  for(let i=0;i<times.length;i++) {
    if(i===omit||times[i]>query)continue;
    const before=state*Math.exp(-.5*(times[i]-last));
    segments.push(decay(last,times[i],state));state=.7*before+.3*values[i];events.push({index:i,time:times[i],value:values[i],before,after:state});last=times[i];
  }
  segments.push(decay(last,query,state));state*=Math.exp(-.5*(query-last));
  return {state,query,events,segments};
}
export function odeClassifier(snapshot, raw, metadata, steps=4, method='rk4', endpoint=1) {
  const standardized=raw.map((v,i)=>(v-metadata.mean[i])/metadata.scale[i]), w=snapshot.state;
  const linear=(z,name)=>dense(z,w[name+'.weight'],w[name+'.bias']);
  const field=(time,z,prefix='field')=>linear(linear([...z,time],prefix+'.input').map(Math.tanh),prefix+'.output');
  const initial=snapshot.kind==='augmented_ode'?[...standardized,0,0]:standardized;
  let result;
  if(snapshot.kind.includes('ode')) result=fixedOde(field,initial,endpoint,steps,method);
  else {const trace=[initial];if(snapshot.kind==='residual')for(let i=0;i<4;i++)trace.push(addScaled(trace.at(-1),field(i/4,trace.at(-1),`blocks.${i}`),.25));result={trace,nfe:snapshot.kind==='residual'?4:0,endpoint:trace.at(-1),stages:[]};}
  const logits=linear(result.endpoint,'readout');
  return {...result,standardized,logits,probabilities:softmax(logits)};
}
export function densityOde(matrix, endpoint) {
  const exponential=matrixExponential2(matrix,endpoint),trace=matrix[0][0]+matrix[1][1];
  return {trace,exponential,logDensityChange:-endpoint*trace,volume:Math.exp(endpoint*trace),density:Math.exp(-endpoint*trace),corners:[[0,0],[1,0],[1,1],[0,1]].map(p=>matrixVector(exponential,p)),probes:[[-1,-1],[-1,1],[1,-1],[1,1]].map(p=>p.reduce((sum,v,i)=>sum+v*matrixVector(matrix,p)[i],0))};
}
