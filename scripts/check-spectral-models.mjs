import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as model from '../src/learn/data/spectral-regularization-models.js';
const id='spectral-normalization-gradient-penalty', folder='docs/teaching/deep-learning-completion/'+id+'/';
const native=JSON.parse(fs.readFileSync(folder+'native-fixtures.json')), sensitivity=JSON.parse(fs.readFileSync(folder+'native-sensitivity-results.json'));
const data=JSON.parse(fs.readFileSync('docs/teaching/drafts/'+id+'/calculated-inputs.json'));
const checks=[];
function close(name, actual, expected, tolerance=1e-10) {
 const a=[actual].flat(Infinity), b=[expected].flat(Infinity); assert.equal(a.length,b.length,name);
 const error=Math.max(...a.map((v,i)=>Math.abs(v-b[i]))); assert.ok(Number.isFinite(error)&&error<=tolerance,name+': '+error);
 checks.push({name,passed:true,maxAbsoluteError:error,tolerance});
}
for(const [i,row] of native.matrix.entries()) close('Native SVD matrix '+i,model.singular2(row.weight).values,row.singular);
for(const row of JSON.parse(fs.readFileSync(folder+'operator-fixtures.json')).cases) {
 const result=model.convolution(row.kernel,row.mode,Array(row.n).fill(1),row.stride);
 close('Basis matrix '+[row.n,row.mode,row.stride,row.kernel],result.matrix,row.matrix,0);
 close('Full operator SVD '+[row.n,row.mode,row.stride,row.kernel],result.operatorNorm,row.norm);
}
for(const row of native.linear) {
 const result=model.linearPenalty(row.weight,2,.1,row.kind);close('Penalty '+[row.weight,row.kind],result.penalty,row.penalty);
 if(!row.zeroConvention||row.kind!=='target-one')close('Penalty derivative '+[row.weight,row.kind],result.gradient,row.gradient);
 else assert.equal(result.gradient,null);
}
for(const row of native.probes) assert.deepEqual(model.probePenalty(2,4,row.points).rows.map(x=>x.slope),row.slopes);
close('Moved probe penalty',model.probePenalty(2,4,[-.5,.5,2.5]).penalty,32/3);
close('Normalization derivative oracle',model.normalizationGradient([[2,1],[0,1]],[[1,-.3],[.2,.7]]).gradient,sensitivity.normalization_gradient.autograd);
assert.equal(model.normalizationGradient([[1,0],[0,1]],[[1,0],[0,1]]),null);
for(const [name,w,u] of [['generic',[[3,0],[0,1]],[1,1]],['orthogonal',[[3,0],[0,1]],[0,1]],['slow_gap',[[1.01,0],[0,1]],[1,1]],['rotated_weight_cached_vector',[[1,0],[0,3]],[1,0]]]) {
 const rows=sensitivity.power[name];if(!rows)throw Error('Missing native power fixture '+name);
 const trace=model.powerTrace(w,u,rows.length).records;
 close('Power '+name,trace.map(row=>[...row.u,...row.v,row.estimate,row.trueNorm]),rows.map(row=>[...row.u,...row.v,row.estimate,row.normalized_operator_norm]));
}
let buffers={u:[1/Math.SQRT2,1/Math.SQRT2],v:[1/Math.SQRT2,1/Math.SQRT2]};
for(const [i,row] of native.library.entries()){const r=model.libraryAccess([[2,1],[0,1]],buffers,row.training);close('Actual PyTorch access '+i,[r.u,r.v,r.effective],[row.u,row.v,row.effective]);buffers=r;}
for(const fit of data.fits){const key=fit.method+'-'+fit.seed,n=native.frozen.find(row=>row.key===key);
 close(key+' all256 generated',data.evaluation_latents.map(z=>model.frozenForward(z,fit.generator_layers).value),fit.generated,1e-6);
 close(key+' all400 critic scores',data.measurements.map(x=>model.frozenForward(x,fit.critic_layers,'critic').value[0]),fit.critic_values,1e-6);
 close(key+' all1681 gradients',fit.grid_coordinates.map(x=>model.frozenForward(x,fit.critic_layers,'critic').jacobian[0]),fit.grid_gradient,1e-6);
 close(key+' fresh critic',n.points.map(x=>model.frozenForward(x,fit.critic_layers,'critic').value),n.critic);
 close(key+' fresh gradients',n.points.map(x=>model.frozenForward(x,fit.critic_layers,'critic').jacobian[0]),n.gradients);
 close(key+' fresh generator',n.points.map(x=>model.frozenForward(x,fit.generator_layers).value),n.generated);
 const swapped=model.swappedGenerator(fit.generator_layers);close(key+' joint coordinate/basis swap',data.evaluation_latents.map(z=>model.frozenForward([...z].reverse(),swapped).value),data.evaluation_latents.map(z=>model.frozenForward(z,fit.generator_layers).value),0);
}
for(const method of ['exact','cap','frobenius','entry','singular-cap','power']){const r=model.normalizeMatrix([[2,1],[0,1]],method);assert.ok(r.effective.flat().every(Number.isFinite));}
close('Positive uniform rescaling null',model.normalizeMatrix([[2,1],[0,1]]).effective,model.normalizeMatrix([[4,2],[0,2]]).effective);
assert.ok(model.powerTrace([[1,0],[0,0]],[0,1]).error);assert.ok(model.powerTrace([[1,0],[0,1]],[0,0]).error);
assert.equal(model.normalizeMatrix([[0,0],[0,0]]).trueNorm,0);
close('Fresh winning radius',model.linearMargin([[1,0],[0,1]],[0,0],[1.5,.25]).radius,1.25/Math.SQRT2);
assert.equal(model.linearMargin([[1,0],[1,0]],[1,0],[0,0]).radius,Infinity);
assert.equal(model.linearMargin([[1,0],[1,0]],[0,0],[0,0]).radius,0);
fs.writeFileSync(folder+'model-checks.json',JSON.stringify({topicId:id,passed:true,checks,limits:['Native frozen models reused; no new GAN training.','Browser rendering and interactions checked separately.']},null,2)+'\n');
console.log(checks.length+' spectral model and native-reference checks passed.');
