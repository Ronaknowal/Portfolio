import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { normalizeLocations, responseNormalize, blockBudget, defaultFusion, foldBranches, reconstruct, maxDifference } from '../src/learn/data/convnext-models.js';
const evidence = 'docs/teaching/evidence/convnext-models.json';
writeFileSync(evidence, JSON.stringify({passed:false,status:'running'}));
const json = path => JSON.parse(readFileSync(path, 'utf8'));
const assets='public/learn-assets/convnext-modern-cnn-designs/';
const native=json('docs/teaching/evidence/convnext-native.json'); assert.equal(native.passed,true);
const saved=json(assets+'saved-models.json'),oracle=json(assets+'author-check-results.json');
const checks=[]; let maximumError=0, maximumMseError=0;
const close=(actual,expected,tolerance=1e-6)=>{ assert.equal(actual.flat(Infinity).length,expected.flat(Infinity).length); const difference=maxDifference(actual,expected); assert.ok(Number.isFinite(difference)&&difference<=tolerance,`difference ${difference}, tolerance ${tolerance}`); return difference; };
assert.throws(()=>close([1],[1.1],1e-5));assert.throws(()=>close([],[]));assert.throws(()=>close([NaN],[0]));
checks.push('Numerical guard rejects incorrect, empty and nonfinite subjects');
const locations=[[1,3,7],[101,103,107]],shifted=[[1,3,7],[111,113,117]];
close(normalizeLocations(locations).output,oracle.normalization_axes.three_channel.original[0],1e-12);
close(normalizeLocations([[1,3,7],[105,103,107]]).output,oracle.normalization_axes.three_channel.edited[0],1e-12);
close(normalizeLocations(shifted).output,normalizeLocations(locations).output,0);
close(normalizeLocations([[1,3],[101,103]],'whole').output,oracle.normalization_axes.single_group[0],1e-12);
checks.push('Channel and whole-specimen normalization match independent NumPy axes; changed-cell and exact shift-null fixtures');
const response=responseNormalize([[3,4],[0,12]],[.5,-.5],[0,0]);
close(response.output,[[oracle.grn.output[0][0][0],oracle.grn.output[1][0][0]],[oracle.grn.output[0][0][1],oracle.grn.output[1][0][1]]],1e-12);
close(responseNormalize([[3,4],[0,0]],[.5,-.5],[0,0]).output,[[5.99999880000048,7.99999840000064],[0,0]],1e-12);
close(responseNormalize([[3,4],[0,24]],[0,0],[0,0]).output,[[3,4],[0,24]],0);
assert.equal(responseNormalize([[0,0],[0,0]],[1,-1],[0,0]).output.flat().every(value=>value===0),true);
checks.push('GRN denominator, signed scales, cross-channel contrast, identity and zero-input cases');
assert.equal(blockBudget(96,4,7,14,14,1).parameters,79296);assert.equal(blockBudget(96,4,7,14,14,2).parameters,79968);
assert.equal(blockBudget(64,3,5,14,14,1).parameters,26688);assert.equal(blockBudget(64,3,5,14,14,1).macs,5130496);
for(const width of [4,64,96,256])for(const expansion of [1,3,6])for(const kernel of [3,5,7,9]) {
 const a=blockBudget(width,expansion,kernel,7,11,1),b=blockBudget(width,expansion,kernel,14,22,1),c=blockBudget(width,expansion,kernel,7,11,2);
 assert.equal(b.macs,4*a.macs);assert.equal(b.parameters,a.parameters);assert.equal(c.parameters-a.parameters,(2*expansion-1)*width);assert.equal(c.macs,a.macs);
}
checks.push('Exact block totals plus144 width/expansion/kernel budget cases with spatial scaling and version differences');
const fusion=foldBranches(defaultFusion);close(fusion.foldedKernel,oracle.fusion.folded_kernel.flat(Infinity),1e-12);assert.ok(fusion.difference<1e-10);assert.ok(Math.abs(fusion.folded[12]-oracle.fusion.center_output)<1e-10);
assert.ok(Math.abs(foldBranches(defaultFusion,true).difference-oracle.fusion.branch_relu_noncommutation_error)<1e-10);
for(const gamma of [-3,0,3])for(const variance of [.01,4,9])for(const coefficient of [-3,0,3]) { const config={...defaultFusion,gamma,variance,kernel:defaultFusion.kernel.map((value,i)=>i===4?coefficient:value)};assert.ok(foldBranches(config).difference<1e-10); }
checks.push('Independent folded kernel/bias/output fixture and nonlinear counterexample;27 signed/extreme folding cases');
const start=performance.now();
for(const fixture of native.fixtures) {
 const model=saved.find(row=>row.global_response===fixture.grn);
 const result=reconstruct(model.model_state,fixture.pixels,fixture.mask);
 maximumError=Math.max(maximumError,close(result.output,fixture.output,1e-5));
 maximumMseError=Math.max(maximumMseError,Math.abs(result.mse-fixture.mse));assert.ok(Math.abs(result.mse-fixture.mse)<1e-5);
}
const elapsed=performance.now()-start;
for(const model of saved)for(const example of model.examples) {
 const baseline=reconstruct(model.model_state,example.input,example.visible_patches),changed=example.input.map(row=>[...row]);changed[0][0]=1-changed[0][0];
 const hidden=reconstruct(model.model_state,changed,example.visible_patches);close(hidden.output,baseline.output,0);assert.notEqual(hidden.mse,baseline.mse);
 const visible=example.input.map(row=>[...row]);visible[0][2]=1-visible[0][2];assert.ok(maxDifference(reconstruct(model.model_state,visible,example.visible_patches).output,baseline.output)>1e-4);
 assert.throws(()=>reconstruct(model.model_state,example.input,Array.from({length:4},()=>[1,1,1,1])));
}
checks.push('All1536 fresh native outputs and24 losses agree; allfour hidden edits exactly preserve reconstruction, visible contrasts change it and invalid masks reject');
const metrics=json('src/learn/data/convnext-measurements.json');assert.equal(metrics.runs.length,6);assert.ok(metrics.runs.every(run=>run.trace.every(point=>point.development_masked_mse>=0&&point.development_masked_mse<=.3)));assert.ok(metrics.runs.every(run=>!run.model_state&&!run.examples));
checks.push('Allsix conserved traces fit the declared chart domain; topic metadata contains no saved weights');
const files=['src/learn/data/convnext-models.js','src/learn/data/convnext-measurements.json','src/learn/components/lesson-labs/ConvNeXtLabs.jsx','src/learn/components/lesson-labs/convnext.css','src/learn/data/topics/convnext-modern-cnn-designs.jsx','scripts/generate-convnext-lesson.mjs','scripts/verify-convnext-models.mjs'];
const report={passed:true,checks,maximumError,maximumMseError,nodeInferenceMillisecondsFor24Cases:elapsed,sources:Object.fromEntries(files.map(file=>[file,createHash('sha256').update(readFileSync(file)).digest('hex')])),limits:['Timing is a local Node correctness-run observation, not a browser/device benchmark','Actual browser, keyboard, responsive and final integration checks remain with the increment owner']};
writeFileSync(evidence,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
