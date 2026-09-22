import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { shiftedTracks, encodeSequence, sequenceStep, sequenceIndex, sequenceTokens } from '../src/learn/data/seq2seq-models.js';
const output='docs/teaching/evidence/seq2seq-independent.json';
fs.writeFileSync(output,JSON.stringify({passed:false,status:'running'}));
const oracle=JSON.parse(fs.readFileSync('docs/teaching/evidence/seq2seq-independent-native.json'));
const {weights}=JSON.parse(fs.readFileSync('public/learn-code/sequence-to-sequence-encoder-decoder/seed-one-inference.json'));
assert.equal(oracle.passed,true);
let maxError=0, values=0;
function close(actual, expected) {
 const left=actual.flat(Infinity),right=expected.flat(Infinity);assert.equal(left.length,right.length);assert.ok(left.length>0);
 for(let i=0;i<left.length;i++){const difference=Math.abs(left[i]-right[i]);assert.ok(Number.isFinite(difference)&&difference<1e-10,`value${i}: ${difference}`);maxError=Math.max(maxError,difference);values++;}
}
assert.throws(()=>close([NaN],[0]));assert.throws(()=>close([1],[2]));assert.throws(()=>close([],[]));
for(const row of oracle.cases){
 const tracks=shiftedTracks(row.form,row.padding);assert.deepEqual(tracks.inputs,row.inputs);assert.deepEqual(tracks.targets,row.targets);
 const encoded=encodeSequence(weights,row.lemma,row.feature);close(encoded.context,row.context);
 let state=encoded.context;
 for(let i=0;i<row.inputs.length;i++){
  const result=sequenceStep(weights,state,sequenceIndex[row.inputs[i]]);close(result.state,row.states[i]);close(result.probabilities,row.probabilities[i]);state=result.state;
 }
 if(row.padding>0) assert.equal(tracks.inputs[row.form.length+1],'<eos>');
}
// At the longest target there is no padding, so no next input after its EOS target.
const longest=oracle.cases.find(row=>row.padding===0);assert.equal(shiftedTracks(longest.form).inputs.length,longest.form.length+1);
const files=['src/learn/data/seq2seq-models.js','src/learn/components/lesson-labs/Seq2SeqLabs.jsx','src/learn/components/lesson-labs/seq2seq-labs.css','src/learn/data/topics/sequence-to-sequence-encoder-decoder.jsx','scripts/verify-seq2seq-independent.py','scripts/verify-seq2seq-independent.mjs'];
fs.writeFileSync(output,JSON.stringify({passed:true,reviewer:'convnext_finish',cases:oracle.cases.length,comparedValues:values,maxAbsoluteDifference:maxError,checks:['Actual public batch shift agrees with visual tracks through EOS and every padded input','Browser sequential source contexts match independent native unsorted packed batch, including min/max allowed source lengths','Every browser teacher-forced state and probability matches native batch through real tokens, EOS and ignored padding','Nonfinite,empty and deliberately incorrect fixtures fail the comparison guard'],sources:Object.fromEntries(files.map(file=>[file,createHash('sha256').update(fs.readFileSync(file)).digest('hex')])),limits:'Complements existing native/generated/beam evidence without rerunning fits. Browser interaction and final build evidence are separate.'},null,2)+'\n');
console.log(JSON.stringify({passed:true,cases:oracle.cases.length,values,maxError}));
