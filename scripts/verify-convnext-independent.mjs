import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { blockBudget, normalizeLocations, responseNormalize, foldBranches } from '../src/learn/data/convnext-models.js';
const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/convnext-independent-native.json'));
assert.equal(native.passed, true);
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
for (const [file, value] of Object.entries(native.sourceHashes)) assert.equal(hash(file), value, file);
let maximumError = 0;
const close = (a,b) => { const left=a.flat(Infinity),right=b.flat(Infinity); assert.equal(left.length,right.length); left.forEach((v,i)=> {const delta=Math.abs(v-right[i]);maximumError=Math.max(maximumError,delta);assert.ok(delta<1e-10,`${v} != ${right[i]}`);}); };
for(const row of native.fixtures.normalization) for(const mode of ['channels','whole']) close(normalizeLocations(row.input,mode).output,row[mode]);
for(const row of native.fixtures.grn) close(responseNormalize(row.maps,row.scale,row.shift).output,row.output);
for(const row of native.fixtures.budgets) assert.equal(blockBudget(row.channels,4,7,3,5,row.version).parameters,row.parameters);
for(const row of native.fixtures.folding) {const output=foldBranches(row.config);close(output.separate,row.expected);close(output.folded,row.expected);}
const files = ['src/learn/data/convnext-models.js','src/learn/data/topics/convnext-modern-cnn-designs.jsx','src/learn/components/lesson-labs/ConvNeXtLabs.jsx','src/learn/components/lesson-labs/convnext.css','scripts/verify-convnext-independent.mjs'];
const report={passed:true,reviewer:'root; separate from topic author convnext_finish',groups:native.checks,maximumError,sourceHashes:Object.fromEntries(files.map(file=>[file,hash(file)])),limits:'Numerical source review; browser/layout evidence is separate. No fits or published benchmark claims remeasured.'};
fs.writeFileSync('docs/teaching/evidence/convnext-independent.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:true,groups:report.groups.length,maximumError}));
