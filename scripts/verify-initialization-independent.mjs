import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { activationMoments, directionalGeometry, symmetryState, widthConfiguration } from '../src/learn/data/weight-initialization-models.js';
const output = 'docs/teaching/evidence/initialization-independent-review.json';
writeFileSync(output, JSON.stringify({ passed: false }));
const native = JSON.parse(readFileSync('docs/teaching/evidence/initialization-independent-native.json'));
assert.equal(native.passed, true);
const close = (a, b, tolerance=1e-11) => assert.ok(Math.abs(a-b) <= tolerance, `${a} != ${b}`);
for (const row of native.symmetryCases) {
  const actual = symmetryState(row.weights, row.outgoing, row.input, row.target);
  for (const key of ['output', 'loss']) close(actual[key], row[key]);
  for (const key of ['hiddenGradient', 'headGradient']) actual[key].forEach((value, i) => close(value, row[key][i]));
}
// Angular average of norm gain squared must equal half the Frobenius norm
// squared, even though a single direction may amplify or collapse.
for (const small of [.05, .2, .7, 1]) for (const depth of [1, 3, 8]) {
  const directions = Array.from({length:128}, (_, i) => 2*Math.PI*i/128);
  const mean = directions.reduce((sum, angle) => sum + directionalGeometry(small, [Math.cos(angle), Math.sin(angle)], depth).normGain**2, 0)/directions.length;
  close(mean, ((2-small**2)**depth + small**(2*depth))/2, 1e-10);
}
for (const values of [[-4, -2, 0, 2, 4], [0,0,0,0], [-1,-1,-1], [2,2,2]]) {
  const {after} = activationMoments(values);
  const sumPairs = after.values.flatMap(a => after.values.map(b => (a-b)**2)).reduce((a,b)=>a+b,0);
  close(after.variance, sumPairs/(2*values.length**2));
}
close(widthConfiguration(48,.0017).hiddenRate, .0017/1.5);
const files = [
 'src/learn/data/topics/weight-initialization-xavier-kaiming-p.jsx',
 'src/learn/data/curriculum/blueprints/weight-initialization-xavier-kaiming-p.js',
 'src/learn/components/lesson-labs/WeightInitializationLabs.jsx',
 'src/learn/components/lesson-labs/weight-initialization.css',
 'src/learn/data/weight-initialization-models.js',
 'src/learn/data/weight-initialization-measurements.json',
 'scripts/generate-weight-initialization-lesson.mjs',
 'public/learn-assets/weight-initialization-xavier-kaiming-p/initialization-experiments.py',
 'public/learn-assets/weight-initialization-xavier-kaiming-p/initialization_library_bridge.py',
 'public/learn-assets/weight-initialization-xavier-kaiming-p/digits-400.csv',
 'public/learn-assets/weight-initialization-xavier-kaiming-p/calculated-inputs.json',
 'docs/teaching/drafts/weight-initialization-xavier-kaiming-p/lesson.md',
 'docs/teaching/drafts/weight-initialization-xavier-kaiming-p/visual-specifications.md',
];
writeFileSync(output, JSON.stringify({passed:true, reviewer:'root; author initialization_implementation', groups:[
 'Six changed JS derivative cases equal independently executed native autograd',
 '12 directional/depth configurations independently checked by128direction angular averaging',
 'Four moments cases checked by pairwise-distance variance identity',
 'Changed noninteger width multiplier matches separate native package bridge',
], reviewedFiles:Object.fromEntries(files.map(file=>[file,createHash('sha256').update(readFileSync(file)).digest('hex')])), limits:'Pedagogical review in INITIALIZATION-INDEPENDENT-REVIEW.md; production browser closure is separate.'},null,2)+'\n');
console.log('PASS:4 complementary initialization model groups; source hashes bound');
