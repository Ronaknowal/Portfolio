// Complementary independent review; explicit spatial loops and fresh derivative probes.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { frozenVision, windowRead, composeInfluence, dinoDirection, gramGeometry, patchProjection, visionMACs } from '../src/learn/data/vision-transformer-models.js';
const topic = 'vision-transformers-vit-deit-swin-dinov2', directory = `docs/teaching/deep-learning-completion/${topic}`;
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
let seed = 92673, comparisons = 0, maximumError = 0;
const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
function equal(a, b, tolerance = 1e-10) {
  if (Array.isArray(b)) { assert.equal(a.length, b.length); return b.forEach((v, i) => equal(a[i], v, tolerance)); }
  assert.ok(Number.isFinite(a) && Number.isFinite(b));
  const error = Math.abs(a - b); maximumError = Math.max(maximumError, error); comparisons++;
  assert.ok(error <= tolerance, `${a} vs ${b}: ${error} > ${tolerance}`);
}
const asset = read(`public/learn-assets/${topic}/plain-vit.json`);
for (const test of read(`${directory}/independent-forward-fixtures.json`).cases) {
  const result = frozenVision(asset.state, test.image, test);
  equal(result.logits, test.logits, 1e-10); equal(result.features[0], test.cls, 1e-10);
  equal(result.probabilities.reduce((a,b)=>a+b,0), 1);
  if (test.movePositions) equal(result.logits, frozenVision(asset.state, test.image).logits, 1e-10);
}
// Enumerate rooms directly from spatial lower/upper bounds, not the runtime mask.
function spatialReference(values, width, side, shift, bias) {
  return values.map((_, receiver) => {
    const r = Math.floor(receiver / width), c = receiver % width;
    const top = Math.floor((r - shift) / side) * side + shift;
    const left = Math.floor((c - shift) / side) * side + shift;
    let numerator = 0, denominator = 0;
    for (let y = Math.max(0,top); y < Math.min(width,top+side); y++) {
      for (let x = Math.max(0,left); x < Math.min(width,left+side); x++) {
        const mass = Math.exp(bias ? bias[r-y+side-1][c-x+side-1] : 0);
        numerator += mass * values[y*width+x]; denominator += mass;
      }
    }
    return numerator / denominator;
  });
}
for (let trial=0; trial<40; trial++) {
  const side=trial%2+2, shift=trial%side, width=6;
  const values=Array.from({length:36},()=>32*random()-16);
  const bias=Array.from({length:2*side-1},()=>Array.from({length:2*side-1},()=>8*random()-4));
  const actual=windowRead(values,width,width,side,shift,{bias});
  equal(actual.output,spatialReference(values,width,side,shift,bias));
  const first=windowRead(values,width,width,side,0), second=windowRead(first.output,width,width,side,shift);
  equal(second.output,spatialReference(spatialReference(values,width,side,0),width,side,shift));
  const influence=composeInfluence(second.weights,first.weights);
  const donor=trial%36, receiver=(trial*7)%36, edited=[...values]; edited[donor]+=.37;
  const perturbed=spatialReference(spatialReference(edited,width,side,0),width,side,shift);
  equal((perturbed[receiver]-second.output[receiver])/.37,influence[receiver][donor]);
  const teacher=Array.from({length:3},()=>2*random()-1), center=Array.from({length:3},()=>random());
  const student=Array.from({length:3},()=>2*random()-1), tt=.2+random(), ts=.2+random();
  const direction=dinoDirection(teacher,center,student,tt,ts);
  for(let j=0;j<3;j++) { const p=[...student],m=[...student];p[j]+=1e-5;m[j]-=1e-5;
    equal(direction.gradient[j],(dinoDirection(teacher,center,p,tt,ts).loss-dinoDirection(teacher,center,m,tt,ts).loss)/2e-5,2e-8); }
  equal(dinoDirection(teacher,center,student,tt,ts,120).target,direction.target);
  const angles=Array.from({length:4},()=>360*random()), rotation=360*random();
  const geometry=gramGeometry(angles,angles);
  equal(gramGeometry(angles.map(a=>a+rotation),angles).loss,0);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++) equal(geometry.gram[i][j],Math.cos((angles[i]-angles[j])*Math.PI/180));
  const pixels=Array.from({length:4},()=>16*random()), weights=Array.from({length:2},()=>Array.from({length:4},()=>8*random()-4)), biases=[random(),random()];
  equal(patchProjection(pixels,weights,biases).output,weights.map((w,i)=>w.reduce((total,v,j)=>total+v*pixels[j],biases[i])));
}
for(const [h,w,p,c,d,b]of[[96,160,8,3,64,6],[512,384,16,3,768,12],[32,48,4,1,32,2]]) {
  const n=h/p*w/p, s=n+1, result=visionMACs(h,w,p,c,d,b);
  const count=BigInt(n*p*p*c*d)+BigInt(b)*(12n*BigInt(s)*BigInt(d)**2n+2n*BigInt(s)**2n*BigInt(d))+BigInt(d*1000);
  assert.equal(BigInt(result.total),count); equal(result.pairs,2*s*s*d);
}
const result={passed:true,comparisons,maximumError,nativeFullModels:12,spatialTrials:40,dinoFiniteDifferenceCoordinates:120,
  checks:['Independent functional PyTorch conv/LN/SDPA/GELU on 12 new inputs and arbitrary permutations','Direct spatial room enumeration with fresh signed values and nonzero relative bias','Two-layer coefficients versus changed-input derivative','Fresh DINO finite differences and common-shift invariance','Gram angle identity and orthogonal null','Patch projection and exact BigInt MAC totals'],
  reusedEvidence:['native-checks.json','library-checks.json','model-checks.json'],limitations:'No retraining or optional pretrained checkpoint execution.'};
fs.writeFileSync(`${directory}/independent-model-checks.json`,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
