import fs from 'node:fs';
import assert from 'node:assert/strict';
const labPath='src/learn/components/lesson-labs/DpStateFamiliesLabs.jsx';
let lab=fs.readFileSync(labPath,'utf8');
for(const [before,after] of [
  ['label="intermediate"','label="product"'],
  ['<small>boundary</small></span>{step.live','<small>left</small></span>{step.live'],
  ['<small>boundary</small></span></div>','<small>right</small></span></div>'],
  ['`${branch.count} ways`','`${branch.count} ${branch.count === 1 ? \'way\' : \'ways\'}`'],
]){assert(lab.includes(before),before);lab=lab.replaceAll(before,after);}
fs.writeFileSync(labPath,lab);
const bodyPath='src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx';
let body=fs.readFileSync(bodyPath,'utf8');
for(const [before,after] of [['and130 (1), totaling19','and 130 (1), totaling 19'],['numbers are101 and121','numbers are 101 and 121'],['version returns0','version returns 0'],['version returns1','version returns 1']]){assert(body.includes(before),before);body=body.replaceAll(before,after);}
fs.writeFileSync(bodyPath,body);
