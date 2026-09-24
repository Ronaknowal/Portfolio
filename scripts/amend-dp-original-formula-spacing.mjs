import fs from 'node:fs';
import assert from 'node:assert/strict';
const path='src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx';
const before=fs.readFileSync(path,'utf8');
const formula='current=max(next_one,reward+next_two)';
assert.equal(before.split(formula).length,2);
fs.writeFileSync(path,before.replace(formula,'current = max(next_one, reward + next_two)'));
