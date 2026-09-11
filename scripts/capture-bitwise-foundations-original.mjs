import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { arrayMapExamples } from '../src/learn/data/array-map-foundations-examples.js';
const recordPath = 'docs/teaching/evidence/bitwise-foundations-original.json';
assert(!fs.existsSync(recordPath));
const sourcePaths = ['src/learn/data/topics/arrays-strings-hash-maps.jsx','src/learn/data/curriculum/blueprints/arrays-strings-hash-maps.js','src/learn/data/practice/arrays-strings-hash-maps.js','src/learn/data/array-map-foundations-examples.js','src/learn/data/array-map-foundations-model.js','src/learn/components/lesson-labs/ArrayMapFoundationsLabs.jsx','src/learn/components/lesson-labs/systems-structures.css'];
const sources = sourcePaths.map(file => {
  const bytes = fs.readFileSync(file), archive = path.join('scratch/bitwise-foundations-original',file);
  fs.mkdirSync(path.dirname(archive), {recursive:true}); fs.writeFileSync(archive,bytes);
  return {path:file,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,archive};
});
const inventory = JSON.parse(execFileSync(process.execPath,['scripts/build-curriculum-inventory.mjs','--topic','arrays-strings-hash-maps'],{encoding:'utf8'}));
fs.writeFileSync(recordPath,JSON.stringify({capturedAt:new Date().toISOString(),topicId:'arrays-strings-hash-maps',sources,inventory,originalExamples:arrayMapExamples,scope:'Preserve existing arrays/text/hash teaching and six programs while adding a connected deeper bitset/word/XOR branch; publication identity and module order remain unchanged.'},null,2)+'\n');
console.log(`Archived ${sources.length} original files and ${Object.keys(arrayMapExamples).length} complete examples.`);
