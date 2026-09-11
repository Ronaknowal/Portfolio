const fs = require('node:fs');
const assert = require('node:assert/strict');
const { parse } = require('@babel/parser');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const pde = 'partial-differential-equations-conservation-boundary-conditions';
const numerical = 'numerical-pdes-grids-finite-elements-stability';
const body = `src/learn/data/topics/${pde}.jsx`;
const source = fs.readFileSync(body, 'utf8');
parse(source, { sourceType: 'module', plugins: ['jsx'] });
assert(source.includes('export default') && /<Sources(?:\s|>)/.test(source));
const manifestPath = 'src/learn/data/lesson-manifest.json';
const manifest = read(manifestPath);
assert(!manifest[pde]);
manifest[pde] = `./topics/${pde}.jsx`;
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
const indexPath = 'src/learn/data/curriculum/blueprints/index.js';
let index = fs.readFileSync(indexPath, 'utf8');
assert(!index.includes('numericalPdesBlueprint'));
index = `import numericalPdesBlueprint from './${numerical}.js';\n` + index;
const closing = index.lastIndexOf('};');
assert(closing > 0);
index = index.slice(0, closing) + '  "Numerical PDEs: Grids, Finite Elements & Stability": numericalPdesBlueprint,\n' + index.slice(closing);
fs.writeFileSync(indexPath, index);
const ledgerPath = 'docs/teaching/dsa-math-foundations-progress.json';
const ledger = read(ledgerPath);
Object.assign(ledger.topics.find(topic => topic.id === pde), {
  status: 'in-progress', designRecord: 'docs/teaching/PARTIAL-DIFFERENTIAL-EQUATIONS-LESSON-DESIGN.md',
  reviewStage: 'Complete published draft; author native/prose verification complete, actual-route browser verification underway. Independent and integration review remain pending.'
});
Object.assign(ledger.topics.find(topic => topic.id === numerical), {
  status: 'in-progress', designRecord: 'docs/teaching/NUMERICAL-PDES-LESSON-DESIGN.md',
  reviewStage: 'Complete root design and proposed fixtures assessed independently; individual brief indexed; full author implementation underway. No body is registered yet.'
});
fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
console.log('Registered complete PDE56 draft and assessed Numerical PDE57 brief; neither is implementation-reviewed.');
