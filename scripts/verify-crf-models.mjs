import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { chainDistribution, independentDistribution, sameWinners, initialFactors, factorPresets, normalizationBranches, legalBio, legalBioPath, BIO_LABELS, logPartition } from '../src/learn/data/crf-models.js';
const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);
const cases = [];
for (const [name, values] of Object.entries(factorPresets)) {
  const result = chainDistribution(values);
  close(Math.exp(logPartition([values.slice(0, 2).map(Math.log), values.slice(2, 4).map(Math.log)], [values.slice(4, 6).map(Math.log), values.slice(6, 8).map(Math.log)])), result.partition);
  close(result.paths.reduce((sum, path) => sum + path.probability, 0), 1);
  result.nodes.forEach(row => close(row[0] + row[1], 1));
  cases.push({ name, partition: result.partition, winners: result.winners });
}
assert.deepEqual(chainDistribution(initialFactors).paths.map(path => path.mass), [3, 24, 1, 2]);
assert.deepEqual(chainDistribution(factorPresets['Pair reverses the choice']).nodes, [[.84, .16], [.48, .52]]);
assert.equal(sameWinners(chainDistribution(factorPresets['Four-way tie']), independentDistribution(factorPresets['Four-way tie'])), true);
assert.deepEqual(chainDistribution(factorPresets['Four-way tie']).winners, ['AA', 'AB', 'BA', 'BB']);
assert.deepEqual(chainDistribution([1,3,2,1,1,1,1,16]).winners, ['BB']);
assert.deepEqual(independentDistribution([1,3,2,1,1,1,1,16]).winners, ['BA']);
const scaled = chainDistribution(initialFactors.map((value, index) => index >= 4 ? value * 2 : value));
scaled.paths.forEach((path, index) => close(path.probability, chainDistribution(initialFactors).paths[index].probability));
const epsilon = 1e-5;
const unary = [[3,1],[1,2]].map(row => row.map(Math.log));
const basePair = [[0,Math.log(4)],[0,0]];
const plus = basePair.map(row => [...row]), minus = basePair.map(row => [...row]);
plus[0][1] += epsilon; minus[0][1] -= epsilon;
close((logPartition(unary, plus) - logPartition(unary, minus)) / (2 * epsilon), .8);
close((Math.log(4) + epsilon - logPartition(unary, plus) - (Math.log(4) - epsilon - logPartition(unary, minus))) / (2 * epsilon), .2);
const branchCases = [[.5,.01,1,'decrease'], [.5,1,.01,'increase'], [.5,1,1,'unchanged'], [.3,1,1,'unchanged'], [.25,3,1,'increase']];
for (const [p,a,b,direction] of branchCases) {
  const result = normalizationBranches(p,a,b);
  close(result.global[0], p*a/(p*a+(1-p)*b));
  close(result.global.reduce((sum,value)=>sum+value,0), 1);
  assert.equal(result.direction, direction);
}
close(normalizationBranches(.25,3,1).global[0], .5);
const expectedLegal = { START: ['O','B-PER','B-ORG'], O: ['O','B-PER','B-ORG'], 'B-PER': ['O','B-PER','I-PER','B-ORG'], 'I-PER': ['O','B-PER','I-PER','B-ORG'], 'B-ORG': ['O','B-PER','B-ORG','I-ORG'], 'I-ORG': ['O','B-PER','B-ORG','I-ORG'] };
for (const [previous, legal] of Object.entries(expectedLegal)) for (const current of BIO_LABELS) assert.equal(legalBio(previous,current), legal.includes(current));
assert.equal(legalBioPath(['B-PER','I-PER','O']), true);
assert.equal(legalBioPath(['O','I-PER','O']), false);
close(Math.exp(logPartition([[0,0,10]], [], [true,true,false])), 2);
assert.equal(logPartition([[0,0,10]], [], [false,false,false]), -Infinity);
for (const bad of ['',0,-1,17,Infinity,NaN]) assert.throws(()=>chainDistribution(initialFactors.map((value,index)=>index ? value : bad)));
for (const bad of ['',0,-1,11,Infinity,NaN]) assert.throws(()=>normalizationBranches(.5,bad,1));
const source = fs.readFileSync('src/learn/data/topics/conditional-random-fields-crf.jsx','utf8');
// Parse the actual JS string literals: a backslash escaping defect becomes a control character.
const { parse } = await import('@babel/parser');
const exampleTree = parse(fs.readFileSync('src/learn/data/crf-examples.js', 'utf8'), { sourceType: 'module' });
const displayedProgram = exampleTree.program.body.find(node => node.type === 'VariableDeclaration' && node.declarations[0].id.name === 'program').declarations[0].init.value;
assert.equal(displayedProgram, fs.readFileSync('public/learn-assets/crf/crf_example.py', 'utf8').replaceAll('\r\n', '\n'));
const tree = parse(source, { sourceType: 'module', plugins: ['jsx'] });
let mathExpressions = 0;
const katex = (await import('katex')).default;
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && ['Math','MathBlock'].includes(node.openingElement.name.name)) {
    const value = node.children.find(child => child.type === 'JSXExpressionContainer')?.expression.value;
    assert.equal(typeof value, 'string');
    assert.ok(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value));
    katex.renderToString(value,{throwOnError:true}); mathExpressions++;
  }
  for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
}
visit(tree);
assert.ok(mathExpressions > 50);
const files = ['src/learn/data/crf-models.js','src/learn/data/topics/conditional-random-fields-crf.jsx','scripts/verify-crf-models.mjs'];
const result = { status:'passed', cases, mathExpressions, checks:['independent log-space recurrence versus direct products','normalization, marginal sums, contrasting and null presets','full tie-set comparison','global factor-scale invariance','logZ and gold-log-likelihood finite differences','label-bias reverse/equal/changed-prior/transfer','all 30 BIO predecessor edges, impossible and legal masked partition','invalid inputs rejected','actual parsed JSX math strings render without errors'], files:Object.fromEntries(files.map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/crf-models.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({status:result.status, presets:cases.length, mathExpressions}));
