import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { parse } from '@babel/parser';
import katex from 'katex';
import * as model from '../src/learn/data/single-variable-calculus-models.js';
import { singleVariableCalculusExamples } from '../src/learn/data/single-variable-calculus-examples.js';

const directory = 'scratch/single-variable-calculus-verification';
fs.mkdirSync(directory, { recursive: true });
const data = { rates: [], limits: [], compositions: [], extrema: [], accumulations: [], growth: [], taylors: [], improper: [], examples: singleVariableCalculusExamples };
for (let t = 4; t <= 12; t++) for (const h of [-1, -.5, -.1, -.01, -.0001, .0001, .01, .1, .5, 1]) data.rates.push(model.motionRate(t / 4, h));
for (const kind of ['smooth', 'hole', 'jump']) for (const epsilon of [.01, .03, .1, .3, .5, .83, .84, .8399999999999999, .8400000000000001, 1]) for (const delta of [.001, .002, .01, .02, .1, .2, .25, .5]) data.limits.push(model.limitGuarantee(kind, epsilon, delta));
for (const x of [-1, -.5, -1 / 3, 0, .25, .5, 1]) for (const h of [-.5, -.1, -.01, 0, .01, .1, .5]) data.compositions.push(model.compositionChange(x, h));
for (const kind of ['motion', 'inflection', 'cusp']) for (let a = 0; a < 16; a++) for (let b = a + 1; b <= 16; b++) data.extrema.push(model.extremaCandidates(kind, a / 4, b / 4));
for (const kind of ['motion', 'inflection', 'cusp']) for (const [a,b] of [[2,2+2*Number.EPSILON],[1,1+Number.EPSILON],[0,1+Number.EPSILON],[0,1e-323],[3-2*Number.EPSILON,3]]) data.extrema.push(model.extremaCandidates(kind,a,b));
for (let upper = 1; upper <= 16; upper++) for (const n of [1, 2, 3, 4, 8, 16, 32, 64, 127, 128]) for (const tag of ['left', 'midpoint', 'right']) data.accumulations.push(model.motionAccumulation(upper / 4, n, tag));
for (let rate = -16; rate <= 16; rate++) for (let period = 1; period <= 8; period++) data.growth.push(model.exponentialRate(rate / 20, period / 4));
for (const kind of ['exp', 'log']) for (let degree = 0; degree <= 12; degree++) for (const x of (kind === 'exp' ? [-2, -1.9, -1, -.25, 0, .01, .5, 1, 2] : [-.9, -.8, -.1, 0, .01, .5, .9, 1, 1.05, 1.5])) data.taylors.push(model.taylorApproximation(kind, degree, x));
for (const kind of ['tail', 'endpoint']) for (const power of [0, .5, .9, 1 - Number.EPSILON, 1, 1 + Number.EPSILON, 1.5, 2, 3]) for (let q = 0; q <= 24; q++) data.improper.push(model.improperPowerIntegral(kind, power, q / 4));
for(const x of [.05,.1,.15,.2]) data.taylors.push(model.taylorApproximation('exp',12,x));
const invalid = [
  () => model.motionPosition(-1), () => model.motionVelocity(Infinity), () => model.motionRate(0, .1), () => model.motionRate(2, 0), () => model.motionRate(2, 1e-10),
  () => model.limitGuarantee('unknown', .1, .1), () => model.limitGuarantee('smooth', NaN, .1), () => model.limitGuarantee('hole', .1, 0),
  () => model.compositionChange(2, .1), () => model.compositionChange(0, Infinity), () => model.extremaCandidates('motion', 2, 2), () => model.extremaCandidates('cusp', 4, 1), () => model.extremaCandidates('bad', 0, 4),
  () => model.motionAccumulation(4, 129, 'left'), () => model.motionAccumulation(4, 1.5, 'left'), () => model.motionAccumulation(.1, 4, 'left'), () => model.motionAccumulation(4, 4, 'bad'),
  () => model.exponentialRate(.9, 1), () => model.exponentialRate(0, 0), () => model.taylorApproximation('log', 2, -1), () => model.taylorApproximation('exp', 13, 0), () => model.taylorApproximation('bad', 2, 0),
  () => model.improperPowerIntegral('bad', 1, 1), () => model.improperPowerIntegral('tail', -1, 1), () => model.improperPowerIntegral('tail', 1, 7),
];
invalid.forEach(call => assert.throws(call, RangeError));
const source = 'src/learn/data/topics/single-variable-calculus-limits-derivatives-integrals.jsx';
const ast = parse(fs.readFileSync(source, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const formulas = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name?.name === 'MathBlock') {
    const expression = node.children.find(item => item.type === 'JSXExpressionContainer').expression;
    assert.equal(expression.type, 'TaggedTemplateExpression');
    assert.equal(expression.quasi.expressions.length, 0);
    const tex = expression.quasi.quasis.map(item => item.value.raw).join('');
    katex.renderToString(tex, { throwOnError: true, displayMode: true });
    formulas.push(tex);
  }
  for (const [key, value] of Object.entries(node)) if (!['loc', 'start', 'end'].includes(key)) {
    if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
  }
}
visit(ast);
assert(formulas.length > 20);
data.syntax = { equations: formulas.length, invalidRejected: invalid.length };
data.sources = [source, 'src/learn/data/single-variable-calculus-models.js', 'src/learn/data/single-variable-calculus-examples.js', 'src/learn/components/lesson-labs/SingleVariableCalculusLabs.jsx', 'src/learn/components/lesson-labs/single-variable-calculus-labs.css', 'src/learn/data/curriculum/blueprints/single-variable-calculus-limits-derivatives-integrals.js'].map(file => ({ path: file, sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') }));
fs.writeFileSync(`${directory}/cases.json`, JSON.stringify(data));
const result = spawnSync(path.resolve('scratch/lesson-tools/Scripts/python.exe'), ['scripts/verify-single-variable-calculus.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout); process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status || 1);
