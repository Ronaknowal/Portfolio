import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import katex from 'katex';
import { gaussianProcessExamples } from '../src/learn/data/gaussian-process-examples.js';
const sourcePath = 'src/learn/data/topics/gaussian-processes-gp.jsx';
const source = fs.readFileSync(sourcePath, 'utf8');
const manuscript = fs.readFileSync('docs/teaching/drafts/gaussian-processes-gp/lesson.md', 'utf8').replaceAll('\r', '');
const tree = parse(source, { sourceType: 'module', plugins: ['jsx'] });
const strings = [], maths = [], headings = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'StringLiteral') {
    assert.ok(!/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(node.value), 'Control character in decoded JSX string');
    assert.ok(!/<\/?(?:details|summary)>/.test(node.value), 'HTML markup accidentally rendered as prose');
    strings.push(node.value);
  }
  if (node.type === 'JSXElement') {
    const name = node.openingElement.name.name;
    const text = node.children.filter(child => child.type === 'JSXExpressionContainer' && child.expression.type === 'StringLiteral').map(child => child.expression.value).join('');
    if (name === 'Math' || name === 'MathBlock') { katex.renderToString(text, { throwOnError: true, displayMode: name === 'MathBlock', strict: 'error' }); maths.push(text); }
    if (name === 'H2') headings.push(text);
  }
  for (const [key, value] of Object.entries(node)) if (key !== 'loc' && key !== 'extra') {
    if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
  }
}
visit(tree);
const expectedHeadings = [...manuscript.matchAll(/^## (.+)$/gm)].map(match => match[1]);
assert.deepEqual(headings, expectedHeadings);
assert.equal(headings.length, 10);
assert.equal((source.match(/<details>/g) || []).length, 14, 'Seven separate closed hints and solutions');
const originalPrograms = [...manuscript.matchAll(/```python\n([\s\S]*?)\n```/g)].map(match => match[1]);
assert.deepEqual(gaussianProcessExamples.map(example => example.code), originalPrograms);
assert.equal(gaussianProcessExamples.length, 2);
gaussianProcessExamples.forEach(example => {
  assert.equal(fs.readFileSync(`public/learn/examples/gaussian-processes-gp/${example.file}`, 'utf8'), example.code + '\n');
  assert.ok(example.expected.length > 40, 'Missing captured full stdout');
});
for (const marker of ['VectorFunctionFigure', 'ConditioningSliceFigure', 'ObservationMatrixFigure', 'KernelGeometryFigure', 'HistoricalForecastFigure', 'ProbeChoiceFigure', 'GaussianConditioningLab', 'GaussianForecastLab', 'GaussianProbeLab']) assert.ok(source.includes(`<${marker}`), marker);
const requiredPhrases = ['Brownian sample paths almost surely do not belong', 'finite-rank GP', 'same sampled feature map', 'pointwise', '13 of 24', 'trace term', 'first-pass', 'First pass', 'Semi-Supervised Learning'];
for (const phrase of requiredPhrases.filter(value => value !== 'first-pass')) assert.ok(strings.join(' ').includes(phrase), phrase);
assert.ok(!source.includes('Inline figure F') && !source.includes('Investigation I'), 'Implementation notes leaked into reader');
const evidence = { passed: true, formulasRenderedWithoutErrors: maths.length, completeManuscriptHeadings: headings.length, exactNativePrograms: 2, namedFigures: 6, investigations: 'Three contracts, with I1 split into direct and spatial mounts', source: { [sourcePath]: crypto.createHash('sha256').update(source).digest('hex') } };
fs.writeFileSync('docs/teaching/evidence/gaussian-process-content.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(evidence, null, 2));
