import fs from 'node:fs';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { parse } from '@babel/parser';
import { optimalTransportExamples } from '../src/learn/data/optimal-transport-examples.js';

const directory = 'scratch/optimal-transport-review/native';
fs.mkdirSync(directory, { recursive: true });
const ast = parse(fs.readFileSync('scratch/optimal-transport-review/original-lesson.jsx', 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const blocks = {};
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'CodeBlock') {
    const language = node.openingElement.attributes.find(attribute => attribute.name?.name === 'language')?.value?.value;
    const expression = node.children.find(child => child.type === 'JSXExpressionContainer')?.expression;
    if (expression?.type === 'TemplateLiteral' && expression.expressions.length === 0) blocks[language] = expression.quasis[0].value.cooked;
  }
  for (const child of Object.values(node)) if (Array.isArray(child)) child.forEach(visit); else if (child && typeof child === 'object') visit(child);
}
visit(ast);
assert.ok(blocks.python && blocks.output);
const examples = {
  originalScaling: {
    title: 'Preserved first scaling example: a tiny complete loop',
    question: 'Predict why the regularized plan keeps both marginals but moves a little mass along the more expensive routes.',
    code: blocks.python,
  },
  ...optimalTransportExamples,
};
// Preserve the original program exactly even when this script is rerun.
examples.originalScaling.code = blocks.python;
fs.writeFileSync(directory + '/input.json', JSON.stringify(examples, null, 2));
execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/prepare-optimal-transport-native.py'], { stdio: 'inherit' });
const prepared = JSON.parse(fs.readFileSync(directory + '/prepared.json'));
assert.equal(prepared.originalScaling.code, blocks.python);
assert.equal(prepared.originalScaling.expected, blocks.output.trim());
fs.writeFileSync('src/learn/data/optimal-transport-examples.js', 'export const optimalTransportExamples = ' + JSON.stringify(prepared, null, 2) + ';\n');
console.log('Eight complete programs captured; original code/output retained exactly.');
