import fs from 'node:fs';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { parse } from '@babel/parser';
import { rateDistortionExamples } from '../src/learn/data/rate-distortion-examples.js';

const directory = 'scratch/rate-distortion-review/native';
fs.mkdirSync(directory, { recursive: true });
const ast = parse(fs.readFileSync('scratch/rate-distortion-review/original-lesson.jsx', 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const original = {};
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'CodeBlock') {
    const language = node.openingElement.attributes.find(attribute => attribute.name?.name === 'language')?.value?.value;
    const expression = node.children.find(child => child.type === 'JSXExpressionContainer')?.expression;
    if (expression?.type === 'TemplateLiteral' && expression.expressions.length === 0) original[language] = expression.quasis[0].value.cooked;
  }
  for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
}
visit(ast);
assert.ok(original.python && original.output);
const examples = {
  originalBinary: { title: 'The original fair-binary calculation, within its stated domain', question: 'Why must these sample budgets stay between zero and one half?', language: 'python', code: original.python },
  ...rateDistortionExamples,
};
examples.originalBinary.code = original.python;
fs.writeFileSync(directory + '/input.json', JSON.stringify(examples, null, 2));
execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/prepare-rate-distortion-native.py'], { stdio: 'inherit' });
const prepared = JSON.parse(fs.readFileSync(directory + '/prepared.json'));
assert.equal(prepared.originalBinary.code, original.python);
assert.equal(prepared.originalBinary.expected, original.output.trim());
fs.writeFileSync('src/learn/data/rate-distortion-examples.js', 'export const rateDistortionExamples = ' + JSON.stringify(prepared, null, 2) + ';\n');
console.log('Complete native examples prepared; original program and output preserved exactly.');
