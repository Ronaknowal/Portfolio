import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import generateModule from '@babel/generator';
import postcss from 'postcss';

const generate = generateModule.default || generateModule;
const paths = [
  'src/learn/data/dp-state-families-models.js',
  'src/learn/components/lesson-labs/DpStateFamiliesLabs.jsx',
];
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
const normalize = value => {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['start', 'end', 'loc', 'extra', 'tokens'].includes(key))
    .map(([key, child]) => [key, normalize(child)]));
  return value;
};
const records = [];
for (const path of paths) {
  const original = fs.readFileSync(path, 'utf8');
  const tree = parse(original, { sourceType: 'module', plugins: ['jsx'] });
  const formatted = generate(tree, { comments: true, compact: false, retainLines: false }).code + '\n';
  assert.deepEqual(normalize(parse(formatted, { sourceType: 'module', plugins: ['jsx'] })), normalize(tree));
  fs.writeFileSync(path, formatted);
  records.push({ path, before: digest(original), after: digest(formatted), normalizedAstEqual: true });
}
const cssPath = 'src/learn/components/lesson-labs/dp-state-families-labs.css';
const css = fs.readFileSync(cssPath, 'utf8');
const tree = postcss.parse(css);
const cssIdentity = node => {
  const { type, name, params, selector, prop, value, important, text } = node;
  return { type, name, params, selector, prop, value, important, text, children: node.nodes?.map(cssIdentity) };
};
function formatNode(node, depth = 0) {
  const indent = '  '.repeat(depth);
  if (node.type === 'decl') return `${indent}${node.prop}: ${node.value}${node.important ? ' !important' : ''};`;
  if (node.type === 'comment') return `${indent}/*${node.text}*/`;
  const head = node.type === 'atrule' ? `@${node.name}${node.params ? ' ' + node.params : ''}` : node.selector;
  return `${indent}${head} {\n${node.nodes.map(child => formatNode(child, depth + 1)).join('\n')}\n${indent}}`;
}
const formattedCss = tree.nodes.map(node => formatNode(node)).join('\n\n') + '\n';
assert.deepEqual(cssIdentity(postcss.parse(formattedCss)), cssIdentity(tree));
fs.writeFileSync(cssPath, formattedCss);
records.push({ path: cssPath, before: digest(css), after: digest(formattedCss), parsedCssEqual: true });
fs.writeFileSync('docs/teaching/evidence/dp-state-families-formatting.json', JSON.stringify({ checkedAt: new Date().toISOString(), records }, null, 2) + '\n');
console.log(JSON.stringify(records, null, 2));
