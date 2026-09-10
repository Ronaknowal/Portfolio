const assert = require('node:assert/strict');
const fs = require('node:fs');
const { parse } = require('@babel/parser');
const generate = require('@babel/generator').default;
const postcss = require('postcss');

const files = [
  'src/learn/data/topics/exponential-families-sufficient-statistics.jsx',
  'src/learn/data/exponential-family-models.js',
  'src/learn/data/exponential-family-examples.js',
  'src/learn/components/lesson-labs/ExponentialFamilyLabs.jsx',
  'src/learn/data/curriculum/blueprints/exponential-families-sufficient-statistics.js',
];
function comparable(value) {
  if (Array.isArray(value)) return value.map(comparable);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).filter(([key]) => !['start', 'end', 'loc', 'extra'].includes(key)).map(([key, child]) => [key, comparable(child)]));
}
for (const file of files) {
  const before = fs.readFileSync(file, 'utf8');
  const ast = parse(before, { sourceType: 'module', plugins: ['jsx'] });
  const after = generate(ast, { compact: false, comments: true, jsescOption: { minimal: true } }, before).code + '\n';
  assert.deepEqual(comparable(parse(after, { sourceType: 'module', plugins: ['jsx'] })), comparable(ast), file);
  fs.writeFileSync(file, after);
}
const cssFile = 'src/learn/components/lesson-labs/exponential-family-labs.css';
const css = postcss.parse(fs.readFileSync(cssFile, 'utf8'));
function formatCss(node, depth = 0) {
  const indent = '  '.repeat(depth);
  if (node.type === 'decl') return `${indent}${node.prop}: ${node.value}${node.important ? ' !important' : ''};`;
  if (node.type === 'comment') return `${indent}/* ${node.text} */`;
  if (node.type === 'root') return node.nodes.map(child => formatCss(child)).join('\n\n') + '\n';
  const label = node.type === 'atrule' ? `@${node.name} ${node.params}` : node.selector;
  return `${indent}${label} {\n${node.nodes.map(child => formatCss(child, depth + 1)).join('\n')}\n${indent}}`;
}
function cssMeaning(root) {
  return root.nodes.map(node => ({ type: node.type, selector: node.selector, name: node.name, params: node.params, prop: node.prop, value: node.value, important: node.important, text: node.text, children: node.nodes ? cssMeaning(node) : undefined }));
}
const formattedCss = formatCss(css);
assert.deepEqual(cssMeaning(postcss.parse(formattedCss)), cssMeaning(css));
fs.writeFileSync(cssFile, formattedCss);
fs.mkdirSync('scratch/exponential-family-verification', { recursive: true });
const result = { checkedAt: new Date().toISOString(), files: [...files, cssFile], javascriptAstAndStringsPreserved: true, cssMeaningPreserved: true };
fs.writeFileSync('scratch/exponential-family-verification/formatting-results.json', JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));

