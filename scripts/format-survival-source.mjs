import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import generator from '@babel/generator';
import postcss from 'postcss';
import crypto from 'node:crypto';
const generate = generator.default || generator;
const files = ['src/learn/data/topics/survival-analysis-cox-regression-kaplan-meier-hazard-models.jsx', 'src/learn/components/lesson-labs/SurvivalLabs.jsx', 'src/learn/data/curriculum/blueprints/survival-analysis-cox-regression-kaplan-meier-hazard-models.js'];
const ignore = new Set(['start', 'end', 'loc', 'extra', 'comments', 'leadingComments', 'trailingComments', 'innerComments']);
function normalized(value) {
  if (Array.isArray(value)) return value.map(normalized);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([key]) => !ignore.has(key)).map(([key, entry]) => [key, normalized(entry)]));
  return value;
}
for (const file of files) {
  const source = fs.readFileSync(file, 'utf8');
  const original = parse(source, { sourceType: 'module', plugins: ['jsx'] });
  const formatted = generate(original, { jsescOption: { minimal: true }, comments: true }).code + '\n';
  assert.deepEqual(normalized(parse(formatted, { sourceType: 'module', plugins: ['jsx'] })), normalized(original));
  fs.writeFileSync(file, formatted);
}
const cssFile = 'src/learn/components/lesson-labs/survival-labs.css';
const originalCss = fs.readFileSync(cssFile, 'utf8');
const css = postcss.parse(originalCss);
function formatCss(node, depth = 0) {
  const indent = '  '.repeat(depth);
  if (node.type === 'root') return node.nodes.map(child => formatCss(child)).join('\n\n') + '\n';
  if (node.type === 'decl') return `${indent}${node.prop}: ${node.value}${node.important ? ' !important' : ''};`;
  if (node.type === 'comment') return `${indent}/* ${node.text} */`;
  if (node.type === 'rule' || node.type === 'atrule') {
    const label = node.type === 'rule' ? node.selector : `@${node.name} ${node.params}`;
    return `${indent}${label} {\n${node.nodes.map(child => formatCss(child, depth + 1)).join('\n')}\n${indent}}`;
  }
  throw new Error(`Unsupported CSS node ${node.type}`);
}
function cssMeaning(node) { return { type: node.type, selector: node.selector, prop: node.prop, value: node.value, important: node.important, name: node.name, params: node.params, nodes: node.nodes?.map(cssMeaning) }; }
const formattedCss = formatCss(css);
assert.deepEqual(cssMeaning(postcss.parse(formattedCss)), cssMeaning(css));
fs.writeFileSync(cssFile, formattedCss);
fs.writeFileSync('scratch/survival/format-conservation.json', JSON.stringify({ timestamp: new Date().toISOString(), files, normalizedAstConserved: true, cssFile, cssAstConserved: true, originalCssSha256: crypto.createHash('sha256').update(originalCss).digest('hex'), finalCssSha256: crypto.createHash('sha256').update(formattedCss).digest('hex') }, null, 2));
console.log('Formatted three semantic JavaScript sources and CSS; normalized ASTs conserved.');
