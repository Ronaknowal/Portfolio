const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { parse } = require('@babel/parser');
const files = [
  'src/learn/data/topics/convex-duality-lagrangian-methods-kkt-conditions.jsx',
  'src/learn/data/topics/randomized-algorithms-sampling-error-guarantees.jsx',
  'src/learn/data/topics/external-memory-algorithms-b-trees-i-o-complexity.jsx',
];
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const normalized = node => {
  if (Array.isArray(node)) return node.map(normalized);
  if (!node || typeof node !== 'object') return node;
  return Object.fromEntries(Object.entries(node)
    .filter(([key]) => !['start', 'end', 'loc', 'extra'].includes(key))
    .map(([key, value]) => [key, normalized(value)]));
};
const results = [];
for (const file of files) {
  const original = fs.readFileSync(file, 'utf8');
  const ast = parse(original, { sourceType: 'module', plugins: ['jsx'] });
  const edits = [];
  const visit = node => {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'JSXText') {
      const raw = original.slice(node.start, node.end);
      if (raw.includes('>')) edits.push({ start: node.start, end: node.end, text: raw.replaceAll('>', '&gt;') });
    }
    for (const child of Object.values(node)) {
      if (Array.isArray(child)) child.forEach(visit);
      else if (child && typeof child === 'object') visit(child);
    }
  };
  visit(ast);
  let updated = original;
  for (const edit of edits.sort((a, b) => b.start - a.start)) updated = updated.slice(0, edit.start) + edit.text + updated.slice(edit.end);
  assert.deepEqual(normalized(parse(updated, { sourceType: 'module', plugins: ['jsx'] })), normalized(ast), file);
  if (edits.length) fs.writeFileSync(file, updated);
  results.push({ file, textNodesEscaped: edits.length, beforeSha256: hash(original), afterSha256: hash(updated), normalizedAstEqual: true });
}
fs.mkdirSync('scratch/lesson-comparison-entities', { recursive: true });
fs.writeFileSync('scratch/lesson-comparison-entities/source-results.json', JSON.stringify({ at: new Date().toISOString(), results }, null, 2) + '\n');
console.log(JSON.stringify(results));
