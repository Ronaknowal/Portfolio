const fs = require('node:fs');
const { parse } = require('@babel/parser');
const file = 'src/learn/data/topics/decision-theory-risk-cost-sensitive-decisions.jsx';
let source = fs.readFileSync(file, 'utf8');
const edits = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  const text = node.type === 'JSXText' ? node : node.type === 'JSXAttribute' && ['question', 'hint'].includes(node.name.name) && node.value?.type === 'StringLiteral' ? node.value : null;
  if (text) {
    const before = source.slice(text.start, text.end);
    const after = before.replace(/([A-Za-z]{3,})(?=\d)/g, '$1 ').replace(/\b(FN|FP)(?=\d)/g, '$1 ');
    if (after !== before) edits.push({ start: text.start, end: text.end, after });
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(parse(source, { sourceType: 'module', plugins: ['jsx'] }));
for (const edit of edits.sort((left, right) => right.start - left.start)) source = source.slice(0, edit.start) + edit.after + source.slice(edit.end);
parse(source, { sourceType: 'module', plugins: ['jsx'] });
fs.writeFileSync(file, source);
console.log(`Repaired ${edits.length} prose/teaching-prompt spacing spans; code, formulas, links and IDs untouched.`);
