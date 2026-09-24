import fs from 'node:fs';
import { parse } from '@babel/parser';
import generator from '@babel/generator';
const generate = generator.default || generator;
const files = ['src/learn/data/topics/survival-analysis-cox-regression-kaplan-meier-hazard-models.jsx', 'src/learn/components/lesson-labs/SurvivalLabs.jsx'];
const changes = [];
function polish(text) {
  return text
    .replace(/\b(Python|NumPy|SciPy|pandas|lifelines|scikit-survival|scikit-learn|Cox|seed|shape|scale|cause|group|interval|day|time|at|by|before|after|through|until|than|of|is|to)(\d)/g, '$1 $2')
    .replace(/(\d)(days?|seconds?|hours?|pump-days|pump-day|failures?|events?|subjects?|people|records?|pumps?|units?|per|for|with|and|at|instead|rather|then|or|if|when|gives|versus|time units)\b/g, '$1 $2')
    .replace(/([,;])(?=[A-Za-z])/g, '$1 ')
    .replace(/\b(is|gives|has|with|and|of|the|by|from|to|then|for)(?=[A-Z](?:\b|[(_=]))/g, '$1 ')
    .replace(/\b(in|by|for|with|of|is|through)(?=\()/g, '$1 ');
}
for (const file of files) {
  const ast = parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  function walk(node) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'JSXText') {
      const next = polish(node.value);
      if (next !== node.value) { changes.push({ file, before: node.value.slice(0, 120), after: next.slice(0, 120) }); node.value = next; }
    }
    for (const [key, value] of Object.entries(node)) {
      if (['loc', 'extra', 'comments'].includes(key)) continue;
      if (Array.isArray(value)) value.forEach(walk); else if (value && typeof value === 'object') walk(value);
    }
  }
  walk(ast);
  fs.writeFileSync(file, generate(ast, { jsescOption: { minimal: true }, comments: true }).code + '\n');
}
fs.writeFileSync('scratch/survival/prose-spacing.json', JSON.stringify({ checkedAt: new Date().toISOString(), scope: 'Word/number spacing in JSX prose only; no mathematical string, executable example or numerical model edits.', changes }, null, 2) + '\n');
console.log(`Polished ${changes.length} JSX prose nodes.`);
