const fs = require('node:fs');
const assert = require('node:assert/strict');
const parser = require('@babel/parser');
const generate = require('@babel/generator').default;
const postcss = require('postcss');
function normalize(value) {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([key]) => !['loc', 'start', 'end', 'extra', 'tokens', 'comments', 'leadingComments', 'trailingComments', 'innerComments', 'raws', 'source', 'inputs'].includes(key)).map(([key, child]) => [key, normalize(child)]));
  return value;
}
const records = [];
for (const file of ['src/learn/data/topics/real-analysis-sequences-modes-of-convergence.jsx', 'src/learn/components/lesson-labs/RealAnalysisLabs.jsx', 'src/learn/data/real-analysis-models.js', 'src/learn/data/curriculum/blueprints/real-analysis-sequences-modes-of-convergence.js']) {
  const before = fs.readFileSync(file, 'utf8');
  const ast = parser.parse(before, { sourceType: 'module', plugins: ['jsx'] });
  const after = generate(ast, { compact: false, concise: false, jsescOption: { minimal: true } }, before).code + '\n';
  assert.deepEqual(normalize(parser.parse(after, { sourceType: 'module', plugins: ['jsx'] })), normalize(ast));
  fs.writeFileSync(file, after);
  records.push({ file, normalizedASTConserved: true });
}
const file = 'src/learn/components/lesson-labs/real-analysis-labs.css';
const css = postcss.parse(fs.readFileSync(file, 'utf8'));
const before = normalize(css.toJSON());
css.walk(node => {
  let depth = 0;
  for (let parent = node.parent; parent && parent.type !== 'root'; parent = parent.parent) depth++;
  node.raws.before = '\n' + '  '.repeat(depth);
  if (node.type === 'decl') node.raws.between = ': ';
  if (node.nodes) { node.raws.between = ' '; node.raws.after = '\n' + '  '.repeat(depth); node.raws.semicolon = true; }
});
const formatted = css.toString().trim() + '\n';
assert.deepEqual(normalize(postcss.parse(formatted).toJSON()), before);
fs.writeFileSync(file, formatted);
records.push({ file, normalizedASTConserved: true });
fs.mkdirSync('scratch/real-analysis-verification', { recursive: true });
fs.writeFileSync('scratch/real-analysis-verification/format-conservation.json', JSON.stringify({ checkedAt: new Date().toISOString(), records }, null, 2));
console.log('Five owned production files formatted with normalized AST conservation.');

