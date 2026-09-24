const fs = require('node:fs');
const assert = require('node:assert/strict');
const parser = require('@babel/parser');
const generate = require('@babel/generator').default;
const postcss = require('postcss');
const esbuild = require('esbuild');

function normalize(value) {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value)
      .filter(([key]) => !['loc', 'start', 'end', 'extra', 'tokens', 'comments', 'leadingComments', 'trailingComments', 'innerComments', 'raws', 'source', 'inputs'].includes(key))
      .map(([key, child]) => [key, normalize(child)]));
  }
  return value;
}

const records = [];
for (const file of ['src/learn/data/topics/naive-bayes-probabilistic-classifiers.jsx', 'src/learn/components/lesson-labs/NaiveBayesLabs.jsx', 'src/learn/data/naive-bayes-models.js', 'src/learn/data/curriculum/blueprints/naive-bayes-probabilistic-classifiers.js']) {
  const before = fs.readFileSync(file, 'utf8');
  const ast = parser.parse(before, { sourceType: 'module', plugins: ['jsx'] });
  const after = generate(ast, { compact: false, concise: false, jsescOption: { minimal: true } }, before).code + '\n';
  assert.deepEqual(normalize(parser.parse(after, { sourceType: 'module', plugins: ['jsx'] })), normalize(ast));
  assert.equal(esbuild.transformSync(after, { loader: file.endsWith('jsx') ? 'jsx' : 'js' }).warnings.length, 0);
  fs.writeFileSync(file, after);
  records.push({ file, normalizedAstConserved: true, transformWarnings: 0 });
}
const cssFile = 'src/learn/components/lesson-labs/naive-bayes-labs.css';
const css = postcss.parse(fs.readFileSync(cssFile, 'utf8'));
const originalTree = normalize(css.toJSON());
css.walk(node => {
  let depth = 0;
  for (let parent = node.parent; parent && parent.type !== 'root'; parent = parent.parent) depth += 1;
  node.raws.before = '\n' + '  '.repeat(depth);
  if (node.type === 'decl') node.raws.between = ': ';
  if (node.nodes) { node.raws.between = ' '; node.raws.after = '\n' + '  '.repeat(depth); node.raws.semicolon = true; }
});
const formatted = css.toString().trim() + '\n';
assert.deepEqual(normalize(postcss.parse(formatted).toJSON()), originalTree);
fs.writeFileSync(cssFile, formatted);
records.push({ file: cssFile, normalizedAstConserved: true });
fs.writeFileSync('scratch/naive-bayes-verification/format-conservation.json', JSON.stringify({ checkedAt: new Date().toISOString(), records }, null, 2) + '\n');
console.log('Five owned source files formatted; semantic AST and JSX transforms conserved.');
