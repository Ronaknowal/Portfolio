const assert = require('node:assert/strict');
const fs = require('node:fs');
const { parse } = require('@babel/parser');
const generate = require('@babel/generator').default;

const files = [
  'src/learn/data/topics/algorithm-correctness-loop-invariants-termination.jsx',
  'src/learn/data/algorithm-correctness-models.js',
  'src/learn/components/lesson-labs/AlgorithmCorrectnessLabs.jsx',
  'src/learn/data/curriculum/blueprints/algorithm-correctness-loop-invariants-termination.js',
];
function withoutLocations(value) {
  if (Array.isArray(value)) return value.map(withoutLocations);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['start', 'end', 'loc', 'extra'].includes(key))
    .map(([key, child]) => [key, withoutLocations(child)]));
}

fs.mkdirSync('scratch/algorithm-correctness-verification', { recursive: true });
for (const file of files) {
  const before = fs.readFileSync(file, 'utf8');
  const ast = parse(before, { sourceType: 'module', plugins: ['jsx'] });
  const after = generate(ast, { compact: false, comments: true, jsescOption: { minimal: true } }, before).code + '\n';
  assert.deepEqual(withoutLocations(parse(after, { sourceType: 'module', plugins: ['jsx'] })), withoutLocations(ast), file);
  fs.writeFileSync(file, after);
}
const result = { checkedAt: new Date().toISOString(), files, astAndStringsPreserved: true };
fs.writeFileSync('scratch/algorithm-correctness-verification/formatting-results.json', JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
