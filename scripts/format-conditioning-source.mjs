import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import generator from '@babel/generator';

const files = [
  'src/learn/data/topics/conditioning-stability-numerical-analysis.jsx',
  'src/learn/data/conditioning-stability-models.js',
  'src/learn/components/lesson-labs/ConditioningStabilityLabs.jsx',
  'src/learn/data/curriculum/blueprints/conditioning-stability-numerical-analysis.js',
];
const normalized = ast => JSON.stringify(ast, (key, value) =>
  ['start', 'end', 'loc', 'extra', 'tokens'].includes(key) ? undefined : value);
const digest = text => crypto.createHash('sha256').update(text).digest('hex');
const records = [];
for (const file of files) {
  const before = fs.readFileSync(file, 'utf8');
  const ast = parse(before, { sourceType: 'module', plugins: ['jsx'] });
  const after = generator.default(ast, { comments: true, compact: false, retainLines: false }).code + '\n';
  assert.equal(normalized(parse(after, { sourceType: 'module', plugins: ['jsx'] })), normalized(ast), file);
  fs.writeFileSync(file, after);
  records.push({ file, before: digest(before), after: digest(after), normalizedAstEqual: true });
}
fs.mkdirSync('scratch/conditioning-stability-format', { recursive: true });
fs.writeFileSync(path.resolve('scratch/conditioning-stability-format/results.json'), JSON.stringify({ completedAt: new Date().toISOString(), records }, null, 2));
console.log('Four owned modules formatted with exact normalized AST and JSX/string equality.');
