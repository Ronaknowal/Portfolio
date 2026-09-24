import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import generator from '@babel/generator';

const files = [
  'src/learn/data/topics/gradient-boosted-trees-xgboost-lightgbm-catboost.jsx',
  'src/learn/data/gradient-boosted-trees-models.js',
  'src/learn/components/lesson-labs/GradientBoostedTreeLabs.jsx',
];
const generate = generator.default || generator;
const digest = text => crypto.createHash('sha256').update(text).digest('hex');
const syntax = source => parse(source, { sourceType: 'module', plugins: ['jsx'] });
const normalize = value => {
  if (Array.isArray(value)) return value.map(normalize);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).filter(([key]) => !['start', 'end', 'loc', 'extra', 'tokens'].includes(key)).map(([key, child]) => [key, normalize(child)]));
};
const records = [];
for (const path of files) {
  const before = fs.readFileSync(path, 'utf8');
  const tree = syntax(before);
  const after = generate(tree, { comments: true, compact: false, retainLines: false }).code + '\n';
  assert.deepEqual(normalize(syntax(after)), normalize(tree), path);
  fs.writeFileSync(path, after);
  records.push({ path, beforeSha256: digest(before), afterSha256: digest(after), normalizedAstUnchanged: true });
}
fs.writeFileSync('scratch/gradient-boosted-trees-verification/formatting.json', JSON.stringify({ checkedAt: new Date().toISOString(), records }, null, 2));
console.log(records);
