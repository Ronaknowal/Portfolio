import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parse } from '@babel/parser';

const sourcePath = 'src/learn/data/topics/naive-bayes-probabilistic-classifiers.jsx';
const bytes = fs.readFileSync(sourcePath);
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const target = 'docs/teaching/evidence/naive-bayes-original-review.json';
if (fs.existsSync(target)) throw new Error('Original record already exists; preserve it.');
const ast = parse(bytes.toString('utf8'), { sourceType: 'module', plugins: ['jsx'] });
const programs = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'CodeBlock') {
    const lang = node.openingElement.attributes.find(item => item.name?.name === 'language');
    if (lang?.value?.value === 'python') {
      const expr = node.children.find(item => item.type === 'JSXExpressionContainer')?.expression;
      if (expr?.type !== 'TemplateLiteral' || expr.expressions.length) throw new Error('Nonliteral original program');
      programs.push({ code: expr.quasis[0].value.cooked, line: node.loc.start.line });
    }
  }
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc') continue;
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(ast);
const dir = 'scratch/naive-bayes-original';
fs.mkdirSync(dir, { recursive: true });
const python = path.resolve('scratch/lesson-tools/Scripts/python.exe');
const records = programs.map((program, index) => {
  const file = `${dir}/program-${index + 1}.py`;
  fs.writeFileSync(file, program.code);
  const result = spawnSync(python, [file], { encoding: 'utf8', timeout: 120000 });
  return { ...program, sha256: hash(program.code), standaloneExit: result.status,
    stdout: result.stdout, stderr: result.stderr, error: result.error?.message || null };
});
const record = { capturedAt: new Date().toISOString(), scope: 'Exact original body and all six original Python blocks; independent standalone execution before rewrite. Errors and mismatched comments are preserved, not certified.',
  source: { path: sourcePath, sha256: hash(bytes), encoding: 'base64', bytes: bytes.toString('base64') }, programs: records };
fs.writeFileSync(target, JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ sourceHash: record.source.sha256, programs: records.map((p, i) => ({ number: i + 1, line: p.line, exit: p.standaloneExit, stdout: p.stdout, error: p.stderr.slice(-500) })) }, null, 2));
