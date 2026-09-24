import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parse } from '@babel/parser';

const sourcePath = 'src/learn/data/topics/ensemble-methods-stacking.jsx';
const target = 'docs/teaching/evidence/ensemble-original-review.json';
if (fs.existsSync(target)) throw new Error('Original archive already exists.');
const bytes = fs.readFileSync(sourcePath);
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const ast = parse(bytes.toString('utf8'), { sourceType: 'module', plugins: ['jsx'] });
const programs = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'CodeBlock') {
    const expression = node.children.find(child => child.type === 'JSXExpressionContainer')?.expression;
    if (expression?.type !== 'TemplateLiteral' || expression.expressions.length) throw new Error('Unexpected original code expression.');
    programs.push({ line: node.loc.start.line, code: expression.quasis[0].value.cooked });
  }
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc') continue;
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(ast);
const directory = 'scratch/ensemble-methods';
fs.mkdirSync(directory, { recursive: true });
const python = path.resolve('scratch/lesson-tools/Scripts/python.exe');
const records = programs.map((program, index) => {
  const file = `${directory}/original-program-${index + 1}.py`;
  fs.writeFileSync(file, program.code);
  const run = spawnSync(python, [file], {
    encoding: 'utf8', timeout: 180000,
    env: { ...process.env, LOKY_MAX_CPU_COUNT: '2', OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1' }
  });
  return { ...program, sha256: hash(program.code), standaloneExit: run.status, stdout: run.stdout, stderr: run.stderr, error: run.error?.message || null };
});
const record = {
  capturedAt: new Date().toISOString(),
  scope: 'Exact original body and its three Python blocks, each executed as a standalone program before rewriting. Resource parallelism capped at two workers without changing code. Failures and old output comments retained as observed history, not endorsed current results.',
  source: { path: sourcePath, sha256: hash(bytes), encoding: 'base64', bytes: bytes.toString('base64') },
  programs: records
};
fs.writeFileSync(target, JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ sourceHash: record.source.sha256, programs: records.map(program => ({ line: program.line, exit: program.standaloneExit, stdout: program.stdout, error: program.stderr.slice(-600) })) }, null, 2));
