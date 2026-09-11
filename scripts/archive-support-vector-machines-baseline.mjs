import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';

const topic = 'support-vector-machines-svm';
const sourcePath = `src/learn/data/topics/${topic}.jsx`;
const destination = 'docs/teaching/archive/support-vector-machines-before-rewrite';
if (fs.existsSync(destination)) throw new Error('Original archive already exists.');
const source = fs.readFileSync(sourcePath);
const ast = parse(source.toString('utf8'), { sourceType: 'module', plugins: ['jsx'] });
const snippets = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'CodeBlock') {
    const child = node.children.find(item => item.type === 'JSXExpressionContainer'
      && item.expression.type === 'TemplateLiteral' && item.expression.expressions.length === 0);
    if (child) snippets.push(child.expression.quasis[0].value.cooked);
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(ast);
fs.mkdirSync(destination, { recursive: true });
fs.writeFileSync(path.join(destination, `${topic}.jsx`), source);
snippets.forEach((snippet, index) => fs.writeFileSync(path.join(destination, `original-snippet-${index + 1}.py`), snippet));
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const record = {
  capturedAt: new Date().toISOString(), topic, sourcePath, sha256: hash(source), bytes: source.length,
  snippets: snippets.map((snippet, index) => ({ path: `${destination}/original-snippet-${index + 1}.py`, sha256: hash(snippet) })),
  note: 'Exact preservation, not verification. Original CodeBlocks include dependent fragments and comments-only output/sweep material. Printed assertions in the old body are not accepted as executed evidence.',
  parentBaseline: 'docs/teaching/evidence/classical-ml-supervised-baseline.json',
  supportOwnership: 'Original imports shared content/viz only; no dedicated support source to replace.'
};
fs.writeFileSync(path.join(destination, 'manifest.json'), `${JSON.stringify(record, null, 2)}\n`);
console.log(JSON.stringify(record, null, 2));
