import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';

const topic = 'gradient-boosted-trees-xgboost-lightgbm-catboost';
const sourcePath = `src/learn/data/topics/${topic}.jsx`;
const destination = 'docs/teaching/archive/gradient-boosted-trees-before-rewrite';
if (fs.existsSync(destination)) throw new Error('Baseline already archived; do not overwrite.');
const source = fs.readFileSync(sourcePath);
const ast = parse(source.toString('utf8'), { sourceType: 'module', plugins: ['jsx'] });
const programs = [];
const outputs = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement') {
    const name = node.openingElement.name.name;
    const literal = node.children.find(child => child.type === 'JSXExpressionContainer'
      && child.expression.type === 'TemplateLiteral' && child.expression.expressions.length === 0);
    if (literal && name === 'CodeBlock') programs.push(literal.expression.quasis[0].value.cooked);
    if (literal && name === 'Callout' && node.openingElement.attributes.some(attribute => attribute.name?.name === 'type' && attribute.value?.value === 'output')) {
      outputs.push(literal.expression.quasis[0].value.cooked);
    }
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(ast);
fs.mkdirSync(destination, { recursive: true });
fs.writeFileSync(path.join(destination, `${topic}.jsx`), source);
programs.forEach((program, index) => fs.writeFileSync(path.join(destination, `original-program-${index + 1}.py`), program));
outputs.forEach((output, index) => fs.writeFileSync(path.join(destination, `original-output-${index + 1}.txt`), output));
const baselinePath = 'docs/teaching/evidence/classical-ml-supervised-baseline.json';
const baselineBytes = fs.readFileSync(baselinePath);
const baseline = JSON.parse(baselineBytes);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const record = {
  capturedAt: new Date().toISOString(), topic, sourcePath,
  sha256: hash(source), bytes: source.length,
  originalProgramCount: programs.length, originalOutputCount: outputs.length,
  parentBaseline: { path: baselinePath, sha256: hash(baselineBytes), capturedAt: baseline.capturedAt },
  ownership: 'The old topic has no dedicated support imports; shared visual/content implementations are not mutated.',
  note: 'Preservation evidence only. Historical claims of executed stdout are not accepted without rerunning these extracted programs.'
};
fs.writeFileSync(path.join(destination, 'manifest.json'), `${JSON.stringify(record, null, 2)}\n`);
console.log(JSON.stringify(record, null, 2));
