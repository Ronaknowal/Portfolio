import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import traverseModule from '@babel/traverse';

const traverse = traverseModule.default || traverseModule;
const source = 'src/learn/data/topics/survival-analysis-cox-regression-kaplan-meier-hazard-models.jsx';
const directory = 'docs/teaching/archive/survival-original';
const destination = 'docs/teaching/evidence/survival-original.json';
if (fs.existsSync(destination)) throw new Error('Preserve the existing original archive.');
fs.mkdirSync(directory, { recursive: true });
const bytes = fs.readFileSync(source);
const archive = path.join(directory, path.basename(source) + '.txt');
fs.writeFileSync(archive, bytes);
const ast = parse(bytes.toString('utf8'), { sourceType: 'module', plugins: ['jsx'] });
const programs = [];
traverse(ast, {
  JSXElement({ node }) {
    if (node.openingElement.name.name !== 'CodeBlock') return;
    const expression = node.children.find(child => child.type === 'JSXExpressionContainer')?.expression;
    if (expression?.type !== 'TemplateLiteral' || expression.expressions.length) throw new Error('Review nonliteral extraction.');
    const code = expression.quasis[0].value.cooked;
    const codePath = path.join(directory, `program-${programs.length}.py`);
    fs.writeFileSync(codePath, code, 'utf8');
    programs.push({ index: programs.length, sourceLine: node.loc.start.line, path: codePath, sha256: crypto.createHash('sha256').update(code).digest('hex'), code });
  }
});
fs.writeFileSync(destination, JSON.stringify({ capturedAt: new Date().toISOString(), source, archive, sourceSha256: crypto.createHash('sha256').update(bytes).digest('hex'), programs, sharedBaseline: 'docs/teaching/evidence/classical-ml-supervised-baseline.json', priorTopicOwnedSupportFiles: [], note: 'Exact original body and code, including comments, preserved. Notebook dependencies will be restored explicitly for execution; claims and output comments are not presumed correct.' }, null, 2) + '\n');
console.log(JSON.stringify({ source, programs: programs.length, archive }));
