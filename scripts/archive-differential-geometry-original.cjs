const fs = require('node:fs');
const crypto = require('node:crypto');
const parser = require('@babel/parser');
const sourcePath = 'src/learn/data/topics/differential-geometry-riemannian-manifolds.jsx';
const destination = 'docs/teaching/evidence/differential-geometry-original-content.json';
if (fs.existsSync(destination)) throw new Error('Original archive already exists; do not overwrite it.');
const bytes = fs.readFileSync(sourcePath);
const hash = crypto.createHash('sha256').update(bytes).digest('hex');
if (hash !== '4f9a22535e04bf93314583373a2dcde737579ea8a54e5af2426b4bc3ed07989c') throw new Error('Unexpected original source.');
const source = bytes.toString('utf8');
const ast = parser.parse(source, { sourceType: 'module', plugins: ['jsx'] });
const blocks = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXElement' && node.openingElement.name.name === 'CodeBlock') {
    const language = node.openingElement.attributes.find(item => item.name?.name === 'language').value.value;
    const expression = node.children.find(item => item.type === 'JSXExpressionContainer').expression;
    if (expression.expressions.length) throw new Error('Original code unexpectedly interpolated.');
    blocks.push({ language, text: expression.quasis[0].value.cooked });
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') visit(value);
  }
}
visit(ast);
fs.writeFileSync(destination, JSON.stringify({ topicId: 'differential-geometry-riemannian-manifolds',
  archivedAt: new Date().toISOString(), sourcePath, sha256: hash, fullSource: source, blocks }, null, 2));
console.log(JSON.stringify({ sha256: hash, blocks: blocks.length }));
