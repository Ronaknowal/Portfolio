import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import generate from '@babel/generator';

const paths = [
  'src/learn/data/topics/random-variables-expectation-covariance.jsx',
  'src/learn/components/lesson-labs/RandomVariableLabs.jsx',
  'src/learn/data/random-variables-models.js',
];
function readable(text) {
  return text.replace(/([A-Za-z]{2,})(?=[0-9−])/g, '$1 ')
    .replace(/\b(is|at|by|probability|density|mass)(?=\.[0-9])/g, '$1 ')
    .replace(/\b(over|in|on|probability|variance|therefore|add|is)(?=[([])/g, '$1 ');
}
function walk(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXText') {
    node.value = readable(node.value);
    node.extra = { raw: node.value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'), rawValue: node.value };
  }
  if (node.type === 'JSXAttribute' && ['prompt','hint','title','prerequisites'].includes(node.name.name) && node.value?.type === 'StringLiteral') {
    node.value.value = readable(node.value.value); delete node.value.extra;
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === 'object') walk(value);
  }
}
function clean(node) {
  return JSON.parse(JSON.stringify(node, (key, value) => ['start','end','loc','extra','comments','tokens'].includes(key) ? undefined : value));
}
for (const path of paths) {
  const source = fs.readFileSync(path,'utf8');
  const ast = parse(source,{sourceType:'module',plugins:['jsx']});
  walk(ast);
  const result = generate.default(ast,{retainLines:false,comments:true,jsescOption:{minimal:true}}).code + '\n';
  assert.deepEqual(clean(parse(result,{sourceType:'module',plugins:['jsx']})),clean(ast));
  fs.writeFileSync(path,result);
  console.log('Formatted with normalized AST conservation: '+path);
}
