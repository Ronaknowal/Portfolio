const fs = require('fs');
const parser = require('@babel/parser');
const generate = require('@babel/generator').default;
const postcss = require('postcss');
const assert = require('assert/strict');
const records = [];
function normalized(value) {
  if (Array.isArray(value)) return value.map(normalized);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['loc','start','end','extra','tokens','comments','leadingComments','trailingComments','innerComments'].includes(key))
    .map(([key,child]) => [key,normalized(child)]));
  return value;
}
for (const file of [
  'src/learn/components/lesson-labs/QueueingLabs.jsx',
  'src/learn/data/queueing-models.js',
  'src/learn/data/topics/queueing-theory-m-m-1-m-g-1-littles-law.jsx',
  'scripts/verify-queueing-models.mjs',
  'scripts/review-queueing-lesson.cjs',
]) {
  const before = fs.readFileSync(file,'utf8');
  const ast = parser.parse(before,{sourceType:'unambiguous',plugins:['jsx']});
  const after = generate(ast,{compact:false,concise:false,jsescOption:{minimal:true}},before).code+'\n';
  assert.deepEqual(normalized(parser.parse(after,{sourceType:'unambiguous',plugins:['jsx']})), normalized(ast),file);
  fs.writeFileSync(file,after);
  records.push({file,normalizedASTIdentical:true});
}
const cssFile='src/learn/components/lesson-labs/QueueingLabs.css';
const css = postcss.parse(fs.readFileSync(cssFile,'utf8'));
const cssMeaning=root=>{
  const copy=root.toJSON();
  function strip(value) {
    if(Array.isArray(value)) return value.map(strip);
    if(value&&typeof value==='object') return Object.fromEntries(Object.entries(value).filter(([k])=>!['raws','source','inputs'].includes(k)).map(([k,v])=>[k,strip(v)]));
    return value;
  }
  return strip(copy);
};
const before=cssMeaning(css);
css.walk(node=>{
  let depth=0;for(let p=node.parent;p&&p.type!=='root';p=p.parent)depth++;
  node.raws.before='\n'+'  '.repeat(depth);
  if(node.type==='decl')node.raws.between=': ';
  if(node.nodes){node.raws.between=' ';node.raws.after='\n'+'  '.repeat(depth);node.raws.semicolon=true;}
});
const formatted=css.toString().trim()+'\n';
assert.deepEqual(cssMeaning(postcss.parse(formatted)),before);
fs.writeFileSync(cssFile,formatted);
records.push({file:cssFile,normalizedASTIdentical:true});
fs.writeFileSync('scratch/queueing-authoring/format-conservation.json',JSON.stringify({at:new Date().toISOString(),records},null,2));
console.log('Formatted six owned files with exact normalized AST conservation.');
