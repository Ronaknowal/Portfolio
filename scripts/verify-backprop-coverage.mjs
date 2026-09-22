import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {parse} from '@babel/parser';
const packet='docs/teaching/drafts/backpropagation-automatic-differentiation/lesson.md';
const body='src/learn/data/topics/backprop.jsx';
const markdown=fs.readFileSync(packet,'utf8').replace(/\r/g,'');
const runtime=fs.readFileSync(body,'utf8');
const tree=parse(runtime,{sourceType:'module',plugins:['jsx']});
const walk=(node,fn)=>{if(!node||typeof node!=='object')return;fn(node);for(const value of Object.values(node))if(Array.isArray(value))value.forEach(child=>walk(child,fn));else if(value&&typeof value==='object')walk(value,fn);};
const jsxText=node=>node.type==='JSXText'?node.value:node.type==='JSXExpressionContainer'&&node.expression.type==='StringLiteral'?node.expression.value:(node.children||[]).map(jsxText).join('');
const content={Prose:[],MathBlock:[],CodeBlock:[],H2:[],H3:[]};const tables=[];
walk(tree,node=>{if(node.type==='JSXElement'){
 const name=node.openingElement.name.name;
 if(content[name])content[name].push(jsxText(node));
 if(name==='BackpropTable')tables.push(node.openingElement.attributes.find(a=>a.name?.name==='rows').value.expression.elements.map(row=>row.elements.map(jsxText)));
}});
const normalize=text=>text.replace(/\s/g,'');
const plain=text=>text.replace(/\[([^\]]+)\]\([^\s)]+\)/g,'$1').replace(/\$([^$]+)\$/g,'$1').replace(/`([^`]+)`/g,'$1').replace(/\*\*([^*]+)\*\*/g,'$1');
const missing=[];let checkedParagraphs=0;
for(let block of markdown.replace(/```[\s\S]*?```/g,'').split(/\n\s*\n/)){
 if(!block.trim()||/^(#|\||<|```|\d+\. |- |\$\$|\[Visual)/.test(block))continue;
 if(block.includes('```'))continue;
 block=block.replace('Independent replay of this complete custom-op program belongs to the finishing phase; its piecewise rule is derived here.', 'The complete custom-operation program is included among the independently replayed native examples; its piecewise rule is derived here.').replace('The expected result isTrue for this deterministic block; independent execution of this excerpt is deferred.', 'The executed result is True for this deterministic block.').replace('is expected to pass at those smooth points', 'passes at those tested smooth points');
 const target=normalize(plain(block));
 if(!content.Prose.some(value=>normalize(value)===target))missing.push(block.slice(0,140));else checkedParagraphs++;
}
const sourceCode=[...markdown.matchAll(/```[^\n]*\n([\s\S]*?)\n```/g)].map(m=>m[1]);
assert.deepEqual(content.CodeBlock,sourceCode);
const equations=[...markdown.matchAll(/\$\$\n([\s\S]*?)\n\$\$/g)].map(m=>m[1]);assert.deepEqual(content.MathBlock,equations);
const sourceTables=[];const lines=markdown.split('\n');for(let i=0;i<lines.length;i++){if(lines[i].startsWith('|')){const rows=[];while(i<lines.length&&lines[i].startsWith('|'))rows.push(lines[i++].trim().slice(1,-1).split('|').map(s=>plain(s.trim())));sourceTables.push(rows.slice(2));}}
assert.deepEqual(tables.map(table=>table.map(row=>row.map(normalize))),sourceTables.map(table=>table.map(row=>row.map(normalize))));
const record={status:missing.length?'needs-disposition':'passed',scope:'Full prepared prose/equation/code/table preservation, permitting only whitespace normalization and three native-execution wording updates.',checkedParagraphs,missing,displayEquations:equations.length,codeBlocks:sourceCode.length,tables:tables.length,bodyHash:crypto.createHash('sha256').update(runtime).digest('hex'),packetHash:crypto.createHash('sha256').update(markdown).digest('hex')};
fs.writeFileSync('docs/teaching/evidence/backprop-independent-coverage.json',JSON.stringify(record,null,2)+'\n');console.log(JSON.stringify(record));

