import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {parse} from '@babel/parser';
const root=path.resolve('.'),evidence=path.resolve('scratch/runtime-source-organization');
const safe=file=>{const absolute=path.resolve(file);assert.ok(absolute.startsWith(root+path.sep));return absolute;};
const walk=directory=>fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?walk(path.join(directory,entry.name)):[path.join(directory,entry.name)]);
const sourceFile='src/learn/data/scientific-visual-models.js';
const source=fs.readFileSync(sourceFile,'utf8'),ast=parse(source,{sourceType:'module'});
const namespace=await import('../src/learn/data/scientific-visual-models.js');
const mapping=JSON.parse(fs.readFileSync(path.join(evidence,'file-map.json'),'utf8'));
const snapshots=JSON.parse(fs.readFileSync(path.join(evidence,'export-snapshots.json'),'utf8'));
snapshots.push({file:sourceFile,values:JSON.parse(JSON.stringify(namespace,(_,value)=>typeof value==='function'?{function:value.toString()}:value))});
mapping.splits[sourceFile]={};
fs.copyFileSync(sourceFile,path.join(evidence,'before',sourceFile));
for(const [prefix,destination]of [['csv','src/learn/data/csv-parsing-model.js'],['pivot','src/learn/data/pandas-pivot-model.js']]){
 const declarations=ast.program.body.filter(node=>node.type==='ExportNamedDeclaration'&&(node.declaration.id?.name||node.declaration.declarations?.[0].id.name).startsWith(prefix));
 assert.equal(fs.existsSync(safe(destination)),false);
 fs.writeFileSync(safe(destination),declarations.map(node=>source.slice(node.start,node.end)).join('\n\n')+'\n');
 for(const node of declarations){const name=node.declaration.id?.name||node.declaration.declarations[0].id.name;mapping.splits[sourceFile][name]=[{file:destination,export:name,keys:null}];}
}
for(const file of walk('src/learn').concat(walk('scripts')).filter(file=>/\.(jsx?|mjs|cjs)$/.test(file)&&!file.endsWith('organize-lesson-assets.mjs'))){
 const text=fs.readFileSync(file,'utf8');if(!text.includes('scientific-visual-models'))continue;
 const updated=text.replace(/import\s*\{([^}]+)\}\s*from\s*['"]([^'"]*scientific-visual-models(?:\.js)?)['"];?/g,(_,members,location)=>{
  return ['csv','pivot'].map(prefix=>{const names=members.split(',').map(name=>name.trim()).filter(name=>name.startsWith(prefix));return names.length?`import { ${names.join(', ')} } from '${location.replace(/scientific-visual-models(?:\.js)?$/,prefix==='csv'?'csv-parsing-model.js':'pandas-pivot-model.js')}';`:'';}).filter(Boolean).join('\n');
 });if(updated!==text)fs.writeFileSync(safe(file),updated);
}
fs.unlinkSync(safe(sourceFile));

const original='public/learn-assets/programming-three';const assets=[];
for(const filename of fs.readdirSync(original)){
 const from=safe(path.join(original,filename)),kind=filename.endsWith('.ipynb')?'notebooks':'plots',to=safe(path.join('public/learn-assets',kind,filename));
 assert.ok(fs.statSync(from).isFile());assert.equal(fs.existsSync(to),false);assets.push({from,to});
}
// Verify every resolved source and target before moving any file.
assert.ok(assets.every(({from,to})=>from.startsWith(root+path.sep)&&to.startsWith(root+path.sep)));
fs.writeFileSync(path.join(evidence,'asset-moves.json'),JSON.stringify(assets,null,2));
for(const {from,to}of assets){fs.mkdirSync(path.dirname(to),{recursive:true});fs.renameSync(from,to);}
assert.equal(fs.readdirSync(safe(original)).length,0);fs.rmdirSync(safe(original));
for(const file of ['src/learn/components/lesson-labs/PlotOutput.jsx','src/learn/data/topics/reproducible-notebooks-experiment-structure.jsx']){
 const text=fs.readFileSync(file,'utf8');fs.writeFileSync(safe(file),text.replaceAll('/learn-assets/programming-three/','/learn-assets/'+(file.endsWith('PlotOutput.jsx')?'plots':'notebooks')+'/'));
}
let verifier=fs.readFileSync('scripts/verify-next-three.py','utf8');verifier=verifier.replace("assets=project/'public/learn-assets/programming-three'","assets=project/'public/learn-assets/plots'");fs.writeFileSync(safe('scripts/verify-next-three.py'),verifier);
verifier=fs.readFileSync('scripts/verify-programming-batch-three.mjs','utf8').replace('const assets = path.resolve("public/learn-assets/programming-three");','const assets = path.resolve("public/learn-assets/plots");\nconst notebookAssets = path.resolve("public/learn-assets/notebooks");').replace('fs.mkdirSync(assets, { recursive: true });','fs.mkdirSync(assets, { recursive: true });\nfs.mkdirSync(notebookAssets, { recursive: true });').replace('path.join(assets, artifact)','path.join(artifact.endsWith(".ipynb") ? notebookAssets : assets, artifact)');fs.writeFileSync(safe('scripts/verify-programming-batch-three.mjs'),verifier);
fs.writeFileSync(path.join(evidence,'file-map.json'),JSON.stringify(mapping,null,2));fs.writeFileSync(path.join(evidence,'export-snapshots.json'),JSON.stringify(snapshots));
console.log(`Separated CSV/pivot data; ${assets.length} assets moved into plots/notebooks with resolved paths checked.`);
