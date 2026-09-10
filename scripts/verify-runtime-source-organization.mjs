import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {parse} from '@babel/parser';

const root=path.resolve('.'),evidence=path.resolve('scratch/runtime-source-organization');
const mapping=JSON.parse(fs.readFileSync(path.join(evidence,'file-map.json'),'utf8'));
const snapshots=JSON.parse(fs.readFileSync(path.join(evidence,'export-snapshots.json'),'utf8'));
const serialize=value=>JSON.stringify(value,(_,entry)=>typeof entry==='function'?{function:entry.toString()}:typeof entry==='number'&&!Number.isFinite(entry)?{number:String(entry)}:entry);
let conserved=0;
for(const snapshot of snapshots){
 const exports=mapping.splits[snapshot.file];
 for(const [name,value]of Object.entries(snapshot.values)){
  const destinations=exports[name];assert.ok(destinations,snapshot.file+' '+name);
  let actual;
  for(const destination of destinations){const loaded=await import(pathToFileURL(path.resolve(destination.file)));actual=destinations.length===1?loaded[destination.export]:{...actual,...loaded[destination.export]};}
  assert.deepEqual(JSON.parse(serialize(actual)),value,snapshot.file+' '+name+' exact data/function conservation');conserved++;
 }
}
const files=[...new Set([...Object.values(mapping.splits).flatMap(exports=>Object.values(exports).flat().map(route=>route.file)),...Object.values(mapping.moves)])];
const oldFiles=[...Object.keys(mapping.splits),...Object.keys(mapping.moves)];
for(const file of oldFiles)assert.equal(fs.existsSync(path.resolve(file)),false,'Retired runtime module remains: '+file);
let parsed=0;
for(const file of files){if(!/\.jsx?$/.test(file))continue;parse(fs.readFileSync(file,'utf8'),{sourceType:'module',plugins:['jsx']});parsed++;}
const result={conservedDataExports:conserved,parsedRuntimeModules:parsed,retiredRuntimeFiles:oldFiles.length};
fs.writeFileSync(path.join(evidence,'conservation-results.json'),JSON.stringify(result,null,2));console.log('PASS runtime source organization:',result);
