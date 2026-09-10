// One-time, guarded migration. Runtime code is split by Babel binding dependency
// closures; source code inside examples/functions is copied rather than rewritten.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {parse} from '@babel/parser';
import traverseModule from '@babel/traverse';
const traverse=traverseModule.default;
const repo=path.resolve('.');
const evidence=path.resolve('scratch/runtime-source-organization');
fs.mkdirSync(evidence,{recursive:true});
const parseSource=source=>parse(source,{sourceType:'unambiguous',plugins:['jsx']});
const slash=value=>value.replaceAll('\\','/');
const safePath=relative=>{const absolute=path.resolve(repo,relative);assert.ok(absolute.startsWith(repo+path.sep)&&absolute!==repo,relative);return absolute;};
const read=relative=>fs.readFileSync(safePath(relative),'utf8');
const outputFiles=new Map(),routes=new Map(),sourceFiles=new Set(),snapshots=[];
const data='src/learn/data/',labs='src/learn/components/lesson-labs/';
const addRoute=(oldFile,oldExport,newFile,newExport=oldExport,keys=null)=>{
 const key=slash(safePath(oldFile));const map=routes.get(key)||new Map();const destinations=map.get(oldExport)||[];
 destinations.push({file:newFile,export:newExport,keys});map.set(oldExport,destinations);routes.set(key,map);sourceFiles.add(oldFile);
};
function sourceEdits(source,edits){return edits.sort((a,b)=>b.start-a.start).reduce((text,edit)=>text.slice(0,edit.start)+edit.text+text.slice(edit.end),source);}
function moduleInfo(source){
 const ast=parseSource(source);let scope;traverse(ast,{Program(p){scope=p.scope;p.stop();}});
 const body=ast.program.body;
 const owner=nodePath=>{let current=nodePath;while(current.parentPath&&!current.parentPath.isProgram())current=current.parentPath;return body.indexOf(current.node);};
 const declarations=new Map();
 for(const [name,binding]of Object.entries(scope.bindings))declarations.set(name,{binding,index:owner(binding.path)});
 const exports=new Map();body.forEach((node,index)=>{if(node.type==='ExportNamedDeclaration'&&node.declaration){const declaration=node.declaration;if(declaration.type==='VariableDeclaration')for(const item of declaration.declarations)exports.set(item.id.name,index);else if(declaration.id)exports.set(declaration.id.name,index);}});
 return {ast,scope,body,owner,declarations,exports};
}
async function snapshot(oldFile){
 if(!oldFile.endsWith('.js')||oldFile.includes('/components/'))return;
 const namespace=await import(pathToFileURL(safePath(oldFile)));snapshots.push({file:oldFile,values:namespace});
}
function renderSplit(source,rootExports,rename={}){
 const info=moduleInfo(source),selected=new Set(rootExports.map(name=>{assert.ok(info.exports.has(name),name);return info.exports.get(name);}));
 // Include top-level mutations of selected example collections (e.g. frame).
 for(const name of rootExports)for(const reference of info.declarations.get(name)?.binding.referencePaths||[]){const index=info.owner(reference);if(info.body[index]?.type==='ExpressionStatement')selected.add(index);}
 let changed=true;while(changed){changed=false;for(const {binding,index}of info.declarations.values()){
  if(selected.has(index))continue;
  if(binding.referencePaths.some(reference=>selected.has(info.owner(reference)))){selected.add(index);changed=true;}
 }}
 const imports=[];
 info.body.forEach((node,index)=>{if(node.type!=='ImportDeclaration')return;if(!node.specifiers.length){imports.push(source.slice(node.start,node.end));return;}const needed=node.specifiers.filter(specifier=>info.declarations.get(specifier.local.name)?.binding.referencePaths.some(reference=>selected.has(info.owner(reference))));if(!needed.length)return;const names=needed.map(specifier=>specifier.type==='ImportSpecifier'?`${specifier.imported.name}${specifier.imported.name===specifier.local.name?'':' as '+specifier.local.name}`:specifier.type==='ImportNamespaceSpecifier'?`* as ${specifier.local.name}`:specifier.local.name);const named=needed.every(specifier=>specifier.type==='ImportSpecifier');imports.push(`import ${named?'{ '+names.join(', ')+' }':names.join(', ')} from ${JSON.stringify(node.source.value)};`);});
 let body=info.body.filter((node,index)=>selected.has(index)&&node.type!=='ImportDeclaration').map(node=>source.slice(node.start,node.end)).join('\n\n');
 if(Object.keys(rename).length){const local=moduleInfo(body),edits=[];for(const [before,after]of Object.entries(rename)){const binding=local.scope.getBinding(before);assert.ok(binding,before);for(const node of [binding.identifier,...binding.referencePaths.map(reference=>reference.node)].filter(node=>node.type==='Identifier'))if(!edits.some(edit=>edit.start===node.start&&edit.end===node.end))edits.push({start:node.start,end:node.end,text:after});}body=sourceEdits(body,edits);}
 return imports.join('\n')+(imports.length?'\n\n':'')+body+'\n';
}
async function splitExports(oldFile,groups,{source=read(oldFile)}={}){
 await snapshot(oldFile);
 const original=moduleInfo(source);const covered=new Set();
 for(const group of groups){
  const [newFile,names,rename={}]=group;names.forEach(name=>covered.add(name));
  outputFiles.set(newFile,renderSplit(source,names,rename));
  for(const name of names)addRoute(oldFile,name,newFile,rename[name]||name);
 }
 assert.deepEqual([...covered].sort(),[...original.exports.keys()].sort(),oldFile+' export conservation');
}
async function splitObject(oldFile,oldExport,groups){
 await snapshot(oldFile);const source=read(oldFile),info=moduleInfo(source),declaration=info.body[info.exports.get(oldExport)].declaration.declarations.find(node=>node.id.name===oldExport);assert.equal(declaration.init.type,'ObjectExpression');
 const covered=[];
 for(const [newFile,newExport,predicate]of groups){const properties=declaration.init.properties.filter(property=>predicate(property.key.name||property.key.value));const keys=properties.map(property=>property.key.name||property.key.value);covered.push(...keys);outputFiles.set(newFile,`// Independently runnable ${newExport.replace(/Examples$/,'')} examples; verified code/output pairs.\nexport const ${newExport} = {\n${properties.map(property=>source.slice(property.start,property.end)).join(',\n')}\n};\n`);addRoute(oldFile,oldExport,newFile,newExport,keys);}
 assert.deepEqual(covered.sort(),declaration.init.properties.map(property=>property.key.name||property.key.value).sort(),oldFile+' property conservation');
}

await splitObject(data+'programming-batch-one-examples.js','programmingExamples',[
 [data+'python-core-examples.js','pythonCoreExamples',key=>key.startsWith('basics')],
 [data+'oop-core-examples.js','objectOrientedCoreExamples',key=>key.startsWith('oop')],
 [data+'iterator-core-examples.js','iteratorCoreExamples',key=>key.startsWith('iter')]
]);
await splitObject(data+'programming-batch-two-examples.js','batchTwoExamples',[
 [data+'decorator-core-examples.js','decoratorCoreExamples',key=>key.startsWith('decorator')||key.startsWith('context')],
 [data+'testing-examples.js','testingExamples',key=>key.startsWith('testing')],
 [data+'numpy-reference-examples.js','numpyReferenceExamples',key=>key.startsWith('numpy')]
]);
await splitExports(data+'programming-batch-three-examples.js',[
 [data+'plotting-examples.js',['plottingExamples']], [data+'notebook-examples.js',['notebookExamples']], [data+'api-design-examples.js',['apiExamples']]
]);
await splitExports(data+'programming-batch-four-examples.js',[
 [data+'git-examples.js',['gitExamples']], [data+'linux-command-examples.js',['linuxExamples']]
]);
await splitExports(data+'next-three-examples.js',[
 [data+'pandas-practice-examples.js',['pandasNewExamples'],{pandasNewExamples:'pandasPracticeExamples'}],
 [data+'plotting-practice-examples.js',['plottingNewExamples'],{plottingNewExamples:'plottingPracticeExamples'}],
 [data+'git-practice-examples.js',['gitNewExamples'],{gitNewExamples:'gitPracticeExamples'}]
]);
await splitExports(data+'programming-batch-two-traces.js',[
 [data+'decorator-introduction-trace.js',['decoratorTrace']], [data+'debugging-introduction-trace.js',['debuggingTrace']]
]);
await splitExports(data+'workflow-completion-models.js',[
 [data+'bash-workflow-models.js',['argumentCases','argumentTrace','pipelineStatus','publicationTrace']],
 [data+'thread-coordination-models.js',['initialRace','raceStep','initialLocks','lockStep','waitEdges','deadlocked','conditionTrace']]
]);
await splitExports(data+'reliability-models.js',[
 [data+'testing-models.js',['meanTrace','temperatureCases','temperatureMutants','testMatrix','dependencyModel']],
 [data+'notebook-models.js',['initialNotebook','notebookAction','randomConsumers','provenanceModel']],
 [data+'api-design-models.js',['scoreFixtures','apiBoundary','ownershipTrace','compatibilityModel']]
]);
await splitObject(data+'reliability-examples.js','reliabilityExamples',[
 [data+'testing-practice-examples.js','testingPracticeExamples',key=>key==='testingRepair'],
 [data+'notebook-practice-examples.js','notebookPracticeExamples',key=>key==='notebookTransfer'],
 [data+'api-practice-examples.js','apiPracticeExamples',key=>['ownership','apiTransfer'].includes(key)]
]);
let iterationSource=read(data+'iteration-decorator-examples.js');
iterationSource=iterationSource.replace("import { programmingExamples } from './programming-batch-one-examples.js';","import { iteratorCoreExamples } from './iterator-core-examples.js';").replace("import { batchTwoExamples } from './programming-batch-two-examples.js';","import { decoratorCoreExamples } from './decorator-core-examples.js';");
iterationSource=iterationSource.replace("Object.fromEntries(Object.entries(programmingExamples).filter(([key])=>key.startsWith('iter')))",'{...iteratorCoreExamples}').replace("Object.fromEntries(Object.entries(batchTwoExamples).filter(([key])=>key.startsWith('decorator')||key.startsWith('context')))",'{...decoratorCoreExamples}');
await splitExports(data+'iteration-decorator-examples.js',[[data+'iterator-examples.js',['iterationExamples']],[data+'decorator-examples.js',['decoratorExamples']]],{source:iterationSource});
await splitExports(data+'iteration-decorator-models.js',[[data+'iterator-models.js',['cursorTrace','generatorTrace','pipelineTrace']],[data+'decorator-context-models.js',['decoratorOrderTrace','contextTrace','exitStackTrace']]]);
await splitExports(data+'data-foundations-models.js',[
 [data+'scientific-file-models.js',['measurementFields','measurementCases','measurementRows','measurementModel','oldMeasurementFile','newMeasurementFile','publicationTrace']],
 [data+'sql-models.js',['sqlSensors','sqlReadings','joinModel','transactionTrace']]
]);
await splitExports(data+'data-foundations-examples.js',[[data+'scientific-file-examples.js',['measurementValidator','fileExamples']],[data+'sql-examples.js',['sensorDatabase','featureQuery','sqlExamples']]]);
await splitExports(labs+'WorkflowCompletionLabs.jsx',[[labs+'BashWorkflowLabs.jsx',['BashArgumentsLab','BashStatusLab','BashPublicationLab']],[labs+'ThreadCoordinationLabs.jsx',['ThreadRaceLab','ThreadConditionLab','ThreadDeadlockLab']]]);
await splitExports(labs+'ReliabilityLabs.jsx',[[labs+'TestingLabs.jsx',['DebugExecutionLab','TestDiscriminationLab','DependencyConstraintsLab']],[labs+'NotebookLabs.jsx',['NotebookKernelLab','RandomStreamLab','ProvenanceLab']],[labs+'ApiDesignLabs.jsx',['ApiBoundaryLab','ApiOwnershipLab','ApiCompatibilityLab']]]);
await splitExports(labs+'IterationDecoratorLabs.jsx',[[labs+'IteratorLabs.jsx',['CursorOwnershipLab','GeneratorFrameLab','PullPipelineLab']],[labs+'DecoratorContextLabs.jsx',['DecoratorOrderLab','ContextLifetimeLab','ExitStackLab']]]);
await splitExports(labs+'DataFoundationsLabs.jsx',[[labs+'ScientificFileLabs.jsx',['FileSchemaLab','FilePublicationLab']],[labs+'SqlLabs.jsx',['SqlJoinLab','SqlTransactionLab']]]);
await splitExports(labs+'WorkflowVisuals.jsx',[
 [labs+'NotebookFigures.jsx',['NotebookStatePicture']], [labs+'ApiDesignFigures.jsx',['ApiCheckingPicture']],
 [labs+'BashFigures.jsx',['BashArgumentPicture','BashPipelinePicture']], [labs+'ThreadFigures.jsx',['ThreadRacePicture']], [labs+'GitFigures.jsx',['GitRemotePicture']]
]);
await splitExports(labs+'ScientificConceptVisuals.jsx',[
 [labs+'NumpyFigures.jsx',['NumpyShapeComparison']],[labs+'ScientificFileFigures.jsx',['CsvBoundaryDiagram']],[labs+'SqlFigures.jsx',['SqlRelationshipDiagram']],
 [labs+'PandasPivotLab.jsx',['PandasPivotLab']],[labs+'PlotOwnershipFigure.jsx',['PlotOwnershipDiagram']]
]);
await splitExports(labs+'PythonMechanismFigures.jsx',[
 [labs+'PythonReferenceFigure.jsx',['ReferenceSetupFigure']],[labs+'BoundMethodFigure.jsx',['BoundMethodRetentionFigure']],
 [labs+'IteratorFigures.jsx',['CursorOwnershipMap']],[labs+'DecoratorContextFigures.jsx',['DecoratorBindingFigure','ContextRouteMap']],[labs+'TestingBoundaryFigure.jsx',['TestingBoundaryFigure']]
]);

const directMoves=new Map([
 [data+'bash-completion-examples.js',data+'bash-workflow-examples.js'],
 [data+'thread-completion-examples.js',data+'thread-coordination-examples.js'],
 [labs+'NextThreeElements.jsx',labs+'LessonInvestigation.jsx'],
 [labs+'next-three.css',labs+'lesson-investigations.css'],
 [labs+'WorkflowCompletionElements.jsx',labs+'RunnableExample.jsx'],
 [labs+'WorkflowCompletion.css',labs+'workflow-labs.css'],
 [labs+'programming-three.css',labs+'plot-output.css'],
 [labs+'WorkflowVisuals.css',labs+'workflow-figures.css']
]);
for(const [before,after]of directMoves){outputFiles.set(after,read(before));sourceFiles.add(before);}

const mapping=Object.fromEntries([...routes].map(([file,exports])=>[slash(path.relative(repo,file)),Object.fromEntries(exports)]));
fs.writeFileSync(path.join(evidence,'file-map.json'),JSON.stringify({splits:mapping,moves:Object.fromEntries(directMoves)},null,2));
const serialize=value=>JSON.stringify(value,(_,entry)=>typeof entry==='function'?{function:entry.toString()}:typeof entry==='number'&&!Number.isFinite(entry)?{number:String(entry)}:entry);
fs.writeFileSync(path.join(evidence,'export-snapshots.json'),serialize(snapshots));
for(const file of sourceFiles){const copy=path.join(evidence,'before',file);fs.mkdirSync(path.dirname(copy),{recursive:true});fs.copyFileSync(safePath(file),copy);}

function resolveImport(file,specifier){if(!specifier.startsWith('.'))return null;const target=path.resolve(path.dirname(safePath(file)),specifier);return [target,target+'.js',target+'.jsx',target+'.css'].find(candidate=>routes.has(slash(candidate))||directMoves.has(slash(path.relative(repo,candidate))));}
function relativeImport(from,to){let result=slash(path.relative(path.dirname(safePath(from)),safePath(to)));if(!result.startsWith('.'))result='./'+result;return result;}
function migrationImports(file,source){
 const references=[...sourceFiles].map(name=>path.basename(name).replace(/\.(jsx?|css)$/,''));
 if(!references.some(name=>source.includes(name)))return source;
 let info;try{info=moduleInfo(source);}catch(error){error.message=file+': '+error.message;throw error;}const edits=[];
 for(const node of info.body){
  if(node.type!=='ImportDeclaration')continue;
  const old=resolveImport(file,node.source.value);if(!old)continue;
  const moved=directMoves.get(slash(path.relative(repo,old)));
  if(moved){edits.push({start:node.source.start,end:node.source.end,text:JSON.stringify(relativeImport(file,moved))});continue;}
  const map=routes.get(slash(old)),imports=[],aggregates=[];
  for(const specifier of node.specifiers){
   const local=specifier.local.name;
   if(specifier.type==='ImportNamespaceSpecifier'){
    const all=[...new Set([...map.values()].flat().map(route=>route.file))];const variables=all.map((destination,index)=>`${local}Module${index+1}`);
    all.forEach((destination,index)=>imports.push(`import * as ${variables[index]} from ${JSON.stringify(relativeImport(file,destination))};`));aggregates.push(`const ${local} = { ${variables.map(name=>'...'+name).join(', ')} };`);continue;
   }
   assert.equal(specifier.type,'ImportSpecifier',file);let destinations=map.get(specifier.imported.name);assert.ok(destinations,file+' '+specifier.imported.name);
   if(destinations.length>1){
    const references=info.scope.getBinding(local)?.referencePaths||[];
    const keys=references.map(reference=>reference.parentPath.isMemberExpression()&&reference.parent.object===reference.node&&!reference.parent.computed?reference.parent.property.name:null);
    if(keys.length&&keys.every(Boolean))destinations=destinations.filter(destination=>keys.some(key=>destination.keys.includes(key)));
    if(destinations.length>1){destinations.forEach((destination,index)=>imports.push(`import { ${destination.export} as ${local}Part${index+1} } from ${JSON.stringify(relativeImport(file,destination.file))};`));aggregates.push(`const ${local} = { ${destinations.map((_,index)=>`...${local}Part${index+1}`).join(', ')} };`);continue;}
   }
   const destination=destinations[0];imports.push(`import { ${destination.export}${destination.export===local?'':' as '+local} } from ${JSON.stringify(relativeImport(file,destination.file))};`);
  }
  edits.push({start:node.start,end:node.end,text:imports.join('\n')+(aggregates.length?'\n'+aggregates.join('\n'):'')});
 }
 // A few verification scripts use dynamic imports of these old shared modules.
 traverse(info.ast,{CallExpression(p){if(p.node.callee.type!=='Import'||p.node.arguments[0]?.type!=='StringLiteral')return;const literal=p.node.arguments[0],old=resolveImport(file,literal.value);if(!old)return;const moved=directMoves.get(slash(path.relative(repo,old)));if(moved){edits.push({start:literal.start,end:literal.end,text:JSON.stringify(relativeImport(file,moved))});return;}const map=routes.get(slash(old));let destinations=[...new Set([...map.values()].flat().map(route=>route.file))];const pattern=p.parentPath.parentPath?.node?.id;if(pattern?.type==='ObjectPattern'){const names=pattern.properties.map(property=>property.key.name);destinations=[...new Set(names.flatMap(name=>map.get(name)||[]).map(route=>route.file))];}edits.push({start:p.node.start,end:p.node.end,text:destinations.length===1?`import(${JSON.stringify(relativeImport(file,destinations[0]))})`:`Promise.all([${destinations.map(destination=>`import(${JSON.stringify(relativeImport(file,destination))})`).join(', ')}]).then(modules => Object.assign({}, ...modules))`});}});
 return sourceEdits(source,edits);
}
function walk(directory){return fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?walk(path.join(directory,entry.name)):[path.join(directory,entry.name)]);}
const excluded=new Set(['src/learn/data/topics/index.js','src/learn/components/TopicContent.jsx','src/learn/Reader.jsx','src/learn/components/PlaceholderContent.jsx','src/learn/data/track-definitions.js']);
const consumers=walk(safePath('src/learn')).concat(walk(safePath('scripts'))).map(absolute=>slash(path.relative(repo,absolute))).filter(file=>/\.(jsx?|mjs|cjs)$/.test(file)&&!file.startsWith('src/learn/data/curriculum/')&&!excluded.has(file)&&!sourceFiles.has(file)&&file!=='scripts/reorganize-runtime-sources.mjs');
for(const file of consumers){const original=read(file),updated=migrationImports(file,original);if(updated!==original)outputFiles.set(file,updated);}
for(const [file,source]of outputFiles)if(/\.(jsx?|mjs|cjs)$/.test(file))outputFiles.set(file,migrationImports(file,source));

// Verify every absolute source/destination before any move/removal.
const verifiedPaths=[...new Set([...sourceFiles,...outputFiles.keys()])].map(safePath);assert.ok(verifiedPaths.every(absolute=>absolute.startsWith(repo+path.sep)));
for(const file of outputFiles.keys())if(!consumers.includes(file))assert.equal(fs.existsSync(safePath(file)),false,'Destination already exists: '+file);
fs.writeFileSync(path.join(evidence,'verified-workspace-paths.json'),JSON.stringify(verifiedPaths,null,2));
for(const [file,source]of outputFiles){fs.mkdirSync(path.dirname(safePath(file)),{recursive:true});fs.writeFileSync(safePath(file),source);}
for(const file of sourceFiles)fs.unlinkSync(safePath(file));

// Deep conservation of every data export, including inherited/mutated example sets.
let conserved=0;
for(const snapshot of snapshots){const map=routes.get(slash(safePath(snapshot.file)));for(const [name,value]of Object.entries(snapshot.values)){const destinations=map.get(name);assert.ok(destinations,snapshot.file+' '+name);let actual;for(const destination of destinations){const loaded=await import(pathToFileURL(safePath(destination.file)));actual=destinations.length===1?loaded[destination.export]:{...actual,...loaded[destination.export]};}assert.deepEqual(JSON.parse(serialize(actual)),JSON.parse(serialize(value)),snapshot.file+' '+name+' value conservation');conserved++;}}
fs.writeFileSync(path.join(evidence,'migration-results.json'),JSON.stringify({dataExportsConserved:conserved,removedSourceFiles:sourceFiles.size,writtenFiles:outputFiles.size,checkedPaths:verifiedPaths.length},null,2));
console.log(`Runtime migration: ${conserved} data exports conserved; ${sourceFiles.size} old modules replaced; ${outputFiles.size} source/consumer files written. Build deferred to integration.`);
