import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {parse} from '@babel/parser';
import traverseModule from '@babel/traverse';
const traverse=traverseModule.default;
const root=path.resolve('.'),labs='src/learn/components/lesson-labs/';
const read=file=>fs.readFileSync(file,'utf8');
const write=(file,source)=>{const absolute=path.resolve(file);assert.ok(absolute.startsWith(root+path.sep));fs.writeFileSync(absolute,source);};
const astOf=source=>parse(source,{sourceType:'unambiguous',plugins:['jsx']});
const edit=(source,changes)=>changes.sort((a,b)=>b.start-a.start).reduce((text,change)=>text.slice(0,change.start)+change.text+text.slice(change.end),source);
function renameBinding(source,scope,before,after){
 const binding=scope.getBinding(before);if(!binding||before===after)return source;assert.ok(!scope.hasOwnBinding(after),'Binding collision: '+after);
 const changes=[],nodes=[binding.identifier,...binding.referencePaths.map(reference=>reference.node)];
 for(const node of nodes){if(!['Identifier','JSXIdentifier'].includes(node.type)||changes.some(change=>change.start===node.start&&change.end===node.end))continue;changes.push({start:node.start,end:node.end,text:after});}
 return edit(source,changes);
}
function programScope(source){let scope;traverse(astOf(source),{Program(p){scope=p.scope;p.stop();}});return scope;}
function extractHelpers(sourceFile,names,targetFile,preamble=''){
 const source=read(sourceFile),ast=astOf(source),helpers=ast.program.body.filter(node=>(node.type==='FunctionDeclaration'&&names.includes(node.id.name))||(node.type==='VariableDeclaration'&&node.declarations.some(declaration=>names.includes(declaration.id.name))));
 assert.equal(helpers.length,names.length);write(targetFile,preamble+helpers.map(node=>'export '+source.slice(node.start,node.end)).join('\n\n')+'\n');
}
function replaceHelpers(file,names,sharedFile){
 let source=read(file),ast=astOf(source);const removed=ast.program.body.filter(node=>(node.type==='FunctionDeclaration'&&names.includes(node.id.name))||(node.type==='VariableDeclaration'&&node.declarations.some(declaration=>names.includes(declaration.id.name))));
 const used=removed.flatMap(node=>node.id?[node.id.name]:node.declarations.map(declaration=>declaration.id.name));
 source=edit(source,removed.map(node=>({start:node.start,end:node.end,text:''})));source=`import { ${used.join(', ')} } from './${sharedFile}';\n`+source;write(file,source);
}
extractHelpers(labs+'NotebookLabs.jsx',['display','Choices','Feedback'],labs+'LabControls.jsx',"import { useId } from 'react';\n\n");
for(const file of ['NotebookLabs.jsx','TestingLabs.jsx','ApiDesignLabs.jsx'])replaceHelpers(labs+file,['display','Choices','Feedback'],'LabControls.jsx');
extractHelpers(labs+'ThreadCoordinationLabs.jsx',['Lab','State','Log'],labs+'MechanismLab.jsx',"import { useId } from 'react';\nimport './workflow-labs.css';\n\n");
for(const file of ['BashWorkflowLabs.jsx','ThreadCoordinationLabs.jsx'])replaceHelpers(labs+file,['Lab','State','Log'],'MechanismLab.jsx');
extractHelpers(labs+'ScientificFileLabs.jsx',['StepControls','Prediction'],labs+'DataLabControls.jsx');
for(const file of ['ScientificFileLabs.jsx','SqlLabs.jsx'])replaceHelpers(labs+file,['StepControls','Prediction'],'DataLabControls.jsx');

const helperRenames={display:'formatLabValue',Choices:'LabChoices',Feedback:'LabFeedback',Lab:'MechanismLab',State:'StatePanel',Log:'ExecutionLog',StepControls:'DataStepControls',Prediction:'DataPrediction'};
for(const file of ['LabControls.jsx','NotebookLabs.jsx','TestingLabs.jsx','ApiDesignLabs.jsx','MechanismLab.jsx','BashWorkflowLabs.jsx','ThreadCoordinationLabs.jsx','DataLabControls.jsx','ScientificFileLabs.jsx','SqlLabs.jsx']){
 let source=read(labs+file);
 for(const [before,after]of Object.entries(helperRenames))source=renameBinding(source,programScope(source),before,after);
 // Renaming an imported binding also renames its source export when this is the
 // shared helper migration; never rewrite unrelated lesson APIs.
 const ast=astOf(source),changes=[];
 for(const declaration of ast.program.body)if(declaration.type==='ImportDeclaration'&&/\/(LabControls|MechanismLab|DataLabControls)\.jsx$/.test(declaration.source.value))for(const specifier of declaration.specifiers)if(helperRenames[specifier.imported?.name])changes.push({start:specifier.imported.start,end:specifier.imported.end,text:helperRenames[specifier.imported.name]});
 write(labs+file,edit(source,changes));
}

const localRenames={
 'NotebookLabs.jsx':{NotebookKernelLab:{s:'notebookState',setS:'setNotebookState',act:'executeNotebookAction'},RandomStreamLab:{m:'streamModel'},ProvenanceLab:{m:'provenance'}},
 'TestingLabs.jsx':{DebugExecutionLab:{s:'executionState'},DependencyConstraintsLab:{m:'compatibility'}},
 'ApiDesignLabs.jsx':{ApiBoundaryLab:{s:'boundaryState'},ApiOwnershipLab:{s:'ownershipState'},ApiCompatibilityLab:{m:'compatibility'}},
 'IteratorLabs.jsx':{CursorOwnershipLab:{s:'cursorState'},GeneratorFrameLab:{s:'generatorState'},PullPipelineLab:{s:'pipelineState'}},
 'DecoratorContextLabs.jsx':{DecoratorOrderLab:{s:'wrapperState'},ContextLifetimeLab:{s:'contextState'},ExitStackLab:{s:'cleanupState'}},
 'ThreadCoordinationLabs.jsx':{ThreadConditionLab:{s:'conditionState'}},
 'BashWorkflowLabs.jsx':{BashStatusLab:{strict:'pipefailEnabled',setStrict:'setPipefailEnabled'}}
};
for(const [file,functions]of Object.entries(localRenames)){
 let source=read(labs+file);
 for(const [functionName,bindings]of Object.entries(functions))for(const [before,after]of Object.entries(bindings)){
  let scope;traverse(astOf(source),{FunctionDeclaration(p){if(p.node.id.name===functionName){scope=p.scope;p.stop();}}});if(scope)source=renameBinding(source,scope,before,after);
 }
 write(labs+file,source);
}

const topicFiles=fs.readdirSync('src/learn/data/topics').filter(file=>file.endsWith('.jsx'));
const aliases=[];
for(const name of topicFiles){
 const file='src/learn/data/topics/'+name;let source=read(file);
 if(!/import[^;]*\bas (ex|added|native|practice|reference|foundations)\b/.test(source))continue;
 const ast=astOf(source),bindings=ast.program.body.filter(node=>node.type==='ImportDeclaration'&&/examples/.test(node.source.value)).flatMap(node=>node.specifiers).filter(specifier=>specifier.type==='ImportSpecifier'&&specifier.imported.name!==specifier.local.name&&['ex','added','native','practice','reference','foundations'].includes(specifier.local.name)).map(specifier=>[specifier.local.name,specifier.imported.name]);
 for(const [before,after]of bindings){source=renameBinding(source,programScope(source),before,after);aliases.push({file,before,after});}write(file,source);
}

// Consolidate named imports and remove unused imported symbols introduced by
// the split, without touching example strings or reformatting whole lessons.
const candidates=fs.readdirSync(labs).filter(file=>file.endsWith('.jsx')).map(file=>labs+file).concat(topicFiles.map(file=>'src/learn/data/topics/'+file));
for(const file of candidates){
 let source=read(file);if(!/(LessonInvestigation|Models\.js|models\.js|Examples|examples|LabControls|MechanismLab|DataLabControls)/.test(source))continue;
 let ast;try{ast=astOf(source);}catch{continue;}
 const scope=programScope(source),groups=new Map(),changes=[];
 for(const node of ast.program.body){if(node.type!=='ImportDeclaration'||!node.specifiers.length||!node.specifiers.every(specifier=>specifier.type==='ImportSpecifier'))continue;
  const used=node.specifiers.filter(specifier=>scope.getBinding(specifier.local.name)?.referencePaths.length);
  if(!used.length){changes.push({start:node.start,end:node.end,text:''});continue;}
  const previous=groups.get(node.source.value);if(previous){previous.specifiers.push(...used);changes.push({start:node.start,end:node.end,text:''});}else groups.set(node.source.value,{node,specifiers:used});
 }
 for(const [location,{node,specifiers}]of groups)changes.push({start:node.start,end:node.end,text:`import { ${specifiers.map(specifier=>specifier.imported.name+(specifier.local.name!==specifier.imported.name?' as '+specifier.local.name:'')).join(', ')} } from ${JSON.stringify(location)};`});
 source=edit(source,changes);write(file,source);
}
fs.writeFileSync('scratch/runtime-source-organization/lesson-alias-map.json',JSON.stringify(aliases,null,2));
console.log(`Shared controls extracted; meaningful state names applied; ${aliases.length} example import aliases clarified.`);
