import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import * as testingModels from "../src/learn/data/testing-models.js";
import * as notebookModels from "../src/learn/data/notebook-models.js";
import * as apiModels from "../src/learn/data/api-design-models.js";
const models = { ...testingModels, ...notebookModels, ...apiModels };
const root=path.resolve('scratch/reliability-review');fs.mkdirSync(root,{recursive:true});
const runRoot=fs.mkdtempSync(path.join(root,'run-'));const python=path.resolve('scratch/lesson-tools/Scripts/python.exe');
const runtime=path.join(runRoot,'runtime');fs.mkdirSync(path.join(runtime,'kernels/python3'),{recursive:true});
fs.writeFileSync(path.join(runtime,'kernels/python3/kernel.json'),JSON.stringify({argv:[python,'-m','ipykernel_launcher','-f','{connection_file}'],display_name:'Reliability verification',language:'python'}));
const env={...process.env,PYTHONIOENCODING:'utf-8',PYTHONPATH:'',JUPYTER_PATH:runtime,JUPYTER_RUNTIME_DIR:path.join(runtime,'jupyter'),IPYTHONDIR:path.join(runtime,'ipython')};
const examples={};
for(const slug of ['testing-debugging-dependency-management','reproducible-notebooks-experiment-structure','code-documentation-type-hints-api-design']){
 for(const reference of await collectLessonExamples(`src/learn/data/topics/${slug}.jsx`)){const key=`${slug}/${reference.key}`;examples[key]=reference.example;assert.ok(examples[key],key);}
}
const directories={};
for(const [key,ex] of Object.entries(examples)){
 const directory=path.join(runRoot,key.replaceAll('/','--'));fs.mkdirSync(directory,{recursive:true});directories[key]=directory;
 for(const[name,code]of Object.entries(ex.files||{}))fs.writeFileSync(path.join(directory,name),code);
 const filename=ex.filename||'lesson.py';fs.writeFileSync(path.join(directory,filename),ex.code);
 const r=spawnSync(python,['-B',filename],{cwd:directory,env,encoding:'utf8',timeout:90000});assert.equal(r.status,0,key+'\n'+r.stderr);assert.equal(r.stdout.replaceAll('\r\n','\n').trimEnd(),ex.output.trimEnd(),key);console.log(key+': output passed');
}
const fixture={examples,directories,mean:[],matrix:[],dependency:[],boundary:[],ownership:[],compatibility:[],random:[],provenance:[],notebook:[]};
for(const values of [[18],[18,24],[0,24]])for(const early of [true,false])fixture.mean.push({values,early,states:models.meanTrace(values,early)});
for(let mask=0;mask<16;mask++){const selected=models.temperatureCases.filter((_,i)=>mask&(1<<i)).map(c=>c.id);fixture.matrix.push({selected,rows:models.testMatrix(selected).map(({fn,...rest})=>rest)});}
for(const modern of [false,true])fixture.dependency.push({modern,...models.dependencyModel(modern)});
for(const kind of Object.keys(models.scoreFixtures))for(const minimum of [0,.85,1])fixture.boundary.push({kind,minimum,states:models.apiBoundary(kind,minimum)});
for(const mode of ['shared','mutate','copy'])fixture.ownership.push({mode,states:models.ownershipTrace(mode)});
for(const change of ['same','rename','exclusive','units'])fixture.compatibility.push({change,...models.compatibilityModel(change)});
for(const separate of [false,true])for(const extra of [false,true])fixture.random.push(models.randomConsumers(separate,extra));
for(const change of ['none','sameMean','offset','data','code'])for(const complete of [true,false])fixture.provenance.push({change,complete,...models.provenanceModel(change,complete)});
for(const actions of [[['all'],['edit',5],['display']],[['all'],['edit',5],['inputs'],['calculate'],['display']],[['all'],['restart'],['display']],[['restart'],['calculate']],[['edit',5],['all']]]){let state=models.initialNotebook();const states=[];for(const[action,value]of actions){state=models.notebookAction(state,action,value);states.push(state);}fixture.notebook.push({actions,states});}
const fixturePath=path.join(runRoot,'fixtures.json');fs.writeFileSync(fixturePath,JSON.stringify(fixture));
const r=spawnSync(python,['-B',path.resolve('scripts/verify-reliability-native.py'),fixturePath],{cwd:runRoot,env,encoding:'utf8',timeout:180000});process.stdout.write(r.stdout);assert.equal(r.status,0,r.stderr);fs.writeFileSync(path.join(root,'latest.json'),JSON.stringify({runRoot,fixturePath,examples:Object.keys(examples).length,checkedAt:new Date().toISOString()},null,2));console.log(`${Object.keys(examples).length} displayed programs, independent checks and kernels passed. Evidence: ${runRoot}`);
