import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import {pandasExamples} from '../src/learn/data/pandas-examples.js';
import { plottingExamples } from "../src/learn/data/plotting-examples.js";
import { gitExamples } from "../src/learn/data/git-examples.js";
import { pandasPracticeExamples as pandasNewExamples } from "../src/learn/data/pandas-practice-examples.js";
import { plottingPracticeExamples as plottingNewExamples } from "../src/learn/data/plotting-practice-examples.js";
import { gitPracticeExamples as gitNewExamples } from "../src/learn/data/git-practice-examples.js";
import {alignmentModel,cleaningModel,groupingModel} from '../src/learn/data/pandas-foundations-model.js';
import {joinOrders,lookupRows,modelJoin} from '../src/learn/data/pandas-join-model.js';
import {coordinateModel,histogramModel,intervalModel} from '../src/learn/data/plotting-foundations-model.js';
import {stagingTrace,branchTrace,remoteTrace,conflictTrace} from '../src/learn/data/git-foundations-model.js';

const root=path.resolve('.'),dir=path.join(root,'scratch/next-three-review');
fs.mkdirSync(dir,{recursive:true});
const fixture={examples:{pandas:{...pandasExamples,...pandasNewExamples},plotting:{...plottingExamples,...plottingNewExamples}},alignment:[],cleaning:[],grouping:[],joins:[],coordinates:[],histograms:[],intervals:[],staging:[],branches:[],remotes:[],conflicts:[]};
for(const mode of ['labels','positions']) for(const reversed of [false,true]) for(const incomplete of [false,true]) fixture.alignment.push({mode,reversed,incomplete,result:alignmentModel(mode,reversed,incomplete)});
for(const policy of ['known','positive']) fixture.cleaning.push({policy,result:cleaningModel(policy)});
for(const keepMissing of [true,false]) for(const fillZero of [true,false]) fixture.grouping.push({keepMissing,fillZero,result:groupingModel(keepMissing,fillZero)});
for(const how of ['left','inner','outer']) for(const duplicate of [false,true]) for(const validate of [false,true]) fixture.joins.push({how,duplicate,validate,left:joinOrders,right:lookupRows(duplicate),result:modelJoin(how,duplicate,validate)});
for(const view of ['full','zoom','log']) fixture.coordinates.push({view,result:coordinateModel(view)});
for(const layout of ['three','two']) for(const density of [false,true]) fixture.histograms.push({layout,density,result:histogramModel(layout,density)});
for(const kind of ['sd','sem']) for(const repeated of [false,true]) fixture.intervals.push({kind,repeated,result:intervalModel(kind,repeated)});
for(const option of [false,true]) {fixture.staging.push({option,trace:stagingTrace(option)});fixture.branches.push({option,trace:branchTrace(option)});fixture.remotes.push({option,trace:remoteTrace(option)});}
for(const resolution of ['combined','ours']) fixture.conflicts.push({resolution,trace:conflictTrace(resolution)});
fs.writeFileSync(path.join(dir,'native-fixtures.json'),JSON.stringify(fixture));
for(const [id,old,added] of [['pandas-data-wrangling-joins-grouping',pandasExamples,pandasNewExamples],['matplotlib-scientific-plotting',plottingExamples,plottingNewExamples],['git-github-collaborative-version-control',gitExamples,gitNewExamples]]) {
  const source=fs.readFileSync('src/learn/data/topics/'+id+'.jsx','utf8');
  const displayed=await collectLessonExamples('src/learn/data/topics/'+id+'.jsx');
  for(const examples of [old,added]) {
    const references=displayed.filter(reference=>Object.hasOwn(examples,reference.key)).map(reference=>reference.key);
    assert.deepEqual([...new Set(references)].sort(),Object.keys(examples).sort(),id+' complete displayed-example coverage');
  }
}
const python=path.resolve(process.env.LESSON_PYTHON||'scratch/lesson-tools/Scripts/python.exe');
const run=spawnSync(python,['scripts/verify-next-three.py',dir],{cwd:root,encoding:'utf8',env:{...process.env,MPLBACKEND:'Agg',MPLCONFIGDIR:path.join(dir,'mpl'),PYTHONIOENCODING:'utf-8',GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:path.join(dir,'empty-gitconfig'),GIT_TERMINAL_PROMPT:'0'},maxBuffer:10*1024*1024});
process.stdout.write(run.stdout||'');process.stderr.write(run.stderr||'');
assert.equal(run.status,0,run.error?.message||'Native checks failed');
const bash=process.env.LESSON_BASH||'C:/Program Files/Git/bin/bash.exe';
const gitDir=path.join(dir,'bash-examples');fs.mkdirSync(gitDir,{recursive:true});
const results=[];
for(const [id,example] of Object.entries({...gitExamples,...gitNewExamples})) {
  const file=path.join(gitDir,id+'.sh');fs.writeFileSync(file,example.code);
  const result=spawnSync(bash,['--noprofile','--norc',file],{cwd:gitDir,encoding:'utf8',timeout:30000,env:{...process.env,TMPDIR:gitDir.replaceAll('\\','/'),GIT_CONFIG_GLOBAL:path.join(dir,'empty-gitconfig'),GIT_CONFIG_NOSYSTEM:'1',GIT_TERMINAL_PROMPT:'0'},maxBuffer:4*1024*1024});
  assert.equal(result.status,0,id+': '+result.stderr+' '+result.error?.message);
  assert.equal(result.stdout.replaceAll('\r\n','\n').trimEnd(),example.output.trimEnd(),id+' exact output');
  results.push({id,passed:true});
}
fs.writeFileSync(path.join(dir,'git-example-results.json'),JSON.stringify({git:spawnSync('git',['--version'],{encoding:'utf8'}).stdout.trim(),bash:spawnSync(bash,['--version'],{encoding:'utf8'}).stdout.split('\n')[0],examples:results},null,2));
console.log('PASS: '+results.length+' displayed Git Bash examples, including regression search and the independent staged-file task.');
