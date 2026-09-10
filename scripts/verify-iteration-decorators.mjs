import { mkdir,writeFile,readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import { iterationExamples } from "../src/learn/data/iterator-examples.js";
import { decoratorExamples } from "../src/learn/data/decorator-examples.js";
import { cursorTrace } from "../src/learn/data/iterator-models.js";
import { generatorTrace } from "../src/learn/data/iterator-models.js";
import { pipelineTrace } from "../src/learn/data/iterator-models.js";
import { decoratorOrderTrace } from "../src/learn/data/decorator-context-models.js";
import { contextTrace } from "../src/learn/data/decorator-context-models.js";
import { exitStackTrace } from "../src/learn/data/decorator-context-models.js";
const examples={iteration:iterationExamples,decorators:decoratorExamples};
for(const [name,id] of [['iteration','iterators-iterables-generators'],['decorators','decorators-context-managers']]){
  const jsx=await readFile(`src/learn/data/topics/${id}.jsx`,'utf8');
  const refs=(await collectLessonExamples(`src/learn/data/topics/${id}.jsx`)).map(reference=>reference.key);
  assert.deepEqual([...new Set(refs)].sort(),Object.keys(examples[name]).sort(),`Every ${name} fixture is displayed and every displayed fixture is exported`);
}
const cursors=[],generators=[],pipelines=[],orders=[],contexts=[],stacks=[];
for(const shared of [false,true])for(const empty of [false,true])cursors.push({shared,empty,states:cursorTrace(shared,empty)});
for(const action of ['exhaust','close-started','close-created'])generators.push({action,states:generatorTrace(action)});
for(const limit of [1,2])for(const bad of [false,true])pipelines.push({limit,bad,states:pipelineTrace(limit,bad)});
for(const outer of ['cap','double'])for(const value of [3,8,12])orders.push({outer,value,states:decoratorOrderTrace(outer,value)});
for(const path of ['success','body-fails','enter-fails'])for(const suppress of [false,true])contexts.push({path,suppress,states:contextTrace(path,suppress)});
for(const fail of ['A','B','C','none'])stacks.push({fail,states:exitStackTrace(fail)});
await mkdir('scratch/iteration-decorators-review',{recursive:true});
await writeFile('scratch/iteration-decorators-review/fixtures.json',JSON.stringify({examples,cursors,generators,pipelines,orders,contexts,stacks},null,2));
console.log('Exported 25 displayed programs and 27 lab configurations for independent native Python verification.');
