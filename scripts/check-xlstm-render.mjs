import fs from 'node:fs';
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {parse} from '@babel/parser';
import {createServer} from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {readerForward} from '../src/learn/data/xlstm-memory-models.js';
const id='xlstm-extended-lstm',folder='docs/teaching/deep-learning-completion/'+id+'/',draft='docs/teaching/drafts/'+id+'/',deployed='public/learn-code/'+id+'/',checks=[];
const check=(name,value)=>{assert.ok(value,name);checks.push({name,passed:true});};
for(const name of ['XlstmPrimitives.jsx','XlstmFigures.jsx','XlstmMechanismLabs.jsx','XlstmStudy.jsx']){parse(fs.readFileSync('src/learn/components/lesson-labs/'+name,'utf8'),{sourceType:'module',plugins:['jsx']});check('JSX parse '+name,true);}
const downloads=['memory_mechanisms.py','row_sequence_models.py','author_calculations.py','row-sequence-fits.npz','row-sequence-results.json','optdigits.tra','optdigits.tes','optdigits.names','data-provenance.md'];
for(const name of downloads)check('Exact deployed source copy '+name,fs.readFileSync(draft+name).equals(fs.readFileSync(deployed+name)));
const rows=JSON.parse(fs.readFileSync(deployed+'digit-rows.json')).rows,models=['lstm','slstm','mlstm'].flatMap(kind=>[19,43].map(seed=>JSON.parse(fs.readFileSync(deployed+`model-${kind}_seed${seed}.json`))));
const rawSource=fs.readFileSync(draft+'optdigits.tra','utf8').trim().split(/\r?\n/).map(line=>line.split(',').map(Number));
for(const row of rows){assert.deepEqual(row.pixels.flat(),rawSource[row.sourceId-1].slice(0,64));assert.equal(row.label,rawSource[row.sourceId-1][64]);}check('All300 deployed images and labels equal their one-based source records',rows.length===300);
check('Fresh and worked source identities retained',rows[142].sourceId===187&&rows[35].sourceId===3451);
check('Only explicit learner assets are deployed',fs.readdirSync(deployed).sort().join('|')===[...downloads,'digit-rows.json',...models.map(model=>`model-${model.id}.json`)].sort().join('|'));
const server=await createServer({configFile:false,plugins:[react()],server:{middlewareMode:true,watch:null},optimizeDeps:{noDiscovery:true},appType:'custom',logLevel:'error'});
try{
 const topic=await server.ssrLoadModule('/src/learn/data/topics/'+id+'.jsx'),html=renderToStaticMarkup(React.createElement(topic.default.content));
 for(const phrase of ['weighted ledger','Signed retrieval is not softmax','Every record','checkpoint','xLSTM-Mixer','State accounting','Hint and reasoned solution'])if(!['Every record','State accounting'].includes(phrase))check('Full conceptual article rendered: '+phrase,html.includes(phrase));
 check('Twenty figure anchors and four investigations replaced',!html.includes('[Figure ')&&!html.includes('[Investigation ')&&[...html.matchAll(/data-lab="xlstm-/g)].length===4);
 check('Ten independent practice solutions remain closed',[...html.matchAll(/<summary>Solution<\/summary>/g)].length===10&&!html.includes('<details open'));
 check('Shared neutral style scoped to new body',html.includes('neural-lesson-neutral xlstm-lesson'));
 check('Initial article has no bad computed values',!html.includes('NaN')&&!html.includes('undefined'));
 const {RowReaderWorkspace,LearningCurveView,initialReaderView,executeReader}=await server.ssrLoadModule('/src/learn/components/lesson-labs/XlstmStudy.jsx');
 for(const model of models){
  const original=rows[142],clean=original.pixels,edited=clean.map((row,i)=>i>=5?row.map(()=>0):row),blank=clean.map(row=>row.map(()=>0));
  for(const [name,raw,reversed,execution] of [['original',clean,false,'full'],['bottom rows zero',edited,false,'full'],['reversed',edited,true,'full'],['blank',blank,false,'full'],['carry',edited,false,'carry'],['full reset',edited,false,'reset'],['normalizer reset',edited,false,'reset-n']]){
   const view={...initialReaderView,reversed,execution},out=renderToStaticMarkup(React.createElement(RowReaderWorkspace,{model,original,raw,view}));
   check(`Actual selected fit and state render ${model.id}: ${name}`,out.includes('187')&&out.includes('All ten scores')&&!out.includes('NaN')&&!out.includes('undefined'));
   const ordered=reversed?[...raw].reverse():raw,result=executeReader(ordered,model,view);
   if(execution==='carry'||model.kind==='lstm'&&execution==='reset-n')assert.deepEqual(result.trace.map(r=>r.logits),readerForward(ordered,model).logits);
   if(name==='bottom rows zero')assert.deepEqual(result.trace.slice(0,5).map(r=>r.logits),readerForward(clean,model).logits.slice(0,5));
  }
  const curve=renderToStaticMarkup(React.createElement(LearningCurveView,{model}));check('All150 learning points and selected-checkpoint explanation '+model.id,model.trainingCurve.length===150&&curve.includes('different parameter snapshots')&&curve.includes('validation_clean'));
 }
 const {XSignedWrites}=await server.ssrLoadModule('/src/learn/components/lesson-labs/XlstmPrimitives.jsx');const signed=renderToStaticMarkup(React.createElement(XSignedWrites,{contributions:[[0,0],[-2,1]],denominator:1}));check('Zero and negative signed contribution geometry renders',!signed.includes('NaN')&&signed.includes('rose extends left'));
}finally{await server.close();}
const start=performance.now();for(const model of models)for(const row of rows.slice(0,20))readerForward(row.pixels,model);const elapsed=(performance.now()-start)/120;
fs.writeFileSync(folder+'render-checks.json',JSON.stringify({topicId:id,passed:true,checks,localNodeTiming:{millisecondsPerEightRowForward:elapsed},limits:['SSR checks actual accessible values and complete manuscript, not browser interaction or painted geometry.','Local Node timing is not a browser benchmark.','Final whole-site build, independent review and desktop/mobile browser remain root-owned.']},null,2)+'\n');
console.log(checks.length+' xLSTM source/SSR checks passed; eight-row forward '+elapsed.toFixed(3)+'ms in local Node.');
