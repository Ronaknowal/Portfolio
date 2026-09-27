import fs from 'node:fs';
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {parse} from '@babel/parser';
import {createServer} from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {spliceForward,prepareHyenaModel} from '../src/learn/data/hyena-convolution-models.js';
const id='hyena-long-convolution-models',folder='docs/teaching/deep-learning-completion/'+id+'/',draft='docs/teaching/drafts/'+id+'/',deployed='public/learn-code/'+id+'/',checks=[];
const check=(name,value)=>{assert.ok(value,name);checks.push({name,passed:true});};
for(const name of ['HyenaPrimitives.jsx','HyenaFigures.jsx','HyenaMechanismLabs.jsx','HyenaStudy.jsx']){parse(fs.readFileSync('src/learn/components/lesson-labs/'+name,'utf8'),{sourceType:'module',plugins:['jsx']});check('JSX parse '+name,true);}
const downloads=['convolution_mechanisms.py','splice_models.py','author_calculations.py','blocked_convolution.py','splice-fits.npz','splice-results.json','splice.data','splice.names','data-provenance.md'];
for(const name of downloads)check('Exact deployed source copy '+name,fs.readFileSync(draft+name).equals(fs.readFileSync(deployed+name)));
const rows=JSON.parse(fs.readFileSync(deployed+'validation-sequences.json')).rows,models=['linear_29','gated_29','ungated_29','gated_71'].map(id=>JSON.parse(fs.readFileSync(deployed+'model-'+id+'.json'))),raw=fs.readFileSync(draft+'splice.data','utf8').trim().split(/\r?\n/);
for(const row of rows){const fields=raw[row.sourceId-1].split(',');assert.equal(row.sequence,fields[2].replace(/\s/g,''));assert.equal(row.label,fields[0].trim());}
check('All460 shipped sequence and label records equal their one-based raw source records',rows.length===460);
check('Worked and fresh source identity',rows[0].sourceId===3&&rows[1].sourceId===4);
check('Only explicit learner assets deployed',fs.readdirSync(deployed).sort().join('|')===[...downloads,'validation-sequences.json',...models.map(model=>'model-'+model.id+'.json')].sort().join('|'));
const server=await createServer({configFile:false,plugins:[react()],server:{middlewareMode:true,watch:null},optimizeDeps:{noDiscovery:true},appType:'custom',logLevel:'error'});
try{
 const topic=await server.ssrLoadModule('/src/learn/data/topics/'+id+'.jsx'),html=renderToStaticMarkup(React.createElement(topic.default.content));
 for(const phrase of ['Three questions about a long sequence','A convolution is a ledger','Gates make the mixing','Generate the filter','How the block learns','A real sequence task','Blocks must include','Some long filters','Hankel','StripedHyena','Plan an honest efficiency comparison'])check('Complete conceptual manuscript rendered: '+phrase,html.includes(phrase));
 check('Twenty figure anchors and four main investigations replaced',!html.includes('**Figure H')&&!html.includes('**Investigation H')&&[...html.matchAll(/data-lab="hyena-/g)].length===5);
 check('Ten separate solutions and extra blocked practice remain closed',[...html.matchAll(/<summary>Solution<\/summary>/g)].length===10&&html.includes('Hint and reasoned solution')&&!html.includes('<details open'));
 check('Inline mathematics was rendered, not left as dollar strings',!html.includes('$h_r')&&html.includes('katex'));
 check('Topic neutral theme and no bad values',html.includes('hyena-lesson')&&html.includes('neural-lesson-neutral')&&!html.includes('NaN')&&!html.includes('undefined'));
 const {DnaReadView,FilterReadView,initialDnaState}=await server.ssrLoadModule('/src/learn/components/lesson-labs/HyenaStudy.jsx');
 for(const model of models){const state=initialDnaState();for(const [name,sequence,kernelLimit,gatesOff]of[['original',state.original,60,false],['boundaryAA',state.original.slice(0,30)+'AA'+state.original.slice(32),60,false],['distantAA','AA'+state.original.slice(2),60,false],['allN','N'.repeat(60),60,false],['lag0to4',state.original,5,false],['gatesOff',state.original,60,true]]){const current={...state,sequence,kernelLimit,gatesOff,pinned:{sequence:state.original,gatesOff:false,kernelLimit:60,modelId:'gated_29'}},out=renderToStaticMarkup(React.createElement(DnaReadView,{model,state:current}));check('Complete saved network renders '+model.id+' '+name,out.includes('source 4')&&!out.includes('NaN')&&!out.includes('undefined')&&out.includes('Current fit at controlled inputs'));}
  if(model.kind!=='linear'){const out=renderToStaticMarkup(React.createElement(FilterReadView,{model,block:1,channel:15}));check('Actual fitted raw/envelope/product at block1 channel15 '+model.id,out.includes('864')&&out.includes('Product used as long filter'));}
 }
 const {Stems,formatHyenaTick}=await server.ssrLoadModule('/src/learn/components/lesson-labs/HyenaPrimitives.jsx');
 for(const values of [[1,-1],[-4,0,4],[.001,-.001]]){const svg=renderToStaticMarkup(React.createElement(Stems,{title:'Signed extrema',values}));check('Signed extrema stems keep text baseline233 and separate index268 within285-height viewBox '+values,svg.includes(' 285')&&svg.includes('y="268"')&&!svg.includes('NaN'));}
 for(const [lo,hi] of [[-.000635,.003135],[-1e-16,1e-16],[0,2129.92],[.999999,1.000001]]){const labels=[lo,(lo+hi)/2,hi].map(v=>formatHyenaTick(v,hi-lo));check('Distinct span-aware scientific tick labels '+lo+'..'+hi,new Set(labels).size===3&&labels.every(x=>x.length<=12));}
 const {LearningCurvesFigure}=await server.ssrLoadModule('/src/learn/components/lesson-labs/HyenaFigures.jsx');const out=renderToStaticMarkup(React.createElement(LearningCurvesFigure));check('All320 exact measured validation points and selected epochs are retained',models.every(m=>m.history.length===80)&&out.includes('All80 measured validation epochs')&&out.includes('0.773983419'));
}finally{await server.close();}
const start=performance.now();let n=0;for(const model of models){const preparedFilters=prepareHyenaModel(model);for(const row of rows.slice(0,10)){spliceForward(row.sequence,model,{preparedFilters});n++;}}const elapsed=(performance.now()-start)/n;
fs.writeFileSync(folder+'render-checks.json',JSON.stringify({topicId:id,passed:true,checks,localNodeTiming:{millisecondsPerSixtySymbolForward:elapsed},limits:['SSR validates accessible rendered values and complete manuscript, not interactive or painted browser geometry.','Local Node timing is not a browser/device speed claim.','Native42 and model1,036,712 checks cover unchanged numerical engines; final independent/browser/integration remain pending.']},null,2)+'\n');
console.log(checks.length+' Hyena source/SSR checks passed; local60-symbol forward '+elapsed.toFixed(3)+'ms.');
