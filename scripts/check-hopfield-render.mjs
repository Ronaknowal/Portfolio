import fs from 'node:fs';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { parse } from '@babel/parser';
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { digitRead,prepareDigitBank,energyContours,occludeDigit } from '../src/learn/data/hopfield-memory-models.js';
const id='modern-hopfield-networks',folder='docs/teaching/deep-learning-completion/'+id+'/',draft='docs/teaching/drafts/'+id+'/',deployed='public/learn-code/'+id+'/',checks=[];
const check=(name,condition)=>{assert.ok(condition,name);checks.push({name,passed:true});};
for(const name of ['HopfieldPrimitives.jsx','HopfieldFigures.jsx','HopfieldLabs.jsx','HopfieldDigitLab.jsx']){parse(fs.readFileSync('src/learn/components/lesson-labs/'+name,'utf8'),{sourceType:'module',plugins:['jsx']});check('JSX parse '+name,true);}
for(const name of ['associative_memory.py','digit_memory.py','digit-memory-fits.npz','digit-results.json','mechanism-results.json','optdigits.tra','optdigits.tes','optdigits.names','data-provenance.md'])check('Deployed exact copy '+name,fs.readFileSync(draft+name).equals(fs.readFileSync(deployed+name)));
const bank=JSON.parse(fs.readFileSync(deployed+'digit-bank.json')),models=['fixed','seed17','seed41'].map(id=>JSON.parse(fs.readFileSync(deployed+'model-'+id+'.json')));
const sourceRows=fs.readFileSync(draft+'optdigits.tra','utf8').trim().split(/\r?\n/).map(row=>row.split(',').map(Number));
for(const role of ['memory','validation'])for(const row of bank[role]){assert.deepEqual(row.pixels,sourceRows[row.sourceId-1].slice(0,64));assert.equal(row.label,sourceRows[row.sourceId-1][64]);}check('Every deployed image and label equal the one-based raw source row',true);
const server=await createServer({configFile:false,plugins:[react()],server:{middlewareMode:true,watch:null},optimizeDeps:{noDiscovery:true},appType:'custom',logLevel:'error'});
try{const topic=await server.ssrLoadModule('/src/learn/data/topics/'+id+'.jsx'),html=renderToStaticMarkup(React.createElement(topic.default.content));
 for(const phrase of ['A memory is more than a label','Every other feature casts a signed vote','Actual energy contours','Shape','Higher-order','fixed training samples','Hint and reasoned solution'])if(phrase!=='Shape')check('Full article render '+phrase,html.includes(phrase));
 check('All ten independent practice solutions closed',[...html.matchAll(/<summary>Solution<\/summary>/g)].length===10);
 check('All four distinct investigations visible',[...html.matchAll(/data-lab="hopfield-/g)].length===4);
 check('Prepared placeholder anchors are absent',!html.includes('[Figure ')&&!html.includes('[Investigation '));
 check('Neutral scoped body applied',html.includes('neural-lesson-neutral hopfield-lesson'));
 const {DigitReadView}=await server.ssrLoadModule('/src/learn/components/lesson-labs/HopfieldDigitLab.jsx');
 const {Bars}=await server.ssrLoadModule('/src/learn/components/lesson-labs/HopfieldPrimitives.jsx');const probabilityBars=renderToStaticMarkup(React.createElement(Bars,{title:'blank class masses',labels:Array.from({length:10},(_,i)=>'class'+i),values:Array(10).fill(.1),fixedMaximum:1}));check('Blank0.1 masses use ten27px bars on fixed270px unit scale',[...probabilityBars.matchAll(/width="27"/g)].length===10);check('Probability axis states fixed zero-to-one scale',probabilityBars.includes('Fixed class-mass scale from0 to1'));
 for(const model of models)for(const raw of [bank.validation[31].pixels,occludeDigit(bank.validation[31].pixels),Array(64).fill(0)]){const result=renderToStaticMarkup(React.createElement(DigitReadView,{bank,model,raw,original:bank.validation[31]}));check('Actual digit render '+model.id+' '+raw.slice(0,8).join(''),result.includes('3748')&&result.includes('All200 memories contribute')&&!result.includes('NaN')&&!result.includes('undefined'));if(raw.every(x=>x===0))check('Blank cue explicitly shows all-class tie '+model.id,result.includes('All ten classes tie'));}
}finally{await server.close();}
const model=models[1],prepared=prepareDigitBank(bank,model),start=performance.now();for(let i=0;i<100;i++)digitRead(bank.validation[i].pixels,bank,model,prepared);const readMs=(performance.now()-start)/100,gridStart=performance.now();energyContours([[1,0],[-1,0],[.5,1]],2);const gridMs=performance.now()-gridStart;
fs.writeFileSync(folder+'render-checks.json',JSON.stringify({topicId:id,passed:true,checks,localNodeTiming:{millisecondsPerDigitRead:readMs,millisecondsPer61By41Grid:gridMs},limits:['SSR verifies computed accessible content, not browser interaction/paint.','Timing is this local Node process, not claimed browser latency.','Root owns final production build and desktop/mobile browser.']},null,2)+'\n');console.log(checks.length+' source/deploy/render checks passed; digit read '+readMs.toFixed(3)+'ms, grid '+gridMs.toFixed(3)+'ms.');
