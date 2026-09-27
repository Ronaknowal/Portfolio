import fs from 'node:fs';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { parse } from '@babel/parser';
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { frozenForward, linearPenalty } from '../src/learn/data/spectral-regularization-models.js';
const id='spectral-normalization-gradient-penalty',folder='docs/teaching/deep-learning-completion/'+id+'/',draft='docs/teaching/drafts/'+id+'/',deployed='public/learn-code/'+id+'/';
const checks=[];const check=(name,condition)=>{assert.ok(condition,name);checks.push({name,passed:true});};
const data=JSON.parse(fs.readFileSync(draft+'calculated-inputs.json')),dataset=JSON.parse(fs.readFileSync(deployed+'dataset.json'));
for(const file of ['critic-regularization-study.py','sensitivity-calculations.py','calculated-inputs.json','sensitivity-results.json','digits-400.csv','data-provenance.md'])check('Deployed byte identity '+file,fs.readFileSync(draft+file).equals(fs.readFileSync(deployed+file)));
for(const fit of data.fits){assert.deepEqual(JSON.parse(fs.readFileSync(deployed+'model-'+fit.method+'-'+fit.seed+'.json')),fit);checks.push({name:'Lossless selected fit '+fit.method+'-'+fit.seed,passed:true});}
assert.deepEqual(dataset.rows.map(row=>row.profile),data.measurements);check('All400 profiles in deployed data, 241/79/80 roles',dataset.rows.length===400&&['fit','development','assessment'].map(role=>dataset.rows.filter(row=>row.role===role).length).join('/')==='241/79/80');
check('Zero-strength penalty is constant even at zero weight',linearPenalty([0,0],0,.1,'target-one').gradient.every(v=>v===0));
for(const name of ['SpectralPrimitives.jsx','SpectralMechanismFigures.jsx','SpectralMechanismLabs.jsx','SpectralMeasuredLab.jsx']){parse(fs.readFileSync('src/learn/components/lesson-labs/'+name,'utf8'),{sourceType:'module',plugins:['jsx']});checks.push({name:'JSX parse '+name,passed:true});}
const server=await createServer({configFile:false,plugins:[react()],server:{middlewareMode:true,watch:null},optimizeDeps:{noDiscovery:true},appType:'custom',logLevel:'error'});
try{
 const topic=await server.ssrLoadModule('/src/learn/data/topics/'+id+'.jsx');
 const html=renderToStaticMarkup(React.createElement(topic.default.content));
 for(const phrase of ['A critic that gives directions','Differentiate first','Cross-correlation kernel','cached u and v','GroupSort input a','Linear ODE rate','Read the complete executed','Hint and reasoned solution'])check('Full article render '+phrase,html.includes(phrase));
 check('All nine independent practice solutions are retained closed', [...html.matchAll(/<summary>Solution<\/summary>/g)].length===9);
 const {StudyWorkspace}=await server.ssrLoadModule('/src/learn/components/lesson-labs/SpectralMeasuredLab.jsx');
 for(const [a,b]of[[6,0],[5,5]]){
  const result=renderToStaticMarkup(React.createElement(StudyWorkspace,{dataset,fit:data.fits[a],comparison:data.fits[b]}));
  for(const phrase of ['Actual measured dataset','Every declared method and seed','matrix-product bound','Joint maximum difference','Final frozen critic'])check('Real workspace '+a+'/'+b+' '+phrase,result.includes(phrase));
  check('Real workspace finite text '+a+'/'+b,!result.includes('NaN')&&!result.includes('undefined'));
 }
 const view={imageIndex:137,role:'assessment',checkpoint:0,point:[.19,.83],slice:31,z:[1.17,-.33],pinned:[-.22,.87]};
 const {vector}=await server.ssrLoadModule('/src/learn/components/lesson-labs/SpectralPrimitives.jsx');
 for(const index of [0,4,8]){
  const result=renderToStaticMarkup(React.createElement(StudyWorkspace,{dataset,fit:data.fits[index],comparison:data.fits[1],view,onViewChange:()=>{}}));
  check('Controlled image/checkpoint survive model '+index,result.includes('source '+dataset.rows[137].sourceId)&&result.includes('actual generated checkpoint 1'));
  check('Controlled latent and pinned inputs survive model '+index,result.includes('Pinned latent '+vector(view.pinned)+'; current latent '+vector(view.z)));
  check('Controlled critic probe survives model '+index,result.includes(vector(view.point)));
  check('Current model recomputes output at retained latent '+index,result.includes(vector(frozenForward(view.z,data.fits[index].generator_layers).value)));
 }
}finally{await server.close();}
const timed=[];
for(const fit of data.fits){const start=performance.now();for(let i=0;i<100;i++){frozenForward([i/100,-.4],fit.generator_layers);frozenForward([i/100,.4],fit.critic_layers,'critic');}timed.push({fit:fit.method+'-'+fit.seed,millisecondsPerGeneratorAndCritic:(performance.now()-start)/100});}
fs.writeFileSync(folder+'render-checks.json',JSON.stringify({topicId:id,passed:true,checks,boundedNativeJsTiming:timed,limits:['Timing is this local Node runtime, not browser latency or training speed.','SSR checks accessible text and calculations; canvas paint, mobile geometry and live controls remain the browser pass.']},null,2)+'\n');console.log(checks.length+' deploy, render and boundary checks passed; all nine forwards timed.');
