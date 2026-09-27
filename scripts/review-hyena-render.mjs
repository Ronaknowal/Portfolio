import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createServer} from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {modalScan} from '../src/learn/data/hyena-convolution-models.js';

const folder='docs/teaching/deep-learning-completion/hyena-long-convolution-models/';
const checks=[];
const server=await createServer({configFile:false,plugins:[react()],server:{middlewareMode:true,watch:null},optimizeDeps:{noDiscovery:true},appType:'custom',logLevel:'error'});
const check=(name,condition)=>{assert.ok(condition,name);checks.push({name,passed:true});};
try {
  const {Plot,Probabilities}=await server.ssrLoadModule('/src/learn/components/lesson-labs/HyenaPrimitives.jsx');
  const base=modalScan([2,0,-1,3,1,-.5],[1,0],[.5,-.25]).map(row=>row.output);
  for(const scale of [1,.001,1e-7,-.001]) {
    const output=renderToStaticMarkup(React.createElement(Plot,{title:'Independent tiny-value probe',xLabel:'position',yLabel:'output',series:[{label:'Current modal output',values:base.map((v,t)=>[t,v*scale])}]}));
    const ticks=[...output.matchAll(/<text x="95"[^>]*>([^<]+)<\/text>/g)].map(match=>match[1]);
    check('Distinct signed vertical ticks at scale '+scale,ticks.length===3&&new Set(ticks).size===3);
    check('Nominal figure stays keyboard-scrollable at scale '+scale,output.includes('tabindex="0"')&&output.includes('min-width:460px'));
  }
  const out=renderToStaticMarkup(React.createElement(Probabilities,{title:'Independent fixed probability domain',current:[.1,.2,.7],reference:[1/3,1/3,1/3]}));
  const widths=[...out.matchAll(/<rect[^>]*width="([^"]+)"/g)].map(match=>Number(match[1]));
  check('All class bars preserve the same unit probability scale',widths.length===3&&widths.every((width,i)=>Math.abs(width-[31,62,217][i])<1e-12));
  check('Probability axis remains explicitly zero to one',out.includes('>0</text>')&&out.includes('>0.5</text>')&&out.includes('>1</text>'));
} finally {await server.close();}
fs.writeFileSync(folder+'independent-render-checks.json',JSON.stringify({passed:true,checks,limits:['Scoped SSR checks inspect numeric axis and probability geometry only. Root owns interactive and painted browser review.']},null,2)+'\n');
console.log(checks.length+' independent Hyena display checks passed.');
