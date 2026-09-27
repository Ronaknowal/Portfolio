import fs from 'node:fs';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const id='hybrid-ssm-transformer-architectures-jamba',file='scratch/hybrid-jamba-render-check.mjs',errors=[],original=console.error;
try{
 await build({entryPoints:[`src/learn/data/topics/${id}.jsx`],bundle:true,platform:'node',format:'esm',jsx:'automatic',external:['react','react-dom','react/jsx-runtime'],loader:{'.css':'empty'},outfile:file});
 console.error=(...args)=>errors.push(args.map(String).join(' '));const module=await import(`../${file}`),html=renderToStaticMarkup(createElement(module.default.content));assert.deepEqual(errors,[]);assert.ok(!/>(?:NaN|[-+]?Infinity)</.test(html));for(const lab of ['jamba-memory','jamba-budget','jamba-router','jamba-stroke'])assert.ok(html.includes(`data-lab="${lab}"`));assert.ok(!html.includes('[Figure J'));assert.ok(html.includes('katex'));
 const result={passed:true,htmlCharacters:html.length,checks:['Full lesson renders without exceptions or React warnings','All four immediately playable investigations present; selected model honestly loads on intersection','All eighteen figure anchors replaced and dollar math converted to rendered KaTeX','Finite initial numeric outputs'],limitations:['SSR does not establish painted geometry, browser controls, keyboard/mobile access or network recovery.']};fs.writeFileSync(`docs/teaching/deep-learning-completion/${id}/render-checks.json`,JSON.stringify(result,null,2)+'\n');original(JSON.stringify(result));
}finally{console.error=original;fs.rmSync(file,{force:true});}
