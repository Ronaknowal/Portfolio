// Final production checks for inline lesson controls: calibration states and themed figure actions.
const fs=require('fs'),assert=require('assert/strict'),crypto=require('crypto');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,'')),hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const fonts=read(process.env.LEARNING_FONT_FIXTURES || 'scratch/kmeans-revision-review/fonts/manifest.json');
const base=process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4193';
const receipt='docs/teaching/evidence/control-affordance-independent-browser.json';
const writesEvidence=!process.argv.includes('--no-evidence');
const evidence={startedAt:new Date().toISOString(),base,records:[],screenshots:[],errors:[],passed:false};
if(writesEvidence)fs.writeFileSync(receipt,JSON.stringify(evidence,null,2)+'\n');
(async()=>{const browser=await chromium.launch({channel:'msedge',headless:true});const {records,screenshots,errors}=evidence;try{
 const context=await browser.newContext({viewport:{width:1366,height:1000}});
 await context.route('https://fonts.googleapis.com/**',r=>r.fulfill({path:fonts.stylesheet,contentType:'text/css',headers:{'access-control-allow-origin':'*'}}));
 await context.route('https://fonts.gstatic.com/**',r=>fonts.files[r.request().url()]?r.fulfill({path:fonts.files[r.request().url()],contentType:'font/ttf',headers:{'access-control-allow-origin':'*'}}):r.abort());
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 const open=async id=>{await page.goto(`${base}/learn/path/full-curriculum/${id}?module=classical-ml`,{waitUntil:'domcontentloaded'});await page.locator('.reader-article h2,.reader-article h3').first().waitFor();await page.evaluate(()=>document.fonts.ready);};
 await open('calibration-conformal-prediction');const figure=page.locator('[data-calibration-figure="conditioning-fork"]');const ranges=figure.locator('input[type=range]');assert.equal(await ranges.count(),4);
 const set=async values=>{for(let i=0;i<4;i++)await ranges.nth(i).fill(String(values[i]));};
 const cases=[
  {name:'initial complementary confidence conceals error',values:[.2,.3,.8,.9],points:[2,1],desc:['0.2 against0.3','0.8 against0.8'],text:/Pooling by confidence hides errors/},
  {name:'two distinct perfectly calibrated forecasts',values:[.3,.3,.9,.9],points:[2,2],desc:['0.3 against0.3','0.7 against0.7'],text:/Each distinct class-1 forecast also matches/},
  {name:'equal forecasts pool exactly at tie boundary',values:[.5,.2,.5,.8],points:[1,1],desc:['0.5 against0.5','0.5 against0.5'],text:/Each distinct class-1 forecast also matches/},
  {name:'deterministic endpoints',values:[0,0,1,1],points:[2,1],desc:['0 against0','1 against1'],text:/Each distinct class-1 forecast also matches/},
  {name:'matching means do not establish conditional calibration',values:[.6,.7,.8,.7],points:[2,2],desc:['0.6 against0.7','0.6 against0.7'],text:/At least one confidence-conditioned point is off the diagonal; agreement of overall averages alone is insufficient/},
  {name:'decimal-complement pooling avoids binary roundoff split',values:[.45,.3,.55,.9],points:[2,1],desc:['0.45 against0.3','0.55 against0.8'],text:/Both readings now expose a discrepancy/},
 ];
 for(const item of cases){await set(item.values);const plots=figure.locator('svg.cal-plot');assert.equal(await plots.count(),2);for(let i=0;i<2;i++){assert.equal(await plots.nth(i).locator('circle').count(),item.points[i]);assert((await plots.nth(i).locator('desc').textContent()).replaceAll(' ','').includes(item.desc[i].replaceAll(' ','')),item.name);}assert.match(await figure.innerText(),item.text);records.push({check:item.name,values:item.values,points:item.points});}
 await figure.getByRole('button',{name:'Match forecasts to the group rates'}).click();assert.equal(await ranges.nth(0).inputValue(),'.3'.replace(/^\./,'0.'));assert.equal(await ranges.nth(2).inputValue(),'0.9');
 await figure.getByRole('button',{name:/^Reset:/}).click();assert.equal(await ranges.nth(0).inputValue(),'0.2');await ranges.nth(0).focus();await page.keyboard.press('ArrowRight');assert.equal(await ranges.nth(0).inputValue(),'0.21');assert.match(await figure.innerText(),/0\.21/);records.push({check:'Calibration buttons reset/match and keyboard ArrowRight update values and visible output'});
 for(const width of[390,320]){await page.setViewportSize({width,height:1000});await figure.getByRole('button',{name:/^Reset:/}).click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);const boxes=await ranges.evaluateAll(nodes=>nodes.map(n=>({width:n.getBoundingClientRect().width,height:n.getBoundingClientRect().height})));assert(boxes.every(b=>b.width>100&&b.height>=44));records.push({check:'Calibration responsive controls',width,boxes});if(width===390){const file='docs/teaching/evidence/screenshots/control-affordance-calibration-390.png';await figure.scrollIntoViewIfNeeded();const navStyle=await page.addStyleTag({content:'.learn-nav {visibility:hidden !important}'});try{await figure.screenshot({path:file});}finally{await navStyle.evaluate(node=>node.remove());}screenshots.push({path:file,sha256:hash(file)});}}
 for(const [id,selector,buttonName] of[
  ['pca-dimensionality-reduction','.pca-figure .pca-buttons',/Overlay the cultivar labels/],
  ['bias-variance-tradeoff-learning-curves','.bv-figure .bv-buttons',/Hide the individual fold values/],
  ['end-to-end-supervised-learning-error-analysis','.ete-investigation',/Use current values as comparison baseline/],
 ]){for(const width of[1366,390,320]){await page.setViewportSize({width,height:1000});await open(id);const button=page.locator(selector).getByRole('button',{name:buttonName}).first();await button.scrollIntoViewIfNeeded();const before=await button.evaluate(n=>({bg:getComputedStyle(n).backgroundColor,fg:getComputedStyle(n).color,height:n.getBoundingClientRect().height,left:n.getBoundingClientRect().left,right:n.getBoundingClientRect().right}));assert.notEqual(before.bg,'rgb(107, 107, 107)');assert.equal(before.fg,'rgb(226, 181, 90)');assert(before.height>=44&&before.left>=0&&before.right<=width+1);await button.click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);records.push({check:'Previously native-grey control remains usable',id,width,style:before});}}
 assert.deepEqual(errors,[]);Object.assign(evidence,{capturedAt:new Date().toISOString(),sourceHashes:Object.fromEntries(['scripts/verify-inline-lesson-controls.cjs','src/index.css','src/learn/components/topic-content.css','src/learn/components/lesson-labs/CalibrationFigures.jsx','src/learn/components/lesson-labs/calibration-labs.css','src/learn/data/calibration-models.js'].map(p=>[p,hash(p)])),manifestHash:hash('dist/.vite/manifest.json'),records,screenshots,errors,passed:true});console.log(JSON.stringify({passed:true,groups:records.length,screenshots},null,2));
 }catch(error){evidence.failure=error.message;throw error;}finally{await browser.close();if(writesEvidence)fs.writeFileSync(receipt,JSON.stringify(evidence,null,2)+'\n');}})().catch(e=>{console.error(e);process.exitCode=1;});
