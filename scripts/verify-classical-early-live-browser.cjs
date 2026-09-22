const fs=require('node:fs');const crypto=require('node:crypto');const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const families=['LinearLogistic','DecisionTree','Knn','GradientBoostedTree','SupportVectorMachine','NaiveBayes','EnsembleMethods','Recommender','Multioutput','Survival','Pca','ClusteringEvaluation','Dbscan','AnomalyDetection','Gmm','Manifold','Ica','Nmf','Scaling','Validation','Regularization'];
const topicDir='src/learn/data/topics';const topics=fs.readdirSync(topicDir).filter(f=>f.endsWith('.jsx')&&families.some(x=>fs.readFileSync(topicDir+'/'+f,'utf8').includes('/'+x+'Labs'))).map(f=>f.slice(0,-4));
const base=process.env.LEARNING_BASE_URL||'http://127.0.0.1:4190';const receipt='docs/teaching/evidence/live-exploration-classical-early-browser.json';const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const selector='[data-live-exploration],.lesson-investigation';const records=[];const sourceFiles=[...topics.map(id=>`${topicDir}/${id}.jsx`),...fs.readdirSync('src/learn/components/lesson-labs').filter(f=>f==='AnomalyTemperatureLab.jsx'||families.some(x=>f===x+'Labs.jsx'||f===x+'Shared.jsx')).map(f=>'src/learn/components/lesson-labs/'+f)];
async function semanticChecks(page,id){
 const checks=[];
 if(id==='pca-dimensionality-reduction'){
  const lab=page.locator('.pca-investigation').first();await lab.getByLabel('Proposed angle',{exact:false}).fill('0');
  assert.match(await lab.locator('.pca-readout').first().innerText(),/retained 10 \+ residual 10/);
  await lab.getByLabel('Proposed angle',{exact:false}).fill('45');
  assert.match(await lab.locator('.pca-readout').first().innerText(),/retained 18 \+ residual 2/);checks.push('Projection angle moves actual conserved retained/residual sums 10/10→18/2');
 }
 if(id==='gaussian-mixture-models-gmm-em-algorithm'){
  const lab=page.locator('.gm-investigation').first();await lab.getByRole('button',{name:'Identical components',exact:true}).click();await lab.getByLabel('Weight of A',{exact:true}).fill('0.75');
  assert.match(await lab.locator('.gm-bars').getAttribute('aria-label'),/component A 0\.75, component B 0\.25/);checks.push('Identical-component responsibilities equal edited mixing weights');
  const em=page.locator('.gm-investigation').nth(1);const likelihood=async()=>Number((await em.locator('.gm-readout').innerText()).match(/log-likelihood (-?\d+(?:\.\d+)?)/)[1]);const before=await likelihood();await em.getByRole('button',{name:'Compute E-step',exact:true}).click();assert.equal(await likelihood(),before);await em.getByRole('button',{name:'Apply M-step',exact:true}).click();assert(await likelihood()>=before);await em.getByLabel('Initial Left mean',{exact:true}).fill('-1.5');assert.match(await em.locator('.gm-readout').innerText(),/Iteration 0, parameters applied/);checks.push('E-step preserves likelihood; M-step increases it; setup edit restarts the actual trace');
 }
 if(id==='independent-component-analysis-ica'){
  const scale=page.locator('[data-ica-lab="scale"]');const before=await scale.locator('table').innerText();await scale.getByRole('button',{name:'Set c = 0',exact:true}).click();assert.match(await scale.locator('[role="status"]').innerText(),/c = 0 is invalid/);assert.equal(await scale.locator('table').innerText(),before);await scale.getByRole('button',{name:'Reset',exact:true}).click();assert.equal(await scale.locator('.ic-error').count(),0);checks.push('Invalid zero compensation retains explicitly labelled last valid table; Reset clears error');
 }
 if(id==='non-negative-matrix-factorization-nmf'){
  const lab=page.locator('.nm-investigation').first();await lab.getByLabel('Amount a of component 1',{exact:true}).fill('3');assert.match(await lab.locator('.lesson-live-note').innerText(),/feature 3 is now 4 against 3 in the saved reference/);await lab.getByRole('button',{name:'Save current mixture as reference',exact:true}).click();await lab.getByLabel('Amount a of component 1',{exact:true}).fill('2');assert.match(await lab.locator('.lesson-live-note').innerText(),/feature 3 is now 3 against 4 in the saved reference/);checks.push('NMF edited contributions update immediately and explicit saved reference remains fixed');
 }
 if(id==='clustering-evaluation-validation-silhouette-ari-nmi'){
  const lab=page.locator('[data-live-exploration]').last();const before=await lab.locator('.ce-readout').innerText();await lab.getByText('Change the six probe locations (strictly increasing, within ±40)',{exact:true}).click();await lab.getByLabel('B location',{exact:true}).fill('0');assert.match(await lab.locator('.ce-error').innerText(),/strictly increasing/);assert.equal(await lab.locator('.ce-readout').innerText(),before);await lab.getByRole('button',{name:'Reset',exact:true}).click();checks.push('Invalid coincident probe locations preserve valid fitted centers and show explicit error');
 }
 if(id==='cross-validation-hyperparameter-tuning'){
  const lab=page.locator('[data-cv-lab="folds"]');for(let row=0;row<7;row++)await lab.getByLabel(`validation fold of row ${row}`,{exact:true}).selectOption('0');assert(await lab.locator('[data-cv-error]').count());assert.equal(await lab.locator('.lesson-live-note').count(),0);await lab.getByRole('button',{name:'Reset',exact:true}).click();assert.equal(await lab.locator('[data-cv-error]').count(),0);assert.equal(await lab.locator('.lesson-live-note').count(),1);checks.push('Empty validation fold shows error and clears invalid output; Reset restores calculation');
 }
 return checks;
}
async function main(){
 const reuse=process.env.LIVE_REUSE_DESKTOP==='1';
 const reusePath='docs/teaching/evidence/live-exploration-classical-early-desktop.json';
 const repairedTopic='survival-analysis-cox-regression-kaplan-meier-hazard-models';
 const currentHashes=Object.fromEntries(sourceFiles.map(file=>[file,hash(file)]));
 if(reuse){
  const previous=JSON.parse(fs.readFileSync(reusePath,'utf8'));
  assert.equal(previous.status,'passed-desktop-only');assert.equal(previous.records.length,21);
  for(const[file,digest]of Object.entries(previous.sourceHashes))if(file!==`${topicDir}/${repairedTopic}.jsx`)assert.equal(hash(file),digest,`Cannot reuse changed desktop source: ${file}`);
  records.push(...previous.records.filter(record=>record.id!==repairedTopic));
 }
 fs.writeFileSync(receipt,JSON.stringify({status:'running',checkedAt:new Date().toISOString()},null,2));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1366,height:1000}});
 const fonts=JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json','utf8').replace(/^\uFEFF/,''));
 await context.route(url=>url.href===fonts.stylesheetUrl,r=>r.fulfill({path:fonts.stylesheet,contentType:'text/css'}));for(const [url,file]of Object.entries(fonts.files))await context.route(url,r=>r.fulfill({path:file,contentType:'font/ttf',headers:{'access-control-allow-origin':'*'}}));
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
 for(const id of reuse?[repairedTopic]:topics){
   await page.goto(`${base}/learn/path/full-curriculum/${id}?module=classical-ml`,{waitUntil:'commit'});await page.waitForSelector(selector,{timeout:20000});await page.evaluate(()=>document.fonts.ready);
   const count=await page.locator(selector).count();assert(count>0,id+' labs');
   const liveText=await page.locator(selector).allTextContents();assert(!liveText.some(t=>/record (?:a|your) prediction|commit prediction|check prediction|choose a prediction|your prediction matches|calculated without a recorded prediction/i.test(t)),id+' obsolete learner prediction UI');
   let exercised=0;
   for(let i=0;i<count;i++){
     const lab=page.locator(selector).nth(i);const control=lab.locator('input[type="range"]:visible:not(:disabled),select:visible:not(:disabled),input[type="number"]:visible:not(:disabled)').first();
     if(!await control.count())continue;
     const before=await lab.evaluate(el=>({text:el.innerText,svg:[...el.querySelectorAll('svg')].map(x=>x.outerHTML).join('')}));
     const kind=await control.evaluate(el=>el.tagName==='SELECT'?'select':el.type);
     if(kind==='select'){
       const options=await control.locator('option').evaluateAll(els=>els.filter(e=>!e.disabled).map(e=>e.value));const current=await control.inputValue();const next=options.find(x=>x!==current&&x!=='');if(next===undefined)continue;await control.selectOption(next);
     }else{
       const values=await control.evaluate(el=>({value:Number(el.value),min:el.min===''?-10:Number(el.min),max:el.max===''?10:Number(el.max),step:el.step==='any'||el.step===''?0.1:Number(el.step)}));
       const next=values.value+values.step<=values.max?values.value+values.step:values.value-values.step;if(next<values.min)continue;
       await control.fill(String(Number(next.toPrecision(10))));
     }
     await page.waitForTimeout(30);
     if(id==='survival-analysis-cox-regression-kaplan-meier-hazard-models'&&i===2){
       const nullView=await lab.evaluate(el=>({text:el.innerText,svg:[...el.querySelectorAll('svg')].map(x=>x.outerHTML).join('')}));
       assert.deepEqual(nullView,before,'AFT and PH both reduce to the baseline at multiplier one');
       await lab.getByRole('slider',{name:/^Multiplier/}).fill('2');
     }
     // The learning rate changes a future update, not the unchanged initial
     // parameter point. Advance the actual solver, then check its arithmetic.
     if(id==='linear-logistic-regression'&&i===1){
       await lab.getByRole('button',{name:'Next step',exact:true}).click();
       assert.match(await lab.innerText(),/b=0\.900000, w=1\.800000/,'rate .2 applies the actual first gradient');
     }
     const after=await lab.evaluate(el=>({text:el.innerText,svg:[...el.querySelectorAll('svg')].map(x=>x.outerHTML).join('')}));
     assert.notDeepEqual(after,before,`${id} lab ${i+1}: valid edit did not reach visible text/geometry`);exercised++;
     const reset=lab.getByRole('button',{name:'Reset',exact:true});if(await reset.count())await reset.click();
   }
   assert.equal(errors.length,0,`${id}: ${errors.join('; ')}`);
   const semantic=await semanticChecks(page,id);assert.equal(errors.length,0,`${id}: ${errors.join('; ')}`);
   records.push({id,labs:count,controlChanges:exercised,semanticChecks:semantic});console.log(`PASS ${id}: ${count} labs / ${exercised} edits / ${semantic.length} mechanism checks`);
 }
 await page.setViewportSize({width:390,height:844});
 for(const id of topics){
  await page.goto(`${base}/learn/path/full-curriculum/${id}?module=classical-ml`,{waitUntil:'commit'});await page.waitForSelector(selector,{timeout:20000});await page.evaluate(()=>document.fonts.ready);
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2);assert(!overflow,id+' phone page overflow');
 }
 assert.equal(errors.length,0,errors.join('; '));
 for(const[file,digest]of Object.entries(currentHashes))assert.equal(hash(file),digest,`Source changed during browser verification: ${file}`);
 fs.writeFileSync(receipt,JSON.stringify({status:'passed',checkedAt:new Date().toISOString(),base,browser:browser.version(),records:records.sort((a,b)=>topics.indexOf(a.id)-topics.indexOf(b.id)),phone:{width:390,height:844,topicIds:topics,pageOverflow:false},reusedDesktop:reuse?{record:reusePath,sha256:hash(reusePath),topics:20,sourceUnchangedVerified:true,freshDesktopTopic:repairedTopic}:null,sourceHashes:currentHashes,verifierSha256:hash(__filename),limitations:'One Chromium engine. Control smoke checks establish an observable update; selected arithmetic, algorithm-step and invalid-input cases have explicit assertions. Direct model math is unchanged. Reused desktop groups are named explicitly; all21phone routes run fresh.'},null,2)+'\n');
 }catch(error){fs.writeFileSync(receipt,JSON.stringify({status:'failed',checkedAt:new Date().toISOString(),records,error:error.message,errors},null,2)+'\n');throw error;}finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
