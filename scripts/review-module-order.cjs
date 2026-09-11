const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {execFileSync}=require('node:child_process');
(async()=>{
 const {tracks}=await import('./lib/authoring-curriculum.mjs');
 const {learningPaths,getLearningRoute}=await import('./lib/authoring-curriculum.mjs');
 const {topicCatalogue}=await import('../src/learn/data/curriculum/topic-catalogue.js');
 // Publication changes with authoring; refresh the source-based inventory rather
 // than baking today's first planned lesson into the route test.
 execFileSync(process.execPath,['scripts/build-curriculum-inventory.mjs'],{encoding:'utf8'});
 const inventory=JSON.parse(fs.readFileSync('docs/curriculum/curriculum-inventory.json','utf8'));
 const published=new Set(inventory.topics.filter(t=>t.publicationStatus==='published').map(t=>t.id));
 const program=tracks.find(t=>t.id==='programming-scientific-computing');
 const dsa=tracks.find(t=>t.id==='data-structures-algorithms');
 const mathematics=tracks.find(t=>t.id==='math-foundations');
 const reviewedMathematicsStart=54, reviewedMathematicsEnd=57;
 const newMathematicsIds=mathematics.topicIds.slice(reviewedMathematicsStart,reviewedMathematicsEnd);
 assert.equal(newMathematicsIds.length,3);
 assert.equal(reviewedMathematicsEnd,mathematics.topicIds.length);
 assert.ok(mathematics.topicIds.every(id=>published.has(id)));
 assert.ok(newMathematicsIds.every(id=>published.has(id)));
 const firstPlannedIndex=dsa.topicIds.findIndex(id=>!published.has(id));
 const leadingPublishedIds=dsa.topicIds.slice(0,firstPlannedIndex<0?dsa.topicIds.length:firstPlannedIndex);
 assert.ok(leadingPublishedIds.length>0,'DSA navigation fixture needs published lessons');
 const nextPlannedId=firstPlannedIndex<0?null:dsa.topicIds[firstPlannedIndex];
 const plannedFixture=tracks.map(track=>({track,index:track.topicIds.findIndex(id=>!published.has(id))})).filter(item=>item.index>0).sort((a,b)=>a.index-b.index)[0];
 assert.ok(plannedFixture,'A published-prefix/planned-entry fixture is required');
 const full=getLearningRoute(learningPaths.find(p=>p.id==='full-curriculum'));
 const base=process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173',root='scratch/module-order-review';
 fs.mkdirSync(root,{recursive:true});
 const browser=await chromium.launch({channel:'msedge',headless:true}),errors=[],results=[];
 try {
  for(const width of [1440,390]) {
   const context=await browser.newContext({viewport:{width,height:1000}}),page=await context.newPage();
   page.on('pageerror',error=>errors.push(error.message));
   const ready=async id=>{await page.locator(`.reader-topic.is-current[data-topic-id="${id}"]`).waitFor();assert.equal(await page.locator('.reader-header h1').innerText(),topicCatalogue[id].title);};
   const navigate=async(selector,id)=>{await page.locator(selector).click();await page.waitForURL(url=>url.pathname.endsWith('/'+id));await ready(id);};
   await page.goto(base+'/learn');await page.locator('.path-card').first().waitFor();
   for(const path of learningPaths) {
    const route=getLearningRoute(path),card=page.locator(`[data-path-id="${path.id}"]`),count=route.topicIds.filter(id=>published.has(id)).length;
    assert.equal(await card.locator('.path-card__meta').innerText(),`${route.moduleCount} modules · ${route.topicIds.length.toLocaleString()} topics · 0 completed`);
    assert.equal(await card.locator('.path-card__availability').innerText(),`${count} published · ${route.topicIds.length-count} planned`);
   }
   for(const path of learningPaths) {
    const route=getLearningRoute(path);await page.goto(`${base}/learn/path/${path.id}`);
    await page.waitForURL(url=>url.pathname.endsWith('/'+route.topicIds[0]));await ready(route.topicIds[0]);
    assert.equal(await page.locator('.reader-group').count(),route.moduleCount);
   }
   await page.goto(`${base}/learn/path/full-curriculum/${program.topicIds[0]}`);await ready(program.topicIds[0]);
   assert.deepEqual(await page.locator(`[data-module-id="${program.id}"] .reader-topic`).evaluateAll(nodes=>nodes.map(n=>n.dataset.topicId)),program.topicIds);
   for(let i=0;i<program.topicIds.length;i++) {
    const id=program.topicIds[i];await ready(id);
    assert.ok((await page.locator('.reader-header__meta').innerText()).includes(`${i+1} of ${program.topicIds.length} topics on this route`));
    const next=full.steps[i+1];assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[next.topicId].title));
    if(i)assert.ok((await page.locator('.reader-footer__previous').innerText()).includes(topicCatalogue[program.topicIds[i-1]].title));
    await navigate('.reader-footer__next',next.topicId);
   }
   assert.equal(new URL(page.url()).searchParams.get('module'),'data-structures-algorithms');
   // Walk the published DSA sequence, including its complete-module boundary.
   await page.goto(`${base}/learn/path/full-curriculum/${leadingPublishedIds[0]}?module=${dsa.id}`);await ready(leadingPublishedIds[0]);
   for(let index=0;index<leadingPublishedIds.length;index++){
    const id=leadingPublishedIds[index];await ready(id);
    await page.waitForFunction(()=>{const button=document.querySelector('.reader-complete');return button&&!button.disabled;});
    await page.locator('.reader-complete').click();
    assert.ok(new URL(page.url()).pathname.endsWith('/'+id),'Marking complete must not silently advance the route');
    assert.equal(await page.locator(`[data-module-id="${dsa.id}"] .reader-group__completed`).innerText(),`${index+1} completed`);
    const at=full.steps.findIndex(step=>step.topicId===id&&step.moduleId===dsa.id);
    const destination=full.steps[at+1].topicId;
    await page.locator('.reader-footer__next').focus();await page.keyboard.press('Enter');
    await page.waitForURL(url=>url.pathname.endsWith('/'+destination));await ready(destination);
   }
   if(nextPlannedId){await ready(nextPlannedId);assert.ok(await page.locator('.reader-complete').isDisabled());}
   else{
    const at=full.steps.findIndex(step=>step.topicId===dsa.topicIds.at(-1)&&step.moduleId===dsa.id);
    await ready(full.steps[at+1].topicId);
    assert.equal(new URL(page.url()).searchParams.get('module'),full.steps[at+1].moduleId);
   }
   assert.equal(await page.getByText(/next published lesson|skipping ahead/i).count(),0);
   const completedDsa=new Set(leadingPublishedIds);
   const dsaRoute=getLearningRoute([dsa.id]);
   const expectedDsaResume=dsaRoute.topicIds.find(id=>!completedDsa.has(id))||dsaRoute.topicIds[0];
   await page.goto(`${base}/learn/track/${dsa.id}`);
   await page.waitForURL(url=>url.pathname.endsWith('/'+expectedDsaResume));await ready(expectedDsaResume);
   assert.equal(await page.locator(`[data-module-id="${dsa.id}"] .reader-group__completed`).innerText(),`${leadingPublishedIds.length} completed`);
   // Seed a separate existing progress history to keep planned-resume coverage
   // after DSA becomes fully published. Actual completion controls were exercised above.
   const plannedPrefix=plannedFixture.track.topicIds.slice(0,plannedFixture.index);
   const plannedResume=plannedFixture.track.topicIds[plannedFixture.index];
   await page.evaluate(ids=>{
    const progress=JSON.parse(localStorage.getItem('kd-progress')||'{}');
    for(const id of ids)progress[id]=true;
    localStorage.setItem('kd-progress',JSON.stringify(progress));
   },plannedPrefix);
   await page.goto(`${base}/learn/track/${plannedFixture.track.id}`);
   await page.waitForURL(url=>url.pathname.endsWith('/'+plannedResume));await ready(plannedResume);
   assert.ok(await page.locator('.reader-complete').isDisabled());
   assert.equal(await page.locator(`[data-module-id="${plannedFixture.track.id}"] .reader-group__completed`).innerText(),`${plannedPrefix.length} completed`);
   // Complete the final mathematics segment and follow the real next module.
   // Publication status must never replace the syllabus successor.
   await page.goto(`${base}/learn/path/full-curriculum/${newMathematicsIds[0]}?module=${mathematics.id}`);
   for(let index=0;index<newMathematicsIds.length;index++) {
    const id=newMathematicsIds[index];
    await ready(id);
    assert.ok((await page.locator('.reader-header__meta').innerText()).includes(`${index+reviewedMathematicsStart+1} of ${mathematics.topicIds.length} topics on this route`));
    assert.ok((await page.locator('.reader-footer__previous').innerText()).includes(topicCatalogue[mathematics.topicIds[index+reviewedMathematicsStart-1]].title));
    const step=full.steps.findIndex(item=>item.topicId===id&&item.moduleId===mathematics.id);
    const destinationStep=full.steps[step+1],destination=destinationStep.topicId;
    assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[destination].title));
    await page.waitForFunction(()=>{const button=document.querySelector('.reader-complete');return button&&!button.disabled;});
    await page.locator('.reader-complete').click();
    assert.ok(new URL(page.url()).pathname.endsWith('/'+id));
    await navigate('.reader-footer__next',destination);
    assert.equal(new URL(page.url()).searchParams.get('module'),destinationStep.moduleId);
   }
   assert.equal(await page.locator(`[data-module-id="${mathematics.id}"] .reader-group__completed`).innerText(),'3 completed');
   const lastMathStep=full.steps.findIndex(item=>item.topicId===mathematics.topicIds.at(-1)&&item.moduleId===mathematics.id);
   const mathematicsSuccessor=full.steps[lastMathStep+1];
   await ready(mathematicsSuccessor.topicId);
   assert.notEqual(mathematicsSuccessor.moduleId,mathematics.id);
   assert.equal(await page.locator('.reader-complete').isDisabled(),!published.has(mathematicsSuccessor.topicId));
   assert.equal(await page.getByText(/next published lesson|skipping ahead/i).count(),0);
   // A shared lesson follows the module it was selected from, even on reload.
   const module=tracks.find(t=>t.id==='robotics-embodied-ai'),shared='sim-to-real-transfer-domain-randomization';
   await page.goto(`${base}/learn/path/full-curriculum/${program.topicIds[0]}`);await ready(program.topicIds[0]);
   const group=page.locator(`[data-module-id="${module.id}"]`);await group.locator('.reader-group__toggle').click();
   await group.locator(`[data-topic-id="${shared}"]`).click();await ready(shared);await page.reload();await ready(shared);
   assert.equal(new URL(page.url()).searchParams.get('module'),module.id);
   const at=module.topicIds.indexOf(shared),next=module.topicIds[at+1];
   assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[next].title));
   await navigate('.reader-footer__next',next);await navigate('.reader-footer__previous',shared);
   // Route numbering belongs to the current module, and completion remains shared.
   const complete=page.locator('.reader-complete');
   if(!await complete.isDisabled()) {await complete.click();await page.goto(`${base}/learn/track/reinforcement-learning/${shared}`);await ready(shared);assert.match(await page.locator('.reader-complete').innerText(),/Completed/);}
   await page.goto(`${base}/learn/path/full-curriculum/iterators-iterables-generators?module=absent`);await ready('iterators-iterables-generators');
   await page.locator('.reader-prerequisites summary').click();assert.ok(await page.locator('.reader-prerequisites a').count()>=2);
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
   await page.locator('.reader-header').screenshot({path:`${root}/header-${width}.png`});
   await page.locator('.reader-sidebar').screenshot({path:`${root}/sidebar-${width}.png`});
   results.push({width,programmingSteps:program.topicIds.length,paths:learningPaths.length,sharedModuleContext:true,plannedResume:true,plannedFixture:plannedFixture.track.id,seededPlannedPrefix:plannedPrefix,plannedResumeId:plannedResume,dsaCompletedIds:leadingPublishedIds,newMathematicsIds,mathematicsSuccessor,mathematicsModuleBoundary:true,nextPlannedId,expectedDsaResume,counts:true,prerequisiteLinks:true});
   await context.close();
  }
  assert.deepEqual(errors,[]);fs.writeFileSync(`${root}/results.json`,JSON.stringify({results,errors},null,2));
  console.log(`PASS: all ${program.topicIds.length} programming steps match sidebar; ${learningPaths.length} path entries/counts; ${leadingPublishedIds.length} DSA completions preserve module boundaries and resume; planned resume in ${plannedFixture.track.id}; shared-topic progress and prerequisite links at 1440/390px.`);
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
