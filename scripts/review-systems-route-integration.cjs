const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
  const {learningPaths,getLearningRoute}=await import('./lib/authoring-curriculum.mjs');
  const {topicCatalogue}=await import('../src/learn/data/curriculum/topic-catalogue.js');
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const errors=[],results=[];
  try{
    for(const width of [1440,390]){
      const page=await browser.newPage({viewport:{width,height:1000}});page.on('pageerror',e=>errors.push(e.message));
      await page.goto('http://127.0.0.1:5173/learn');await page.locator('.path-card').first().waitFor();
      for(const path of learningPaths){
        const route=getLearningRoute(path);
        assert.equal(await page.locator(`[data-path-id="${path.id}"] .path-card__meta`).innerText(),`${route.moduleCount} modules · ${route.topicIds.length.toLocaleString()} topics · 0 completed`);
      }
      for(const id of ['embodied-intelligence','research-revision']){
        const route=getLearningRoute(learningPaths.find(p=>p.id===id));
        const topic='linked-lists-stacks-queues',index=route.topicIds.indexOf(topic);
        await page.goto(`http://127.0.0.1:5173/learn/path/${id}/${topic}`);await page.locator('.reader-article .lesson-intro').waitFor();
        assert.equal(await page.locator('.reader-group').count(),route.moduleCount);
        for(const group of route.navigationGroups){
          const row=page.locator(`[data-module-id="${group.id}"]`);
          assert.equal(await row.locator('.reader-group__size > span').first().textContent(),`${group.topicIds.length} ${group.topicIds.length===1?'topic':'topics'}`);
          assert.equal(await row.locator('.reader-group__completed').textContent(),'0 completed');
        }
        // Supporting prerequisites remain available; module contents own order.
        for(const pre of ['arrays-strings-hash-maps','object-oriented-programming-in-python'])assert.ok(route.topicIds.includes(pre));
        assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[route.topicIds[index+1]].title));
        await page.locator('.reader-footer__next').click();await page.waitForURL(url=>url.pathname.endsWith('/'+route.topicIds[index+1]));
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
        results.push({width,path:id,modules:route.moduleCount,topics:route.topicIds.length,countsAndNext:true});
      }
      await page.close();
    }
    assert.deepEqual(errors,[]);fs.writeFileSync('scratch/systems-three-review/route-integration.json',JSON.stringify({results,errors},null,2));
    console.log('PASS: all seven hub counts agree with resolved data; both affected focused paths show correct module/topic/completion counts and next navigation at 1440/390.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
