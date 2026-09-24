const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory=path.resolve('scratch/random-matrix-browser');
(async()=>{
  const {randomMatrixExamples:examples}=await import(pathToFileURL(path.resolve('src/learn/data/random-matrix-examples.js')));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const records=[];
  try {
    for(const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});
      await page.routeWebSocket('**',socket=>socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/random-matrix-theory?module=math-foundations',{waitUntil:'domcontentloaded',timeout:60000});
      const root=page.locator('.random-matrix-lesson');
      await root.waitFor();await page.evaluate(()=>document.fonts.ready);
      assert.equal(await root.locator('.katex-error').count(),0);
      assert.equal(await root.locator('.katex-display').count(),18);
      assert.equal(await root.locator('.rm-practice').count(),8);
      const captures=[];
      const mass=root.locator('.python-example').filter({has:page.getByRole('heading',{name:examples.mass.title,exact:true})});
      const normalize=value=>value.replace(/\s+/g,' ').trim();
      const blocks=mass.locator(':scope > div');
      assert(normalize(await blocks.nth(0).innerText()).includes(normalize(examples.mass.code)));
      assert.equal(normalize(await blocks.nth(1).innerText()).replace(/^OUTPUT /,''),normalize(examples.mass.expected));
      assert(normalize(await mass.evaluate(node=>node.previousElementSibling.textContent)).includes(normalize(examples.mass.question)));
      assert(normalize(await mass.evaluate(node=>node.nextElementSibling.textContent)).includes('narrow endpoint layer'));
      const targets=[
        [root.locator('.katex-display').nth(7),'final-finite-bounds'],
        [root.locator('.katex-display').nth(13),'final-inverse-equation'],
        [root.locator('.python-example').first(),'final-first-program'],
        [root.locator('.python-example').first().locator('.lesson-note'),'final-first-output'],
        [root.locator('.lesson-sources'),'final-sources'],
        [mass,'final-mass-program'],
        [mass.locator('.lesson-note'),'final-mass-output'],
      ];
      for(const [target,name] of targets) {
        await target.evaluate(node=>window.scrollTo({top:node.getBoundingClientRect().top+scrollY-90,behavior:'instant'}));
        await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
        await page.waitForTimeout(200);
        const filename=name+'-'+width+'.png';
        await page.screenshot({path:path.join(directory,filename)});captures.push(filename);
      }
      const external=await root.locator('.lesson-sources a[href^="https"]').evaluateAll(nodes=>nodes.map(node=>({href:node.href,target:node.target,rel:node.rel})));
      assert(external.every(link=>link.target==='_blank'&&link.rel.includes('noreferrer')));
      assert.equal(await root.locator('a[href*="queueing-theory-m-m-1-m-g-1-little-s-law"]').count(),1);
      const geometry=await root.locator('.rm-inline').first().locator('svg').evaluate(node=>{
        const lines=[...node.querySelectorAll('path')].map(path=>path.getAttribute('d'));
        const colors=[...node.querySelectorAll('text[fill]')].map(text=>getComputedStyle(text).fill);
        return {lines,colors};
      });
      assert.deepEqual(geometry.lines,['M50,115 H310 M180,205 V25','M105,190 L255,40','M155,90 L205,140']);
      assert(geometry.colors.every(color=>color!=='rgb(0, 0, 0)'));
      records.push({width,captures,external,geometry,displayedMassHelperCodeOutputQuestionAndExplanation:true});
      await page.close();
    }
    fs.writeFileSync(path.join(directory,'final-reading-results.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,records},null,2));
    console.log('Final ordinary-reading, source-link and exact static-figure checks passed at 1440/390/320.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
