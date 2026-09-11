const {chromium} = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const output = path.resolve('scratch/queueing-reading-final');
fs.mkdirSync(output,{recursive:true});
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const records=[];
  try {
    for(const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1050},reducedMotion:'reduce'});
      const errors=[];
      page.on('pageerror',e=>errors.push(e.message));
      page.on('console',e=>{if(e.type()==='error')errors.push(e.text());});
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/queueing-theory-m-m-1-m-g-1-little-s-law');
      const lesson=page.locator('.queueing-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(()=>document.fonts.ready);
      for(const section of [0,2,5,7,9]) {
        await lesson.locator('h2').nth(section).evaluate(n=>window.scrollTo({top:n.getBoundingClientRect().top+scrollY-90,behavior:'instant'}));
        await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
        await page.waitForTimeout(200);
        await page.screenshot({path:path.join(output,`reading-${section+1}-${width}.png`)});
      }
      async function shot(locator,name) {
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility='hidden'));
        await locator.screenshot({path:path.join(output,`${name}-${width}.png`)});
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility=''));
      }
      const figures=lesson.locator('figure.queueing-figure');
      assert.equal(await figures.count(),3);
      await shot(figures.nth(0),'timeline');
      await shot(figures.nth(1),'birth-death');
      const birthRows=await figures.nth(1).locator('.queueing-birth-flow>span').evaluateAll(nodes=>nodes.map(n=>n.getBoundingClientRect().top));
      assert(Math.max(...birthRows)-Math.min(...birthRows)<20);
      const variability=lesson.getByRole('region',{name:'Service inspection and residual work investigation',exact:true});
      await variability.getByRole('combobox',{name:'Service distribution'}).selectOption('rareLong');
      await shot(variability,'rare-long');
      assert.equal(await variability.locator('.queueing-triangles svg').count(),2);
      for(const svg of await variability.locator('.queueing-triangles svg').all()) {
        const bounds=await svg.boundingBox();assert(bounds.width<=width);
      }
      const area=lesson.getByRole('region',{name:'Occupancy area investigation',exact:true});
      await area.getByRole('slider',{name:'Observation horizon'}).fill('5');
      await shot(area,'censored-area');
      const finite=lesson.getByRole('region',{name:'Finite capacity and admission investigation',exact:true});
      assert(await finite.locator('.queueing-state-list strong').allTextContents().then(x=>x.includes('1 job')));
      await shot(finite,'finite-buffer');
      await shot(lesson.locator('.lesson-sources'),'sources');
      const math=lesson.locator('.katex-display');
      for(const [index,label]of [[1,'occupancy-equation'],[5,'tail-derivation'],[9,'pk-derivation']])await shot(math.nth(index),label);
      const geometry=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,errors:document.querySelectorAll('.queueing-lesson .katex-error').length,font:getComputedStyle(document.querySelector('.queueing-lesson p')).fontFamily}));
      assert.equal(geometry.errors,0);assert(geometry.scroll<=width+1);assert.deepEqual(errors,[]);
      records.push({width,geometry,errors,finalGrammarAndStaticGeometry:true});
      await page.close();
    }
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify({at:new Date().toISOString(),passed:true,records},null,2));
    console.log('Final ordinary reading and static-figure checks passed at all three widths.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
