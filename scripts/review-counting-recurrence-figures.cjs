const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/counting-combinatorics-browser');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const records=[],errors=[];
  try{
    for(const width of [1440,390,320]){
      const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});
      await page.routeWebSocket('**',socket=>socket.close());
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',message=>{if(message.type()==='error'&&!message.text().includes('[vite]'))errors.push(message.text());});
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/counting-combinatorics-mathematical-induction?module=math-foundations',{waitUntil:'domcontentloaded'});
      const lesson=page.locator('.counting-lesson'); await lesson.waitFor(); await page.evaluate(()=>document.fonts.ready);
      const fonts=await page.evaluate(()=>[...document.fonts].filter(font=>font.status==='loaded').map(font=>font.family)); assert(fonts.length);
      const tiling=lesson.locator('figure[aria-label="Five length-four tilings split by first tile"]');
      const partition=lesson.locator('figure[aria-label="Insert D into a new or existing set-partition block"]');
      assert.equal(await lesson.locator('figure').count(),6);
      const rows=await tiling.locator('.counting-tile-strip').evaluateAll(nodes=>nodes.map(node=>[...node.children].map(tile=>({value:Number(tile.textContent),span:getComputedStyle(tile).gridColumnStart,width:tile.getBoundingClientRect().width,unit:(node.getBoundingClientRect().width-3*4)/4}))));
      assert.deepEqual(rows.map(row=>row.map(tile=>tile.value)),[[1,1,1,1],[1,1,2],[1,2,1],[2,1,1],[2,2]]);
      for(const row of rows){assert.equal(row.reduce((sum,tile)=>sum+tile.value,0),4); for(const tile of row){assert.equal(tile.span,`span ${tile.value}`);assert(Math.abs(tile.width-(tile.value*tile.unit+(tile.value-1)*4))<1);}}
      const partitions=await partition.locator('.counting-partition').evaluateAll(nodes=>nodes.map(node=>[...node.children].map(block=>block.textContent)));
      assert.deepEqual(partitions,[['ABC'],['ABC','D'],['AB','C'],['ABD','C'],['AB','CD']]);
      const captures=[];
      async function shot(target,name){await target.evaluate(node=>window.scrollTo({top:node.getBoundingClientRect().top+scrollY-85,behavior:'instant'}));await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));await page.waitForTimeout(150);const box=await target.boundingBox();assert(box.y>0&&box.y<150);const file=`${name}-${width}.png`;await page.screenshot({path:path.join(directory,file)});captures.push(file);}
      await shot(tiling,'final-tiling-decomposition');await shot(partition,'final-partition-decomposition');
      await shot(partition.locator('figcaption'),'final-partition-reading');
      for(const target of [tiling,partition]){
        const overflow=await target.evaluate(node=>{const outer=node.getBoundingClientRect();return [...node.querySelectorAll('*')].filter(child=>{const box=child.getBoundingClientRect();return box.left<outer.left-1||box.right>outer.right+1;}).map(child=>child.className);});assert.deepEqual(overflow,[]);
      }
      assert.equal(await lesson.locator('.katex-error').count(),0);
      assert.equal(await lesson.locator('.katex-display').count(),14);
      assert((await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.every(node=>node.scrollWidth<=node.clientWidth+1))));
      assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)));
      records.push({width,fonts,rows,partitions,captures});await page.close();
    }
    assert.deepEqual(errors,[]);fs.writeFileSync(path.join(directory,'final-recurrence-results.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,records,errors,scope:'Final two inline recurrence representations; comprehensive earlier controls remain unchanged.'},null,2));
    console.log('Final recurrence figures and unchanged14 equation fits pass at1440/390/320 with intended fonts.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
