const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const folder = path.resolve('scratch/hypothesis-testing-browser');
fs.mkdirSync(folder,{recursive:true});
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const results=[];
  try {
    for(const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1050},reducedMotion:'reduce'});
      const errors=[],warnings=[],failedRequests=[];
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',event=>{if(event.type()==='error')errors.push({text:event.text(),location:event.location()});if(event.type()==='warning')warnings.push(event.text());});
      page.on('requestfailed',request=>failedRequests.push({url:request.url(),failure:request.failure()}));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/hypothesis-testing-confidence-intervals');
      const lesson=page.locator('.hypothesis-lesson');
      await lesson.locator('h2').last().waitFor();
      const checkpoints=lesson.locator('.lesson-check');
      assert.equal(await checkpoints.count(),2);
      const checkpointAnswers=['The fraction improved is 4/5; the mean saving is 2 ms.','the t critical value also changes because the degrees of freedom increase'];
      for(let index=0;index<2;index++){
        const checkpoint=checkpoints.nth(index);
        assert((await checkpoint.locator('p').first().innerText()).length>100,'A complete question is rendered');
        assert.equal(await checkpoint.locator('details').getAttribute('open'),null);
        await checkpoint.locator('summary').focus();await page.keyboard.press('Enter');
        assert(await checkpoint.locator('details > div').isVisible());
        assert((await checkpoint.locator('details > div').innerText()).includes(checkpointAnswers[index]));
        await checkpoint.evaluate(node=>window.scrollTo(0,node.getBoundingClientRect().top+scrollY-100));
        await page.screenshot({path:path.join(folder,'checkpoint-'+(index+1)+'-'+width+'.png')});
      }
      for(let index=0;index<10;index++){
        await lesson.locator('h2').nth(index).evaluate(node=>window.scrollTo(0,node.getBoundingClientRect().top+scrollY-100));
        await page.screenshot({path:path.join(folder,`reading-${index+1}-${width}.png`)});
      }
      for(let index=0;index<2;index++){
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(node=>node.style.visibility='hidden'));
        await lesson.locator('.hypothesis-inline').nth(index).screenshot({path:path.join(folder,`inline-${index+1}-${width}.png`)});
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(node=>node.style.visibility=''));
      }
      const visited=[];
      for(let labIndex=0;labIndex<8;labIndex++){
        const lab=lesson.locator('.hypothesis-lab').nth(labIndex);
        const controls=lab.locator('input:enabled,select:enabled,button:enabled');
        await controls.first().focus();
        for(let index=0;index<await controls.count();index++){
          const control=controls.nth(index);
          assert(await control.evaluate(node=>node===document.activeElement));
          const style=await control.evaluate(node=>({label:node.getAttribute('aria-label')||node.textContent,outline:getComputedStyle(node).outlineStyle,width:getComputedStyle(node).outlineWidth,height:node.getBoundingClientRect().height}));
          assert.notEqual(style.outline,'none');assert(parseFloat(style.width)>0);assert(style.height>=43);
          visited.push(style);await page.keyboard.press('Tab');
        }
      }
      const sample=lesson.getByRole('slider',{name:'Observations per experiment',exact:true});
      await sample.focus();await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');assert.equal(await sample.inputValue(),'10');
      const select=lesson.getByRole('combobox',{name:'Coverage spread model'});await select.focus();await page.keyboard.press('End');await page.keyboard.press('Enter');assert.equal(await select.inputValue(),'estimated');
      const disclosure=lesson.locator('.hypothesis-practice summary').first();await disclosure.focus();await page.keyboard.press('Enter');assert.equal(await lesson.locator('.hypothesis-practice details[open]').count(),1);
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>node.open=true));
      const math=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map((node,index)=>({index,text:node.textContent.slice(0,120),width:node.clientWidth,scroll:node.scrollWidth})));
      const overflowingMath=math.filter(row=>row.scroll>row.width+2);
      const clippedLabels=await lesson.locator('svg.hypothesis-plot').evaluateAll(nodes=>nodes.flatMap((svg,index)=>{
        const boundary=svg.getBoundingClientRect();
        return [...svg.querySelectorAll('text')].filter(node=>{const box=node.getBoundingClientRect();return box.left<boundary.left-1||box.right>boundary.right+1||box.top<boundary.top-1||box.bottom>boundary.bottom+1;}).map(node=>({index,text:node.textContent}));
      }));
      let keyboardTables=0;
      const tables=lesson.locator('.lesson-table-wrap');
      for(let index=0;index<await tables.count();index++){
        const table=tables.nth(index);
        if(await table.evaluate(node=>node.scrollWidth>node.clientWidth+1)){
          await table.focus();for(let press=0;press<6;press++)await page.keyboard.press('ArrowRight');
          await page.waitForFunction(node=>node.scrollLeft>0,await table.elementHandle());keyboardTables++;
        }
      }
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      await lesson.locator('.hypothesis-practice').first().evaluate(node=>window.scrollTo(0,node.getBoundingClientRect().top+scrollY-100));
      await page.screenshot({path:path.join(folder,`practice-open-${width}.png`)});
      await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(node=>node.style.visibility='hidden'));
      await lesson.locator('.lesson-sources').screenshot({path:path.join(folder,`sources-${width}.png`)});
      const result={width,ordinarySections:10,inlineFigures:2,nonemptyCheckpoints:2,keyboardControls:visited.length,visited,keyboardTables,mathCount:math.length,overflowingMath,clippedLabels,overflow,errors,warnings,failedRequests};
      results.push(result);fs.writeFileSync(path.join(folder,'reading-in-progress.json'),JSON.stringify(results,null,2));
      assert.deepEqual(overflowingMath,[]);assert.deepEqual(clippedLabels,[]);assert.equal(overflow,false);
      assert.equal(await lesson.locator('.katex-error').count(),0);assert.equal(await lesson.locator('p p,p div,p section').count(),0);
      assert.deepEqual(errors,[]);assert.deepEqual(warnings.filter(text=>/nest|hydration|descendant/i.test(text)),[]);
      await page.close();
    }
  }finally{await browser.close();}
  fs.writeFileSync(path.join(folder,'reading-results.json'),JSON.stringify({at:new Date().toISOString(),status:'passed',results},null,2));console.log(results.map(({width,keyboardControls,keyboardTables,mathCount})=>({width,keyboardControls,keyboardTables,mathCount})));
})().catch(error=>{console.error(error);process.exitCode=1;});
