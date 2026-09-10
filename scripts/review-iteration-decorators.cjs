const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
 const fixtures=JSON.parse(fs.readFileSync('scratch/iteration-decorators-review/fixtures.json','utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const errors=[],results=[];
 const topics=[['iterators-iterables-generators',14,[['iterator-ownership','cursors',['shared','empty']],['generator-frame','generators',['action']],['iterator-pipeline','pipelines',['limit','bad']]]],['decorators-context-managers',11,[['decorator-order','orders',['outer','value']],['context-lifetime','contexts',['path','suppress']],['context-exit-stack','stacks',['fail']]]]];
 try{
  for(const width of [1440,390]){
   const page=await browser.newPage({viewport:{width,height:1100}});
   page.on('pageerror',e=>errors.push(e.message));
   for(const [id,programs,labs] of topics){
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/'+id);
    await page.locator('.reader-article .lesson-intro').waitFor();
    assert.equal(await page.locator('.reader-article [data-investigation]').count(),3);
    await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=true));
    assert.equal(await page.locator('.reader-article .python-example').count(),programs);
    for(const hash of await page.locator('.reader-article a[href^="#"]').evaluateAll(ns=>ns.map(n=>n.hash.slice(1))))assert.ok(await page.evaluate(hash=>!!document.getElementById(hash),hash),'Missing anchor '+hash);
    assert.ok((await page.locator('.nt-resources a').count())>=3);
    for(const [labId,kind,keys] of labs){
     const lab=page.locator(`[data-investigation="${labId}"]`);
     for(const config of fixtures[kind]){
      for(let i=0;i<keys.length;i++)await lab.locator('select').nth(i).selectOption(String(config[keys[i]]));
      await lab.getByRole('button',{name:'Reset',exact:true}).click();
      for(let i=0;i<config.states.length;i++){
       assert.equal(await lab.locator('.nt-feedback').innerText(),config.states[i].note);
       assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'page overflow');
       const clipped=await lab.locator('.id-slot,.id-frame>div,.id-flow-node,.id-call-layer,.id-lifetime span,select,button').evaluateAll(ns=>ns.filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>({text:n.textContent,scroll:n.scrollWidth,client:n.clientWidth})));
       assert.deepEqual(clipped,[],'clipped text/control '+labId);
       if(i<config.states.length-1)await lab.getByRole('button',{name:'Next step',exact:true}).click();
      }
      assert.ok(await lab.getByRole('button',{name:'Next step',exact:true}).isDisabled());
      if(config.states.length>1){await lab.getByRole('button',{name:'Back',exact:true}).click();assert.equal(await lab.locator('.nt-feedback').innerText(),config.states.at(-2).note);}
     }
     const config=fixtures[kind].find(c=>c.shared===true&&!c.empty||c.action==='exhaust'||c.outer==='cap'&&c.value===8||c.path==='body-fails'&&!c.suppress||c.bad===true&&c.limit===2||c.fail==='C')||fixtures[kind][0];
     for(let i=0;i<keys.length;i++)await lab.locator('select').nth(i).selectOption(String(config[keys[i]]));
     const picturedStep=({cursors:2,generators:2,pipelines:9,orders:4,contexts:3,stacks:3})[kind];
     for(let i=0;i<Math.min(picturedStep,config.states.length-1);i++)await lab.getByRole('button',{name:'Next step',exact:true}).click();
     const nav=page.locator('.learn-nav');await nav.evaluate(n=>n.style.visibility='hidden');
     try{await lab.screenshot({path:`scratch/iteration-decorators-review/${labId}-${width}.png`});}
     finally{await nav.evaluate(n=>n.style.visibility='');}
     const button=lab.getByRole('button',{name:'Reset',exact:true});await button.focus();await page.keyboard.press('Enter');
     assert.ok(await lab.getByRole('button',{name:'Back',exact:true}).isDisabled());
     assert.equal(await lab.locator('.nt-feedback').innerText(),config.states[0].note);
    }
    await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=false));
    const hint=page.locator('.reader-article summary').filter({hasText:/^Hint:/}).first();await hint.focus();await page.keyboard.press('Enter');assert.ok(await hint.evaluate(n=>n.parentElement.open));
    results.push({id,width,labs:3,programs,allPresetsAndSteps:true,keyboardAndDisclosure:true,overflow:false});
   }
   await page.close();
  }
  assert.deepEqual(errors,[]);
  fs.writeFileSync('scratch/iteration-decorators-review/browser-results.json',JSON.stringify({results,errors},null,2));
  console.log(JSON.stringify({results,errors},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
