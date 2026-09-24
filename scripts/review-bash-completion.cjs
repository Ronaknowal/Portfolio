const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
 const fixtures=JSON.parse(fs.readFileSync('scratch/bash-completion-review/fixtures.json','utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const errors=[],results=[];
 try{
  for(const width of [1440,390]){
   const page=await browser.newPage({viewport:{width,height:1000}});
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/bash-scripting-command-line-automation');
   await page.locator('.reader-article .lesson-intro').waitFor();
   assert.equal(await page.locator('.reader-article .wc-lab').count(),3);
   assert.equal(await page.locator('[data-bash-example]').count(),5);
   assert.equal(await page.locator('[data-report-file]').count(),5);
   await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=true));
   assert.equal(await page.locator('[data-bash-practice]').count(),1);
   assert.ok((await page.locator('.reader-footer__next').innerText()).includes('OS Processes'));
   for(const hash of await page.locator('.reader-article a[href^="#"]').evaluateAll(ns=>ns.map(n=>n.hash.slice(1))))assert.ok(await page.evaluate(hash=>!!document.getElementById(hash),hash),'missing anchor '+hash);
   assert.ok((await page.locator('.nt-resources a').count())>=3);
   const geometry=async()=>{
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'page overflow');
    assert.deepEqual(await page.locator('.wc-lab button,.wc-lab select,.wc-state').evaluateAll(ns=>ns.filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>n.textContent)),[],'lab clipping');
   };
   const argumentsLab=page.locator('[data-investigation="bash-arguments"]');
   for(const c of fixtures.argumentsCases){
    await argumentsLab.locator('select').nth(0).selectOption(c.id);await argumentsLab.locator('select').nth(1).selectOption(String(c.quoted));
    for(let step=0;step<4;step++){
     assert.equal(await argumentsLab.locator('.wc-feedback').innerText(),c.phases[step]);await geometry();
     if(step<3)await argumentsLab.getByRole('button',{name:'Next step',exact:true}).click();
    }
    assert.ok((await argumentsLab.locator('.wc-state').nth(2).innerText()).includes(c.argv.length+' arguments'));
    for(const [i,arg]of c.argv.entries())assert.ok((await argumentsLab.locator('.wc-state').nth(2).innerText()).includes((i+1)+': '+JSON.stringify(arg)));
    await argumentsLab.getByRole('button',{name:'Back',exact:true}).click();assert.equal(await argumentsLab.locator('.wc-feedback').innerText(),c.phases[2]);
   }
   const statusLab=page.locator('[data-investigation="bash-status"]');
   for(const c of fixtures.pipelineCases){
    await statusLab.locator('select').nth(0).selectOption(String(c.producer));await statusLab.locator('select').nth(1).selectOption(String(c.consumer));await statusLab.getByRole('checkbox').setChecked(c.pipefail);
    assert.ok((await statusLab.locator('.wc-feedback').innerText()).startsWith('Pipeline status: '+c.expected+'.'));await geometry();
   }
   const publication=page.locator('[data-investigation="bash-publication"]');
   for(const [index,states]of fixtures.publication.entries()){
    await publication.locator('select').selectOption(String(!!index));
    for(let step=0;step<states.length;step++){
     assert.equal(await publication.locator('.wc-feedback').innerText(),states[step].step);
     const boxes=await publication.locator('.wc-state').allTextContents();
     assert.ok(boxes[0].includes(states[step].staged)&&boxes[1].includes(states[step].visible)&&boxes[2].includes(states[step].status));await geometry();
     if(step<states.length-1)await publication.getByRole('button',{name:'Next step',exact:true}).click();
    }
   }
   await argumentsLab.locator('select').nth(0).selectOption('spaces');await argumentsLab.locator('select').nth(1).selectOption('false');
   for(let i=0;i<3;i++)await argumentsLab.getByRole('button',{name:'Next step',exact:true}).click();
   await statusLab.locator('select').nth(0).selectOption('4');await statusLab.locator('select').nth(1).selectOption('0');await statusLab.getByRole('checkbox').uncheck();
   for(const [lab,name]of [[argumentsLab,'arguments'],[statusLab,'statuses'],[publication,'publication']]){
    await page.locator('.learn-nav').evaluate(n=>n.style.visibility='hidden');
    try{await lab.screenshot({path:`scratch/bash-completion-review/${name}-${width}.png`});}
    finally{await page.locator('.learn-nav').evaluate(n=>n.style.visibility='');}
   }
   await argumentsLab.getByRole('button',{name:'Reset',exact:true}).focus();await page.keyboard.press('Enter');assert.ok(await argumentsLab.getByRole('button',{name:'Back',exact:true}).isDisabled());
   await statusLab.getByRole('checkbox').focus();await page.keyboard.press('Space');assert.ok(await statusLab.getByRole('checkbox').isChecked());
   await publication.getByRole('button',{name:'Reset',exact:true}).focus();await page.keyboard.press('Enter');assert.ok(await publication.getByRole('button',{name:'Back',exact:true}).isDisabled());
   await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=false));const hint=page.locator('summary').filter({hasText:'Hint: separate argument policy'});await hint.focus();await page.keyboard.press('Enter');assert.ok(await hint.evaluate(n=>n.parentElement.open));
   results.push({width,labs:3,programs:5,reportFiles:5,independentSolution:true,allModelConfigurations:true,keyboardAndAnchors:true,overflow:false});
   await page.close();
  }
  assert.deepEqual(errors,[]);fs.writeFileSync('scratch/bash-completion-review/browser-results.json',JSON.stringify({results,errors},null,2));console.log(JSON.stringify({results,errors},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
