const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const out=path.resolve('scratch/workflow-visual-review');fs.mkdirSync(out,{recursive:true});
const topics={
 notebook:'reproducible-notebooks-experiment-structure',api:'code-documentation-type-hints-api-design',
 bash:'bash-scripting-command-line-automation',threads:'threads-concurrency-locks-deadlocks',git:'git-github-collaborative-version-control'
};
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});const errors=[],results=[];
 for(const width of [1440,390]){
  const page=await browser.newPage({viewport:{width,height:1000}});page.on('pageerror',e=>errors.push(String(e)));
  for(const [name,slug] of Object.entries(topics)){
   await page.goto(`http://127.0.0.1:5173/learn/path/full-curriculum/${slug}?module=programming-scientific-computing`);
   await page.locator('[data-visual]').first().waitFor();
   const inspect=async id=>{
    const figure=page.locator(`[data-visual="${id}"]`);await figure.scrollIntoViewIfNeeded();
    const measurement=await figure.evaluate(el=>({client:el.clientWidth,scroll:el.scrollWidth,caption:el.querySelector('figcaption')?.innerText,text:el.innerText}));
    assert.ok(measurement.scroll<=measurement.client+2,`${id} figure overflow at ${width}`);assert.ok(measurement.caption);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false,slug+' page overflow');
    await page.locator('.learn-nav').evaluateAll(ns=>ns.forEach(n=>n.style.visibility='hidden'));
    await figure.screenshot({path:path.join(out,`${id}-${width}.png`)});
    await page.locator('.learn-nav').evaluateAll(ns=>ns.forEach(n=>n.style.visibility=''));
    results.push({id,width,...measurement});
   };
   if(name==='notebook'){
    const f=page.locator('[data-visual="notebook-three-stores"]');assert.equal(await f.locator('mark').innerText(),'5');assert.deepEqual(await f.locator('dd').allTextContents(),['2','18.0']);await inspect('notebook-three-stores');
    const lab=page.locator('[data-investigation="notebook-kernel"]');await lab.getByRole('button',{name:'Run all',exact:true}).click();await lab.locator('select').selectOption('5');await lab.getByRole('button',{name:'Run Display',exact:true}).click();assert.equal(await lab.locator('output').innerText(),'18');
    await lab.getByRole('button',{name:'Run Inputs',exact:true}).click();await lab.getByRole('button',{name:'Run Calculate',exact:true}).click();await lab.getByRole('button',{name:'Run Display',exact:true}).click();assert.equal(await lab.locator('output').innerText(),'15');
   }
   if(name==='api'){assert.match(await page.locator('[data-visual="api-checking-paths"]').innerText(),/"haha"/);await inspect('api-checking-paths');}
   if(name==='bash'){
    const lab=page.locator('[data-investigation="bash-arguments"]');
    for(const kind of ['spaces','wildcard','empty'])for(const quoted of [false,true]){
     await lab.locator('select').nth(0).selectOption(kind);await lab.locator('select').nth(1).selectOption(String(quoted));
     for(let i=0;i<3;i++)await lab.getByRole('button',{name:'Next step',exact:true}).click();
     const expected=quoted?1:kind==='empty'?0:2;
     assert.match(await lab.locator('.wv-argument-receiver > strong').innerText(),new RegExp(`receives ${expected} positional argument`));
     if(kind==='empty'&&quoted)assert.equal(await lab.locator('.wv-arguments code').innerText(),'""');
     await lab.getByRole('button',{name:'Back',exact:true}).click();assert.match(await lab.locator('.wv-argument-receiver > strong').innerText(),/prepared/);
     await lab.getByRole('button',{name:'Reset',exact:true}).click();
    }
    await lab.locator('select').nth(0).selectOption('spaces');await lab.locator('select').nth(1).selectOption('false');for(let i=0;i<3;i++)await lab.getByRole('button',{name:'Next step',exact:true}).click();await inspect('bash-argument-boundaries');
    const status=page.locator('[data-investigation="bash-status"]');
    for(const a of [0,4])for(const b of [0,2])for(const strict of [false,true]){
     await status.locator('select').nth(0).selectOption(String(a));await status.locator('select').nth(1).selectOption(String(b));await status.getByRole('checkbox').setChecked(strict);
     const expected=strict?b||a:b;assert.equal(await status.locator('.wv-pipeline-result strong').innerText(),String(expected));
    }
    await status.locator('select').nth(1).selectOption('0');await inspect('bash-output-status-channels');
    const checkbox=status.getByRole('checkbox');await checkbox.focus();await page.keyboard.press('Space');assert.equal(await checkbox.isChecked(),false);
   }
   if(name==='threads'){
    const lab=page.locator('[data-investigation="thread-race"]');
    for(const worker of ['A','B','A','B','A','B'])await lab.getByRole('button',{name:'Advance '+worker,exact:true}).click();assert.equal(await lab.locator('.wv-shared-counter output').innerText(),'1');assert.match(await lab.locator('[aria-live]').innerText(),/observed 1/);await inspect('thread-execution-lanes');
    await lab.locator('select').selectOption('true');await lab.getByRole('button',{name:'Advance A',exact:true}).click();await lab.getByRole('button',{name:'Advance B',exact:true}).click();assert.match(await lab.locator('.wv-worker').nth(1).innerText(),/Cannot enter: A holds/);assert.equal(await lab.locator('.wv-worker').nth(1).locator('output').innerText(),'—');
    for(const worker of ['A','A','B','B','B'])await lab.getByRole('button',{name:'Advance '+worker,exact:true}).click();assert.equal(await lab.locator('.wv-shared-counter output').innerText(),'2');await lab.getByRole('button',{name:'Reset',exact:true}).click();
    const advance=lab.getByRole('button',{name:'Advance A',exact:true});await advance.focus();await page.keyboard.press('Enter');assert.equal(await lab.locator('.wv-worker').nth(0).locator('output').innerText(),'0');assert.equal(await advance.evaluate(e=>e===document.activeElement),true);
   }
   if(name==='git'){
    const lab=page.locator('[data-investigation="git-remotes"]');
    for(const local of [false,true]){
     await lab.locator('select').selectOption(String(local));for(let i=0;i<3;i++)await lab.getByRole('button',{name:'Next step',exact:true}).click();
     const v=lab.locator('[data-visual="git-repository-boundary"]');assert.match(await v.locator('.wv-remote-server strong').innerText(),/main → B/);assert.match(await v.locator('.wv-git-local-flow').innerText(),/origin\/main → B/);assert.match(await v.locator('.wv-git-local-flow').innerText(),new RegExp(`main → ${local?'C':'A'}`));
     await lab.getByRole('button',{name:'Next step',exact:true}).click();assert.equal(await v.locator('.wv-working-file code').innerText(),local?'version 1 + local note':'version 2');
     if(local)assert.match(await v.locator('.wv-integrate-link').innerText(),/Refused/);await lab.getByRole('button',{name:'Reset',exact:true}).click();
    }
    for(let i=0;i<3;i++)await lab.getByRole('button',{name:'Next step',exact:true}).click();await inspect('git-repository-boundary');
    const next=lab.getByRole('button',{name:'Next step',exact:true});await next.focus();await page.keyboard.press('Enter');assert.match(await lab.locator('.wv-integrate-link').innerText(),/Refused/);
   }
  }
  await page.close();
 }
 assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'browser-results.json'),JSON.stringify({date:new Date().toISOString(),results,errors},null,2));await browser.close();console.log(`PASS: ${results.length} desktop/mobile visual inspections, notebook stale/recomputed values, six argv cases, eight status combinations, both race protocols, both remote histories and keyboard operations.`);
})().catch(e=>{console.error(e);process.exit(1);});
