const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');const assert=require('node:assert/strict');
(async()=>{const browser=await chromium.launch({channel:'msedge',headless:true});const results=[];try{for(const width of [1440,390,320]){
  const page=await browser.newPage({viewport:{width,height:1000}});await page.routeWebSocket('**',s=>s.close());const errors=[],failed=[];page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>failed.push(r.url()));
  await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/random-variables-expectation-covariance?module=math-foundations',{waitUntil:'networkidle'});await page.evaluate(()=>document.fonts.ready);
  const lesson=page.locator('.random-variable-lesson');assert.equal(await lesson.getByText('Keep the objects separate',{exact:true}).count(),1);
  for(const [name,target] of [['object-legend',lesson.locator('.lesson-table-wrap').first()],['bernoulli-bridge',lesson.locator('p').filter({hasText:'A Bernoulli variable B is 1'})]]){
    await target.evaluate(n=>window.scrollTo(0,window.scrollY+n.getBoundingClientRect().top-140));await page.screenshot({path:`scratch/random-variables-browser/final-${name}-${width}.png`});
  }
  const text=await lesson.locator('.lesson-sources').innerText();assert(text.includes('the block-independence construction'));assert(!text.includes('slides 32–39'));
  const noise=page.getByRole('region',{name:'Shared noise investigation',exact:true});
  for(const [action,expected] of [['Average (A+B)/2','1S + 0.5e_A + 0.5e_B'],['Difference B−A','0S − 1e_A + 1e_B']]){
    await noise.getByRole('button',{name:action,exact:true}).focus();await page.keyboard.press('Enter');assert((await noise.locator('.rv-combine').innerText()).includes(expected));
  }
  assert((await noise.locator('.rv-noise').innerText()).includes('A noise contribution'));
  await noise.locator('.rv-noise').evaluate(n=>window.scrollTo(0,window.scrollY+n.getBoundingClientRect().top-140));await page.screenshot({path:`scratch/random-variables-browser/final-noise-labels-${width}.png`});
  const squared=page.getByRole('region',{name:'Squared uniform transformation investigation',exact:true});assert.equal(await squared.locator('thead th').last().innerText(),'Mass');
  await squared.locator('table').evaluate(n=>window.scrollTo(0,window.scrollY+n.getBoundingClientRect().top-140));await page.screenshot({path:`scratch/random-variables-browser/final-preimage-table-${width}.png`});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);results.push({width,objects:4,bernoulliBridge:true,sourceAnnotation:true,errors,failed});await page.close();
}}finally{await browser.close();}fs.writeFileSync('scratch/random-variables-browser/final-text-results.json',JSON.stringify({checkedAt:new Date().toISOString(),results},null,2));console.log(JSON.stringify(results));})().catch(e=>{console.error(e);process.exitCode=1;});
