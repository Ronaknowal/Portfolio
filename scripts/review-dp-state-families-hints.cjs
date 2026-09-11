const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const hashes=require('./dp-state-families-source-hashes.cjs');
const directory='scratch/dp-state-families-hints';fs.mkdirSync(directory,{recursive:true});
(async()=>{
 const sourceHashes=hashes();const browser=await chromium.launch({channel:'msedge',headless:true});const results=[];
 try{
 for(const width of [1440,390,320]){
  const page=await browser.newPage({viewport:{width,height:1100}});await page.routeWebSocket('**',socket=>socket.close());
  const errors=[],failedRequests=[];page.on('pageerror',error=>errors.push(error.message));page.on('requestfailed',request=>failedRequests.push(request.url()));
  await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms');
  const lesson=page.locator('.dynamic-programming-lesson');await lesson.waitFor();await page.evaluate(()=>document.fonts.ready);
  const fonts=await page.evaluate(()=>[...document.fonts].map(font=>({family:font.family,status:font.status})));
  assert(fonts.some(font=>font.family.replaceAll('"','')==='Space Grotesk'&&font.status==='loaded'));
  const checkpoints=await lesson.locator('[data-dpf-checkpoint]').all();assert.equal(checkpoints.length,5);
  for(const [index,checkpoint] of checkpoints.entries()){
   const hint=checkpoint.locator('details').nth(0),answer=checkpoint.locator('details').nth(1);
   assert.equal(await hint.getAttribute('open'),null);assert.equal(await answer.getAttribute('open'),null);
   assert.equal(await checkpoint.locator(':scope>p').count(),1);
   await hint.locator('summary').focus();await page.keyboard.press('Enter');
   assert.notEqual(await hint.getAttribute('open'),null);assert.equal(await answer.getAttribute('open'),null);
   assert(await hint.locator('p').isVisible());
   await checkpoint.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-95,behavior:'instant'}));
   await page.screenshot({path:`${directory}/hint-${index}-${width}.png`});
   await answer.locator('summary').focus();await page.keyboard.press('Enter');
   assert.notEqual(await answer.getAttribute('open'),null);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  }
  const summaries=await lesson.locator('details>summary').all();
  for(const summary of summaries){if(await summary.evaluate(node=>node.parentElement.open))continue;await summary.focus();await page.keyboard.press('Enter');assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
  assert.equal(await lesson.locator('details[open]').count(),55);
  assert.equal(await lesson.locator('.python-example').count(),20);
  assert.deepEqual(errors,[]);assert.deepEqual(failedRequests,[]);
  results.push({width,hints:5,expandedDisclosures:55,programs:20,fonts,errors,failedRequests});await page.close();
 }
 assert.deepEqual(hashes(),sourceHashes);
 fs.writeFileSync('docs/teaching/evidence/dp-state-families-hint-browser.json',JSON.stringify({checkedAt:new Date().toISOString(),sourceHashes,results},null,2)+'\n');
 console.log(JSON.stringify(results.map(({width,hints,expandedDisclosures})=>({width,hints,expandedDisclosures}))));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
