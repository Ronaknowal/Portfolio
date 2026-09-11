const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const hashes=require('./dp-state-families-source-hashes.cjs');
const directory='scratch/dp-state-families-reading';
fs.mkdirSync(directory,{recursive:true});
(async()=>{
 const sourceHashes=hashes();
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const results=[];
 try{
  for(const width of [1440,390,320]){
   const page=await browser.newPage({viewport:{width,height:1100}});
   await page.routeWebSocket('**',socket=>socket.close());
   const errors=[],failedRequests=[];
   page.on('pageerror',error=>errors.push(error.message));
   page.on('requestfailed',request=>failedRequests.push(request.url()));
   await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms');
   const lesson=page.locator('.dynamic-programming-lesson');
   await lesson.waitFor();
   await page.evaluate(()=>document.fonts.ready);
   const fonts=await page.evaluate(()=>[...document.fonts].map(font=>({family:font.family,status:font.status})));
   assert(fonts.some(font=>font.family.replaceAll('"','')==='Space Grotesk'&&font.status==='loaded'));
   const disclosures=await lesson.locator('details>summary').all();
   const overflow=[];
   for(const disclosure of disclosures){
    await disclosure.focus();await page.keyboard.press('Enter');
    const size=await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth);
    if(size>1)overflow.push({summary:await disclosure.innerText(),size});
   }
   async function capture(target,name){
    await target.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-95,behavior:'instant'}));
    await page.screenshot({path:`${directory}/${name}-${width}.png`});
   }
   await capture(lesson.locator('p').filter({hasText:'For nonadjacent rewards, a value-only backward loop'}),'original-compression-formula');
   for(const [part,name] of [['Derive the changed recurrence and its endpoint rule','digit-changed-solution'],['signed singleton','balloon-signed-solution'],['Change a two-node chain','tree-nonempty-solution']])await capture(lesson.locator('.lesson-check').filter({hasText:part}),name);
   for(const [index,figure] of(await lesson.locator('.dpf-inline').all()).entries())await capture(figure,`inline-${index}`);
   const bodyReading=await lesson.locator('p').evaluateAll(nodes=>nodes.filter(node=>/C\(i,j\) = min|B\(l,r\) = max|state.*position,tight/.test(node.textContent)).map(node=>({text:node.textContent,width:node.clientWidth,scroll:node.scrollWidth})));
   results.push({width,fonts,disclosures:disclosures.length,overflow,errors,failedRequests,bodyReading});
   await page.close();
  }
  assert.deepEqual(hashes(),sourceHashes);
  const record={checkedAt:new Date().toISOString(),sourceHashes,results};
  fs.writeFileSync('docs/teaching/evidence/dp-state-families-reading.json',JSON.stringify(record,null,2)+'\n');
  for(const result of results){assert.deepEqual(result.overflow,[]);assert.deepEqual(result.errors,[]);assert.deepEqual(result.failedRequests,[]);}
  console.log(JSON.stringify(results.map(({width,disclosures,overflow})=>({width,disclosures,overflow}))));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
