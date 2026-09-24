const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const hashes=require('./dp-state-families-source-hashes.cjs');
const directory='scratch/dp-state-families-select-label';
fs.mkdirSync(directory,{recursive:true});
(async()=>{
 const sourceHashes=hashes();
 const {matrixChainPlan}=await import('../src/learn/data/dp-state-families-models.js');
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
   const lab=page.getByRole('region',{name:'Interval split investigation',exact:true});
   await lab.waitFor();await page.evaluate(()=>document.fonts.ready);
   const values=[];
   for(const dimensions of [[8,2,12,3,6],[3,7,2,5],[2,2,2,2],[20,20,20,20,20,20,20]]){
    await lab.getByLabel('Matrix-chain dimensions',{exact:true}).fill(dimensions.join(','));
    await lab.getByRole('button',{name:'Apply dimensions',exact:true}).click();
    const model=matrixChainPlan(dimensions);
    for(const cell of model.cells.filter(cell=>cell.candidates.length)){
     await lab.getByRole('button',{name:`Inspect interval ${cell.left} to ${cell.right}`,exact:true}).click();
     for(const candidate of cell.candidates){
      const select=lab.getByLabel('Candidate final split',{exact:true});
      await select.selectOption(String(candidate.split));
      const measurement=await select.evaluate(node=>{
       const style=getComputedStyle(node);const canvas=document.createElement('canvas');const context=canvas.getContext('2d');context.font=style.font;
       const text=node.selectedOptions[0].textContent;
       return {text,textWidth:context.measureText(text).width,available:node.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight)-28};
      });
      assert(measurement.textWidth<=measurement.available,JSON.stringify(measurement));
      assert.equal(await lab.locator('[data-dpf-status]').innerText(),`Left ${candidate.first} + right ${candidate.second} + final ${candidate.merge} = ${candidate.total}. Best interval cost: ${cell.cost}.`);
      values.push(measurement);
     }
    }
   }
   await lab.getByRole('button',{name:'Reset interval lab',exact:true}).focus();await page.keyboard.press('Enter');
   const heading=lab.locator('h4').first();
   await heading.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-90,behavior:'instant'}));
   await page.screenshot({path:`${directory}/chosen-split-${width}.png`});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
   assert.deepEqual(errors,[]);assert.deepEqual(failedRequests,[]);
   results.push({width,values,errors,failedRequests});await page.close();
  }
  assert.deepEqual(hashes(),sourceHashes);
  fs.writeFileSync('docs/teaching/evidence/dp-state-families-select-label-browser.json',JSON.stringify({checkedAt:new Date().toISOString(),sourceHashes,results},null,2)+'\n');
  console.log(JSON.stringify(results.map(result=>({width:result.width,selectedStates:result.values.length}))));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
