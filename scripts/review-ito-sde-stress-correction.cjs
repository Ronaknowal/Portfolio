const { chromium }=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});const results=[];
  try{
    for(const width of [1440,390,320]){
      const page=await browser.newPage({viewport:{width,height:1000}});await page.routeWebSocket('**',socket=>socket.close());
      const errors=[],warnings=[],failed=[];
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',message=>{if(['warning','error'].includes(message.type())&&!message.text().startsWith('[vite]'))warnings.push(message.text());});
      page.on('requestfailed',request=>failed.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/it-calculus-stochastic-differential-equations?module=math-foundations',{waitUntil:'networkidle'});
      await page.evaluate(()=>document.fonts.ready);
      const solver=page.getByRole('region',{name:'Coupled SDE solver investigation',exact:true});
      for(const originalNoise of ['0','0.3','0.6','1']){
        await solver.getByLabel('Solver coarse steps',{exact:true}).selectOption({value:'1'});
        await solver.getByLabel('Solver noise scale',{exact:true}).selectOption({value:originalNoise});
        await solver.getByLabel('Solver fixture',{exact:true}).selectOption({value:'stress'});
        const steps=solver.getByLabel('Solver coarse steps',{exact:true});
        const sigma=solver.getByLabel('Solver noise scale',{exact:true});
        assert.ok(await steps.isDisabled());assert.ok(await sigma.isDisabled());
        assert.equal(await steps.inputValue(),'1');assert.equal(await steps.locator('option:checked').innerText(),'1');
        assert.equal(await sigma.inputValue(),'1');assert.equal(await sigma.locator('option:checked').innerText(),'σ=1');
        assert.ok((await solver.innerText()).includes('new: -0.6'));
        if(originalNoise==='0.3'){
          await solver.evaluate(element=>window.scrollTo(0,scrollY+element.getBoundingClientRect().top-175));
          await page.screenshot({path:`scratch/ito-sde-browser/stress-active-controls-reading-${width}.png`});
          await page.addStyleTag({content:'.learn-nav{visibility:hidden!important}'});
          await solver.screenshot({path:`scratch/ito-sde-browser/stress-active-controls-${width}.png`});
          await page.addStyleTag({content:'.learn-nav{visibility:visible!important}'});
        }
        await solver.getByLabel('Solver fixture',{exact:true}).selectOption({value:'seeded'});
        assert.equal(await steps.locator('option:checked').innerText(),'128');
        assert.equal(await sigma.inputValue(),originalNoise);
      }
      await solver.getByLabel('Solver fixture',{exact:true}).selectOption({value:'stress'});
      await solver.getByRole('button',{name:'Reset coupled solvers',exact:true}).focus();await page.keyboard.press('Enter');
      assert.equal(await solver.getByLabel('Solver coarse steps',{exact:true}).locator('option:checked').innerText(),'16');
      assert.equal(await solver.getByLabel('Solver noise scale',{exact:true}).inputValue(),'0.6');
      const fonts=await page.evaluate(()=>[...document.fonts].some(font=>font.family==='Space Grotesk'&&font.status==='loaded'));
      assert.ok(fonts);assert.deepEqual(errors,[]);assert.deepEqual(warnings,[]);assert.deepEqual(failed,[]);
      results.push({width,fonts,checkedSeededNoiseValues:4,stressStepDisplay:'1',stressSigmaDisplay:'1',seededValuesRestored:true,keyboardReset:true,errors,warnings,failed});
      await page.close();
    }
  }finally{await browser.close();}
  const record={checkedAt:new Date().toISOString(),results};
  fs.writeFileSync('scratch/ito-sde-browser/stress-correction-results.json',JSON.stringify(record,null,2));console.log(JSON.stringify(record,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
