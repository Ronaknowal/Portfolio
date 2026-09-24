const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/concentration-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const model = await import(pathToFileURL(path.resolve('src/learn/data/concentration-inequalities-models.js')));
  const { concentrationInequalityExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/concentration-inequalities-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440,390]) {
      const page = await browser.newPage({ viewport:{width,height:1000}, reducedMotion:'reduce' });
      await page.routeWebSocket('**', socket=>socket.close());
      page.on('pageerror',error=>errors.push(error.message));
      await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173')+'/learn/path/full-curriculum/concentration-inequalities-hoeffding-bernstein-chernoff?module=mathematical-statistical-foundations');
      const lesson=page.locator('.concentration-lesson'); await lesson.waitFor();
      const record={width,anchors:[],tailCases:0,samplingCases:0,budgetCases:0,familyCases:0,screenshots:[]};
      const fact=(region,name)=>region.locator('.concentration-facts > div').filter({has:page.locator('dt',{hasText:name})}).locator('dd');
      const shot=async(locator,name)=>{
        await locator.evaluate(element=>window.scrollTo(0,element.getBoundingClientRect().top+window.scrollY-90));
        const filename=`${name}-${width}.png`; await page.screenshot({path:path.join(directory,filename)}); record.screenshots.push(filename);
      };
      const keyButton=async(region,name,key='Enter')=>{await region.getByRole('button',{name,exact:true}).focus();await page.keyboard.press(key);};
      const slider=async(region,name,target)=>{
        const input=region.getByRole('slider',{name,exact:true});
        await input.focus(); await page.keyboard.press('Home');
        const minimum=Number(await input.getAttribute('min'));
        for(let value=minimum;value<target;value++) await page.keyboard.press('ArrowRight');
        assert.equal(await input.inputValue(),String(target));
      };
      for(const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()){
        const href=await anchor.getAttribute('href'); await anchor.focus();await page.keyboard.press('Enter');
        await page.waitForFunction(hash=>location.hash===hash,href);await page.waitForTimeout(80);
        const target=page.locator(`[id="${href.slice(1)}"]`),box=await target.boundingBox();
        assert(box&&box.y>=45&&box.y<200,`${href}: ${box?.y}`);record.anchors.push({href,top:box.y});
        await shot(target,`reading-${record.anchors.length}`);
      }
      assert.equal(record.anchors.length,8);
      const tail=lesson.getByRole('region',{name:'Exact count tails and bounds',exact:true});
      for(const [n,p,k] of [[100,10,20],[20,25,0],[20,25,5],[20,25,10],[20,25,21],[100,0,1],[100,100,100],[100,100,101],[200,90,1],[200,2,200]]){
        await tail.getByLabel('Independent observations n').selectOption(String(n));await tail.getByLabel('True success probability p').selectOption(String(p));
        await slider(tail,'Inclusive count threshold',k);
        const state=model.countTail(n,p,k);
        assert.equal(await fact(tail,'Actual finite-law probability').innerText(),model.formatLogProbability(state.exactLog,state.exactComplementLog));
        assert.equal(await tail.locator('svg rect').count(),n+1);
        const boundCells=await tail.locator('tbody td').allTextContents();
        assert.deepEqual(boundCells,Object.values(state.bounds).map(value=>model.formatLogProbability(value)));
        record.tailCases++;
        if(n===200&&p===90) await shot(tail.locator('.concentration-facts'),'near-one-probability');
      }
      await keyButton(tail,'Reset tail comparison','Space');assert.equal(await tail.getByRole('slider').inputValue(),'8');await shot(tail.locator('figure'),'tail-default');
      const witness=lesson.getByRole('region',{name:'Exponential Markov witness',exact:true});
      const lambda=witness.getByRole('slider'); await lambda.focus();await page.keyboard.press('Home');
      assert.equal(await fact(witness,'Selected raw upper bound').innerText(),'1');
      await page.keyboard.press('End');assert.equal(await lambda.inputValue(),'2');
      await keyButton(witness,'Use exact optimum');assert(Math.abs(Number(await lambda.inputValue())-Math.log(3))<1e-12);
      await witness.locator('summary').focus();await page.keyboard.press('Enter');assert.equal(await witness.locator('tbody tr').count(),7);
      await shot(witness.locator('figure'),'witness-optimum');
      await shot(witness.locator('details'),'witness-contributions');
      await keyButton(witness,'Reset exponential witness');assert.equal(await lambda.inputValue(),'1');
      const budget=lesson.getByRole('region',{name:'Sample and variance budgets',exact:true});
      for(const [n,delta,w,v,eps] of [[100,.05,1,.09,.1],[100,.05,1,0,.1],[400,.01,1,.25,.05],[1000,.005,10,.01,.01]]){
        for(const [name,value] of [['Observation count n',n],['Failure budget δ',delta],['Known range width',w],['Variance upper bound / width²',v],['Absolute tolerance ε',eps]]) await budget.getByLabel(name).selectOption(String(value));
        const expected=model.precisionBudget(n,delta,w,v,eps);
        assert(Math.abs(Number(await fact(budget,'Bernstein quadratic radius').innerText())-expected.bernsteinExact)<1e-5);
        assert.equal(await fact(budget,'Sufficient Hoeffding sample count').innerText(),expected.needed.toLocaleString('en-US'));record.budgetCases++;
      }
      await keyButton(budget,'Reset precision budget','Space');await shot(budget.locator('figure'),'budget-default');
      const sampling=lesson.getByRole('region',{name:'Sampling dependence comparison',exact:true});
      for(const n of [1,10,20]) for(const eps of [20,80]){
        await sampling.getByLabel('Number of observations n').selectOption(String(n));await slider(sampling,'Absolute tolerance in percentage points',eps);
        const expected=model.samplingComparison(n,eps);
        const shown=await fact(sampling,'Exact finite-law event probability').allTextContents();
        assert.deepEqual(shown,expected.laws.map(law=>model.formatLogProbability(law.logFailure)));record.samplingCases++;
      }
      await keyButton(sampling,'Reset sampling comparison');
      for(let index=0;index<3;index++) await shot(sampling.locator('.concentration-law-comparison > figure').nth(index),`sampling-${index+1}`);
      const family=lesson.getByRole('region',{name:'Simultaneous error budget',exact:true});
      for(const relation of ['independent','identical']) for(const k of [1,100]){
        await family.getByLabel('Relation between check failures').selectOption(relation);await slider(family,'Number of fixed checks',k);
        const expected=model.familyBudget(100,k,.05,relation);
        assert.equal(await fact(family,'Actual allocated family failure').innerText(),model.formatLogProbability(expected.allocatedFamilyLog));record.familyCases++;
      }
      await keyButton(family,'Reset family budget','Space');await shot(family.locator('figure'),'family-default');
      for(const proof of await lesson.locator('.concentration-proof').all()){
        await proof.locator('summary').focus();await page.keyboard.press('Enter');assert.equal(await proof.getAttribute('open'),'');await shot(proof,`proof-${record.screenshots.length}`);
      }
      const displayed=await lesson.locator('.python-example').all(); assert.equal(displayed.length,9);
      assert.equal(await lesson.locator('.concentration-program > p').count(),9);
      for(const element of displayed){
        const title=await element.locator('h3').innerText();const example=Object.values(examples).find(value=>value.title===title);assert(example,title);
        const blocks=await element.locator(':scope > div').all(); assert.equal(blocks.length,2);
        assert(normalize(await blocks[0].innerText()).includes(normalize(example.code)),title+' code');
        assert.equal(normalize(await blocks[1].innerText()).replace(/^OUTPUT /,''),normalize(example.expected),title+' output');
      }
      const checkpoints=await lesson.locator('.lesson-check').all();assert.equal(checkpoints.length,9);
      for(const checkpoint of checkpoints){
        for(const detail of await checkpoint.locator('details').all())assert.equal(await detail.getAttribute('open'),null);
        const hints=checkpoint.locator('.concentration-hint');
        if(await hints.count()){
          await hints.locator('summary').focus();await page.keyboard.press('Enter');
          assert.equal(await checkpoint.locator('.concentration-solution').getAttribute('open'),null);
          assert((await hints.innerText()).length>80);
          await checkpoint.locator('.concentration-solution summary').focus();await page.keyboard.press('Space');
          assert((await checkpoint.locator('.concentration-solution').innerText()).length>150);
        }else{await checkpoint.locator('summary').focus();await page.keyboard.press('Enter');assert((await checkpoint.locator('details').innerText()).length>150);}
      }
      record.independentHints=await lesson.locator('.concentration-hint[open]').count();assert.equal(record.independentHints,8);
      assert.equal(await lesson.locator('.lesson-sources a').count(),5);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      record.mathBounds=await lesson.locator('.katex-display').evaluateAll(elements=>elements.map(element=>({width:element.clientWidth,scroll:element.scrollWidth,text:element.textContent.slice(0,70)})));
      assert(record.mathBounds.every(item=>item.scroll<=item.width+2),JSON.stringify(record.mathBounds));
      record.svgTextOverflow=await lesson.locator('svg text').evaluateAll(elements=>elements.flatMap(element=>{const text=element.getBBox(),box=element.ownerSVGElement.viewBox.baseVal;return text.x<-.5||text.x+text.width>box.width+.5?[{text:element.textContent,x:text.x,width:text.width}]:[]}));
      assert.deepEqual(record.svgTextOverflow,[]);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      records.push(record);await page.close();
    }
    assert.deepEqual(errors,[]);
    const result={checkedAt:new Date().toISOString(),records,errors};
    fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
