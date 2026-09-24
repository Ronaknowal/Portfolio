const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const output = path.resolve('docs/teaching/projects/evidence/depth-revision');
const report = { passed: false, checks: [], screenshots: [], errors: [], checkedAt: new Date().toISOString() };
fs.mkdirSync(output, {recursive:true});
fs.writeFileSync(path.join(output,'depth-browser.json'),JSON.stringify(report,null,2));

async function open(page,stage) {
  await page.goto(`${base}/learn/projects/typed-decision-model/${stage}`,{waitUntil:'domcontentloaded'});
  await page.locator('.tdp-stage').waitFor();
  await page.evaluate(()=>document.fonts.ready);
}
async function screenshot(page,selector,name) {
  await page.locator(selector).scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(output,name)});
  report.screenshots.push(name);
}
async function bounds(page,label) {
  const result=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,clipped:[...document.querySelectorAll('.tdp-stage button,.tdp-stage input,.tdp-stage textarea,.tdp-stage select')].filter(element=>{const r=element.getBoundingClientRect();return r.width&&r.height&&(r.left < -1||r.right > innerWidth+1);}).map(element=>element.id||element.textContent)}));
  assert.ok(result.scroll<=result.width+1,label+' page overflow');
  assert.deepEqual(result.clipped,[],label+' clipped controls');
  report.checks.push({case:label,...result});
}
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const context=await browser.newContext({viewport:{width:1366,height:1000}});
    const page=await context.newPage();
    const errors=[]; page.on('pageerror',error=>errors.push(error.message));
    await open(page,'data');
    const trace=page.getByRole('region',{name:'Trace a request into tokens and candidate positions'});
    await trace.waitFor();
    assert.match(await trace.locator('.tdp-trace-summary').innerText(),/35 tokens.*8, 15, 22.*target index 0/);
    await trace.getByRole('button',{name:'Move first candidate to the end'}).click();
    assert.match(await trace.locator('.tdp-trace-summary').innerText(),/target index 2/);
    await trace.getByRole('button',{name:'Reset trace'}).click();
    await trace.getByRole('checkbox',{name:'Include Delivery'}).uncheck();
    assert.match(await trace.locator('.tdp-trace-summary').innerText(),/28 tokens.*8, 15/);
    await trace.getByRole('button',{name:'Try an unseen paraphrase'}).click();
    assert.ok(await trace.locator('.tdp-token-grid').getByText(/UNK/).count()>0);
    await trace.getByRole('textbox',{name:'Request text'}).fill('123');
    assert.match(await trace.getByRole('status').innerText(),/a–z/);
    await trace.getByRole('textbox',{name:'Request text'}).fill('word '.repeat(150));
    assert.match(await trace.getByRole('status').innerText(),/128/);
    await trace.getByRole('button',{name:'Reset trace'}).click();
    await screenshot(page,'.tdp-encoding-lab','encoding-desktop.png');
    report.checks.push({case:'Actual tokenizer, reordered target, variable candidates, unknown words and explicit rejection',passed:true});

    await open(page,'architecture');
    const attention=page.getByRole('region',{name:'Explore an attention row and padding mask'});
    const before=await attention.locator('.tdp-attention-output').innerText();
    await attention.getByRole('checkbox',{name:'Mask the padding key'}).uncheck();
    assert.notEqual(await attention.locator('.tdp-attention-output').innerText(),before);
    assert.match(await attention.innerText(),/deliberately disabled/);
    await attention.getByRole('button',{name:'Reset attention'}).click();
    await attention.getByRole('slider',{name:'First query coordinate'}).press('Home');
    assert.notEqual(await attention.locator('.tdp-attention-output').innerText(),before);
    await attention.getByRole('slider',{name:'First query coordinate'}).press('ArrowRight');
    assert.equal(await attention.getByRole('slider',{name:'First query coordinate'}).inputValue(),'-1.9');
    await screenshot(page,'.tdp-attention-lab','attention-desktop.png');
    report.checks.push({case:'Attention mask, reset and interior keyboard input change visible weights and output',passed:true});

    await open(page,'training');
    const training=page.getByRole('region',{name:'Train a shared scoring head one step at a time'});
    const initial=await training.locator('.tdp-step-loss').innerText();
    await training.getByRole('combobox',{name:'Correct candidate'}).selectOption('1');
    assert.notEqual(await training.locator('.tdp-step-loss').innerText(),initial);
    await training.getByRole('button',{name:'Apply one SGD step'}).click();
    assert.match(await training.innerText(),/1 step applied/);
    await training.getByRole('slider',{name:'SGD learning rate'}).press('Home');
    assert.equal(await training.getByRole('button',{name:'Apply one SGD step'}).isEnabled(),false);
    const losses=(await training.locator('.tdp-step-loss').innerText()).match(/\d+\.\d+/g);assert.equal(losses[0],losses[1]);
    await training.getByRole('button',{name:'Reset training'}).click();
    assert.equal(await training.locator('.tdp-step-loss').innerText(),initial);
    const rateSlider=training.getByRole('slider',{name:'SGD learning rate'});
    await rateSlider.scrollIntoViewIfNeeded();
    const rateBounds=await rateSlider.boundingBox();
    await page.mouse.move(rateBounds.x+rateBounds.width*.25,rateBounds.y+rateBounds.height/2);
    await page.mouse.down();
    await page.mouse.move(rateBounds.x+rateBounds.width*.72,rateBounds.y+rateBounds.height/2,{steps:8});
    await page.mouse.up();
    assert.ok(Number(await rateSlider.inputValue())>1);
    assert.notEqual(await training.locator('.tdp-step-loss').innerText(),initial);
    await training.getByRole('button',{name:'Reset training'}).click();
    await screenshot(page,'.tdp-training-lab','training-desktop.png');
    await page.setViewportSize({width:390,height:950});
    await screenshot(page,'.tdp-training-lab','training-mobile.png');
    report.checks.push({case:'Target-dependent gradient, real applied step, pointer drag, zero-rate null case and reset',passed:true});

    let sourceCount=0;
    for(const stage of ['define','data','baseline','architecture','training','calibration','evaluation','serving']) {
      await open(page,stage);
      assert.ok(await page.locator('.tdp-concept-links a').count()>=2,stage+' local concept links missing');
      const sources=page.locator('.tdp-source');
      for(let index=0;index<await sources.count();index++) {
        const source=sources.nth(index);
        await source.locator('summary').click();
        await source.locator('pre').waitFor();
        assert.ok((await source.locator('pre').innerText()).length>80);
        sourceCount++;
      }
      for(const width of [1366,390,320]) {
        await page.setViewportSize({width,height:950});
        await bounds(page,`${stage}, expanded source, ${width}px`);
      }
    }
    report.checks.push({case:'Every stage has relevant topic links and all canonical code excerpts load',sourceCount,passed:true});
    assert.deepEqual(errors,[]);
    report.passed=true;
    console.log(`PASS: ${report.checks.length} depth integration groups, ${sourceCount} source excerpts, four screenshot candidates.`);
    await context.close();
  }catch(error){report.errors.push(error.stack);console.error(error.stack);process.exitCode=1;}
  finally{await browser.close();fs.writeFileSync(path.join(output,'depth-browser.json'),JSON.stringify(report,null,2)+'\n');}
})();
