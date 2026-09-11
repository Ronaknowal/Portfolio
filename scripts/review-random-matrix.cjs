const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/random-matrix-browser');
fs.mkdirSync(directory,{recursive:true});
const normalize=value=>value.replace(/\s+/g,' ').trim();
(async()=>{
  const {randomMatrixExamples:examples}=await import(pathToFileURL(path.resolve('src/learn/data/random-matrix-examples.js')));
  const model=await import(pathToFileURL(path.resolve('src/learn/data/random-matrix-models.js')));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const records=[], errors=[];
  try {
    for(const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});
      await page.routeWebSocket('**',socket=>socket.close());
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',message=>{if(message.type()==='error'&&!message.text().includes('[vite]')) errors.push(message.text());});
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/random-matrix-theory?module=math-foundations');
      const lesson=page.locator('.random-matrix-lesson');
      await lesson.waitFor();
      await page.evaluate(()=>document.fonts.ready);
      const record={width,anchors:[],captures:[],programs:[],states:[],fonts:await page.evaluate(()=>[...document.fonts].filter(font=>font.status==='loaded').map(font=>font.family))};
      const shot=async(element,name)=>{
        await element.evaluate(node=>window.scrollTo({top:node.getBoundingClientRect().top+scrollY-85,behavior:'instant'}));
        await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
        await page.waitForTimeout(200);
        const file=name+'-'+width+'.png';
        await page.screenshot({path:path.join(directory,file)});
        record.captures.push(file);
      };
      const press=async(region,name)=>{
        const button=region.getByRole('button',{name,exact:true});
        await page.keyboard.press('Tab');
        await button.focus();
        assert(await button.evaluate(node=>document.activeElement===node&&getComputedStyle(node).outlineStyle!=='none'));
        await page.keyboard.press('Enter');
      };
      await shot(lesson.locator('.lesson-intro'),'ordinary-intro');
      for(const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href=await link.getAttribute('href');
        const target=lesson.locator('[id="'+href.slice(1)+'"]');
        assert.equal(await target.count(),1,'anchor exists '+href);
        await link.focus();await page.keyboard.press('Enter');
        await page.waitForFunction(id=>{const node=document.getElementById(id);return node&&node.getBoundingClientRect().top>=45&&node.getBoundingClientRect().top<=140;},href.slice(1));
        record.anchors.push(href);
        await shot(target,'ordinary-section-'+record.anchors.length);
      }
      for(const [index,figure] of (await lesson.locator('.rm-inline').all()).entries()) await shot(figure,'inline-'+index);
      for(const investigation of await lesson.locator('[data-investigation]').all()) await shot(investigation,(await investigation.getAttribute('data-investigation'))+'-initial');
      assert.equal(await lesson.locator('[data-investigation]').count(),5);
      for(const investigation of await lesson.locator('[data-investigation]').all()) await shot(investigation.locator('svg'),(await investigation.getAttribute('data-investigation'))+'-plot');
      const covariance=lesson.locator('[data-investigation="random-covariance"]');
      for(const shape of ['32,8','64,16','48,48','24,48']) for(const law of ['gaussian','sign']) for(const centered of [false,true]) {
        await covariance.getByLabel('Matrix shape',{exact:true}).selectOption(shape);
        await covariance.getByLabel('Entry law',{exact:true}).selectOption(law);
        await covariance.getByLabel('Subtract sample means',{exact:true}).setChecked(centered);
        const [rows,columns]=shape.split(',').map(Number);
        const expected=model.randomMatrixSpectrum({rows,columns,law,centered,seed:7});
        assert((await covariance.locator('[aria-live]').innerText()).includes('largest '+model.formatRandomMatrix(expected.largest)));
        assert((await covariance.locator('.rm-zero').innerText()).includes(expected.zeroCount+'/'+columns));
        record.states.push({lab:'covariance',shape,law,centered,zeroCount:expected.zeroCount});
      }
      await shot(covariance,'covariance-wide-centered');
      await press(covariance,'Draw another sample');
      assert((await covariance.innerText()).includes('Seed 8'));
      const redrawnCovariance=model.randomMatrixSpectrum({rows:24,columns:48,law:'sign',centered:true,seed:8});
      assert((await covariance.locator('[aria-live]').innerText()).includes('largest '+model.formatRandomMatrix(redrawnCovariance.largest)));
      await press(covariance,'Reset experiment');
      assert.equal(await covariance.getByLabel('Matrix shape',{exact:true}).inputValue(),'64,16');
      assert.equal(await covariance.getByLabel('Entry law',{exact:true}).inputValue(),'gaussian');
      assert.equal(await covariance.getByLabel('Subtract sample means',{exact:true}).isChecked(),false);
      const finite=lesson.locator('[data-investigation="random-finite-calibration"]');
      for(const nullModel of ['iid','duplicate']) for(const observedModel of ['iid','duplicate','spike']) {
        await finite.getByLabel('Null generation',{exact:true}).selectOption(nullModel);
        await finite.getByLabel('Observed model',{exact:true}).selectOption(observedModel);
        const expected=model.matrixNullCalibration(nullModel,observedModel);
        assert((await finite.locator('[aria-live]').innerText()).includes(expected.exceedances+'/59'));
        record.states.push({lab:'finite',nullModel,observedModel,score:expected.rankScore});
      }
      await shot(finite,'finite-matched-null');await press(finite,'Reset comparison');
      const spike=lesson.locator('[data-investigation="random-spike"]');
      for(const columns of ['16','32']) {
        await spike.getByLabel('Features with 64 observations',{exact:true}).selectOption(columns);
        const slider=spike.getByRole('slider',{name:'Population eigenvalue ℓ',exact:true});
        for(const [key,population] of [['Home',1],['End',5]]) {
          await slider.focus();await page.keyboard.press(key);
          const expected=model.randomMatrixSpectrum({rows:64,columns:Number(columns),seed:7,spike:population});
          assert((await spike.locator('[aria-live]').innerText()).includes('actual sample value '+model.formatRandomMatrix(expected.largest)));
          record.states.push({lab:'spike',columns,population});
        }
      }
      await shot(spike,'spike-high');await press(spike,'Draw another sample');
      assert((await spike.locator('[aria-live]').innerText()).includes('actual sample value '+model.formatRandomMatrix(model.randomMatrixSpectrum({rows:64,columns:32,spike:5,seed:8}).largest)));
      await press(spike,'Reset experiment');
      const wigner=lesson.locator('[data-investigation="random-wigner"]');
      for(const size of ['12','32','48']) for(const law of ['gaussian','sign']) for(const scaled of [false,true]) {
        await wigner.getByLabel('Symmetric matrix size',{exact:true}).selectOption(size);
        await wigner.getByLabel('Symmetric entry law',{exact:true}).selectOption(law);
        await wigner.getByLabel('Divide matrix by square root of size',{exact:true}).setChecked(scaled);
        const expected=model.wignerSpectrum(Number(size),7,law);
        assert((await wigner.locator('[aria-live]').innerText()).includes(model.formatRandomMatrix(expected.secondMoment*(scaled?1:Number(size)))));
        record.states.push({lab:'wigner',size,law,scaled});
      }
      await shot(wigner,'wigner-sign');await press(wigner,'Draw another sample');
      assert.equal(normalize(await wigner.locator('.rm-values').textContent()),normalize(model.wignerSpectrum(48,8,'sign').values.map(value=>model.formatRandomMatrix(value)).join(', ')));
      await press(wigner,'Reset experiment');
      const gap=lesson.locator('[data-investigation="random-level-gap"]');
      const difference=gap.getByRole('slider',{name:'Diagonal difference d',exact:true});
      const coupling=gap.getByRole('slider',{name:'Coupling c',exact:true});
      await coupling.focus();await page.keyboard.press('Home');
      assert((await gap.locator('[aria-live]').innerText()).includes('Current gap 0;'));
      await shot(gap,'gap-crossing');
      await difference.focus();await page.keyboard.press('End');
      await coupling.focus();await page.keyboard.press('End');
      assert((await gap.locator('[aria-live]').innerText()).includes('Current gap 5;'));
      await shot(gap,'gap-maximum');await press(gap,'Reset levels');
      for(const example of Object.values(examples)) {
        const card=lesson.locator('.python-example').filter({has:page.getByText(example.title,{exact:true})});
        assert.equal(await card.count(),1,example.title);
        const preceding=await card.evaluate(node=>node.previousElementSibling.textContent);
        assert(normalize(preceding).includes(normalize(example.question)),example.title+' visible prompt');
        const blocks=card.locator(':scope > div');
        assert(normalize(await blocks.nth(0).innerText()).includes(normalize(example.code)),example.title+' code');
        assert.equal(normalize(await blocks.nth(1).innerText()).replace(/^OUTPUT /,''),normalize(example.expected),example.title+' output');
        const summary=card.locator('summary').filter({hasText:/output/i});
        if(await summary.count()) {await summary.focus();await page.keyboard.press('Enter');}
        assert(normalize(await card.innerText()).includes(normalize(example.expected)),example.title+' output');
        record.programs.push(example.title);
      }
      for(const practice of await lesson.locator('.rm-practice').all()) {
        for(const summary of await practice.locator('summary').all()) {await summary.focus();await page.keyboard.press('Enter');}
        assert((await practice.innerText()).length>250);
      }
      await shot(lesson.locator('.rm-practice').nth(6),'practice-changed-program');
      const geometry=await lesson.evaluate(root=>{
        const outside=[];
        for(const node of root.querySelectorAll('.katex-display,.rm-inline,.rm-plot,.rm-controls,.rm-practice')) {
          const bounds=node.getBoundingClientRect();
          if(bounds.left<0||bounds.right>innerWidth+1||node.scrollWidth>node.clientWidth+2) outside.push({tag:node.tagName,className:node.className,width:bounds.width,scroll:node.scrollWidth,formula:node.querySelector('annotation')?.textContent});
        }
        const svgLabels=[];
        for(const svg of root.querySelectorAll('.rm-plot')) for(const node of svg.querySelectorAll('text')) {
          const bounds=node.getBBox();if(bounds.x<0||bounds.x+bounds.width>361) svgLabels.push({text:node.textContent,x:bounds.x,width:bounds.width});
        }
        return {outside,svgLabels,math:root.querySelectorAll('.katex-display').length,bodyOverflow:document.documentElement.scrollWidth>innerWidth+1};
      });
      record.geometry=geometry;
      records.push(record);
      fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),records,errors},null,2));
      assert.deepEqual(geometry.outside,[],'visible content fits at '+width);
      assert.deepEqual(geometry.svgLabels,[],'SVG label fit '+width);
      assert(!geometry.bodyOverflow);
      await page.close();
    }
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,widths:records.map(record=>record.width),states:records.map(record=>record.states.length),errors}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
