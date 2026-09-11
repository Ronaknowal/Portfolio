const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/ensemble-methods/browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const m = await import('../src/learn/data/ensemble-methods-models.js');
  const { ensembleExamples: examples } = await import('../src/learn/data/ensemble-methods-examples.js');
  const map = JSON.parse(fs.readFileSync('src/learn/data/ensemble-prediction-map.json'));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const attachments = process.argv.includes('--attachments');
  const reading = process.argv.includes('--reading') || attachments;
  try {
    if (process.argv.includes('--probe')) {
      const page = await browser.newPage();
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/ensemble-methods-stacking', { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.waitForTimeout(3000);
      console.log(await page.evaluate(() => ({ overlay: document.querySelector('vite-error-overlay')?.shadowRoot?.textContent, lesson: !!document.querySelector('.ensemble-lesson') })));
      return;
    }
    const widths = attachments ? [390] : process.argv.includes('--320') ? [320] : [1440, 390, 320];
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/ensemble-methods-stacking', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.ensemble-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      let states = 0;
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.waitForTimeout(80);
        await page.screenshot({ path: path.join(directory, name + '-' + width + '.png') });
      };
      const lab = id => lesson.locator('[data-investigation="' + id + '"]');
      const slide = async (area, name, value) => { await area.getByRole('slider', { name, exact: true }).fill(String(value)); states += 1; };
      assert.equal(await lesson.locator('h2').count(), 13);
      assert.equal(await lesson.locator('[data-investigation]').count(), 6);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      await capture(lesson.locator('h2').first(), 'reading-start');

      if (!reading) {
      const vote=lab('ensemble-votes');
      assert((await vote.innerText()).includes('Mean delay probability: 0.3433'));
      await slide(vote,'Model A: probability of delay',.8);
      await slide(vote,'Model B: probability of delay',.4);
      await slide(vote,'Model C: probability of delay',.3);
      await slide(vote,'Relative weight of model C',2);
      assert((await vote.innerText()).includes('Mean delay probability: 0.45'));
      await slide(vote,'Model A: probability of delay',1);
      assert((await vote.innerText()).includes('Probability mean → on time'));
      const slider=vote.getByRole('slider',{name:'Model A: probability of delay',exact:true});
      await slider.focus();await page.keyboard.press('ArrowLeft');assert.equal(await slider.inputValue(),'0.99');states+=1;
      await vote.getByRole('button',{name:'Reset forecasts'}).focus();await page.keyboard.press('Enter');
      assert.equal(await slider.inputValue(),'0.51');

      const error=lab('ensemble-errors');
      for(const preset of ['complementary','copies','sameSide']) {
        await error.getByRole('combobox').selectOption(preset);
        for(const weight of [0,.2,.5,1]) {
          await slide(error,'Weight on forecast A',weight);
          const expected=m.errorBlendState(preset,weight);
          const text=await error.innerText();assert(text.includes(m.ensembleNumber(expected.mse)));
          const plots=await error.locator('svg').first().getAttribute('aria-label');assert(plots.includes('Shipment A'));
        }
      }
      await error.getByRole('combobox').selectOption('complementary');
      await slide(error,'Weight on forecast A',.2);
      if(width===390)await capture(error.locator('.ensemble-error-rows'),'changed-residuals');

      const bootstrap=lab('ensemble-bootstrap');
      for(const preset of ['repeated','allRows','smallBags']) {
        await bootstrap.getByRole('combobox',{name:'Draw preset'}).selectOption(preset);
        for(let bag=0;bag<3;bag+=1) {
          if(bag)await bootstrap.getByRole('button',{name:'Next step',exact:true}).click();
          await bootstrap.getByRole('combobox',{name:'Inspect outcome row'}).selectOption('0');
          const expected=m.bootstrapState(preset,bag,0);
          const tokens=await bootstrap.locator('.ensemble-draws li').allTextContents();
          assert.deepEqual(tokens,expected.draws[bag].map(i=>String.fromCharCode(65+i)));
          assert((await bootstrap.innerText()).includes('All-model mean: '+m.ensembleNumber(expected.ensemblePrediction)));
          if(preset==='allRows')assert((await bootstrap.innerText()).includes('unavailable — no eligible model'));
          states+=1;
        }
      }
      await bootstrap.getByRole('combobox',{name:'Draw preset'}).selectOption('repeated');
      await bootstrap.getByRole('button',{name:'Next step',exact:true}).click();
      assert((await bootstrap.innerText()).includes('Threshold: x ≤ 3.'));
      await capture(bootstrap.locator('.ensemble-draws'),'bootstrap-gap');

      const boost=lab('ensemble-boosting');
      for(const preset of ['mixed','perfect','contradiction']) {
        await boost.getByRole('combobox').selectOption(preset);
        const trace=m.signedBoostingTrace(preset);
        for(let index=0;index<trace.frames.length;index+=1) {
          if(index)await boost.getByRole('button',{name:'Next step',exact:true}).click();
          const frame=trace.frames[index];
          assert((await boost.innerText()).includes('Weighted mistake mass ε = '+m.ensembleNumber(frame.stump.error,6)));
          if(frame.status==='accepted')assert((await boost.innerText()).includes('α = ½ log[(1−ε)/ε] = '+m.ensembleNumber(frame.alpha,6)));
          else assert((await boost.innerText()).includes(frame.status==='perfect'?'Perfect weak learner: stop.':'No positive edge: stop.'));
          states+=1;
        }
      }
      await boost.getByRole('combobox').selectOption('mixed');
      await boost.getByRole('button',{name:'Next step',exact:true}).click();
      await capture(boost.locator('figure'),'boosting-second-fit');
      const scroll=boost.locator('.ensemble-plot-scroll');
      await scroll.focus();await page.keyboard.press('ArrowRight');await page.waitForTimeout(180);
      if(width<600)assert(await scroll.evaluate(node=>node.scrollLeft>0));
      await boost.getByRole('button',{name:'Back',exact:true}).click();assert((await boost.innerText()).includes('0.804719'));

      const oof=lab('ensemble-oof');
      for(const mode of ['honest','leaky']) {
        await oof.getByRole('combobox').selectOption(mode);
        await oof.getByRole('button',{name:'Back',exact:true}).click();
        assert((await oof.innerText()).includes('No folds filled yet'));
        assert((await oof.innerText()).includes('not available'));
        for(let fold=1;fold<=3;fold+=1) {await oof.getByRole('button',{name:'Next step',exact:true}).click();states+=1;}
        const expected=m.oofOwnershipState(mode,3,2.5);
        assert((await oof.innerText()).includes('Learned convex weight on nearest neighbor: '+m.ensembleNumber(expected.weightNearest,6)));
        for(const query of [0,.5,2.5,5]) {
          await slide(oof,'New input x',query);
          assert((await oof.innerText()).includes(m.ensembleNumber(m.oofOwnershipState(mode,3,query).ensemble)));
        }
      }
      await oof.getByRole('combobox').selectOption('honest');
      await oof.getByRole('button',{name:'Next step',exact:true}).click();await oof.getByRole('button',{name:'Next step',exact:true}).click();
      await capture(oof.locator('.ensemble-ownership'),'oof-completed');

      const boundaries=lab('ensemble-boundaries');
      for(const name of Object.keys(map.models)) {
        await boundaries.getByRole('combobox').selectOption(name);
        for(const [x,y] of [[0,24],[12,12],[16,12],[24,0]]) {
          await slide(boundaries,'Horizontal grid index',x);await slide(boundaries,'Vertical grid index',y);
          assert((await boundaries.locator('.ensemble-readout').innerText()).includes('P(class 1)='+m.ensembleNumber(map.models[name].probabilities[y*25+x],6)));
          const actual=await boundaries.locator('rect[data-probability]').evaluateAll(nodes=>nodes.map(node=>Number(node.dataset.probability)));
          assert.deepEqual(actual,map.models[name].probabilities);
        }
      }
      await boundaries.getByRole('button',{name:'Reset fitted-map view'}).click();
      await capture(boundaries.locator('figure'),'native-prediction-map');
      }

      // Open every teaching disclosure once, including the full original program and changed report.
      let opened=0;
      for(const detail of await lesson.locator('details').all()) {
        if(!await detail.evaluate(node=>node.open)) {await detail.locator(':scope > summary').click();opened+=1;}
      }
      const plain=await lesson.innerText();
      for(const example of examples) { assert(plain.includes(example.question),example.id+' question');assert(plain.includes(example.output),example.id+' output'); }
      const allCode=(await lesson.locator('.python-example').allTextContents()).join('\n');
      for(const example of examples)assert(allCode.includes(example.code.trim()),example.id+' complete code');
      assert.equal(await lesson.locator('section.lesson-check').count(),12);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const wideMath=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.flatMap(node=>node.scrollWidth>node.clientWidth+2?[{text:node.textContent,width:node.clientWidth,scroll:node.scrollWidth}]:[]));
      assert.deepEqual(wideMath,[],'Displayed equation exceeds available width');
      const overflow=await page.evaluate(()=>({window:innerWidth,document:document.documentElement.scrollWidth}));
      assert(overflow.document<=overflow.window+1,JSON.stringify(overflow));
      const invalid=await lesson.locator('input').evaluateAll(nodes=>nodes.filter(node=>!node.checkValidity()).map(node=>node.outerHTML));assert.deepEqual(invalid,[]);
      for(const [heading,name] of [[5,'bound-derivation'],[7,'api-reading'],[11,'context-reading']]) {
        if(width!==1440 || heading===5)await capture(lesson.locator('h2').nth(heading),name);
      }
      if(width===320) {await capture(lesson.locator('section.lesson-check').last(),'changed-report');await capture(lesson.locator('.lesson-sources'),'resources');}
      if(reading) for(const index of [1,4])await capture(lesson.locator('.katex-display').nth(index),'final-equation-'+index);
      if(attachments) {
        await capture(lesson.locator('.ensemble-calibration'),'calibration-law');
        await capture(lesson.locator('section.lesson-check').last().locator('.python-example > .lesson-note'),'changed-report-output');
      }
      assert.deepEqual(errors,[]);
      records.push({width,states,opened,mathCount:await lesson.locator('.katex-display').count(),wideMath,overflow,errors,fonts:true});
      fs.writeFileSync(path.join(directory,'width-'+width+(reading?'-reading':'')+'.json'),JSON.stringify({checkedAt:new Date().toISOString(),record:records.at(-1)},null,2)+'\n');
      await page.close();
    }
  } catch(error) {
    const current=browser.contexts().flatMap(context=>context.pages()).at(-1);
    const overlay=current ? await current.evaluate(()=>document.querySelector('vite-error-overlay')?.shadowRoot?.textContent).catch(()=>null) : null;
    fs.writeFileSync(path.join(directory,'interrupted.json'),JSON.stringify({at:new Date().toISOString(),records,error:String(error),overlay},null,2)+'\n');
    throw error;
  } finally {await browser.close();}
  fs.writeFileSync(path.join(directory,attachments?'attachments-results.json':reading?'reading-results.json':'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),records},null,2)+'\n');
  console.log(JSON.stringify(records,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
