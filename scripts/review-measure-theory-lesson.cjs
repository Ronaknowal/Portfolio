const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const output = path.resolve('scratch/measure-theory-browser');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const model = await import('../src/learn/data/measure-theory-models.js');
  const { measureTheoryExamples } = await import('../src/learn/data/measure-theory-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [], warnings = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', event => {
        if (event.type() === 'error') errors.push(event.text());
        if (event.type() === 'warning') warnings.push(event.text());
      });
      page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/measure-theory-probability-spaces');
      const lesson = page.locator('.measure-lesson');
      await lesson.locator('h2').last().waitFor();
      const lab = name => lesson.locator('[data-measure-lab="' + name + '"]');
      const metric = (region, label) => region.locator('.measure-metrics > div').filter({ has: page.locator('dt', { hasText: label }) }).locator('dd');
      async function range(region, label, value) {
        await region.getByRole('slider', { name: label, exact: true }).fill(String(value));
      }
      async function capture(region, filename) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await region.screenshot({ path: path.join(output, filename + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      if (width !== 320 && !process.env.MEASURE_READING_ONLY) {
        const events = lab('events');
        for (const partition of Object.keys(model.INFORMATION_PARTITIONS)) {
          await events.getByRole('combobox', { name: 'Observed information' }).selectOption(partition);
          for (const mask of [0, 3, 7, 21, 42, 48, 63]) {
            await events.getByRole('button', { name: 'Empty event', exact: true }).click();
            for (let face = 1; face <= 6; face++) {
              if (mask & (1 << (face-1))) await events.getByRole('button', { name: 'Face ' + face, exact: true }).click();
            }
            const expected = model.informationState(partition, mask);
            assert.equal(await metric(events, 'Observable from this information?').innerText(), expected.observable ? 'Yes: union of whole cells' : 'No: at least one cell is split');
            assert.equal(await metric(events, 'Events in this information algebra').innerText(), String(expected.observableEventCount));
            states++;
          }
        }
        await events.getByRole('combobox').selectOption('pairs');
        await events.getByRole('button', { name: 'Faces 1, 2, 3', exact: true }).click();
        await capture(events, 'events-split');
        for (const threshold of [0, 1, 2, 3]) {
          await range(lab('preimage'), 'Output threshold t', threshold);
          const expected = model.preimageState(threshold);
          assert.equal(await metric(lab('preimage'), 'Its probability').innerText(), expected.preimage.length + '/6 = ' + model.measureNumber(expected.probability));
          states++;
        }
        await range(lab('preimage'), 'Output threshold t', 1);
        await capture(lab('preimage'), 'preimage');
        for (const w of [0, .25, .5, 1]) {
          await range(lab('mixture'), 'Atomic probability w', w);
          for (const [key, interval] of Object.entries({ half: [0,.5], point:[0,0], positive:[.25,.75], all:[0,1], outside:[-.25,-.1] })) {
            await lab('mixture').getByRole('combobox').selectOption(key);
            assert.equal(await metric(lab('mixture'), 'Total event probability').innerText(), model.measureNumber(model.mixedMeasureState(w, ...interval).probability));
            states++;
          }
        }
        await range(lab('mixture'), 'Atomic probability w', .25);
        await lab('mixture').getByRole('combobox').selectOption('point');
        await capture(lab('mixture'), 'mixture-point');
        for (let level = 0; level <= 6; level++) {
          await range(lab('simple'), 'Refinement n', level);
          assert.equal(await metric(lab('simple'), 'Lower simple integral').innerText(), model.measureNumber(model.simpleIntegralState(level).lowerIntegral, 6));
          states++;
        }
        await range(lab('simple'), 'Refinement n', 2);
        await capture(lab('simple'), 'simple');
        for (const mode of Object.keys(model.LIMIT_MODES)) {
          await lab('limits').getByRole('combobox').selectOption(mode);
          for (const n of [1, 4, 16, 64]) {
            await range(lab('limits'), 'Sequence index n', n);
            for (const x of [0, .25, 1]) {
              await range(lab('limits'), 'Fixed observation x', x);
              const expected = model.limitIntegralState(mode, n, x);
              assert.equal(await metric(lab('limits'), 'Current integral').innerText(), model.measureNumber(expected.integral));
              assert.equal(await metric(lab('limits'), 'Its pointwise limit').innerText(), model.measureNumber(expected.pointwiseLimit));
              states++;
            }
          }
          await range(lab('limits'), 'Sequence index n', 4);
          await range(lab('limits'), 'Fixed observation x', .25);
          await capture(lab('limits'), 'limit-' + mode);
        }
        for (const [column, row, a, b] of [[2,1,2,3],[0,0,.5,.5],[3,3,3,3],[0,3,3,.5],[3,0,.5,3]]) {
          const region = lab('joint');
          await range(region, 'Cell column, from left', column);
          await range(region, 'Cell row, from bottom', row);
          await range(region, 'Horizontal scale a', a);
          await range(region, 'Vertical scale b', b);
          assert.equal(await metric(region, 'Unchanged cell probability').innerText(), model.measureNumber(model.jointCellState(column,row,a,b).probability));
          states++;
        }
        await range(lab('joint'), 'Cell column, from left', 2);
        await range(lab('joint'), 'Cell row, from bottom', 1);
        await range(lab('joint'), 'Horizontal scale a', 2);
        await range(lab('joint'), 'Vertical scale b', 3);
        await capture(lab('joint'), 'joint');
        const conditional = lab('conditional');
        for (const partition of Object.keys(model.INFORMATION_PARTITIONS)) {
          await conditional.getByRole('combobox', { name: 'Prediction information' }).selectOption(partition);
          for (const sampling of ['fair', 'missing-six']) {
            await conditional.getByRole('combobox', { name: 'Outcome probabilities' }).selectOption(sampling);
            assert.equal(await metric(conditional, 'Mean squared residual').innerText(), model.measureNumber(model.conditionalMeanState(partition,sampling).risk));
            states++;
          }
        }
        assert(await conditional.getByRole('slider', { name: 'Chosen prediction on the null cell' }).isVisible());
        await range(conditional, 'Chosen prediction on the null cell', 20);
        assert.equal(await metric(conditional, 'Mean squared residual').innerText(), '0');
        await capture(conditional, 'conditional-null');
        for (const invalid of ['1,2', '1,2,3,4,5,', '1,2,3,4,5,Infinity', '1,2,3,4,5,101']) {
          await conditional.getByRole('textbox').fill(invalid);
          await conditional.getByRole('button', { name: 'Apply losses' }).click();
          assert(await conditional.getByRole('alert').isVisible());
          assert.equal(await metric(conditional, 'Mean squared residual').innerText(), '0');
          states++;
        }
        await capture(conditional, 'conditional-invalid');
        await conditional.getByRole('textbox').fill('1, 3, 2, 8, 7, 9');
        await conditional.getByRole('button', { name: 'Apply losses' }).click();
        assert.equal(await conditional.getByRole('alert').count(), 0);
        await conditional.getByRole('combobox', { name: 'Prediction information' }).selectOption('pairs');
        await conditional.getByRole('combobox', { name: 'Outcome probabilities' }).selectOption('fair');
        assert.equal(await metric(conditional, 'Mean squared residual').innerText(), model.measureNumber(model.conditionalMeanState('pairs','fair',[1,3,2,8,7,9]).risk));
        await capture(conditional, 'conditional-changed');
        for (const source of ['fair','missing-six']) {
          for (const target of Object.keys(model.TARGET_MEASURES)) {
            await lab('ratios').getByRole('combobox',{ name:'Source probability P' }).selectOption(source);
            await lab('ratios').getByRole('combobox',{ name:'Target probability Q' }).selectOption(target);
            const expected = model.reweightState(source,target);
            assert.equal(await metric(lab('ratios'), 'Reweighted source expectation').innerText(), expected.supported ? model.measureNumber(expected.weightedMean) : 'Identity unavailable');
            states++;
          }
        }
        await capture(lab('ratios'), 'ratio-support');
      }
      const visited = [];
      for (const name of ['events','preimage','mixture','simple','limits','joint','conditional','ratios']) {
        const controls = lab(name).locator('input:enabled,select:enabled,button:enabled');
        await controls.first().focus();
        for (let index = 0; index < await controls.count(); index++) {
          const control = controls.nth(index);
          assert(await control.evaluate(node => node === document.activeElement));
          const style = await control.evaluate(node => ({ outline:getComputedStyle(node).outlineStyle, width:getComputedStyle(node).outlineWidth, height:node.getBoundingClientRect().height }));
          assert.notEqual(style.outline,'none');assert(parseFloat(style.width)>0);assert(style.height>=43);
          visited.push(style);await page.keyboard.press('Tab');
        }
      }
      const keyboardSlider = lab('simple').getByRole('slider');
      await keyboardSlider.focus();await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');
      assert.equal(await keyboardSlider.inputValue(),'1');
      for (let index=0;index<2;index++) {
        const checkpoint=lesson.locator('.lesson-check').nth(index);
        assert((await checkpoint.locator('p').first().innerText()).length>80);
        await checkpoint.locator('summary').focus();await page.keyboard.press('Enter');
        assert((await checkpoint.locator('details > div').innerText()).length>150);
      }
      for (let index=0;index<10;index++) {
        const heading = lesson.locator('h2').nth(index);
        const anchor = lesson.locator('.lesson-intro nav a').nth(index);
        assert.equal((await anchor.getAttribute('href')).slice(1), await heading.getAttribute('id'));
        await heading.evaluate(node=>window.scrollTo(0,node.getBoundingClientRect().top+scrollY-100));
        await page.screenshot({path:path.join(output,'reading-'+(index+1)+'-'+width+'.png')});
      }
      for (let index=0;index<3;index++) await capture(lesson.locator('.measure-inline').nth(index),'inline-'+index);
      for (const example of Object.values(measureTheoryExamples)) {
        const block = lesson.locator('.python-example').filter({has:page.locator('h3',{hasText:example.title})});
        assert.equal(await block.count(),1);
        assert((await block.innerText()).includes(example.code));
        assert((await block.innerText()).includes(example.expected));
        const question=await block.evaluate(node=>node.previousElementSibling.textContent);
        assert(question.includes(example.question));
      }
      assert.equal(await lesson.locator('.measure-practice').count(),9);
      for (let index=0;index<9;index++) {
        const task=lesson.locator('.measure-practice').nth(index);
        for(const summary of await task.locator('summary').all()) {await summary.focus();await page.keyboard.press('Enter');}
        assert.equal(await task.locator('details[open]').count(),2);
        assert((await task.locator('details').last().innerText()).length>150);
      }
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>node.open=true));
      const maths=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map((node,index)=>({index,width:node.clientWidth,scroll:node.scrollWidth,text:node.textContent.slice(0,100)})));
      const overflowMath=maths.filter(row=>row.scroll>row.width+2);
      const clippedLabels=await lesson.locator('.measure-plot,.measure-cell-plot').evaluateAll(nodes=>nodes.flatMap((svg,index)=>{
        const box=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].filter(node=>{const b=node.getBoundingClientRect();return b.left<box.left-1||b.right>box.right+1||b.top<box.top-1||b.bottom>box.bottom+1;}).map(node=>({index,text:node.textContent}));
      }));
      let keyboardTables=0;
      for(const table of await lesson.locator('.lesson-table-wrap').all()){
        if(await table.evaluate(node=>node.scrollWidth>node.clientWidth+1)){
          await table.focus();for(let index=0;index<6;index++)await page.keyboard.press('ArrowRight');
          await page.waitForFunction(node=>node.scrollLeft>0,await table.elementHandle());keyboardTables++;
        }
      }
      const pageOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      const sources=lesson.locator('.lesson-sources');
      const resourceLinks=await sources.locator('a').evaluateAll(nodes=>nodes.map(node=>({text:node.textContent,url:node.href})));
      assert(resourceLinks.length>=6);
      await capture(sources,'sources');
      await lesson.locator('.measure-practice').nth(5).evaluate(node=>window.scrollTo(0,node.getBoundingClientRect().top+scrollY-100));
      await page.screenshot({path:path.join(output,'practice-changed-'+width+'.png')});
      const result={width,states,keyboardControls:visited.length,keyboardTables,ordinarySections:10,inlineFigures:3,examples:12,checkpoints:2,practice:9,maths,overflowMath,clippedLabels,pageOverflow,resourceLinks,errors,warnings,failedRequests};
      results.push(result);
      fs.writeFileSync(path.join(output,'in-progress.json'),JSON.stringify(results,null,2));
      assert.equal(await lesson.locator('.katex-error').count(),0);
      assert.equal(await lesson.locator('p p,p div,p section').count(),0);
      assert.deepEqual(overflowMath,[]);assert.deepEqual(clippedLabels,[]);assert.equal(pageOverflow,false);
      assert.deepEqual(errors,[]);assert.deepEqual(warnings.filter(text=>/nest|hydration|descendant/i.test(text)),[]);
      assert.deepEqual(failedRequests,[]);
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(output,process.env.MEASURE_READING_ONLY ? 'reading-results.json' : 'results.json'),JSON.stringify({status:'passed',at:new Date().toISOString(),results},null,2));
  console.log(results.map(({width,states,keyboardControls,maths})=>({width,states,keyboardControls,mathCount:maths.length})));
})().catch(error=>{console.error(error);process.exitCode=1;});
