const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/exponential-family-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { exponentialFamilyExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/exponential-family-examples.js')));
  const model = await import(pathToFileURL(path.resolve('src/learn/data/exponential-family-models.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/exponential-families-sufficient-statistics?module=mathematical-statistical-foundations');
      const lesson = page.locator('.exponential-family-lesson'); await lesson.waitFor();
      const record = { width, anchors: [], captures: [], equations: [], controls: [], code: [] };
      const shot = async (locator, name) => {
        await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + scrollY - 90));
        const filename = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, filename) }); record.captures.push(filename);
      };
      const fact = (region, name) => region.locator('.family-facts > div').filter({ has: page.locator('dt', { hasText: name }) }).locator('dd');
      const button = async (region, name, key = 'Enter') => { await region.getByRole('button', { name, exact: true }).focus(); await page.keyboard.press(key); };
      await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
      for (const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await link.getAttribute('href'); await link.focus(); await page.keyboard.press('Enter');
        await page.waitForFunction(value => location.hash === value, href); await page.waitForTimeout(60);
        const box = await page.locator(`[id="${href.slice(1)}"]`).boundingBox();
        assert(box && box.y >= 45 && box.y < 200, `${href} top ${box?.y}`);
        record.anchors.push({ href, top: box.y });
        if (width !== 320) await shot(page.locator(`[id="${href.slice(1)}"]`), `ordinary-section-${record.anchors.length}`);
      }
      assert.equal(record.anchors.length, 10);
      for (const [index, figure] of (await lesson.locator('.family-figure').all()).entries()) await shot(figure, `ordinary-figure-${index + 1}`);
      for (const [index, lab] of (await lesson.locator('.family-lab').all()).entries()) await shot(lab, `ordinary-lab-${index + 1}`);
      for (const [index, equation] of (await lesson.locator('.katex-display').all()).entries()) {
        record.equations.push(await equation.evaluate(element => ({ client: element.clientWidth, scroll: element.scrollWidth })));
        if (width === 320) await shot(equation, `equation-${index + 1}`);
      }
      const summary = lesson.locator('[data-family-lab="sufficiency"]');
      await button(summary, 'Move one success A → B'); assert.equal(await fact(summary, 'Likelihood ratio: yours / reference').innerText(), '1');
      await summary.getByLabel('Probability model').selectOption('groups');
      assert.equal(await fact(summary, 'Likelihood ratio: yours / reference').innerText(), '0.107143');
      const probability = summary.getByRole('slider', { name: 'First probability', exact: true });
      await probability.focus(); await page.keyboard.press('ArrowLeft');
      assert.equal(await probability.inputValue(), '75');
      assert.equal(await fact(summary, 'Likelihood ratio: yours / reference').innerText(), '0.142857');
      await summary.getByRole('button', { name: 'Observation 1', exact: true }).focus(); await page.keyboard.press('Space');
      assert((await summary.locator('.family-feedback').innerText()).includes('totals now differ'));
      await button(summary, 'Reset comparison'); assert.equal(await fact(summary, 'Your total / reference total').innerText(), '6 / 6');
      record.controls.push('model change, exact ratio, keyboard bit/probability, count mismatch and reset');

      const weights = lesson.locator('[data-family-lab="normalizer"]');
      const eta = weights.getByRole('slider'); await eta.focus(); await page.keyboard.press('End');
      assert.equal(await eta.inputValue(), '5');
      assert.equal(await fact(weights, 'A′ = expected X').innerText(), model.formatFamilyNumber(model.finiteExponentialFamily(5, 0, [1,2,1]).mean[0]));
      await page.keyboard.press('Home'); assert.equal(await eta.inputValue(), '-5');
      await button(weights, 'Reset weights', 'Space'); assert.equal(await fact(weights, 'A″ = variance of X').innerText(), '0.5');
      record.controls.push('normalizer endpoint tilts, visible mean/variance and keyboard reset');

      const moments = lesson.locator('[data-family-lab="moments"]');
      await button(moments, 'Fit observed moments');
      const fit = model.finiteMomentFit([2,5,3]);
      const rangeValues = await moments.getByRole('slider').evaluateAll(elements => elements.map(element => Number(element.value)));
      record.fittedRangeValues = rangeValues;
      assert(Math.abs(rangeValues[0] - fit.parameters[0]) < 1e-12, `slider changed exact fit ${rangeValues}`);
      assert.equal(await fact(moments, 'Modeled (E[X], E[X²])').innerText(), '0.1, 0.5');
      await shot(moments.locator('.family-moment-layout'), 'fitted-moments');
      await button(moments, 'Remove zero category'); assert(await moments.getByRole('button', {name:'Fit observed moments',exact:true}).isDisabled());
      assert((await moments.locator('.family-feedback').innerText()).includes('Boundary target'));
      await shot(moments.locator('.family-moment-layout'), 'boundary-moments');
      for (const value of [-1,0,1]) await moments.getByRole('spinbutton', {name:`Count at ${value}`,exact:true}).fill('0');
      assert((await moments.locator('.family-feedback').innerText()).includes('No data'));
      await button(moments, 'Reset moments', 'Space');
      const firstParameter = moments.getByRole('slider', { name:'Linear η₁', exact:true });
      await firstParameter.focus(); await page.keyboard.press('End'); assert.equal(await firstParameter.inputValue(),'5');
      await button(moments, 'Reset moments');
      record.controls.push('finite fit retains exact coordinates, boundary, empty dataset, parameter keyboard and reset');

      const prior = lesson.locator('[data-family-lab="coordinates"]');
      for(const name of ['Prior alpha','Prior beta']) { await prior.getByRole('slider',{name,exact:true}).focus(); await page.keyboard.press('Home'); }
      assert.equal(await fact(prior,'Density at p=0.5').innerText(),'1');
      assert.equal(await fact(prior,'Density at η=0').innerText(),'0.25');
      assert.equal(await fact(prior,'Probability in either interval').innerText(),'0.5');
      await shot(prior.locator('.family-density-pair'),'uniform-coordinate-density');
      await prior.getByRole('slider',{name:'Prior alpha',exact:true}).focus();await page.keyboard.press('End');
      const expectedPrior=model.betaCoordinateModel(8,1);
      assert.equal(await fact(prior,'Probability in either interval').innerText(),model.formatFamilyNumber(expectedPrior.intervalMass));
      await button(prior,'Reset prior','Space');assert.equal(await fact(prior,'Probability in either interval').innerText(),'0.6875');
      record.controls.push('uniform versus log-odds density, skewed prior, interval mass, keyboard and reset');

      const displayed=await lesson.locator('.python-example').all(); assert.equal(displayed.length,9);
      assert.equal(await lesson.locator('.family-example > p').count(),9);
      for(const exampleElement of displayed){
        const title=await exampleElement.locator('h3').innerText();const example=Object.values(examples).find(value=>value.title===title);assert(example);
        const blocks=await exampleElement.locator(':scope > div').all();assert.equal(blocks.length,2);
        assert(normalize(await blocks[0].innerText()).includes(normalize(example.code)));
        assert.equal(normalize(await blocks[1].innerText()).replace(/^OUTPUT /,''),normalize(example.expected));
        const state=await blocks[0].evaluate(element=>({client:element.clientWidth,scroll:element.scrollWidth,overflow:getComputedStyle(element).overflowX}));
        assert(state.scroll<=state.client+1||['auto','scroll'].includes(state.overflow));record.code.push({title,...state});
      }
      await shot(lesson.locator('.family-example').nth(3),'program-question');
      await shot(lesson.locator('.python-example').nth(3).locator('.lesson-note'),'program-output');
      const exercises=await lesson.locator('.family-exercise').all();assert.equal(exercises.length,9);
      for(const exercise of exercises){
        const details=await exercise.locator('details').all();assert.equal(details.length,2);
        assert.equal(await details[0].getAttribute('open'),null);assert.equal(await details[1].getAttribute('open'),null);
        await details[0].locator('summary').focus();await page.keyboard.press('Enter');
        assert.equal(await details[1].getAttribute('open'),null);
        await details[1].locator('summary').focus();await page.keyboard.press('Space');
        assert((await details[1].innerText()).length>150);
      }
      await shot(exercises[2],'changed-task-solution');
      await shot(lesson.locator('.lesson-sources'),'sources');
      record.externalLinks=await lesson.locator('.lesson-sources a').evaluateAll(elements=>elements.map(element=>({href:element.href,target:element.target,rel:element.rel})));
      assert(record.externalLinks.length>=5&&record.externalLinks.every(link=>link.href.startsWith('https:')&&link.target==='_blank'&&link.rel.includes('noreferrer')));
      record.overflow=await lesson.locator('.family-lab,.family-figure,.katex-display').evaluateAll(elements=>elements.filter(element=>element.scrollWidth>element.clientWidth+2).map(element=>({className:element.className,client:element.clientWidth,scroll:element.scrollWidth})));
      record.svgTextOverflow=await lesson.locator('svg text').evaluateAll(elements=>elements.flatMap(element=>{const bounds=element.getBBox(),box=element.ownerSVGElement.viewBox.baseVal;return bounds.x<-.5||bounds.x+bounds.width>box.width+.5||bounds.y<-.5||bounds.y+bounds.height>box.height+.5?[{text:element.textContent,x:bounds.x,width:bounds.width}]:[]}));
      record.pageOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      record.mathErrors=await lesson.locator('.katex-error').count();
      records.push(record);fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),records,errors},null,2));
      await page.close();
    }
    assert.deepEqual(errors,[]);
    assert(records.every(record=>!record.pageOverflow&&!record.mathErrors&&!record.overflow.length&&!record.svgTextOverflow.length),JSON.stringify(records.map(record=>({width:record.width,overflow:record.overflow,svg:record.svgTextOverflow}))));
    console.log(JSON.stringify({checkedAt:new Date().toISOString(),records,errors},null,2));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
