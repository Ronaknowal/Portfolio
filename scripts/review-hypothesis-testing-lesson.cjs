const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const folder = path.resolve('scratch/hypothesis-testing-browser');
fs.mkdirSync(folder, { recursive: true });
async function range(lab, name, value) {
  await lab.getByRole('slider', { name, exact: true }).evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}
async function capture(page, lab, file) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await lab.screenshot({ path: path.join(folder, file) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function assertGeometry(lesson) {
  const bad = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap((svg, index) => {
    const border = svg.getBoundingClientRect();
    return [...svg.querySelectorAll('text')].filter(node => {
      const box = node.getBoundingClientRect();
      return box.left < border.left - 1 || box.right > border.right + 1 || box.top < border.top - 1 || box.bottom > border.bottom + 1;
    }).map(node => ({ plot: index, text: node.textContent }));
  }));
  assert.deepEqual(bad, [], 'All plot labels fit their SVG');
  assert.equal(await lesson.locator('svg').evaluateAll(nodes => nodes.some(node => /NaN|Infinity/.test(node.innerHTML))), false);
}
(async () => {
  const m = await import(pathToFileURL(path.resolve('src/learn/data/hypothesis-testing-models.js')));
  const { hypothesisExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/hypothesis-testing-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [], warnings = [], requests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', event => {
        if (event.type() === 'error') errors.push(event.text());
        if (event.type() === 'warning') warnings.push(event.text());
      });
      page.on('requestfailed', request => requests.push({ url: request.url(), reason: request.failure() }));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/hypothesis-testing-confidence-intervals');
      const lesson = page.locator('.hypothesis-lesson');
      await lesson.locator('[data-lab="proportion-coverage"]').waitFor();
      const counts = {};
      const coverage = lesson.locator('[data-lab="confidence-coverage"]');
      counts.coverage = 0;
      for (const n of [5,25,100]) for (const level of [90,95,99]) for (const mode of ['known','estimated']) {
        await range(coverage,'Observations per experiment',n);
        await coverage.getByRole('combobox',{name:'Coverage confidence level'}).selectOption(String(level));
        await coverage.getByRole('combobox',{name:'Coverage spread model'}).selectOption(mode);
        const expected = m.intervalCoverageState(n,level,0,mode);
        assert.match(await coverage.locator('dd').first().innerText(), new RegExp(`^${expected.covered}/40 cover`));
        assert.equal(await coverage.locator('circle').count(),40);
        counts.coverage++;
      }
      await coverage.getByRole('button',{name:'Reset coverage'}).click();
      await capture(page,coverage,`coverage-known-${width}.png`);
      await coverage.getByRole('button',{name:'Draw 40 new experiments'}).click();
      assert.match(await coverage.locator('dd').first().innerText(),new RegExp(`^${m.intervalCoverageState(25,95,1).covered}/40 cover`));
      await coverage.getByRole('combobox',{name:'Coverage spread model'}).selectOption('estimated');
      await capture(page,coverage,`coverage-estimated-${width}.png`);
      await coverage.getByRole('button',{name:'Reset coverage'}).click();

      const prediction=lesson.locator('[data-lab="mean-prediction"]');
      for (const n of [5,25,100]) {
        await range(prediction,'Sample size for interval comparison',n);
        const expected=m.predictionWidths(n);
        assert.equal(await prediction.locator('dd').first().innerText(),`[${m.inferenceNumber(100-expected.meanHalfWidth)}, ${m.inferenceNumber(100+expected.meanHalfWidth)}]`);
      }
      counts.prediction=3;
      await capture(page,prediction,`prediction-${width}.png`);
      const tail=lesson.locator('[data-lab="null-tail"]');
      counts.tails=0;
      for (const shift of [-2,3]) for (const spread of [.25,2]) for (const reference of [-1,2]) for (const alpha of [.01,.05,.1]) for (const alternative of ['two-sided','greater','less']) {
        await range(tail,'Shift every saving',shift); await range(tail,'Multiply residual spread',spread); await range(tail,'Null mean saving',reference);
        await tail.getByRole('combobox',{name:'Test significance level'}).selectOption(String(alpha));
        await tail.getByRole('combobox',{name:'Test alternative'}).selectOption(alternative);
        const expected=m.pairedTailState({shift,spread,reference,alpha,alternative});
        assert.equal(await tail.locator('dd').nth(2).innerText(),m.inferenceNumber(expected.p,6));
        assert.equal(await tail.locator('dd').nth(3).innerText(),`[${m.inferenceNumber(expected.low)}, ${m.inferenceNumber(expected.high)}] ms`);
        counts.tails++;
      }
      await tail.getByRole('button',{name:'Reset paired test'}).click();
      await capture(page,tail,`tail-two-sided-${width}.png`);
      await tail.getByRole('combobox',{name:'Test alternative'}).selectOption('greater');
      await capture(page,tail,`tail-one-sided-${width}.png`);
      await tail.getByRole('button',{name:'Reset paired test'}).click();

      const effect=lesson.locator('[data-lab="practical-effect"]');
      counts.effects=0;
      for (const preset of Object.keys(m.EFFECT_PRESETS)) for (const tolerance of [.25,.5,2]) {
        await effect.getByRole('combobox').selectOption(preset);
        await range(effect,'Predeclared equivalence tolerance',tolerance);
        const expected=m.practicalEffectState(preset,tolerance);
        assert.match(await effect.locator('dd').last().innerText(), new RegExp(`^p=${m.inferenceNumber(expected.equivalenceP,6).replaceAll('.','\\.')};`));
        counts.effects++;
      }
      await effect.getByRole('combobox').selectOption('precise'); await range(effect,'Predeclared equivalence tolerance',.5);
      await capture(page,effect,`effect-equivalence-${width}.png`);
      await effect.getByRole('combobox').selectOption('shifted');
      await capture(page,effect,`effect-shifted-${width}.png`);

      const power=lesson.locator('[data-lab="planned-power"]');
      counts.power=0;
      for (const n of [5,25,200]) for (const sigma of [2,4,10]) for (const delta of [0,2,4]) for (const alpha of [.01,.1]) for (const alternative of ['greater','two-sided']) {
        await range(power,'Planned independent observations',n); await range(power,'Known population SD',sigma); await range(power,'Hypothetical true saving',delta);
        await power.getByRole('combobox',{name:'Power significance level'}).selectOption(String(alpha)); await power.getByRole('combobox',{name:'Power test direction'}).selectOption(alternative);
        assert.equal(await power.locator('dd').nth(1).innerText(),m.inferencePercent(m.plannedPowerState(n,sigma,delta,alpha,alternative).power));
        counts.power++;
      }
      await range(power,'Planned independent observations',25); await range(power,'Known population SD',4); await range(power,'Hypothetical true saving',2);
      await power.getByRole('combobox',{name:'Power significance level'}).selectOption('0.05');
      await power.getByRole('combobox',{name:'Power test direction'}).selectOption('greater');
      await capture(page,power,`power-${width}.png`);

      const proportion=lesson.locator('[data-lab="proportion-coverage"]');
      counts.proportions=0;
      for (const n of [5,10,15,20]) {
        await range(proportion,'Independent Bernoulli trials',n);
        for (const observed of [0,Math.floor(n/2),n]) for (const truth of [0,.04,.2,1]) {
          await range(proportion,'Observed successes',observed); await range(proportion,'Hypothetical true success probability',truth);
          const expected=m.proportionCoverageState(n,observed,truth);
          const texts=await proportion.locator('tbody tr').allTextContents();
          for (const [index,key] of ['wald','wilson','exact'].entries()) assert(texts[index].includes(m.inferencePercent(expected.coverage[key])));
          counts.proportions++;
        }
      }
      await range(proportion,'Independent Bernoulli trials',10); await range(proportion,'Observed successes',0); await range(proportion,'Hypothetical true success probability',.2);
      await capture(page,proportion,`proportion-zero-${width}.png`);
      await range(proportion,'Hypothetical true success probability',.04);
      await capture(page,proportion,`proportion-undercoverage-${width}.png`);
      const chances=lesson.locator('[data-lab="multiple-chances"]');
      counts.chances=0;
      for (const count of [1,20,100]) for (const looks of [1,9,12]) {
        await range(chances,'Independent true-null tests',count); await range(chances,'Maximum fair-coin looks',looks);
        assert((await chances.innerText()).includes(m.inferencePercent(m.familyErrorState(count).independentFamilyError)));
        assert((await chances.innerText()).includes(m.inferencePercent(m.optionalLooksState(looks).cumulativeError)));
        counts.chances++;
      }
      await range(chances,'Independent true-null tests',20);
      await capture(page,chances,`repeated-chances-${width}.png`);
      const signs=lesson.locator('[data-lab="sign-flip"]');
      counts.signs=0;
      for (const preset of ['original','allPositive','centered']) {
        await signs.getByRole('combobox').selectOption(preset);
        assert.equal(await signs.getByRole('slider').inputValue(),'0');
        for (let mask=0;mask<32;mask++) {
          await range(signs,'Sign assignment index',mask);
          const expected=m.signFlipState(preset,mask);
          assert.equal(await signs.locator('dd').nth(2).innerText(),`${m.inferenceNumber(expected.current.mean)} ms`);
          assert.equal(await signs.locator('dd').last().innerText(),`${expected.extremeCount}/32 = ${m.inferenceNumber(expected.p,6)}`);
          counts.signs++;
        }
      }
      await signs.getByRole('combobox').selectOption('original');
      await capture(page,signs,`sign-flip-${width}.png`);
      await assertGeometry(lesson);

      for (let index=0;index<await lesson.locator('h2').count();index++) {
        await lesson.locator('h2').nth(index).evaluate(node=>window.scrollTo(0,node.getBoundingClientRect().top+scrollY-100));
        await page.screenshot({path:path.join(folder,`reading-${index+1}-${width}.png`)});
      }
      const anchorLinks=lesson.locator('a[href^="#"]');
      for (let index=0;index<await anchorLinks.count();index++) {
        const href=await anchorLinks.nth(index).getAttribute('href');
        assert.equal(await lesson.locator(`[id="${href.slice(1)}"]`).count(),1);
        await anchorLinks.nth(index).click();
        assert.equal(new URL(page.url()).hash,href);
      }
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>node.open=true));
      for (const example of Object.values(examples)) {
        const block=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:example.title,exact:true})});
        assert.equal(await block.count(),1);
        const text=await block.innerText();
        assert(text.includes(example.code),`${example.title}: full code`);
        assert(text.includes(example.expected),`${example.title}: full stdout`);
      }
      assert.equal(await lesson.locator('.hypothesis-practice').count(),8);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      assert.equal(await lesson.locator('p p,p div,p section').count(),0);
      const math=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({text:node.textContent.slice(0,100),width:node.clientWidth,scroll:node.scrollWidth})));
      const overflowingMath=math.filter(row=>row.scroll>row.width+2);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      await capture(page,lesson.locator('.lesson-sources'),`sources-${width}.png`);
      fs.writeFileSync(path.join(folder,'in-progress.json'),JSON.stringify({width,counts,math,overflowingMath,overflow,errors,warnings,requests},null,2));
      assert.deepEqual(overflowingMath,[]);
      assert.equal(overflow,false);
      assert.deepEqual(errors,[]);
      assert.deepEqual(warnings.filter(text=>/nest|hydration|descendant/i.test(text)),[]);
      results.push({width,counts,examples:Object.keys(examples).length,anchors:await anchorLinks.count(),practice:8,mathCount:math.length,overflowingMath,overflow,errors,warnings,requests});
      await page.close();
    }
  } finally {await browser.close();}
  fs.writeFileSync(path.join(folder,'results.json'),JSON.stringify({at:new Date().toISOString(),status:'passed',results},null,2));
  console.log(results);
})().catch(error=>{console.error(error);process.exitCode=1;});
