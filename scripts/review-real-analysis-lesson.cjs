const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/real-analysis-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { realAnalysisExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/real-analysis-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  const sourceHashes = JSON.parse(fs.readFileSync('scratch/real-analysis-verification/results.json','utf8')).sourceHashes;
  for (const [file, expected] of Object.entries(sourceHashes)) assert.equal(hash(file), expected, `Native/browser source mismatch: ${file}`);
  try {
    for (const width of (process.env.ANALYSIS_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/real-analysis-sequences-modes-of-convergence?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.real-analysis-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      const record = { width, anchors: [], states: [], keyboard: [], captures: [], programs: [], fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) };
      assert(record.fonts.length > 0, 'Actual intended fonts loaded');
      const lab = name => lesson.getByRole('region', { name, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(80);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) }); record.captures.push(file);
      }
      async function press(region, name) {
        const button = region.getByRole('button', { name, exact: true });
        await page.keyboard.press('Tab');
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter'); record.keyboard.push(name);
      }
      async function range(region, name, value) {
        const input = region.getByRole('slider', { name, exact: true });
        await input.fill(String(value)); await input.dispatchEvent('input');
      }
      try {
        await shot(lesson.locator('.lesson-intro'), 'analysis-intro');
        for (const depth of await lesson.locator('details.analysis-depth').all()) { await depth.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); }
        assert.equal(await lesson.locator('.analysis-lab').count(), 9);
        for (const select of await lesson.locator('.analysis-field select').all()) {
          const original = await select.inputValue();
          const choices = await select.locator('option').evaluateAll(options => options.map(option => option.value));
          await select.selectOption(choices[0]); await select.focus();
          await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
          assert.equal(await select.inputValue(), choices[1]);
          record.keyboard.push(`${await select.getAttribute('aria-label')}: ArrowDown/Enter`);
          await select.selectOption(original);
        }
        for (const slider of await lesson.locator('.analysis-field input[type="range"]').all()) {
          const original = await slider.inputValue();
          const maximum = Number(await slider.getAttribute('max'));
          await slider.focus(); await page.keyboard.press(Number(original) === maximum ? 'ArrowLeft' : 'ArrowRight');
          assert.notEqual(await slider.inputValue(), original);
          record.keyboard.push(`${await slider.getAttribute('aria-label')}: arrow change`);
          await slider.fill(original); await slider.dispatchEvent('input');
        }
        for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
          const href = await anchor.getAttribute('href');
          const target = lesson.locator(`[id="${href.slice(1)}"]`);
          assert.equal(await target.count(), 1, href);
          await anchor.focus(); await page.keyboard.press('Enter');
          await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 40 && box.top <= 150; }, href.slice(1));
          record.anchors.push(href); await shot(target, `reading-section-${record.anchors.length}`);
        }
        for (const [index, figure] of (await lesson.locator('figure').all()).entries()) await shot(figure, `reading-figure-${index}`);
        // Each select option and range boundary is inspected, then reset.
        for (const [index, region] of (await lesson.locator('.analysis-lab').all()).entries()) {
          const reset = region.getByRole('button', { name: /^Reset / });
          for (const select of await region.locator('select').all()) {
            const label = await select.getAttribute('aria-label');
            for (const value of await select.locator('option').evaluateAll(nodes => nodes.map(node => node.value))) {
              await select.selectOption(value);
              assert.equal(await select.inputValue(), value);
              assert(!(await region.innerText()).match(/NaN|Infinity|undefined/));
              record.states.push({ lab: index, label, value });
            }
            await reset.click();
          }
          for (const slider of await region.getByRole('slider').all()) {
            const label = await slider.getAttribute('aria-label');
            for (const value of [await slider.getAttribute('min'), await slider.getAttribute('max')]) {
              await range(region, label, value);
              assert.equal(await slider.inputValue(), value);
              assert(!(await region.innerText()).match(/NaN|Infinity|undefined/));
              record.states.push({ lab: index, label, value });
            }
            await reset.click();
          }
          await shot(region, `investigation-${index}`);
          await press(region, await reset.innerText());
        }
        const tail = lab('Choose where the entire safe tail starts');
        assert((await tail.innerText()).includes('exactly on the tolerance boundary'));
        await range(tail, 'Proposed start N', 10);
        assert((await tail.innerText()).includes('This N certifies the tail'));
        await shot(tail, 'strict-tail-passes');
        const bracket = lab('Keep a shrinking bracket around an unknown real number');
        await bracket.getByLabel('Squared target', { exact: true }).selectOption('5');
        await range(bracket, 'Bisection steps', 0);
        await press(bracket, 'Bisect once');
        assert((await bracket.innerText()).includes('4 ≤ 5 ≤ 6.25'));
        await shot(bracket, 'changed-exact-bracket');
        const cauchy = lab('Look beyond the next small step');
        await cauchy.getByLabel('Series increments', { exact: true }).selectOption('telescoping');
        await cauchy.getByLabel('Block begins after N', { exact: true }).selectOption('16');
        assert((await cauchy.innerText()).includes('1/17 − 1/33'));
        const power = lab('Fix one point, then let the difficult point move');
        await power.getByLabel('Function domain', { exact: true }).selectOption('compact-subinterval');
        await range(power, 'Power index n', 8);
        assert((await power.innerText()).includes('0.10011'));
        await shot(power, 'compact-domain-bound');
        const triangle = lab('Find the error that a coarse grid misses');
        await triangle.getByLabel('Triangle amplitude', { exact: true }).selectOption('unit-area');
        assert((await triangle.innerText()).includes('42.66667'));
        await shot(triangle, 'missed-unit-area-peak');
        const derivative = lesson.locator('.analysis-lab').nth(5);
        await derivative.locator('select').nth(1).selectOption('2');
        assert((await derivative.innerText()).includes('stronger derivative theorem applies'));
        await shot(derivative, 'both-derivative-bounds');
        const series = lab('A function-series bound does not control its derivative');
        await series.getByLabel('Series evaluation point', { exact: true }).selectOption('-1');
        assert((await series.innerText()).includes('converge conditionally'));
        await shot(series, 'changed-series-endpoint');
        const bernstein = lab('Build an approximation from nearby weighted values');
        await bernstein.getByLabel('Polynomial degree n', { exact: true }).selectOption('100');
        await bernstein.getByLabel('Corner location c', { exact: true }).selectOption('0.7');
        await range(bernstein, 'Weighted evaluation x', 0.7);
        assert((await bernstein.innerText()).includes('0.05'));
        await shot(bernstein, 'changed-corner-weights');
        const sweep = lab('A shrinking chance can keep revisiting one observer');
        await sweep.getByLabel('Dyadic block k', { exact: true }).selectOption('3');
        await range(sweep, 'Interval position j', 4);
        assert((await sweep.innerText()).includes('[0.5, 0.625)'));
        assert.equal(await sweep.locator('.analysis-readout').nth(2).locator('strong').innerText(), '1');
        await shot(sweep, 'half-open-boundary-hit');
        for(const [index,practice] of (await lesson.locator('.analysis-practice').all()).entries()) {
          const details=practice.locator(':scope > details'); assert.equal(await details.count(),2); assert.equal(await details.nth(1).getAttribute('open'),null);
          for(const disclosure of await details.all()) { await disclosure.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); }
          if([2,6,10,13].includes(index)) await shot(practice,`changed-practice-${index}`);
        }
        for(const example of Object.values(examples)) {
          const program=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:example.title,exact:true})}); assert.equal(await program.count(),1,example.title);
          for(const [index,expected] of [[0,example.code],[1,example.expected]]) assert.equal(normalize(await program.locator(':scope > div').nth(index).evaluate(node=>[...node.childNodes].filter(child=>child.nodeType===3).map(child=>child.textContent).join(''))),normalize(expected));
          assert.equal(normalize(await program.evaluate(node=>node.previousElementSibling.textContent)),normalize(`Before running: ${example.question}`)); record.programs.push(example.title);
        }
        await shot(lesson.locator('.python-example').last(),'approximation-complete-program');
        await shot(lesson.locator('.python-example').last().locator(':scope > div').last(),'changed-output');
        const math=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({width:node.clientWidth,scroll:node.scrollWidth,text:node.textContent}))); record.math=math;
        assert(math.every(box=>box.scroll<=box.width+1),`Math overflow ${width}: ${JSON.stringify(math.filter(box=>box.scroll>box.width+1))}`);
        for(const [index,equation] of (await lesson.locator('.katex-display').all()).entries()) await shot(equation,`equation-${index}`);
        const clipped=await lesson.locator('svg').evaluateAll(nodes=>nodes.flatMap(svg=>{const rect=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].flatMap(text=>{const box=text.getBoundingClientRect();return box.left<rect.left-2||box.right>rect.right+2||box.top<rect.top-2||box.bottom>rect.bottom+2?[text.textContent]:[];});}));
        assert.deepEqual(clipped,[],`SVG clipping ${width}`); assert.equal(await lesson.locator('.katex-error').count(),0);
        assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)),`Document overflow ${width}`);
        record.sources=await lesson.locator('.lesson-sources a').evaluateAll(nodes=>nodes.map(node=>({href:node.href,target:node.target,rel:node.rel})));
        assert(record.sources.every(link=>link.href.startsWith('https:')&&link.target==='_blank'&&link.rel.includes('noreferrer')));
        await shot(lesson.locator('.lesson-sources'),'learning-resources');
        records.push(record); fs.writeFileSync(path.join(directory,'progress.json'),JSON.stringify({checkedAt:new Date().toISOString(),records,errors},null,2));
      } catch(error) {
        await page.screenshot({path:path.join(directory,`failure-${width}.png`)});
        fs.writeFileSync(path.join(directory,'failure.json'),JSON.stringify({checkedAt:new Date().toISOString(),record,errors,error:String(error)},null,2)); throw error;
      }
      await page.close();
    }
    assert.deepEqual(errors,[]);
    for (const [file, expected] of Object.entries(sourceHashes)) assert.equal(hash(file), expected, `Source changed during browser review: ${file}`);
    fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,records,errors,sourceHashes},null,2));
    console.log('Real Analysis actual-font reading, interaction, keyboard, programs and geometry passed at all requested widths.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
