const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/ordinary-differential-equations-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { ordinaryDifferentialEquationsExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/ordinary-differential-equations-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of (process.env.ODE_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/ordinary-differential-equations-linear-systems?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.ode-lesson');
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
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter'); record.keyboard.push(name);
      }
      async function range(region, name, value) {
        const input = region.getByRole('slider', { name, exact: true });
        await input.fill(String(value)); await input.dispatchEvent('input');
      }
      try {
        await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
        for (const select of await lesson.locator('.ode-control select').all()) {
          const original = await select.inputValue();
          const choices = await select.locator('option').evaluateAll(options => options.map(option => option.value));
          await select.selectOption(choices[0]); await select.focus();
          await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
          assert.equal(await select.inputValue(), choices[1]);
          record.keyboard.push(`${await select.getAttribute('aria-label')}: ArrowDown/Enter`);
          await select.selectOption(original);
        }
        for (const slider of await lesson.locator('.ode-control input[type="range"]').all()) {
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
        const field = lab('Rate and direction field investigation');
        for (const [kind, initial, expected] of [['cooling',0,'equilibrium'],['logistic',10,'equilibrium'],['logistic',2,'rising'],['logistic',20,'falling']]) {
          await field.getByLabel('Rate law',{exact:true}).selectOption(kind); await range(field,'Initial state',initial);
          assert((await field.locator('.ode-readout').innerText()).includes(expected)); record.states.push(['field',kind,initial]);
        }
        await shot(field.locator('figure'),'logistic-above-capacity');
        await press(field,'Reset rate field');
        const fieldSlider=field.getByRole('slider',{name:'Initial state',exact:true}); await fieldSlider.focus(); await page.keyboard.press('ArrowRight'); assert.equal(await fieldSlider.inputValue(),'11'); record.keyboard.push('Initial state ArrowRight'); await press(field,'Reset rate field');
        const waiting=lab('Nonunique waiting solutions investigation');
        for(const departure of [0,1.5,3]) { await range(waiting,'Departure time',departure); assert((await waiting.innerText()).includes(`At t = 3 it equals ${(3-departure)**2}`)); record.states.push(['waiting',departure]); }
        await shot(waiting,'waiting-zero-until-horizon'); await press(waiting,'Reset waiting family');
        const oscillator=lab('Position velocity and energy investigation');
        for(const damping of [0,2,4,6]) { await oscillator.getByLabel('Damping c',{exact:true}).selectOption(String(damping)); await range(oscillator,'Motion inspection time',0); assert((await oscillator.locator('.ode-state-strip').innerText()).includes('E = 2 J')); record.states.push(['oscillator',damping]); }
        await oscillator.getByLabel('Initial velocity',{exact:true}).selectOption('2'); await range(oscillator,'Motion inspection time',1); await shot(oscillator.locator('.ode-two-plots'),'overdamped-moving-state'); await press(oscillator,'Reset oscillator');
        const columns=lab('Fundamental matrix columns investigation');
        for(const preset of ['node','saddle','rotation','spiral','jordan','nilpotent']) { await columns.getByLabel('Linear system',{exact:true}).selectOption(preset); record.states.push(['matrix',preset]); assert.equal(await columns.locator('.ode-matrix-readout .ode-vector').count(),4); }
        assert((await columns.innerText()).includes('[1, 1]')); await shot(columns.locator('figure'),'nilpotent-drift');
        await columns.getByLabel('Linear system',{exact:true}).selectOption('saddle'); await columns.getByLabel('Initial vector',{exact:true}).selectOption('first'); assert((await columns.innerText()).includes('[2.7183, 0]')); await shot(columns.locator('figure'),'growing-mode'); await press(columns,'Reset matrix columns');
        const forcing=lab('Initial and forced response investigation');
        await range(forcing,'Input switch time',0); assert((await forcing.locator('.ode-readout').innerText()).includes('0 + 0 =')); record.states.push(['forcing','zero first interval']);
        await range(forcing,'Second interval power',20); await range(forcing,'Response inspection time',0); assert((await forcing.locator('.ode-readout').innerText()).includes('40 + 0 + 0 = 40')); record.states.push(['forcing','initial']);
        await range(forcing,'Response inspection time',6); await range(forcing,'Input switch time',3); await range(forcing,'First interval power',0); await shot(forcing.locator('figure'),'late-heating-contributions'); await press(forcing,'Reset input response');
        const order=lab('Order of two continuous stages investigation');
        await order.getByLabel('Stage initial state',{exact:true}).selectOption('second'); assert((await order.innerText()).includes('Final state: [1, 2]'));
        await order.getByLabel('Stage order',{exact:true}).selectOption('lower'); assert((await order.innerText()).includes('Final state: [1, 1]')); record.states.push(['order','both changed-initial']); await shot(order.locator('.ode-stage-chain'),'swapped-stages'); await press(order,'Reset stage order');
        const numerical=lab('Numerical stages and error investigation');
        for(const [method,count] of [['euler',1],['midpoint',2],['rk4',4]]) { await numerical.getByLabel('Step method',{exact:true}).selectOption(method); assert.equal(await numerical.locator('tbody tr').count(),count); assert((await numerical.locator('.ode-readout').innerText()).includes('8 steps; final time 5')); record.states.push(['stages',method,count]); }
        await shot(numerical.locator('.ode-table-wrap'),'rk4-actual-stages');
        await numerical.getByLabel('Step method',{exact:true}).selectOption('euler'); await numerical.getByLabel('Decay rate k',{exact:true}).selectOption('4');
        for(const [step,factor] of [[0.5,'−1'],[0.6,'-1.4']]) { await numerical.getByLabel('Requested step h',{exact:true}).selectOption(String(step)); assert((await numerical.innerText()).includes(step===0.5?'factor is -1.':'factor is -1.4.')); record.states.push(['unstable',step]); }
        await shot(numerical.locator('figure'),'unstable-euler'); await shot(numerical.locator('.ode-readout'),'unstable-final-error'); await press(numerical,'Reset numerical steps');
        const boundary=lab('Boundary conditions and uniqueness investigation');
        assert((await boundary.innerText()).includes('infinitely many solutions'));
        await boundary.getByLabel('Requested endpoint value',{exact:true}).selectOption('1'); assert((await boundary.innerText()).includes('no solution')); await shot(boundary.locator('figure'),'inconsistent-boundary');
        await boundary.getByLabel('Right endpoint',{exact:true}).selectOption('half-pi'); assert((await boundary.innerText()).includes('one solution')); record.states.push(['boundary','many/none/one']); await press(boundary,'Reset boundary conditions');
        for(const depth of await lesson.locator('details.ode-depth').all()) { assert.equal(await depth.getAttribute('open'),null); await depth.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); record.keyboard.push(await depth.locator(':scope > summary').innerText()); }
        for(const [index,practice] of (await lesson.locator('.ode-practice').all()).entries()) {
          const details=practice.locator(':scope > details'); assert.equal(await details.count(),2); assert.equal(await details.nth(1).getAttribute('open'),null);
          for(const disclosure of await details.all()) { await disclosure.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); }
          if([2,6,9,13].includes(index)) await shot(practice,`changed-practice-${index}`);
        }
        for(const example of Object.values(examples)) {
          const program=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:example.title,exact:true})}); assert.equal(await program.count(),1,example.title);
          for(const [index,expected] of [[0,example.code],[1,example.expected]]) assert.equal(normalize(await program.locator(':scope > div').nth(index).evaluate(node=>[...node.childNodes].filter(child=>child.nodeType===3).map(child=>child.textContent).join(''))),normalize(expected));
          assert.equal(normalize(await program.evaluate(node=>node.previousElementSibling.textContent)),normalize(`Before running: ${example.question}`)); record.programs.push(example.title);
        }
        await shot(lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:examples.events.title,exact:true})}),'events-complete-program');
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
    fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,records,errors},null,2));
    console.log('ODE actual-font reading, interaction, keyboard, programs and geometry passed at all requested widths.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
