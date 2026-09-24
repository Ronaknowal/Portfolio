const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/counting-combinatorics-browser');
const normalize = text => text.replace(/\s+/g, ' ').trim();
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const { countingCombinatoricsExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/counting-combinatorics-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of (process.env.COUNTING_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/counting-combinatorics-mathematical-induction?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.counting-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      const record = { width, anchors: [], states: [], captures: [], programs: [], keyboard: [], fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) };
      assert(record.fonts.length > 0, 'actual intended fonts loaded');
      const region = name => lesson.getByRole('region', { name, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(100);
        const bounds = await target.boundingBox();
        assert(bounds.y >= 0 && bounds.y < 180, `capture arrival ${name}: ${JSON.stringify(bounds)}`);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) }); record.captures.push(file);
      }
      async function press(target, name) {
        const button = target.getByRole('button', { name, exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'), name);
        await page.keyboard.press('Enter'); record.keyboard.push(name);
      }
      try {
        await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
        for (const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
          const href = await link.getAttribute('href');
          const target = lesson.locator(`[id="${href.slice(1)}"]`);
          assert.equal(await target.count(), 1, href);
          await link.focus(); await page.keyboard.press('Enter');
          await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 40 && box.top <= 150; }, href.slice(1));
          record.anchors.push(href); await shot(target, `ordinary-section-${record.anchors.length}`);
        }
        for (const [index, figure] of (await lesson.locator('figure').all()).entries()) await shot(figure, `inline-${index}`);
        for (const [index, lab] of (await lesson.locator('.counting-lab').all()).entries()) await shot(lab, `initial-lab-${index}`);
        const fibers = region('Outcome identity and fibers investigation');
        assert((await fibers.getByRole('status').last().innerText()).includes('12 ÷ 2 = 6'));
        await press(fibers, 'Show unequal repeat groups');
        assert((await fibers.getByRole('status').last().innerText()).includes('1, 2'));
        await press(fibers, 'Next'); assert.equal(await fibers.locator('.counting-fiber .counting-words > .counting-word').count(), 2);
        await shot(fibers.locator('.counting-fiber'), 'unequal-fiber');
        for (const labels of [0, 1, 2, 4]) for (const positions of [0, 1, 3]) for (const repeats of ['false', 'true']) {
          await fibers.getByLabel('Available labels', { exact: true }).selectOption(String(labels));
          await fibers.getByLabel('Positions', { exact: true }).selectOption(String(positions));
          await fibers.getByLabel('Repeated labels', { exact: true }).selectOption(repeats);
          const expected = positions === 0 ? 1 : repeats === 'true' ? labels ** positions : positions > labels ? 0 : Array.from({ length: positions }, (_, index) => labels - index).reduce((a,b) => a*b,1);
          assert((await fibers.getByRole('status').first().innerText()).startsWith(`${expected} ordered descriptions`));
          record.states.push(['choices', labels, positions, repeats]);
        }
        await fibers.getByLabel('Positions').selectOption('0'); await fibers.getByLabel('Available labels').selectOption('0');
        await shot(fibers.locator('.counting-fiber'), 'empty-choice'); await press(fibers, 'Reset');
        const allocations = region('Stars bars and allocations investigation');
        for (const [rule, expected] of [['free',21],['positive',6],['cap',12]]) {
          await allocations.getByLabel('Allocation rule').selectOption(rule);
          assert((await allocations.getByRole('status').innerText()).includes(`Exact count: ${expected}.`));
          record.states.push(['allocations', rule, expected]);
        }
        await press(allocations,'Next'); await press(allocations,'Previous');
        await allocations.getByLabel('Allocation rule').selectOption('free');
        await press(allocations,'Move C → B');
        assert((await allocations.getByRole('status').innerText()).includes('(0, 1, 4)'));
        await shot(allocations.locator('.counting-containers'), 'moved-allocation');
        await allocations.getByLabel('Total tokens').selectOption('2'); await allocations.getByLabel('Allocation rule').selectOption('positive');
        assert((await allocations.getByRole('status').innerText()).includes('Exact count: 0.'));
        assert(await allocations.getByRole('button',{name:'Next',exact:true}).isDisabled());
        await shot(allocations,'infeasible-allocation');
        await allocations.getByLabel('Allocation rule').selectOption('free'); await allocations.getByLabel('Total tokens').selectOption('0');
        assert((await allocations.getByRole('status').innerText()).includes('Exact count: 1.'));
        assert.equal(await allocations.locator('.counting-tokens').filter({hasText:'empty'}).count(),3); await press(allocations,'Reset');
        const overlap = region('Inclusion exclusion contribution investigation');
        for (const [label, weight, total] of [['1 · Add singles',3,13],['2 · Subtract pairs',0,7],['3 · Add triple',1,8]]) {
          await press(overlap,label); assert((await overlap.getByRole('status').innerText()).includes(`weight: ${weight}. Signed total at this stage: ${total}.`));
          record.states.push(['overlap',label,weight,total]);
        }
        await shot(overlap.locator('.counting-selected-object'),'triple-restored');
        await press(overlap,'Object 12 in set C'); assert((await overlap.getByRole('status').innerText()).includes('Actual union size: 8.'));
        await press(overlap,'Empty all sets'); assert((await overlap.getByRole('status').innerText()).includes('Actual union size: 0.')); await press(overlap,'Reset');
        const induction = region('Induction base coverage investigation');
        assert((await induction.getByRole('status').innerText()).includes('6×4 + 2×7 = 38'));
        await press(induction,'Base 18 certificate'); assert((await induction.getByRole('status').innerText()).includes('Unsupported'));
        await shot(induction.locator('.counting-proof-chain'),'unsupported-proof-chain');
        assert((await induction.innerText()).includes('so this target is representable'));
        await induction.getByLabel('Token values and theorem').selectOption('threeFive');
        await induction.getByLabel('Target total').selectOption('29');
        assert((await induction.getByRole('status').innerText()).includes('8×3 + 1×5 = 29'));
        await press(induction,'Base 8 certificate'); assert((await induction.getByRole('status').innerText()).includes('Unsupported'));
        await press(induction,'Reset'); record.states.push(['induction','supported/unsupported, both presets']);
        const paths = region('Balanced paths and reflection investigation');
        for (const pairs of [1,2,3,4,5]) {
          await paths.getByLabel('Parenthesis pairs').selectOption(String(pairs));
          const before = await paths.locator('svg').getAttribute('aria-label');
          await press(paths,'Reflect at first −1');
          assert((await paths.getByRole('status').innerText()).includes('endpoint: +2'));
          const points = await paths.locator('svg polyline').getAttribute('points');
          assert(points.split(' ').length === 2*pairs+1);
          if(pairs===4) await shot(paths.locator('svg'),'reflected-path');
          await press(paths,'Undo at first +1'); assert.equal(await paths.locator('svg').getAttribute('aria-label'),before);
          await paths.getByLabel('Path family').selectOption('valid');
          assert((await paths.getByRole('status').innerText()).includes('First return'));
          await paths.getByLabel('Path family').selectOption('bad'); record.states.push(['paths',pairs]);
        }
        await paths.getByLabel('Parenthesis pairs').selectOption('0');
        assert((await paths.innerText()).includes('no bad paths'));
        await paths.getByLabel('Path family').selectOption('valid'); assert((await paths.getByRole('status').innerText()).includes('One empty balanced word'));
        await shot(paths.locator('svg'),'empty-balanced-path'); await press(paths,'Reset');
        const coefficients = region('Generating coefficient construction investigation');
        assert((await coefficients.getByRole('status').innerText()).includes('x^3: 6.'));
        await press(coefficients,'Remove last factor'); assert((await coefficients.getByRole('status').innerText()).includes('x^3: 3.'));
        await press(coefficients,'Include next factor');
        for(const [label,value] of [['Station A capacity','1'],['Station B capacity','2'],['Station C capacity','2']]) await coefficients.getByLabel(label).selectOption(value);
        await coefficients.getByLabel('Target degree').selectOption('4'); assert((await coefficients.getByRole('status').innerText()).includes('x^4: 3.'));
        await shot(coefficients.locator('.counting-coefficients'),'changed-coefficient');
        await coefficients.getByLabel('Target degree').selectOption('13'); assert((await coefficients.getByRole('status').innerText()).includes('x^13: 0.'));
        await coefficients.getByLabel('Factors included').selectOption('0'); await coefficients.getByLabel('Target degree').selectOption('0');
        assert((await coefficients.getByRole('status').innerText()).includes('x^0: 1.')); await press(coefficients,'Reset'); record.states.push(['coefficients','changed/empty/outside']);
        const rotation = region('Cyclic pattern symmetry investigation');
        await press(rotation,'Inspect alternating 0101'); assert((await rotation.innerText()).includes('2 images × 2 fixing shifts = 4'));
        await press(rotation,'Rotate one site'); assert((await rotation.locator('svg').getAttribute('aria-label')).includes('1010'));
        await shot(rotation.locator('.counting-orbit-view'),'alternating-orbit');
        await rotation.getByLabel('Outcome convention').selectOption('true'); assert((await rotation.getByRole('status').innerText()).startsWith('16 outcomes'));
        for(const [sites,count] of [[3,4],[4,6],[5,8],[6,14]]) {
          await rotation.getByLabel('Ring sites').selectOption(String(sites)); await rotation.getByLabel('Outcome convention').selectOption('false');
          assert((await rotation.getByRole('status').innerText()).startsWith(`${count} outcomes`)); record.states.push(['rotations',sites,count]);
        }
        await press(rotation,'Reset');
        // A native keyboard change of a select is separate from programmatic state coverage.
        const keyboardSelect = rotation.getByLabel('Ring sites'); await keyboardSelect.focus();
        await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
        assert.equal(await keyboardSelect.inputValue(),'5'); record.keyboard.push('Ring sites ArrowDown/Enter'); await press(rotation,'Reset');
        for (const example of Object.values(examples)) {
          const program = lesson.locator('.python-example').filter({has: page.getByRole('heading',{name:example.title,exact:true})});
          assert.equal(await program.count(),1);
          const code = program.locator(':scope > div');
          for(const [position,expected] of [[0,example.code],[1,example.expected]]) assert.equal(normalize(await code.nth(position).evaluate(node => [...node.childNodes].filter(child=>child.nodeType===3).map(child=>child.textContent).join(''))),normalize(expected));
          assert.equal(normalize(await program.evaluate(node=>node.previousElementSibling.textContent)),normalize(`Before running: ${example.question}`));
          record.programs.push(example.title);
        }
        assert.equal(await lesson.locator('.python-example').count(),13);
        await shot(lesson.locator('.python-example').nth(9),'reflection-program');
        await shot(lesson.locator('.python-example').last().locator(':scope > div').last(),'capstone-output');
        for(const [index,practice] of (await lesson.locator('.counting-practice').all()).entries()) {
          const details=practice.locator(':scope > details'); assert.equal(await details.count(),2);
          assert.equal(await details.nth(1).getAttribute('open'),null);
          for(const disclosure of await details.all()) { await disclosure.locator('summary').focus(); await page.keyboard.press('Enter'); }
          if([0,2,6,7,10,11].includes(index)) await shot(practice,`practice-${index}`);
        }
        const math=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({width:node.clientWidth,scroll:node.scrollWidth,text:node.textContent})));
        record.math=math; assert(math.every(box=>box.scroll<=box.width+1),`Math overflow ${width}: ${JSON.stringify(math.filter(box=>box.scroll>box.width+1))}`);
        for(const [index,equation] of (await lesson.locator('.katex-display').all()).entries()) await shot(equation,`equation-${index}`);
        const clipped=await lesson.locator('svg').evaluateAll(nodes=>nodes.flatMap(svg=>{const rect=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].flatMap(text=>{const box=text.getBoundingClientRect();return box.left<rect.left-2||box.right>rect.right+2||box.top<rect.top-2||box.bottom>rect.bottom+2?[text.textContent]:[];});}));
        assert.deepEqual(clipped,[],`SVG clipping ${width}`);
        assert.equal(await lesson.locator('.katex-error').count(),0);
        assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)),`document overflow ${width}`);
        const sources=await lesson.locator('.lesson-sources a').evaluateAll(nodes=>nodes.map(node=>({href:node.href,target:node.target,rel:node.rel})));
        assert(sources.every(link=>link.href.startsWith('https:')&&link.target==='_blank'&&link.rel.includes('noreferrer')));
        record.sources=sources; await shot(lesson.locator('.lesson-sources'),'learning-resources');
        records.push(record);
        fs.writeFileSync(path.join(directory,'progress.json'),JSON.stringify({checkedAt:new Date().toISOString(),records,errors},null,2));
      } catch(error) {
        await page.screenshot({path:path.join(directory,`failure-${width}.png`)});
        fs.writeFileSync(path.join(directory,'failure.json'),JSON.stringify({checkedAt:new Date().toISOString(),record,errors,error:String(error)},null,2));
        throw error;
      }
      await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,records,errors},null,2));
    console.log('Counting actual-font 1440/390/320 reading, states, keyboard, anchors, programs and fit passed.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
