const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const directory = path.resolve('scratch/graph-fundamentals-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { graphFundamentalsExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/graph-fundamentals-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const errors = [];
  const environmentMessages = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        if (message.location().url.endsWith('/@vite/client') && message.text().startsWith('[vite] failed to connect to websocket.')) environmentMessages.push({width,text:message.text()});
        else errors.push(message.text());
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/graph-fundamentals-adjacency-laplacian-connectivity?module=math-foundations');
      const lesson = page.locator('.graph-fundamentals-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const record = { width, captures: [], anchors: [], controls: [], programs: [] };
      const shot = async (element, name) => {
        await element.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 90));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        // Let the browser finish rasterizing a long-page jump before recording pixels.
        await page.waitForTimeout(150);
        const filename = name + '-' + width + '.png';
        await page.screenshot({ path: path.join(directory, filename) });
        record.captures.push(filename);
      };
      const press = async (region, name, key = 'Enter') => {
        const button = region.getByRole('button', { name, exact: true });
        await page.keyboard.press('Tab');
        await button.focus();
        assert(await button.evaluate(node => node === document.activeElement && getComputedStyle(node).outlineStyle !== 'none'), name + ' visible focus');
        await page.keyboard.press(key);
      };
      const setRange = async (region, name, key) => {
        const slider = region.getByRole('slider', {name,exact:true});
        await slider.focus();
        await page.keyboard.press(key);
      };
      await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
      for (const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await link.getAttribute('href');
        await link.focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(hash => location.hash === hash, href);
        const target = lesson.locator('[id="' + href.slice(1) + '"]');
        await page.waitForFunction(id => {
          const y=document.getElementById(id).getBoundingClientRect().top;
          return y>=45 && y<=140;
        }, href.slice(1));
        const top = await target.evaluate(node => node.getBoundingClientRect().top);
        record.anchors.push({href,top});
        if ([1,4,6,8,9].includes(record.anchors.length)) await shot(target,'ordinary-section-'+record.anchors.length);
      }
      assert.equal(record.anchors.length,10);
      for (const figure of await lesson.locator('[data-graph-figure]').all()) await shot(figure, await figure.getAttribute('data-graph-figure'));
      for (const investigation of await lesson.locator('[data-investigation]').all()) await shot(investigation, await investigation.getAttribute('data-investigation')+'-initial');
      assert.equal(await lesson.locator('[data-investigation]').count(),5);
      assert.equal(await lesson.locator('[data-graph-figure]').count(),4);

      const walks = lesson.locator('[data-investigation="graph-matrix-walks"]');
      assert((await walks.locator('.gf-result').innerText()).includes('= 2.'));
      await walks.getByLabel('Walk destination',{exact:true}).selectOption('0');
      assert((await walks.locator('.gf-result').innerText()).includes('= 4.'));
      await walks.getByLabel('Walk length',{exact:true}).selectOption('0');
      assert((await walks.locator('.gf-result').innerText()).includes('= 1.'));
      await walks.getByLabel('Walk destination',{exact:true}).selectOption('2');
      assert((await walks.locator('.gf-result').innerText()).includes('= 0.'));
      await walks.getByLabel('Directed arrows, including C→B',{exact:true}).check();
      await walks.getByLabel('Add C–D bridge, weight 1',{exact:true}).check();
      await walks.getByLabel('Walk start',{exact:true}).selectOption('2');
      await walks.getByLabel('Walk destination',{exact:true}).selectOption('1');
      await walks.getByLabel('Walk length',{exact:true}).selectOption('1');
      assert((await walks.locator('.gf-result').innerText()).includes('= 1.'));
      await shot(walks.locator('.gf-linked'), 'directed-linked-matrix');
      await press(walks,'Reset graph and walk','Space');
      record.controls.push('weighted return, zero-step identity/absence, directed reverse edge and reset');

      const energy = lesson.locator('[data-investigation="graph-edge-energy"]');
      assert((await energy.locator('.gf-result').innerText()).includes('9'));
      await press(energy,'Make each component constant');
      assert((await energy.locator('.gf-result').innerText()).includes('Edge sum = 0'));
      await setRange(energy,'Bridge C–D weight','ArrowRight');
      assert((await energy.locator('.gf-result').innerText()).includes('Edge sum = 4'));
      await shot(energy.locator('.gf-energy-bars'),'bridge-energy-change');
      await press(energy,'Reset energy');
      await setRange(energy,'Signal at B','ArrowRight');
      assert((await energy.locator('.gf-row-equation').innerText()).includes('Σ = 0'));
      assert((await energy.locator('.gf-result').innerText()).includes('Edge sum = 6'));
      await shot(energy.locator('.gf-row-equation'),'local-cancellation');
      await press(energy,'Reset energy');
      record.controls.push('componentwise constants, positive bridge, signed local cancellation with nonzero energy, keyboard ranges/reset');

      const normalization = lesson.locator('[data-investigation="graph-normalization"]');
      await normalization.getByLabel('Graph normalization operator',{exact:true}).selectOption('naiveIdentity');
      assert.equal(await normalization.locator('td.gf-selected').innerText(),'1');
      await normalization.getByLabel('Graph normalization operator',{exact:true}).selectOption('symmetric');
      assert.equal(await normalization.locator('td.gf-selected').innerText(),'0');
      await normalization.getByLabel('Graph normalization operator',{exact:true}).selectOption('transition');
      assert.equal(await normalization.locator('td.gf-selected').innerText(),'1');
      await normalization.getByLabel('Add a unit self-loop at B',{exact:true}).check();
      await shot(normalization.locator('.gf-linked'),'loop-and-isolate');
      await normalization.getByLabel('Graph normalization operator',{exact:true}).selectOption('symmetric');
      await normalization.getByLabel('Normalization signal',{exact:true}).selectOption('root');
      record.normalizedRoot = await normalization.locator('.gf-node-values').last().innerText();
      await press(normalization,'Reset normalization');
      record.controls.push('product-versus-naive isolated diagonals, stochastic hold, loop and root-degree signal');

      const averaging = lesson.locator('[data-investigation="graph-averaging"]');
      await press(averaging,'Apply one update');
      assert((await averaging.locator('.gf-result').innerText()).includes('ordinary mean 2'));
      await averaging.getByLabel('Averaging rule',{exact:true}).selectOption('neighbor');
      await press(averaging,'Apply one update');
      assert((await averaging.locator('.gf-result').innerText()).includes('ordinary mean 1;'));
      await press(averaging,'Apply one update');
      assert((await averaging.locator('.gf-result').innerText()).includes('ordinary mean 2;'));
      await press(averaging,'Inspect step 24');
      await shot(averaging.locator('.gf-linked'),'alternating-neighbor');
      await averaging.getByLabel('Averaging rule',{exact:true}).selectOption('lazy');
      await press(averaging,'Inspect step 24');
      assert((await averaging.locator('.gf-result').innerText()).includes('ordinary mean 1.5;'));
      await averaging.getByLabel('Averaging rule',{exact:true}).selectOption('exchange');
      await averaging.getByLabel('Exchange step size',{exact:true}).selectOption('0.75');
      await press(averaging,'Inspect step 24');
      await shot(averaging.locator('.gf-linked'),'growing-discrete-mode');
      await press(averaging,'Previous update');
      assert((await averaging.locator('.gf-result').innerText()).includes('Step 23'));
      await press(averaging,'Reset averaging');
      record.controls.push('ordinary-versus-weighted mean, two-cycle, lazy damping, growing beyond-bound mode, previous/reset');

      const anchors = lesson.locator('[data-investigation="graph-harmonic-anchors"]');
      assert((await anchors.locator('.gf-result').innerText()).includes('Unanchored component'));
      assert((await anchors.locator('.gf-node-values').first().innerText()).includes('undetermined'));
      await anchors.getByLabel('Anchor D at 2',{exact:true}).check();
      assert((await anchors.locator('.gf-result').innerText()).includes('unique harmonic'));
      await anchors.getByLabel('Anchor D at 2',{exact:true}).uncheck();
      await anchors.getByLabel('Connect C–D with weight 1',{exact:true}).check();
      assert((await anchors.locator('.gf-result').innerText()).includes('unique harmonic'));
      await setRange(anchors,'Anchor C','ArrowRight');
      await shot(anchors.locator('.gf-network'),'connected-interpolation');
      await press(anchors,'Reset anchors');
      record.controls.push('partial undetermined result, independent anchor/bridge repairs, changed boundary and reset');

      for (const program of await lesson.locator('.python-example').all()) {
        const title = await program.locator('h3').innerText();
        const example = Object.values(examples).find(candidate => candidate.title === title);
        assert(example,title);
        const previous = await program.evaluate(node => node.previousElementSibling.textContent);
        assert.equal(normalize(previous),normalize('Before running: '+example.question));
        const blocks = program.locator(':scope > div');
        assert(normalize(await blocks.nth(0).innerText()).includes(normalize(example.code)));
        assert.equal(normalize(await blocks.nth(1).innerText()).replace(/^OUTPUT /,''),normalize(example.expected));
        record.programs.push(title);
      }
      assert.equal(record.programs.length,11);
      const firstProgram = lesson.locator('.python-example').first();
      await shot(firstProgram,'visible-question-program');
      await shot(firstProgram.locator('.lesson-note'),'visible-program-output');
      for (const task of await lesson.locator('.gf-practice').all()) {
        const hint=task.locator('details').nth(0), solution=task.locator('details').nth(1);
        assert.equal(await hint.getAttribute('open'),null);
        assert.equal(await solution.getAttribute('open'),null);
        await hint.locator('summary').focus(); await page.keyboard.press('Enter');
        assert.equal(await solution.getAttribute('open'),null);
        await solution.locator('summary').focus(); await page.keyboard.press('Space');
        assert((await solution.innerText()).length>150);
      }
      assert.equal(await lesson.locator('.gf-practice').count(),10);
      await shot(lesson.locator('.gf-practice').nth(6),'changed-harmonic-solution');
      await lesson.locator('details').evaluateAll(elements=>elements.forEach(element=>{element.open=true}));
      record.equations=[];
      for(const [index,equation] of (await lesson.locator('.katex-display').all()).entries()){
        record.equations.push(await equation.evaluate(element=>({text:element.textContent,client:element.clientWidth,scroll:element.scrollWidth})));
        if(width===320)await shot(equation,'equation-'+(index+1));
      }
      await shot(lesson.locator('.lesson-sources'),'sources');
      record.links=await lesson.locator('.lesson-sources a').evaluateAll(elements=>elements.map(element=>({href:element.href,target:element.target,rel:element.rel})));
      assert.equal(record.links.length,7);
      assert(record.links.every(link=>link.target==='_blank'&&link.rel.includes('noreferrer')));
      record.fonts=await page.evaluate(()=>Array.from(document.fonts).map(face=>({family:face.family,status:face.status})));
      for(const family of ['Space Grotesk','JetBrains Mono'])assert(record.fonts.some(face=>face.family.replaceAll('"','')===family&&face.status==='loaded'),'Font '+family);
      record.overflow=await lesson.locator('[data-investigation],[data-graph-figure],.katex-display').evaluateAll(elements=>elements.filter(element=>element.scrollWidth>element.clientWidth+2).map(element=>({className:element.className,client:element.clientWidth,scroll:element.scrollWidth,text:element.textContent.slice(0,150)})));
      record.svgTextOverflow=await lesson.locator('svg text').evaluateAll(elements=>elements.flatMap(element=>{const box=element.getBoundingClientRect(),svg=element.ownerSVGElement.getBoundingClientRect();return box.left<svg.left-1||box.right>svg.right+1||box.top<svg.top-1||box.bottom>svg.bottom+1?[element.textContent]:[]}));
      record.pageOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      record.mathErrors=await lesson.locator('.katex-error').count();
      records.push(record);
      fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),records,errors,environmentMessages},null,2));
      await page.close();
    }
    assert.deepEqual(errors,[]);
    assert(records.every(record=>!record.pageOverflow&&!record.mathErrors&&!record.overflow.length&&!record.svgTextOverflow.length),JSON.stringify(records.map(record=>({width:record.width,overflow:record.overflow,svg:record.svgTextOverflow}))));
    console.log(JSON.stringify({checkedAt:new Date().toISOString(),allPassed:true,viewports:records.map(record=>({width:record.width,anchors:record.anchors.length,programs:record.programs.length,equations:record.equations.length,captures:record.captures.length,controls:record.controls}))},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
