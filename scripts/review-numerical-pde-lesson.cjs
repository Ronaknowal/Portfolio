const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/numerical-pde-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();
(async () => {
  const { numericalPdeExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/numerical-pde-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  const files = ['src/learn/data/topics/numerical-pdes-grids-finite-elements-stability.jsx','src/learn/data/numerical-pde-models.js','src/learn/data/numerical-pde-examples.js','src/learn/components/lesson-labs/NumericalPdeLabs.jsx','src/learn/components/lesson-labs/numerical-pde-labs.css','src/learn/data/curriculum/blueprints/numerical-pdes-grids-finite-elements-stability.js'];
  const sourceHashes = Object.fromEntries(files.map(file => [file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  try {
    for (const width of (process.env.NPDE_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      const record = { width, states: [], anchors: [], keyboard: [], captures: [], programs: [] };
      try {
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-pdes-grids-finite-elements-stability?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
        const lesson = page.locator('.npde-lesson');
        await lesson.waitFor();
        await page.evaluate(() => document.fonts.ready);
        record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
        assert(record.fonts.some(font => font.includes('Space Grotesk')));
        assert(record.fonts.some(font => font.includes('JetBrains Mono')));
        await page.addStyleTag({ content: 'html { scroll-behavior:auto!important; }' });
        async function shot(target, name) {
          await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
          await page.waitForTimeout(100);
          const file = name+'-'+width+'.png';
          await page.screenshot({ path: path.join(directory,file) }); record.captures.push(file);
        }
        const lab = title => lesson.getByRole('region', { name: title, exact: true });
        async function range(region,label,value) {
          const control = region.getByLabel(label, { exact: false });
          assert.equal(await control.count(),1,label);
          await control.fill(String(value)); await control.dispatchEvent('input');
        }
        async function state(region,label,check) {
          const text = normalize(await region.innerText());
          assert(check(text),label+'\n'+text);
          const clipped = await region.locator('svg').evaluateAll(nodes=>nodes.flatMap(svg=>{const box=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].filter(text=>{const rect=text.getBoundingClientRect();return rect.left<box.left-2||rect.right>box.right+2;}).map(text=>text.textContent);}));
          assert.deepEqual(clipped,[],label+' graph tick fit');
          record.states.push(label); assert.deepEqual(errors,[]);
        }
        async function reset(region) {
          await region.getByRole('button',{name:'Reset',exact:true}).focus(); await page.keyboard.press('Enter');
          record.keyboard.push('Reset: '+await region.getAttribute('aria-label'));
        }
        await shot(lesson.locator('.lesson-intro'),'reading-intro');
        for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
          const href=await anchor.getAttribute('href'); const target=lesson.locator(href);
          assert.equal(await target.count(),1); await anchor.focus(); await page.keyboard.press('Enter');
          await page.waitForFunction(id => { const box=document.getElementById(id).getBoundingClientRect(); return box.top>=35&&box.top<=155; },href.slice(1));
          record.anchors.push(href); await shot(target,'reading-section-'+record.anchors.length);
        }
        assert.equal(record.anchors.length,15);
        for (const detail of await lesson.locator('.npde-section > details').all()) {
          await detail.locator(':scope > summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await detail.getAttribute('open'),null); record.keyboard.push('Optional section Enter');
        }
        const poisson=lab('From a stencil row to a field certificate');
        await poisson.getByLabel('Intervals N',{exact:true}).selectOption('32');
        await state(poisson,'N32 direct certified',text=>text.includes('Certified within'));
        await poisson.getByLabel('Solve',{exact:true}).selectOption('jacobi');
        await state(poisson,'zero updates not certified',text=>text.includes('Not yet certified'));
        await range(poisson,'Jacobi updates',1500);
        await state(poisson,'1500 updates certified',text=>text.includes('Certified within'));
        await shot(poisson.locator('.npde-budget'),'exact-field-budget');
        await poisson.getByLabel('Endpoint values',{exact:true}).selectOption('tilted');
        await range(poisson,'Interior row',31); await shot(poisson.locator('.npde-stencil'),'right-boundary-stencil');
        await poisson.getByLabel('Manufactured target',{exact:true}).selectOption('linear');
        await poisson.getByLabel('Solve',{exact:true}).selectOption('direct');
        await state(poisson,'linear exact mesh and nonzero boundary',text=>!text.includes('outside numeric range'));
        await shot(poisson.locator('figure'),'nonzero-boundary-field'); await reset(poisson);
        const refinement=lab('A refinement test can tell the wrong story');
        await refinement.getByLabel('Refinement target').selectOption('quadratic'); assert.equal(await refinement.locator('svg').count(),0);
        await state(refinement,'quadratic order undefined',text=>(text.match(/undefined/g)||[]).length===4);
        await shot(refinement,'exactness-trap'); await reset(refinement);
        const diffusion=lab('Watch a mode, not just a stability badge');
        await diffusion.getByLabel('Steps to time .1',{exact:true}).selectOption('2');
        await diffusion.getByLabel('Initial sine mode',{exact:true}).selectOption('7');
        await range(diffusion,'Displayed time step',1);
        await state(diffusion,'large-step high-mode oscillation',text=>text.includes('r=3.2')&&text.includes('not met'));
        await shot(diffusion.locator('figure'),'high-mode-first-step');
        await range(diffusion,'Displayed time step',2); await shot(diffusion.locator('figure'),'high-mode-second-step');
        await diffusion.getByLabel('Steps to time .1',{exact:true}).selectOption('64');
        await range(diffusion,'Displayed time step',64);
        await state(diffusion,'stable refined high-mode',text=>text.includes('condition r≤.5: met'));
        await shot(diffusion.locator('figure'),'stable-high-mode');
        await diffusion.getByLabel('Diffusion intervals',{exact:true}).selectOption('16');
        assert.equal(await diffusion.getByLabel('Initial sine mode',{exact:true}).inputValue(),'1');
        await diffusion.getByLabel('Initial sine mode',{exact:true}).selectOption('15');
        await diffusion.getByLabel('Steps to time .1',{exact:true}).selectOption('8');
        await range(diffusion,'Displayed time step',8);
        await state(diffusion,'large amplified field retains axis scale',text=>text.includes('r=3.2')&&!text.includes('outside numeric range'));
        await shot(diffusion.locator('figure'),'large-amplification-scale'); await reset(diffusion);
        const material=lab('Two materials share one steady flux');
        await range(material,'Right conductivity',1);
        await state(material,'identical materials',text=>text.includes('q=1/(R₁+R₂)=1;'));
        await range(material,'Interface position',.9);
        await state(material,'interface immaterial when k equal',text=>text.includes('q=1/(R₁+R₂)=1;'));
        await reset(material); await shot(material.locator('figure'),'series-resistance-field');
        const neumann=lab('A boundary gauge cannot repair missing heat');
        await range(neumann,'Right outward flux',0);
        await state(neumann,'incompatible flux rejected',text=>text.includes('Incompatible')&&text.includes('No steady solution'));
        await shot(neumann,'incompatible-outflow'); await reset(neumann);
        const transport=lab('Move cell averages through shared faces');
        await transport.getByLabel('Velocity',{exact:true}).selectOption('-1');
        await range(transport,'Courant number',1); await range(transport,'Transport step',3);
        await state(transport,'left exact translation',text=>text.includes('Mass hΣU=0.25')&&text.includes('condition: met'));
        await shot(transport.locator('.npde-cell-strip'),'left-shift-cell-averages');
        await transport.getByLabel('Transport scheme').selectOption('centered');
        await range(transport,'Courant number',.75); await range(transport,'Transport step',8);
        await state(transport,'centered overshoot with conserved mass',text=>text.includes('Mass hΣU=0.25')&&text.includes('minimum -'));
        await shot(transport.locator('figure'),'centered-oscillations');
        await transport.getByLabel('Initial cell averages').selectOption('sine'); await range(transport,'Transport step',16);
        await state(transport,'changed smooth cell averages',text=>!text.includes('outside numeric range'));
        await transport.getByLabel('Transport scheme').selectOption('upwind');
        await transport.getByLabel('Initial cell averages').selectOption('pulse');
        await range(transport,'Courant number',1.5); await range(transport,'Transport step',16);
        await state(transport,'upwind outside convex range is visibly unstable',text=>text.includes('condition: not met')&&text.includes('minimum -'));
        await shot(transport.locator('figure'),'upwind-outside-cfl'); await reset(transport);
        const godunov=lab('Choose a Burgers face flux from the wave');
        await range(godunov,'Left state',2); await range(godunov,'Right state',-2);
        await state(godunov,'stationary shock unique flux',text=>text.includes('Stationary shock')&&text.includes('u²/2 = 2'));
        await shot(godunov,'stationary-shock-flux'); await reset(godunov);
        const hats=lab('Assemble overlapping hats; resolve a point source');
        await hats.getByLabel('Load',{exact:true}).selectOption('point');
        await state(hats,'point between nodes',text=>text.includes('field max 0.0556'));
        await shot(hats.locator('figure').last(),'point-between-nodes');
        await hats.getByLabel('Finite-element mesh').selectOption('aligned');
        await state(hats,'aligned source exact discretization',text=>text.includes('field max 0;'));
        await shot(hats.locator('figure').last(),'source-aligned-node');
        await hats.getByLabel('Finite-element mesh').selectOption('nonuniform');
        await range(hats,'Selected element',2); await shot(hats.locator('.npde-table'),'nonuniform-matrix'); await reset(hats);
        const rectangle=lab('A two-dimensional neighbor is not always a vector neighbor');
        await range(rectangle,'Horizontal index',4); await range(rectangle,'Vertical index',2);
        await state(rectangle,'right edge has boundary rather than wrap',text=>text.includes('Selected unknown index 7')&&text.includes('(5,2): coefficient -6.25; known boundary'));
        await shot(rectangle.locator('.npde-coordinate-grid'),'rectangle-index-map'); await reset(rectangle);
        const triangle=lab('Map a triangle, then map its gradients');
        await triangle.getByLabel('Triangle geometry').selectOption('scaled');
        await state(triangle,'scaled translated triangle',text=>text.includes('Area 3')&&text.includes('-0.5, -0.3333'));
        const vertices=await triangle.locator('polygon').evaluate(node=>[...node.points].map(point=>({x:point.x,y:point.y})));
        assert(Math.abs((vertices[1].x-vertices[0].x)/(vertices[0].y-vertices[2].y)-2/3)<1e-14,'Actual 2:3 triangle aspect ratio');
        await shot(triangle,'mapped-triangle'); await reset(triangle);
        const coarse=lab('Remove error on two resolutions');
        await coarse.getByLabel('Fine-grid error mode').selectOption('7');
        await state(coarse,'high-mode two-grid norms',text=>text.includes('can gain a small smooth component'));
        await shot(coarse.locator('figure'),'coarse-high-mode'); await shot(coarse.locator('.npde-table'),'coarse-norm-comparison'); await reset(coarse);
        for (const slider of await lesson.locator('input[type=range]').all()) {
          const before=await slider.inputValue(); await slider.focus();
          await page.keyboard.press(Number(before)===Number(await slider.getAttribute('max'))?'ArrowLeft':'ArrowRight');
          assert.notEqual(await slider.inputValue(),before);
          assert(await slider.evaluate(node=>getComputedStyle(node).outlineStyle!=='none'));
          await slider.fill(before); await slider.dispatchEvent('input'); record.keyboard.push('Slider arrow');
        }
        for (const select of await lesson.locator('select').all()) {
          const original=await select.inputValue(); const second=await select.locator('option').nth(1).getAttribute('value');
          await select.selectOption({index:0}); await select.focus(); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
          assert.equal(await select.inputValue(),second); await select.selectOption(original); record.keyboard.push('Select ArrowDown');
        }
        for (const [index,practice] of (await lesson.locator('.npde-practice').all()).entries()) {
          assert.equal(await practice.locator(':scope > details').count(),2);
          for (const disclosure of await practice.locator(':scope > details').all()) {
            assert.equal(await disclosure.getAttribute('open'),null);
            await disclosure.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); assert.notEqual(await disclosure.getAttribute('open'),null);
          }
          if ([0,4,9,14].includes(index)) await shot(practice,'explained-practice-'+index);
        }
        assert.equal(await lesson.locator('.npde-practice').count(),15);
        for (const example of Object.values(examples)) {
          const program=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:example.title,exact:true})});
          assert.equal(await program.count(),1); const blocks=program.locator(':scope > div'); assert.equal(await blocks.count(),2);
          for (const [index,expected] of [[0,example.code],[1,example.expected]]) assert.equal(normalize(await blocks.nth(index).evaluate(node=>[...node.childNodes].filter(child=>child.nodeType===3).map(child=>child.textContent).join(''))),normalize(expected));
          assert.equal(normalize(await program.evaluate(node=>node.previousElementSibling.textContent)),normalize('Before running: '+example.question));
          record.programs.push(example.title);
        }
        await shot(lesson.locator('.python-example').last(),'changed-complete-program');
        await shot(lesson.locator('.python-example').last().locator(':scope > div').last(),'changed-accepted-output');
        record.math=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map((node,index)=>({index,width:node.clientWidth,scroll:node.scrollWidth,text:node.textContent})));
        assert(record.math.every(item=>item.scroll<=item.width+1),'Equation overflow '+JSON.stringify(record.math.filter(item=>item.scroll>item.width+1)));
        for (const [index,equation] of (await lesson.locator('.katex-display').all()).entries()) await shot(equation,'equation-'+index);
        await shot(lesson.locator('.npde-inline'),'restriction-reconstruction');
        for (const [index,region] of (await lesson.locator('.npde-investigation').all()).entries()) for (const [pictureIndex,picture] of (await region.locator('svg').all()).entries()) await shot(picture,'ordinary-figure-'+index+'-'+pictureIndex);
        const clipped=await lesson.locator('svg').evaluateAll(nodes=>nodes.flatMap(svg=>{const box=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].flatMap(text=>{const rect=text.getBoundingClientRect();return rect.left<box.left-2||rect.right>box.right+2||rect.top<box.top-2||rect.bottom>box.bottom+2?[text.textContent]:[];});}));
        assert.deepEqual(clipped,[],'SVG label clipping'); assert.equal(await lesson.locator('.katex-error').count(),0);
        assert(!(await lesson.innerText()).includes('outside numeric range')); assert(!(await lesson.innerText()).includes('\\u2212'));
        assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)));
        record.sources=await lesson.locator('.lesson-sources a').evaluateAll(nodes=>nodes.map(node=>({href:node.href,target:node.target,rel:node.rel})));
        assert(record.sources.every(link=>link.href.startsWith('https:')&&link.target==='_blank'&&link.rel.includes('noreferrer')));
        await shot(lesson.locator('.lesson-sources'),'annotated-resources'); assert.deepEqual(errors,[]);
        records.push(record); fs.writeFileSync(path.join(directory,'progress.json'),JSON.stringify({checkedAt:new Date().toISOString(),records,errors},null,2));
      } catch(error) {
        await page.screenshot({path:path.join(directory,'failure-'+width+'.png')});
        fs.writeFileSync(path.join(directory,'failure.json'),JSON.stringify({checkedAt:new Date().toISOString(),record,errors,error:String(error)},null,2)); throw error;
      }
      await page.close();
    }
    assert(files.every(file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')===sourceHashes[file]),'Sources remained stable during review');
    fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,records,errors,sourceHashes},null,2));
    console.log('Numerical PDE actual-font controls, keyboard, anchors, programs, mathematics and reading captures passed.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
