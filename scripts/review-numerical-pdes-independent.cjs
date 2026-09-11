const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/numerical-pdes-independent-review/browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = text => text.replace(/\s+/g, ' ').trim();
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
(async () => {
  const startedAt = new Date().toISOString();
  const start = JSON.parse(fs.readFileSync('scratch/numerical-pdes-independent-review/start.json', 'utf8'));
  const sourceHashes = start.sourceHashes.map(({ path: file }) => ({ path: file, sha256: sha(file) }));
  const { numericalPdeExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/numerical-pde-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const record = { width, states: [], screenshots: [], errors: [], failedRequests: [], keyboard: [], programs: [] };
      page.on('pageerror', e => record.errors.push(String(e)));
      page.on('console', m => { if (m.type() === 'error' && !m.text().includes('[vite]')) record.errors.push(m.text()); });
      page.on('requestfailed', request => record.failedRequests.push({ url: request.url(), failure: request.failure() }));
      try {
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-pdes-grids-finite-elements-stability?module=math-foundations', { waitUntil: 'domcontentloaded' });
        const lesson = page.locator('.npde-lesson');
        await lesson.waitFor();
        await page.evaluate(() => document.fonts.ready);
        record.fonts = await page.evaluate(() => [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family));
        assert(record.fonts.some(f => f.includes('Space Grotesk')));
        assert(record.fonts.some(f => f.includes('JetBrains Mono')));
        await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
        const lab = name => lesson.getByRole('region', { name, exact: true });
        async function shot(target, name) {
          await target.evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
          const file = path.join(directory, `${name}-${width}.png`);
          await page.screenshot({ path: file });
          record.screenshots.push({ path: path.relative(process.cwd(), file).replaceAll('\\', '/'), sha256: sha(file) });
        }
        async function range(region, label, value) {
          const input = region.getByLabel(label, { exact: false });
          assert.equal(await input.count(), 1, label);
          await input.fill(String(value)); await input.dispatchEvent('input');
        }
        async function state(region, name, expected) {
          const text = normalize(await region.innerText());
          if (expected) assert(text.includes(expected), `${name}: ${text}`);
          const geometry = await region.locator('svg').evaluateAll(nodes => nodes.map(svg => {
            const bounds = svg.getBoundingClientRect();
            const labels = [...svg.querySelectorAll('text')].filter(t => { const r = t.getBoundingClientRect(); return r.left < bounds.left - 2 || r.right > bounds.right + 2 || r.top < bounds.top - 2 || r.bottom > bounds.bottom + 2; }).map(t => t.textContent);
            const points = [...svg.querySelectorAll('polyline')].flatMap(p => [...p.points].map(v => [v.x, v.y]));
            return { labels, pointCount: points.length, pointsInside: points.every(([x,y]) => x >= 60-1e-7 && x <= 298+1e-7 && y >= 45-1e-7 && y <= 219+1e-7) };
          }));
          assert(geometry.every(g => g.labels.length === 0 && g.pointsInside), JSON.stringify(geometry));
          assert.deepEqual(record.errors, []);
          record.states.push({ name, geometry });
        }
        async function reset(region) {
          await region.getByRole('button', { name: 'Reset', exact: true }).focus();
          await page.keyboard.press('Enter'); record.keyboard.push('Reset: ' + await region.getAttribute('aria-label'));
        }
        await shot(lesson.locator('.npde-inline'), 'restriction-reading');
        for (const details of await lesson.locator('.npde-section > details').all()) {
          await details.locator(':scope > summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await details.getAttribute('open'), null);
        }
        const p = lab('From a stencil row to a field certificate');
        await p.getByLabel('Intervals N', { exact: true }).selectOption('16');
        await p.getByLabel('Endpoint values', { exact: true }).selectOption('tilted');
        await p.getByLabel('Solve', { exact: true }).selectOption('jacobi');
        await range(p, 'Jacobi updates', 125); await range(p, 'Interior row', 15);
        await state(p, 'nonzero last boundary incomplete', 'Not yet certified');
        await shot(p.locator('.npde-stencil'), 'boundary-row');
        await p.getByLabel('Intervals N', { exact: true }).selectOption('64');
        await p.getByLabel('Solve', { exact: true }).selectOption('direct');
        await state(p, 'fine direct certificate', 'Certified within');
        await shot(p.locator('.npde-budget'), 'field-budget'); await reset(p);
        const refinement = lab('A refinement test can tell the wrong story');
        await refinement.getByLabel('Refinement target').selectOption('linear');
        assert.equal(await refinement.locator('svg').count(), 0);
        await state(refinement, 'linear exactness has undefined rate', 'undefined');
        await shot(refinement, 'linear-exactness'); await reset(refinement);
        const d = lab('Watch a mode, not just a stability badge');
        await d.getByLabel('Diffusion intervals', { exact: true }).selectOption('16');
        await d.getByLabel('Steps to time .1', { exact: true }).selectOption('4');
        await d.getByLabel('Initial sine mode', { exact: true }).selectOption('15');
        await range(d, 'Displayed time step', 3); await state(d, 'overshoot plot range', 'not met');
        await shot(d.locator('figure'), 'diffusion-overshoot');
        await d.getByLabel('Steps to time .1', { exact: true }).selectOption('64');
        await range(d, 'Displayed time step', 13); await state(d, 'refined high-mode common time', 'condition r≤.5: met');
        await reset(d);
        const material = lab('Two materials share one steady flux');
        await range(material, 'Right conductivity', .5); await range(material, 'Interface position', .8);
        await state(material, 'reversed conductivity contrast', 'q=1/(R₁+R₂)');
        await shot(material.locator('figure'), 'material-profile'); await reset(material);
        const balance = lab('A boundary gauge cannot repair missing heat');
        await range(balance, 'Right outward flux', 1.75); await state(balance, 'excess outward flux', 'Incompatible');
        await shot(balance, 'flux-mismatch'); await reset(balance); await state(balance, 'restored full balance', 'Compatible');
        const transport = lab('Move cell averages through shared faces');
        await transport.getByLabel('Velocity', { exact: true }).selectOption('-1');
        await range(transport, 'Courant number', 1); await range(transport, 'Transport step', 3);
        await state(transport, 'left exact three-cell translation', 'minimum 0, maximum 1');
        await shot(transport.locator('.npde-cell-strip'), 'left-translation');
        await transport.getByLabel('Transport scheme', { exact: true }).selectOption('centered');
        await range(transport, 'Courant number', 1.5); await range(transport, 'Transport step', 16);
        await state(transport, 'centered overshoot is visible', 'not met');
        await shot(transport.locator('figure'), 'transport-overshoot'); await reset(transport);
        const g = lab('Choose a Burgers face flux from the wave');
        await range(g, 'Left state', 2.5); await range(g, 'Right state', -2.5);
        await state(g, 'stationary entropy shock', 'no unique face value');
        await shot(g, 'stationary-face');
        await range(g, 'Left state', -2); await range(g, 'Right state', -.5);
        await state(g, 'negative fan fixed-face value', 'Face state -0.5'); await reset(g);
        const h = lab('Assemble overlapping hats; resolve a point source');
        await h.getByLabel('Load', { exact: true }).selectOption('point');
        await h.getByLabel('Finite-element mesh', { exact: true }).selectOption('nonuniform');
        await range(h, 'Selected element', 1); await state(h, 'moved interior load weights', 'Exact discretization errors');
        await shot(h.locator('.npde-table'), 'nonuniform-assembly');
        await shot(h.locator('figure').last(), 'point-field');
        await h.getByLabel('Finite-element mesh', { exact: true }).selectOption('aligned');
        await state(h, 'source-aligned reconstruction', 'field max 0; continuous L2 0; energy 0');
        await reset(h);
        const rectangle = lab('A two-dimensional neighbor is not always a vector neighbor');
        await range(rectangle, 'Horizontal index', 4); await range(rectangle, 'Vertical index', 3);
        await state(rectangle, 'upper-right boundary neighbors', 'Selected unknown index 11');
        await shot(rectangle.locator('.npde-coordinate-grid'), 'corner-indices'); await reset(rectangle);
        const tri = lab('Map a triangle, then map its gradients');
        await tri.getByLabel('Triangle geometry').selectOption('scaled');
        await state(tri, 'stretched element geometry', 'Area 3');
        await shot(tri.locator('svg'), 'triangle-gradients'); await reset(tri);
        const coarse = lab('Remove error on two resolutions');
        await coarse.getByLabel('Fine-grid error mode').selectOption('7');
        await state(coarse, 'energy falls while L2 grows', 'Projection decreases the energy norm');
        await shot(coarse.locator('figure'), 'coarse-high-mode'); await shot(coarse.locator('.npde-table'), 'coarse-norms'); await reset(coarse);
        for (const slider of await lesson.locator('input[type=range]').all()) {
          const old = await slider.inputValue(); await slider.focus();
          await page.keyboard.press(Number(old) < Number(await slider.getAttribute('max')) ? 'ArrowRight' : 'ArrowLeft');
          assert.notEqual(await slider.inputValue(), old); await slider.fill(old); await slider.dispatchEvent('input'); record.keyboard.push('Actual slider arrow');
        }
        const select = tri.getByLabel('Triangle geometry'); await select.focus(); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter'); assert.equal(await select.inputValue(), 'scaled'); record.keyboard.push('Actual select arrows');
        for (const practice of await lesson.locator('.npde-practice').all()) {
          const details = practice.locator(':scope > details'); assert.equal(await details.count(), 2);
          await details.nth(0).locator('summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await details.nth(0).getAttribute('open'), null); assert.equal(await details.nth(1).getAttribute('open'), null);
          await details.nth(1).locator('summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await details.nth(1).getAttribute('open'), null); record.keyboard.push('Independent hint then answer');
        }
        for (const example of Object.values(examples)) {
          const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
          assert.equal(await program.count(),1);
          const blocks=program.locator(':scope > div');
          for (const [index,expected] of [[0,example.code],[1,example.expected]]) assert.equal(normalize(await blocks.nth(index).evaluate(node=>[...node.childNodes].filter(c=>c.nodeType===3).map(c=>c.textContent).join(''))),normalize(expected));
          assert.equal(normalize(await program.evaluate(node=>node.previousElementSibling.textContent)),normalize('Before running: '+example.question)); record.programs.push(example.title);
        }
        await shot(lesson.locator('#npde-reconstruct'), 'reconstruction-proof');
        await shot(lesson.locator('#npde-energy'), 'energy-proof');
        await shot(lesson.locator('.npde-practice').last(), 'changed-report-question');
        await shot(lesson.locator('.python-example').last().locator(':scope > div').last(), 'changed-report-output');
        record.equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(n => ({ width:n.clientWidth,scroll:n.scrollWidth })));
        assert(record.equations.every(n=>n.scroll<=n.width+1));
        assert.equal(await lesson.locator('.katex-error').count(),0);
        assert.equal(await lesson.locator('.npde-investigation').count(),11);
        assert.equal(await lesson.locator('.npde-practice').count(),15);
        record.anchors=await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('href')));
        for(const id of record.anchors) assert.equal(await lesson.locator(id).count(),1);
        assert.equal(record.anchors.length,15);
        assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)));
        assert.deepEqual(record.errors,[]); assert.deepEqual(record.failedRequests,[]);
        records.push(record);
        fs.writeFileSync(path.join(directory,'progress.json'),JSON.stringify({records},null,2));
      } catch(error) {
        await page.screenshot({path:path.join(directory,`failure-${width}.png`)});
        fs.writeFileSync(path.join(directory,'failure.json'),JSON.stringify({record,error:String(error)},null,2)); throw error;
      } finally { await page.close(); }
    }
    for(const entry of sourceHashes) assert.equal(sha(entry.path),entry.sha256,'Stable final browser sources');
    const result={startedAt,completedAt:new Date().toISOString(),passed:true,sourceHashes,records};
    fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify({passed:true,widths:records.map(r=>({width:r.width,states:r.states.length,keyboard:r.keyboard.length,programs:r.programs.length,captures:r.screenshots.length}))}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
