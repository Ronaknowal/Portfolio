const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/multioutput/browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();
const files = ['src/learn/data/topics/multi-label-multi-output-learning.jsx', 'src/learn/data/multioutput-models.js', 'src/learn/data/multioutput-examples.js', 'src/learn/components/lesson-labs/MultioutputLabs.jsx', 'src/learn/components/lesson-labs/MultioutputFigures.jsx', 'src/learn/components/lesson-labs/multioutput-labs.css', 'src/learn/data/curriculum/blueprints/multi-label-multi-output-learning.js'];
(async () => {
  const { multioutputExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/multioutput-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const sourceHashes = Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  const records = [], errors = [];
  try {
    for (const width of (process.env.MULTIOUTPUT_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      const record = { width, states: [], keyboard: [], anchors: [], captures: [], programs: [] };
      records.push(record);
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/multi-label-multi-output-learning?module=classical-ml-supervised', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.multioutput-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(record.fonts.some(font => font.includes('Space Grotesk')));
      assert(record.fonts.some(font => font.includes('JetBrains Mono')));
      await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
      const lab = name => lesson.getByRole('region', { name, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const filename = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, filename) });
        record.captures.push(filename);
      }
      async function state(region, name, predicate) {
        const text = normalize(await region.innerText());
        assert(predicate(text), `${name}: ${text}`);
        const outside = await region.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => { const box = svg.getBoundingClientRect(); return [...svg.querySelectorAll('text')].filter(text => { const rect = text.getBoundingClientRect(); return rect.left < box.left - 2 || rect.right > box.right + 2; }).map(text => text.textContent); }));
        assert.deepEqual(outside, [], `${name} SVG labels`);
        record.states.push(name);
      }
      async function keyboard(control, key, label) { await control.focus(); await page.keyboard.press(key); record.keyboard.push(label || key); }
      async function setRange(region, label, value) {
        const input = region.getByLabel(label, { exact: false });
        assert.equal(await input.count(), 1);
        await input.evaluate((node, next) => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next)); node.dispatchEvent(new Event('input', { bubbles: true })); node.dispatchEvent(new Event('change', { bubbles: true })); }, value);
      }
      async function reset(region) { await keyboard(region.getByRole('button', { name: 'Reset', exact: true }), 'Enter', 'Reset '+await region.getAttribute('aria-label')); }
      // Initial reading and actual arrival, without a screenshot per anchor.
      for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await anchor.getAttribute('href');
        const target = lesson.locator(`[id="${href.slice(1)}"]`);
        assert.equal(await target.count(), 1);
        await keyboard(anchor, 'Enter', 'Anchor '+href);
        await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 35 && box.top <= 180; }, href.slice(1));
        record.anchors.push(href);
      }
      assert.equal(record.anchors.length, 14);
      await shot(lesson.locator('.mo-inline-figure').first(), 'target-schema');
      if (width === 390) await shot(lesson.locator('.mo-inline-figure').nth(1), 'split-reading');
      const grid = lab('Read errors by message and by label');
      await state(grid, 'initial cell and row denominators', text => text.includes('5 / 1 / 2') && text.includes('0.7692'));
      await keyboard(grid.getByRole('button', { name: 'Message 4, hardware, observed 0, predicted 1', exact: true }), 'Enter', 'Correct prediction cell');
      await state(grid, 'corrected false positive', text => text.includes('5 / 0 / 2') && text.includes('0.8333'));
      await keyboard(grid.getByLabel('Message 4 hardware annotation is unknown'), 'Space', 'Unknown annotation');
      await state(grid, 'masked denominators', text => text.includes('Observed decisions 11') && text.includes('uses 3 fully annotated'));
      await shot(grid, 'masked-error-grid'); await reset(grid);
      const joint = lab('One joint table, three different decisions');
      await state(joint, 'known joint marginal and greedy differ', text => text.includes('Greedy AB 00') && text.includes('All joint modes 11'));
      await joint.getByLabel('Chain order', { exact: true }).selectOption('BA');
      await state(joint, 'reverse greedy path', text => text.includes('Greedy BA 11'));
      await shot(joint, 'reverse-probability-tree');
      for (const label of ['00','01','10','11']) await setRange(joint, 'Count '+label, 0);
      await state(joint, 'no probability distribution', text => text.includes('at least one positive count'));
      await setRange(joint, 'Count 11', 3);
      await state(joint, 'zero support branch', text => text.includes('undefined') && text.includes('All joint modes 11'));
      await reset(joint);
      const association = lab('Unmix a co-occurrence table');
      await setRange(association, 'Fraction in high-probability group', .25);
      await state(association, 'changed mixture conditional', text => text.includes('Pooled P(B=1) = 0.3') && text.includes('= 0.7'));
      if (width === 1440) await shot(association, 'conditional-mosaics');
      await reset(association);
      const threshold = lab('Move a decision gate across tied scores');
      await setRange(threshold, 'Validation threshold', .8);
      await state(threshold, 'tied scores stay together', text => text.includes('1 / 2 / 2'));
      await shot(threshold, 'threshold-ties');
      await setRange(threshold, 'Validation threshold', 1.01);
      await state(threshold, 'no selected positives', text => text.includes('0 / 0 / 3') && text.includes('F1 0'));
      await reset(threshold);
      const regression = lab('A shared split must serve two measurements');
      await regression.getByLabel('Energy divisor', { exact: true }).selectOption('100');
      await state(regression, 'shared target scaling', text => text.includes('Best common split: 1.5') && text.includes('5000'));
      await shot(regression, 'shared-split-units');
      await keyboard(regression.getByLabel('Separate output stumps'), 'Space', 'Separate stumps');
      await state(regression, 'separate output splits', text => text.includes('temperature 1.5, energy .5'));
      await reset(regression);
      // Open optional theory before its investigation, then check every range by keyboard.
      for (const detail of await lesson.locator('.mo-optional').all()) {
        await keyboard(detail.locator(':scope > summary'), 'Enter', 'Optional theory');
        assert.notEqual(await detail.getAttribute('open'), null);
      }
      const shrink = lab('Shrink one feature across two outputs');
      await state(shrink, 'group penalty exact coefficients', text => text.includes('Green grouped (1.8, 2.4)') && text.includes('Rose separate (1, 2)'));
      await setRange(shrink, 'Penalty λ', 8);
      await state(shrink, 'entire feature removed', text => text.includes('Green grouped (0, 0)'));
      await reset(shrink);
      if (width === 320) await shot(shrink, 'grouped-coefficients');
      for (const input of await lesson.locator('input[type="range"]').all()) {
        await keyboard(input, 'Home', 'Range Home'); await keyboard(input, 'ArrowRight', 'Range increment');
      }
      for (const region of await lesson.locator('.mo-investigation').all()) await reset(region);
      for (const section of await lesson.locator('.mo-practice').all()) {
        const details = section.locator(':scope > details');
        assert.equal(await details.count(), 2);
        assert.equal(await details.first().locator('summary').innerText(), 'Get a hint');
        for (const detail of await details.all()) {
          assert.equal(await detail.getAttribute('open'), null);
          await keyboard(detail.locator(':scope > summary'), 'Enter', 'Practice hint or explanation');
          assert.notEqual(await detail.getAttribute('open'), null);
        }
      }
      for (const example of Object.values(examples)) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await program.count(), 1);
        assert.equal(normalize(await program.evaluate(node => node.previousElementSibling.textContent)), normalize('Before running: '+example.question));
        const code = await program.locator(':scope > div').evaluateAll(nodes => nodes.map(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join('')));
        assert.equal(normalize(code[0]), normalize(example.code)); assert.equal(normalize(code[1]), normalize(example.expected));
        record.programs.push(example.title);
      }
      if (width === 390) { await shot(lesson.locator('.mo-practice').nth(2), 'changed-joint-practice'); await shot(lesson.locator('.python-example').last().locator(':scope > div').nth(1), 'changed-report-output'); }
      if (width === 320) await shot(lesson.locator('.python-example').first(), 'first-program');
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.getClientRects().length).map(node => ({ text: node.textContent, scroll: node.scrollWidth, client: node.clientWidth })));
      record.equations = equations.length;
      record.equationOverflow = equations.filter(equation => equation.scroll > equation.client + 2);
      assert.deepEqual(record.equationOverflow, [], 'equation overflow');
      const horizontal = await lesson.evaluate(node => { const box = node.getBoundingClientRect(); return [...node.querySelectorAll('p,h2,h3,summary,svg,figure,input,select,button,table')].filter(element => element.getClientRects().length && !element.closest('.lesson-table-wrap,.mo-table-scroll')).filter(element => { const rect = element.getBoundingClientRect(); return rect.right > box.right + 3 || rect.left < box.left - 3; }).map(element => ({ tag: element.tagName, text: element.textContent.slice(0,100) })); });
      record.horizontalOverflow = horizontal;
      assert.deepEqual(horizontal, [], 'reading/control overflow');
      assert.deepEqual(errors, []);
      record.references = await lesson.locator('.lesson-sources a').count();
      await page.close();
    }
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ timestamp: new Date().toISOString(), sourceHashes, records, errors, passed: true }, null, 2));
    console.log(JSON.stringify(records.map(record => ({ width: record.width, states: record.states.length, keyboard: record.keyboard.length, anchors: record.anchors.length, programs: record.programs.length, equations: record.equations, captures: record.captures })), null, 2));
  } catch (error) {
    fs.writeFileSync(path.join(directory, 'latest-failure.json'), JSON.stringify({ timestamp: new Date().toISOString(), sourceHashes, records, errors, failure: error.stack }, null, 2));
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
