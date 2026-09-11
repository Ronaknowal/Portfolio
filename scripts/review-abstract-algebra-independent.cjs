const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = 'scratch/abstract-algebra-independent';
const payload = JSON.parse(fs.readFileSync(`${directory}/payload.json`, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const normalize = value => value.replaceAll('\r\n', '\n').trim();

(async () => {
  for (const source of payload.sources) assert.equal(hash(source.path), source.sha256);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], deliberatelyDisconnectedHmrMessages = [];
      page.on('pageerror', error => errors.push(String(error)));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        if (message.text().startsWith('[vite] failed to connect to websocket.')) deliberatelyDisconnectedHmrMessages.push(message.text());
        else errors.push(message.text());
      });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/abstract-algebra-groups-symmetry-actions?module=math-foundations');
      const lesson = page.locator('.abstract-algebra-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')) && fonts.some(font => font.includes('JetBrains')));
      const record = { width, fonts, captures: [], anchors: [], operatedStates: [] };
      async function shot(target, name) {
        await target.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const file = `${directory}/${name}-${width}.png`; await page.screenshot({ path: file }); record.captures.push({ path: file, sha256: hash(file), opened: false });
      }
      for (const anchor of await lesson.locator('nav a').all()) {
        const id = (await anchor.getAttribute('href')).slice(1);
        await anchor.focus(); await page.keyboard.press('Enter');
        await page.waitForFunction(id => { const rect = document.getElementById(id).getBoundingClientRect(); return rect.top >= 35 && rect.top <= 160; }, id);
        record.anchors.push(id);
      }
      assert.equal(record.anchors.length, 11);
      const composition = lesson.locator('[data-algebra-lab="composition"]');
      await composition.getByLabel('Outer move g', { exact: true }).selectOption('5');
      await composition.getByLabel('Inner move h', { exact: true }).selectOption('6');
      await composition.getByLabel('Track starting vertex', { exact: true }).selectOption('2');
      assert.match(await composition.getByRole('status').innerText(), /vertex is 1 in the first lane and 3 in the second/);
      await composition.getByRole('button', { name: 'Previous stage', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.match(await composition.getByRole('status').innerText(), /vertex is 0 in the first lane and 3 in the second/);
      await shot(composition.locator('.algebra-route').first(), 'changed-composition');
      record.operatedStates.push('rs after r²s; reversed order; intermediate tracked vertex');
      const graph = lesson.locator('.algebra-cayley');
      const graphScroll = graph.locator('..');
      await graphScroll.focus(); await page.keyboard.press('ArrowRight');
      await shot(graphScroll, 'cayley-local-scroll');
      await shot(lesson.locator('.algebra-action-graph'), 'action-state-loops');

      const orbit = lesson.locator('[data-algebra-lab="orbits"]');
      await orbit.getByLabel('Available colors', { exact: true }).selectOption('3');
      for (const [index, value] of [0,1,1,2].entries()) await orbit.getByLabel(`Color at vertex ${index}`, { exact: true }).selectOption(String(value));
      assert.match(await orbit.getByRole('status').innerText(), /8 distinct states/);
      assert.match(await orbit.locator('.algebra-result').innerText(), /21 classes/);
      await shot(orbit.locator('.algebra-orbit-collection'), 'changed-chiral-orbit');
      await orbit.getByLabel('Allowed symmetries', { exact: true }).selectOption('rotations');
      assert.match(await orbit.getByRole('status').innerText(), /4 distinct states/);
      assert.match(await orbit.locator('.algebra-result').innerText(), /24 classes/);
      await shot(orbit.locator('.algebra-result'), 'rotation-fixed-count');
      record.operatedStates.push('changed ternary coloring; D4 versus C4 orbit and total counts');

      const cosets = lesson.locator('[data-algebra-lab="cosets"]');
      await cosets.getByLabel('Representative from H', { exact: true }).selectOption('1');
      assert.match(await cosets.getByRole('status').innerText(), /output block changed/);
      await shot(cosets.locator('.algebra-product-witness'), 'representative-failure');
      await cosets.getByLabel('Subgroup H', { exact: true }).selectOption('rotations');
      await cosets.getByLabel('Representative from H', { exact: true }).selectOption('2');
      assert.match(await cosets.getByRole('status').innerText(), /same output block/);
      await shot(cosets, 'normal-representative');
      record.operatedStates.push('nonnormal and normal representative choices');

      const map = lesson.locator('[data-algebra-lab="equivariance"]');
      await map.getByLabel('Sensor readings', { exact: true }).fill('-20, .25, 19.75, -7.5');
      await map.getByRole('button', { name: 'Apply readings', exact: true }).click();
      await map.getByLabel('Data transformation g', { exact: true }).selectOption('7');
      for (const mode of ['raw','rotations','averaged','shift']) {
        await map.getByLabel('Processing map', { exact: true }).selectOption(mode);
        const expected = payload.states.find(state => state.values[0] === -20 && state.mode === mode && state.g === 7);
        const status = await map.getByRole('status').innerText();
        assert(status.includes(`[${expected.difference.join(', ')}] V`));
        assert(status.includes(`Current-input defect: ${expected.inputSpecificDefect} V`));
        await shot(map.locator('.algebra-route').first(), `changed-map-${mode}`);
        record.operatedStates.push(`changed negative/fractional readings, ${mode} map, r³s`);
      }
      const before = await map.getByRole('status').innerText();
      await map.getByLabel('Sensor readings', { exact: true }).fill('1,2,3,4.1');
      await map.getByRole('button', { name: 'Apply readings', exact: true }).click();
      assert(await map.getByRole('alert').isVisible()); assert.equal(await map.getByRole('status').innerText(), before);
      await shot(map, 'invalid-draft-preserves-state');
      await map.getByRole('button', { name: 'Reset sensor map', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await map.getByRole('alert').count(), 0);
      assert.equal(await map.getByLabel('Sensor readings', { exact: true }).inputValue(), '1, 2, 4, 8');
      record.operatedStates.push('invalid draft preserves calculation; keyboard reset');
      const deeper = lesson.locator('details').filter({ has: page.locator('summary').filter({ hasText: 'Deeper: why this is the nearest tied matrix' }) });
      await deeper.locator('summary').focus(); await page.keyboard.press('Enter'); await shot(deeper, 'projection-proof');
      for (const summary of await lesson.locator('.lesson-check summary').all()) { await summary.focus(); await page.keyboard.press('Enter'); }
      assert.equal(await lesson.locator('.lesson-check').count(), 10);
      await shot(lesson.locator('.lesson-check').last(), 'changed-audit-answer');
      await shot(lesson.locator('.algebra-figure').last(), 'modular-preimages');

      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, question: node.previousElementSibling.textContent.replace(/^Before running:\s*/, ''), blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('')) })));
      assert.equal(programs.length, 10);
      for (const program of programs) {
        const example = payload.examples.find(row => row.title === program.title);
        assert.equal(program.question, example.question);
        assert.equal(normalize(program.blocks[0]), normalize(example.code)); assert.equal(normalize(program.blocks[1]), normalize(example.expected));
      }
      await shot(lesson.locator('.python-example').last(), 'actual-program');
      await shot(lesson.locator('.lesson-sources'), 'references');
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth })));
      assert(equations.every(row => row.scroll <= row.width + 2));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      Object.assign(record, { actualPrograms: programs.length, equations: equations.length, errors, deliberatelyDisconnectedHmrMessages, documentOverflow: false });
      records.push(record); await page.close();
    }
    for (const source of payload.sources) assert.equal(hash(source.path), source.sha256);
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', sources: payload.sources, records }, null, 2));
    console.log(JSON.stringify(records.map(({ width, operatedStates, actualPrograms, equations }) => ({ width, states: operatedStates.length, actualPrograms, equations })), null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
