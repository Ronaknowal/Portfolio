const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = 'scratch/decision-theory-independent';
const packet = JSON.parse(fs.readFileSync(`${directory}/fixtures.json`, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const normalize = value => value.replaceAll('\r\n', '\n').trim();

(async () => {
  for (const source of packet.sources) assert.equal(hash(source.path), source.sha256);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], expectedHmrMessages = [];
      page.on('pageerror', error => errors.push(String(error)));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        if (message.text().startsWith('[vite] failed to connect to websocket.')) expectedHmrMessages.push(message.text());
        else errors.push(message.text());
      });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/decision-theory-risk-cost-sensitive-decisions?module=math-foundations');
      const lesson = page.locator('.decision-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')) && fonts.some(font => font.includes('JetBrains')));
      const record = { width, fonts, states: [], captures: [], programs: [], anchors: [] };
      const regions = lesson.locator('.decision-investigation');
      assert.equal(await regions.count(), 8);
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const file = `${name}-${width}.png`; await page.screenshot({ path: `${directory}/${file}` }); record.captures.push(file);
      }
      async function range(region, name, value) {
        const control = region.locator('label.decision-control').filter({ hasText: name }).locator('input');
        assert.equal(await control.count(), 1); await control.fill(String(value)); await control.dispatchEvent('input');
        record.states.push({ name, value, result: await region.locator('.decision-result').innerText() });
      }
      try {
        for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
          const id = (await anchor.getAttribute('href')).slice(1);
          await anchor.focus(); await page.keyboard.press('Enter');
          await page.waitForFunction(id => { const rect = document.getElementById(id).getBoundingClientRect(); return rect.top >= 35 && rect.top <= 155; }, id);
          record.anchors.push(id);
        }
        const conditional = regions.nth(0);
        const inputs = conditional.getByRole('spinbutton');
        for (const [index, value] of [0, 60, 12, 12].entries()) await inputs.nth(index).fill(String(value));
        await range(conditional, 'Probability faulty', 0.2);
        assert((await conditional.locator('.decision-result').innerText()).includes('tie'));
        await shot(conditional.locator('svg'), 'changed-loss-envelope');
        const procedure = regions.nth(1);
        await range(procedure, 'Prior probability of high state', 0.8);
        await procedure.locator('select').focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
        assert.equal(await procedure.locator('select').inputValue(), '1');
        assert((await procedure.locator('.decision-result').innerText()).includes('Always high or Follow signal'));
        await shot(procedure.locator('.decision-state-columns'), 'changed-state-weighting');
        const provisioning = regions.nth(2);
        await range(provisioning, 'Underage cost', 1); await range(provisioning, 'Order quantity', 2);
        assert((await provisioning.locator('.decision-result').innerText()).includes('Expected cost 2.8.'));
        await shot(provisioning.locator('svg'), 'changed-quantity-criterion');
        const capacity = regions.nth(3);
        await range(capacity, 'Probability faulty in fallback example', 0.075);
        assert((await capacity.innerText()).includes('Predict sound or Use fallback'));
        assert((await capacity.innerText()).includes('samples 101 probabilities'));
        await range(capacity, 'Quarantine slots for all six items', 3);
        assert((await capacity.locator('.decision-result').innerText()).includes('loss 44.4.'));
        await shot(capacity.locator('.decision-region'), 'fallback-sampling-contract');
        await shot(capacity.locator('.decision-items'), 'three-slot-allocation');
        const information = regions.nth(4);
        await range(information, 'Prior faulty probability', 0.02);
        assert((await information.locator('.decision-result').innerText()).includes('0.14'));
        await information.getByRole('checkbox').focus(); await page.keyboard.press('Space');
        assert(await information.getByRole('checkbox').isChecked());
        await shot(information.locator('.decision-tree'), 'changed-information-timing');
        const mixture = regions.nth(5);
        await range(mixture, 'Probability of choosing rule A', 0.5);
        assert((await mixture.locator('.decision-result').innerText()).includes('worst 3.'));
        await shot(mixture.locator('svg'), 'off-optimum-mixture');
        const tail = regions.nth(6);
        await range(tail, 'Tail level alpha', 0.8);
        assert((await tail.locator('.decision-result').innerText()).includes('CVaR 32.5;'));
        await shot(tail.locator('.decision-tail'), 'changed-partial-atom');
        const capstone = regions.nth(7);
        await range(capstone, 'Available quarantine slots', 3);
        assert.equal(await capstone.locator('.decision-candidate-costs p').filter({ hasText: 'optimal' }).count(), 1);
        assert((await capstone.locator('.decision-result').innerText()).includes('37.4'));
        await shot(capstone.locator('.decision-branches'), 'changed-capacity-three-branches');
        await range(capstone, 'Available quarantine slots', 2);
        await range(capstone, 'Price of the one test', 10.4);
        assert.equal(await capstone.locator('.decision-candidate-costs p').filter({ hasText: 'optimal' }).count(), 3);
        for (const region of await regions.all()) {
          const button = region.getByRole('button', { name: 'Reset', exact: true });
          await page.keyboard.press('Tab'); await button.focus();
          assert(await button.evaluate(node => getComputedStyle(node).outlineStyle !== 'none'));
          await page.keyboard.press('Enter');
        }
        for (const details of await lesson.locator('details').all()) {
          if (await details.getAttribute('open') === null) { await details.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); }
          assert.notEqual(await details.getAttribute('open'), null);
        }
        assert((await lesson.innerText()).includes('no finite threshold need attain it'));
        for (const [index, figure] of (await lesson.locator('figure').all()).entries()) await shot(figure, `inline-${index}`);
        for (const example of Object.values(packet.examples)) {
          const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
          assert.equal(await program.count(), 1);
          for (const [index, expected] of [[0, example.code], [1, example.expected]]) assert.equal(normalize(await program.locator(':scope > div').nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(expected));
          assert.equal(normalize(await program.evaluate(node => node.previousElementSibling.textContent)), normalize(`Before running: ${example.question}`));
          record.programs.push(example.title);
        }
        await shot(lesson.locator('.decision-practice').last(), 'changed-practice');
        await shot(lesson.locator('.python-example').last().locator(':scope > div').last(), 'actual-output');
        await shot(lesson.locator('.lesson-sources'), 'alternate-resources');
        record.equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth })));
        assert(record.equations.every(node => node.scroll <= node.width + 1));
        assert.equal(await lesson.locator('.katex-error').count(), 0);
        assert.equal(await lesson.locator('.decision-practice').count(), 14);
        assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
        const clipped = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => { const box = svg.getBoundingClientRect(); return [...svg.querySelectorAll('text')].filter(node => { const r = node.getBoundingClientRect(); return r.left < box.left - 2 || r.right > box.right + 2 || r.top < box.top - 2 || r.bottom > box.bottom + 2; }).map(node => node.textContent); }));
        assert.deepEqual(clipped, []); assert.deepEqual(errors, []);
        record.errors = errors; record.deliberatelyDisconnectedHmrMessages = expectedHmrMessages; records.push(record);
      } catch (error) { await page.screenshot({ path: `${directory}/failure-${width}.png` }); throw error; }
      await page.close();
    }
    for (const source of packet.sources) assert.equal(hash(source.path), source.sha256);
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sources: packet.sources, records }, null, 2));
    console.log('Decision independent changed-state, ordinary reading, native program rendering and keyboard checks passed at three widths.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
