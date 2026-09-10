const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  fs.mkdirSync('scratch/oop-foundations', { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/topic/object-oriented-programming-in-python');
      const lesson = page.locator('.oop-lesson');
      await lesson.waitFor();
      const captureLab = async (lab, name) => {
        // Fixed site navigation intersects tall element screenshots; hide only
        // for this isolated diagram capture, then restore before interaction.
        await page.locator('.learn-nav').evaluate(node => { node.style.visibility = 'hidden'; });
        try { await lab.screenshot({ path: `scratch/oop-foundations/${name}-${width}.png` }); }
        finally { await page.locator('.learn-nav').evaluate(node => { node.style.visibility = ''; }); }
      };
      assert.equal(await lesson.locator('[data-oop-lab]').count(), 4);
      assert.equal(await lesson.locator('nav[aria-label="In this lesson"] a').count(), 7);
      for (const href of await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')))) assert.equal(await page.locator(`[id="${href.slice(1)}"]`).count(), 1);

      const binding = lesson.locator('[data-oop-lab="binding"]');
      assert.equal(await binding.getByRole('button', { name: 'Back one step', exact: true }).isDisabled(), true);
      await binding.getByLabel('Receiving name', { exact: true }).selectOption('alias');
      await binding.getByLabel('Reading to add', { exact: true }).selectOption('24');
      await binding.getByRole('button', { name: 'Step forward', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.match(await binding.locator('.oop-feedback').innerText(), /alias refers to object A/);
      for (let i = 0; i < 3; i++) await binding.getByRole('button', { name: 'Step forward', exact: true }).click();
      assert.match(await binding.locator('.oop-reference-row').first().innerText(), /\[24\]/);
      assert.match(await binding.locator('.oop-reference-row').last().innerText(), /\[\]/);
      await binding.getByRole('button', { name: 'Back one step', exact: true }).click();
      assert.match(await binding.locator('.oop-reference-row').first().innerText(), /\[\]/);
      await binding.getByRole('button', { name: 'Step forward', exact: true }).click();
      await binding.getByRole('button', { name: 'Keep result; prepare another call', exact: true }).click();
      await binding.getByLabel('Receiving name', { exact: true }).selectOption('evening');
      for (let i = 0; i < 4; i++) await binding.getByRole('button', { name: 'Step forward', exact: true }).click();
      assert.match(await binding.locator('.oop-reference-row').last().innerText(), /\[24\]/);
      await captureLab(binding, 'binding');
      await binding.getByRole('button', { name: 'Reset investigation', exact: true }).click();
      assert.equal(await binding.getByLabel('Receiving name', { exact: true }).inputValue(), 'morning');

      const lookup = lesson.locator('[data-oop-lab="lookup"]');
      await lookup.getByRole('button', { name: 'Append 18 through a', exact: true }).click();
      assert.match(await lookup.locator('.oop-lookup-map__instances').innerText(), /\[18\]/);
      await lookup.getByLabel('Receiving object', { exact: true }).selectOption('B');
      await lookup.getByRole('button', { name: 'Assign [99] to b.values', exact: true }).click();
      assert.match(await lookup.locator('.oop-lookup-map__instances .oop-node').last().innerText(), /found here: stop at list B/);
      assert.match(await lookup.locator('.oop-class-node').innerText(), /Objects using C: A/);
      await captureLab(lookup, 'lookup');
      await lookup.getByRole('button', { name: 'Undo action', exact: true }).click();
      assert.match(await lookup.locator('.oop-class-node').innerText(), /Objects using C: A, B/);
      await lookup.getByLabel('Where lists begin', { exact: true }).selectOption('instance');
      await lookup.getByRole('button', { name: 'Append 18 through b', exact: true }).click();
      assert.match(await lookup.locator('.oop-lookup-map__instances .oop-node').first().innerText(), /\[\]/);
      assert.match(await lookup.locator('.oop-lookup-map__instances .oop-node').last().innerText(), /\[18\]/);

      const validation = lesson.locator('[data-oop-lab="validation"]');
      for (const [candidate, status] of [['bool', 'TypeError'], ['text', 'TypeError'], ['nan', 'ValueError'], ['infinity', 'ValueError'], ['huge', 'ValueError'], ['number', 'Accepted']]) {
        await validation.getByLabel('Candidate reading', { exact: true }).selectOption(candidate);
        await validation.getByRole('button', { name: 'Run validation', exact: true }).click();
        assert.match(await validation.locator('.oop-feedback').innerText(), new RegExp(status));
        assert.equal(await validation.locator('.oop-gate--blocked').count(), candidate === 'number' ? 0 : 1);
      }
      await captureLab(validation, 'validation');
      await validation.getByRole('button', { name: 'Reset investigation', exact: true }).click();
      assert.equal(await validation.locator('.oop-gate--passed').count(), 0);

      const composition = lesson.locator('[data-oop-lab="composition"]');
      await composition.getByLabel('Formatter object', { exact: true }).selectOption('fahrenheit');
      await composition.getByLabel('Input in degrees Celsius', { exact: true }).selectOption('0');
      for (let i = 0; i < 4; i++) await composition.getByRole('button', { name: 'Step forward', exact: true }).click();
      assert.match(await composition.locator('.oop-command').innerText(), /Reading: 32.0 F/);
      await captureLab(composition, 'composition');
      await composition.getByRole('button', { name: 'Back one step', exact: true }).click();
      assert.match(await composition.locator('.oop-command').innerText(), /waiting/);

      const mission = lesson.getByRole('region', { name: 'Independent dataset split investigation' });
      await mission.getByText('Hint: identify the invariant and the ownership boundary', { exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await mission.locator('details').first().getAttribute('open'), '');
      await mission.getByText('Show one complete solution and expected output', { exact: true }).click();
      assert.match(await mission.innerText(), /checks passed/);
      await mission.getByText('Transfer: prove the design on changed inputs', { exact: true }).click();
      assert.match(await mission.innerText(), /training and validation sets never overlap/);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Page overflow at ${width}`);
      assert.equal(await lesson.locator('a[href="/learn/topic/pandas-data-wrangling-joins-grouping"]').count(), 1);
      results.push({ width, controls: 'passed', predictionsAndReset: 'passed', keyboard: 'passed', overflow: 'none', labCount: 4 });
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync('scratch/oop-foundations/browser-verification.json', JSON.stringify({ results, pageErrors: errors }, null, 2));
    console.log(JSON.stringify({ results, pageErrors: errors }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
