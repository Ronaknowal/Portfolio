const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');

(async () => {
  fs.mkdirSync('scratch/python-foundations', { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const evidence = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      await page.goto('http://127.0.0.1:5173/learn/topic/python-basics-types-control-flow-functions-modules');
      await page.locator('[data-pyf-lab="references"]').waitFor();
      assert.equal(await page.locator('[data-pyf-lab]').count(), 3);
      const anchors = await page.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(n => n.hash.slice(1)));
      for (const id of anchors) assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      const refs = page.locator('[data-pyf-lab="references"]');
      const nextRef = refs.getByRole('button', { name: 'Next step', exact: true });
      await nextRef.focus();
      await page.keyboard.press('Enter');
      assert.match(await refs.getByRole('status').textContent(), /creates list A/);
      for (let i = 0; i < 3; i++) await nextRef.click();
      assert.match(await refs.locator('.pyf-output').textContent(), /\[18, 21, 24\] \[18, 21, 24\]/);
      await refs.getByRole('button', { name: 'Back', exact: true }).click();
      assert.match(await refs.getByRole('status').textContent(), /append changes A/);
      await refs.getByLabel('Make the backup').selectOption('copy');
      assert.equal(await refs.getByRole('button', { name: 'Back', exact: true }).isDisabled(), true);
      for (let i = 0; i < 4; i++) await nextRef.click();
      assert.match(await refs.locator('.pyf-output').textContent(), /\[18, 21\] \[18, 21, 24\]/);
      await refs.screenshot({ path: `scratch/python-foundations/references-${width}.png` });
      const flow = page.locator('[data-pyf-lab="flow"]');
      await flow.getByLabel('Keep readings at or above').selectOption('0');
      const nextFlow = flow.getByRole('button', { name: 'Next step', exact: true });
      while (await nextFlow.isEnabled()) await nextFlow.click();
      assert.match(await flow.getByRole('status').textContent(), /\[18, 25, 31, 0\]/);
      await flow.screenshot({ path: `scratch/python-foundations/flow-${width}.png` });
      await flow.getByRole('button', { name: 'Reset', exact: true }).click();
      assert.match(await flow.getByRole('status').textContent(), /begins empty/);
      const calls = page.locator('[data-pyf-lab="calls"]');
      await calls.getByLabel("Function's last instruction").selectOption('print');
      await calls.getByLabel('Input temperature').selectOption('100');
      const nextCall = calls.getByRole('button', { name: 'Next step', exact: true });
      for (let i = 0; i < 5; i++) await nextCall.click();
      assert.equal((await calls.locator('.pyf-output pre').textContent()).trim(), '212.0\nNone');
      await calls.getByRole('button', { name: 'Back', exact: true }).click();
      assert.match(await calls.getByRole('status').textContent(), /reaches its end without return/);
      await calls.screenshot({ path: `scratch/python-foundations/calls-${width}.png` });
      await calls.getByLabel("Function's last instruction").selectOption('return');
      for (let i = 0; i < 5; i++) await nextCall.click();
      assert.equal((await calls.locator('.pyf-output pre').textContent()).trim(), '212.0');
      await page.locator('.pyf-mission').first().getByText('Complete changed-input caller and reasoning', { exact: true }).click();
      assert.equal(await page.getByText("{'count': 3, 'mean': 2.0}", { exact: false }).count() > 0, true);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert.equal(overflow, false, `page overflow at ${width}`);
      assert.deepEqual(errors, []);
      evidence.push({ width, labs: 3, keyboardStepping: true, resetAndBack: true, completeOutcomes: true, anchors: anchors.length, overflow: false, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync('scratch/python-foundations/browser-results.json', JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify(evidence, null, 2));
})().catch(error => { console.error(error); process.exit(1); });
