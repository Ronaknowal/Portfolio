const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');

(async () => {
  const { pandasExamples } = await import('../src/learn/data/pandas-examples.js');
  const { pandasNewExamples } = await import("../src/learn/data/pandas-practice-examples.js");
  const examples={...pandasExamples,...pandasNewExamples};
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    fs.mkdirSync('scratch/pandas-lesson', { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('http://127.0.0.1:5173/learn/topic/pandas-data-wrangling-joins-grouping');
      const lab = page.getByRole('region', { name: 'Pandas join explorer', exact: true });
      await lab.waitFor();
      assert.equal(await page.locator('.lesson-guide').count(), 0);
      const ids = await page.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
      for (const id of ids) assert.equal(await page.locator('[id="' + id + '"]').count(), 1, id);
      assert.ok(await page.locator('.lesson-pilot code').evaluateAll(nodes => nodes.every(node => node.textContent.trim())));
      for (const details of await page.locator('.lesson-check details').all()) {
        await details.evaluate(n=>{for(let p=n.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
        await details.locator('summary').click();
        assert.equal(await details.getAttribute('open'), '');
      }
      const outputs = await page.locator('.python-example__output').allTextContents();
      assert.equal(outputs.length, Object.keys(examples).length);
      for (const example of Object.values(examples)) assert.ok(outputs.some(output => output.includes(example.output)), 'Missing rendered example output');
      const resultTable = lab.getByRole('table', { name: 'Join result: read the match indicator', exact: true });
      const checkRows = async count => {
        assert.match(await lab.locator('.lesson-results').innerText(), new RegExp(count + ' result rows'));
        assert.equal(await resultTable.locator('tbody tr').count(), count);
      };
      await checkRows(3);
      for (const [mode, count] of [['inner', 2], ['outer', 4]]) {
        await lab.getByLabel('Join type', { exact: true }).selectOption(mode);
        await checkRows(count);
      }
      assert.match(await resultTable.innerText(), /right_only/);
      await lab.getByLabel('Add a second C1 lookup row', { exact: true }).check();
      assert.match(await lab.locator('.lesson-results').innerText(), /MergeError/);
      assert.equal(await resultTable.count(), 0);
      await lab.getByLabel('Enforce many-to-one validation', { exact: true }).uncheck();
      for (const [mode, count] of [['outer', 5], ['left', 4], ['inner', 3]]) {
        await lab.getByLabel('Join type', { exact: true }).selectOption(mode);
        await checkRows(count);
      }
      await lab.getByRole('button', { name: 'Reset join', exact: true }).focus();
      await page.keyboard.press('Enter');
      await checkRows(3);
      assert.equal(await lab.getByLabel('Add a second C1 lookup row', { exact: true }).isChecked(), false);
      assert.equal(await lab.getByLabel('Enforce many-to-one validation', { exact: true }).isChecked(), true);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Page overflow');
      await lab.screenshot({ path: 'scratch/pandas-lesson/join-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
      console.log(width + 'px: outputs, anchors, solutions, join modes, validation and keyboard reset passed');
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
