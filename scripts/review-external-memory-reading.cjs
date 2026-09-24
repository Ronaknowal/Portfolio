const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/external-memory-browser');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/external-memory-algorithms-b-trees-i-o-complexity?module=data-structures-algorithms');
      const lesson = page.locator('.external-memory-lesson');
      await lesson.waitFor();
      const record = { width, captures: [] };
      const reading = async (locator, name) => {
        await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
        await page.screenshot({ path: path.join(directory, `detail-${name}-${width}.png`) });
        record.captures.push(`detail-${name}-${width}.png`);
      };
      for (let index = 0; index < 2; index++) await reading(lesson.locator('.external-calculation').nth(index), `calculation-${index+1}`);
      await reading(lesson.getByRole('heading', { name: 'Derive the height instead of memorizing it', exact: true }), 'height-proof');
      await reading(lesson.getByRole('heading', { name: 'Deletion needs a reserve before descent', exact: true }), 'deletion-proof');
      await reading(lesson.locator('.external-contrast'), 'bplus-contrast');
      await reading(lesson.locator('.python-example').nth(0), 'native-program');
      await reading(lesson.locator('.python-example').nth(0).locator('.lesson-note'), 'native-output');
      await reading(lesson.locator('.external-exercise').nth(4), 'changed-passes');
      await reading(lesson.locator('.lesson-sources'), 'sources');
      const code = lesson.locator('.python-example').nth(0).locator(':scope > div').nth(0);
      record.codeScrolling = await code.evaluate(element => ({client:element.clientWidth,scroll:element.scrollWidth,mode:getComputedStyle(element).overflowX}));
      if(record.codeScrolling.scroll > record.codeScrolling.client + 1) assert(['auto','scroll'].includes(record.codeScrolling.mode));
      const tree = lesson.getByRole('region', { name: 'B-tree page repairs', exact: true });
      await tree.getByLabel('Tree scenario').selectOption('delete');
      await reading(tree.locator('.external-current'), 'large-tree-controls');
      await reading(tree.locator('figure'), 'large-tree-pages');
      assert.equal(await tree.locator('figure').evaluate(element => element.scrollWidth > element.clientWidth + 1), false);
      if (width === 320) {
        const slider = tree.getByRole('slider');
        await slider.focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
        assert.equal(await slider.inputValue(), '1');
        await tree.getByRole('button', { name: 'Finish tree event', exact: true }).focus();
        await page.keyboard.press('Enter');
        assert((await tree.locator('.external-current').innerText()).includes('Complete delete 1'));
        await tree.getByRole('button', { name: 'Reset tree experiment', exact: true }).focus();
        await page.keyboard.press('Space');
        assert.equal(await slider.inputValue(), '8');
        const crash = lesson.getByRole('region', { name: 'Shadow page crash recovery', exact: true });
        await crash.getByLabel('Publication order').selectOption('early');
        for(let index=0;index<2;index++) {await crash.getByRole('button',{name:'Next commit stage',exact:true}).focus(); await page.keyboard.press('Enter');}
        assert((await crash.getByRole('status').innerText()).includes('missing newRoot'));
        await reading(crash.locator('figure'), 'crash-invalid');
      }
      const merge = lesson.getByRole('region', { name: 'External merge runs and transfers', exact: true });
      await reading(merge.locator('.external-runs'), 'merge-runs-final');
      assert.deepEqual(await lesson.locator('.external-calculation, .external-lab, .external-inline').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.className)), []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      results.push(record);
      await page.close();
    }
    const result = { checkedAt:new Date().toISOString(),results };
    fs.writeFileSync(path.join(directory,'reading-results.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
