const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const directory = path.resolve('scratch/functional-analysis-browser');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/functional-analysis-rkhs', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.functional-analysis-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.waitForTimeout(200);
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
      };
      await capture(lesson.locator('p').filter({ hasText: 'Strictly increasing nodes make the mathematical system invertible.' }), 'native-range-explanation');
      const interpolation = lesson.locator('.python-example').filter({ hasText: 'Interpolate by slopes and by kernels' });
      assert((await interpolation.innerText()).includes('Positive interval energy underflows'));
      assert((await interpolation.innerText()).includes('kernel_energy == 0'));
      await capture(interpolation, 'interpolation-program');
      await capture(lesson.locator('p').filter({ hasText: 'Let Z contain landmarks' }), 'approximation-rank');
      const changedPractice = lesson.locator('section.lesson-check').filter({ hasText: 'H. Evaluate a changed experiment' });
      assert((await changedPractice.innerText()).includes('.155172/.177157'));
      assert((await changedPractice.innerText()).includes('.095880/.091864'));
      await capture(changedPractice.locator('details').nth(1), 'changed-capstone-solution');
      const ridge = lesson.locator('[aria-label="Kernel ridge investigation"]');
      await ridge.getByRole('combobox').selectOption('duplicates');
      await capture(ridge.locator('.functional-plot'), 'duplicate-ridge-plot');
      const mean = lesson.locator('[aria-label="Kernel distribution witness investigation"]');
      await capture(mean.locator('.functional-plot'), 'witness-plot');
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []);
      records.push({ width, exactGuardVisible: true, inverseRankQualificationVisible: true, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });
