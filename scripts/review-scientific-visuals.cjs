const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const { pivotModel } = await import('../src/learn/data/scientific-visual-models.js');
  const { publicationTrace } = await import("../src/learn/data/scientific-file-models.js");
  const directory = 'scratch/scientific-visual-review';
  fs.mkdirSync(directory, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [], results = [];
  const topics = [
    ['numpy-arrays-broadcasting-vectorization', '.sci-shape-comparison'],
    ['scientific-file-formats-schemas-reliable-data-i-o', '.sci-csv-boundaries'],
    ['sql-relational-data-transactions-for-ml', '.sci-relations'],
    ['pandas-data-wrangling-joins-grouping', '.sci-pivot-lab'],
    ['matplotlib-scientific-plotting', '.sci-plot-ownership'],
  ];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      page.on('pageerror', error => errors.push(error.message));
      const capture = async (element, filename) => {
        // Suppress fixed navigation only during tall isolated-figure captures.
        await page.locator('.learn-nav').evaluate(node => { node.style.visibility = 'hidden'; });
        try { await element.screenshot({ path: `${directory}/${filename}-${width}.png` }); }
        finally { await page.locator('.learn-nav').evaluate(node => { node.style.visibility = ''; }); }
      };
      for (const [id, selector] of topics) {
        await page.goto(`http://127.0.0.1:5173/learn/path/full-curriculum/${id}?module=programming-scientific-computing`);
        const visual = page.locator(selector);
        await visual.waitFor({ state: 'visible' });
        await visual.scrollIntoViewIfNeeded();
        await page.evaluate(() => document.fonts.ready);
        await capture(visual, id);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${id}: document overflow at ${width}`);
        assert.equal(await visual.evaluate(element => element.scrollWidth > element.clientWidth + 1), false, `${id}: visual overflow at ${width}`);
        if (id.startsWith('numpy')) {
          assert.equal(await visual.locator('.sci-reading-cells').count(), 3);
          assert.equal(await visual.locator('.sci-reading-cells').nth(1).locator('strong').allTextContents().then(values => values.join(',')), '18,24,30,20,26,32');
        }
        if (id.startsWith('sql')) {
          assert.equal(await visual.locator('.sci-relation-lane').count(), 3);
          assert.equal(await visual.locator('.sci-observation-branch > div').count(), 3);
          assert.match(await visual.innerText(), /B's r3 exists with an unknown value/);
        }
        if (id.startsWith('scientific')) {
          assert.equal(await visual.locator('.sci-field-row > div > span').count(), 9);
          const lab = page.locator('[data-lab="file-publication"]');
          for (const strategy of ['replace', 'direct']) for (const fail of [true, false]) {
            await lab.getByLabel('Write strategy').selectOption(strategy);
            await lab.getByLabel('Writer outcome').selectOption(fail ? 'fail' : 'success');
            const trace = publicationTrace(strategy, fail);
            for (let step = 0; step < trace.length; step++) {
              const artifacts = lab.locator('.sci-file-artifact');
              for (const [index, contents] of [trace[step].destination, trace[step].temporary].entries()) {
                const rows = contents ? contents.trim().split('\n').slice(1) : [];
                assert.equal(await artifacts.nth(index).locator('li').count(), rows.length);
                for (let row = 0; row < rows.length; row++) assert.match(await artifacts.nth(index).locator('li').nth(row).innerText(), new RegExp(rows[row].split(',')[1] + ' °C'));
              }
              if (step < trace.length - 1) await lab.getByRole('button', { name: 'Next step', exact: true }).click();
            }
          }
          await lab.getByLabel('Write strategy').selectOption('replace');
          await lab.getByRole('button', { name: 'Next step', exact: true }).click();
          await lab.getByRole('button', { name: 'Next step', exact: true }).click();
          await capture(lab, 'file-publication');
          const details = lab.getByText('Inspect exact published text', { exact: true });
          await details.focus(); await page.keyboard.press('Enter');
          assert.equal(await lab.locator('.data-state pre').first().isVisible(), true);
        }
        if (id.startsWith('pandas')) {
          const buttons = visual.locator('.sci-pivot-grid button');
          for (const variation of ['unique', 'duplicate', 'missing']) for (const operation of ['pivot', 'sum', 'mean']) {
            await visual.getByLabel('Source records').selectOption(variation);
            await visual.getByLabel('Reshape rule').selectOption(operation);
            const model = pivotModel(variation, operation);
            for (let cell = 0; cell < 4; cell++) {
              await buttons.nth(cell).click();
              assert.equal(await visual.locator('.sci-pivot-records .is-linked').count(), model.cells[cell].sources.length);
              assert.equal(await buttons.nth(cell).getAttribute('aria-pressed'), 'true');
              const expected = model.cells[cell].conflict ? '2 → 1?' : model.blocked ? '—' : String(model.cells[cell].value ?? 'NA');
              assert.equal(await buttons.nth(cell).locator('b').innerText(), expected);
            }
          }
          await visual.getByLabel('Source records').selectOption('duplicate');
          await visual.getByLabel('Reshape rule').selectOption('pivot');
          await buttons.nth(0).focus(); await page.keyboard.press('Enter');
          assert.equal(await buttons.nth(0).getAttribute('aria-pressed'), 'true');
          assert.match(await visual.locator('[role="status"]').innerText(), /returns no partial table/);
          await capture(visual, 'pivot-collision');
          await visual.getByRole('button', { name: 'Reset pivot' }).focus(); await page.keyboard.press('Enter');
          assert.equal(await visual.getByLabel('Source records').inputValue(), 'unique');
          assert.equal(await visual.getByLabel('Reshape rule').inputValue(), 'pivot');
          assert.equal(await buttons.nth(0).getAttribute('aria-pressed'), 'true');
        }
        results.push({ id, width, visual: 'rendered and captured', checks: 'content, no document/visual overflow; relevant controls and keyboard states passed' });
      }
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ results, errors }, null, 2));
    console.log(JSON.stringify({ passed: true, viewports: [1440, 390], topics: 5, pivotConfigurationsPerViewport: 9, cellsPerConfiguration: 4, publicationTracesPerViewport: 4, errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
