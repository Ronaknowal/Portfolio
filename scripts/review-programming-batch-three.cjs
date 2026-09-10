const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');

(async () => {
  const { plottingExamples, notebookExamples, apiExamples } = await Promise.all([import("../src/learn/data/plotting-examples.js"), import("../src/learn/data/notebook-examples.js"), import("../src/learn/data/api-design-examples.js")]).then(modules => Object.assign({}, ...modules));
  const { plottingNewExamples } = await import("../src/learn/data/plotting-practice-examples.js");
  const groups = [
    ['matplotlib-scientific-plotting', {...plottingExamples,...plottingNewExamples}],
    ['reproducible-notebooks-experiment-structure', notebookExamples],
    ['code-documentation-type-hints-api-design', apiExamples],
  ];
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    fs.mkdirSync('scratch/programming-three-browser', { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      for (const [slug, examples] of groups) {
        await page.goto('http://127.0.0.1:5173/learn/topic/' + slug);
        await page.locator('.lesson-intro').waitFor();
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
        for (const example of Object.values(examples)) assert.ok(outputs.some(output => output.includes(example.output)), 'Missing output: ' + slug);
        if (slug.startsWith('matplotlib')) {
          await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=true));
          assert.equal(await page.locator('.lesson-plot img').count(), 9);
          for (const img of await page.locator('.lesson-plot img').all()) {
            await img.scrollIntoViewIfNeeded();
            await img.evaluate(image => image.decode());
            assert.ok(await img.evaluate(image => image.naturalWidth > 0 && image.alt.length > 20));
            const response = await page.request.get(new URL(await img.getAttribute('src'), page.url()).href);
            assert.equal(response.status(), 200);
            assert.match(await response.text(), /<svg/);
          }
          // Default view fits; explicit enlargement scrolls within the figure.
          const scroll = page.locator('.lesson-plot__scroll').first();
          assert.ok(await scroll.evaluate(node => node.scrollWidth <= node.clientWidth + 1));
          await page.getByRole('button', { name: 'Enlarge chart', exact: true }).first().click();
          assert.ok(await scroll.evaluate(node => node.scrollWidth > node.clientWidth));
          await scroll.focus();
          await page.keyboard.press('ArrowRight');
          await page.waitForTimeout(200);
          assert.ok(await scroll.evaluate(node => node.scrollLeft > 0));
          await page.getByRole('button', { name: 'Fit chart', exact: true }).focus();
          await page.keyboard.press('Enter');
          assert.ok(await scroll.evaluate(node => node.scrollWidth <= node.clientWidth + 1));
          await page.locator('.lesson-plot').first().screenshot({ path: 'scratch/programming-three-browser/plot-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
        } else if (slug.startsWith('reproducible')) {
          const lab = page.getByRole('region', { name: 'Notebook state explorer', exact: true });
          const result = lab.locator('.lesson-results');
          await lab.getByRole('button', { name: 'Compute cost', exact: true }).click();
          assert.match(await result.innerText(), /NameError: rate/);
          await lab.getByRole('button', { name: 'Set rate to 2', exact: true }).click();
          await lab.getByRole('button', { name: 'Compute cost', exact: true }).click();
          await lab.getByRole('button', { name: 'Set rate to 3', exact: true }).click();
          await lab.getByRole('button', { name: 'Display cost', exact: true }).click();
          assert.match(await result.innerText(), /→ 20/);
          await lab.getByRole('button', { name: 'Compute cost', exact: true }).click();
          await lab.getByRole('button', { name: 'Display cost', exact: true }).click();
          assert.match(await result.innerText(), /→ 30/);
          await lab.getByRole('button', { name: 'Restart kernel', exact: true }).focus();
          await page.keyboard.press('Enter');
          await lab.getByRole('button', { name: 'Display cost', exact: true }).click();
          assert.match(await result.innerText(), /NameError: cost/);
          for (const link of await page.locator('a[download]').all()) {
            const response = await page.request.get(new URL(await link.getAttribute('href'), page.url()).href);
            assert.equal(response.status(), 200);
            const notebook = await response.json();
            assert.equal(notebook.nbformat, 4);
            assert.equal(notebook.cells.filter(cell => cell.cell_type === 'code').length, 3);
          }
          await lab.screenshot({ path: 'scratch/programming-three-browser/notebook-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
        } else {
          await page.locator('.python-example').last().locator('.python-example__output').screenshot({ path: 'scratch/programming-three-browser/api-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
        }
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), slug + ': page overflow');
        console.log(width + 'px: ' + slug + ': examples, anchors, solutions and visual checks passed');
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
