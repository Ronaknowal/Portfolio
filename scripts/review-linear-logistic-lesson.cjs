const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const folder = 'scratch/linear-logistic-browser';
fs.mkdirSync(folder, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const record = { width, errors: [], screenshots: [], controls: [], keyboard: [] };
      page.on('pageerror', error => record.errors.push(String(error)));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/linear-logistic-regression?module=classical-ml', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.linear-logistic-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(record.fonts.some(name => name.includes('Space Grotesk')));
      assert(record.fonts.some(name => name.includes('JetBrains Mono')));
      await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
      async function shot(target, name) {
        await target.evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const file = path.join(folder, `${name}-${width}.png`);
        await page.screenshot({ path: file }); record.screenshots.push({ path: file.replaceAll('\\', '/'), sha256: sha(file) });
      }
      await shot(lesson.locator('.regression-protocol'), 'information-reading');
      const labs = lesson.locator('[data-investigation]');
      assert.equal(await labs.count(), 5);
      for (const lab of await labs.all()) {
        const name = await lab.getAttribute('data-investigation');
        const initial = normalize(await lab.innerText());
        for (const control of await lab.locator('input[type=range]').all()) {
          for (const boundary of ['min', 'max']) { await control.fill(await control.getAttribute(boundary)); await control.dispatchEvent('input'); }
          await control.focus(); await page.keyboard.press('ArrowLeft'); record.keyboard.push(`${name}: range arrow`);
          record.controls.push({ name, type: 'range', min: await control.getAttribute('min'), max: await control.getAttribute('max') });
        }
        for (const control of await lab.locator('select').all()) {
          for (const value of await control.locator('option').evaluateAll(nodes => nodes.map(node => node.value))) await control.selectOption(value);
          record.controls.push({ name, type: 'select' });
        }
        if (name === 'regression-gradient') {
          for (let step = 0; step < 16; step++) await lab.getByRole('button', { name: 'Next step', exact: true }).click();
          assert((await lab.innerText()).includes('outside the fixed plot window'));
          assert(await lab.getByRole('button', { name: 'Next step', exact: true }).isDisabled());
        }
        if (name === 'regression-residuals') {
          await lab.getByRole('button', { name: 'Fit least squares', exact: true }).click();
          assert((await lab.innerText()).includes('0.100 + 2.100x'));
          await lab.getByRole('button', { name: 'Use mean baseline', exact: true }).click();
          assert((await lab.innerText()).includes('3.250 + 0.000x'));
        }
        await shot(lab, `${name}-changed`);
        const reset = lab.getByRole('button', { name: 'Reset', exact: true });
        await reset.focus(); await page.keyboard.press('Enter'); record.keyboard.push(`${name}: reset Enter`);
        assert.equal(normalize(await lab.innerText()), initial, `${name} reset`);
      }
      const threshold = lesson.locator('[data-investigation="regression-threshold"]');
      await threshold.getByLabel('Warning threshold', { exact: false }).fill('1');
      assert((await threshold.innerText()).includes('precision undefined'));
      await threshold.getByRole('button', { name: 'Reset', exact: true }).click();
      const score = lesson.locator('[data-investigation="regression-logistic-score"]');
      const before = await score.locator('.regression-calculation').innerText();
      await score.getByLabel('Observed label', { exact: false }).selectOption('0');
      assert.equal(await score.locator('.regression-calculation').innerText(), before);
      await score.getByRole('button', { name: 'Reset', exact: true }).click();
      await shot(lesson.getByRole('img', { name: 'Uncertainty about a mean and about one future outcome', exact: true }), 'uncertainty-reading');
      const questions = lesson.locator('.lesson-check');
      assert.equal(await questions.count(), 12);
      for (const question of await questions.all()) {
        const details = question.locator(':scope > details');
        assert.equal(await details.count(), 2);
        assert.equal(await details.nth(1).getAttribute('open'), null);
        await details.nth(0).locator('summary').focus(); await page.keyboard.press('Enter');
        assert.notEqual(await details.nth(0).getAttribute('open'), null);
        assert.equal(await details.nth(1).getAttribute('open'), null);
        await details.nth(1).locator(':scope > summary').focus(); await page.keyboard.press('Enter');
      }
      record.headings = await lesson.locator('h2').allTextContents();
      await lesson.getByText('Compare with one completed capstone report after your attempt', { exact: true }).click();
      const { linearLogisticExamples: examples } = await import(require('node:url').pathToFileURL(path.resolve('src/learn/data/linear-logistic-examples.js')));
      assert.equal(await lesson.locator('.python-example').count(), examples.length);
      for (const example of examples) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await program.count(), 1);
        const blocks = program.locator(':scope > div');
        for (const [index, expected] of [[0, example.code], [1, example.expected]]) assert.equal(normalize(await blocks.nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(expected));
      }
      const anchors = await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      for (const anchor of anchors) assert.equal(await lesson.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
      record.programs = examples.length; record.anchors = anchors;
      assert.equal(record.headings.length, 14);
      record.geometry = await lesson.locator('.regression-chart-scroll svg').evaluateAll(nodes => nodes.map(svg => {
        const bounds = svg.getBoundingClientRect();
        return { title: svg.getAttribute('aria-label'), outsideLabels: [...svg.querySelectorAll('text')].filter(node => { const r = node.getBoundingClientRect(); return r.left < bounds.left - 1 || r.right > bounds.right + 1 || r.top < bounds.top - 1 || r.bottom > bounds.bottom + 1; }).map(node => node.textContent) };
      }));
      assert(record.geometry.every(item => !item.outsideLabels.length), JSON.stringify(record.geometry));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      record.documentWidth = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
      assert(record.documentWidth.scroll <= record.documentWidth.client + 1, JSON.stringify(record.documentWidth));
      assert.deepEqual(record.errors, []);
      records.push(record); await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync('docs/teaching/evidence/linear-logistic-browser-review.json', JSON.stringify({ checkedAt: new Date().toISOString(), browser: 'Edge, actual loaded fonts', sourceHashes: ['src/learn/data/topics/linear-logistic-regression.jsx','src/learn/components/lesson-labs/LinearLogisticLabs.jsx','src/learn/components/lesson-labs/linear-logistic-labs.css','src/learn/data/linear-logistic-models.js','src/learn/data/linear-logistic-examples.js'].map(file => ({ path: file, sha256: sha(file) })), records }, null, 2) + '\n');
  console.log('Regression actual-font 1440/390/320 controls, keyboard, geometry and practice checks passed. Open the recorded images for visual review.');
})().catch(error => { console.error(error); process.exitCode = 1; });
