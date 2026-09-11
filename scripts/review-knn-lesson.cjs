const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const normalize = value => value.replace(/\s+/g, ' ').trim();
const folder = 'scratch/knn-browser';
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
fs.mkdirSync(folder, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const record = { width, controls: 0, screenshots: [], errors: [] };
      page.on('pageerror', error => record.errors.push(String(error)));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/k-nearest-neighbors-knn?module=classical-ml', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.knn-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const labs = lesson.locator('[data-investigation]');
      assert.equal(await labs.count(), 6);
      for (const lab of await labs.all()) {
        const initial = normalize(await lab.innerText());
        for (const range of await lab.locator('input[type=range]').all()) {
          for (const boundary of ['min', 'max']) await range.fill(await range.getAttribute(boundary));
          await range.focus(); await page.keyboard.press('ArrowLeft'); record.controls++;
        }
        for (const select of await lab.locator('select').all()) {
          for (const option of await select.locator('option').evaluateAll(nodes => nodes.map(node => node.value))) await select.selectOption(option);
          await select.focus(); await page.keyboard.press('ArrowUp'); record.controls++;
        }
        await lab.getByRole('button', { name: 'Reset', exact: true }).focus(); await page.keyboard.press('Enter');
        assert.equal(normalize(await lab.innerText()), initial);
      }
      const voting = page.locator('[data-investigation="knn-neighbors"]');
      assert((await voting.locator('.knn-result').innerText()).includes('Prediction: A'));
      await voting.getByLabel('Vote weighting').selectOption('distance');
      assert((await voting.locator('.knn-result').innerText()).includes('Prediction: C'));
      await voting.getByLabel('Query x1', { exact: false }).fill('3');
      await voting.getByLabel('Query x2', { exact: false }).fill('3');
      assert((await voting.locator('.knn-result').innerText()).includes('Exact matches'));
      const units = page.locator('[data-investigation="knn-units"]');
      await units.getByLabel('Coordinate rule').selectOption('true');
      assert((await units.innerText()).includes('Nearest: R'));
      const regression = page.locator('[data-investigation="knn-regression"]');
      await regression.getByLabel('Regression query', { exact: false }).fill('8');
      assert((await regression.innerText()).includes('16.666667'));
      const kd = page.locator('[data-investigation="knn-kdtree"]');
      while (await kd.getByRole('button', { name: 'Next step', exact: true }).isEnabled()) await kd.getByRole('button', { name: 'Next step', exact: true }).click();
      await kd.locator('summary').last().press('Enter');
      const candidates = page.locator('[data-investigation="knn-candidates"]');
      await candidates.getByLabel('Candidate pool').selectOption('lose-close-c');
      assert((await candidates.innerText()).includes('recall@3 = 0.333'));
      const targets = width === 1440 ? [voting.locator('figure'), page.locator('[aria-label="KNN validation log loss from the recorded experiment"]')]
        : width === 390 ? [voting.locator('figure'), units, kd.locator('figure')]
          : [regression.locator('figure'), candidates, page.locator('[data-investigation="knn-volume"] figure')];
      for (let index = 0; index < targets.length; index++) {
        const screenshot = `${folder}/learning-state-${width}-${index}.png`;
        await targets[index].screenshot({ path: screenshot }); record.screenshots.push({ path: screenshot, sha256: sha(screenshot) });
      }
      const questions = lesson.locator('.lesson-check'); assert.equal(await questions.count(), 12);
      for (const question of await questions.all()) {
        const details = question.locator(':scope > details'); assert.equal(await details.count(), 2);
        await details.nth(0).locator('summary').press('Enter');
        assert.equal(await details.nth(1).getAttribute('open'), null);
        await details.nth(1).locator('summary').press('Enter');
      }
      await lesson.getByText('Complete executable response for the changed-data report', { exact: true }).click();
      const { knnExamples } = await import(pathToFileURL(path.resolve('src/learn/data/knn-examples.js')));
      assert.equal(await lesson.locator('.python-example').count(), 11);
      for (const example of knnExamples) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        for (const [index, expected] of [[0, example.code], [1, example.expected]]) assert.equal(normalize(await program.locator(':scope > div').nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(expected));
      }
      const anchors = await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      assert.equal(anchors.length, 13);
      for (const anchor of anchors) assert.equal(await lesson.locator(`[id="${anchor.slice(1)}"]`).count(), 1);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      record.labelIssues = await lesson.locator('.knn-plot svg').evaluateAll(nodes => nodes.flatMap(svg => {
        const owner = svg.getBoundingClientRect();
        return [...svg.querySelectorAll('text')].filter(node => {
          const rect = node.getBoundingClientRect();
          return getComputedStyle(node).fontSize !== '13px' || rect.left < owner.left - 1 || rect.right > owner.right + 1 || rect.top < owner.top - 1 || rect.bottom > owner.bottom + 1;
        }).map(node => node.textContent);
      }));
      assert.deepEqual(record.labelIssues, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      assert.deepEqual(record.errors, []);
      records.push(record); await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync('docs/teaching/evidence/knn-browser-review.json', JSON.stringify({ checkedAt: new Date().toISOString(), records, sourceHashes: ['src/learn/data/topics/k-nearest-neighbors-knn.jsx','src/learn/components/lesson-labs/KnnLabs.jsx','src/learn/components/lesson-labs/knn-labs.css'].map(path => ({path,sha256:sha(path)})) }, null, 2) + '\n');
  console.log('KNN route: desktop/mobile controls, meaningful changed states, keyboard, practice, code/output and label checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
