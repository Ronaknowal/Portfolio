const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const folder = 'scratch/decision-tree-browser';
const normalize = value => value.replace(/\s+/g, ' ').trim();
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
fs.mkdirSync(folder, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const record = { width, errors: [], controls: 0, screenshots: [] };
      page.on('pageerror', error => record.errors.push(String(error)));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/decision-trees-random-forests?module=classical-ml', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.tree-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')) && fonts.some(font => font.includes('JetBrains Mono')));
      await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
      async function shot(target, name) {
        await target.evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        const file = path.join(folder, `${name}-${width}.png`);
        await page.screenshot({ path: file }); record.screenshots.push({ path: file.replaceAll('\\', '/'), sha256: sha(file) });
      }
      const labs = lesson.locator('[data-investigation]');
      assert.equal(await labs.count(), 7);
      for (const lab of await labs.all()) {
        const initial = normalize(await lab.innerText());
        const id = await lab.getAttribute('data-investigation');
        for (const range of await lab.locator('input[type=range]').all()) {
          for (const boundary of ['min', 'max']) await range.fill(await range.getAttribute(boundary));
          await range.focus(); await page.keyboard.press('ArrowLeft'); record.controls++;
        }
        for (const select of await lab.locator('select').all()) {
          for (const option of await select.locator('option').evaluateAll(nodes => nodes.map(node => node.value))) await select.selectOption(option);
          await select.focus(); await page.keyboard.press('ArrowUp'); record.controls++;
        }
        for (const details of await lab.locator('details').all()) {
          await details.locator(':scope > summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await details.getAttribute('open'), null);
          await details.locator(':scope > summary').press('Enter');
        }
        if (width === 320 && ['tree-pruning', 'tree-bootstrap', 'tree-permutation'].includes(id)) await shot(lab, id + '-changed');
        await lab.getByRole('button', { name: 'Reset', exact: true }).focus(); await page.keyboard.press('Enter');
        assert.equal(normalize(await lab.innerText()), initial, id+' reset');
      }
      const partition = lesson.locator('[data-investigation="tree-partitions"]');
      await partition.getByLabel('Query x1', { exact: false }).fill('4.5');
      assert((await partition.innerText()).includes('go left'));
      await partition.getByLabel('Maximum depth', { exact: false }).selectOption('3');
      await shot(partition, 'partition-equality');
      const xor = lesson.locator('[data-investigation="tree-xor"]');
      await xor.getByLabel('Zero-gain policy').selectOption('true');
      assert((await xor.innerText()).includes('4/4'));
      if (width === 390) await shot(xor, 'xor-interaction');
      const bootstrap = lesson.locator('[data-investigation="tree-bootstrap"]');
      await bootstrap.getByLabel('Number of trees', { exact: false }).fill('1');
      await bootstrap.getByLabel('Inspect original row', { exact: false }).selectOption('1');
      assert((await bootstrap.innerText()).includes('unavailable — no eligible tree'));
      const practice = lesson.locator('.lesson-check'); assert.equal(await practice.count(), 13);
      for (const question of await practice.all()) {
        const details = question.locator(':scope > details'); assert.equal(await details.count(), 2);
        await details.nth(0).locator('summary').press('Enter');
        assert.equal(await details.nth(1).getAttribute('open'), null);
        await details.nth(1).locator(':scope > summary').press('Enter');
      }
      await lesson.getByText('Complete executable response for the changed-data report', { exact: true }).click();
      const { decisionTreeExamples } = await import(require('node:url').pathToFileURL(path.resolve('src/learn/data/decision-tree-examples.js')));
      assert.equal(await lesson.locator('.python-example').count(), 10);
      for (const example of decisionTreeExamples) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const blocks = program.locator(':scope > div');
        for (const [index, expected] of [[0, example.code], [1, example.expected]]) assert.equal(normalize(await blocks.nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(expected));
      }
      const anchors = await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      for (const anchor of anchors) assert.equal(await lesson.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
      assert.equal(anchors.length, 14); assert.equal(await lesson.locator('.lesson-sources a').count(), 7);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      record.outsideLabels = await lesson.locator('.tree-plot-scroll svg').evaluateAll(nodes => nodes.flatMap(svg => {
        const b = svg.getBoundingClientRect();
        return [...svg.querySelectorAll('text')].filter(node => { const r = node.getBoundingClientRect(); return r.left < b.left-1 || r.right > b.right+1 || r.top < b.top-1 || r.bottom > b.bottom+1; }).map(node => node.textContent);
      }));
      assert.deepEqual(record.outsideLabels, []);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth+1);
      assert.equal(overflow, false);
      assert.deepEqual(record.errors, []);
      records.push(record); await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync('docs/teaching/evidence/decision-tree-browser-review.json', JSON.stringify({ checkedAt:new Date().toISOString(), records, sources:['src/learn/data/topics/decision-trees-random-forests.jsx','src/learn/components/lesson-labs/DecisionTreeLabs.jsx','src/learn/components/lesson-labs/decision-tree-labs.css'].map(file=>({path:file,sha256:sha(file)}))},null,2)+'\n');
  console.log('Trees: actual-font desktop/390/320 controls, changed outcomes, keyboard, practice, code/output pairs, anchors and geometry passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
