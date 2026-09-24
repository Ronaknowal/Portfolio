const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/bitwise-author/browser';
fs.mkdirSync(directory, { recursive: true });
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');

(async () => {
  const { arrayMapExamples } = await import('../src/learn/data/array-map-foundations-examples.js');
  const { bitwiseExamples } = await import('../src/learn/data/bitwise-foundations-examples.js');
  const native = JSON.parse(fs.readFileSync('scratch/bitwise-author/native-results.json'));
  const sources = native.sources.map(({ path }) => ({ path, sha256: hash(path) }));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], failures = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error', 'warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      page.on('requestfailed', request => { if (!request.failure()?.errorText.includes('ERR_ABORTED')) failures.push([request.url(), request.failure()?.errorText]); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/arrays-strings-hash-maps?module=data-structures-algorithms', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.lesson-pilot').first();
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      let operatedStates = 0, openedDisclosures = 0;
      async function capture(locator, name) {
        await locator.first().evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 112, behavior: 'instant' }));
        await page.waitForTimeout(100);
        const bounds = await locator.first().boundingBox();
        const atPageEnd = await page.evaluate(() => scrollY + innerHeight >= document.documentElement.scrollHeight - 2);
        assert(bounds && bounds.y > 40 && (bounds.y < 180 || (atPageEnd && bounds.y < 900)), `${name}: stale scroll position ${JSON.stringify(bounds)}`);
        const path = `${directory}/${name}-${width}.png`;
        await page.screenshot({ path });
        images.push({ path, sha256: hash(path), opened: false });
      }
      const finish = async locator => {
        let steps = 0;
        while (await locator.isEnabled()) { await locator.click(); operatedStates += 1; if (++steps > 16) throw new Error('Unbounded trace'); }
      };
      assert.equal(await lesson.locator('h2').count(), 15);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      for (const index of [0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13]) await capture(lesson.locator('h2').nth(index), `reading-${index + 1}`);
      // The retained arrays/text/hash investigations still respond on the actual route.
      for (const [id, label, value] of [['array-movement', 'Sequence operation', 'delete'], ['text-units', 'Text sample', 'emoji'], ['hash-buckets', 'Bucket count', '5']]) {
        const original = lesson.locator(`[data-investigation="${id}"]`);
        await original.getByRole('combobox', { name: label, exact: true }).selectOption(value);
        operatedStates += 1;
      }

      const sets = lesson.locator('#packed-set-lab');
      for (const [operation, expected] of [['intersection', 4], ['union', 55], ['difference', 33], ['symmetric', 51], ['complement', 26]]) {
        await sets.getByLabel('Set query', { exact: true }).selectOption(operation);
        assert((await sets.locator('.bitwise-result').innerText()).includes(`mask ${expected}`)); operatedStates += 1;
      }
      await sets.getByLabel('Set query', { exact: true }).selectOption('symmetric');
      await sets.getByRole('button', { name: 'A, bit 1', exact: true }).focus(); await page.keyboard.press('Space');
      assert.equal(await sets.getByRole('button', { name: 'A, bit 1', exact: true }).getAttribute('aria-pressed'), 'true');
      assert((await sets.locator('.bitwise-result').innerText()).includes('mask 49')); operatedStates += 2;
      await sets.getByRole('button', { name: 'B, bit 5', exact: true }).click(); operatedStates += 1;
      await capture(sets, 'changed-sets');
      await sets.getByLabel('Universe width', { exact: true }).selectOption('4');
      assert((await sets.getByRole('status').innerText()).includes('Higher positions were discarded'));
      assert.equal(await sets.getByRole('button', { name: /^A, bit/ }).count(), 4);
      await sets.getByLabel('Universe width', { exact: true }).selectOption('8'); operatedStates += 2;
      await sets.getByRole('button', { name: 'Reset sets', exact: true }).click(); operatedStates += 1;
      assert((await sets.locator('.bitwise-result').innerText()).includes('mask 4'));

      const word = lesson.locator('#word-interpretation-lab');
      assert((await word.innerText()).includes('signed result −3') || (await word.innerText()).includes('signed result -3'));
      await capture(word, 'word-default');
      await word.getByLabel('Word width', { exact: true }).selectOption('4');
      // Six remains after truncation; toggle b3, b2, b0 to obtain 1011.
      for (const position of [3, 2, 0]) await word.getByRole('button', { name: `Pattern, bit ${position}`, exact: true }).click();
      await word.getByLabel('Shift positions', { exact: true }).selectOption('1'); operatedStates += 5;
      assert((await word.innerText()).includes('signed result -3'));
      assert((await word.innerText()).includes('unsigned result 5'));
      assert((await word.innerText()).includes('Discarding overflow gives 6'));
      await capture(word, 'changed-word');
      await capture(word.getByRole('heading', { name: 'Right shift by 1', exact: true }), 'shift-origins');
      for (const shift of ['0', '4']) { await word.getByLabel('Shift positions', { exact: true }).selectOption(shift); operatedStates += 1; }
      assert((await word.innerText()).includes('unsigned result 0'));
      assert((await word.innerText()).includes('signed result -1'));
      await word.getByRole('button', { name: 'Reset word', exact: true }).click(); operatedStates += 1;

      const parity = lesson.locator('#xor-parity-lab');
      for (const [scenario, answer, expected] of [['single', 9, 'Exactly one singleton'], ['two', 11, 'Two singletons'], ['triple', 6, 'promise fails'], ['absent', 0, 'promise fails'], ['zero', 0, 'Exactly one singleton']]) {
        await parity.getByLabel('Parity scenario', { exact: true }).selectOption(scenario); operatedStates += 1;
        await finish(parity.getByRole('button', { name: 'Next', exact: true }));
        assert((await parity.getByRole('status').innerText()).includes(`prefix XOR is ${answer}`));
        assert((await parity.innerText()).includes(expected));
        if (scenario === 'triple') await capture(parity.locator('.bitwise-result'), 'invalid-parity-promise');
      }
      await parity.getByRole('button', { name: 'Previous', exact: true }).focus(); await page.keyboard.press('Enter'); operatedStates += 1;
      await parity.getByRole('button', { name: 'Restart trace', exact: true }).click(); operatedStates += 1;
      const beforeInvalid = await parity.getByRole('status').innerText();
      await parity.getByLabel('Events · 1–12 integers from 0–255', { exact: true }).fill('256, -1');
      await parity.getByRole('button', { name: 'Apply events', exact: true }).click();
      assert(await parity.getByRole('alert').isVisible());
      assert.equal(await parity.getByRole('status').innerText(), beforeInvalid); operatedStates += 1;
      await capture(parity, 'invalid-events');
      await parity.getByLabel('Events · 1–12 integers from 0–255', { exact: true }).fill('3, 7, 3');
      await parity.getByRole('button', { name: 'Apply events', exact: true }).click();
      await finish(parity.getByRole('button', { name: 'Next', exact: true }));
      assert((await parity.getByRole('status').innerText()).includes('prefix XOR is 7')); operatedStates += 1;
      await capture(parity.locator('.bitwise-stream'), 'changed-parity');
      await parity.getByRole('button', { name: 'Reset parity lab', exact: true }).click(); operatedStates += 1;

      const sparse = lesson.locator('#sparse-bit-lab');
      await capture(sparse, 'sparse-default');
      await sparse.getByRole('button', { name: 'Next', exact: true }).click(); operatedStates += 1;
      assert((await sparse.getByRole('status').innerText()).includes('Remove bit 4'));
      await capture(sparse.locator('.bitwise-row').nth(1), 'sparse-borrow');
      await sparse.getByRole('button', { name: 'Previous', exact: true }).click(); operatedStates += 1;
      for (const [value, count] of [[0, 0], [1, 1], [128, 1], [255, 8]]) {
        await sparse.getByLabel('Counting case', { exact: true }).selectOption(String(value)); operatedStates += 1;
        await finish(sparse.getByRole('button', { name: 'Next', exact: true }));
        assert((await sparse.getByRole('status').innerText()).includes(`${count} one`));
        if (value === 0) await capture(sparse, 'zero-count');
      }
      await sparse.getByRole('button', { name: 'Start, bit 3', exact: true }).focus(); await page.keyboard.press('Space'); operatedStates += 1;
      assert.equal(await sparse.getByLabel('Counting case', { exact: true }).inputValue(), 'custom');
      await sparse.getByRole('button', { name: 'Restart trace', exact: true }).click();
      await sparse.getByRole('button', { name: 'Reset counting lab', exact: true }).click(); operatedStates += 2;
      await capture(lesson.locator('.bitwise-figure'), 'two-partitions');

      for (const summary of await lesson.locator('summary').all()) {
        if (await summary.isVisible() && !(await summary.evaluate(node => node.parentElement.open))) {
          await summary.focus(); await page.keyboard.press('Enter');
          assert(await summary.evaluate(node => node.parentElement.open)); openedDisclosures += 1;
        }
      }
      assert.equal(await lesson.locator('.bitwise-practice-task').count(), 4);
      for (const task of await lesson.locator('.bitwise-practice-task').all()) assert.equal(await task.locator('details[open]').count(), 2);
      await capture(lesson.locator('.bitwise-practice-task').nth(1), 'changed-word-answer');
      await capture(lesson.locator('.bitwise-practice-task').last(), 'changed-bitmap-answer');
      const displayed = await lesson.locator('.python-example').allTextContents();
      const examples = [...Object.values(arrayMapExamples), ...Object.values(bitwiseExamples)];
      assert.equal(displayed.length, 11);
      for (const example of examples) {
        const match = displayed.find(text => text.includes(example.code.trim()));
        assert(match, 'Actual full code missing'); assert(match.includes(example.output.trim()), 'Actual output missing');
      }
      assert.equal(await lesson.locator('.dsa-practice__problem').count(), 15);
      for (const number of [136, 191, 231, 461, 260]) assert.equal(await lesson.locator('.dsa-practice__number').filter({ hasText: new RegExp(`^${number}\\.$`) }).count(), 1);
      await capture(lesson.locator('.dsa-practice__stage').nth(2), 'bitwise-practice');
      await capture(lesson.locator('.lesson-sources'), 'references');
      const optionWidths = await lesson.locator('.bitwise-lab select').evaluateAll(nodes => nodes.map(node => {
        const context = document.createElement('canvas').getContext('2d'), style = getComputedStyle(node);
        context.font = `${style.fontSize} ${style.fontFamily}`;
        return { available: node.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) - 25,
          maximum: Math.max(...[...node.options].map(option => context.measureText(option.textContent).width)) };
      }));
      assert(optionWidths.every(row => row.maximum <= row.available), JSON.stringify(optionWidths));
      const overflowingBits = await lesson.locator('.bitwise-cell').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => node.textContent));
      assert.deepEqual(overflowingBits, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []); assert.deepEqual(failures, []);
      results.push({ width, actualFonts: true, operatedStates, openedDisclosures, originalLabs: 3, newLabs: 4,
        completePrograms: 11, practicePlacements: 15, optionWidths, errors, failures, documentOverflow: false });
      await page.close();
    }
    assert.deepEqual(sources, sources.map(({ path }) => ({ path, sha256: hash(path) })));
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sources, results, images }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
