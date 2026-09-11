const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/bitwise-independent-review/browser';
fs.mkdirSync(directory, { recursive: true });
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');

(async () => {
  const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/bitwise-foundations-author-review.json'));
  const sources = author.sources.map(({ path, sha256 }) => ({ path, sha256 }));
  for (const source of sources) assert.equal(hash(source.path), source.sha256);
  const { bitwiseExamples } = await import('../src/learn/data/bitwise-foundations-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error', 'warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/arrays-strings-hash-maps?module=data-structures-algorithms', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.lesson-pilot').first();
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      let operatedStates = 0;
      async function capture(locator, name) {
        await locator.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 100, behavior: 'instant' }));
        await page.waitForTimeout(80);
        const path = `${directory}/${name}-${width}.png`;
        await page.screenshot({ path });
        images.push({ path, sha256: hash(path) });
      }
      async function action(promise) { await promise; operatedStates += 1; }
      const row = (lab, label) => lab.locator('.bitwise-row').filter({ has: page.locator('.bitwise-row__label').filter({ hasText: new RegExp(`^${label}$`) }) });
      async function assertRow(lab, label, expected, wordWidth) {
        const chosen = row(lab, label);
        assert.equal(await chosen.locator('.bitwise-row__value').innerText(), String(expected));
        const bits = await chosen.locator('.bitwise-cell strong').allTextContents();
        assert.equal(bits.join(''), expected.toString(2).padStart(wordWidth, '0'));
      }
      async function setBits(lab, label, desired, wordWidth) {
        for (let position = 0; position < wordWidth; position += 1) {
          const button = lab.getByRole('button', { name: `${label}, bit ${position}`, exact: true });
          const shouldBeOn = Math.floor(desired / 2 ** position) % 2 === 1;
          if ((await button.getAttribute('aria-pressed') === 'true') !== shouldBeOn) await action(button.click());
        }
      }

      await capture(lesson.getByRole('heading', { name: '9. Store a finite set in bits', exact: true }), 'reading-place-values');
      const sets = lesson.locator('#packed-set-lab');
      await action(sets.getByLabel('Universe width', { exact: true }).selectOption('8'));
      await setBits(sets, 'A', 129, 8); await setBits(sets, 'B', 130, 8);
      await action(sets.getByLabel('Set query', { exact: true }).selectOption('symmetric'));
      await assertRow(sets, 'Result', 3, 8);
      await capture(sets.locator('.bitwise-row').first(), 'changed-high-set');
      await action(sets.getByLabel('Set query', { exact: true }).selectOption('complement'));
      await assertRow(sets, 'Result', 126, 8);
      await action(sets.getByLabel('Universe width', { exact: true }).selectOption('4'));
      await assertRow(sets, 'Result', 14, 4);
      await action(sets.getByLabel('Universe width', { exact: true }).selectOption('8'));
      await assertRow(sets, 'A', 1, 8); await assertRow(sets, 'Result', 254, 8);
      await sets.getByRole('button', { name: 'A, bit 0', exact: true }).focus();
      await action(page.keyboard.press('Space')); await assertRow(sets, 'A', 0, 8);
      await assertRow(sets, 'Result', 255, 8);
      await action(sets.getByRole('button', { name: 'Reset sets', exact: true }).click());

      const word = lesson.locator('#word-interpretation-lab');
      await setBits(word, 'Pattern', 129, 8);
      await action(word.getByLabel('Shift positions', { exact: true }).selectOption('7'));
      await assertRow(word, 'Logical', 1, 8); await assertRow(word, 'Arithmetic', 255, 8); await assertRow(word, 'Bounded', 128, 8);
      assert((await word.innerText()).includes('signed result -1'));
      assert.deepEqual(await row(word, 'Logical').locator('.bitwise-cell small').allTextContents(), ['fill', 'fill', 'fill', 'fill', 'fill', 'fill', 'fill', '←b7']);
      await capture(word.locator('.bitwise-interpretations'), 'negative-weight');
      await capture(word.getByRole('heading', { name: 'Right shift by 7', exact: true }), 'seven-place-shift');
      await word.getByLabel('Shift positions', { exact: true }).focus();
      await action(page.keyboard.press('ArrowDown'));
      assert.equal(await word.getByLabel('Shift positions', { exact: true }).inputValue(), '8');
      await assertRow(word, 'Logical', 0, 8); await assertRow(word, 'Arithmetic', 255, 8); await assertRow(word, 'Bounded', 0, 8);
      assert((await word.innerText()).includes('33024'));
      await action(word.getByLabel('Word width', { exact: true }).selectOption('4'));
      assert.equal(await word.getByLabel('Shift positions', { exact: true }).inputValue(), '4');
      await assertRow(word, 'Pattern', 1, 4); await assertRow(word, 'Arithmetic', 0, 4);
      await action(word.getByLabel('Word width', { exact: true }).selectOption('8'));
      await assertRow(word, 'Pattern', 1, 8);
      await action(word.getByRole('button', { name: 'Reset word', exact: true }).click());

      const parity = lesson.locator('#xor-parity-lab');
      await parity.getByLabel('Events · 1–12 integers from 0–255', { exact: true }).fill('255, 1, 255, 128, 1, 0, 128');
      await action(parity.getByRole('button', { name: 'Apply events', exact: true }).click());
      for (const expected of [255, 254, 1, 129, 128, 128, 0]) {
        await parity.getByRole('button', { name: 'Next', exact: true }).focus();
        await action(page.keyboard.press('Enter')); await assertRow(parity, 'Parity', expected, 8);
      }
      assert((await parity.innerText()).includes('Exactly one singleton'));
      await capture(parity.locator('.bitwise-stream'), 'zero-survivor');
      const previous = await parity.getByRole('status').innerText();
      await parity.getByLabel('Events · 1–12 integers from 0–255', { exact: true }).fill('255,,0');
      await action(parity.getByRole('button', { name: 'Apply events', exact: true }).click());
      assert((await parity.getByRole('alert').innerText()).includes('active trace is unchanged'));
      assert.equal(await parity.getByRole('status').innerText(), previous);
      await capture(parity.getByRole('alert'), 'invalid-keeps-finished-trace');
      await parity.getByLabel('Events · 1–12 integers from 0–255', { exact: true }).fill('0, 0, 0');
      await action(parity.getByRole('button', { name: 'Apply events', exact: true }).click());
      assert((await parity.innerText()).includes('promise fails'));
      for (let step = 0; step < 3; step += 1) await action(parity.getByRole('button', { name: 'Next', exact: true }).click());
      await assertRow(parity, 'Parity', 0, 8);
      await action(parity.getByRole('button', { name: 'Reset parity lab', exact: true }).click());

      const sparse = lesson.locator('#sparse-bit-lab');
      await setBits(sparse, 'Start', 164, 8);
      assert((await sparse.getByRole('status').innerText()).includes('Remove bit 2'));
      await assertRow(sparse, 'x', 164, 8); await assertRow(sparse, 'AND', 160, 8);
      await action(sparse.getByRole('button', { name: 'Next', exact: true }).click());
      assert((await sparse.getByRole('status').innerText()).includes('Remove bit 5'));
      await assertRow(sparse, 'x', 160, 8); await assertRow(sparse, 'x − 1', 159, 8); await assertRow(sparse, 'AND', 128, 8);
      await capture(sparse.locator('.bitwise-row').nth(1), 'five-place-borrow');
      await action(sparse.getByRole('button', { name: 'Next', exact: true }).click());
      await action(sparse.getByRole('button', { name: 'Next', exact: true }).click());
      assert((await sparse.getByRole('status').innerText()).includes('3 ones were removed'));
      await action(sparse.getByLabel('Counting case', { exact: true }).selectOption('0'));
      assert(await sparse.getByRole('button', { name: 'Next', exact: true }).isDisabled());
      assert((await sparse.innerText()).includes('is not a positive power of two'));
      await action(sparse.getByRole('button', { name: 'Reset counting lab', exact: true }).click());
      await capture(lesson.locator('.bitwise-figure'), 'separating-partition');

      for (const task of await lesson.locator('.bitwise-practice-task').all()) {
        for (const summary of await task.locator('summary').all()) {
          await summary.focus(); await action(page.keyboard.press('Enter'));
          assert(await summary.evaluate(node => node.parentElement.open));
        }
      }
      await capture(lesson.locator('.bitwise-practice-task').last(), 'bitmap-transfer-feedback');
      await capture(lesson.getByRole('heading', { name: '13. Deeper transfer: separate two singletons', exact: true }), 'reading-partition-proof');
      for (const example of Object.values(bitwiseExamples)) {
        const text = await lesson.locator('.python-example').allTextContents();
        assert(text.some(item => item.includes(example.code) && item.includes(example.output)), example.filename);
      }
      assert.equal(await lesson.locator('.dsa-practice__problem').count(), 15);
      for (const number of [136, 191, 231, 461, 260]) assert.equal(await lesson.locator('.dsa-practice__number').filter({ hasText: new RegExp(`^${number}\\.$`) }).count(), 1);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      assert.deepEqual(await lesson.locator('.bitwise-cell').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent)), []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      assert.deepEqual(errors, []);
      results.push({ width, actualFonts: true, operatedStates, newLabs: 4, completeNewPrograms: 5, practicePlacements: 15, changedPracticeSolutions: 4, errors, documentOverflow: false });
      await page.close();
    }
    for (const source of sources) assert.equal(hash(source.path), source.sha256);
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', sources, results, images }, null, 2) + '\n');
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
