const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/string-matching-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = text => text.replace(/\s+/g, ' ').trim();

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/string-matching-models.js')));
  const { stringMatchingExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/string-matching-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      page.on('pageerror', error => errors.push(`${width}: ${error.message}`));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173') + '/learn/path/full-curriculum/string-matching-prefix-functions-rolling-hashes?module=data-structures-algorithms');
      await page.locator('[data-lab="prefix-border"]').waitFor();
      const record = { width, anchors: [], prefixStates: 0, kmpStates: 0, rollingWindows: 0, streamStates: 0, figures: [], codeExamples: [] };
      async function capture(locator, name) {
        const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await style.evaluate(element => element.remove());
      }
      async function keyButton(lab, name, key = 'Enter') {
        await lab.getByRole('button', { name, exact: true }).focus();
        await page.keyboard.press(key);
      }
      const anchors = page.locator('nav[aria-label="In this lesson"] a');
      for (let index = 0; index < await anchors.count(); index++) {
        const href = await anchors.nth(index).getAttribute('href');
        await anchors.nth(index).focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(hash => location.hash === hash, href);
        await page.waitForTimeout(150);
        const heading = page.locator(`[id="${href.slice(1)}"]`);
        const box = await heading.boundingBox();
        assert(box && box.y >= 0 && box.y < 160, `${width} ${href} arrival ${JSON.stringify(box)}`);
        record.anchors.push({ href, top: box.y });
        await page.screenshot({ path: path.join(directory, `reading-${index + 1}-${width}.png`) });
      }
      const prefix = page.locator('[data-lab="prefix-border"]');
      for (const pattern of ['ababaca', 'aabaaac', 'aaaa', '', '🙂a🙂a']) {
        await prefix.getByLabel('Pattern', { exact: true }).fill(pattern);
        await keyButton(prefix, 'Apply pattern');
        const trace = models.prefixTrace(pattern);
        for (let index = 0; index < trace.states.length; index++) {
          const current = trace.states[index];
          assert.equal(await prefix.locator('[data-result="prefix-action"]').innerText(), current.action);
          assert.deepEqual(await prefix.locator('tbody td').allTextContents(), current.table.map(value => value === null ? '·' : String(value)));
          if (pattern === 'ababaca' && current.kind === 'fallback' && current.matched === 1) await capture(prefix, 'prefix-fallback');
          if (index + 1 < trace.states.length) await keyButton(prefix, 'Next state');
          record.prefixStates++;
        }
        if (trace.states.length > 1) {
          await keyButton(prefix, 'Previous');
          assert.equal(await prefix.locator('[data-result="prefix-action"]').innerText(), trace.states.at(-2).action);
        }
        await keyButton(prefix, 'Reset', 'Space');
        assert.equal(await prefix.locator('[data-result="prefix-action"]').innerText(), trace.states[0].action);
      }
      await prefix.getByLabel('Pattern', { exact: true }).fill('x'.repeat(19));
      await keyButton(prefix, 'Apply pattern');
      assert.match(await prefix.getByRole('alert').innerText(), /18 pattern/);
      assert.equal(await prefix.locator('[data-result="prefix-action"]').innerText(), models.prefixTrace('🙂a🙂a').states[0].action);

      const kmp = page.locator('[data-lab="kmp-alignment"]');
      for (const [text, pattern] of [[models.defaultMatchingText, models.defaultMatchingPattern], ['aaaaa', 'aaa'], ['ab', ''], ['', 'a'], ['ab', 'abcd'], ['a🙂a🙂a', '🙂a'], ['a'.repeat(36) + 'b', 'aaaab']]) {
        await kmp.getByLabel('Text', { exact: true }).fill(text);
        await kmp.getByLabel('Pattern', { exact: true }).fill(pattern);
        await keyButton(kmp, 'Apply search');
        const trace = models.kmpTrace(text, pattern);
        for (let index = 0; index < trace.states.length; index++) {
          const current = trace.states[index];
          assert.equal(await kmp.locator('[data-result="kmp-action"]').innerText(), current.action);
          assert.equal(await kmp.locator('[data-result="kmp-comparisons"]').innerText(), String(current.comparisons));
          assert.equal(await kmp.locator('[data-result="kmp-matches"]').innerText(), current.matches.join(', ') || 'none');
          const rows = kmp.locator('.string-alignment .string-cells');
          assert.equal(await rows.nth(0).locator('.string-known').count(), current.matched);
          assert.equal(await rows.nth(1).locator('.string-known').count(), current.matched);
          if (current.matched > 0) {
            const upper = await rows.nth(0).locator('.string-known').first().boundingBox();
            const lower = await rows.nth(1).locator('.string-known').first().boundingBox();
            assert(Math.abs(upper.x - lower.x) < 1, 'retained text and pattern columns must align');
          }
          if (pattern === models.defaultMatchingPattern && current.kind === 'fallback') await capture(kmp, 'kmp-fallback');
          if (text === 'aaaaa' && current.kind === 'overlap') await capture(kmp, 'kmp-overlap');
          if (index + 1 < trace.states.length) await keyButton(kmp, 'Next state');
          record.kmpStates++;
        }
        if (text.length > 30) await capture(kmp, 'kmp-long-text');
        await keyButton(kmp, 'Reset', 'Space');
        assert.equal(await kmp.locator('[data-result="kmp-action"]').innerText(), trace.states[0].action);
      }
      await kmp.getByLabel('Text', { exact: true }).fill('x'.repeat(41));
      await keyButton(kmp, 'Apply search');
      assert.match(await kmp.getByRole('alert').innerText(), /40 text/);

      const stream = page.locator('[data-lab="chunk-matcher"]');
      for (const [chunks, pattern, faulty] of [['xxa||b|aba', 'aba', false], ['xxa||b|aba', 'aba', true], ['|a|b|', '', false], ['ab|a', 'aba', false], ['|', 'a', false]]) {
        await stream.getByLabel('Chunks separated by |', { exact: true }).fill(chunks);
        await stream.getByLabel('Pattern', { exact: true }).fill(pattern);
        await stream.getByRole('checkbox').setChecked(faulty);
        await keyButton(stream, 'Apply chunks');
        const trace = models.streamTrace(chunks.split('|'), pattern, faulty);
        for (let index = 0; index < trace.states.length; index++) {
          const current = trace.states[index];
          assert.equal(await stream.locator('[data-result="stream-action"]').innerText(), `After ${current.fed} chunks: offset=${current.consumed}, q=${current.matched}; all starts: ${current.matches.join(', ') || 'none'}.`);
          assert.equal(await stream.locator('.string-fed').count(), current.fed);
          if (index + 1 < trace.states.length) await keyButton(stream, 'Feed next chunk');
          record.streamStates++;
        }
        if (chunks === 'xxa||b|aba') await capture(stream, faulty ? 'stream-fault' : 'stream-correct');
        await keyButton(stream, 'Reset', 'Space');
        assert.equal(await stream.locator('.string-fed').count(), 0);
      }
      await stream.getByLabel('Chunks separated by |', { exact: true }).fill('|'.repeat(12));
      await keyButton(stream, 'Apply chunks');
      assert.match(await stream.getByRole('alert').innerText(), /12 decoded chunks/);

      const rolling = page.locator('[data-lab="rolling-fingerprint"]');
      for (const [text, pattern, base, modulus] of [['adbaad', 'ba', 3, 7], ['aaaaa', 'aaa', 3, 7], ['a🙂a🙂a', '🙂a', 31, 1009], ['ab', '', 3, 7], ['a', 'long', 3, 7]]) {
        await rolling.getByLabel('Text', { exact: true }).fill(text);
        await rolling.getByLabel('Pattern', { exact: true }).fill(pattern);
        await rolling.getByLabel('Base', { exact: true }).selectOption(String(base));
        await rolling.getByLabel('Modulus', { exact: true }).selectOption(String(modulus));
        await keyButton(rolling, 'Apply fingerprints');
        const trace = models.rollingTrace(text, pattern, base, modulus);
        assert.equal(await rolling.locator('tbody tr').count(), trace.rows.length);
        for (let index = 0; index < trace.rows.length; index++) {
          const row = trace.rows[index];
          assert.equal(await rolling.locator('[data-result="rolling-action"]').innerText(), `Window start ${row.start}; fingerprint ${row.value}; exact-verification comparisons ${row.checks}. ${row.update ? `Next fingerprint: ${row.update.next}.` : 'Last complete window.'}`);
          if (row.candidate && !row.exact) assert.match(await rolling.locator('[data-result="rolling-verdict"]').innerText(), /collision/);
          if (text === 'adbaad' && index === 0) await capture(rolling, 'rolling-collision');
          if (text === 'adbaad' && row.exact) await capture(rolling, 'rolling-exact');
          if (index + 1 < trace.rows.length) await keyButton(rolling, 'Next window');
          record.rollingWindows++;
        }
        if (!trace.rows.length) assert.match(await rolling.locator('[data-result="rolling-action"]').innerText(), pattern ? /no complete windows/ : /all boundaries/);
      }
      await rolling.getByLabel('Base', { exact: true }).selectOption('31');
      await rolling.getByLabel('Modulus', { exact: true }).selectOption('7');
      await keyButton(rolling, 'Apply fingerprints');
      assert.match(await rolling.getByRole('alert').innerText(), /base < modulus/);
      assert.match(await rolling.locator('[data-result="rolling-action"]').innerText(), /no complete windows/);

      const advanced = page.locator('.string-advanced');
      await advanced.locator(':scope > summary').focus();
      await page.keyboard.press('Enter');
      assert(await advanced.getAttribute('open') !== null);
      for (const figure of ['overlap', 'border-evidence', 'palindromic-prefix', 'unicode-coordinates', 'prefix-cancellation', 'z-reuse']) {
        const locator = page.locator(`[data-figure="${figure}"]`);
        await capture(locator, figure);
        record.figures.push(figure);
      }
      const rendered = page.locator('.string-matching-lesson .python-example');
      assert.equal(await rendered.count(), Object.keys(examples).length);
      for (const example of Object.values(examples)) {
        const block = rendered.filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const texts = await block.locator(':scope > div').evaluateAll(elements => elements.map(element => [...element.childNodes].filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent).join('')).filter(Boolean));
        assert.equal(normalize(texts[0]), normalize(example.code), example.title);
        assert.equal(normalize(texts[1]), normalize(example.expected), example.title);
        record.codeExamples.push(example.title);
      }
      const practice = page.locator('.dsa-practice');
      assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 6);
      assert.equal(await practice.locator('a[target="_blank"]').count(), 6);
      assert.equal(await practice.locator('details[open]').count(), 0);
      const exercise = page.locator('.string-exercise').first();
      await exercise.locator('summary').last().focus();
      await page.keyboard.press('Enter');
      assert.match(await exercise.locator('details').last().innerText(), /2, 1 and 0/);
      await capture(exercise, 'exercise-answer');
      const overflow = await page.evaluate(() => ({ page: document.documentElement.scrollWidth > innerWidth + 1, panels: [...document.querySelectorAll('.string-lab,.string-figure')].filter(element => element.getBoundingClientRect().right > innerWidth + 1).map(element => element.dataset.lab || element.dataset.figure) }));
      assert.deepEqual(overflow, { page: false, panels: [] });
      record.overflow = overflow;
      results.push(record);
      await page.close();
    }
    assert.deepEqual(errors, []);
    const result = { checkedAt: new Date().toISOString(), results, pageErrors: errors };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
