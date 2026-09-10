const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const output = path.resolve('scratch/rate-distortion-review/browser');
(async () => {
  const { rateDistortionExamples } = await import('../src/learn/data/rate-distortion-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => failedRequests.push(request.url()));
      page.on('console', event => { if (event.type() === 'error' && !event.text().includes('WebSocket connection') && !event.text().startsWith('[vite] failed to connect to websocket.')) errors.push(event.text()); });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/rate-distortion-theory?module=math-foundations');
      const lesson = page.locator('.rate-distortion-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      for (const example of Object.values(rateDistortionExamples)) {
        const program = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
        assert((await program.innerText()).includes(example.code));
        assert((await program.innerText()).includes(example.expected));
        const previous = await program.evaluate(node => node.previousElementSibling?.textContent);
        assert(previous.includes(example.question));
      }
      const gaussianProgram = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: rateDistortionExamples.gaussianAllocation.title }) });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      await gaussianProgram.screenshot({ path: path.join(output, 'gaussian-program-final-' + width + '.png') });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      const proof = lesson.locator('p').filter({ hasText: 'That maximum-entropy step can be checked using KL.' });
      assert.equal(await proof.count(), 1);
      await proof.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 85));
      await page.screenshot({ path: path.join(output, 'gaussian-proof-final-' + width + '.png') });
      const practice = lesson.locator('.lesson-check').last();
      assert((await practice.innerText()).includes('compress 30-second'));
      await practice.locator('summary').last().focus(); await page.keyboard.press('Enter');
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      await practice.screenshot({ path: path.join(output, 'practice-audio-final-' + width + '.png') });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => node.textContent));
      const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      const fonts = await page.evaluate(() => ({ prose: document.fonts.check('16px "Space Grotesk"'), mono: document.fonts.check('16px "JetBrains Mono"') }));
      assert.equal(pageOverflow, false); assert.deepEqual(mathOverflow, []); assert.deepEqual(errors, []); assert.deepEqual(failedRequests, []);
      assert(fonts.prose && fonts.mono);
      results.push({ width, pageOverflow, mathOverflow, errors, failedRequests, fonts, completeProgramsAndVisibleQuestions: Object.keys(rateDistortionExamples).length });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(output, 'final-copy-results.json'), JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
  console.log(results);
})().catch(error => { console.error(error); process.exitCode = 1; });
