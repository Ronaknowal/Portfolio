const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const { mutualInformationExamples: examples } = await import('../src/learn/data/mutual-information-examples.js');
  const { mutualInformationExamples: previous } = await import('../scratch/mutual-information-native-verification/pre-prompt-examples.js');
  for (const key of Object.keys(examples)) {
    assert.equal(examples[key].code, previous[key].code);
    assert.equal(examples[key].expected, previous[key].expected);
    assert.equal(examples[key].title, previous[key].title);
  }
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [], failedRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedRequests.push(request.url()));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/mutual-information-information-bottleneck?module=math-foundations', { waitUntil: 'networkidle' });
    await page.locator('.mutual-information-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const prompts = await page.locator('.mutual-information-lesson .python-example').evaluateAll(elements => elements.map(element => ({
      question: element.previousElementSibling.textContent.replace(/^Before running:\s*/, ''),
      title: element.querySelector('h3').textContent,
      previousTag: element.previousElementSibling.tagName,
      fits: element.previousElementSibling.scrollWidth <= element.previousElementSibling.clientWidth + 1,
    })));
    assert.equal(prompts.length, 9);
    for (let index = 0; index < prompts.length; index += 1) {
      const example = Object.values(examples)[index];
      assert.equal(prompts[index].question, example.question);
      assert.equal(prompts[index].title, example.title);
      assert.equal(prompts[index].previousTag, 'P');
      assert.ok(prompts[index].fits);
    }
    const first = page.locator('.mutual-information-lesson .python-example').first();
    await first.evaluate(element => window.scrollTo(0, window.scrollY + element.previousElementSibling.getBoundingClientRect().top - 180));
    await page.screenshot({ path: `scratch/mutual-information-browser-review-fonts/prompts-reading-${width}.png` });
    const geometry = await page.evaluate(() => ({
      page: document.documentElement.scrollWidth,
      loadedOriginalFont: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
      equationOverflow: [...document.querySelectorAll('.mutual-information-lesson .katex-display')].filter(element => element.scrollWidth > element.parentElement.clientWidth + 1).length,
    }));
    assert.equal(geometry.page, width);
    assert.equal(geometry.equationOverflow, 0);
    assert.ok(geometry.loadedOriginalFont);
    assert.deepEqual(errors, []);
    assert.deepEqual(failedRequests, []);
    results.push({ width, prompts, geometry, errors, failedRequests });
    await page.close();
  }
  await browser.close();
  const evidence = { at: new Date().toISOString(), unchangedExecutablePrograms: 9, unchangedOutputs: 9, results };
  fs.writeFileSync('scratch/mutual-information-browser-review-fonts/prompt-review-results.json', JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify(evidence));
})().catch(error => { console.error(error); process.exitCode = 1; });
