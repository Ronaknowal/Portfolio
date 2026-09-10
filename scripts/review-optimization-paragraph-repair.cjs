const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const folder = path.resolve('scratch/optimization-paragraph-repair');
fs.mkdirSync(folder, { recursive: true });
const topics = [
  { id: 'non-convex-optimization-landscape', name: 'nonconvex', selector: '.landscape-lesson' },
  { id: 'constrained-multi-objective-optimization', name: 'constrained', selector: '.constrained-lesson' },
];
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const topic of topics) for (const width of [1440, 390, 320]) {
      const errors = [], warnings = [], failedRequests = [];
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      page.on('pageerror', error => errors.push({ type: 'exception', message: error.message }));
      page.on('console', event => {
        if (event.type() === 'error') errors.push({ type: 'console', message: event.text(), location: event.location() });
        if (event.type() === 'warning') warnings.push(event.text());
      });
      page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
      await page.goto(`http://127.0.0.1:5173/learn/path/full-curriculum/${topic.id}`);
      const lesson = page.locator(topic.selector);
      await lesson.locator('h2').last().waitFor();
      for (let index = 0; index < await lesson.locator('h2').count(); index++) {
        await lesson.locator('h2').nth(index).evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
        await page.screenshot({ path: path.join(folder, `${topic.name}-reading-${index + 1}-${width}.png`) });
      }
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const paragraphs = await lesson.locator('p').evaluateAll(nodes => nodes.map(node => {
        const style = getComputedStyle(node);
        const box = node.getBoundingClientRect();
        return { text: node.textContent.trim(), fontSize: parseFloat(style.fontSize), lineHeight: parseFloat(style.lineHeight), marginBottom: parseFloat(style.marginBottom), width: box.width, height: box.height, nestedParagraphs: node.querySelectorAll('p').length, scrollWidth: node.scrollWidth, clientWidth: node.clientWidth };
      }));
      const styledParagraphs = paragraphs.filter(row => row.marginBottom === 20);
      assert(styledParagraphs.length > 60);
      assert(styledParagraphs.every(row => row.fontSize >= 16 && row.lineHeight > 28 && row.height > 0));
      assert(paragraphs.every(row => row.nestedParagraphs === 0));
      const overflowingParagraphs = paragraphs.filter(row => row.scrollWidth > row.clientWidth + 2);
      assert.deepEqual(overflowingParagraphs, []);
      const nestedBlocks = await lesson.locator('p p, p div, p section, p table, p ul, p ol').count();
      assert.equal(nestedBlocks, 0);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      const proseSequence = await lesson.locator('h2').first().evaluate(heading => {
        const items = [];
        for (let node = heading.nextElementSibling; node && items.length < 4; node = node.nextElementSibling) {
          if (node.tagName !== 'P') continue;
          const box = node.getBoundingClientRect();
          items.push({ text: node.textContent.trim().slice(0, 80), top: box.top, bottom: box.bottom });
        }
        return items;
      });
      assert(proseSequence.length >= 2);
      for (let index = 1; index < proseSequence.length; index++) assert(proseSequence[index].top >= proseSequence[index - 1].bottom + 19);
      const practice = lesson.locator('[class$="-practice"]').first();
      await practice.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
      await page.screenshot({ path: path.join(folder, `${topic.name}-practice-open-${width}.png`) });
      if (topic.name === 'constrained') {
        const epsilon = lesson.locator('p').filter({ hasText: 'For general chosen bounds' });
        assert.equal(await epsilon.count(), 1);
        await epsilon.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 140));
        await page.screenshot({ path: path.join(folder, `${topic.name}-epsilon-${width}.png`) });
      }
      fs.writeFileSync(path.join(folder, 'latest-console.json'), JSON.stringify({ topic, width, errors, warnings, failedRequests }, null, 2));
      assert.deepEqual(errors, []);
      assert.deepEqual(warnings.filter(text => /nest|descendant|hydration/i.test(text)), []);
      results.push({ topic: topic.id, width, paragraphCount: paragraphs.length, styledParagraphs: styledParagraphs.length, ordinarySections: 9, nestedBlocks, overflowingParagraphs, proseSequence, errors, warnings });
      fs.writeFileSync(path.join(folder, 'browser-in-progress.json'), JSON.stringify(results, null, 2));
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(folder, 'browser-results.json'), JSON.stringify({ at: new Date().toISOString(), status: 'passed', results }, null, 2));
  console.log(results.map(({ topic, width, paragraphCount, styledParagraphs, errors }) => ({ topic, width, paragraphCount, styledParagraphs, errors })));
})().catch(error => { console.error(error); process.exitCode = 1; });
