const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve('scratch/queueing-simulation-precision');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const { queueingExamples } = await import('../src/learn/data/queueing-examples.js');
  const example = queueingExamples.find(item => item.id === 'simulation');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/queueing-theory-m-m-1-m-g-1-little-s-law');
      const lesson = page.locator('.queueing-lesson');
      const heading = lesson.getByRole('heading', { name: example.title, exact: true });
      await heading.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const container = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
      const text = (await container.innerText()).replace(/\r\n/g, '\n');
      assert(text.includes(example.code.trim()));
      assert(await lesson.getByText(example.question, { exact: false }).count());
      assert(text.includes(example.expected.trim()));
      const note = lesson.locator('p').filter({ hasText: 'Lindley’s recursion also helps numerical accuracy' });
      await note.waitFor();
      assert((await note.innerText()).includes('subtracting large calendar times'));
      await heading.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
      await page.waitForTimeout(200);
      await page.screenshot({ path: path.join(output, `program-${width}.png`) });
      await note.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
      await page.waitForTimeout(200);
      await page.screenshot({ path: path.join(output, `explanation-${width}.png`) });
      const geometry = await page.evaluate(() => ({ viewport: innerWidth, documentWidth: document.documentElement.scrollWidth, font: getComputedStyle(document.querySelector('.queueing-lesson p')).fontFamily }));
      assert(geometry.documentWidth <= width + 1);
      assert.deepEqual(errors, []);
      records.push({ width, actualCodeQuestionOutputNoteMatch: true, geometry, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const record = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(output, 'browser-results.json'), JSON.stringify(record, null, 2));
  console.log(JSON.stringify(record));
})().catch(error => { console.error(error); process.exit(1); });
