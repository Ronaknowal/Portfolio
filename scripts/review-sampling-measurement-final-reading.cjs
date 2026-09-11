const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/sampling-measurement-browser');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/sampling-measurement-experimental-design', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.sampling-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
      };
      await capture(lesson.locator('h2').nth(5), 'final-reading-units');
      const units = lesson.locator('.sampling-lab[aria-label="Independent units and repeated readings investigation"]');
      assert((await units.innerText()).includes('noninteger readouts are rounded'));
      const readingLabelRight = await units.locator('svg text').filter({ hasText: /^Reading$/ }).evaluate(node => { const box = node.getBBox(); return box.x + box.width; });
      assert(readingLabelRight < 104, 'Reading label leaves a gap before the bar at x=108');
      await capture(units.locator('svg'), 'final-variance-plot');
      for (const name of ['Unit variance', 'Reading variance']) { await units.getByRole('slider', { name, exact: true }).fill('0'); }
      assert((await units.locator('.sampling-readout').innerText()).includes('correlation = undefined; design effect = undefined'));
      await capture(units.locator('.sampling-readout'), 'final-zero-variance');
      const assignment = lesson.locator('.sampling-lab[aria-label="Random assignment investigation"]');
      const table = assignment.locator('.sampling-table-scroll');
      await table.focus();
      for (let n = 0; n < 5; n += 1) await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(150);
      const tableScroll = await table.evaluate(node => ({ width: node.clientWidth, content: node.scrollWidth, left: node.scrollLeft, focused: node === document.activeElement }));
      if (tableScroll.content > tableScroll.width) assert(tableScroll.focused && tableScroll.left > 0);
      await capture(table, 'final-observed-table-scrolled');
      const task = lesson.locator('section.lesson-check').last();
      await task.getByText('Explained solution', { exact: true }).click();
      assert((await task.innerText()).includes('20, 40 and 60 students'));
      assert((await task.innerText()).includes('(1/6)5+(1/3)5+(1/2)5=5'));
      await capture(task.locator('details').last(), 'final-protocol-answer');
      await capture(task.locator('details').last().locator('p').last(), 'final-student-target');
      await lesson.getByText('Deeper: the finite randomization variance and what remains unobserved', { exact: true }).click();
      for (const i of [1, 4, 5, 6, 7, 8]) await capture(lesson.locator('.katex-display').nth(i), `final-equation-${i}`);
      const lastProgram = lesson.locator('.python-example').last();
      await capture(lastProgram.locator('div[style*="white-space: pre"]').last(), 'final-protocol-output');
      const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.clientWidth, content: node.scrollWidth })));
      assert.deepEqual(math.filter(item => item.content > item.width + 2), []);
      const sources = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent, href: node.href })));
      assert.equal(sources.length, 13);
      assert(sources.some(source => source.href.includes('14310x-lecture-4_mp4')));
      assert(sources.some(source => source.href.includes('14310x-lecture-22_mp4')));
      const overflow = await page.evaluate(() => ({ document: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert(overflow.document <= width + 1);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.deepEqual(errors, []);
      records.push({ width, fonts: true, math, sources, displayedStudentWeightedProtocol: true, roundingNote: true, tableScroll, overflow, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { checkedAt: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(directory, 'final-reading-results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ checkedAt: result.checkedAt, passed: true, widths: records.map(record => record.width) }));
})().catch(error => { console.error(error); process.exit(1); });

