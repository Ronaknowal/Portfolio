const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const destination = path.resolve('scratch/conditioning-stability-browser');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/conditioning-stability-numerical-analysis', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.conditioning-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (node, name) => {
        await node.first().evaluate(element => scrollTo({ top: scrollY + element.getBoundingClientRect().top - 84, behavior: 'instant' }));
        await page.waitForTimeout(140);
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.screenshot({ path: path.join(destination, `final-${name}-${width}.png`) });
      };
      await capture(lesson.getByText('Optional Python setup.', { exact: true }), 'setup');
      const setupBeforeProgram = await lesson.evaluate(node => {
        const setup = [...node.querySelectorAll('strong')].find(item => item.textContent === 'Optional Python setup.');
        return Boolean(setup.compareDocumentPosition(node.querySelector('.python-example')) & Node.DOCUMENT_POSITION_FOLLOWING);
      });
      assert(setupBeforeProgram);
      const referenceFigure = lesson.locator('.conditioning-figure').first();
      assert.equal(await referenceFigure.locator('.conditioning-reference-branches > div').count(), 2);
      assert((await referenceFigure.innerText()).includes('the program does not first need to know the exact answer'));
      await capture(referenceFigure, 'reference-branches');
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      for (const [index, label] of [[4, 'sensitivity-equations'], [6, 'joint-bound'], [9, 'summation-bound'], [12, 'fixed-time-bound'], [15, 'discrete-defects'], [16, 'tolerance-rules']]) {
        await capture(lesson.locator('.katex-display').nth(index), label);
      }
      const changedPractice = lesson.locator('.lesson-check').last();
      const solution = changedPractice.locator('details').last();
      assert((await changedPractice.innerText()).includes('ε=2^-18'));
      assert((await solution.innerText()).includes('2863311531/8589934592'));
      assert((await solution.innerText()).includes('1/65536 and passes'));
      await capture(changedPractice, 'changed-report-question');
      await capture(solution, 'changed-report-solution');
      const report = lesson.locator('.python-example').last();
      await capture(report, 'report-program');
      await capture(report.getByText('Expected result', { exact: true }), 'report-output');
      const rounding = lesson.locator('.conditioning-lab[aria-label="Rounding cells investigation"] .conditioning-drawing');
      await rounding.focus();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(160);
      const scrolling = await rounding.evaluate(node => ({ left: node.scrollLeft, width: node.clientWidth, content: node.scrollWidth }));
      if (scrolling.content > scrolling.width) assert(scrolling.left > 0);
      await rounding.evaluate(node => { node.scrollLeft = node.scrollWidth; });
      await capture(rounding, 'rounding-scrolled');
      const referenceLinks = await lesson.locator('.lesson-sources a[href]').evaluateAll(nodes => nodes.map(node => ({ title: node.textContent, href: node.href })));
      assert(referenceLinks.length >= 15);
      assert(referenceLinks.some(link => link.href === 'https://www.youtube.com/watch?v=gv-AB35V2k8'));
      const overflow = await page.evaluate(() => ({ width: innerWidth, page: document.documentElement.scrollWidth }));
      assert(overflow.page <= width + 1);
      assert.deepEqual(errors, []);
      records.push({ width, setupBeforeProgram, scrolling, referenceLinks, errors, overflow });
      await page.close();
    }
    fs.writeFileSync(path.join(destination, 'final-reading-results.json'), JSON.stringify({ completedAt: new Date().toISOString(), records }, null, 2));
    console.log('Final actual-font setup, six derivation contexts, changed report, native display, references and keyboard scrolling passed at all three widths.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

