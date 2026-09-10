const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({viewport: {width, height: 1000}, reducedMotion: 'reduce'});
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/graph-fundamentals-adjacency-laplacian-connectivity?module=math-foundations');
      const lesson = page.locator('.graph-fundamentals-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const record = await lesson.locator('.python-example').first().evaluate(program => {
        const question = program.previousElementSibling;
        const setup = [...document.querySelectorAll('.graph-fundamentals-lesson p')].find(node => node.textContent.includes('Python 3') && node.textContent.includes('standard library'));
        if (!setup) throw new Error('Missing early Python setup.');
        if (!(setup.compareDocumentPosition(program) & Node.DOCUMENT_POSITION_FOLLOWING)) throw new Error('Setup follows the first program.');
        scrollTo(0, scrollY + question.getBoundingClientRect().top - 90);
        return {question: question.textContent, setup: setup.textContent, title: program.querySelector('h3').textContent};
      });
      assert(record.question.startsWith('Before running:'));
      await page.waitForTimeout(180);
      const capture = 'question-before-first-program-' + width + '.png';
      await page.screenshot({path: 'scratch/graph-fundamentals-browser/' + capture});
      records.push({width, ...record, capture});
      await page.close();
    }
    fs.writeFileSync('scratch/graph-fundamentals-browser/reading-results.json', JSON.stringify({checkedAt: new Date().toISOString(), allPassed: true, records}, null, 2));
    console.log(JSON.stringify(records, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
