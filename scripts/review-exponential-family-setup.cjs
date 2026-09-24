const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const root = path.resolve(__dirname, '..');
const directory = path.join(root, 'scratch/exponential-family-browser');
const body = 'src/learn/data/topics/exponential-families-sufficient-statistics.jsx';
const fingerprint = relative => crypto.createHash('sha256').update(fs.readFileSync(path.join(root, relative))).digest('hex');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/exponential-families-sufficient-statistics?module=mathematical-statistical-foundations');
      const lesson = page.locator('.exponential-family-lesson');
      await lesson.waitFor();
      const setup = lesson.locator('p').filter({ hasText: 'Programs use Python 3 and its standard library' });
      assert.equal(await setup.count(), 1);
      const placement = await setup.evaluate(element => ({
        nextIsFirstProgram: element.nextElementSibling?.classList.contains('family-example'),
        precedesAllPrograms: [...document.querySelectorAll('.family-example')].every(program =>
          Boolean(element.compareDocumentPosition(program) & Node.DOCUMENT_POSITION_FOLLOWING)),
        text: element.textContent,
      }));
      assert(placement.nextIsFirstProgram && placement.precedesAllPrograms);
      assert(placement.text.includes('python example.py') && placement.text.includes('py example.py'));
      assert.equal(await lesson.locator('.family-example').count(), 9);
      assert.equal(await lesson.locator('.family-exercise').count(), 9);
      await setup.evaluate(element => scrollTo(0, element.getBoundingClientRect().top + scrollY - 90));
      const screenshot = `first-program-setup-${width}.png`;
      await page.screenshot({ path: path.join(directory, screenshot) });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert(!overflow);
      records.push({ width, placement, overflow, screenshot });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  assert.deepEqual(errors, []);
  const frozen = JSON.parse(fs.readFileSync(path.join(root, 'docs/teaching/evidence/exponential-family-author-review.json'), 'utf8'));
  const unchanged = frozen.sourceFiles.slice(0, 6).filter(file => file.path !== body).map(file => {
    assert.equal(fingerprint(file.path), file.sha256, file.path);
    return file.path;
  });
  const result = {
    checkedAt: new Date().toISOString(), reason: 'Independent reviewer requested setup before the first runnable program.',
    beforeBodySha256: frozen.sourceFiles.find(file => file.path === body).sha256,
    afterBodySha256: fingerprint(body), unchangedModelExampleLabStyleBlueprint: unchanged,
    records, errors,
  };
  fs.writeFileSync(path.join(directory, 'setup-amendment-results.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
})();
