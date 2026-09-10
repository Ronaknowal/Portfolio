const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { spawn } = require('node:child_process');

// Decorator/testing coverage remains here. The current NumPy review owns its
// selection, view/copy, broadcasting and reduction investigations.
function reviewCurrentNumpy() {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(__dirname, 'numpy-foundations-browser.cjs')], { stdio: 'inherit', env: process.env });
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolve() : reject(new Error(`numpy-foundations-browser.cjs exited with ${code}`)));
  });
}

(async () => {
  const { collectLessonExamples } = await import('./lib/lesson-examples.mjs');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    fs.mkdirSync('scratch/programming-batch-two', { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      for (const slug of ['decorators-context-managers', 'testing-debugging-dependency-management']) {
        await page.goto('http://127.0.0.1:5173/learn/topic/' + slug);
        await page.locator('.lesson-lab').waitFor();
        assert.equal(await page.locator('.lesson-guide').count(), 0);
        const routeIds = await page.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(n => n.hash.slice(1)));
        for (const id of routeIds) assert.equal(await page.locator('[id="' + id + '"]').count(), 1, id);
        assert.ok(await page.locator('.lesson-pilot code').evaluateAll(nodes => nodes.every(n => n.textContent.trim().length > 0)));
        for (const details of await page.locator('.lesson-check details').all()) {
          await details.locator('summary').click();
          assert.equal(await details.getAttribute('open'), '');
        }
        const source = fs.readFileSync('src/learn/data/topics/' + slug + '.jsx', 'utf8');
        const expected = (await collectLessonExamples('src/learn/data/topics/'+slug+'.jsx')).map(reference=>reference.example);
        const outputs = await page.locator('.python-example__output').allTextContents();
        assert.equal(outputs.length, expected.length);
        for (const example of expected) assert.ok(outputs.some(s => s.includes(example.output)), 'Missing rendered output');
        const prev = page.getByRole('button', { name: 'Previous step', exact: true });
        const next = page.getByRole('button', { name: 'Next step', exact: true });
        assert.ok(await prev.isDisabled());
        for (let step = 2; step <= 6; step++) {
          await next.click();
          assert.match(await page.locator('.python-trace__state').innerText(), new RegExp('Step ' + step + ' of 6'));
        }
        assert.ok(await next.isDisabled());
        await prev.click();
        assert.match(await page.locator('.python-trace__state').innerText(), /Step 5 of 6/);
        await page.getByRole('button', { name: 'Reset trace', exact: true }).focus();
        await page.keyboard.press('Enter');
        assert.ok(await prev.isDisabled());
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), slug + ': overflow');
        await page.locator('.lesson-lab').screenshot({ path: 'scratch/programming-batch-two/' + slug + '-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
        console.log(width + 'px: ' + slug + ': outputs, navigation, solutions and controls passed');
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
  await reviewCurrentNumpy();
})().catch(error => { console.error(error); process.exitCode = 1; });
