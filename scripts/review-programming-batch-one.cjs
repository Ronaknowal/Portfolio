const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { spawn } = require('node:child_process');

// Keep this public entry point useful: current Python/OOP reviews own their
// multiple labs; this file retains the unchanged iterator lesson's checks.
function reviewCurrentLesson(script) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(__dirname, script)], { stdio: 'inherit', env: process.env });
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolve() : reject(new Error(`${script} exited with ${code}`)));
  });
}

(async () => {
  const programmingExamples = await Promise.all([import("../src/learn/data/python-core-examples.js"), import("../src/learn/data/oop-core-examples.js"), import("../src/learn/data/iterator-core-examples.js")]).then(modules => Object.assign({}, ...modules.flatMap(module=>Object.values(module))));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    fs.mkdirSync('scratch/programming-batch-one', { recursive: true });
    const lessons = [
      ['iterators-iterables-generators', 'iter', 6],
    ];
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      for (const [slug, prefix, steps] of lessons) {
        await page.goto('http://127.0.0.1:5173/learn/topic/' + slug);
        await page.locator('.python-trace').waitFor();
        assert.equal(await page.locator('.lesson-guide').count(), 0);
        const routeIds = await page.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(n => n.hash.slice(1)));
        for (const id of routeIds) assert.equal(await page.locator('[id="' + id + '"]').count(), 1);
        assert.ok(await page.locator('.lesson-pilot code').evaluateAll(nodes => nodes.every(n => n.textContent.trim().length > 0)), 'Empty inline code');
        for (const details of await page.locator('.lesson-check details').all()) {
          await details.locator('summary').click();
          assert.equal(await details.getAttribute('open'), '');
        }
        const examples = Object.entries(programmingExamples).filter(([key]) => key.startsWith(prefix));
        const outputs = await page.locator('.python-example__output').allTextContents();
        assert.equal(outputs.length, examples.length);
        for (const [, example] of examples) assert.ok(outputs.some(s => s.includes(example.output)), 'Missing rendered output: ' + example.output);
        const previous = page.getByRole('button', { name: 'Previous step', exact: true });
        const next = page.getByRole('button', { name: 'Next step', exact: true });
        assert.ok(await previous.isDisabled());
        for (let step = 2; step <= steps; step++) {
          await next.click();
          assert.ok((await page.locator('.python-trace__state').innerText()).includes('Step ' + step + ' of ' + steps));
        }
        assert.ok(await next.isDisabled());
        const state = await page.locator('.python-trace__state').innerText();
        assert.match(state, /Exhausted/);
        await page.locator('.python-trace').screenshot({ path: 'scratch/programming-batch-one/' + prefix + '-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
        await page.getByRole('button', { name: 'Reset trace', exact: true }).focus();
        await page.keyboard.press('Enter');
        assert.ok(await previous.isDisabled());
        assert.ok((await page.locator('.python-trace__state').innerText()).includes('Step 1 of'));
        await next.click();
        await previous.click();
        assert.ok(await previous.isDisabled());
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Page overflow at ' + width);
        console.log(width + 'px: ' + slug + ': outputs, anchors, answers, trace controls and layout passed');
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
  await reviewCurrentLesson('python-foundations-review.cjs');
  await reviewCurrentLesson('review-oop-foundations.cjs');
})().catch(error => { console.error(error); process.exitCode = 1; });
