const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
const evidence = { passed: false, checks: [], screenshots: [], errors: [] };
const output = 'docs/teaching/evidence/published-lesson-ui-repairs.json';
fs.writeFileSync(output, JSON.stringify(evidence, null, 2));
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
    await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.abort());
    const page = await context.newPage();
    page.on('pageerror', error => evidence.errors.push(error.message));
    await page.goto(`${base}/learn/path/full-curriculum/byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram`);
    const trainer = page.locator('[data-bpe-trainer]');
    await trainer.waitFor();
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await trainer.locator('label').evaluate(node => getComputedStyle(node).color), 'rgb(136, 136, 136)', 'Functional labels use the readable secondary color');
    const corpus = trainer.getByRole('textbox', { name: 'Corpus', exact: true });
    assert.match(await trainer.innerText(), /Learned vocabulary \(11 tokens\)/);
    await trainer.getByRole('button', { name: /^Train Step/ }).click();
    assert.match(await trainer.innerText(), /Learned vocabulary \(12 tokens\)/);
    assert.match(await trainer.innerText(), /Merges so far \(1\)/);
    await corpus.fill('aaaa aaaa');
    assert.match(await trainer.innerText(), /Merges so far \(0\)/);
    assert.match(await trainer.innerText(), /Next pair occurs 6 times/);
    assert(!await trainer.getByText('newest', { exact: true }).count());
    await trainer.getByRole('button', { name: /^Train Step/ }).click();
    assert.match(await trainer.innerText(), /Learned vocabulary \(3 tokens\)/);
    await trainer.getByRole('button', { name: 'Restart merges' }).click();
    assert.match(await trainer.innerText(), /Learned vocabulary \(2 tokens\)/);
    evidence.checks.push('Corpus edit immediately resets segmentation; weighted pairs, merge history and cumulative vocabulary update together; restart uses edited corpus.');
    for (const invalid of ['', 'a'.repeat(41), '</w>', 'a '.repeat(601)]) {
      await corpus.fill(invalid);
      assert.equal(await corpus.getAttribute('aria-invalid'), 'true');
      assert(await trainer.getByRole('button', { name: /^Train Step/ }).isDisabled());
      assert(!await trainer.getByText(/Learned vocabulary/).count());
    }
    await corpus.fill('__proto__ constructor toString __proto__');
    assert.equal(await corpus.getAttribute('aria-invalid'), 'false');
    assert(await trainer.getByRole('button', { name: /^Train Step/ }).isEnabled());
    await corpus.fill('x');
    await trainer.getByRole('button', { name: /^Train Step/ }).click();
    assert(await trainer.getByRole('button', { name: /^Train Step/ }).isDisabled());
    assert.match(await trainer.getByRole('status').innerText(), /No adjacent pair remains/);
    await trainer.getByRole('button', { name: 'Restore example' }).click();
    assert.match(await corpus.inputValue(), /^low low low low low /);
    evidence.checks.push('Empty/oversized/reserved-marker input is explained and suppressed; prototype-like words work; terminal state disables step; original example restores.');
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await trainer.getByRole('button', { name: 'Restore example' }).click();
      await corpus.fill('longword'.repeat(5) + ' 😀😀😀');
      await trainer.getByRole('button', { name: /^Train Step/ }).focus();
      await page.keyboard.press('Enter');
      assert.match(await trainer.innerText(), /Merges so far \(1\)/);
      const escaped = await trainer.evaluate(root => [...root.querySelectorAll('button,textarea,span')].filter(node => node.checkVisibility()).filter(node => { const b = node.getBoundingClientRect(); return b.left < -1 || b.right > innerWidth + 1; }).map(node => node.textContent));
      assert.deepEqual(escaped, []);
      evidence.checks.push(`BPE ${width}px long-token layout, labelled corpus and keyboard merge.`);
    }
    await trainer.getByRole('button', { name: 'Restore example' }).click();
    const picture = 'docs/teaching/evidence/screenshots/bpe-trainer-control-review-320.png';
    await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
    await trainer.screenshot({ path: picture });
    evidence.screenshots.push({ path: picture, hash: hash(picture) });
    assert.deepEqual(evidence.errors, []);
    const files = ['src/learn/components/BPETrainer.jsx', 'src/learn/data/bpe-trainer-model.js', 'src/learn/components/viz/TokenStream.jsx', 'src/learn/components/topic-content.css', 'scripts/verify-published-lesson-ui-repairs.cjs'];
    Object.assign(evidence, { passed: true, checkedAt: new Date().toISOString(), manifestHash: hash('dist/.vite/manifest.json'), sourceHashes: Object.fromEntries(files.map(file => [file, hash(file)])) });
    console.log(JSON.stringify({ passed: true, groups: evidence.checks.length }));
  } finally {
    fs.writeFileSync(output, JSON.stringify(evidence, null, 2) + '\n');
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
