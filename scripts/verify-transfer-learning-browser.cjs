const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const preview = process.env.TRANSFER_BROWSER_PREVIEW || 'production';
const output = process.env.TRANSFER_BROWSER_EVIDENCE || 'docs/teaching/evidence/transfer-learning-production-browser.json';
const read = path => JSON.parse(fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const hash = path => digest(fs.readFileSync(path));
const sourcePaths = ['src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx', 'src/learn/components/lesson-labs/TransferLearningLabs.jsx', 'src/learn/components/lesson-labs/transfer-learning.css', 'src/learn/data/transfer-learning-model.js', 'src/learn/data/transfer-learning-experiment.json', 'src/learn/data/transfer-learning-specimens.json', 'scripts/verify-transfer-learning-browser.cjs'];
const report = { status: 'running', startedAt: new Date().toISOString(), base, preview, sourceHashes: Object.fromEntries(sourcePaths.map(path => [path, hash(path)])), manifestHash: preview === 'production' ? hash('dist/.vite/manifest.json') : null, groups: [], screenshots: [], errors: [] };
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
save();
const close = (actual, expected, tolerance = 1e-7) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} vs ${expected}`);
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 }, hasTouch: true, reducedMotion: 'reduce' });
    const fonts = read(process.env.LEARNING_FONT_FIXTURES || 'scratch/kmeans-revision-review/fonts/manifest.json');
    await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css' }));
    await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf' }) : route.abort());
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error' && !message.text().includes('Failed to load resource')) report.errors.push(message.text()); });
    await page.goto(`${base}/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals`, { waitUntil: 'domcontentloaded' });
    await page.locator('[data-testid=transfer-lora]').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const section = id => page.getByTestId(id);
    const spin = (lab, name) => lab.getByRole('spinbutton', { name, exact: true });
    const text = async id => (await section(id).innerText()).replace(/\s+/g, ' ');
    const capture = async (lab, name) => {
      const path = `docs/teaching/evidence/screenshots/transfer-learning-${name}.png`;
      const style = await page.addStyleTag({ content: '.learn-nav{visibility:hidden!important}' });
      try { await lab.screenshot({ path }); } finally { await style.evaluate(node => node.remove()); }
      report.screenshots.push({ path, sha256: hash(path) });
    };
    const group = (name, extra = {}) => report.groups.push({ name, ...extra });
    assert.equal(await page.locator('.transfer-lab').count(), 7);
    assert.equal(await page.locator('.transfer-lesson .katex-error').count(), 0);
    assert.equal(await page.getByText('Hint', { exact: true }).count(), 6);
    assert.equal(await page.getByText('Worked solution', { exact: true }).count(), 6);
    assert.equal(await page.locator('.transfer-lesson details[open]').count(), 0);
    group('All seven visual homes, six independent practice pairs, initial outputs and rendered math load');

    const reuse = section('transfer-reuse');
    await reuse.getByRole('combobox', { name: 'Target training digit' }).selectOption('8');
    assert.ok(await reuse.getByRole('img', { name: /Recorded digit 8/ }).count());
    await reuse.getByRole('checkbox', { name: 'Attach the original head to target features' }).press('Space');
    assert.match(await text('transfer-head-meaning'), /still mean digits 0–4/);
    await reuse.getByRole('button', { name: /Upper tanh/ }).click();
    assert.match(await reuse.innerText(), /Weight shape 16 × 32/);
    await reuse.getByRole('checkbox', { name: 'Attach the original head to target features' }).press('Space');
    await capture(reuse, 'reuse-desktop');
    group('Actual training pixels, architecture stages and head semantics respond to pointer and keyboard controls');

    const freeze = section('transfer-freeze');
    assert.match(await text('transfer-bn-state'), /mean: 0 → 0.2/);
    await freeze.getByRole('button', { name: 'Apply one forward state transition' }).click();
    assert.match(await text('transfer-bn-state'), /mean: 0.2 → 0.38/);
    await freeze.getByRole('checkbox', { name: 'Training mode (unchecked = evaluation)' }).uncheck();
    await spin(freeze, 'Batch value 1').fill('11'); await spin(freeze, 'Batch value 2').fill('13');
    assert.match(await text('transfer-bn-state'), /mean: 0.2 → 0.2/);
    assert.match(await text('transfer-bn-state'), /variance: 1.1 → 1.1/);
    const noGraphOutput = await text('transfer-bn-output');
    await freeze.getByRole('checkbox', { name: 'Record gradients (input also requires gradients)' }).check();
    assert.equal(await text('transfer-bn-output'), noGraphOutput);
    assert.match(await text('transfer-bn-gradients'), /Affine gradients: absent/);
    await freeze.getByRole('checkbox', { name: 'Affine parameters require gradients' }).check();
    assert.equal(await freeze.getByRole('button', { name: 'Apply affine SGD step (rate 0.1)' }).isDisabled(), true);
    await freeze.getByRole('checkbox', { name: 'Optimizer owns the affine parameters' }).check();
    assert.equal(await freeze.getByRole('button', { name: 'Apply affine SGD step (rate 0.1)' }).isEnabled(), true);
    await freeze.getByRole('button', { name: 'Apply affine SGD step (rate 0.1)' }).click();
    assert.notEqual(await text('transfer-bn-output'), noGraphOutput);
    await freeze.getByRole('button', { name: 'Reset freeze experiment' }).click();
    await spin(freeze, 'Batch value 1').fill('');
    assert.equal(await freeze.getByRole('alert').count(), 1);
    assert.match(await freeze.getByRole('alert').innerText(), /Results still use 1/);
    await freeze.getByRole('button', { name: 'Reset freeze experiment' }).click();
    assert.equal(await freeze.getByRole('alert').count(), 0);
    const momentum = freeze.getByRole('slider', { name: 'Running-statistic momentum slider' });
    await momentum.press('End'); close(Number(await spin(freeze, 'Running-statistic momentum').inputValue()), 1);
    assert.match(await text('transfer-bn-state'), /mean: 0 → 2/);
    await momentum.press('Home'); assert.match(await text('transfer-bn-state'), /mean: 0 → 0/);
    await freeze.getByRole('button', { name: 'Reset freeze experiment' }).click();
    await capture(freeze, 'freeze-desktop');
    group('Independent frozen/state/graph/optimizer lanes, actual buffer transitions, editable values, endpoints, invalid state and reset');

    const lora = section('transfer-lora');
    assert.match(await text('transfer-lora-current'), /Output \[2, 1\] · loss 2.5/);
    assert.match(await text('transfer-lora-next'), /\[1.8, 0.9\] · loss next 2.025/);
    await spin(lora, 'Input x row 1, column 1').fill('1'); await spin(lora, 'Input x row 1, column 2').fill('3');
    assert.match(await text('transfer-lora-next'), /\[0.6, 1.8\] · loss next 1.8/);
    await spin(lora, 'Input x row 1, column 2').fill('1');
    assert.match(await text('transfer-lora-current'), /loss 1/);
    assert.match(await text('transfer-lora-gradients'), /∂L\/∂B: \[0\] · \[0\]/);
    await lora.getByRole('button', { name: 'Reset LoRA fixture' }).click();
    await lora.getByRole('button', { name: 'Apply this SGD step' }).click();
    assert.match(await text('transfer-lora-current'), /\[1.8, 0.9\] · loss 2.025/);
    await lora.getByRole('combobox', { name: 'Factor rank' }).selectOption('2');
    assert.equal(await spin(lora, 'A row 2, column 2').inputValue(), '1');
    assert.equal(await spin(lora, 'B row 2, column 2').inputValue(), '0');
    await spin(lora, 'B row 2, column 2').fill('0.3');
    assert.match(await text('transfer-lora-current'), /\[2, 1.9\]/);
    await spin(lora, 'B row 2, column 2').fill('9');
    assert.equal(await lora.getByRole('alert').count(), 1);
    assert.match(await text('transfer-lora-current'), /\[2, 1.9\]/);
    await lora.getByRole('button', { name: 'Reset LoRA fixture' }).click();
    const rate = lora.getByRole('slider', { name: 'SGD learning rate slider' });
    await rate.press('Home'); assert.match(await text('transfer-lora-next'), /\[2, 1\] · loss next 2.5/);
    await rate.press('End'); close(Number(await spin(lora, 'SGD learning rate').inputValue()), 0.5);
    await rate.scrollIntoViewIfNeeded(); const bounds = await rate.boundingBox();
    await rate.click({ position: { x: bounds.width * 0.41, y: bounds.height / 2 } });
    const clickedRate = Number(await rate.inputValue()); close(Number(await spin(lora, 'SGD learning rate').inputValue()), clickedRate);
    assert.ok(clickedRate > 0 && clickedRate < 0.5);
    await lora.getByRole('button', { name: 'Reset LoRA fixture' }).click();
    await capture(lora, 'lora-desktop');
    group('LoRA default, changed/null inputs, real step, rank expansion, invalid values, native endpoints, pointer edits and reset');

    const budget = section('transfer-budget');
    assert.match(await text('transfer-adapter-count'), /233/);
    await budget.getByRole('checkbox', { name: 'Include the new five-output head' }).uncheck();
    assert.match(await text('transfer-adapter-count'), /Total trainable values: 148/);
    await spin(budget, 'LoRA factor width r').fill('3');
    assert.match(await text('transfer-lora-count'), /6 more factor values/);
    const beforeUnit = await text('transfer-adapter-bytes');
    await budget.getByRole('combobox', { name: 'Memory display unit' }).selectOption('GiB');
    assert.match(await text('transfer-adapter-bytes'), /1,776 exact bytes/);
    assert.notEqual(await text('transfer-adapter-bytes'), beforeUnit);
    await spin(budget, 'Adapter feature dimension d').fill('1024'); await spin(budget, 'Adapter bottleneck b').fill('64');
    assert.match(await text('transfer-adapter-count'), /132,160/);
    await spin(budget, 'Adapter feature dimension d').fill('2.5'); assert.equal(await budget.getByRole('alert').count(), 1);
    await budget.getByRole('button', { name: 'Reset parameter budget' }).click();
    assert.match(await text('transfer-adapter-count'), /233/);
    group('Adapter/head accounting, factor counts beyond useful rank, exact bytes/unit invariance, extremes and invalid integer input');

    const evidence = section('transfer-evidence');
    const original = await text('transfer-original-test');
    for (const [value, result] of [[250, 'Linear probe'], [84, 'No eligible candidate'], [5000, 'Scratch'], [400, 'LoRA rank 2'], [500, 'LoRA rank 2']]) {
      await spin(evidence, 'Maximum trainable parameters').fill(String(value)); assert.ok((await text('transfer-budget-winner')).includes(result)); assert.equal(await text('transfer-original-test'), original);
    }
    await evidence.getByRole('combobox', { name: 'Recorded seed' }).selectOption('2');
    await evidence.getByRole('combobox', { name: 'Inspect method' }).selectOption('adapter4');
    assert.ok((await evidence.innerText()).includes('2.350006'));
    assert.equal(await text('transfer-original-test'), original);
    await evidence.getByRole('combobox', { name: 'Recorded seed' }).selectOption('1');
    await evidence.getByRole('combobox', { name: 'Inspect method' }).selectOption('lora2');
    assert.match((await evidence.innerText()).replace(/\s+/g, ' '), /After adaptation: 44\/50/);
    await capture(evidence, 'evidence-desktop');
    await evidence.getByRole('button', { name: 'Reset evidence workspace' }).click();
    group('Budget selection/empty/null cases, seed sensitivity, source retention and immovable single final test report');

    const checkpoint = section('transfer-checkpoint');
    const probabilities = await checkpoint.locator('.transfer-probabilities').innerText();
    await checkpoint.getByRole('checkbox', { name: 'Reverse the declared output labels' }).check();
    const reversed = await checkpoint.locator('.transfer-probabilities').innerText();
    assert.notEqual(reversed, probabilities);
    assert.deepEqual([...probabilities.matchAll(/: ([0-9.]+)/g)].map(match => match[1]), [...reversed.matchAll(/: ([0-9.]+)/g)].map(match => match[1]));
    await checkpoint.getByRole('checkbox', { name: 'Change the preprocessing version' }).check();
    assert.match(await text('transfer-replay-status'), /invalid/);
    await checkpoint.getByRole('combobox', { name: 'Recorded test specimen' }).selectOption('99');
    assert.match(await checkpoint.getByRole('img').getAttribute('aria-label'), /digit 9/);
    await checkpoint.getByRole('button', { name: 'Reset checkpoint comparison' }).click();
    group('Checkpoint class semantics, unchanged numeric probabilities, invalidated preprocessing replay and all test-row choices');

    const programDetails = page.locator('details').filter({ has: page.getByText('Read the full executable Python program', { exact: true }) });
    assert.equal(await programDetails.locator('.transfer-code').count(), 0);
    await programDetails.locator('summary').press('Enter');
    await programDetails.locator('.transfer-code').waitFor();
    assert.ok((await programDetails.innerText()).includes('if __name__ == "__main__":'));
    await programDetails.locator('summary').press('Enter');
    await programDetails.locator('.transfer-code').waitFor({ state: 'detached' });
    assert.equal(await programDetails.locator('.transfer-code').count(), 0);
    for (const name of ['transfer-experiments.py', 'digits-400.csv', 'data-provenance.md']) {
      const anchor = page.locator(`.transfer-downloads a[download="${name}"]`).first();
      const href = await anchor.getAttribute('href');
      let bytes;
      if (href.startsWith('data:')) {
        const comma = href.indexOf(',');
        bytes = href.slice(0, comma).includes(';base64') ? Buffer.from(href.slice(comma + 1), 'base64') : Buffer.from(decodeURIComponent(href.slice(comma + 1)));
      } else {
        const response = await page.request.get(new URL(href, base).href); assert.equal(response.ok(), true);
        bytes = await response.body();
      }
      assert.equal(digest(bytes), hash(`src/learn/assets/transfer-learning/${name}`));
    }
    await page.locator('a[href="#transfer-section-10"]').click();
    const hint = page.getByText('Hint', { exact: true }).first(); await hint.press('Enter');
    assert.equal(await page.getByText('Worked solution', { exact: true }).first().evaluate(node => node.parentElement.open), false);
    await hint.press('Enter');
    group('Lazy complete-program view, exact downloadable bytes and independent keyboard practice disclosures');

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await page.waitForFunction(() => [...document.querySelectorAll('.transfer-plot')].every(plot => Math.abs(plot.viewBox.baseVal.width - plot.getBoundingClientRect().width) < 1));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      const geometry = await page.locator('.transfer-lesson .transfer-lab :is(input,select,button)').evaluateAll(nodes => nodes.map(node => { const r = node.getBoundingClientRect(); return { label: node.getAttribute('aria-label') || node.labels?.[0]?.innerText || node.textContent, width: r.width, left: r.left, right: r.right }; }));
      assert.ok(geometry.every(item => item.width >= 17 && item.left >= 0 && item.right <= width + 1), JSON.stringify(geometry.filter(item => item.width < 17 || item.left < 0 || item.right > width + 1)));
      const slider = evidence.getByRole('slider', { name: 'Maximum trainable parameters slider' });
      await slider.scrollIntoViewIfNeeded(); const rect = await slider.boundingBox(); await page.touchscreen.tap(rect.x + rect.width * 0.15, rect.y + rect.height / 2);
      close(Number(await slider.inputValue()), Number(await spin(evidence, 'Maximum trainable parameters').inputValue()));
      assert.equal(await text('transfer-original-test'), original);
      const plotText = await page.locator('.transfer-plot text').evaluateAll(nodes => nodes.map(node => Number.parseFloat(getComputedStyle(node).fontSize) * node.getScreenCTM().a));
      assert.ok(plotText.every(size => size >= 11.5), JSON.stringify(plotText));
      const schedule = page.locator('.transfer-schedule');
      const schedulePoints = (await schedule.locator('polyline').getAttribute('points')).split(' ').map(pair => pair.split(',').map(Number));
      close(schedulePoints[0][1], 126.5625); close(schedulePoints[1][1], 20); close(schedulePoints[2][1], 126.5625);
      await capture(evidence.locator('.transfer-trace'), `trace-${width}`);
      await capture(schedule, `schedule-${width}`);
      if (width === 390) { await capture(lora, 'lora-390'); await capture(freeze, 'freeze-390'); await capture(evidence, 'evidence-390'); }
      group(`All lab controls remain in bounds, no page overflow and a real slider touch synchronizes outputs at ${width}px`, { controls: geometry.length });
    }
    const mathSvgs = await page.locator('.transfer-lesson .katex svg').evaluateAll(nodes => nodes.map(node => ({ width: node.getBoundingClientRect().width, height: node.getBoundingClientRect().height })));
    assert.ok(mathSvgs.every(size => size.width > 1 && size.height > 1));
    const links = await page.locator('.transfer-lesson a').evaluateAll(nodes => nodes.map(node => getComputedStyle(node).color));
    assert.ok(links.every(color => color === 'rgb(226, 181, 90)'));
    assert.deepEqual(report.errors, []);
    group('KaTeX SVG geometry, amber links, reduced-motion rendering and no console/page errors');
    report.status = 'passed'; report.finishedAt = new Date().toISOString(); save();
    console.log(JSON.stringify({ status: report.status, groups: report.groups.length, screenshots: report.screenshots.length, evidence: output }));
  } catch (error) { report.status = 'failed'; report.errors.push(error.stack); save(); throw error; }
  finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
