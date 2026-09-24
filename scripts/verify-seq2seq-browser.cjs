const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4195';
const folder = 'docs/teaching/evidence/seq2seq-browser';
fs.mkdirSync(folder, { recursive: true });
const report = { passed: false, base, checkedAt: new Date().toISOString(), groups: [], screenshots: [], geometry: [] };
const save = () => fs.writeFileSync(`${folder}/report.json`, JSON.stringify(report, null, 2) + '\n');
save();
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    await require('./lib/lesson-browser-fonts.cjs')(context);
    const page = await context.newPage(), errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => requests.push(request.url()));
    await page.goto(`${base}/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals`, { waitUntil: 'commit' });
    await page.locator('[data-lab="seq2seq-alignment"]').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const lab = name => page.locator(`[data-lab="seq2seq-${name}"]`);
    const expect = async (locator, expression) => assert.match(await locator.innerText(), expression);
    const checkContinuous = async (region, labels) => {
      for (const label of labels) {
        const number = region.getByRole('spinbutton', { name: label, exact: true });
        const slider = region.getByRole('slider', { name: `${label} slider`, exact: true });
        await number.fill('.123');
        assert.equal(Number(await number.inputValue()), .123);
        assert.equal(Number(await slider.inputValue()), .123, `${label} preserves the exact model value`);
      }
    };
    assert.equal(await page.locator('.seq2seq-lesson [data-lab]').count(), 4);
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.ok(!requests.some(url => /seed-one-inference.json|inflection-seq2seq.py|sequence-mechanics.py/.test(url)));
    report.groups.push('Complete lesson, valid equations and no eager saved weights or canonical programs');
    const alignment = lab('alignment');
    await expect(alignment, /receive\s+r\s+and score\s+e/);
    await alignment.getByRole('button', { name: 'Change cared → carts', exact: true }).click();
    await expect(alignment, /receive\s+r\s+and score\s+t/);
    await alignment.getByRole('spinbutton', { name: 'Inspect prediction position (zero-based)', exact: true }).fill('4');
    await expect(alignment, /receive\s+t\s+and score\s+s/);
    await alignment.getByRole('button', { name: 'Reset alignment', exact: true }).click();
    await alignment.getByText('See the mask and its denominator', { exact: true }).click();
    await alignment.getByRole('spinbutton', { name: 'Extra ignored PAD cells', exact: true }).fill('3');
    await expect(alignment, /12 valid targets/);
    await expect(alignment, /valid-token mean = 0\.22314/);
    await alignment.getByRole('spinbutton', { name: 'Inspect prediction position (zero-based)', exact: true }).fill('6');
    await expect(alignment, /receive\s+<eos>\s+and score\s+<pad>/);
    await expect(alignment, /This storage position is excluded from loss/);
    await checkContinuous(alignment, ['Constructed correct-token probability at each valid position']);
    await alignment.getByRole('button', { name: 'Inspect incorrect unshifted inputs', exact: true }).click();
    await expect(alignment, /same-position inputs: answer leakage/);
    await alignment.getByRole('button', { name: 'Reset alignment', exact: true }).click();
    await alignment.getByRole('textbox', { name: 'Reference target', exact: true }).fill('UPPER');
    assert.equal(await alignment.getByRole('textbox', { name: 'Reference target', exact: true }).getAttribute('aria-invalid'), 'true');
    await expect(alignment, /views retain “cared”/);
    await alignment.getByRole('button', { name: 'Reset alignment', exact: true }).click();
    report.groups.push('Target-shift causal boundary, valid-mask denominator, explicit leakage contrast and unsupported-input retention');
    const bridge = lab('bridge');
    await expect(bridge, /Mean NLL\s+0\.81251/);
    const gradientBefore = await bridge.locator('.seq-result').innerText();
    await bridge.getByRole('button', { name: 'Detach context gradient', exact: true }).click();
    await expect(bridge, /Encoder weight derivative 0;/);
    await bridge.getByRole('button', { name: 'Reconnect context gradient', exact: true }).click();
    assert.equal(await bridge.locator('.seq-result').innerText(), gradientBefore);
    await checkContinuous(bridge, ['Source scalar 1', 'Source scalar 2', 'Encoder input weight', 'Encoder update rate']);
    const weight = bridge.getByRole('spinbutton', { name: 'Encoder input weight', exact: true });
    await weight.fill('-0.25');
    const range = bridge.getByRole('slider', { name: 'Encoder input weight slider', exact: true });
    assert.equal(Number(await range.inputValue()), -.25);
    await weight.fill('');
    assert.equal(await weight.getAttribute('aria-invalid'), 'true');
    await expect(bridge, /last valid value: -0\.25/);
    await range.press('Home');
    assert.equal(Number(await weight.inputValue()), -1.5);
    await range.press('ArrowRight');
    assert.ok(Number(await weight.inputValue()) > -1.5);
    assert.equal(Number(await range.inputValue()), Number(await weight.inputValue()));
    await range.scrollIntoViewIfNeeded();
    const bounds = await range.boundingBox();
    await page.mouse.move(bounds.x + bounds.width * .25, bounds.y + bounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(bounds.x + bounds.width * .75, bounds.y + bounds.height / 2, { steps: 8 });
    await page.mouse.up();
    assert.ok(Number(await weight.inputValue()) > 0);
    assert.equal(Number(await range.inputValue()), Number(await weight.inputValue()));
    await bridge.getByRole('spinbutton', { name: 'Encoder update rate', exact: true }).fill('0');
    await expect(bridge, /Zero update rate is an exact null/);
    await bridge.getByRole('button', { name: 'Reset bridge', exact: true }).click();
    report.groups.push('Signed scalar live outputs, detach forward null, invalid numeric retention, true keyboard/pointer slider and zero update');
    const tree = lab('tree');
    await expect(tree, /Beam 2:\s+B, EOS/);
    await tree.getByRole('button', { name: 'Make A end with probability .9', exact: true }).click();
    await expect(tree, /Beam 2:\s+A, EOS/);
    await tree.getByRole('spinbutton', { name: 'P(A) at root', exact: true }).fill('.5');
    await tree.getByRole('spinbutton', { name: 'P(EOS | A)', exact: true }).fill('.5');
    await tree.getByRole('spinbutton', { name: 'P(EOS | B)', exact: true }).fill('.5');
    await tree.getByRole('spinbutton', { name: 'Tiny-tree beam width', exact: true }).fill('1');
    await expect(tree, /Greedy:\s+A, EOS/);
    await tree.getByText('Change the length-ranking policy', { exact: true }).click();
    await checkContinuous(tree, ['P(A) at root', 'P(EOS | A)', 'P(EOS | B)', 'Length score alpha']);
    await tree.getByRole('spinbutton', { name: 'Length score alpha', exact: true }).fill('1');
    await expect(tree, /Winner: longer answer/);
    await tree.getByRole('button', { name: 'Reset probability tree', exact: true }).click();
    report.groups.push('Changing normalized tree, deterministic tie, bounded frontier and explicit length-ranking direction');
    const fitted = lab('fitted');
    await fitted.getByText('Open the saved inflection model', { exact: true }).click();
    await fitted.getByRole('textbox', { name: 'Source lemma', exact: true }).waitFor();
    await expect(fitted, /Generated\s+emoves/);
    await expect(fitted, /first P\(e\) 0\.6076/);
    assert.equal(await fitted.getByRole('textbox').count(), 2, 'Only source and optional prefix, never a target/guess');
    await fitted.getByRole('button', { name: 'Change source to emmode', exact: true }).click();
    await expect(fitted, /Generated\s+ememmis/);
    await expect(fitted, /Edited query; no reference spelling/);
    await fitted.getByRole('button', { name: 'Reset fitted investigation', exact: true }).click();
    await fitted.getByRole('button', { name: 'Force first character a', exact: true }).click();
    await expect(fitted, /Generated\s+amves/);
    await expect(fitted, /0\.547886/);
    await fitted.getByRole('button', { name: 'Reset fitted investigation', exact: true }).click();
    await fitted.getByRole('combobox', { name: 'Decoder initial context', exact: true }).selectOption('zero');
    await expect(fitted, /Generated\s+pled/);
    await fitted.getByRole('combobox', { name: 'Decoder initial context', exact: true }).selectOption('lactate');
    await expect(fitted, /Generated\s+lactated/);
    await fitted.getByRole('button', { name: 'Reset fitted investigation', exact: true }).click();
    await fitted.getByRole('spinbutton', { name: 'Generated-token cap (includes EOS)', exact: true }).fill('3');
    await expect(fitted, /Generated\s+emo/);
    await expect(fitted, /CAPPED: the model did not choose EOS/);
    await fitted.getByRole('button', { name: 'Reset fitted investigation', exact: true }).click();
    await fitted.getByText('Compare actual candidate-owned beam search', { exact: true }).click();
    await fitted.getByRole('spinbutton', { name: 'Fitted-model beam width', exact: true }).fill('3');
    const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/seq2seq-native.json')).beamFixtures.find(row => row.query.lemma === 'emmove' && row.query.feature === 'past' && row.width === 3);
    assert.ok(native);
    const resultTable = fitted.getByRole('table', { name: 'Final retained beam candidates and state ownership', exact: true });
    assert.equal(await resultTable.locator('tbody tr').count(), 3);
    assert.ok((await resultTable.locator('tbody tr').first().innerText()).includes(native.tokens.map(token => ['<pad>', '<bos>', '<eos>', '<past>', '<participle>', '<third_person>', ...'abcdefghijklmnopqrstuvwxyz'][token]).join(' ')));
    await fitted.getByRole('textbox', { name: 'Source lemma', exact: true }).fill('lactate');
    await fitted.getByRole('textbox', { name: 'Source lemma', exact: true }).fill('emmove');
    await expect(fitted, /Generated\s+emoves/);
    report.groups.push('Actual saved-model source/prefix/context interventions, cap/EOS, native beam3 and quick-edit freshness');
    for (const file of ['inflection-seq2seq.py', 'sequence-mechanics.py']) {
      const disclosure = page.locator('.seq-program').filter({ has: page.locator(`a[href$="${file}"]`) });
      await disclosure.locator('summary').click();
      await disclosure.locator('pre').waitFor();
      assert.equal((await disclosure.locator('pre').innerText()).replace(/\r\n/g, '\n'), fs.readFileSync(`public/learn-code/sequence-to-sequence-encoder-decoder/${file}`, 'utf8').replace(/\r\n/g, '\n'));
      await disclosure.locator('summary').click();
    }
    report.groups.push('Complete canonical source fetched on demand with exact displayed bytes and keyboard scrolling');
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      const metrics = await page.evaluate(() => {
        const root = document.querySelector('.seq2seq-lesson');
        const visible = [...root.querySelectorAll('svg.neural-chart')].map(svg => ({ width: svg.getBoundingClientRect().width, height: svg.getBoundingClientRect().height }));
        const radicals = [...root.querySelectorAll('.katex svg')].filter(svg => svg.getClientRects().length).map(svg => svg.getBoundingClientRect().height);
        const controls = [...root.querySelectorAll('input, select, button')].filter(node => node.getClientRects().length).map(node => ({ label: node.getAttribute('aria-label') || node.textContent || node.type, height: node.getBoundingClientRect().height }));
        return { pageWidth: document.documentElement.clientWidth, pageScrollWidth: document.documentElement.scrollWidth, plots: visible, radicals, controls };
      });
      assert.ok(metrics.pageScrollWidth <= metrics.pageWidth + 1, `page overflow at ${width}`);
      assert.ok(metrics.plots.every(plot => plot.width > 100 && plot.height > 80));
      assert.ok(metrics.radicals.every(height => height > 1));
      assert.ok(metrics.controls.every(control => control.height >= 44), `control under 44 px at ${width}: ${JSON.stringify(metrics.controls.filter(control => control.height < 44))}`);
      report.geometry.push({ width, ...metrics });
      for (const name of width === 320 ? ['alignment', 'fitted'] : ['bridge', 'tree', 'fitted']) {
        await lab(name).scrollIntoViewIfNeeded();
        const file = `${folder}/${name}-${width}.png`;
        // Element captures can expand beyond the viewport. Hide unrelated fixed
        // navigation only during capture so it cannot obscure the lesson figure.
        await lab(name).screenshot({ path: file, style: '.workspace-nav, .learning-skip { visibility: hidden !important; }' });
        report.screenshots.push(file);
      }
      if (width !== 390) {
        for (const name of ['evidence', 'timelines']) {
          const figure = page.locator(`.seq-${name}`);
          await figure.scrollIntoViewIfNeeded();
          const file = `${folder}/${name}-${width}.png`;
          await figure.screenshot({ path: file, style: '.workspace-nav, .learning-skip { visibility: hidden !important; }' });
          report.screenshots.push(file);
        }
      }
    }
    const theme = await bridge.evaluate(node => ({ background: getComputedStyle(node).backgroundColor, button: getComputedStyle(node.querySelector('button')).backgroundColor }));
    assert.equal(theme.background, 'rgb(16, 16, 16)');
    assert.equal(theme.button, 'rgb(25, 25, 25)');
    assert.deepEqual(errors, []);
    report.groups.push('Desktop/390/320 geometry, equations, neutral/amber theme and no runtime errors; images captured for inspection');
    report.captureNote = 'Element screenshots hide only unrelated fixed global navigation and skip-link chrome during capture; lesson styles and content are unchanged.';
    // Isolate fetch failure and retry from the already loaded investigation.
    const failure = await context.newPage();
    let block = true;
    await failure.route('**/seed-one-inference.json', route => block ? route.abort() : route.continue());
    await failure.goto(`${base}/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals`, { waitUntil: 'commit' });
    const failedLab = failure.locator('[data-lab="seq2seq-fitted"]');
    await failedLab.waitFor();
    await failedLab.getByText('Open the saved inflection model', { exact: true }).click();
    await failedLab.getByRole('button', { name: 'Retry saved model', exact: true }).waitFor();
    block = false;
    await failedLab.getByRole('button', { name: 'Retry saved model', exact: true }).click();
    await failedLab.getByRole('textbox', { name: 'Source lemma', exact: true }).waitFor();
    await expect(failedLab, /Generated\s+emoves/);
    report.groups.push('Actual deferred model network failure, visible recovery and successful retry');
    report.sourceHashes = Object.fromEntries(['src/learn/data/topics/sequence-to-sequence-encoder-decoder.jsx', 'src/learn/components/lesson-labs/Seq2SeqLabs.jsx', 'src/learn/components/lesson-labs/seq2seq-labs.css', 'src/learn/data/seq2seq-models.js'].map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
    report.passed = true; save();
    console.log(JSON.stringify({ passed: true, base, groups: report.groups.length, screenshots: report.screenshots.length }));
  } catch (error) { report.error = error.stack; save(); throw error; }
  finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
