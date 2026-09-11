const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = 'scratch/real-analysis-independent/browser';
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const normalize = value => value.replace(/\s+/g, ' ').trim();
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const payload = JSON.parse(fs.readFileSync('scratch/real-analysis-independent/payload.json', 'utf8'));
  const sources = payload.sources.map(source => ({ path: source.path, sha256: hash(source.path) }));
  const records = [];
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      const record = { width, states: [], keyboard: [], captures: [], errors };
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/real-analysis-sequences-modes-of-convergence?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.real-analysis-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(record.fonts.includes('Space Grotesk') && record.fonts.includes('JetBrains Mono'));
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      const lab = name => lesson.getByRole('region', { name, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const filename = `${name}-${width}.png`;
        await page.screenshot({ path: `${directory}/${filename}` });
        record.captures.push(filename);
      }
      async function slider(region, name, value) {
        const input = region.getByRole('slider', { name, exact: true });
        await input.fill(String(value)); await input.dispatchEvent('input');
      }
      async function keyboardReset(region, label) {
        const button = region.getByRole('button', { name: label, exact: true });
        await button.focus(); await page.keyboard.press('Enter');
        record.keyboard.push(label);
      }
      await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
      for (const details of await lesson.locator('details.analysis-depth').all()) {
        await details.locator(':scope > summary').focus(); await page.keyboard.press('Enter');
      }
      const tail = lab('Choose where the entire safe tail starts');
      await tail.getByLabel('Requested tolerance', { exact: true }).selectOption('20');
      await slider(tail, 'Proposed start N', 19);
      assert((await tail.innerText()).includes('exactly on the tolerance boundary'));
      await tail.getByRole('slider').focus(); await page.keyboard.press('ArrowRight');
      assert((await tail.innerText()).includes('This N certifies the tail'));
      record.states.push('strict N=19 fails, keyboard N=20 passes');
      record.keyboard.push('Proposed start N: ArrowRight');
      await shot(tail, 'strict-tail'); await keyboardReset(tail, 'Reset sequence tail');

      const bracket = lab('Keep a shrinking bracket around an unknown real number');
      await bracket.getByLabel('Squared target', { exact: true }).selectOption('2');
      await slider(bracket, 'Bisection steps', 16);
      const readouts = await bracket.locator('.analysis-readout strong').allTextContents();
      assert.equal(readouts[0], '1.4141998291015625');
      assert.equal(readouts[1], '1.4142150878906250');
      assert((await bracket.innerText()).includes('Rounded endpoint squares'));
      record.states.push({ exactBracket16: readouts });
      await shot(bracket.locator('.analysis-readouts'), 'exact-decimal-bracket');
      await keyboardReset(bracket, 'Reset exact bracket');

      const block = lab('Look beyond the next small step');
      await block.getByLabel('Series increments', { exact: true }).selectOption('telescoping');
      await block.getByLabel('Block begins after N', { exact: true }).selectOption('32');
      assert((await block.innerText()).includes('1/33 − 1/65'));
      record.states.push('telescoping N32'); await keyboardReset(block, 'Reset Cauchy block');

      const power = lab('Fix one point, then let the difficult point move');
      await slider(power, 'Fixed input x', 1);
      assert.equal(await power.locator('.analysis-readout strong').first().innerText(), '0');
      await power.getByLabel('Function domain', { exact: true }).selectOption('compact-subinterval');
      await slider(power, 'Power index n', 16);
      assert((await power.innerText()).includes('now lies outside'));
      record.states.push('closed endpoint zero error; compact domain excludes moving witness');
      await shot(power, 'compact-domain'); await keyboardReset(power, 'Reset power convergence');

      const triangle = lab('Find the error that a coarse grid misses');
      await triangle.getByLabel('Triangle index n', { exact: true }).selectOption('256');
      await triangle.getByLabel('Triangle amplitude', { exact: true }).selectOption('unit-area');
      assert.deepEqual((await triangle.locator('.analysis-readout strong').allTextContents()).slice(0, 3), ['0', '256', '1']);
      record.states.push('unit-area triangle n256 missed by grid');
      await shot(triangle.locator('.analysis-pair'), 'missed-peak'); await keyboardReset(triangle, 'Reset moving triangle');

      const derivative = lab('A small curve can retain a large slope');
      await derivative.getByLabel('Oscillation index n', { exact: true }).selectOption('16');
      await derivative.getByLabel('Amplitude rule', { exact: true }).selectOption('2');
      assert.deepEqual((await derivative.locator('.analysis-readout strong').allTextContents()).slice(0, 2), ['0.00391', '0.0625']);
      record.states.push('derivative and function distinct scales n16 power2');
      await shot(derivative.locator('.analysis-pair'), 'paired-derivative'); await keyboardReset(derivative, 'Reset derivative comparison');

      const series = lab('A function-series bound does not control its derivative');
      const endpoint = series.getByLabel('Series evaluation point', { exact: true });
      await endpoint.selectOption('-1');
      assert.equal(await endpoint.locator('option:checked').innerText(), '−1 (left endpoint)');
      assert((await series.innerText()).includes('converge conditionally'));
      assert(Number(await series.locator('.analysis-readout strong').first().innerText()) < 0);
      await endpoint.focus(); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
      assert.equal(await endpoint.inputValue(), '-0.5');
      record.keyboard.push('Series point: ArrowDown/Enter');
      await page.keyboard.press('Escape');
      await endpoint.selectOption('-1');
      await endpoint.evaluate(node => node.blur());
      record.states.push('explicit negative endpoint option, negative function sum and conditional derivative');
      await shot(series, 'negative-endpoint'); await keyboardReset(series, 'Reset series endpoint');
      assert.equal(await endpoint.inputValue(), '1');

      const bernstein = lab('Build an approximation from nearby weighted values');
      await bernstein.getByLabel('Polynomial degree n', { exact: true }).selectOption('4');
      await bernstein.getByLabel('Corner location c', { exact: true }).selectOption('0.5');
      await slider(bernstein, 'Weighted evaluation x', 0.5);
      assert.deepEqual(await bernstein.locator('.analysis-readout strong').allTextContents(), ['0', '0.1875', '0.1875', '0.25']);
      const marker = bernstein.locator('.analysis-plot circle.rose-fill');
      assert.equal(Number(await marker.getAttribute('cx')), 168);
      const expectedY = 176 - (0.1875 + 0.3) * 146 / 1.4;
      assert(Math.abs(Number(await marker.getAttribute('cy')) - expectedY) < 1e-10);
      record.states.push('degree4 corner=.5 exact3/16 selected value and actual plotted marker');
      await shot(bernstein.locator('svg'), 'weighted-polynomial'); await keyboardReset(bernstein, 'Reset polynomial approximation');

      const writer = lab('A shrinking chance can keep revisiting one observer');
      await writer.getByLabel('Dyadic block k', { exact: true }).selectOption('5');
      await writer.getByLabel('Fixed observer U', { exact: true }).selectOption('0.25');
      await slider(writer, 'Interval position j', 7);
      assert.equal(await writer.locator('.analysis-readout strong').nth(2).innerText(), '0');
      await writer.getByRole('slider').focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await writer.locator('.analysis-readout strong').nth(2).innerText(), '1');
      record.keyboard.push('Typewriter j7→8: ArrowRight');
      record.states.push('half-open boundary observer1/4 excludes j7, includes j8');
      await shot(writer, 'dyadic-observer'); await keyboardReset(writer, 'Reset interval sweep');

      for (const [index, figure] of (await lesson.locator('[data-analysis-figure="continuity-domain"], [data-analysis-figure="uniform-error-path"]').all()).entries()) await shot(figure, `proof-figure-${index}`);
      const lastPractice = lesson.locator('.analysis-practice').last();
      for (const summary of await lastPractice.locator(':scope > details > summary').all()) {
        await summary.focus(); await page.keyboard.press('Enter'); record.keyboard.push('Changed capstone disclosure');
      }
      assert((await lastPractice.innerText()).includes('12/2525'));
      await shot(lastPractice, 'changed-capstone');
      record.programs = [];
      for (const example of payload.examples) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await program.count(), 1);
        for (const [index, text] of [[0, example.code], [1, example.expected]]) {
          assert.equal(normalize(await program.locator(':scope > div').nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(text));
        }
        assert.equal(normalize(await program.evaluate(node => node.previousElementSibling.textContent)), normalize(`Before running: ${example.question}`));
        record.programs.push(example.id);
      }
      await shot(lesson.locator('.python-example').last().locator(':scope > div').last(), 'exact-output');
      record.math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth })));
      assert(record.math.every(node => node.scroll <= node.width + 1));
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      await shot(lesson.locator('.lesson-sources'), 'learning-resources');
      assert.deepEqual(errors, []);
      records.push(record); await page.close();
    }
    for (const source of sources) assert.equal(hash(source.path), source.sha256);
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sources, records }, null, 2));
    console.log('Independent actual-font three-width source/program/control/plot reading passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
