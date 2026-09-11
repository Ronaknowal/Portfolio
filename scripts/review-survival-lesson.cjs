const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/survival/browser');
fs.mkdirSync(directory, { recursive: true });
const files = ['src/learn/data/topics/survival-analysis-cox-regression-kaplan-meier-hazard-models.jsx', 'src/learn/data/survival-models.js', 'src/learn/data/survival-examples.js', 'src/learn/components/lesson-labs/SurvivalLabs.jsx', 'src/learn/components/lesson-labs/SurvivalFigures.jsx', 'src/learn/components/lesson-labs/survival-labs.css', 'src/learn/components/lesson-labs/survival-figures.css', 'src/learn/data/curriculum/blueprints/survival-analysis-cox-regression-kaplan-meier-hazard-models.js'];
const normalize = value => value.replace(/\s+/g, ' ').trim();
(async () => {
  const { survivalExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/survival-examples.js')));
  const sourceHashes = Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [], failedRequests = [];
  try {
    for (const width of (process.env.SURVIVAL_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => { if (!request.url().includes('favicon')) failedRequests.push(request.url()); });
      const record = { width, states: [], keyboard: [], anchors: [], programs: [], captures: [] };
      records.push(record);
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/survival-analysis-cox-regression-kaplan-meier-hazard-models?module=classical-ml', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.survival-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(record.fonts.some(font => font.includes('Space Grotesk')) && record.fonts.some(font => font.includes('JetBrains Mono')));
      await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
      const lab = title => lesson.getByRole('region', { name: title, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(150);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) });
        record.captures.push(file);
      }
      async function keyboard(target, key, label) { await target.focus(); await page.keyboard.press(key); record.keyboard.push(label || key); }
      async function setRange(region, name, value) {
        const input = region.getByRole('slider', { name, exact: false });
        assert.equal(await input.count(), 1);
        await input.evaluate((node, next) => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next)); node.dispatchEvent(new Event('input', { bubbles: true })); node.dispatchEvent(new Event('change', { bubbles: true })); }, value);
        assert(Math.abs(Number(await input.inputValue()) - value) < 1e-10);
      }
      async function reset(region) { await keyboard(region.getByRole('button', { name: 'Reset', exact: true }), 'Enter', `reset ${await region.getAttribute('aria-label')}`); }
      async function state(region, name, parts) {
        const text = normalize(await region.innerText());
        for (const part of parts) assert(text.includes(part), `${name}: missing ${part}\n${text}`);
        const outside = await region.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => {
          const bounds = svg.getBoundingClientRect();
          return [...svg.querySelectorAll('text')].filter(text => { const box = text.getBoundingClientRect(); return box.left < bounds.left - 2 || box.right > bounds.right + 2 || box.top < bounds.top - 2 || box.bottom > bounds.bottom + 2; }).map(text => text.textContent);
        }));
        assert.deepEqual(outside, [], `${name}: SVG label fit`);
        record.states.push(name);
      }
      for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const target = await anchor.getAttribute('href');
        await anchor.click();
        const heading = page.locator(`[id="${target.slice(1)}"]`);
        assert.equal(await heading.count(), 1);
        const top = await heading.evaluate(node => node.getBoundingClientRect().top);
        assert(top >= -2 && top < 250, `${target} actual arrival ${top}`);
        record.anchors.push(target);
      }
      const km = lab('Count the pumps still at risk');
      await state(km, 'KM tied event and censor', ['6 at risk', '0.729167']);
      if (width === 390) await shot(km.locator('.sv-lanes'), 'risk-lanes-default');
      await setRange(km, 'Administrative follow-up ends', 5);
      await setRange(km, 'Inspect day', 5);
      await state(km, 'Administrative cap updates indicators', ['Day 5:', '4 at risk', '4 censored']);
      await keyboard(km.getByLabel('Make every observed endpoint censored'), 'Space', 'all-censored toggle');
      await state(km, 'No observed events', ['S(5) = 1']);
      await reset(km);
      await km.getByLabel("Reverse one record's event status").selectOption('0');
      await state(km, 'Changed first endpoint', ['0.833333']);
      await reset(km);

      const restricted = lab('Read area, uncertainty and the end of follow-up');
      await state(restricted, 'Original area and interval', ['6.575521 days', '0.273438', 'pointwise']);
      await setRange(restricted, 'Restricted horizon in days', 7);
      await state(restricted, 'Censor-row interval wording', ['risk-table row at day 7']);
      await restricted.getByLabel('Observation scenario').selectOption('none');
      await state(restricted, 'Median not reached', ['not reached', 'not available']);
      await restricted.getByLabel('Observation scenario').selectOption('all');
      await setRange(restricted, 'Restricted horizon in days', 9);
      await state(restricted, 'Exhausted final risk set', ['not available']);
      await reset(restricted);
      if (width === 320) await shot(restricted.locator('.sv-plot'), 'restricted-area');

      const hazard = lab('Separate a hazard multiplier from a slower clock');
      await setRange(hazard, 'Multiplier', 2);
      await state(hazard, 'Time ratio 2 with shape 2', ['0.25', '19.9813']);
      await hazard.getByLabel('What the multiplier changes').selectOption('hazard');
      await state(hazard, 'Hazard ratio 2', ['0.7071']);
      await setRange(hazard, 'Weibull shape', 0.5);
      await setRange(hazard, 'Already running for days', 0);
      await state(hazard, 'Unbounded zero-age rate', ['unbounded as age approaches zero']);
      await reset(hazard);
      if (width === 390) await shot(hazard.locator('.sv-chart-pair'), 'hazard-clock');

      const cox = lab('Let the event compete inside its risk set');
      const betaInput = cox.getByLabel('Cox coefficient beta', { exact: false });
      assert(Math.abs(Number(await betaInput.inputValue()) - Math.log(2)) < 1e-14);
      await state(cox, 'Exact log2 preset and Efron tie', ['0.06153846']);
      await cox.getByLabel('Tie denominator').selectOption('breslow');
      await state(cox, 'Breslow tie', ['0.04733728']);
      await setRange(cox, 'Cox coefficient beta', 0);
      await cox.getByLabel('Event-time risk set').selectOption('1');
      await state(cox, 'Uniform local competition', ['0.16666667', 'score contribution -1']);
      await keyboard(cox.getByRole('button', { name: 'Use beta = log(2)', exact: true }), 'Enter', 'exact logarithm preset');
      assert(Math.abs(Number(await betaInput.inputValue()) - Math.log(2)) < 1e-14);
      await reset(cox);
      if (width === 1440 || width === 320) await shot(cox.locator('.sv-risk-weights'), 'cox-risk-weights');

      const ph = lab('A hazard change and a survival crossing happen at different times');
      await state(ph, 'Survival crossing occurs at 6', ['0.548812', 'hazard ratio = 2']);
      await ph.getByLabel('Constructed population').selectOption('mixture');
      await setRange(ph, 'Inspect elapsed days', 5);
      await state(ph, 'Conditional versus pooled PH', ['0.634167']);
      if (width === 390) await shot(ph.locator('.sv-chart-pair'), 'pooled-hazards');
      await reset(ph);

      const concordance = lab('Rank only the pairs whose ordering can be compared');
      await state(concordance, 'Five comparable pairs', ['5 comparable pairs = 0.9']);
      await concordance.getByLabel('Risk-score ordering').selectOption('reversed');
      await state(concordance, 'Reversed risk', ['5 comparable pairs = 0.1']);
      await concordance.getByLabel('Risk-score ordering').selectOption('tied');
      await state(concordance, 'All tied scores', ['5 comparable pairs = 0.5']);
      await concordance.getByLabel('Risk-score ordering').selectOption('transformed');
      await state(concordance, 'Strict score transform', ['5 comparable pairs = 0.9']);
      await reset(concordance);
      if (width === 320) await shot(concordance.locator('.sv-pair-board'), 'pair-board');

      const censor = lab('Recover a horizon loss from observable outcomes');
      await state(censor, 'Known observation-law expectation', ['0.24', '0.274286', '0.3']);
      await setRange(censor, 'Chance follow-up lasts to day 10', 0.1);
      await state(censor, 'Rare observation increases weight', ['0.24']);
      await setRange(censor, 'Predicted survival at day 5', 0.5);
      await state(censor, 'Changed proper probability loss', ['0.25']);
      await reset(censor);
      if (width === 1440) await shot(censor.locator('.sv-observation-flow'), 'censor-probability-tree');

      const competing = lab('Send first-event mass out of one common surviving pool');
      await state(competing, 'Final first-event mass', ['0.388889', '0.222222']);
      await setRange(competing, 'Observe through day', 4);
      await state(competing, 'Earlier mass transition', ['Day 4, risk set 3', '0.444444']);
      await setRange(competing, 'Analytic competing hazard per day', 0);
      await state(competing, 'Remove analytic competing rate', ['0.393469']);
      await reset(competing);
      if (width === 390) await shot(competing.locator('.sv-mass-strip'), 'competing-mass');

      // Reach every type of control using actual keyboard operation, not only synthetic input events.
      for (const region of await lesson.locator('.sv-lab').all()) {
        for (const input of await region.locator('input[type="range"]').all()) {
          await keyboard(input, 'ArrowRight', `range ${await input.getAttribute('id')}`);
        }
        for (const select of await region.locator('select').all()) {
          await keyboard(select, 'End', `select ${await select.getAttribute('aria-label')}`);
          await page.keyboard.press('Enter');
        }
        await reset(region);
      }
      for (const practice of await lesson.locator('.sv-practice').all()) {
        const summaries = practice.locator(':scope > details > summary');
        assert.equal(await summaries.nth(0).innerText(), 'Get a hint');
        assert.equal(await summaries.nth(1).innerText(), 'Show the explained solution');
      }
      for (const summary of await lesson.locator('details > summary').all()) {
        const open = await summary.evaluate(node => node.parentElement.open);
        if (!open) await keyboard(summary, 'Enter', 'open teaching disclosure');
      }
      for (const example of Object.values(examples)) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await program.count(), 1);
        assert.equal(normalize(await program.evaluate(node => node.previousElementSibling.textContent)), normalize('Before running: ' + example.question));
        const blocks = await program.locator(':scope > div').evaluateAll(nodes => nodes.map(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join('')));
        assert.equal(normalize(blocks[0]), normalize(example.code));
        assert.equal(normalize(blocks[1]), normalize(example.expected));
        record.programs.push(example.title);
      }
      record.equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, text: node.textContent, scroll: node.scrollWidth, client: node.clientWidth })));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      record.equationOverflow = record.equations.filter(row => row.scroll > row.client + 2);
      record.geometryOverflow = await lesson.evaluate(node => { const bounds = node.getBoundingClientRect(); return [...node.querySelectorAll('p,h2,h3,summary,svg,figure,input,select,button')].filter(element => element.getClientRects().length && !element.closest('.lesson-table-wrap,.sv-table-scroll')).filter(element => { const box = element.getBoundingClientRect(); return box.left < bounds.left - 3 || box.right > bounds.right + 3; }).map(element => element.textContent.slice(0, 120)); });
      assert.deepEqual(record.geometryOverflow, []);
      if (width === 1440) await shot(lesson.locator('h2').nth(8), 'ties-reading');
      if (width === 390) await shot(lesson.locator('h2').nth(10), 'evaluation-reading');
      if (width === 320) {
        await shot(lesson.locator('.sv-practice').nth(3), 'changed-cox-practice');
        await shot(lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: examples.changedReport.title, exact: true }) }).locator(':scope > div').nth(1), 'changed-report-output');
      }
      await page.close();
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(failedRequests, []);
    const result = { checkedAt: new Date().toISOString(), sourceHashes, records, errors, failedRequests, passed: records.every(record => record.equationOverflow.length === 0) };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(records.map(record => ({ width: record.width, states: record.states.length, keyboard: record.keyboard.length, anchors: record.anchors.length, programs: record.programs.length, equationOverflow: record.equationOverflow })), null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
