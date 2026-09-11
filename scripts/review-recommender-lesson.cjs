const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/recommender-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { recommenderExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/recommender-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  const files = ['src/learn/data/topics/recommender-systems-collaborative-filtering-matrix-factorization.jsx', 'src/learn/data/recommender-models.js', 'src/learn/data/recommender-examples.js', 'src/learn/components/lesson-labs/RecommenderLabs.jsx', 'src/learn/components/lesson-labs/RecommenderFigures.jsx', 'src/learn/components/lesson-labs/recommender-labs.css', 'src/learn/data/curriculum/blueprints/recommender-systems-collaborative-filtering-matrix-factorization.js'];
  const sourceHashes = Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  try {
    for (const width of (process.env.RECOMMENDER_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      const record = { width, states: [], keyboard: [], anchors: [], captures: [], programs: [] };
      records.push(record);
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/recommender-systems-collaborative-filtering-matrix-factorization?module=classical-ml-supervised', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.rec-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(record.fonts.some(font => font.includes('Space Grotesk')));
      assert(record.fonts.some(font => font.includes('JetBrains Mono')));
      await page.addStyleTag({ content: 'html { scroll-behavior:auto!important; }' });
      const lab = title => lesson.getByRole('region', { name: title, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(100);
        const filename = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, filename) });
        record.captures.push(filename);
      }
      async function state(region, label, condition) {
        const text = normalize(await region.innerText());
        assert(condition(text), `${label}\n${text}`);
        const outside = await region.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => { const box = svg.getBoundingClientRect(); return [...svg.querySelectorAll('text')].filter(text => { const rect = text.getBoundingClientRect(); return rect.left < box.left - 2 || rect.right > box.right + 2; }).map(text => text.textContent); }));
        assert.deepEqual(outside, [], `${label}: SVG text boundaries`);
        record.states.push(label);
        assert.deepEqual(errors, []);
      }
      async function reset(region) {
        await region.getByRole('button', { name: 'Reset', exact: true }).focus();
        await page.keyboard.press('Enter');
        record.keyboard.push('Reset '+await region.getAttribute('aria-label'));
      }
      async function range(region, label, value) {
        const control = region.getByLabel(label, { exact: false });
        assert.equal(await control.count(), 1, label);
        await control.fill(String(value));
        await control.dispatchEvent('input');
      }
      await shot(lesson.locator('.lesson-intro'), 'reading-intro');
      for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await anchor.getAttribute('href');
        const target = lesson.locator(`[id="${href.slice(1)}"]`);
        assert.equal(await target.count(), 1, href);
        await anchor.focus(); await page.keyboard.press('Enter');
        await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 35 && box.top <= 170; }, href.slice(1));
        record.anchors.push(href);
        await shot(target, 'reading-section-'+record.anchors.length);
      }
      assert.equal(record.anchors.length, 14);
      const feedback = lab('Read a missing cell correctly');
      await state(feedback, 'missing explicit cell omitted', text => text.includes('no target') && text.includes('weight 0'));
      await feedback.getByLabel('Objective', { exact: true }).selectOption('implicit');
      await state(feedback, 'missing implicit weak target', text => text.includes('weight 1') && text.includes('loss 0.25'));
      await feedback.getByRole('button', { name: 'User 0, item 5: rating 4; count 15', exact: true }).focus(); await page.keyboard.press('Enter');
      record.keyboard.push('Select observed feedback cell');
      await state(feedback, 'observed repeated count', text => text.includes('weight 31') && text.includes('loss 7.75'));
      await shot(feedback, 'feedback-observed'); await reset(feedback); await shot(feedback, 'feedback-missing');
      const neighbors = lab('Audit a neighbor before trusting its score');
      await state(neighbors, 'raw single-overlap prediction', text => text.includes('3.063'));
      await neighbors.getByLabel("Center by each co-rater's mean").check();
      await range(neighbors, 'Minimum co-raters', 2);
      await state(neighbors, 'supported centered fallback', text => text.includes('No usable neighbors') && text.includes('3.25'));
      await neighbors.getByLabel('Allow negative weights').focus(); await page.keyboard.press('Space');
      record.keyboard.push('Toggle signed neighbor weights');
      await state(neighbors, 'signed out-of-scale estimate', text => text.includes('5.5'));
      await shot(neighbors, 'neighbor-signed'); await reset(neighbors);
      const factors = lab('Move the factors using one shared old state');
      await state(factors, 'actual simultaneous step', text => text.includes('1.758') && text.includes('1.35'));
      await range(factors, 'Learning rate η', 1);
      await state(factors, 'large step increases objective', text => text.includes('5.699'));
      await range(factors, 'Requested updates', 8);
      await state(factors, 'bounded divergent trace', text => text.includes('Stopped before'));
      await shot(factors.locator('svg'), 'factor-divergence'); await reset(factors); await shot(factors, 'factor-update');
      const rotation = lab('Rotate the coordinates without changing recommendations');
      await range(rotation, 'Shared rotation in degrees', 90);
      await state(rotation, 'rotation score invariance', text => text.includes('I0 score 1') && text.includes('I1 score 0.5') && text.includes('I2 score 1'));
      await shot(rotation, 'factor-rotation'); await reset(rotation);
      const implicit = lab('See the all-pair confidence bowl');
      await state(implicit, 'all-pair optimum', text => text.includes('0.861') && text.includes('0.083'));
      await implicit.getByLabel('Include missing-item target 0 with weight 1').uncheck();
      await state(implicit, 'removed missing term changes optimum', text => text.includes('0.852') && text.includes('0.111'));
      await shot(implicit.locator('svg'), 'implicit-omitted');
      await range(implicit, 'Confidence multiplier α', 12); await range(implicit, 'Once-per-user penalty λ', .1);
      await state(implicit, 'high-confidence small-penalty contour', text => text.includes('Minimum at p'));
      await reset(implicit); await shot(implicit, 'implicit-system'); await shot(implicit.locator('svg'), 'implicit-contours');
      const pair = lab('Train a preference gap rather than a star rating');
      await range(pair, "Sampled item's first factor", 1.5);
      await state(pair, 'reversed pair partially repaired', text => text.includes('−') && text.includes('-0.5') && text.includes('-0.217'));
      await shot(pair, 'pair-gap'); await reset(pair);
      const slate = lab('Move the slate and watch the ranking evidence');
      await state(slate, 'original metrics', text => text.includes('0.387') && text.includes('AP@K 0.25'));
      await slate.getByRole('button', { name: 'Move item 0 up', exact: true }).focus(); await page.keyboard.press('Enter');
      record.keyboard.push('Move slate item with Enter');
      await state(slate, 'earlier relevant rank', text => text.includes('NDCG@K 0.613'));
      await slate.getByLabel('Candidate generator misses item 1').check();
      await state(slate, 'candidate recall ceiling', text => text.includes('Best possible candidate Recall@K 0.5'));
      await shot(slate, 'slate-retrieval-miss');
      await slate.getByLabel('Evaluation labels', { exact: true }).selectOption('none');
      await range(slate, 'Visible slots K', 5);
      await state(slate, 'empty relevance and short slate', text => text.includes('undefined') && text.includes('4/5'));
      await shot(slate.locator('.rec-metrics'), 'slate-no-labels'); await reset(slate);
      const policy = lab("Separate exposure from a policy's expected reward");
      await state(policy, 'policy expectation differs from logged mean', text => text.includes('0.48') && text.includes('0.6') && text.includes('0.765'));
      await range(policy, 'Logging probability of A', 1);
      await state(policy, 'unsupported policy', text => text.includes('Support fails') && text.includes('undefined'));
      await shot(policy, 'policy-no-support');
      await range(policy, 'Target probability of A', 1);
      await state(policy, 'supported boundary policy', text => text.includes('Support holds'));
      await reset(policy); await shot(policy, 'policy-tree');
      for (const control of await lesson.locator('input[type="range"]').all()) {
        await control.focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
        record.keyboard.push('Range Home/ArrowRight '+await control.getAttribute('id'));
      }
      for (const region of await lesson.locator('.rec-investigation').all()) await reset(region);
      let figureIndex = 0;
      for (const figure of await lesson.locator('.rec-figure').all()) await shot(figure, 'inline-figure-'+(++figureIndex));
      const programs = await lesson.locator('.python-example').all();
      assert.equal(programs.length, 14);
      for (const example of Object.values(examples)) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await program.count(), 1);
        assert.equal(normalize(await program.locator('h3').innerText()), normalize(example.title));
        const prior = await program.evaluate(node => node.previousElementSibling?.textContent);
        assert.equal(normalize(prior), normalize('Before running: '+example.question));
        const code = await program.locator(':scope > div').evaluateAll(nodes => nodes.map(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join('')));
        assert.equal(normalize(code[0]), normalize(example.code));
        assert.equal(normalize(code[1]), normalize(example.expected));
        record.programs.push(example.title);
      }
      await shot(programs[0], 'first-program');
      await shot(lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: examples.implicitLibrary.title, exact: true }) }), 'implicit-library-code');
      await shot(programs.at(-1).locator(':scope > div').nth(1), 'changed-experiment-output');
      let practice = 0;
      for (const section of await lesson.locator('.rec-practice').all()) {
        const details = section.locator('details');
        assert.equal(await details.count(), 2);
        assert.equal(await details.nth(0).locator('summary').innerText(), 'Get a hint');
        for (const detail of await details.all()) {
          assert.equal(await detail.getAttribute('open'), null);
          await detail.locator('summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await detail.getAttribute('open'), null);
          record.keyboard.push('Open practice disclosure');
        }
        practice += 1;
      }
      assert.equal(practice, 12);
      for (const detail of await lesson.locator(':scope > details').all()) {
        await detail.locator(':scope > summary').focus(); await page.keyboard.press('Enter');
        assert.notEqual(await detail.getAttribute('open'), null);
        record.keyboard.push('Open optional derivation');
      }
      await shot(lesson.locator('.rec-practice').nth(4), 'changed-gradient-practice');
      await shot(lesson.locator('.rec-practice').last(), 'changed-capstone-practice');
      const equations = await lesson.locator('.katex-display').all();
      record.equations = equations.length;
      record.mathOverflow = [];
      for (const [index, equation] of equations.entries()) {
        const geometry = await equation.evaluate(node => ({ width: node.clientWidth, scroll: node.scrollWidth }));
        if (geometry.scroll > geometry.width + 2) record.mathOverflow.push({ index, ...geometry });
        await shot(equation, 'equation-'+index);
      }
      assert.deepEqual(record.mathOverflow, [], 'display equations must fit');
      record.geometry = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth }));
      assert(record.geometry.document <= width+2 && record.geometry.body <= width+2, JSON.stringify(record.geometry));
      assert.deepEqual(errors, []);
      await shot(lesson.locator('.lesson-sources'), 'references');
      await page.close();
    }
  } catch (error) {
    errors.push(error.stack);
    process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ timestamp: new Date().toISOString(), sourceHashes, records, errors }, null, 2));
    console.log(JSON.stringify({ widths: records.map(record => ({ width: record.width, states: record.states.length, keyboard: record.keyboard.length, programs: record.programs.length, equations: record.equations })), errors }, null, 2));
  }
})();
