const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const output = path.resolve('scratch/combinatorial-optimization-review/browser');
fs.mkdirSync(output, { recursive: true });
(async () => {
  const { combinatorialOptimizationExamples: examples } = await import('../src/learn/data/combinatorial-optimization-examples.js');
  const models = await import('../src/learn/data/combinatorial-optimization-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true }), results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], failedRequests = [], warnings = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', event => { if (event.type() === 'error') errors.push(event.text()); if (event.type() === 'warning') warnings.push(event.text()); });
      page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/combinatorial-optimization-approximation-algorithms?module=math-foundations');
      const lesson = page.locator('.combinatorial-lesson');
      await lesson.locator('h2').last().waitFor(); await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].map(font => ({ family: font.family, status: font.status })));
      assert(fonts.some(font => font.status === 'loaded'));
      const lab = id => lesson.locator('[data-combinatorial-lab="' + id + '"]');
      const metric = (region, label) => region.locator('.combinatorial-metrics > div').filter({ has: page.getByText(label, { exact: true }) }).locator('dd');
      const reset = region => region.getByRole('button', { name: 'Reset investigation' }).click();
      const next = region => region.getByRole('button', { name: 'Next state', exact: true }).click();
      async function finish(region) {
        for (let count = 0; count < 100; count++) {
          if (await region.getByRole('button', { name: 'Next state', exact: true }).isDisabled()) return;
          await next(region);
        }
        throw Error('Unbounded stepper');
      }
      async function capture(region, name) {
        await region.scrollIntoViewIfNeeded(); await page.waitForTimeout(180);
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await region.screenshot({ path: path.join(output, name + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      const matroid = lab('matroid');
      assert.equal(await metric(matroid, 'Greedy value').innerText(), '3');
      assert.equal(await metric(matroid, 'Tiny exact optimum').innerText(), '4');
      assert((await matroid.innerText()).includes('No augmentation exists'));
      await capture(matroid, 'matroid-interval'); states++;
      await matroid.getByRole('combobox', { name: 'Feasibility rule' }).selectOption('graphic');
      assert.equal(await metric(matroid, 'Greedy value').innerText(), '14');
      await capture(matroid, 'matroid-forest'); states++;
      await matroid.getByRole('combobox', { name: 'Smaller set I' }).selectOption('3');
      assert((await matroid.innerText()).includes('not satisfied'));
      await matroid.getByRole('combobox', { name: 'Larger set J' }).selectOption('11');
      assert((await matroid.innerText()).includes('Legal augmentation')); states++;
      await matroid.getByRole('combobox', { name: 'Feasibility rule' }).selectOption('uniform');
      await matroid.getByRole('slider').fill('-3');
      assert.equal(await metric(matroid, 'Greedy value').innerText(), '7'); states++;
      await reset(matroid);
      const search = lab('branch-bound');
      assert.equal(await metric(search, 'Remaining upper bound').innerText(), '59');
      await next(search); await next(search); await capture(search, 'search-middle'); states++;
      await finish(search);
      assert.equal(await metric(search, 'Feasible incumbent value').innerText(), '57');
      assert.equal(await metric(search, 'Optimality certified?').innerText(), 'Yes');
      await capture(search, 'search-final'); states++;
      await search.getByRole('slider').fill('0');
      assert.equal(await metric(search, 'Remaining upper bound').innerText(), '0'); states++; await reset(search);
      const assignment = lab('assignment');
      for (let index = 0; index < 3; index++) await next(assignment);
      assert.equal(await metric(assignment, 'Current total cost').innerText(), '5');
      assert((await assignment.locator('.combinatorial-residual-path').innerText()).includes('cost change -1'));
      await capture(assignment, 'assignment-refund'); states++;
      await assignment.getByRole('combobox').selectOption('signed'); await finish(assignment);
      assert.equal(await metric(assignment, 'Current total cost').innerText(), String(models.assignmentTrace({ costs: [[-4, 2, null], [0, 3, 2], [1, null, -1]] }).cost)); states++;
      await assignment.getByRole('combobox').selectOption('missing'); await finish(assignment);
      assert((await assignment.innerText()).includes('Requested size is impossible')); states++;
      await assignment.getByText('Inspect the final potential certificate', { exact: true }).click();
      await capture(assignment, 'assignment-missing-certificate'); await reset(assignment);
      const cover = lab('cover');
      await finish(cover);
      assert.equal(await metric(cover, 'Current cost').innerText(), '3');
      assert.equal(await metric(cover, 'Tiny exact cover cost').innerText(), '2');
      await capture(cover, 'cover-equal'); states++;
      await cover.getByRole('combobox', { name: 'Candidate data' }).selectOption('weighted'); await finish(cover);
      assert.equal(await metric(cover, 'Current cost').innerText(), '4'); states++;
      await cover.getByRole('combobox', { name: 'Optimization objective' }).selectOption('coverage'); await finish(cover);
      assert.equal(await metric(cover, 'Behaviors covered').innerText(), '5 / 6');
      await capture(cover, 'coverage-budget'); states++;
      await cover.getByRole('combobox', { name: 'Candidate data' }).selectOption('missing'); await finish(cover);
      assert.equal(await metric(cover, 'Tiny maximum possible coverage').innerText(), '5'); states++;
      await cover.getByRole('combobox', { name: 'Optimization objective' }).selectOption('cover'); await finish(cover);
      assert.equal(await metric(cover, 'Tiny exact cover cost').innerText(), 'infeasible'); states++; await reset(cover);
      const vertex = lab('vertex-cover');
      await next(vertex);
      assert.equal(await metric(vertex, 'Dual lower bound').innerText(), '2');
      await capture(vertex, 'vertex-growing'); states++; await finish(vertex);
      assert.equal(await metric(vertex, 'Selected-vertex cost').innerText(), '5');
      assert.equal(await metric(vertex, 'Dual lower bound').innerText(), '3');
      await vertex.getByText('Compare a separately solved fractional relaxation', { exact: true }).click();
      assert.equal(await metric(vertex, 'LP optimum').innerText(), '4.5');
      assert.equal(await metric(vertex, 'Rounded cover cost').innerText(), '9');
      await capture(vertex, 'vertex-fractional'); states++;
      await vertex.getByRole('combobox').selectOption('star'); await finish(vertex);
      assert.equal(await metric(vertex, 'Tiny exact cover optimum').innerText(), '6');
      await vertex.getByRole('slider').fill('0');
      assert.equal(await metric(vertex, 'Uncovered edges').innerText(), '0'); states++;
      await vertex.getByRole('combobox').selectOption('cycle'); await finish(vertex);
      assert.equal(await metric(vertex, 'Tiny exact cover optimum').innerText(), '2'); states++; await reset(vertex);
      const scaling = lab('scaling');
      for (const denominator of [2, 4, 10, 20]) {
        await scaling.getByRole('combobox').selectOption(String(denominator));
        const model = models.scaledKnapsackState({ epsilonDenominator: denominator });
        assert.equal(await metric(scaling, 'True selected value').innerText(), String(model.value));
        assert.equal(await metric(scaling, 'DP update cells').innerText(), String(model.operations)); states++;
      }
      await capture(scaling, 'scaling-fine');
      await scaling.getByRole('slider').fill('0');
      assert.equal(await metric(scaling, 'True selected value').innerText(), '0');
      assert.equal(await metric(scaling, 'DP update cells').innerText(), '0'); states++; await reset(scaling);
      let keyboardControls = 0;
      for (const region of await lesson.locator('[data-combinatorial-lab]').all()) {
        for (const control of await region.locator('input:enabled,select:enabled,button:enabled').all()) {
          await control.focus(); await page.keyboard.press('Tab'); await page.keyboard.press('Shift+Tab');
          assert(await control.evaluate(node => node === document.activeElement));
          const style = await control.evaluate(node => ({ outline: getComputedStyle(node).outlineStyle, width: getComputedStyle(node).outlineWidth }));
          assert.notEqual(style.outline, 'none'); assert(parseFloat(style.width) >= 2); keyboardControls++;
        }
      }
      await search.getByRole('slider').focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
      assert.equal(await search.getByRole('slider').inputValue(), '1'); await reset(search);
      await assignment.getByRole('button', { name: 'Next state', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await metric(assignment, 'Current total cost').innerText(), '1'); await reset(assignment);
      assert.equal(await lesson.locator('h2').count(), 11);
      for (let index = 0; index < 11; index++) {
        const heading = lesson.locator('h2').nth(index), anchor = lesson.locator('.lesson-intro nav a').nth(index);
        assert.equal((await anchor.getAttribute('href')).slice(1), await heading.getAttribute('id'));
        await anchor.click();
        assert.equal(await page.evaluate(() => decodeURIComponent(location.hash).slice(1)), await heading.getAttribute('id'));
        await page.waitForTimeout(150);
        await page.screenshot({ path: path.join(output, 'reading-' + (index + 1) + '-' + width + '.png') });
      }
      for (const example of Object.values(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
        assert.equal(await block.count(), 1);
        assert((await block.innerText()).includes(example.code)); assert((await block.innerText()).includes(example.expected));
        assert((await block.evaluate(node => node.previousElementSibling?.textContent)).includes(example.question));
      }
      const practices = lesson.locator('section.lesson-check');
      assert.equal(await practices.count(), 11);
      for (const practice of await practices.all()) for (const summary of await practice.locator('summary').all()) {
        await summary.focus(); await page.keyboard.press('Enter');
        assert(await summary.evaluate(node => node.parentElement.open));
      }
      await capture(practices.last(), 'practice-scheduling');
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      for (let index = 0; index < 3; index++) await capture(lesson.locator('.combinatorial-inline').nth(index), 'inline-' + index);
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.clientWidth, scroll: node.scrollWidth, text: node.textContent.slice(0, 130) })));
      const overflowMath = equations.filter(row => row.scroll > row.width + 2);
      const figures = await lesson.locator('svg').evaluateAll(nodes => nodes.map((svg, index) => {
        const bounds = svg.getBoundingClientRect();
        return { index, minimumTextPx: Math.min(...[...svg.querySelectorAll('text')].map(node => parseFloat(getComputedStyle(node).fontSize) * bounds.width / svg.viewBox.baseVal.width)), clipped: [...svg.querySelectorAll('text')].filter(node => { const b = node.getBoundingClientRect(); return b.left < bounds.left - 1 || b.right > bounds.right + 1 || b.top < bounds.top - 1 || b.bottom > bounds.bottom + 1; }).map(node => node.textContent) };
      }));
      for (const row of overflowMath) await capture(lesson.locator('.katex-display').nth(row.index), 'equation-overflow-' + row.index);
      let keyboardRegions = 0;
      for (const region of await lesson.locator('.lesson-table-wrap,.combinatorial-incidence,.combinatorial-tree').all()) if (await region.evaluate(node => node.scrollWidth > node.clientWidth + 1)) {
        await region.focus(); for (let i = 0; i < 6; i++) await page.keyboard.press('ArrowRight');
        await page.waitForFunction(node => node.scrollLeft > 0, await region.elementHandle()); keyboardRegions++;
      }
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      const intentionalViteDiagnostics = errors.filter(text => text.includes('WebSocket connection') || text.startsWith('[vite] failed to connect to websocket.'));
      const applicationErrors = errors.filter(text => !intentionalViteDiagnostics.includes(text));
      const record = { width, states, keyboardControls, keyboardRegions, fonts, equations, overflowMath, figures, pageOverflow, applicationErrors, intentionalViteDiagnostics, warnings, failedRequests };
      results.push(record); fs.writeFileSync(path.join(output, 'in-progress.json'), JSON.stringify(results, null, 2));
      assert.deepEqual(applicationErrors, []); assert.deepEqual(failedRequests, []); assert.equal(pageOverflow, false);
      assert.equal(await lesson.locator('.katex-error').count(), 0); assert.equal(await lesson.locator('p p,p div,p section').count(), 0);
      if (!process.env.COMBINATORIAL_AUDIT) {
        assert.deepEqual(overflowMath, []); assert.deepEqual(figures.flatMap(figure => figure.clipped), []);
        assert(figures.every(figure => figure.minimumTextPx >= 14));
      }
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), browser: browser.version(), results }, null, 2));
  console.log(JSON.stringify(results.map(({ width, states, keyboardControls, overflowMath, figures }) => ({ width, states, keyboardControls, overflowMath, figures })), null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
