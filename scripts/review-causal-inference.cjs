const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const directory = path.resolve('scratch/causal-inference-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { causalInferenceExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/causal-inference-examples.js')));
  const model = await import(pathToFileURL(path.resolve('src/learn/data/causal-inference-models.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [], environmentMessages = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        const url = message.location().url;
        if ((url.endsWith('/@vite/client') && message.text().startsWith('[vite] failed to connect to websocket.')) ||
            (url.startsWith('https://fonts.googleapis.com/') && message.text() === 'Failed to load resource: net::ERR_NETWORK_ACCESS_DENIED')) {
          environmentMessages.push({ width, url, text: message.text() });
        } else errors.push(message.text());
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/causal-inference-do-calculus?module=math-foundations');
      const lesson = page.locator('.causal-inference-lesson');
      await lesson.waitFor();
      const record = { width, anchors: [], captures: [], controls: [], code: [] };
      const shot = async (locator, name) => {
        await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + scrollY - 90));
        const filename = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, filename) });
        record.captures.push(filename);
      };
      const fact = (region, name) => region.locator('.causal-facts > div').filter({ has: page.locator('dt', { hasText: name }) }).locator('dd');
      const pressButton = async (region, name, key = 'Enter') => {
        const button = region.getByRole('button', { name, exact: true });
        await button.focus();
        const focused = await button.evaluate(element => element === document.activeElement && getComputedStyle(element).outlineStyle !== 'none');
        assert(focused, `visible keyboard focus: ${name}`);
        await page.keyboard.press(key);
      };
      const rangeKey = async (region, name, key) => {
        const slider = region.getByRole('slider', { name, exact: true });
        await slider.focus(); await page.keyboard.press(key);
        return Number(await slider.inputValue());
      };

      await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
      for (const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await link.getAttribute('href');
        await link.focus(); await page.keyboard.press('Enter');
        await page.waitForFunction(value => location.hash === value, href);
        await page.waitForTimeout(65);
        const heading = page.locator(`[id="${href.slice(1)}"]`);
        const bounds = await heading.boundingBox();
        assert(bounds && bounds.y >= 45 && bounds.y < 200, `${href}: top ${bounds?.y}`);
        record.anchors.push({ href, top: bounds.y });
        if (width !== 320) await shot(heading, `ordinary-section-${record.anchors.length}`);
      }
      assert.equal(record.anchors.length, 12);
      for (const [index, figure] of (await lesson.locator('[data-causal-figure]').all()).entries()) await shot(figure, `ordinary-figure-${index + 1}`);
      for (const [index, lab] of (await lesson.locator('[data-causal-lab]').all()).entries()) {
        await shot(lab, `ordinary-lab-${index + 1}`);
        const graph = lab.locator('.causal-graph').first();
        if (await graph.count()) await shot(graph, `ordinary-lab-graph-${index + 1}`);
      }

      const paths = lesson.locator('[data-causal-lab="paths"]');
      for (const [preset, node, initiallySeparated] of [['fork', 'Z', false], ['chain', 'M', false], ['collider', 'C', true], ['descendant', 'D', true]]) {
        await paths.getByLabel('Path structure').selectOption(preset);
        assert((await paths.locator('.causal-feedback').innerText()).includes(initiallySeparated ? 'All X–Y paths' : 'An active X–Y path'));
        const checkbox = paths.getByRole('checkbox', { name: node, exact: true });
        await checkbox.focus(); await page.keyboard.press('Space');
        assert((await paths.locator('.causal-feedback').innerText()).includes(initiallySeparated ? 'An active X–Y path' : 'All X–Y paths'));
        assert.equal(await paths.locator('.causal-conditioned-ring').count(), 1);
        if (preset === 'descendant') await shot(paths.locator('.causal-investigation-layout'), 'conditioned-descendant');
      }
      await pressButton(paths, 'Reset path');
      assert.equal(await paths.getByLabel('Path structure').inputValue(), 'fork');
      assert.equal(await paths.locator('input:checked').count(), 0);
      record.controls.push('four motifs, keyboard conditioning including descendant, path status, unchanged edges and reset');

      const adjustment = lesson.locator('[data-causal-lab="adjustment"]');
      assert.equal(await fact(adjustment, 'Raw observed risk difference').innerText(), '0.22');
      await pressButton(adjustment, 'Randomize assignment');
      assert.equal(await fact(adjustment, 'Raw observed risk difference').innerText(), '0.1');
      await pressButton(adjustment, 'Reverse targeting', 'Space');
      assert.equal(await fact(adjustment, 'Raw observed risk difference').innerText(), '-0.02');
      assert.equal(await fact(adjustment, 'Adjusted observed risk difference').innerText(), '0.1');
      await shot(adjustment.locator('.causal-weight-comparison'), 'reversed-composition');
      await rangeKey(adjustment, 'Offer probability in low group', 'Home');
      assert((await fact(adjustment, 'Adjusted observed risk difference').innerText()).includes('Not identified'));
      assert((await adjustment.locator('.causal-feedback').innerText()).includes('lacks one treatment state'));
      await rangeKey(adjustment, 'Offer probability in high group', 'Home');
      assert((await adjustment.locator('.causal-mixture').first().innerText()).includes('No such cases'));
      await shot(adjustment.locator('.causal-weight-comparison'), 'no-treated-cases');
      await pressButton(adjustment, 'Reset population');
      await rangeKey(adjustment, 'High-activity population share', 'End');
      await rangeKey(adjustment, 'Offer probability in low group', 'Home');
      assert.equal(await fact(adjustment, 'Adjusted observed risk difference').innerText(), '0.1');
      await pressButton(adjustment, 'Reset population', 'Space');
      assert.equal(await fact(adjustment, 'Raw observed risk difference').innerText(), '0.22');
      record.controls.push('randomization, assignment reversal, zero overlap, absent treatment group, irrelevant empty stratum and reset');

      const rules = lesson.locator('[data-causal-lab="rules"]');
      for (const [key, preset] of Object.entries(model.CAUSAL_RULE_PRESETS)) {
        await rules.getByLabel('Rule and causal story').selectOption(key);
        const result = model.inspectDoRule(preset);
        assert((await rules.locator('.causal-feedback').innerText()).includes(result.valid ? 'Separation test passes.' : 'Separation test fails.'));
        assert.equal(await rules.locator('path.causal-edge.is-cut').count(), result.removed.length);
        assert.equal(await rules.locator('.causal-rule-expression code').count(), 2);
        if (key === 'exchangeFails') assert.equal(await rules.locator('.causal-rule-expression code').first().innerText(), 'p(y | do(z))');
        if (key === 'deletion') assert.equal(await rules.locator('.causal-rule-expression code').last().innerText(), 'p(y)');
        if (key === 'deletionFails') {
          assert((await fact(rules, 'Eligible action nodes Z(W)').innerText()).includes('Empty'));
          await shot(rules.locator('.causal-paired-graphs'), 'rule-three-ancestor');
        }
      }
      await pressButton(rules, 'Reset rule');
      assert.equal(await rules.getByLabel('Rule and causal story').inputValue(), 'observation');
      record.controls.push('six rule cases, incoming/outgoing cut counts, rule-three ancestor exception and reset');

      const frontdoor = lesson.locator('[data-causal-lab="frontdoor"]');
      assert.equal(await fact(frontdoor, 'Model-known do risks X=0 / X=1').innerText(), '0.25 / 0.6');
      const direct = frontdoor.getByRole('checkbox');
      await direct.focus(); await page.keyboard.press('Space');
      assert.equal(await fact(frontdoor, 'Model-known do risks X=0 / X=1').innerText(), '0.25 / 0.7');
      assert.equal(await frontdoor.locator('.causal-frontdoor-chain strong').last().innerText(), '0.3 / 0.65');
      assert((await frontdoor.locator('.causal-feedback').innerText()).includes('violates full mediation'));
      assert.equal(await frontdoor.locator('.causal-edge').count(), 5);
      await shot(frontdoor.locator('.causal-investigation-layout'), 'frontdoor-direct-path');
      await rangeKey(frontdoor, 'P(M=1 | X=0)', 'Home');
      await rangeKey(frontdoor, 'P(M=1 | X=1)', 'End');
      assert((await frontdoor.locator('.causal-feedback').innerText()).includes('zero mass'));
      assert((await frontdoor.locator('.causal-frontdoor-chain strong').last().innerText()).includes('Not identified'));
      await pressButton(frontdoor, 'Reset frontdoor');
      await rangeKey(frontdoor, 'P(M=1 | X=0)', 'Home');
      await rangeKey(frontdoor, 'P(M=1 | X=1)', 'Home');
      assert.equal(await frontdoor.locator('.causal-frontdoor-chain strong').last().innerText(), '0.2 / 0.2');
      assert((await frontdoor.locator('.causal-feedback').innerText()).includes('matches the model-known'));
      await pressButton(frontdoor, 'Reset frontdoor', 'Space');
      record.controls.push('frontdoor valid calculation, direct-path violation, unsupported mediator/treatment pairs, constant mediator supported target and reset');

      const worlds = lesson.locator('[data-causal-lab="counterfactual"]');
      assert.equal(await fact(worlds, 'Counterfactual success given the evidence').innerText(), '0.166667');
      await rangeKey(worlds, 'Both-outcome success overlap', 'Home');
      assert.equal(await fact(worlds, 'Counterfactual success given the evidence').innerText(), '0');
      await rangeKey(worlds, 'Both-outcome success overlap', 'End');
      assert.equal(await fact(worlds, 'Counterfactual success given the evidence').innerText(), '0.333333');
      for (const observedTreatment of [0, 1]) for (const observedOutcome of [0, 1]) {
        await worlds.getByLabel('Observed treatment', { exact: true }).selectOption(String(observedTreatment));
        await worlds.getByLabel('Observed outcome', { exact: true }).selectOption(String(observedOutcome));
        await worlds.getByLabel('Counterfactual treatment', { exact: true }).selectOption(String(observedTreatment));
        assert.equal(await fact(worlds, 'Counterfactual success given the evidence').innerText(), String(observedOutcome));
      }
      await pressButton(worlds, 'Reset worlds');
      await shot(worlds.locator('.causal-response-types'), 'paired-worlds-posterior');
      record.controls.push('counterfactual coupling endpoints, unchanged experimental marginals, all four consistency states and reset');

      const programs = await lesson.locator('.python-example').all();
      assert.equal(programs.length, 10);
      assert.equal(await lesson.locator('.causal-example > p').count(), 10);
      for (const element of programs) {
        const title = await element.locator('h3').innerText();
        const example = Object.values(examples).find(value => value.title === title);
        assert(example, title);
        const blocks = await element.locator(':scope > div').all();
        assert.equal(blocks.length, 2);
        assert(normalize(await blocks[0].innerText()).includes(normalize(example.code)));
        assert.equal(normalize(await blocks[1].innerText()).replace(/^OUTPUT /, ''), normalize(example.expected));
        const dimensions = await blocks[0].evaluate(node => ({ client: node.clientWidth, scroll: node.scrollWidth, overflow: getComputedStyle(node).overflowX }));
        assert(dimensions.scroll <= dimensions.client + 1 || ['auto', 'scroll'].includes(dimensions.overflow));
        record.code.push({ title, ...dimensions });
      }
      await shot(lesson.locator('.causal-example').nth(2), 'preserved-example-question');
      await shot(programs[2].locator('.lesson-note'), 'preserved-example-output');
      const exercises = await lesson.locator('.causal-exercise').all();
      assert.equal(exercises.length, 11);
      for (const exercise of exercises) {
        const hints = exercise.locator('details').nth(0), solution = exercise.locator('details').nth(1);
        assert.equal(await hints.getAttribute('open'), null);
        assert.equal(await solution.getAttribute('open'), null);
        await hints.locator('summary').focus(); await page.keyboard.press('Enter');
        assert.equal(await solution.getAttribute('open'), null);
        await solution.locator('summary').focus(); await page.keyboard.press('Space');
        assert((await solution.innerText()).length > 150);
      }
      await shot(exercises[7], 'changed-counterfactual-solution');
      // Open longer derivations/tables only after preserving normal unexpanded reading captures.
      await lesson.locator('details').evaluateAll(elements => elements.forEach(element => { element.open = true; }));
      record.equations = [];
      for (const [index, equation] of (await lesson.locator('.katex-display').all()).entries()) {
        record.equations.push(await equation.evaluate(element => ({ client: element.clientWidth, scroll: element.scrollWidth })));
        if (width === 320) await shot(equation, `equation-${index + 1}`);
      }
      await shot(lesson.locator('.lesson-sources'), 'sources');
      record.externalLinks = await lesson.locator('.lesson-sources a').evaluateAll(elements => elements.map(element => ({ href: element.href, target: element.target, rel: element.rel })));
      assert.equal(record.externalLinks.length, 5);
      assert(record.externalLinks.every(link => link.href.startsWith('https:') && link.target === '_blank' && link.rel.includes('noreferrer')));
      record.mathErrors = await lesson.locator('.katex-error').count();
      record.fonts = await page.evaluate(() => ({
        status: document.fonts.status,
        faces: Array.from(document.fonts).map(face => ({ family: face.family, weight: face.weight, status: face.status })),
        lessonFamily: getComputedStyle(document.querySelector('.causal-inference-lesson')).fontFamily
      }));
      if (process.env.REQUIRE_WEB_FONTS === '1') {
        for (const family of ['Space Grotesk', 'JetBrains Mono']) {
          assert(record.fonts.faces.some(face => face.family.replaceAll('"', '') === family && face.status === 'loaded'), `Public font was not loaded: ${family}`);
        }
      }
      record.overflow = await lesson.locator('.causal-lab,[data-causal-figure],.katex-display').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 2).map(element => ({ className: element.className, client: element.clientWidth, scroll: element.scrollWidth })));
      record.pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      // Use screen coordinates: SVG text is positioned inside translated node groups.
      record.svgTextOverflow = await lesson.locator('svg text').evaluateAll(elements => elements.flatMap(element => {
        const text = element.getBoundingClientRect(), svg = element.ownerSVGElement.getBoundingClientRect();
        return text.left < svg.left - 1 || text.right > svg.right + 1 || text.top < svg.top - 1 || text.bottom > svg.bottom + 1 ? [{ text: element.textContent }] : [];
      }));
      records.push(record);
      fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), records, errors, environmentMessages }, null, 2));
      await page.close();
    }
    assert.deepEqual(errors, []);
    assert(records.every(record => !record.pageOverflow && !record.mathErrors && !record.overflow.length && !record.svgTextOverflow.length), JSON.stringify(records.map(record => ({ width: record.width, overflow: record.overflow, svg: record.svgTextOverflow }))));
    console.log(JSON.stringify({ checkedAt: new Date().toISOString(), viewports: records.map(record => ({ width: record.width, controls: record.controls, anchors: record.anchors.length, equations: record.equations.length, programs: record.code.length, captures: record.captures.length })), errors, allPassed: true }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
