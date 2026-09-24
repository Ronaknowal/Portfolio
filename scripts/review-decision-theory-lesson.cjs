const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/decision-theory-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const { decisionTheoryExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/decision-theory-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of (process.env.DECISION_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/decision-theory-risk-cost-sensitive-decisions?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.decision-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      const record = { width, anchors: [], states: [], keyboard: [], captures: [], programs: [], fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) };
      assert(record.fonts.some(font => font.includes('Space Grotesk')), 'Actual site sans font loaded');
      assert(record.fonts.some(font => font.includes('JetBrains Mono')), 'Actual site code font loaded');
      const lab = name => lesson.getByRole('region', { name, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(80);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) }); record.captures.push(file);
      }
      async function range(region, label, value) {
        const input = region.locator('label.decision-control').filter({ hasText: label }).locator('input');
        assert.equal(await input.count(), 1, label);
        await input.fill(String(value)); await input.dispatchEvent('input');
      }
      async function reset(region) {
        const button = region.getByRole('button', { name: 'Reset', exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter');
        record.keyboard.push(`Reset: ${await region.getAttribute('aria-label')}`);
      }
      try {
        await shot(lesson.locator('.lesson-intro'), 'reading-intro');
        for (const anchor of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
          const href = await anchor.getAttribute('href');
          const target = lesson.locator(`[id="${href.slice(1)}"]`);
          assert.equal(await target.count(), 1, href);
          await anchor.focus(); await page.keyboard.press('Enter');
          await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 35 && box.top <= 155; }, href.slice(1));
          record.anchors.push(href); await shot(target, `reading-section-${record.anchors.length}`);
        }
        assert.equal(record.anchors.length, 13);
        for (const slider of await lesson.locator('input[type="range"]').all()) {
          const original = await slider.inputValue(), maximum = Number(await slider.getAttribute('max'));
          await slider.focus(); await page.keyboard.press(Number(original) === maximum ? 'ArrowLeft' : 'ArrowRight');
          assert.notEqual(await slider.inputValue(), original);
          record.keyboard.push(normalize(await slider.locator('..').innerText()));
          await slider.fill(original); await slider.dispatchEvent('input');
        }
        for (const select of await lesson.locator('select').all()) {
          const original = await select.inputValue();
          await select.selectOption({ index: 0 }); await select.focus();
          await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
          assert.equal(await select.inputValue(), '1');
          record.keyboard.push('Select ArrowDown/Enter'); await select.selectOption(original);
        }
        const conditional = lab('Read the lower loss envelope');
        for (const [p, text] of [[0, 'Best action: Release'], [0.125, 'Best actions (tie): Release or Quarantine'], [1, 'Best action: Quarantine']]) {
          await range(conditional, 'Probability faulty', p); assert((await conditional.locator('.decision-result').innerText()).includes(text)); record.states.push(['loss-table', p, text]);
        }
        await range(conditional, 'Probability faulty', 0.125); await shot(conditional.locator('svg'), 'risk-envelope-tie');
        for (const input of await conditional.getByRole('spinbutton').all()) await input.fill('4');
        assert((await conditional.locator('.decision-result').innerText()).includes('identical'));
        record.states.push(['loss-table', 'identical rows']); await shot(conditional, 'identical-loss-table'); await reset(conditional);
        const procedure = lab('Hold the state fixed, then average the states');
        await range(procedure, 'Prior probability of high state', 0.1);
        assert((await procedure.locator('.decision-result').innerText()).includes('Best rule: Always low'));
        await procedure.locator('select').selectOption('0'); assert((await procedure.locator('.decision-result').innerText()).includes('= 0.1'));
        record.states.push(['procedure', 'prior .1']); await shot(procedure.locator('.decision-state-columns'), 'fixed-state-versus-prior');
        await range(procedure, 'Prior probability of high state', 0.2); assert((await procedure.locator('.decision-result').innerText()).includes('Best rules: Always low or Follow signal')); record.states.push(['procedure', 'exact prior .2 tie']); await reset(procedure);
        const provision = lab('See the shortage behind the weighted average');
        for (const [quantity, cost] of [[2, 10], [10, 6]]) {
          await range(provision, 'Order quantity', quantity); assert((await provision.locator('.decision-result').innerText()).includes(`Expected cost ${cost}.`)); record.states.push(['provisioning', quantity, cost]);
        }
        await shot(provision.locator('.decision-demand'), 'demand-surplus-quantile'); await shot(provision.locator('svg'), 'asymmetric-cost-curve'); await reset(provision);
        const capacity = lab('Add a real fallback, then share limited slots');
        await range(capacity, 'Probability faulty in fallback example', 0.075);
        assert((await capacity.innerText()).includes('Predict sound or Use fallback')); record.states.push(['fallback', '.075 tie']);
        await range(capacity, 'Probability faulty in fallback example', 0.7); assert((await capacity.innerText()).includes('Predict faulty or Use fallback')); record.states.push(['fallback', '.7 tie']);
        await range(capacity, 'Fixed fallback loss', 0); assert((await capacity.innerText()).includes('Best: Use fallback'));
        await range(capacity, 'Fixed fallback loss', 2); await range(capacity, 'Probability faulty in fallback example', 0.5); assert((await capacity.innerText()).includes('Best: Predict faulty'));
        for (const [slots, total] of [[0, 106.4], [2, 50.4], [6, 44.4]]) {
          await range(capacity, 'Quarantine slots for all six items', slots); assert((await capacity.locator('.decision-result').innerText()).includes(`loss ${total}.`)); record.states.push(['capacity', slots, total]);
        }
        await reset(capacity); await shot(capacity.locator('.decision-region'), 'fallback-region'); await shot(capacity.locator('.decision-items'), 'capacity-two-slots');
        const information = lab('Put the information on the correct side of the decision');
        assert((await information.locator('.decision-result').innerText()).includes('4.84'));
        await information.getByRole('checkbox').focus(); await page.keyboard.press('Space');
        assert(await information.getByRole('checkbox').isChecked()); assert((await information.locator('.decision-result').innerText()).includes('8.4'));
        record.keyboard.push('Information timing Space'); record.states.push(['test', 'late signal 8.4']); await shot(information.locator('.decision-tree'), 'information-too-late');
        await reset(information); await range(information, 'Prior faulty probability', 0.02); assert((await information.locator('.decision-result').innerText()).includes('0.14'));
        record.states.push(['test', 'prior .02 value .14']); await range(information, 'Prior faulty probability', 0.01); assert((await information.locator('.decision-result').innerText()).includes('value 0;'));
        await range(information, 'Prior faulty probability', 0); await range(information, 'False-positive P(+ | sound)', 0);
        assert((await information.innerText()).includes('undefined: impossible')); record.states.push(['test', 'null positive branch']); await shot(information.locator('.decision-tree'), 'impossible-signal'); await reset(information);
        await shot(information.locator('.decision-tree'), 'contingent-test-tree');
        const mix = lab('Balance the two state risks');
        for (const [weight, worst] of [[0, 4], [0.4, 2.4], [1, 6]]) { await range(mix, 'Probability of choosing rule A', weight); assert((await mix.locator('.decision-result').innerText()).includes(`worst ${worst}.`)); record.states.push(['mixture', weight, worst]); }
        await reset(mix); await shot(mix.locator('svg'), 'minimax-certificate-plane');
        const tail = lab('Change the consequence criterion explicitly');
        for (const [alpha, cvar] of [[0.8, 32.5], [0.9, 55], [0.95, 100], [0.99, 100]]) {
          await range(tail, 'Tail level alpha', alpha); assert((await tail.locator('.decision-result').innerText()).includes(`CVaR ${cvar};`)); record.states.push(['tail', alpha, cvar]);
        }
        await range(tail, 'Sure final outcome', 90); assert((await tail.innerText()).includes('Prefer the lottery.'));
        await range(tail, 'Sure final outcome', 95); assert((await tail.innerText()).includes('Prefer the sure outcome.')); record.states.push(['utility', '90 lottery, 95 sure']);
        await reset(tail); await shot(tail.locator('svg'), 'utility-chord'); await shot(tail.locator('.decision-tail'), 'atom-aware-tail');
        const capstone = lab('Test first; reallocate after the result');
        assert((await capstone.locator('.decision-result').innerText()).includes('= 41.'));
        assert.equal(await capstone.locator('.decision-candidate-costs p').filter({ hasText: 'optimal' }).count(), 2);
        await range(capstone, 'Price of the one test', 10.4); assert.equal(await capstone.locator('.decision-candidate-costs p').filter({ hasText: 'optimal' }).count(), 3);
        record.states.push(['capstone', 'price 10.4 three-way tie']);
        await range(capstone, 'Price of the one test', 12); assert((await capstone.locator('.decision-candidate-costs p').last().innerText()).includes('optimal'));
        record.states.push(['capstone', 'price 12 no test']); await shot(capstone.locator('.decision-candidate-costs'), 'no-test-beats-priced-information'); await reset(capstone);
        await range(capstone, 'Available quarantine slots', 1); await capstone.locator('select').selectOption('4');
        assert((await capstone.locator('.decision-result').innerText()).includes('58.2')); record.states.push(['capstone', 'capacity 1']); await shot(capstone.locator('.decision-branches'), 'changed-capacity-contingent-policy'); await reset(capstone);
        for (const [index, figure] of (await lesson.locator('figure').all()).entries()) await shot(figure, `inline-figure-${index}`);
        for (const details of await lesson.locator(':scope > details').all()) {
          await details.locator(':scope > summary').focus(); await page.keyboard.press('Enter'); assert.notEqual(await details.getAttribute('open'), null); record.keyboard.push('Optional derivation Enter');
        }
        for (const [index, practice] of (await lesson.locator('.decision-practice').all()).entries()) {
          const disclosures = practice.locator(':scope > details'); assert.equal(await disclosures.count(), 2);
          assert.equal(await disclosures.nth(1).getAttribute('open'), null);
          for (const disclosure of await disclosures.all()) { await disclosure.locator('summary').focus(); await page.keyboard.press('Enter'); assert.notEqual(await disclosure.getAttribute('open'), null); }
          if ([0, 9, 12, 13].includes(index)) await shot(practice, `explained-practice-${index}`);
        }
        record.practice = await lesson.locator('.decision-practice').count();
        for (const example of Object.values(examples)) {
          const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
          assert.equal(await program.count(), 1);
          const blocks = program.locator(':scope > div'); assert.equal(await blocks.count(), 2);
          for (const [index, expected] of [[0, example.code], [1, example.expected]]) assert.equal(normalize(await blocks.nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(expected));
          assert.equal(normalize(await program.evaluate(node => node.previousElementSibling.textContent)), normalize(`Before running: ${example.question}`));
          record.programs.push(example.title);
        }
        await shot(lesson.locator('.python-example').last(), 'complete-capstone-program');
        await shot(lesson.locator('.python-example').last().locator(':scope > div').last(), 'executed-capstone-output');
        record.math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth, text: node.textContent })));
        assert(record.math.every(box => box.scroll <= box.width + 1), `Math overflow ${width}: ${JSON.stringify(record.math.filter(box => box.scroll > box.width + 1))}`);
        for (const [index, equation] of (await lesson.locator('.katex-display').all()).entries()) await shot(equation, `equation-${index}`);
        const clipped = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => { const rect = svg.getBoundingClientRect(); return [...svg.querySelectorAll('text')].flatMap(text => { const box = text.getBoundingClientRect(); return box.left < rect.left - 2 || box.right > rect.right + 2 || box.top < rect.top - 2 || box.bottom > rect.bottom + 2 ? [text.textContent] : []; }); }));
        assert.deepEqual(clipped, [], `SVG clipping ${width}`);
        assert.equal(await lesson.locator('.katex-error').count(), 0);
        assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), `Document overflow ${width}`);
        assert(!(await lesson.innerText()).includes('\\u2212'), 'No literal Unicode escape in rendered prose');
        record.sources = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, target: node.target, rel: node.rel })));
        assert(record.sources.every(link => link.href.startsWith('https:') && link.target === '_blank' && link.rel.includes('noreferrer')));
        await shot(lesson.locator('.lesson-sources'), 'annotated-learning-resources');
        records.push(record);
        fs.writeFileSync(path.join(directory, 'progress.json'), JSON.stringify({ checkedAt: new Date().toISOString(), records, errors }, null, 2));
      } catch (error) {
        await page.screenshot({ path: path.join(directory, `failure-${width}.png`) });
        fs.writeFileSync(path.join(directory, 'failure.json'), JSON.stringify({ checkedAt: new Date().toISOString(), record, errors, error: String(error) }, null, 2));
        throw error;
      }
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, errors }, null, 2));
    console.log('Decision actual-font reading, controls, keyboard, program/output, anchors and geometry passed at all requested widths.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
