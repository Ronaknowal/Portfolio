const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/sets-logic-browser');
const normalize = text => text.replace(/\s+/g, ' ').trim();
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const { setsLogicExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/sets-logic-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of (process.env.SETS_LOGIC_WIDTHS || '1440,390,320').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/sets-logic-relations-proof-techniques?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.sets-logic-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const record = { width, anchors: [], states: [], captures: [], programs: [], fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) };
      assert(record.fonts.length > 0, 'actual webfonts loaded');
      const region = name => lesson.getByRole('region', { name, exact: true });
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(100);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) }); record.captures.push(file);
      }
      async function press(target, name) {
        const button = target.getByRole('button', { name, exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'), name);
        await page.keyboard.press('Enter');
      }
      await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
      for (const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await link.getAttribute('href');
        const target = lesson.locator(`[id="${href.slice(1)}"]`);
        assert.equal(await target.count(), 1);
        await link.focus(); await page.keyboard.press('Enter');
        await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 40 && box.top <= 150; }, href.slice(1));
        record.anchors.push(href); await shot(target, `ordinary-section-${record.anchors.length}`);
      }
      for (const [index, figure] of (await lesson.locator('figure').all()).entries()) await shot(figure, `inline-${index}`);
      for (const lab of await lesson.locator('.sets-logic-lab').all()) await shot(lab, `initial-${(await lab.getAttribute('aria-label')).split(' ')[0]}`);
      const sets = region('Set membership regions investigation');
      assert((await sets.getByRole('status').innerText()).includes('Selected: {Bo}'));
      await press(sets, 'Ada badged');
      assert((await sets.getByRole('status').innerText()).includes('Selected: {Ada, Bo}'));
      for (const [key, expected] of [['union', '{Ada, Bo, Cam}'], ['difference', '∅'], ['reverseDifference', '{Cam}'], ['complement', '{Cam, Dee}'], ['symmetricDifference', '{Cam}']]) {
        await sets.getByLabel('Selection rule', { exact: true }).selectOption(key);
        assert((await sets.getByRole('status').innerText()).includes(`Selected: ${expected}`)); record.states.push(['set', key]);
      }
      await press(sets, 'Make both empty'); await sets.getByLabel('Selection rule').selectOption('complement');
      assert((await sets.getByRole('status').innerText()).includes('{Ada, Bo, Cam, Dee}'));
      await shot(sets.locator('svg'), 'set-empty-complement');
      await press(sets, 'Put everyone in both'); await sets.getByLabel('Selection rule').selectOption('intersection');
      await shot(sets.locator('svg'), 'set-all-overlap');
      await press(sets, 'Reset');
      const truth = region('Truth assignments and argument validity investigation');
      for (const [key, counters, admitted] of [['consequent', 1, 2], ['ponens', 0, 1], ['tollens', 0, 1], ['contradiction', 0, 0], ['equivalence', 0, 3]]) {
        await truth.getByLabel('Argument to investigate').selectOption(key);
        assert.equal(await truth.locator('.sets-logic-countermodel').count(), counters);
        assert.equal(await truth.locator('.sets-logic-admitted,.sets-logic-countermodel').count(), admitted);
        if (key === 'contradiction') assert((await truth.getByRole('status').innerText()).includes('No world'));
        record.states.push(['argument', key]);
      }
      await press(truth, 'Reset'); await shot(truth.locator('.sets-logic-worlds'), 'countermodel-worlds');
      await press(truth, 'Q'); await truth.getByLabel('Proposed conclusion').selectOption('implication');
      assert((await truth.getByRole('status').innerText()).includes('Valid argument'));
      await press(truth, 'Reset');
      const quantified = region('Quantifier order and witnesses investigation');
      for (const [rows, columns] of [[3, 3], [0, 3], [3, 0], [0, 0], [1, 1], [2, 1]]) {
        await quantified.getByLabel('Jobs in the domain').selectOption(String(rows));
        await quantified.getByLabel('Reviewers in the domain').selectOption(String(columns));
        const each = rows === 0 || (columns >= rows);
        const common = columns > 0 && rows <= 1;
        for (const [mode, expected] of [['each', each], ['common', common], ['notEach', !each], ['notCommon', !common]]) {
          await quantified.getByLabel('Quantified claim').selectOption(mode);
          assert.equal(normalize(await quantified.getByRole('status').innerText()), `This claim is ${expected}.`);
          record.states.push(['quantifiers', rows, columns, mode]);
        }
      }
      await quantified.getByLabel('Jobs in the domain').selectOption('0');
      await quantified.getByLabel('Reviewers in the domain').selectOption('0');
      await shot(quantified, 'quantifier-empty-domains');
      await press(quantified, 'Reset'); await press(quantified, 'Let Bo review every job');
      await quantified.getByLabel('Quantified claim').selectOption('common');
      assert((await quantified.getByRole('status').innerText()).includes('true'));
      await shot(quantified, 'quantifier-common-witness');
      await press(quantified, 'Scan has reviewer Bo');
      assert((await quantified.getByRole('status').innerText()).includes('false'));
      await press(quantified, 'Clear assignments'); assert((await quantified.getByRole('status').innerText()).includes('false')); await press(quantified, 'Reset');
      const relation = region('Relation properties and grouping investigation');
      for (const key of ['nearby', 'moduloThree', 'identity', 'divisors', 'noLeast']) {
        await relation.getByLabel('Relation preset').selectOption(key);
        for (const property of ['reflexive', 'symmetric', 'transitive', 'antisymmetric']) {
          await relation.getByLabel('Inspect an axiom').selectOption(property);
          record.states.push(['relation', key, property]);
        }
        if (key === 'moduloThree') { assert.equal(await relation.locator('.sets-logic-classes > div').count(), 3); await shot(relation.locator('.sets-logic-classes'), 'relation-classes'); }
        if (key === 'noLeast') { assert((await relation.innerText()).includes('Least: ∅')); await shot(relation.locator('svg').last(), 'order-no-least'); }
      }
      await relation.getByLabel('Relation preset').selectOption('identity');
      await press(relation, 'Relation 0 to 0'); await relation.getByLabel('Inspect an axiom').selectOption('reflexive');
      assert((await relation.locator('.sets-logic-properties').innerText()).includes('fails at (0)'));
      await shot(relation.locator('svg').first(), 'missing-reflexive-loop');
      await press(relation, 'Reset'); await shot(relation.locator('svg').first(), 'missing-transitive-edge');
      const square = region('Induction square border investigation');
      for (let index = 0; index < 3; index++) await press(square, 'Back');
      assert(await square.getByRole('button', { name: 'Back', exact: true }).isDisabled());
      for (let n = 0; n < 8; n++) {
        assert.equal(await square.locator('svg rect').count(), (n + 1) ** 2);
        assert((await square.getByRole('status').innerText()).startsWith(`${n * n} + ${2 * n + 1} = ${(n + 1) ** 2}`));
        record.states.push(['square', n]);
        if (n < 7) await press(square, 'Next square');
      }
      assert(await square.getByRole('button', { name: 'Next square', exact: true }).isDisabled());
      await shot(square, 'induction-last-square'); await press(square, 'Reset');
      const diagonal = region('Diagonal missing subset investigation');
      assert.equal(await diagonal.locator('.sets-logic-constructed').count(), 0);
      await press(diagonal, 'Reveal the diagonal subset');
      assert((await diagonal.locator('.sets-logic-constructed').innerText()).includes('D = {1, 3}'));
      for (let row = 0; row < 4; row++) { await diagonal.getByLabel('Compare D with this row').selectOption(String(row)); assert((await diagonal.getByRole('status').innerText()).includes(`D ≠ f(${row})`)); record.states.push(['diagonal', row]); }
      await press(diagonal, 'Element 0 in subset f(0)');
      assert((await diagonal.locator('.sets-logic-constructed').innerText()).includes('D = {0, 1, 3}'));
      await press(diagonal, 'Element 2 in subset f(0)');
      assert((await diagonal.locator('.sets-logic-constructed').innerText()).includes('D = {0, 1, 3}'));
      await shot(diagonal.locator('.sets-logic-constructed'), 'diagonal-changed-output'); await press(diagonal, 'Reset');
      assert.equal(await diagonal.locator('.sets-logic-constructed').count(), 0);
      for (const [index, example] of Object.values(examples).entries()) {
        const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(normalize(await program.locator('h3').innerText()), normalize(example.title));
        const codes = program.locator(':scope > div');
        assert.equal(normalize(await codes.nth(0).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(example.code));
        assert.equal(normalize(await codes.nth(1).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''))), normalize(example.expected));
        const previous = await program.evaluate(node => node.previousElementSibling.textContent);
        assert.equal(normalize(previous), normalize(`Before running: ${example.question}`));
        record.programs.push(example.title);
      }
      assert.equal(await lesson.locator('.python-example').count(), 11);
      await shot(lesson.locator('.python-example').nth(2), 'quantifier-program');
      await shot(lesson.locator('.python-example').last().locator(':scope > div').last(), 'policy-output');
      for (const [index, practice] of (await lesson.locator('.sets-logic-practice').all()).entries()) {
        const disclosures = practice.locator(':scope > details');
        assert.equal(await disclosures.count(), 2);
        assert.equal(await disclosures.nth(1).getAttribute('open'), null);
        await disclosures.nth(0).locator('summary').focus(); await page.keyboard.press('Enter');
        await disclosures.nth(1).locator('summary').focus(); await page.keyboard.press('Enter');
        if ([2, 3, 6, 9, 10].includes(index)) await shot(practice, `practice-${index}`);
      }
      const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth })));
      assert(math.every(box => box.scroll <= box.width + 1), `equation fit at ${width}: ${JSON.stringify(math)}`);
      for (const [index, equation] of (await lesson.locator('.katex-display').all()).entries()) await shot(equation, `equation-${index}`);
      const labels = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => {
        const rect = svg.getBoundingClientRect();
        return [...svg.querySelectorAll('text')].flatMap(text => { const box = text.getBoundingClientRect(); return box.left < rect.left - 2 || box.right > rect.right + 2 || box.top < rect.top - 2 || box.bottom > rect.bottom + 2 ? [text.textContent] : []; });
      }));
      assert.deepEqual(labels, [], `SVG label clipping at ${width}`);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), `document overflow ${width}`);
      const sources = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, target: node.target, rel: node.rel })));
      assert(sources.every(link => link.href.startsWith('https:') && link.target === '_blank' && link.rel.includes('noreferrer')));
      await shot(lesson.locator('.lesson-sources'), 'learning-resources');
      record.math = math; record.sources = sources; records.push(record); await page.close();
      fs.writeFileSync(path.join(directory, 'progress.json'), JSON.stringify({ checkedAt: new Date().toISOString(), records, errors }, null, 2));
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, errors }, null, 2));
    console.log('Sets/logic actual-font desktop and mobile interactions, ordinary reading, programs, anchors and keyboard passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });


