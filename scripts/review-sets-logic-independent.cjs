const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const output = 'scratch/sets-logic-independent';
fs.mkdirSync(output, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/sets-logic-relations-proof-techniques?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.sets-logic-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const region = name => lesson.getByRole('region', { name, exact: true });
      const captures = [];
      async function shot(target, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        const file = `${output}/${name}-${width}.png`;
        await target.screenshot({ path: file });
        captures.push(file);
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      async function enterButton(target, name) {
        const button = target.getByRole('button', { name, exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter');
      }
      const sets = region('Set membership regions investigation');
      await enterButton(sets, 'Put everyone in both');
      assert.equal(await sets.locator('svg .sets-logic-amber').count(), 4);
      await shot(sets.locator('svg'), 'all-four-in-overlap');
      await sets.getByRole('combobox').selectOption('symmetricDifference');
      assert((await sets.getByRole('status').innerText()).includes('Selected: ∅'));
      await enterButton(sets, 'Make both empty');
      await sets.getByRole('combobox').selectOption('complement');
      assert((await sets.getByRole('status').innerText()).includes('{Ada, Bo, Cam, Dee}'));
      await enterButton(sets, 'Reset');

      const truth = region('Truth assignments and argument validity investigation');
      await enterButton(truth, 'P → Q');
      await enterButton(truth, 'Q');
      await truth.getByLabel('Proposed conclusion').selectOption('conjunction');
      assert.equal(await truth.locator('.sets-logic-countermodel').count(), 3);
      await enterButton(truth, 'P'); await enterButton(truth, 'Q');
      assert.equal(await truth.locator('.sets-logic-countermodel').count(), 0);
      assert.equal(await truth.locator('.sets-logic-admitted').count(), 1);
      await shot(truth.locator('.sets-logic-worlds'), 'changed-conjunction-argument');
      await enterButton(truth, 'Reset');

      const quantifiers = region('Quantifier order and witnesses investigation');
      await quantifiers.getByLabel('Reviewers in the domain').selectOption('2');
      await enterButton(quantifiers, 'Let Bo review every job');
      await quantifiers.getByLabel('Quantified claim').selectOption('common');
      assert((await quantifiers.getByRole('status').innerText()).includes('true'));
      await enterButton(quantifiers, 'Scan has reviewer Bo');
      assert((await quantifiers.getByRole('status').innerText()).includes('false'));
      await quantifiers.getByLabel('Quantified claim').selectOption('each');
      assert((await quantifiers.getByRole('status').innerText()).includes('true'));
      await shot(quantifiers, 'separate-witnesses-no-common');
      await quantifiers.getByLabel('Jobs in the domain').selectOption('0');
      await quantifiers.getByLabel('Reviewers in the domain').selectOption('0');
      for (const [mode, expected] of [['each', 'true'], ['common', 'false'], ['notEach', 'false'], ['notCommon', 'true']]) {
        await quantifiers.getByLabel('Quantified claim').selectOption(mode);
        assert((await quantifiers.getByRole('status').innerText()).includes(expected));
      }
      await enterButton(quantifiers, 'Reset');

      const relation = region('Relation properties and grouping investigation');
      await relation.getByLabel('Relation preset').selectOption('identity');
      await enterButton(relation, 'Relation 0 to 0');
      await relation.getByLabel('Inspect an axiom').selectOption('reflexive');
      assert((await relation.locator('.sets-logic-properties').innerText()).includes('fails at (0)'));
      await shot(relation.locator('svg').first(), 'missing-loop');
      await enterButton(relation, 'Reset');
      await enterButton(relation, 'Relation 0 to 2');
      assert((await relation.locator('.sets-logic-properties').innerText()).includes('fails at (0, 2, 3)'));
      await shot(relation.locator('svg').first(), 'changed-transitivity-obligation');
      await relation.getByLabel('Relation preset').selectOption('moduloThree');
      assert.deepEqual(await relation.locator('.sets-logic-classes strong').allTextContents(), ['{0, 3}', '{1, 4}', '{2, 5}']);
      const pan = relation.getByRole('region', { name: 'Editable relation matrix; scroll horizontally if necessary' });
      if (width === 320) {
        await pan.focus(); await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => document.querySelector('.sets-logic-matrix-scroll').scrollLeft > 0);
      }
      await relation.getByLabel('Relation preset').selectOption('noLeast');
      assert((await relation.innerText()).includes('Least: ∅'));
      await shot(relation.locator('svg').last(), 'incomparable-minimal-elements');
      await enterButton(relation, 'Reset');

      const square = region('Induction square border investigation');
      await enterButton(square, 'Back');
      assert.equal(await square.locator('svg rect').count(), 9);
      assert((await square.getByRole('status').innerText()).startsWith('4 + 5 = 9'));
      await shot(square, 'changed-square-border');
      await enterButton(square, 'Reset');
      const diagonal = region('Diagonal missing subset investigation');
      await enterButton(diagonal, 'Reveal the diagonal subset');
      await enterButton(diagonal, 'Element 2 in subset f(0)');
      assert((await diagonal.locator('.sets-logic-constructed').innerText()).includes('D = {1, 3}'));
      await enterButton(diagonal, 'Element 1 in subset f(1)');
      assert((await diagonal.locator('.sets-logic-constructed').innerText()).includes('D = {3}'));
      await diagonal.getByLabel('Compare D with this row').selectOption('1');
      assert((await diagonal.getByRole('status').innerText()).includes('f(1) says in, while D says out'));
      await shot(diagonal, 'changed-diagonal-escape');
      await enterButton(diagonal, 'Reset');

      for (let index = 0; index < 5; index++) await shot(lesson.locator('.sets-logic-figure').nth(index), 'inline-' + index);
      const practice = lesson.locator('.sets-logic-practice').nth(6);
      await practice.locator('summary').last().focus(); await page.keyboard.press('Enter');
      assert((await practice.innerText()).includes('G is not'));
      await shot(practice, 'representative-independence-practice');
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      assert.deepEqual(errors, []);
      records.push({ width, captures, fonts, errors, keyboard: 'All changed buttons activated with Enter; matrix pan tested at320.' });
      console.log('Sets independent browser passed', width);
      await page.close();
    }
    const source = 'src/learn/data/sets-logic-models.js';
    fs.writeFileSync(output + '/browser-results.json', JSON.stringify({ reviewedAt: new Date().toISOString(), passed: true, modelSha256: createHash('sha256').update(fs.readFileSync(source)).digest('hex'), records }, null, 2) + '\n');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
