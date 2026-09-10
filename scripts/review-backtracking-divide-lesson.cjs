const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve(__dirname, '../scratch/backtracking-divide-lesson-review');
fs.mkdirSync(directory, { recursive: true });

async function capture(page, target, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await target.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

function subsetOracle(values, target) {
  const answers = [];
  for (let mask = 0; mask < 2 ** values.length; mask++) {
    const answer = values.flatMap((_, index) => mask & (1 << index) ? [index] : []);
    if (answer.reduce((sum, index) => sum + values[index], 0) === target) answers.push(answer);
  }
  return answers.map(JSON.stringify).sort();
}

function intervalOracle(values, low, high, kind) {
  let best = null;
  for (let start = low; start < high; start++) {
    for (let end = start + 1; end <= high; end++) {
      if (kind === 'prefix' && start !== low || kind === 'suffix' && end !== high) continue;
      const sum = values.slice(start, end).reduce((a, b) => a + b, 0);
      if (!best || sum > best.sum) best = { sum, start, end };
    }
  }
  return best;
}

async function stepAll(lab, check) {
  let states = 0;
  while (true) {
    await check();
    states++;
    const next = lab.getByRole('button', { name: 'Next event', exact: true });
    if (await next.isDisabled()) return states;
    assert.ok(states < 700);
    await next.click();
  }
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/backtracking-divide-and-conquer?module=data-structures-algorithms');
    const subsets = page.locator('[data-bd-lab="subsets"]');
    await subsets.waitFor({ timeout: 60000 });
    const queens = page.locator('[data-bd-lab="queens"]');
    const summary = page.locator('[data-bd-lab="summary"]');
    assert.equal(await page.locator('.python-example').count(), 9);
    assert.equal(await page.locator('[data-bd-lab]').count(), 3);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1);
    await page.screenshot({ path: path.join(directory, `reading-entry-${width}.png`) });
    await capture(page, page.locator('.bd-inline'), `inversion-boundary-${width}.png`);
    let subsetStates = 0;
    let pruneCaptured = false;
    for (const prune of [true, false]) {
      await subsets.getByRole('checkbox', { name: 'Use positive-value pruning', exact: true }).setChecked(prune);
      await subsets.getByRole('button', { name: 'Apply search', exact: true }).click();
      subsetStates += await stepAll(subsets, async () => {
        const message = await subsets.locator('.bd-status').innerText();
        const current = subsets.locator('.bd-choice-node--current');
        assert.equal(await current.count(), 1);
        if (message.startsWith('Prune:') && !pruneCaptured) {
          await capture(page, subsets, `subset-pruning-${width}.png`);
          pruneCaptured = true;
        }
      });
      assert.match(await subsets.locator('.bd-answer-strip strong').innerText(), /1 saved answers/);
      assert.match(await subsets.locator('.bd-status').innerText(), new RegExp(`${prune ? 13 : 15} entered calls`));
      assert.equal(await subsets.locator('.bd-path .bd-empty').innerText(), 'empty');
    }
    for (const fixture of [{ values: [2, 2], target: 2 }, { values: [], target: 0 }, { values: [], target: 1 }, { values: [1, 2, 3], target: 9 }, { values: [1, 2, 3], target: 3 }]) {
      await subsets.getByLabel('Values · at most three', { exact: true }).fill(fixture.values.join(','));
      await subsets.getByLabel('Target sum', { exact: true }).fill(String(fixture.target));
      await subsets.getByRole('button', { name: 'Apply search', exact: true }).click();
      await subsets.getByRole('button', { name: 'Finish exploration', exact: true }).click();
      const answers = await subsets.locator('.bd-answer-strip > div > span').evaluateAll(nodes => nodes.map(node => JSON.stringify((node.textContent.match(/positions \[(.*?)\]/)[1] || '').split(',').filter(value => value.trim()).map(Number))));
      assert.deepEqual(answers.sort(), subsetOracle(fixture.values, fixture.target));
    }
    await subsets.getByLabel('Values · at most three', { exact: true }).fill('-2, 3');
    await subsets.getByRole('button', { name: 'Apply search', exact: true }).click();
    assert.match(await subsets.getByRole('alert').innerText(), /1 through 9/);
    await subsets.getByRole('button', { name: 'Reset example', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await subsets.getByRole('alert').count(), 0);
    const scrollingTree = subsets.locator('.bd-diagram-scroll');
    await scrollingTree.focus();
    assert.equal(await scrollingTree.evaluate(node => getComputedStyle(node).outlineStyle), 'solid');
    if (width < 400) {
      const before = await scrollingTree.evaluate(node => node.scrollLeft);
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(200);
      assert.ok(await scrollingTree.evaluate((node, previous) => node.scrollLeft > previous, before));
    }

    let queenStates = 0;
    let attackCaptured = false;
    queenStates += await stepAll(queens, async () => {
      const queenCoordinates = await queens.locator('.bd-square--queen small').allTextContents();
      const points = queenCoordinates.map(value => value.split(',').map(Number));
      for (let a = 0; a < points.length; a++) for (let b = a + 1; b < points.length; b++) {
        assert.notEqual(points[a][0], points[b][0]);
        assert.notEqual(points[a][1], points[b][1]);
        assert.notEqual(Math.abs(points[a][0] - points[b][0]), Math.abs(points[a][1] - points[b][1]));
      }
      if ((await queens.locator('.bd-status').innerText()).startsWith('Reject')) {
        assert.ok(await queens.locator('.bd-square--attacker').count() > 0);
        assert.equal(await queens.locator('.bd-chessboard svg line').count(), await queens.locator('.bd-square--attacker').count());
        if (!attackCaptured) {
          await capture(page, queens, `queen-rejection-${width}.png`);
          attackCaptured = true;
        }
      }
    });
    assert.match(await queens.locator('.bd-status').innerText(), /2 solutions/);
    assert.equal(await queens.locator('.bd-square--queen').count(), 0);
    await queens.getByLabel('Board size', { exact: true }).selectOption('5');
    await queens.getByRole('button', { name: 'Apply board', exact: true }).click();
    let solutions = 0;
    while (!(await queens.getByRole('button', { name: 'Next solution', exact: true }).isDisabled())) {
      await queens.getByRole('button', { name: 'Next solution', exact: true }).click();
      assert.equal(await queens.locator('.bd-square--queen').count(), 5);
      solutions++;
      if (solutions === 1) await capture(page, queens, `queen-solution-${width}.png`);
    }
    assert.equal(solutions, 10);
    await queens.getByRole('button', { name: 'Finish exploration', exact: true }).click();
    assert.equal(await queens.locator('.bd-square--queen').count(), 0);

    let summaryNodes = 0;
    for (const values of [[-2, 4, -1, 3, -5, 2], [-8, -3, -6], [0, 0], [5], [9, -9, 9, -9, 9, -9, 9, -9]]) {
      await summary.getByLabel('Signed array · one to eight integers', { exact: true }).fill(values.join(','));
      await summary.getByRole('button', { name: 'Apply array', exact: true }).click();
      const buttons = summary.locator('.bd-split-level button');
      assert.equal(await buttons.count(), values.length * 2 - 1);
      for (let index = 0; index < await buttons.count(); index++) {
        await buttons.nth(index).click();
        const [low, high] = (await buttons.nth(index).getAttribute('aria-label')).match(/\d+/g).map(Number);
        const rows = await summary.locator('.bd-summary-table').last().locator('tr').evaluateAll(nodes => nodes.map(node => node.textContent));
        assert.ok(rows[0].includes(String(values.slice(low, high).reduce((a, b) => a + b, 0))));
        for (const kind of ['prefix', 'suffix', 'best']) {
          const expected = intervalOracle(values, low, high, kind);
          await summary.getByLabel('Highlight interval', { exact: true }).selectOption(kind);
          assert.match(await summary.locator('.bd-status').innerText(), new RegExp(`${kind} sum ${expected.sum}, interval \\[${expected.start}, ${expected.end}\\)`));
          assert.equal(await summary.locator('.bd-array-cell--selected').count(), expected.end - expected.start);
        }
        summaryNodes++;
      }
      if (values.length === 6 || values.length === 3) {
        await buttons.first().click();
        await summary.getByLabel('Highlight interval', { exact: true }).selectOption('crossing');
        await capture(page, summary, `summary-${values.length === 6 ? 'crossing' : 'negative'}-${width}.png`);
      }
    }
    await summary.getByLabel('Signed array · one to eight integers', { exact: true }).fill('1, nope');
    await summary.getByRole('button', { name: 'Apply array', exact: true }).click();
    assert.ok(await summary.getByRole('alert').count());
    await summary.getByRole('button', { name: 'Reset array', exact: true }).click();
    assert.equal(await summary.getByRole('alert').count(), 0);
    const practice = page.locator('[data-practice-topic="backtracking-divide-and-conquer"]');
    assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 6);
    assert.equal(await practice.locator('details[open]').count(), 0);
    await practice.locator('.dsa-practice__extension > summary').focus();
    await page.keyboard.press('Enter');
    assert.equal(await practice.locator('.dsa-practice__extension').evaluate(node => node.open), true);
    await capture(page, practice.locator('.dsa-practice__stage').first(), `practice-${width}.png`);
    const overflow = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
    assert.ok(overflow.scroll <= width + 1);
    results.push({ width, subsetStates, subsetFixtures: 5, queenStates, fiveQueenSolutions: solutions, summaryNodes, anchors: anchors.length, overflow, keyboard: 'reset, local tree scroll, selectors and optional practice passed' });
    await page.close();
  }
  assert.deepEqual(errors, []);
  await browser.close();
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ results, errors }, null, 2));
  console.log(JSON.stringify({ results, errors }, null, 2));
})().catch(error => { console.error(error); process.exit(1); });
