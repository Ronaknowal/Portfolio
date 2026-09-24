const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const directory = path.resolve(__dirname, '../scratch/greedy-exchange-lesson-review');
fs.mkdirSync(directory, { recursive: true });

async function capture(page, target, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await target.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

// Independent tiny-instance checker: occupied integer slots, not the model's
// pairwise compatibility function. All browser intervals have integer endpoints.
function intervalOracle(requests, objective) {
  let optimum = 0;
  for (let mask = 0; mask < 2 ** requests.length; mask++) {
    const occupied = new Set();
    let valid = true;
    let score = 0;
    for (let index = 0; index < requests.length; index++) {
      if (!(mask & (1 << index))) continue;
      const [start, finish, value = 1] = requests[index];
      for (let time = start; time < finish; time++) {
        if (occupied.has(time)) valid = false;
        occupied.add(time);
      }
      score += objective === 'count' ? 1 : value;
    }
    if (valid) optimum = Math.max(optimum, score);
  }
  return optimum;
}

async function inspectSelection(lab, requests, objective, final) {
  const kept = await lab.locator('.gx-interval--selected text').allTextContents();
  const occupied = new Set();
  let score = 0;
  for (const id of kept) {
    const [start, finish, value = 1] = requests[id.charCodeAt(0) - 65];
    for (let time = start; time < finish; time++) {
      assert.ok(!occupied.has(time), 'Highlighted selection must be feasible');
      occupied.add(time);
    }
    score += objective === 'count' ? 1 : value;
  }
  const comparison = await lab.locator('.gx-comparison').innerText();
  assert.match(comparison, new RegExp(`score ${score}\\b`));
  assert.match(comparison, new RegExp(`optimum: ${intervalOracle(requests, objective)},`));
  if (final) assert.ok(!comparison.includes('Finish the trace'));
  return kept;
}

async function inspectDeadline(lab, jobs) {
  const rows = await lab.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.children].map(cell => cell.textContent)));
  let completion = 0;
  const latenesses = [];
  for (const [id, actualCompletion, deadline, lateness] of rows) {
    completion += jobs[id][0];
    assert.equal(Number(actualCompletion), completion);
    assert.equal(Number(deadline), jobs[id][1]);
    assert.equal(Number(lateness), completion - jobs[id][1]);
    latenesses.push(Number(lateness));
  }
  const current = lab.locator('h4', { hasText: 'Current order' }).locator('..');
  const maximum = Math.max(...latenesses);
  assert.match(await current.innerText(), new RegExp(`Maximum signed lateness ${maximum}; maximum tardiness ${Math.max(0, maximum)}`));
  const stripOrder = await current.locator('.gx-job strong').allTextContents();
  assert.deepEqual(stripOrder, rows.map(row => row[0]));
  return { order: stripOrder, maximum };
}

function wholeOracle(capacity) {
  const items = [[10, 60], [20, 100], [30, 120]];
  let best = 0;
  for (let mask = 0; mask < 8; mask++) {
    let weight = 0;
    let value = 0;
    items.forEach(([w, v], index) => {
      if (mask & (1 << index)) { weight += w; value += v; }
    });
    if (weight <= capacity) best = Math.max(best, value);
  }
  return best;
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/greedy-algorithms-exchange-arguments?module=data-structures-algorithms');
    const intervals = page.locator('[data-gx-lab="intervals"]');
    await intervals.waitFor({ timeout: 60000 });
    const deadlines = page.locator('[data-gx-lab="deadlines"]');
    const capacityLab = page.locator('[data-gx-lab="capacity"]');
    assert.equal(await page.locator('.python-example').count(), 9);
    assert.equal(await page.locator('[data-gx-lab]').count(), 3);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    assert.equal(anchors.length, 10);
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1);
    await page.screenshot({ path: path.join(directory, `reading-entry-${width}.png`) });
    for (let index = 0; index < 5; index++) {
      await capture(page, page.locator('.gx-inline').nth(index), `inline-${index + 1}-${width}.png`);
    }

    const fixtures = [
      { requests: [[0, 6], [1, 3], [3, 5], [5, 7], [6, 9], [7, 9]], objective: 'count' },
      { requests: [[0, 4], [4, 8], [3, 5]], objective: 'count' },
      { requests: [[0, 5, 10], [0, 2, 4], [2, 5, 4]], objective: 'value' },
      { requests: [[0, 2], [0, 2], [2, 4]], objective: 'count' },
      { requests: [], objective: 'count' },
      { requests: [[0, 16]], objective: 'count' },
    ];
    let selectionStates = 0;
    for (const { requests, objective } of fixtures) {
      for (const rule of ['finish', 'start', 'duration']) {
        await intervals.locator('textarea').fill(requests.map(row => row.join(',')).join('\n'));
        await intervals.getByLabel('Candidate rule', { exact: true }).selectOption(rule);
        await intervals.getByRole('button', { name: 'Apply intervals and rule', exact: true }).click();
        await intervals.getByLabel('Score this selection by', { exact: true }).selectOption(objective);
        for (let step = 0; step <= requests.length; step++) {
          const kept = await inspectSelection(intervals, requests, objective, step === requests.length);
          selectionStates++;
          if (step === requests.length && rule === 'finish' && objective === 'count') assert.equal(kept.length, intervalOracle(requests, objective));
          if (step < requests.length) await intervals.getByRole('button', { name: 'Next request', exact: true }).click();
        }
        assert.equal(await intervals.getByRole('button', { name: 'Next request', exact: true }).isDisabled(), true);
        if (requests.length === 3 && (objective === 'value' || rule === 'duration')) {
          await capture(page, intervals, `interval-${objective}-${requests[0][1]}-${rule}-${width}.png`);
        }
      }
    }
    const beforeInvalid = await intervals.locator('.gx-comparison').innerText();
    for (const bad of ['-1,3', '2,2', '0,2,nope', '0,1\n1,2\n2,3\n3,4\n4,5\n5,6\n6,7']) {
      await intervals.locator('textarea').fill(bad);
      await intervals.getByRole('button', { name: 'Apply intervals and rule', exact: true }).click();
      assert.equal(await intervals.getByRole('alert').count(), 1);
      assert.equal(await intervals.locator('.gx-comparison').innerText(), beforeInvalid);
    }
    await intervals.getByRole('button', { name: 'Reset appointments', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await intervals.getByRole('alert').count(), 0);
    await intervals.getByRole('button', { name: 'Shortest-duration trap', exact: true }).click();
    await intervals.getByLabel('Candidate rule', { exact: true }).selectOption('duration');
    await intervals.getByRole('button', { name: 'Apply intervals and rule', exact: true }).click();
    await intervals.getByRole('button', { name: 'Finish selection', exact: true }).click();
    assert.deepEqual(await inspectSelection(intervals, fixtures[1].requests, 'count', true), ['C']);
    await intervals.getByRole('button', { name: 'Weighted-value trap', exact: true }).click();
    await intervals.getByRole('button', { name: 'Finish selection', exact: true }).click();
    assert.deepEqual(await inspectSelection(intervals, fixtures[2].requests, 'value', true), ['B', 'C']);
    await page.keyboard.press('Tab');
    await intervals.locator('.gx-scroll').focus();
    assert.equal(await intervals.locator('.gx-scroll').evaluate(node => getComputedStyle(node).outlineStyle), 'solid');
    if (width === 320) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(200);
      assert.ok(await intervals.locator('.gx-scroll').evaluate(node => node.scrollLeft > 0));
    }

    let scheduleStates = 0;
    for (const jobs of [{ A: [4, 9], B: [3, 5], C: [2, 7] }, { A: [1, 10], B: [1, 10], C: [1, 10] }, { A: [1, 30], B: [8, 0], C: [1, 30] }]) {
      for (const [id, [processing, deadline]] of Object.entries(jobs)) {
        await deadlines.getByLabel(`Job ${id} processing time`, { exact: true }).fill(String(processing));
        await deadlines.getByLabel(`Job ${id} deadline`, { exact: true }).fill(String(deadline));
      }
      await deadlines.getByRole('button', { name: 'Apply jobs', exact: true }).click();
      await inspectDeadline(deadlines, jobs);
      scheduleStates++;
      for (const pair of [0, 1, 0, 1, 0, 1]) {
        const before = await inspectDeadline(deadlines, jobs);
        await deadlines.getByLabel('Adjacent pair', { exact: true }).selectOption(String(pair));
        await deadlines.getByRole('button', { name: 'Exchange adjacent jobs', exact: true }).click();
        const after = await inspectDeadline(deadlines, jobs);
        const expected = [...before.order];
        [expected[pair], expected[pair + 1]] = [expected[pair + 1], expected[pair]];
        assert.deepEqual(after.order, expected);
        if (jobs[before.order[pair]][1] > jobs[before.order[pair + 1]][1]) assert.ok(after.maximum <= before.maximum);
        scheduleStates++;
      }
      await deadlines.getByRole('button', { name: 'Order all by deadline', exact: true }).click();
      const ordered = await inspectDeadline(deadlines, jobs);
      assert.deepEqual(ordered.order, Object.keys(jobs).sort((a, b) => jobs[a][1] - jobs[b][1] || a.localeCompare(b)));
      await capture(page, deadlines, `deadline-${jobs.A[1]}-${width}.png`);
    }
    const previousSchedule = await deadlines.locator('tbody').innerText();
    await deadlines.getByLabel('Job A processing time', { exact: true }).fill('0');
    await deadlines.getByRole('button', { name: 'Apply jobs', exact: true }).click();
    assert.equal(await deadlines.getByRole('alert').count(), 1);
    assert.equal(await deadlines.locator('tbody').innerText(), previousSchedule);
    await deadlines.getByRole('button', { name: 'Reset jobs', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await deadlines.getByRole('alert').count(), 0);

    let capacityStates = 0;
    for (const divisible of [true, false]) {
      await capacityLab.getByRole('checkbox', { name: 'Allow fractional items', exact: true }).setChecked(divisible);
      const slider = capacityLab.getByLabel('Capacity', { exact: true });
      await slider.focus();
      await page.keyboard.press('Home');
      for (let capacity = 0; capacity <= 60; capacity++) {
        assert.equal(await slider.inputValue(), String(capacity));
        const rows = await capacityLab.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.children].map(cell => cell.textContent)));
        let remaining = capacity;
        let total = 0;
        for (const [index, row] of rows.entries()) {
          const weight = [10, 20, 30][index];
          const amount = divisible ? Math.min(remaining, weight) : remaining >= weight ? weight : 0;
          remaining -= amount;
          assert.equal(row[1], `${amount}/${weight}`);
          const fraction = row[2].split('/').map(Number);
          assert.equal(fraction[0] / (fraction[1] || 1), amount / weight);
          const value = amount * [6, 5, 4][index];
          assert.equal(Number(row[3]), value);
          total += value;
        }
        assert.match(await capacityLab.locator('.gx-status').innerText(), new RegExp(`Density-first value: ${total}\\.`));
        assert.equal(await capacityLab.locator('.gx-capacity-track').getAttribute('aria-label'), `Used capacity ${capacity - remaining} of ${capacity}`);
        if (!divisible) assert.match(await capacityLab.locator('.gx-status').innerText(), new RegExp(`whole-item optimum is ${wholeOracle(capacity)},`));
        if ([0, 31, 50].includes(capacity)) await capture(page, capacityLab, `capacity-${capacity}-${divisible ? 'fractional' : 'whole'}-${width}.png`);
        capacityStates++;
        if (capacity < 60) { await slider.focus(); await page.keyboard.press('ArrowRight'); }
      }
    }
    await capacityLab.getByRole('button', { name: 'Reset capacity', exact: true }).click();
    assert.equal(await capacityLab.getByLabel('Capacity', { exact: true }).inputValue(), '50');
    assert.equal(await capacityLab.getByRole('checkbox').isChecked(), true);

    const practice = page.locator('[data-practice-topic="greedy-algorithms-exchange-arguments"]');
    assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 4);
    assert.equal(await practice.locator('details[open]').count(), 0);
    const disclosure = practice.locator('details').first();
    await disclosure.locator('summary').focus();
    await page.keyboard.press('Enter');
    assert.equal(await disclosure.evaluate(node => node.open), true);
    await page.keyboard.press('Space');
    assert.equal(await disclosure.evaluate(node => node.open), false);
    await capture(page, practice.locator('.dsa-practice__stage').first(), `practice-${width}.png`);
    const overflow = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
    assert.ok(overflow.scroll <= width + 1, JSON.stringify(overflow));
    results.push({ width, selectionStates, intervalFixtures: fixtures.length, scheduleStates, capacityStates, anchors: anchors.length, overflow, keyboard: 'resets, local scroll, disclosures and all slider values passed' });
    await page.close();
  }
  assert.deepEqual(errors, []);
  await browser.close();
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ results, errors }, null, 2));
  console.log(JSON.stringify({ results, errors }, null, 2));
})().catch(error => { console.error(error); process.exit(1); });
