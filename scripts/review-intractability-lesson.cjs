const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, '../scratch/intractability-lesson-review');
fs.mkdirSync(output, { recursive: true });

async function capture(page, target, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = 'hidden'; }));
  await target.screenshot({ path: path.join(output, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = ''; }));
}
async function click(lab, label) {
  await lab.getByRole('button', { name: label, exact: true }).click();
}
async function openDetail(lab, label) {
  const summary = lab.locator('summary').filter({ hasText: label }).first();
  const details = summary.locator('..');
  if (!(await details.evaluate(element => element.open))) await summary.click();
  return details;
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/reductions-p-np-computational-intractability?module=data-structures-algorithms');
    const certificate = page.locator('[data-lab="certificate"]');
    await certificate.waitFor({ timeout: 60000 });
    const reduction = page.locator('[data-lab="reduction"]');
    const cover = page.locator('[data-lab="cover"]');
    const encoding = page.locator('[data-lab="encoding"]');
    assert.equal(await page.locator('.python-example').count(), 9);
    await page.screenshot({ path: path.join(output, `reading-entry-${width}.png`) });
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const id of anchors) {
      assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      await page.locator(`.lesson-intro a[href="#${id}"]`).click();
      await page.waitForFunction(anchor => { const top = document.getElementById(anchor).getBoundingClientRect().top; return top >= -1 && top < innerHeight; }, id);
    }
    await capture(page, page.locator('.intract-oracle-boundary').locator('..'), `reduction-flow-${width}.png`);
    await capture(page, page.locator('.intract-path').locator('..'), `path-restriction-${width}.png`);
    assert.match(await certificate.locator('.intract-result').innerText(), /Rejected/);
    const x3 = certificate.getByRole('button', { name: 'Toggle x3', exact: true });
    await x3.focus();
    await page.keyboard.press('Enter');
    assert.match(await certificate.locator('.intract-result').innerText(), /Accepted/);
    await capture(page, certificate, `certificate-accepted-${width}.png`);
    await page.keyboard.press('Space');
    assert.match(await certificate.locator('.intract-result').innerText(), /Rejected/);
    const completeTable = await openDetail(certificate, 'Compare all eight');
    assert.equal(await completeTable.locator('tbody tr').count(), 8);
    await openDetail(certificate, 'Change the formula');
    await certificate.getByLabel('Clause draft', { exact: true }).fill('4');
    await click(certificate, 'Apply formula');
    assert.match(await certificate.getByRole('alert').innerText(), /Applied formula is unchanged/);
    await click(certificate, 'Contradictory clauses');
    for (let state = 0; state < 8; state += 1) {
      for (let index = 0; index < 3; index += 1) {
        const control = certificate.getByRole('button', { name: `Toggle x${index + 1}`, exact: true });
        if ((await control.getAttribute('aria-pressed')) !== String(Boolean(state & (1 << index)))) await control.click();
      }
      assert.match(await certificate.locator('.intract-result').innerText(), /Rejected/);
    }
    assert.equal(await completeTable.locator('tbody tr').filter({ hasText: 'Yes' }).count(), 0);
    await capture(page, certificate, `certificate-contradiction-${width}.png`);
    await click(certificate, 'Default formula');

    const clause1 = reduction.getByRole('group', { name: 'Clause 1', exact: true });
    const clause2 = reduction.getByRole('group', { name: 'Clause 2', exact: true });
    const clause3 = reduction.getByRole('group', { name: 'Clause 3', exact: true });
    await clause1.getByRole('button').first().click();
    await clause2.getByRole('button').first().click();
    await clause3.getByRole('button').first().click();
    assert.match(await reduction.locator('.intract-result').innerText(), /not a clique/);
    assert.equal(await reduction.locator('.intract-conflict').count(), 2);
    await capture(page, reduction, `reduction-conflict-${width}.png`);
    await clause2.getByRole('button').nth(1).click();
    assert.match(await reduction.locator('.intract-result').innerText(), /Valid clique.*T, T, F/);
    await capture(page, reduction, `reduction-witness-${width}.png`);
    await openDetail(reduction, 'Change the formula');
    await click(reduction, 'Contradictory clauses');
    await click(reduction, 'Search this tiny graph');
    assert.match(await reduction.getByRole('status').innerText(), /no target clique/);
    await click(reduction, 'Repeated literals');
    assert.equal(await reduction.locator('svg rect').count(), 9);
    await click(reduction, 'Search this tiny graph');
    assert.match(await reduction.locator('.intract-result').innerText(), /Valid clique/);
    await reduction.getByLabel('Clause draft', { exact: true }).fill('1\n1');
    await click(reduction, 'Apply formula');
    await click(reduction, 'Search this tiny graph');
    assert.match(await reduction.locator('.intract-result').innerText(), /Target: 2.*Valid clique/);
    assert.equal(await reduction.locator('svg rect').count(), 2);
    await click(reduction, 'Default formula');
    await click(reduction, 'Search this tiny graph');

    await encoding.getByLabel('Target exponent', { exact: true }).selectOption('40');
    assert.match(await encoding.innerText(), /41 binary digits/);
    assert.match(await encoding.innerText(), /1,099,511,627,777 slots/);
    await capture(page, encoding, `encoding-large-${width}.png`);

    assert.match(await cover.locator('.intract-result').innerText(), /6 edges remain/);
    await click(cover, 'Use matching endpoints');
    assert.match(await cover.locator('.intract-result').innerText(), /Every edge is covered/);
    assert.match(await cover.locator('.intract-bounds').innerText(), /3 ≤ OPT/);
    assert.match(await cover.locator('.intract-bounds').innerText(), /OPT ≤ 6/);
    await capture(page, cover, `cover-bound-${width}.png`);
    await cover.getByLabel('Cover budget', { exact: true }).selectOption('3');
    assert.match(await cover.locator('.intract-budget').innerText(), /A cover within budget/);
    await click(cover, 'Use exact budget answer');
    assert.match(await cover.locator('.intract-result').innerText(), /\(3\).*Every edge/);
    await openDetail(cover, 'See the finite branch tree');
    await openDetail(cover, 'Reveal this tiny');
    assert.match(await cover.locator('.intract-budget').innerText(), /Minimum cover size 3/);
    await capture(page, cover, `cover-exact-${width}.png`);
    await openDetail(cover, 'Change the graph');
    await cover.getByLabel('Cover edge draft', { exact: true }).fill('A A');
    await click(cover, 'Apply graph');
    assert.match(await cover.getByRole('alert').innerText(), /excludes self-loops/);
    assert.match(await cover.locator('.intract-result').innerText(), /\(3\).*Every edge/);
    await click(cover, 'Star graph');
    await click(cover, 'Use matching endpoints');
    assert.match(await cover.locator('.intract-bounds').innerText(), /1 ≤ OPT/);
    await cover.getByLabel('Cover budget', { exact: true }).selectOption('1');
    await click(cover, 'Use exact budget answer');
    assert.match(await cover.locator('.intract-result').innerText(), /Selected: A \(1\).*Every edge/);
    await click(cover, 'Empty graph');
    await cover.getByLabel('Cover budget', { exact: true }).selectOption('0');
    await click(cover, 'Use exact budget answer');
    assert.match(await cover.locator('.intract-result').innerText(), /Selected: ∅ \(0\).*Every edge/);
    await cover.getByLabel('Cover edge draft', { exact: true }).fill('A B\nB A\nB C');
    await click(cover, 'Apply graph');
    await click(cover, 'Clear selected vertices');
    const vertexB = cover.getByRole('button', { name: 'Toggle vertex B', exact: true });
    await vertexB.focus();
    await page.keyboard.press('Space');
    assert.match(await cover.locator('.intract-result').innerText(), /Selected: B \(1\).*Every edge/);

    const exercise = page.locator('summary').filter({ hasText: 'Exercise 2 · break' });
    await exercise.focus();
    await page.keyboard.press('Enter');
    const parent = exercise.locator('..');
    assert.equal(await parent.evaluate(element => element.open), true);
    await parent.locator('summary').filter({ hasText: 'Hint' }).focus();
    await page.keyboard.press('Space');
    assert.equal(await parent.locator('details').first().evaluate(element => element.open), true);
    const leetcode = page.locator('a[href^="https://leetcode.com/problems/"]');
    assert.equal(await leetcode.count(), 3);
    for (const link of await leetcode.all()) {
      assert.equal(await link.getAttribute('target'), '_blank');
      assert.match(await link.getAttribute('rel'), /noopener/);
    }
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false, `Page overflow ${width}`);
    assert.deepEqual(errors, []);
    results.push({ width, anchors: anchors.length, nativePrograms: 9, officialPracticeLinks: 3, contradictionAssignments: 8, keyboard: true, overflow, errors });
    await page.close();
  }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(results, null, 2));
  await browser.close();
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exit(1); });
