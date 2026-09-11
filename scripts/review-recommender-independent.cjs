const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/recommender-independent';

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('WebSocket')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/recommender-systems-collaborative-filtering-matrix-factorization', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.rec-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const lab = name => lesson.locator('.rec-investigation').filter({ has: page.getByRole('heading', { name, exact: true }) });
      const evidence = lab('Read a missing cell correctly');
      await evidence.getByRole('combobox', { name: 'Objective', exact: true }).selectOption('implicit');
      assert((await evidence.innerText()).includes('loss 0.25'));
      await evidence.getByRole('button', { name: 'User 0, item 0: rating 5; count 10', exact: true }).click();
      assert((await evidence.innerText()).includes('loss 5.25'));
      const bowl = lab('See the all-pair confidence bowl');
      await bowl.getByRole('slider', { name: /Confidence multiplier/ }).fill('0');
      await bowl.getByRole('slider', { name: /Once-per-user penalty/ }).fill('2');
      const checkbox = bowl.getByRole('checkbox');
      await checkbox.focus();
      await page.keyboard.press('Space');
      assert(!(await checkbox.isChecked()));
      assert((await bowl.innerText()).includes('Minimum at p = (0.455, 0.182)'));
      const geometry = await bowl.locator('polyline').evaluateAll(lines => lines.map(line => [...line.points].every(p => p.x >= 47 && p.x <= 257 && p.y >= 40 && p.y <= 230)));
      assert(geometry.every(Boolean));
      const slate = lab('Move the slate and watch the ranking evidence');
      await slate.getByRole('combobox').selectOption('graded');
      await slate.getByRole('checkbox').check();
      await slate.getByRole('slider').fill('5');
      await slate.getByRole('button', { name: 'Move item 3 up', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.deepEqual(await slate.locator('.rec-item-title').evaluateAll(nodes => nodes.map(node => node.innerText.split('\n')[0].trim())), ['Item 2', 'Item 0', 'Item 3', 'Item 4']);
      assert((await slate.innerText()).includes('filled slots: 4/5'));
      const metric = name => slate.locator('dl > div').filter({ has: page.locator('dt', { hasText: name }) }).locator('dd');
      assert.equal(await metric(/^Precision@K$/).innerText(), '0.4');
      assert.equal(await metric(/^Recall@K$/).innerText(), '0.667');
      const policy = lab("Separate exposure from a policy's expected reward");
      await policy.getByRole('slider', { name: /Logging probability of A/ }).fill('1');
      assert((await policy.innerText()).includes('Support fails'));
      await policy.getByRole('slider', { name: /Target probability of A/ }).fill('1');
      assert((await policy.innerText()).includes('Support holds'));
      assert((await policy.innerText()).includes('one-request IPS variance is 0.24'));
      const scopes = { labs: await lesson.locator('.rec-investigation').count(), programs: await lesson.locator('.python-example').count(), anchors: await lesson.locator('nav a').evaluateAll(nodes => nodes.map(n => n.hash).filter(h => !document.getElementById(h.slice(1)))) };
      assert.equal(scopes.programs, 14); assert.equal(scopes.labs, 8); assert.deepEqual(scopes.anchors, []);
      const fit = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth, equations: [...document.querySelectorAll('.katex-display')].map(node => ({ client: node.clientWidth, scroll: node.scrollWidth })) }));
      assert(fit.scroll <= width + 1); assert(fit.equations.every(item => item.scroll <= item.client + 1));
      const target = width === 390 ? slate : bowl;
      await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 85, behavior: 'instant' }));
      await page.screenshot({ path: `${directory}/changed-${width}.png` });
      assert.deepEqual(errors, []);
      records.push({ width, scope: 'Missing versus observed implicit term; changed Gram system and keyboard removal; graded shortened slate and keyboard reorder; support-failing and supported logger endpoint.', font: 'Space Grotesk loaded', errors, fit, geometry, scopes, image: `${directory}/changed-${width}.png` });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), scope: 'Complementary operated states; author full suite reused. Neighbor example display amendment is separately verified by its author.', records }, null, 2) + '\n');
  console.log('Passed complementary recommender interactions at 1440, 390, 320.');
})().catch(error => { console.error(error); process.exitCode = 1; });
