const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const topics = [
  ['python-basics-types-control-flow-functions-modules', ['reference-setup']],
  ['object-oriented-programming-in-python', ['bound-method']],
  ['iterators-iterables-generators', ['cursor-ownership']],
  ['decorators-context-managers', ['decorator-binding', 'context-route']],
  ['testing-debugging-dependency-management', ['test-boundary']],
];
(async () => {
  fs.mkdirSync('scratch/python-visual-review', { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [], results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1100 } });
      page.on('pageerror', error => { errors.push(error.message); console.error('Page error:', error.message); });
      for (const [topic, figures] of topics) {
        await page.goto(`http://127.0.0.1:5173/learn/path/full-curriculum/${topic}?module=programming-scientific-computing`);
        try { await page.locator('.reader-article .lesson-intro').waitFor(); }
        catch (error) { console.error('Current URL:', page.url(), 'Page:', (await page.locator('body').innerText()).slice(0, 1800)); throw error; }
        for (const figure of figures) {
          const node = page.locator(`[data-python-figure="${figure}"]`);
          await node.evaluate(n => { let parent = n.parentElement; while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; } });
          await node.scrollIntoViewIfNeeded();
          assert.equal(await node.count(), 1);
          const overflow = await node.evaluate(n => ({ box: n.getBoundingClientRect().right > innerWidth + 1, content: n.scrollWidth > n.clientWidth + 2 }));
          assert.deepEqual(overflow, { box: false, content: false }, `${figure} overflows at ${width}`);
          const clipped = await node.locator('svg text').evaluateAll(labels => labels.filter(label => {
            const b = label.getBBox(), v = label.ownerSVGElement.viewBox.baseVal;
            return b.x < -1 || b.y < -1 || b.x + b.width > v.width + 1 || b.y + b.height > v.height + 1;
          }).map(n => n.textContent));
          assert.deepEqual(clipped, [], `${figure} SVG labels outside viewBox`);
          const overlaps = await node.locator('svg').evaluateAll(svgs => svgs.flatMap(svg => {
            const labels = [...svg.querySelectorAll('text')].filter(n => n.textContent.trim());
            const pairs = [];
            for (let i = 0; i < labels.length; i++) for (let j = i + 1; j < labels.length; j++) {
              const a = labels[i].getBoundingClientRect(), b = labels[j].getBoundingClientRect();
              if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1) pairs.push([labels[i].textContent, labels[j].textContent]);
            }
            return pairs;
          }));
          assert.deepEqual(overlaps, [], `${figure} text labels overlap`);
          for (const svg of await node.locator('svg').all()) assert.ok((await svg.getAttribute('aria-label'))?.length > 30, 'Missing meaningful text equivalent');
          const nav = page.locator('.learn-nav');
          await nav.evaluate(n => n.style.visibility = 'hidden');
          await node.screenshot({ path: `scratch/python-visual-review/${figure}-${width}.png` });
          await nav.evaluate(n => n.style.visibility = '');
        }
        if (topic === 'iterators-iterables-generators') {
          const lab = page.locator('[data-investigation="iterator-ownership"]');
          for (const shared of [false, true]) for (const empty of [false, true]) {
            await lab.locator('select').nth(0).selectOption(String(shared));
            await lab.locator('select').nth(1).selectOption(String(empty));
            const expected = empty ? 'a: END → b: END' : shared ? 'a: 18 → b: 21' : 'a: 18 → b: 18';
            const next = lab.getByRole('button', { name: 'Next step', exact: true });
            await next.focus(); await page.keyboard.press('Enter'); await page.keyboard.press('Enter');
            assert.ok((await lab.locator('.id-output').innerText()).includes(expected));
            const text = await lab.locator('[data-python-figure="cursor-ownership"] svg').getAttribute('aria-label');
            assert.ok(text.includes(shared ? 'share one cursor' : 'independent cursors'));
            assert.equal(await lab.locator('.pmf-cursor-map .pmf-object').count(), shared ? 1 : 2);
            const reset = lab.getByRole('button', { name: 'Reset', exact: true });
            await reset.focus(); await page.keyboard.press('Enter');
            assert.ok(await lab.getByRole('button', { name: 'Back', exact: true }).isDisabled());
          }
          await lab.locator('select').nth(1).selectOption('false');
          await lab.getByRole('button', { name: 'Next step', exact: true }).click();
          await lab.getByRole('button', { name: 'Next step', exact: true }).click();
          await page.locator('.learn-nav').evaluate(n => n.style.visibility = 'hidden');
          try { await lab.screenshot({ path: `scratch/python-visual-review/cursor-shared-active-${width}.png` }); }
          finally { await page.locator('.learn-nav').evaluate(n => n.style.visibility = ''); }
        }
        if (topic === 'decorators-context-managers') {
          const lab = page.locator('[data-investigation="context-lifetime"]');
          for (const path of ['success', 'body-fails', 'enter-fails']) for (const suppress of [false, true]) {
            await lab.locator('select').nth(0).selectOption(path);
            await lab.locator('select').nth(1).selectOption(String(suppress));
            while (!(await lab.getByRole('button', { name: 'Next step', exact: true }).isDisabled())) await lab.getByRole('button', { name: 'Next step', exact: true }).click();
            const route = lab.locator('[data-python-figure="context-route"]');
            const caught = path === 'enter-fails' || path === 'body-fails' && !suppress;
            assert.ok((await route.locator('svg').getAttribute('aria-label')).includes(caught ? 'outer' : 'continue after with'));
            assert.ok((await route.locator('.pmf-resource-state').innerText()).includes(path === 'enter-fails' ? 'not acquired' : 'closed'));
            if (path === 'enter-fails') assert.ok((await route.innerText()).includes('NOT CALLED'));
            await lab.getByRole('button', { name: 'Back', exact: true }).click();
            await lab.getByRole('button', { name: 'Next step', exact: true }).focus(); await page.keyboard.press('Enter');
          }
          await page.locator('.learn-nav').evaluate(n => n.style.visibility = 'hidden');
          try { await lab.screenshot({ path: `scratch/python-visual-review/context-entry-failure-${width}.png` }); }
          finally { await page.locator('.learn-nav').evaluate(n => n.style.visibility = ''); }
        }
        results.push({ topic, width, figures, layout: 'pass', diagramsAccessible: true });
      }
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync('scratch/python-visual-review/browser-results.json', JSON.stringify({ results, errors }, null, 2));
    console.log(`Python visual review passed: ${results.length} topic/viewport checks; 6 visual forms; cursor/context presets, keyboard, bounds and text equivalents. No page errors.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
