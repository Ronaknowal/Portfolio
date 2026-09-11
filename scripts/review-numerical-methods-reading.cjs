const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/numerical-methods-browser');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-methods-finite-differences-quadrature-root-finding?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.numerical-methods-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const captures = [], geometry = [];
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(150);
        const filename = `final-${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, filename) }); captures.push(filename);
      }
      assert.equal(await lesson.locator('.katex-display').count(), 8);
      for (const [index, equation] of (await lesson.locator('.katex-display').all()).entries()) {
        const size = await equation.evaluate(node => ({ width: node.clientWidth, scroll: node.scrollWidth }));
        assert(size.scroll <= size.width + 1, `equation ${index} at ${width}: ${JSON.stringify(size)}`);
        await shot(equation, `equation-${index}`);
        geometry.push(size);
      }
      for (const [index, plot] of (await lesson.locator('svg.nm-plot').all()).entries()) await shot(plot, `plot-${index}`);
      const newton = lesson.locator('[data-investigation="numerical-newton"]');
      const tangent = newton.locator('svg > path[clip-path]').nth(1);
      const tangentCoordinates = (await tangent.getAttribute('d')).match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi).map(Number);
      // A true line is clipped by the SVG rectangle, not flattened onto its edge.
      const slopes = [];
      for (let index = 2; index < tangentCoordinates.length; index += 2) slopes.push((tangentCoordinates[index + 1] - tangentCoordinates[index - 1]) / (tangentCoordinates[index] - tangentCoordinates[index - 2]));
      assert(slopes.every(value => Math.abs(value - slopes[0]) < 1e-10));
      const difference = lesson.locator('[data-investigation="numerical-difference"]');
      await difference.getByLabel('Function offset', { exact: true }).selectOption('100000000');
      const slider = difference.getByRole('slider', { name: 'Negative power of ten', exact: true });
      await slider.focus(); await page.keyboard.press('End'); await page.keyboard.press('ArrowLeft');
      await shot(difference.locator('svg').first(), 'lost-derivative-geometry');
      await shot(difference.locator('svg').nth(1), 'actual-error-plot');
      await shot(difference.locator('.nm-readout'), 'difference-readout');
      const adaptive = lesson.locator('[data-investigation="numerical-adaptive"]');
      await adaptive.getByLabel('Adaptive function', { exact: true }).selectOption('blind');
      assert((await adaptive.innerText()).includes('vertical coordinates are multiplied by 10⁵'));
      assert((await adaptive.locator('svg').textContent()).includes('1.65'));
      assert.equal(await adaptive.locator('svg circle').count(), 5);
      for (const circle of await adaptive.locator('svg circle').all()) assert.equal(Number(await circle.getAttribute('cy')), 166);
      await shot(adaptive.locator('svg'), 'blind-scale');
      await shot(adaptive.locator('.nm-readout'), 'blind-readout');
      const first = lesson.locator('.python-example').first();
      await shot(first, 'original-program'); await shot(first.locator('.lesson-note'), 'original-output');
      for (const [index, item] of (await lesson.locator('.nm-practice').all()).entries()) {
        await item.locator(':scope > details > summary').nth(1).click();
        if ([0, 5, 8].includes(index)) await shot(item, `practice-${index}`);
      }
      await shot(lesson.locator('.lesson-sources'), 'sources');
      for (const svg of await lesson.locator('svg.nm-plot').all()) {
        const clipped = await svg.evaluate(node => {
          const box = node.getBoundingClientRect();
          return [...node.querySelectorAll('text')].filter(text => { const rect = text.getBoundingClientRect(); return rect.left < box.left - 2 || rect.right > box.right + 2; }).map(text => text.textContent);
        });
        assert.deepEqual(clipped, [], `plot labels at ${width}`);
      }
      const external = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, target: node.target, rel: node.rel })));
      assert(external.every(link => link.href.startsWith('https:') && link.target === '_blank' && link.rel.includes('noreferrer')));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
      records.push({ width, captures, geometry, straightTangentClip: true, blindSamplesScaledCorrectly: true, external, fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) });
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'final-reading-results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, errors }, null, 2));
    console.log('Final equation/plot/ordinary reading, numerical geometry and source links passed at 1440/390/320.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
