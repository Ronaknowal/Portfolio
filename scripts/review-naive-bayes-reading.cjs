const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/naive-bayes-browser');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge' });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 } });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/naive-bayes-probabilistic-classifiers', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.naive-bayes-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      const metrics = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, i) => {
        const r = node.getBoundingClientRect();
        const spans = [...node.querySelectorAll('.katex-html span')].map(n => n.getBoundingClientRect()).filter(s => s.width > 0);
        return { i, left: r.left, right: r.right, actualLeft: Math.min(...spans.map(s => s.left)), actualRight: Math.max(...spans.map(s => s.right)) };
      }));
      fs.writeFileSync(path.join(directory, 'math-actual-bounds-' + width + '.json'), JSON.stringify(metrics, null, 2));
      assert(metrics.every(m => m.actualLeft >= m.left - 1 && m.actualRight <= m.right + 1), JSON.stringify(metrics.filter(m => m.actualLeft < m.left - 1 || m.actualRight > m.right + 1)));
      const capture = async (node, name) => {
        await node.first().evaluate(n => scrollTo({ top: n.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.waitForTimeout(120);
        await page.screenshot({ path: path.join(directory, 'final-' + name + '-' + width + '.png') });
      };
      await capture(lesson.locator('.lesson-intro'), 'intro');
      for (const i of [2, 9, 10, 16]) await capture(lesson.locator('.katex-display').nth(i).locator('..').locator('..'), 'equation-' + i);
      await capture(lesson.getByText('Derive the Gaussian estimates and identify the degenerate case', { exact: true }), 'gaussian-proof');
      await capture(lesson.locator('section.lesson-check').filter({ hasText: 'Diagnose a rare-class remedy' }).locator('details').last(), 'rare-class-answer');
      await capture(lesson.locator('section.lesson-check').last().locator('details').last().locator('p').last(), 'report-consequence');
      const copies = lesson.locator('[data-investigation="copied-alarm"]');
      const alarmBounds = await copies.locator('svg').evaluate(svg => {
        const box = svg.querySelector('rect').getBBox();
        const text = svg.querySelector('text').getBBox();
        return { boxLeft: box.x, boxRight: box.x + box.width, textLeft: text.x, textRight: text.x + text.width };
      });
      assert(alarmBounds.textLeft > alarmBounds.boxLeft + 5 && alarmBounds.textRight < alarmBounds.boxRight - 5);
      await capture(copies.locator('figure'), 'copied-node');
      const fonts = await page.evaluate(() => [...document.fonts].filter(f => ['Space Grotesk', 'JetBrains Mono'].includes(f.family.replaceAll('"', ''))).map(f => ({ family: f.family, status: f.status })));
      assert(fonts.some(f => f.family.includes('Space') && f.status === 'loaded'));
      assert(fonts.some(f => f.family.includes('JetBrains') && f.status === 'loaded'));
      assert.deepEqual(errors, []);
      records.push({ width, metrics, fonts, errors, alarmBounds });
      await page.close();
    }
    fs.writeFileSync(path.join(directory, 'final-reading-results.json'), JSON.stringify({ completedAt: new Date().toISOString(), records }, null, 2));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
