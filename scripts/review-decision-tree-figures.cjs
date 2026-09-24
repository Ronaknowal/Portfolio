const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/decision-trees-random-forests?module=classical-ml', { waitUntil: 'domcontentloaded' });
      await page.locator('.tree-lesson').waitFor();
      await page.evaluate(() => document.fonts.ready);
      const partition = page.locator('[data-investigation="tree-partitions"]');
      await partition.getByLabel('Maximum depth').selectOption('3');
      await partition.getByLabel('Query x1', { exact: false }).fill('4.5');
      const ids = ['tree-partitions'];
      if (width === 390) {
        await page.getByLabel('Zero-gain policy').selectOption('true');
        ids.push('tree-xor');
      }
      if (width === 320) ids.push('tree-pruning', 'tree-variance');
      for (const id of ids) {
        const figure = page.locator(`[data-investigation="${id}"] .tree-figure`);
        const svg = figure.locator('svg');
        const labels = await svg.evaluate(node => [...node.querySelectorAll('text')].map(label => {
          const bounds = label.getBoundingClientRect();
          const owner = node.getBoundingClientRect();
          return { text: label.textContent, font: getComputedStyle(label).font, inside: bounds.left >= owner.left - 1 && bounds.right <= owner.right + 1 && bounds.top >= owner.top - 1 && bounds.bottom <= owner.bottom + 1 };
        }));
        assert(labels.every(label => label.inside && label.font.includes('13px')));
        const screenshot = `scratch/decision-tree-browser/${id}-final-figure-${width}.png`;
        await figure.screenshot({ path: screenshot });
        records.push({ width, id, labels, screenshot });
      }
      await page.close();
    }
  } finally { await browser.close(); }
  const paths = ['src/learn/components/lesson-labs/DecisionTreeLabs.jsx', 'src/learn/components/lesson-labs/decision-tree-labs.css'];
  const sources = paths.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
  fs.writeFileSync('docs/teaching/evidence/decision-tree-figure-amendment.json', JSON.stringify({ checkedAt: new Date().toISOString(), records, sources, change: 'Scoped chart typography overrides an older global mobile rule; y-axis clearance restored. Reuses unchanged interaction/native evidence.' }, null, 2) + '\n');
  console.log('Tree figures: corrected actual fonts and label extents passed at all three widths.');
})().catch(error => { console.error(error); process.exitCode = 1; });
