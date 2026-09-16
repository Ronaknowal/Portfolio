// Read-only triage of currently rendered lesson SVGs. Not a substitute for
// screenshots, numerical validation or reviewing informative interactive states.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

async function main() {
  const args = process.argv.slice(2);
  const value = flag => args.includes(flag) ? args[args.indexOf(flag) + 1] : undefined;
  const manifest = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
  const topics = args.includes('--all-published') ? Object.keys(manifest) : (value('--topics') || '').split(',').filter(Boolean);
  if (!topics.length || topics.some(id => !manifest[id])) throw new Error('Supply --all-published or --topics <published-id,...>');
  const widths = (value('--widths') || '1366,390,320').split(',').map(Number);
  if (widths.some(width => !Number.isInteger(width) || width < 240 || width > 2560)) throw new Error('Unsupported viewport width');
  const output = value('--output') || 'docs/teaching/evidence/lesson-visual-layout-audit.json';
  const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
  const componentFiles = directory => fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const filename = path.posix.join(directory, entry.name);
    return entry.isDirectory() ? componentFiles(filename) : /\.(jsx?|css)$/.test(entry.name) ? [filename] : [];
  });
  const sourceFiles = [...new Set([
    ...topics.map(id => 'src/learn/data/' + manifest[id].replace(/^\.\//, '')),
    ...componentFiles('src/learn/components'),
  ])];
  const sourceHashes = Object.fromEntries(sourceFiles.map(filename => [filename, hash(filename)]));
  const buildManifestHash = hash('dist/.vite/manifest.json');
  const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  let nextIndex = 0;
  const startedAt = new Date().toISOString();
  try {
    await Promise.all(Array.from({ length: Math.min(3, topics.length) }, async () => {
      const page = await browser.newPage({ viewport: { width: widths[0], height: 1000 } });
      page.setDefaultTimeout(15000);
      while (nextIndex < topics.length) {
        const id = topics[nextIndex++];
        try {
          await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173'}/learn/path/full-curriculum/${id}`, { waitUntil: 'domcontentloaded' });
          await page.locator('.reader-article').last().waitFor();
          await page.evaluate(() => document.fonts.ready);
          for (const width of widths) {
            await page.setViewportSize({ width, height: 1000 });
            await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            const figures = await page.evaluate(inspectLessonVisualLayout);
            records.push({ id, width, figures });
          }
        } catch (error) { records.push({ id, error: error.message }); }
        if (nextIndex % 25 === 0) console.log(`Inspected ${Math.min(nextIndex, topics.length)} / ${topics.length} lesson routes`);
      }
      await page.close();
    }));
  } finally { await browser.close(); }
  for (const [filename, digest] of Object.entries(sourceHashes)) assert.equal(hash(filename), digest, `Source changed during audit: ${filename}`);
  assert.equal(hash('dist/.vite/manifest.json'), buildManifestHash, 'Build changed during audit');
  const report = {
    startedAt, completedAt: new Date().toISOString(), topicCount: topics.length, widths,
    stage: 'Geometry triage; findings are candidates for source/rendered review, not confirmed defects or a whole-site visual approval.',
    buildManifestHash,
    verifierHash: hash(__filename), inspectorHash: hash('scripts/lib/lesson-visual-layout.cjs'),
    sourceHashes,
    records,
  };
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ output, topics: topics.length, errors: records.filter(r => r.error).length, flaggedFigures: records.reduce((n, r) => n + (r.figures || []).filter(f => f.issues.length).length, 0) }));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
