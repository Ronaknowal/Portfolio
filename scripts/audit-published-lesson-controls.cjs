// Bounded UI triage of published lessons outside the completed teaching rollout.
// Findings require rendered/source review; this does not certify lesson content.
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest = read('src/learn/data/lesson-manifest.json');
const reviewed = new Set(read('docs/teaching/evidence/live-exploration-runtime-scope.json').topics.map(t => t.id));
const requested = process.argv.find(arg => arg.startsWith('--topics='))?.slice(9).split(',');
const topics = requested || Object.keys(manifest).filter(id => !reviewed.has(id));
const output = process.env.LEARNING_AUDIT_OUTPUT || 'docs/teaching/evidence/published-lesson-control-triage.json';
const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const report = { startedAt: new Date().toISOString(), scope: topics, manifestHash: hash('dist/.vite/manifest.json'), sourceHashes: Object.fromEntries(['src/learn/components/BPETrainer.jsx', 'src/learn/data/bpe-trainer-model.js', 'src/learn/components/viz/TokenStream.jsx', 'src/learn/components/viz/StepTrace.jsx', 'src/learn/components/topic-content.css'].map(file => [file, hash(file)])), records: [], errors: [], limitations: 'Initial visible state at desktop/phone; controls and labels are triage candidates. This is not an exhaustive interaction, accessibility or teaching-quality certification.' };
function inspect() {
  const visible = node => node.checkVisibility();
  const name = node => node.getAttribute('aria-label') || (node.getAttribute('aria-labelledby') || '').split(/\s+/).map(id => document.getElementById(id)?.textContent || '').join(' ').trim() || [...(node.labels || [])].map(label => label.textContent).join(' ').trim() || (node.tagName === 'BUTTON' ? node.textContent.trim() : '') || node.getAttribute('title');
  const controls = [...document.querySelectorAll('.reader-article button,.reader-article input,.reader-article select,.reader-article textarea,[role="slider"]')].filter(visible).map(node => {
    const style = getComputedStyle(node), rect = node.getBoundingClientRect(), label = name(node);
    const issues = [];
    if (!label) issues.push('missing accessible label');
    if (node.tagName === 'BUTTON' && /^rgb\((\d+), \1, \1\)$/.test(style.backgroundColor) && Number(style.backgroundColor.match(/\d+/)[0]) > 65) issues.push('native-like gray background');
    if (rect.width < 24 || rect.height < 24) issues.push('target below 24px');
    if (rect.left < -1 || rect.right > innerWidth + 1) issues.push('outside viewport: check local scrolling');
    if ((node.type === 'range' || node.type === 'checkbox' || node.type === 'radio') && style.accentColor === 'auto') issues.push('native accent');
    return { tag: node.tagName, type: node.type || node.getAttribute('role'), name: label?.slice(0, 120), class: node.className?.baseVal ?? node.className, parent: node.parentElement.className?.baseVal ?? node.parentElement.className, issues, size: [rect.width, rect.height], background: style.backgroundColor };
  });
  const overflow = [...document.querySelectorAll('.reader-article *')].filter(visible).filter(node => {
    if (node.closest('svg,.katex,.katex-display,pre,table')) return false;
    if (getComputedStyle(node).position === 'absolute' || getComputedStyle(node).position === 'fixed') return false;
    const box = node.getBoundingClientRect();
    if (box.right <= innerWidth + 2 && box.left >= -2) return false;
    for (let p = node.parentElement; p && p !== document.body; p = p.parentElement) {
      if (['auto','scroll'].includes(getComputedStyle(p).overflowX)) return false;
    }
    return true;
  }).slice(0, 8).map(node => ({tag: node.tagName, class: String(node.className), text: node.textContent.slice(0, 100)}));
  return { controls: controls.length, candidates: controls.filter(c => c.issues.length), overflow, articleError: !!document.querySelector('.lesson-load-error'), ranges: controls.filter(c => c.type === 'range' || c.type === 'slider').length };
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
    await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.abort());
    let index = 0;
    await Promise.all(Array.from({ length: 2 }, async () => {
      const page = await context.newPage();
      while (index < topics.length) {
        const id = topics[index++], errors = [];
        const listener = error => errors.push(error.message);
        page.on('pageerror', listener);
        try {
          await page.goto(`${base}/learn/path/full-curriculum/${id}`, { waitUntil: 'domcontentloaded' });
          await page.locator('.reader-article h2,.reader-article h3,.lesson-load-error').first().waitFor();
          await page.evaluate(() => document.fonts.ready);
          for (const width of [1366, 390, 320]) {
            await page.setViewportSize({ width, height: 1000 });
            await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            report.records.push({ id, width, ...await page.evaluate(inspect) });
          }
          if (errors.length) report.errors.push({ id, errors });
        } catch (error) { report.errors.push({ id, error: error.message }); }
        page.off('pageerror', listener);
      }
      await page.close();
    }));
    report.completedAt = new Date().toISOString();
    report.verifierHash = hash(__filename);
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ topics: topics.length, records: report.records.length, errors: report.errors.length, flagged: report.records.filter(r => r.candidates.length || r.overflow.length).length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
