const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest = read('dist/.vite/manifest.json');
const lessons = read('src/learn/data/lesson-manifest.json');
const ids = read('docs/teaching/evidence/deep-learning-core-baseline.json').ids;
const next = [...ids.slice(1), 'weight-initialization-xavier-kaiming-p'];
const publishedChunks = Object.entries(lessons).map(([id, path]) => ({ id, file: manifest[`src/learn/data/${path.slice(2)}`].file }));
const receipt = 'docs/teaching/evidence/deep-learning-core-production-integration.json';
const report = { status: 'running', checkedAt: new Date().toISOString(), base, manifestHash: hash('dist/.vite/manifest.json'), checks: [], routes: [] };
const save = () => fs.writeFileSync(receipt, JSON.stringify(report, null, 2) + '\n');
save();
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
    for (const [i, id] of ids.entries()) {
      const context = await browser.newContext({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
      await context.route('https://fonts.googleapis.com/**', r => r.fulfill({ path: fonts.stylesheet, contentType: 'text/css' }));
      await context.route('https://fonts.gstatic.com/**', r => fonts.files[r.request().url()] ? r.fulfill({ path: fonts.files[r.request().url()], contentType: 'font/ttf' }) : r.abort());
      const page = await context.newPage(), requests = new Set(), errors = [];
      page.on('request', request => requests.add(new URL(request.url()).pathname.slice(1)));
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/${id}?module=deep-learning-fundamentals`);
      await page.locator('.reader-article[aria-busy=false] .lesson-intro, .reader-article[aria-busy=false] h2').first().waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert.equal(await page.locator('.lesson-load-error,.katex-error').count(), 0);
      const loaded = publishedChunks.filter(chunk => requests.has(chunk.file)).map(chunk => chunk.id);
      assert.deepEqual(loaded, [id], `${id}: unrelated published lesson loaded`);
      const links = await page.locator('.reader-article a[href^="#"]').evaluateAll(nodes => nodes.map(node => ({ href: node.getAttribute('href'), exists: !!document.getElementById(decodeURIComponent(node.hash.slice(1))) })));
      assert.ok(links.every(link => link.exists), JSON.stringify(links.filter(link => !link.exists)));
      const math = await page.locator('.katex .sqrt svg,.katex .accent svg').evaluateAll(nodes => nodes.map(node => ({ width: node.getBoundingClientRect().width, height: node.getBoundingClientRect().height })));
      assert.ok(math.every(box => box.width > 1 && box.height > 1), `${id}: collapsed math geometry`);
      const javascriptBytes = [...requests].filter(path => path.endsWith('.js') && fs.existsSync(`dist/${path}`)).reduce((sum, path) => sum + fs.statSync(`dist/${path}`).size, 0);
      const linkColors = await page.locator('.reader-article a[href]').evaluateAll(nodes => [...new Set(nodes.map(node => getComputedStyle(node).color))]);
      assert.ok(linkColors.every(color => { const rgb = color.match(/[\d.]+/g).map(Number); return rgb[0] >= rgb[1] && rgb[1] > rgb[2]; }), `${id}: non-theme link ${linkColors}`);
      if (i === 2 || i === 3) {
        const stem = i === 2 ? 'loss-functions' : 'normalization';
        const filename = i === 2 ? 'loss-experiments.py' : 'normalization-experiments.py';
        const programChunk = manifest[`src/learn/data/${stem}-program.js`].file;
        assert.ok(!requests.has(programChunk), 'Closed program must not load source');
        assert.equal(await page.locator('.neural-program > div').count(), 0);
        await page.getByText('Read the complete CPU program', { exact: true }).click();
        await page.locator('.neural-program > div').waitFor();
        assert.ok(requests.has(programChunk));
        const expected = fs.readFileSync(`public/learn-assets/${id}/${filename}`, 'utf8');
        const shown = await page.locator('.neural-program > div').evaluate(node => [...node.childNodes].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent).join(''));
        assert.equal(shown.trim().replace(/\r\n/g, '\n'), expected.trim().replace(/\r\n/g, '\n'));
        await page.getByText('Read the complete CPU program', { exact: true }).click();
        await page.locator('.neural-program > div').waitFor({ state: 'detached' });
        assert.equal(await page.locator('.neural-program > div').count(), 0);
        for (const file of [filename, 'digits-400.csv', 'data-provenance.md']) {
          const response = await page.request.get(`${base}/learn-assets/${id}/${file}`);
          assert.equal(response.status(), 200);
          assert.equal(createHash('sha256').update(await response.body()).digest('hex'), hash(`public/learn-assets/${id}/${file}`));
        }
      }
      await page.locator('.reader-footer__next').click();
      await page.waitForURL(`**/${next[i]}?module=deep-learning-fundamentals`);
      assert.deepEqual(errors, []);
      report.routes.push({ id, loadedLessonBodies: loaded, sectionLinks: links.length, mathSvgCount: math.length, next: next[i], initialJavascriptBytes: javascriptBytes, linkColors });
      await context.close();
    }
    report.checks.push('Five fresh-context routes request only their own published body', 'All lesson section anchors exist', 'Rendered radical/accent geometry remains nonzero', 'Every next action follows the exact module sequence', 'Themed lesson links', 'Loss/Normalization programs fetch and mount only on demand, preserve complete source and unmount on close', 'Exact HTTP program/data/provenance downloads', 'No runtime page errors');
    report.status = 'passed';
  } catch (error) { report.status = 'failed'; report.failure = error.stack; process.exitCode = 1; }
  finally { await browser.close(); report.verifierHash = hash(__filename); save(); console.log(JSON.stringify(report)); }
})();
