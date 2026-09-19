// Production integration for the five prepared lessons at Classical ML 29–33.
// Reuses source-bound topic reviews rather than rerunning earlier module suites.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4188';
const read = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const baseline = read('docs/teaching/evidence/classical-query-evaluation-baseline.json');
const publication = read('src/learn/data/lesson-manifest.json');
const manifest = read('dist/.vite/manifest.json');
const initialBuildHash = hash('dist/.vite/manifest.json');
const scope = baseline.scope;
const source = id => `src/learn/data/${publication[id].replace(/^\.\//, '')}`;
const body = id => manifest[source(id)].file;
const bodies = new Set(Object.keys(publication).map(body));
const outlines = new Set(Object.entries(manifest).filter(([key]) => key.includes('/generated/outlines/')).map(([, value]) => value.file));
const entry = Object.keys(manifest).find(key => manifest[key].isEntry && key.endsWith('index.html'));
const records = [];
const route = id => `${base}/learn/path/full-curriculum/${id}?module=classical-ml`;
function closure(roots) {
  const visited = new Set();
  const visit = key => { if (visited.has(key)) return; visited.add(key); (manifest[key].imports || []).forEach(visit); };
  roots.forEach(visit);
  return new Set([...visited].map(key => manifest[key].file));
}
function checkNetwork(requests, ids, routeSource) {
  const loaded = new Set(requests.filter(url => url.startsWith(base)).map(url => new URL(url).pathname.slice(1)));
  assert.deepEqual([...loaded].filter(path => bodies.has(path)).sort(), ids.map(body).sort());
  assert.deepEqual([...loaded].filter(path => outlines.has(path)), []);
  const allowed = closure([entry, routeSource, ...ids.map(source)]);
  const scripts = [...loaded].filter(path => path.endsWith('.js'));
  assert.deepEqual(scripts.filter(path => !allowed.has(path)), []);
  return { lessonBodies: ids.map(body), scripts, decodedJsBytes: scripts.reduce((sum, path) => sum + fs.statSync(`dist/${path}`).size, 0), gzipJsEstimate: scripts.reduce((sum, path) => sum + gzipSync(fs.readFileSync(`dist/${path}`)).length, 0) };
}

(async () => {
  const { tracks, allTopicsOrdered } = await import('../src/learn/data/generated/navigation.js');
  const topicMap = Object.fromEntries(allTopicsOrdered.map(topic => [topic.id, topic]));
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.deepEqual(scope, module.topicIds.slice(28, 33));
  assert.deepEqual(publication, baseline.originalPublication);
  assert.equal(bodies.size, Object.keys(publication).length);
  for (const id of Object.keys(publication)) assert.equal(manifest[source(id)].isDynamicEntry, true);
  const reviewedSources = new Set([...scope.map(source), 'src/learn/components/topic-content.css']);
  for (const name of ['crf', 'gaussian-process', 'semi-supervised', 'active-learning', 'evaluation-metrics']) {
    const receipt = read(`docs/teaching/evidence/${name}-author-review.json`);
    const bindings = receipt.sourceHashes || receipt.reviewedFiles || receipt.source;
    assert.ok(bindings && Object.keys(bindings).length, `Missing author binding: ${name}`);
    for (const [path, digest] of Object.entries(bindings)) {
      if (!path.startsWith('src/') && !path.startsWith('public/')) continue;
      assert.equal(hash(path), digest, `Author-reviewed source changed: ${path}`);
      assert.ok(fs.statSync(path).mtimeMs <= fs.statSync('dist/.vite/manifest.json').mtimeMs, `Rebuild changed source: ${path}`);
      reviewedSources.add(path);
    }
  }
  const sourceHashes = Object.fromEntries([...reviewedSources].map(path => [path, hash(path)]));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const fresh = async () => {
    const context = await browser.newContext({ viewport: { width: 1366, height: 950 }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    const requests = [], errors = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    return { context, page, requests, errors };
  };
  const ready = async (page, id) => {
    await page.waitForFunction(({ id, title }) => location.pathname.endsWith(`/${id}`) && document.querySelector('.reader-header h1')?.textContent === title && document.querySelector('.reader-article')?.getAttribute('aria-busy') === 'false' && document.querySelector('.reader-complete')?.disabled === false, { id, title: topicMap[id].title });
  };
  const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  try {
    {
      const { context, page, requests, errors } = await fresh();
      await page.goto(`${base}/learn`, { waitUntil: 'domcontentloaded' });
      await page.locator('.path-card').first().waitFor();
      await settle(page);
      records.push({ check: 'Hub loads no lesson body or outline', ...checkNetwork(requests, [], 'src/learn/LearnHub.jsx') });
      assert.deepEqual(errors, []);
      await context.close();
    }
    for (const id of scope) {
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(id), { waitUntil: 'domcontentloaded' });
      await ready(page, id);
      const index = module.topicIds.indexOf(id);
      const group = page.locator('[data-module-id="classical-ml"]');
      assert.deepEqual(await group.locator('[data-topic-id]').evaluateAll(elements => elements.map(element => element.dataset.topicId)), module.topicIds);
      assert.equal(await group.locator('.reader-group__completed').innerText(), '0 completed');
      assert.ok((await page.locator('.reader-header__meta').innerText()).includes(`${index + 1} of ${module.topicIds.length} topics on this route`));
      assert.ok((await page.locator('.reader-footer__previous').innerText()).includes(topicMap[module.topicIds[index - 1]].title));
      assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicMap[module.topicIds[index + 1]].title));
      assert.equal(await page.locator('.katex-error').count(), 0);
      assert.deepEqual(errors, []);
      records.push({ check: 'Selected body only, full ordered module, compact completion count and correct adjacent lessons', id, ...checkNetwork(requests, [id], 'src/learn/Reader.jsx') });
      if (id === scope[0]) {
        await page.locator('.reader-complete').click();
        assert.equal(new URL(page.url()).pathname.split('/').at(-1), id);
        await page.reload();
        await ready(page, id);
        assert.equal(await group.locator('.reader-group__completed').innerText(), '1 completed');
        assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], id), true);
        records.push({ check: 'Completion persists under unchanged stable ID without auto-advancing' });
      }
      await context.close();
    }
    {
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(scope[0]));
      await ready(page, scope[0]);
      await page.locator('.reader-footer__next').click();
      await ready(page, scope[1]);
      const firstBodyRequests = requests.filter(url => new URL(url).pathname === `/${body(scope[0])}`).length;
      await page.goBack();
      await ready(page, scope[0]);
      assert.equal(requests.filter(url => new URL(url).pathname === `/${body(scope[0])}`).length, firstBodyRequests);
      records.push({ check: 'Next/Back preserves sequence and reuses already imported body', ...checkNetwork(requests, scope.slice(0, 2), 'src/learn/Reader.jsx') });
      assert.deepEqual(errors, []);
      await context.close();
    }
    {
      const { context, page, errors } = await fresh();
      let release;
      const pending = new Promise(resolve => { release = resolve; });
      await page.route(`**/${body(scope[0])}`, async request => { await pending; await request.continue(); });
      try {
        const requested = page.waitForRequest(request => new URL(request.url()).pathname === `/${body(scope[0])}`);
        await Promise.all([page.goto(route(scope[0]), { waitUntil: 'domcontentloaded' }), requested]);
        await page.locator('.lesson-loading').waitFor();
        assert.ok(await page.locator('.reader-complete').isDisabled());
        await page.locator('.reader-footer__next').click();
        await ready(page, scope[1]);
        release();
        await page.evaluate(async url => { await import(url); }, `${base}/${body(scope[0])}`);
        await settle(page);
        await ready(page, scope[1]);
        assert.deepEqual(errors, []);
        records.push({ check: 'Slow import leaves shell usable and cannot replace the newer destination' });
      } finally { release(); await context.close(); }
    }
    for (const failure of ['import', 'render']) {
      const { context, page, errors } = await fresh();
      let attempts = 0;
      await page.route(`**/${body(scope[0])}*`, request => {
        attempts += 1;
        if (attempts > 1) return request.continue();
        if (failure === 'import') return request.abort('failed');
        return request.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled scoped render failure")}};' });
      });
      await page.goto(route(scope[0]), { waitUntil: 'domcontentloaded' });
      const alert = page.locator('.lesson-load-error');
      await alert.waitFor();
      assert.equal(await alert.getAttribute('role'), 'alert');
      assert.ok(await page.locator('.reader-complete').isDisabled());
      assert.equal(await page.locator('.planned-lesson').count(), 0);
      if (failure === 'import') {
        await alert.getByRole('button', { name: 'Try again', exact: true }).focus();
        await page.keyboard.press('Enter');
        await settle(page);
        await page.waitForFunction(() => document.querySelector('.lesson-load-error') || document.querySelector('.reader-complete')?.disabled === false);
      }
      const reloadRequired = await alert.count() > 0;
      if (reloadRequired) await Promise.all([page.waitForEvent('domcontentloaded'), alert.getByRole('button', { name: 'Reload page', exact: true }).click()]);
      await ready(page, scope[0]);
      assert.ok(attempts > 1);
      assert.deepEqual(errors.filter(error => !error.includes('Controlled scoped render failure')), []);
      records.push({ check: `${failure} failure recovery keeps shell and completion guard`, attempts, reloadRequired });
      await context.close();
    }
    for (const [path, digest] of Object.entries(sourceHashes)) assert.equal(hash(path), digest);
    assert.equal(hash('dist/.vite/manifest.json'), initialBuildHash);
    fs.writeFileSync('docs/teaching/evidence/classical-query-evaluation-integration.json', JSON.stringify({ status: 'passed', checkedAt: new Date().toISOString(), base, scope, catalogueCount: allTopicsOrdered.length, publicationCount: Object.keys(publication).length, moduleCount: tracks.length, moduleTopics: module.topicIds.length, buildManifestSha256: initialBuildHash, sourceHashes, records, byteMeaning: 'Actual requested production JS files; decoded file bytes and gzip estimates, not measured wire compression or latency.' }, null, 2) + '\n');
    console.log(`PASS: ${records.length} production integration groups across five lessons.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
