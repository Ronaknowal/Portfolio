// Scoped production integration for Classical ML positions 34–39.
// Numerical and pedagogical review belong to the topic-specific audit records.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4188';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const baseline = read('docs/teaching/evidence/classical-final-audit-baseline.json');
const publication = read('src/learn/data/lesson-manifest.json');
const build = read('dist/.vite/manifest.json');
const scope = baseline.scope;
const source = id => `src/learn/data/${publication[id].replace(/^\.\//, '')}`;
const body = id => build[source(id)].file;
const bodies = new Set(Object.keys(publication).map(body));
const outlines = new Set(Object.entries(build).filter(([key]) => key.includes('/generated/outlines/')).map(([, value]) => value.file));
const entry = Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html'));
const evidencePath = 'docs/teaching/evidence/classical-final-audit-integration.json';
const records = [];
let activeCheck = 'catalogue and build identity';
const route = id => `${base}/learn/path/full-curriculum/${id}?module=classical-ml`;
const buildManifestSha256 = hash('dist/.vite/manifest.json');
const sourceHashes = Object.fromEntries([...new Set([
  ...Object.keys(baseline.sourceHashes),
  'src/learn/components/topic-content.css', 'src/learn/data/lesson-manifest.json',
])].map(file => [file, hash(file)]));

function networkEvidence(requests, ids, routeSource) {
  const loaded = new Set(requests.filter(url => url.startsWith(`${base}/`)).map(url => new URL(url).pathname.slice(1)));
  assert.deepEqual([...loaded].filter(file => bodies.has(file)).sort(), ids.map(body).sort());
  assert.deepEqual([...loaded].filter(file => outlines.has(file)), []);
  const allowed = new Set();
  const visit = key => { if (allowed.has(build[key].file)) return; allowed.add(build[key].file); (build[key].imports || []).forEach(visit); };
  [entry, routeSource, ...ids.map(source)].forEach(visit);
  const scripts = [...loaded].filter(file => file.endsWith('.js'));
  assert.deepEqual(scripts.filter(file => !allowed.has(file)), []);
  return { lessonBodies: ids.map(body), decodedJsBytes: scripts.reduce((n, file) => n + fs.statSync(`dist/${file}`).size, 0), gzipJsEstimate: scripts.reduce((n, file) => n + gzipSync(fs.readFileSync(`dist/${file}`)).length, 0) };
}

async function inspectPage(page, topicMap) {
  const result = await page.locator('.reader-article').evaluate(root => {
    const visible = node => node.getClientRects().length > 0;
    const ids = Array.from(document.querySelectorAll('[id]'), node => node.id);
    const anchors = Array.from(root.querySelectorAll('a[href]'));
    return {
      duplicateIds: ids.filter((id, index) => ids.indexOf(id) !== index),
      missingAnchors: anchors.filter(a => a.getAttribute('href').startsWith('#') && !document.getElementById(decodeURIComponent(a.hash.slice(1)))).map(a => a.getAttribute('href')),
      links: [...new Set(anchors.map(a => a.getAttribute('href')))],
      defaultBlueLinks: anchors.filter(visible).filter(a => ['rgb(0, 0, 238)', 'rgb(85, 26, 139)'].includes(getComputedStyle(a).color)).map(a => a.textContent),
      collapsedMath: Array.from(root.querySelectorAll('.katex svg')).filter(visible).filter(svg => svg.getBoundingClientRect().height < 1).map(svg => svg.closest('.katex').textContent.slice(0, 150)),
      unlabeledControls: Array.from(root.querySelectorAll('input:not([type=hidden]), select, textarea')).filter(visible).filter(node => !node.labels?.length && !node.getAttribute('aria-label') && !node.getAttribute('aria-labelledby')).map(node => node.outerHTML.slice(0, 150)),
      pageOverflow: document.documentElement.scrollWidth > innerWidth + 1,
      mathErrors: root.querySelectorAll('.katex-error').length,
    };
  });
  for (const key of ['duplicateIds', 'missingAnchors', 'defaultBlueLinks', 'collapsedMath', 'unlabeledControls']) assert.deepEqual(result[key], [], `${page.url()} ${key}`);
  assert.equal(result.pageOverflow, false, page.url());
  assert.equal(result.mathErrors, 0);
  for (const href of result.links.filter(href => href.startsWith('/learn/'))) {
    const url = new URL(href, base);
    assert.ok(topicMap[url.pathname.split('/').at(-1)], `Unknown lesson link: ${href}`);
  }
  const downloads = result.links.filter(href => href.startsWith('/learn-assets/') || href.startsWith('/learn/downloads/') || href.startsWith('/learn/examples/'));
  for (const href of downloads) {
    const file = `public${decodeURIComponent(new URL(href, base).pathname)}`;
    assert.ok(fs.existsSync(file), `Missing download ${file}`);
    const response = await page.request.get(new URL(href, base).href);
    assert.equal(response.status(), 200, href);
    assert.ok((await response.body()).equals(fs.readFileSync(file)), `Served download differs: ${href}`);
  }
  return { links: result.links.length, downloads: downloads.length, layoutAndControls: 'passed' };
}

(async () => {
  const { tracks, allTopicsOrdered } = await import('../src/learn/data/generated/navigation.js');
  const { getLearningRoute, learningPaths } = await import('../src/learn/data/curriculum.js');
  const topicMap = Object.fromEntries(allTopicsOrdered.map(topic => [topic.id, topic]));
  const module = tracks.find(track => track.id === 'classical-ml');
  const steps = getLearningRoute(learningPaths.find(path => path.id === 'full-curriculum')).steps;
  assert.deepEqual(scope, module.topicIds.slice(33));
  assert.deepEqual(module.topicIds, baseline.moduleTopicIds);
  assert.deepEqual(publication, baseline.publication);
  assert.ok(module.topicIds.every(id => publication[id]));
  for (const id of Object.keys(publication)) assert.equal(build[source(id)].isDynamicEntry, true);
  for (const file of Object.keys(sourceHashes)) assert.ok(fs.statSync(file).mtimeMs <= fs.statSync('dist/.vite/manifest.json').mtimeMs, `Build predates ${file}`);
  fs.writeFileSync(evidencePath, JSON.stringify({ status: 'in-progress', checkedAt: new Date().toISOString(), scope }, null, 2) + '\n');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const ready = (page, id) => page.waitForFunction(({ id, title }) => location.pathname.endsWith(`/${id}`) && document.querySelector('.reader-header h1')?.textContent === title && document.querySelector('.reader-article')?.getAttribute('aria-busy') === 'false' && document.querySelector('.reader-complete')?.disabled === false, { id, title: topicMap[id].title });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 950 }, reducedMotion: 'reduce' });
    const hub = await context.newPage();
    const hubRequests = [];
    hub.on('request', request => hubRequests.push(request.url()));
    await hub.goto(`${base}/learn`);
    await hub.locator('.path-card').first().waitFor();
    records.push({ check: 'Hub isolation', ...networkEvidence(hubRequests, [], 'src/learn/LearnHub.jsx') });
    await context.close();
    for (const width of [1366, 390, 320]) {
      for (const id of scope) {
        const context = await browser.newContext({ viewport: { width, height: 950 }, reducedMotion: 'reduce' });
        const page = await context.newPage(), requests = [], errors = [];
        page.on('request', request => requests.push(request.url()));
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(route(id)); await ready(page, id); await page.evaluate(() => document.fonts.ready);
        const group = page.locator('[data-module-id="classical-ml"]');
        assert.deepEqual(await group.locator('[data-topic-id]').evaluateAll(nodes => nodes.map(node => node.dataset.topicId)), module.topicIds);
        const stepIndex = steps.findIndex(step => step.topicId === id && step.moduleId === 'classical-ml');
        for (const [selector, step] of [['.reader-footer__previous', steps[stepIndex - 1]], ['.reader-footer__next', steps[stepIndex + 1]]]) {
          assert.ok((await page.locator(selector).innerText()).includes(topicMap[step.topicId].title));
        }
        const inspection = await inspectPage(page, topicMap);
        assert.deepEqual(errors, []);
        records.push({ check: 'Ordered published lesson, local links/downloads, formula geometry, theme and layout', id, width, ...inspection, ...networkEvidence(requests, [id], 'src/learn/Reader.jsx') });
        if (width === 1366) {
          await page.locator('.reader-complete').click(); await page.reload(); await ready(page, id);
          assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], id), true);
          assert.equal(await group.locator('.reader-group__completed').innerText(), '1 completed');
        }
        await context.close();
      }
    }
    // Consecutive navigation retains earlier CSS: inspect the actual mixed-load state.
    const mixed = await browser.newContext({ viewport: { width: 1366, height: 950 } });
    const page = await mixed.newPage(), requests = [], errors = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(route(scope[0])); await ready(page, scope[0]);
    for (const id of scope.slice(1)) {
      await page.locator('.reader-footer__next').click(); await ready(page, id);
      await inspectPage(page, topicMap);
    }
    assert.deepEqual(errors, []);
    records.push({ check: 'Six-topic Next sequence and CSS coexistence', ...networkEvidence(requests, scope, 'src/learn/Reader.jsx') });
    await mixed.close();
    for (const failure of ['import', 'render']) {
      activeCheck = `${failure} failure recovery`;
      const context = await browser.newContext({ viewport: { width: 1366, height: 950 } });
      const page = await context.newPage();
      const id = scope[3]; // Newly published body, previously a separate outline.
      let attempts = 0;
      await page.route(`**/${body(id)}*`, request => {
        attempts += 1;
        if (attempts > 1) return request.continue();
        if (failure === 'import') return request.abort('failed');
        return request.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled final-audit render failure")}};' });
      });
      await page.goto(route(id));
      const alert = page.locator('.lesson-load-error');
      await alert.waitFor();
      assert.equal(await alert.getAttribute('role'), 'alert');
      assert.equal(await page.locator('.planned-lesson').count(), 0);
      assert.ok(await page.locator('.reader-complete').isDisabled());
      await Promise.all([page.waitForEvent('domcontentloaded'), alert.getByRole('button', { name: 'Reload page', exact: true }).click()]);
      await ready(page, id);
      assert.ok(attempts > 1);
      records.push({ check: 'Published lesson failure is recoverable and cannot mark completion', failure, id, attempts });
      await context.close();
    }
    for (const [file, digest] of Object.entries(sourceHashes)) assert.equal(hash(file), digest, `Source changed during integration: ${file}`);
    assert.equal(hash('dist/.vite/manifest.json'), buildManifestSha256);
    fs.writeFileSync(evidencePath, JSON.stringify({ status: 'passed', checkedAt: new Date().toISOString(), scope, sourceHashes, buildManifestSha256, moduleTopics: module.topicIds.length, published: Object.keys(publication).length, records, byteMeaning: 'Actual requested production JS files, decoded file bytes and gzip estimates; not network latency measurements.' }, null, 2) + '\n');
    console.log(`PASS: ${records.length} production integration groups, six lessons at three widths.`);
  } finally { await browser.close(); }
})().catch(error => {
  fs.writeFileSync(evidencePath, JSON.stringify({ status: 'failed', checkedAt: new Date().toISOString(), scope, sourceHashes, buildManifestSha256, activeCheck, records, error: error.stack }, null, 2) + '\n');
  console.error(error); process.exitCode = 1;
});
