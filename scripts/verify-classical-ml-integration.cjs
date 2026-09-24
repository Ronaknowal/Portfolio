// Production checks for the authorized Classical ML increment. Does not rerun
// completed modules or each lesson's already recorded numerical/interaction suite.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { gzipSync } = require('node:zlib');
const { createHash } = require('node:crypto');
const { parse } = require('@babel/parser');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const origin = new URL(base).origin;
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const ledger = read('docs/teaching/classical-ml-supervised-progress.json');
const baseline = read(ledger.baseline);
const manifest = read('src/learn/data/lesson-manifest.json');
const build = read('dist/.vite/manifest.json');
const initialManifestHash = hash('dist/.vite/manifest.json');
const initialLedgerHash = hash('docs/teaching/classical-ml-supervised-progress.json');
const checkedSourceHashes = new Map();
const source = id => `src/learn/data/${manifest[id].replace(/^\.\//, '')}`;
const file = key => {
  assert.ok(build[key], `Missing production entry: ${key}`);
  return build[key].file;
};
const lessonFiles = new Set(Object.keys(manifest).map(id => file(source(id))));
const outlineFiles = new Set(Object.keys(build).filter(key => key.startsWith('src/learn/data/generated/outlines/')).map(file));
const entry = Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html'));
assert.ok(entry, 'Missing Vite HTML entry');
const url = id => `${base}/learn/path/full-curriculum/${id}?module=classical-ml`;

function buildClosure(roots) {
  const visited = new Set();
  const files = new Set();
  function visit(key) {
    if (visited.has(key)) return;
    visited.add(key);
    files.add(file(key));
    for (const imported of build[key].imports || []) visit(imported);
  }
  roots.forEach(visit);
  return files;
}

function sourceClosure(filename) {
  const visited = new Set();
  function visit(absolute) {
    const relative = path.relative(process.cwd(), absolute).replaceAll('\\', '/');
    assert.ok(!relative.startsWith('../'), 'Source import escaped the repository');
    if (visited.has(relative)) return;
    visited.add(relative);
    if (!/\.(js|jsx)$/.test(absolute)) return;
    const ast = parse(fs.readFileSync(absolute, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
    for (const declaration of ast.program.body) {
      if (!['ImportDeclaration', 'ExportNamedDeclaration', 'ExportAllDeclaration'].includes(declaration.type)) continue;
      const specifier = declaration.source?.value;
      if (!specifier?.startsWith('.')) continue;
      const target = path.resolve(path.dirname(absolute), specifier);
      const resolved = [target, ...['.js', '.jsx', '.css', '.json'].map(extension => target + extension), path.join(target, 'index.js'), path.join(target, 'index.jsx')]
        .find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
      assert.ok(resolved, `Unresolved import ${specifier} from ${relative}`);
      visit(resolved);
    }
  }
  visit(path.resolve(filename));
  return [...visited];
}

function checkNetwork(requests, ids) {
  const local = new Set(requests.map(request => new URL(request)).filter(request => request.origin === origin).map(request => request.pathname.slice(1)));
  assert.deepEqual([...local].filter(item => lessonFiles.has(item)).sort(), ids.map(id => file(source(id))).sort(), 'Unrelated or missing lesson body');
  assert.deepEqual([...local].filter(item => outlineFiles.has(item)), [], 'Published lesson fetched a syllabus outline');
  const allowed = buildClosure([entry, 'src/learn/Reader.jsx', ...ids.map(source)]);
  assert.deepEqual([...local].filter(item => item.endsWith('.js') && !allowed.has(item)), [], 'JavaScript outside the selected dependency closure');
  const scripts = [...local].filter(item => item.endsWith('.js'));
  return {
    scripts,
    rawScriptBytes: scripts.reduce((sum, filename) => sum + fs.statSync(path.join('dist', filename)).size, 0),
    gzipScriptBytes: scripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join('dist', filename))).length, 0),
    byteMeaning: 'Built files requested in a fresh browser context; gzip estimate is not a measured network compression or latency.',
  };
}

(async () => {
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { topicMap } = await import('../src/learn/data/catalogue.js');
  assert.deepEqual(Object.keys(topicCatalogue), baseline.catalogueIds, 'Catalogue IDs/order changed');
  assert.deepEqual(manifest, baseline.publicationMappings, 'Publication mapping changed');
  const module = tracks.find(item => item.id === 'classical-ml');
  assert.deepEqual(module.topicIds, baseline.moduleOrder, 'Classical ML order changed');
  assert.deepEqual(ledger.topics.map(topic => topic.id), module.topicIds.slice(0, 10));
  for (const id of Object.keys(manifest)) assert.equal(build[source(id)].isDynamicEntry, true, `${id} lost its lazy body entry`);
  assert.equal(lessonFiles.size, Object.keys(manifest).length, 'Distinct body entries were bundled together');

  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const fresh = async () => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    const requests = [], errors = [], failedAssets = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    const isLocalAsset = address => {
      const parsed = new URL(address);
      return parsed.origin === origin && /\.(js|css)$/.test(parsed.pathname);
    };
    page.on('requestfailed', request => {
      if (isLocalAsset(request.url())) failedAssets.push({ url: request.url(), error: request.failure()?.errorText });
    });
    page.on('response', response => {
      if (isLocalAsset(response.url()) && response.status() >= 400) failedAssets.push({ url: response.url(), status: response.status() });
    });
    return { context, page, requests, errors, failedAssets };
  };
  const ready = async (page, topic) => {
    await page.waitForFunction(({ id, title }) => {
      const complete = document.querySelector('.reader-complete');
      return window.location.pathname.split('/').at(-1) === id
        && document.querySelector('.reader-header h1')?.textContent === title
        && document.querySelector('.reader-article .lesson-pilot')
        && document.querySelector('.reader-article')?.getAttribute('aria-busy') === 'false'
        && complete && !complete.disabled;
    }, topic);
  };
  const settleRenderTurn = page => page.evaluate(() => new Promise(resolve => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  }));
  const rememberSources = sources => {
    for (const filename of sources) {
      const current = hash(filename);
      if (checkedSourceHashes.has(filename)) assert.equal(current, checkedSourceHashes.get(filename), `Source changed during review: ${filename}`);
      checkedSourceHashes.set(filename, current);
    }
  };
  rememberSources([
    'src/learn/Reader.jsx',
    'src/learn/components/TopicContent.jsx',
    'src/learn/components/LessonBoundary.jsx',
    'src/learn/hooks/useTopicResource.js',
    'src/learn/data/lesson-loader.js',
    'src/learn/data/lesson-manifest.json',
    ledger.baseline,
  ]);
  try {
    for (const [index, topic] of ledger.topics.entries()) {
      assert.ok(topic.reviewedFiles, `${topic.id} needs reviewed source versions before integration`);
      for (const [filename, expected] of Object.entries(topic.reviewedFiles)) assert.equal(hash(filename), expected, `Reviewed version changed: ${filename}`);
      const sources = sourceClosure(source(topic.id));
      rememberSources([...sources, ...Object.keys(topic.reviewedFiles)]);
      assert.deepEqual(sources.filter(filename => filename.startsWith('src/learn/data/topics/')), [source(topic.id)]);
      assert.ok(!sources.some(filename => filename.includes('/data/practice/')), 'ML imported unrelated DSA practice');
      const dataFiles = sources.filter(filename => filename.startsWith('src/learn/data/'));
      assert.deepEqual(dataFiles.filter(filename => !(filename in topic.reviewedFiles)), [], 'Lesson imports unreviewed or unrelated data');
      assert.match(topicMap[topic.id].readTime, /min.*read/i, 'Reading-time units missing');

      const { context, page, requests, errors, failedAssets } = await fresh();
      await page.goto(url(topic.id), { waitUntil: 'domcontentloaded' });
      await ready(page, topic);
      assert.equal(await page.locator('.reader-header h1').innerText(), topic.title);
      const metadata = await page.locator('.reader-header__meta').innerText();
      assert.ok(metadata.includes(`${index + 1} of ${module.topicIds.length} topics on this route`));
      assert.ok(metadata.includes(topicMap[topic.id].readTime));
      const group = page.locator('[data-module-id="classical-ml"]');
      assert.equal(await group.locator('.reader-group__completed').innerText(), '0 completed');
      assert.deepEqual(await group.locator('[data-topic-id]').evaluateAll(items => items.map(item => item.dataset.topicId)), module.topicIds);
      assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicMap[module.topicIds[index + 1]].title));
      if (index) assert.ok((await page.locator('.reader-footer__previous').innerText()).includes(ledger.topics[index - 1].title));
      assert.equal(await page.locator('.katex-error').count(), 0);
      assert.deepEqual(errors, []);
      assert.deepEqual(failedAssets, [], 'A selected lesson asset failed to load');
      const network = checkNetwork(requests, [topic.id]);
      records.push({ case: 'Selected lesson, source ownership, module sequence and compact counts', topicId: topic.id, sourceFiles: sources, ...network });

      if (index === 0) {
        await page.locator('.reader-complete').click();
        assert.equal(new URL(page.url()).pathname.split('/').at(-1), topic.id, 'Completion silently advanced the lesson');
        await page.waitForFunction(() => document.querySelector('[data-module-id="classical-ml"] .reader-group__completed')?.textContent === '1 completed');
        assert.equal(await group.locator('.reader-group__completed').innerText(), '1 completed');
        await page.reload({ waitUntil: 'domcontentloaded' });
        await ready(page, topic);
        assert.equal(await page.locator('[data-module-id="classical-ml"] .reader-group__completed').innerText(), '1 completed');
        assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('kd-progress'))['linear-logistic-regression']), true);
        records.push({ case: 'Completion persists under the stable ID and stays on the chosen lesson' });
      }
      if (index === 9) {
        await page.locator('.reader-footer__next').click();
        await page.waitForURL(url(module.topicIds[10]));
        await page.waitForFunction(title => document.querySelector('.reader-header h1')?.textContent === title, topicMap[module.topicIds[10]].title);
        assert.equal(await page.locator('.reader-header h1').innerText(), topicMap[module.topicIds[10]].title);
        records.push({ case: 'Topic ten continues to the actual unmodified successor', successor: module.topicIds[10] });
      }
      await context.close();
    }

    // One representative slow import verifies shell availability and stale-result isolation.
    {
      const { context, page, errors } = await fresh();
      let release;
      const pending = new Promise(resolve => { release = resolve; });
      await page.route(`**/${file(source(ledger.topics[0].id))}`, async request => { await pending; await request.continue(); });
      const firstRequest = page.waitForRequest(request => new URL(request.url()).pathname === `/${file(source(ledger.topics[0].id))}`);
      try {
        await Promise.all([
          page.goto(url(ledger.topics[0].id), { waitUntil: 'domcontentloaded' }),
          firstRequest,
        ]);
        await page.locator('.lesson-loading').waitFor();
        assert.equal(await page.locator('.lesson-loading').getAttribute('role'), 'status');
        assert.ok(await page.locator('.reader-complete').isDisabled());
        await page.locator('.reader-footer__next').click();
        await ready(page, ledger.topics[1]);
        const destination = await page.locator('.reader-article .lesson-pilot').elementHandle();
        release();
        // Await the same browser module-map entry, including its evaluation,
        // instead of assuming that an arbitrary 200 ms delay was sufficient.
        await page.evaluate(async address => { await import(address); }, `${base}/${file(source(ledger.topics[0].id))}`);
        await settleRenderTurn(page);
        await ready(page, ledger.topics[1]);
        assert.ok(await destination.evaluate(element => element.isConnected && document.querySelector('.reader-article .lesson-pilot') === element), 'A late lesson replaced the destination body');
        assert.equal(await page.locator('.reader-article .lesson-pilot').count(), 1);
        assert.deepEqual(errors, []);
        records.push({ case: 'Slow lesson does not block sequence navigation or replace the destination when it arrives' });
      } finally {
        release();
        await context.close();
      }
    }

    // The injected failures are browser-only; no production source is changed.
    for (const failure of ['import', 'render']) {
      const { context, page, errors } = await fresh();
      let attempts = 0;
      await page.route(`**/${file(source(ledger.topics[0].id))}*`, async request => {
        attempts += 1;
        if (attempts !== 1) return request.continue();
        if (failure === 'import') return request.abort('failed');
        return request.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled ML render failure")}};' });
      });
      await page.goto(url(ledger.topics[0].id), { waitUntil: 'domcontentloaded' });
      const error = page.locator('.lesson-load-error');
      await error.waitFor();
      assert.equal(await error.getAttribute('role'), 'alert');
      assert.ok(await page.locator('.reader-complete').isDisabled());
      assert.equal(await page.locator('.planned-lesson').count(), 0);
      let reloadRequired = true;
      if (failure === 'import') {
        await error.getByRole('button', { name: 'Try again', exact: true }).focus();
        await page.keyboard.press('Enter');
        // The old alert may still exist at the end of the keyboard event.
        // Flush its queued render/effect turn before classifying retry results.
        await settleRenderTurn(page);
        await page.waitForFunction(() => document.querySelector('.reader-article .lesson-pilot') || document.querySelector('.lesson-load-error'));
        reloadRequired = await page.locator('.lesson-load-error').count() > 0;
      }
      if (reloadRequired) {
        await Promise.all([
          page.waitForEvent('domcontentloaded'),
          error.getByRole('button', { name: 'Reload page', exact: true }).click(),
        ]);
      }
      await ready(page, ledger.topics[0]);
      assert.ok(attempts > 1);
      assert.deepEqual(errors.filter(message => !message.includes('Controlled ML render failure')), []);
      records.push({ case: `${failure} failure preserves shell and completion guard, then recovers`, attempts, reloadRequired });
      await context.close();
    }
    for (const [filename, expected] of checkedSourceHashes) assert.equal(hash(filename), expected, `Source changed before evidence freeze: ${filename}`);
    assert.equal(hash('dist/.vite/manifest.json'), initialManifestHash, 'Production build changed during review');
    assert.equal(hash('docs/teaching/classical-ml-supervised-progress.json'), initialLedgerHash, 'Reviewed ledger changed during review');
    fs.writeFileSync('docs/teaching/evidence/classical-ml-production-review.json', JSON.stringify({
      checkedAt: new Date().toISOString(), browser: browser.version(), base,
      catalogueCount: baseline.catalogueIds.length, publicationCount: Object.keys(manifest).length,
      moduleTopicCount: module.topicIds.length, buildManifestSha256: initialManifestHash,
      reviewedLedgerSha256: initialLedgerHash, sourceHashes: Object.fromEntries(checkedSourceHashes), records,
      scope: 'Current ten-topic integration; unchanged per-topic numerical/browser evidence and completed earlier modules reused.',
    }, null, 2) + '\n');
    console.log(`PASS: ${records.length} production integration cases across the ten Classical ML lessons.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
