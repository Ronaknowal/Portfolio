const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');

(async () => {
  const { tracks, learningPaths, getLearningRoute } = await import('./lib/authoring-curriculum.mjs');
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const publication = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
  const baseline = JSON.parse(fs.readFileSync('docs/curriculum/professional-modules-baseline.json', 'utf8'));
  const originalIds = new Set(baseline.topicIds);
  const selectedTracks = tracks.filter(track => ['quantitative-finance', 'system-design'].includes(track.id));
  const selectedPaths = learningPaths.filter(route => ['quant-trading', 'system-design-engineer', 'full-curriculum'].includes(route.id));
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4183';
  const outputDirectory = 'scratch/professional-curriculum-review';
  fs.mkdirSync(outputDirectory, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const screenshots = [];
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 } });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn`);
      await page.locator('.path-card').first().waitFor();
      const catalogueRequests = [];
      const captureCatalogueRequest = request => catalogueRequests.push(request.url());
      page.on('request', captureCatalogueRequest);
      await page.getByRole('tab', { name: 'Search catalogue', exact: true }).click();
      for (const [query, id, moduleId] of [
        ['HyperLogLog', 'hyperloglog-hll-approximate-distinct-counting', 'system-design'],
        ['Snowflake', 'distributed-ids-ordering-uniqueness-guarantees', 'system-design'],
        ['t digest', 'streaming-quantiles-kll-t-digest-reservoir-sampling', 'system-design'],
        ['Newey West', 'financial-inference-hac-errors-bootstrap-multiple-testing', 'quantitative-finance'],
        ['SABR', 'local-volatility-stochastic-volatility-jumps-rough-models', 'quantitative-finance'],
      ]) {
        await page.getByRole('combobox', { name: 'Module', exact: true }).selectOption(moduleId);
        await page.locator('#topic-search').fill(query);
        const result = page.locator(`.topic-result[data-topic-id="${id}"]`);
        await result.waitFor();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
        if (query === 'Snowflake') assert.match(await result.innerText(), /Curriculum scope: Snowflake IDs/);
        results.push({ width, check: 'named concept search with module filter', query, topic: id });
      }
      assert.ok(!catalogueRequests.some(url => selectedTracks.some(track => track.topicIds.some(id => url.includes(`/assets/${id}-`)))), 'Searching fetched lesson or outline chunks');
      page.off('request', captureCatalogueRequest);
      await page.locator('#topic-search').fill('zzzz-no-such-concept-zzzz');
      await page.locator('.catalogue-empty').waitFor();
      await page.getByRole('tab', { name: 'Guided paths', exact: true }).click();
      for (const route of selectedPaths) {
        const resolved = getLearningRoute(route);
        const card = page.locator(`[data-path-id="${route.id}"]`);
        const published = resolved.topicIds.filter(id => publication[id]).length;
        assert.equal(await card.locator('.path-card__meta').innerText(), `${resolved.moduleCount} modules · ${resolved.topicIds.length.toLocaleString()} topics · 0 completed`);
        assert.equal(await card.locator('.path-card__availability').innerText(), `${published} published · ${resolved.topicIds.length - published} planned`);
        results.push({ width, check: 'resolved path counts', path: route.id });
      }
      for (const id of [
        'bloom-cuckoo-xor-filters-approximate-membership',
        'hyperloglog-hll-approximate-distinct-counting',
        'count-min-sketch-count-sketch-streaming-heavy-hitters',
        'streaming-quantiles-kll-t-digest-reservoir-sampling',
        'merkle-trees-content-addressing-data-integrity',
        'design-studio-experimentation-platforms-assignment-metric-integrity',
        'stochastic-control-hjb-equations-optimal-stopping-in-finance',
        'rfq-markets-dealer-pricing-electronic-otc-trading',
      ]) {
        await page.goto(`${base}/learn/topic/${id}`);
        await page.locator('.planned-lesson .syllabus-sequence').waitFor();
        assert.deepEqual(await page.locator('.syllabus-subtopics li').allTextContents(), topicCatalogue[id].subtopics);
        assert.deepEqual(await page.locator('.syllabus-sequence li').allTextContents(), topicCatalogue[id].blueprint.sequence);
        assert.ok(await page.locator('.reader-complete').isDisabled());
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
        if (id === 'hyperloglog-hll-approximate-distinct-counting') {
          await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
          const filename = path.join(outputDirectory, `hyperloglog-${width}.png`);
          await page.screenshot({ path: filename, fullPage: false });
          const bytes = fs.readFileSync(filename);
          screenshots.push({ path: filename.replaceAll('\\', '/'), sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length });
        }
        results.push({ width, check: 'new topic concept scope, sequence, planned state and responsive layout', topic: id });
      }
      await page.goto(`${base}/learn`);
      await page.getByRole('tab', { name: 'Modules', exact: true }).click();
      for (const track of selectedTracks) {
        const card = page.locator('.track-card').filter({ has: page.getByRole('heading', { name: track.title, exact: true }) });
        assert.match(await card.locator('.track-card__eyebrow').innerText(), new RegExp(`${track.topicIds.length} TOPICS.*${track.sections.length} SECTIONS`));
        await card.getByRole('button', { name: 'View outline', exact: true }).click();
        assert.equal(await card.locator('.track-topic').count(), track.topicIds.length);
        assert.deepEqual(await card.locator('.track-topic').allTextContents(), track.topicIds.map(id => topicCatalogue[id].title));
        results.push({ width, check: 'complete module outline in syllabus order', module: track.id });
      }
      for (const track of selectedTracks) {
        const first = track.topicIds[0];
        const requests = [];
        const recordRequest = request => requests.push(request.url());
        page.on('request', recordRequest);
        await page.goto(`${base}/learn/track/${track.id}/${first}`);
        await page.locator('.planned-lesson .syllabus-sequence').waitFor();
        assert.equal(await page.locator('.reader-header h1').innerText(), topicCatalogue[first].title);
        assert.ok(await page.locator('.reader-complete').isDisabled());
        assert.deepEqual(await page.locator(`[data-module-id="${track.id}"] .reader-topic`).evaluateAll(nodes => nodes.map(node => node.dataset.topicId)), track.topicIds);
        assert.ok(requests.some(url => url.includes(`/assets/${first}-`)), `Selected outline was not loaded: ${first}`);
        assert.ok(!requests.some(url => track.topicIds.slice(1).filter(id => !originalIds.has(id)).some(id => url.includes(`/assets/${id}-`))), 'Unvisited outlines loaded eagerly');
        page.off('request', recordRequest);
        await page.locator('.reader-footer__next').click();
        await page.waitForURL(url => url.pathname.endsWith(`/${track.topicIds[1]}`));
        await page.waitForFunction(title => document.querySelector('.reader-header h1')?.textContent === title, topicCatalogue[track.topicIds[1]].title);
        await page.locator('.reader-footer__previous').click();
        await page.waitForURL(url => url.pathname.endsWith(`/${first}`));
        await page.locator('.planned-lesson .syllabus-sequence').waitFor();
        // Footer navigation uses the site's smooth scroll. Wait for that real
        // transition and a paint before capturing, rather than recording a
        // transient empty viewport while traveling up the previous long page.
        await page.evaluate(() => { window.__reviewHeaderVisibleSince = null; });
        await page.waitForFunction(title => {
          // A single zero-scroll frame can precede scroll anchoring after the
          // asynchronous outline expands. Require a sustained visible header.
          const heading = document.querySelector('.reader-header h1');
          const bounds = heading?.getBoundingClientRect();
          if (!(heading?.textContent === title && window.scrollY < 1 && bounds.y >= 0 && bounds.y < innerHeight)) {
            window.__reviewHeaderVisibleSince = null;
            return false;
          }
          window.__reviewHeaderVisibleSince ??= performance.now();
          return performance.now() - window.__reviewHeaderVisibleSince > 250;
        }, topicCatalogue[first].title);
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const headerBounds = await page.locator('.reader-header h1').boundingBox();
        assert.ok(headerBounds && headerBounds.y >= 0 && headerBounds.y < 1000, `Reader heading not visible: ${JSON.stringify({ width, module: track.id, headerBounds })}`);
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
        assert.equal(overflow, false, `Page overflow at ${width}: ${track.id}`);
        const filename = path.join(outputDirectory, `${track.id}-${width}.png`);
        await page.screenshot({ path: filename, fullPage: false });
        const bytes = fs.readFileSync(filename);
        screenshots.push({ path: filename.replaceAll('\\', '/'), sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length });
        // One new topic from every section tests lazy outlines, their complete
        // rendered scope and prerequisite links without claiming content review.
        for (const section of track.sections) {
          const id = section.topicIds.find(value => !originalIds.has(value));
          if (!id) continue;
          await page.goto(`${base}/learn/path/full-curriculum/${id}?module=${track.id}`);
          await page.locator('.planned-lesson .syllabus-sequence').waitFor();
          assert.deepEqual(await page.locator('.syllabus-sequence li').allTextContents(), topicCatalogue[id].blueprint.sequence);
          assert.equal(await page.locator('.syllabus-prerequisites a').count(), topicCatalogue[id].prerequisiteIds.length);
          assert.ok(await page.locator('.reader-complete').isDisabled());
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
          results.push({ width, check: 'planned scope, prerequisites and responsive layout', module: track.id, section: section.name, topic: id });
        }
        results.push({ width, check: 'module navigation, planned state and lazy outline loading', module: track.id });
      }
      // Existing stable progress survives finance regrouping; shared networking
      // also has one identity when accessed from the new system-design module.
      const financeId = 'market-instruments-returns-cash-flow-accounting';
      const sharedId = 'networking-foundations-packets-transport-dns-sockets';
      const saved = { [financeId]: true, [sharedId]: true };
      await page.evaluate(value => localStorage.setItem('kd-progress', JSON.stringify(value)), saved);
      await page.goto(`${base}/learn/track/quantitative-finance/${financeId}`);
      await page.locator(`.reader-topic.is-current[data-topic-id="${financeId}"]`).waitFor();
      assert.ok((await page.locator(`.reader-topic.is-current[data-topic-id="${financeId}"]`).getAttribute('class')).includes('is-complete'));
      await page.goto(`${base}/learn/track/system-design/${sharedId}?module=system-design`);
      await page.locator(`[data-module-id="system-design"] .reader-topic.is-current[data-topic-id="${sharedId}"]`).waitFor();
      assert.ok((await page.locator(`[data-module-id="system-design"] .reader-topic.is-current`).getAttribute('class')).includes('is-complete'));
      assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem('kd-progress'))), saved);
      results.push({ width, check: 'retained and shared progress identities' });
      await context.close();
    }
    assert.deepEqual(errors, []);
    const evidence = { status: 'passed', checked: new Date().toISOString(), base, browser: 'Microsoft Edge / Playwright', widths: [1440, 390], results, screenshots, pageErrors: errors };
    fs.writeFileSync('docs/curriculum/professional-curriculum-browser-evidence.json', JSON.stringify(evidence, null, 2) + '\n');
    console.log(`PASS: ${results.length} scoped browser checks; module/path counts, all outline memberships, 24 section samples at each width, navigation, progress, responsive layout and lazy loading.`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
