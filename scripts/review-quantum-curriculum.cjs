const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');

(async () => {
  const { tracks, learningPaths, getLearningRoute } = await import('./lib/authoring-curriculum.mjs');
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const publication = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
  const track = tracks.find(track => track.id === 'quantum-ai');
  const path = learningPaths.find(path => path.id === 'quantum-computing');
  const route = getLearningRoute(path);
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4183';
  const outputDirectory = 'scratch/quantum-curriculum-review';
  fs.mkdirSync(outputDirectory, { recursive: true });
  const results = [];
  const screenshots = [];
  const errors = [];
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [1440, 390]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 } });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn`);
      for (const selectedPath of [path, learningPaths.find(path => path.id === 'full-curriculum')]) {
        const resolved = getLearningRoute(selectedPath);
        const card = page.locator(`[data-path-id="${selectedPath.id}"]`);
        await card.waitFor();
        assert.equal(await card.locator('.path-card__meta').innerText(), `${resolved.moduleCount} modules · ${resolved.topicIds.length.toLocaleString()} topics · 0 completed`);
        const published = resolved.topicIds.filter(id => publication[id]).length;
        assert.equal(await card.locator('.path-card__availability').innerText(), `${published} published · ${resolved.topicIds.length - published} planned`);
        results.push({ width, check: 'resolved path counts', path: selectedPath.id });
      }
      await page.getByRole('tab', { name: 'Modules', exact: true }).click();
      const moduleCard = page.locator('.track-card').filter({ has: page.getByRole('heading', { name: track.title, exact: true }) });
      await moduleCard.getByRole('button', { name: 'View outline', exact: true }).click();
      assert.deepEqual(await moduleCard.locator('.track-topic').allTextContents(), track.topicIds.map(id => topicCatalogue[id].title));
      assert.match(await moduleCard.locator('.track-card__eyebrow').innerText(), new RegExp(`${track.topicIds.length} TOPICS.*${track.sections.length} SECTIONS`));
      results.push({ width, check: 'complete module order and section counts' });

      const requests = [];
      const captureRequest = request => requests.push(request.url());
      page.on('request', captureRequest);
      await page.getByRole('tab', { name: 'Search catalogue', exact: true }).click();
      await page.getByRole('combobox', { name: 'Module', exact: true }).selectOption('quantum-ai');
      for (const [query, id] of [
        ['QSVT', 'block-encodings-quantum-signal-processing-qsvt'],
        ['GKP', 'bosonic-quantum-codes-cat-binomial-gkp-encodings'],
        ['ML KEM', 'post-quantum-cryptography-ml-security-implications'],
        ['Rydberg', 'neutral-atom-qubits-optical-tweezers-rydberg-blockade'],
        ['Kraus', 'quantum-channels-kraus-operators-complete-positivity'],
        ['QIR', 'quantum-circuit-languages-additional-tools'],
      ]) {
        await page.locator('#topic-search').fill(query);
        await page.locator(`.topic-result[data-topic-id="${id}"]`).waitFor();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
        results.push({ width, check: 'named subtopic search', query, topic: id });
      }
      assert.ok(!requests.some(url => track.topicIds.some(id => url.includes(`/assets/${id}-`))), 'Search loaded quantum content eagerly');
      requests.length = 0;
      const first = track.topicIds[0];
      await page.goto(`${base}/learn/path/quantum-computing/${first}?module=quantum-ai`);
      await page.locator('.planned-lesson .syllabus-sequence').waitFor();
      assert.deepEqual(await page.locator('[data-module-id="quantum-ai"] .reader-topic').evaluateAll(nodes => nodes.map(node => node.dataset.topicId)), track.topicIds);
      assert.ok(requests.some(url => url.includes(`/assets/${first}-`)), 'Selected outline not fetched');
      assert.ok(!requests.some(url => track.topicIds.slice(1).some(id => url.includes(`/assets/${id}-`))), 'Unvisited quantum outline fetched');
      page.off('request', captureRequest);
      results.push({ width, check: 'full quantum path sequence and lazy selected-outline loading' });

      await page.locator('.reader-footer__next').click();
      await page.waitForURL(url => url.pathname.endsWith(`/${track.topicIds[1]}`));
      await page.waitForFunction(title => document.querySelector('.reader-header h1')?.textContent === title, topicCatalogue[track.topicIds[1]].title);
      await page.locator('.reader-footer__previous').click();
      await page.waitForURL(url => url.pathname.endsWith(`/${first}`));
      await page.locator('.planned-lesson .syllabus-sequence').waitFor();
      await page.evaluate(() => { window.__reviewHeaderVisibleSince = null; });
      await page.waitForFunction(title => {
        const heading = document.querySelector('.reader-header h1');
        const bounds = heading?.getBoundingClientRect();
        if (!(heading?.textContent === title && window.scrollY < 1 && bounds.y >= 0 && bounds.y < innerHeight)) {
          window.__reviewHeaderVisibleSince = null;
          return false;
        }
        window.__reviewHeaderVisibleSince ??= performance.now();
        return performance.now() - window.__reviewHeaderVisibleSince > 250;
      }, topicCatalogue[first].title);
      const screenshotPath = `${outputDirectory}/quantum-computing-${width}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      screenshots.push({ path: screenshotPath, sha256: createHash('sha256').update(fs.readFileSync(screenshotPath)).digest('hex') });
      results.push({ width, check: 'previous/next respects module sequence and settled heading' });

      for (const section of track.sections) {
        const id = section.topicIds[0];
        const topic = topicCatalogue[id];
        await page.goto(`${base}/learn/track/quantum-ai/${id}`);
        await page.locator('.planned-lesson .syllabus-subtopics').waitFor();
        assert.deepEqual(await page.locator('.syllabus-subtopics li').allTextContents(), topic.subtopics);
        assert.equal(await page.locator('.syllabus-prerequisites a').count(), topic.prerequisiteIds.length);
        if (topic.blueprint) assert.deepEqual(await page.locator('.syllabus-sequence li').allTextContents(), topic.blueprint.sequence);
        else assert.equal(await page.locator('.syllabus-sequence').count(), 0);
        assert.ok(await page.locator('.reader-complete').isDisabled());
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
        results.push({ width, check: 'planned scope, prerequisites and responsive page', section: section.name, topic: id });
      }
      const retainedId = 'qubits-superposition-entanglement';
      const saved = { [retainedId]: true, 'python-basics-types-control-flow-functions-modules': true };
      await page.evaluate(value => localStorage.setItem('kd-progress', JSON.stringify(value)), saved);
      await page.goto(`${base}/learn/track/quantum-ai/${retainedId}`);
      await page.locator(`.reader-topic.is-current[data-topic-id="${retainedId}"]`).waitFor();
      assert.ok((await page.locator(`.reader-topic.is-current[data-topic-id="${retainedId}"]`).getAttribute('class')).includes('is-complete'));
      assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem('kd-progress'))), saved);
      results.push({ width, check: 'retained stable identity and stored progress' });
      await context.close();
    }
    assert.deepEqual(errors, []);
    const evidence = { status: 'passed', checked: new Date().toISOString(), base, browser: 'Microsoft Edge / Playwright', widths: [1440, 390],
      path: { id: path.id, modules: route.moduleCount, topics: route.topicIds.length }, results, screenshots, pageErrors: errors };
    fs.writeFileSync('docs/curriculum/quantum-curriculum-browser-evidence.json', JSON.stringify(evidence, null, 2) + '\n');
    console.log(`PASS: ${results.length} quantum browser checks; path/module counts, named search, 12 section samples at each width, navigation, planned state, progress and lazy loading.`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
