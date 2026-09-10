const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const { learningPaths, getPathTopicIds } = await import('./lib/authoring-curriculum.mjs');
  const baseline = new Set(JSON.parse(fs.readFileSync('docs/curriculum/pre-expansion-topic-ids.json', 'utf8')));
  const samples = ['hardware-systems', 'computational-neuroscience'].map(trackId => Object.values(topicCatalogue).find(t => t.trackId === trackId && t.blueprint && t.prerequisiteIds.length && !baseline.has(t.id)));
  assert.ok(samples.every(Boolean));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const dir = 'scratch/curriculum-review';
  fs.mkdirSync(dir, { recursive: true });
  const errors = [];
  const results = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto('http://127.0.0.1:5173/learn');
      await page.locator('.path-card').first().waitFor();
      assert.equal(await page.locator('.path-card').count(), 7);
      for (const title of ['Neural Engineering', 'GPU Engineering']) {
        const card = page.locator('.path-card').filter({ has: page.getByRole('heading', { name: title, exact: true }) });
        await card.locator('summary').click();
        assert.ok(await card.locator('li').count() >= 4);
        await card.getByRole('button', { name: 'Follow this path' }).click();
        await page.locator('.reader-header h1').waitFor();
        const route = learningPaths.find(p => p.title === title);
        const firstId = getPathTopicIds(route)[0];
        await page.waitForURL(`**/learn/path/${route.id}/${firstId}`);
        await page.locator(`.reader-topic.is-current[data-topic-id="${firstId}"]`).waitFor();
        await page.goto('http://127.0.0.1:5173/learn');
        await page.locator('.path-card').first().waitFor();
      }
      await page.screenshot({ path: `${dir}/paths-${width}.png`, fullPage: true });
      for (const topic of samples) {
        const route = learningPaths.find(p => p.id === (topic.trackId === 'hardware-systems' ? 'gpu-engineer' : 'neural-engineer'));
        await page.goto(`http://127.0.0.1:5173/learn/path/${route.id}/${topic.id}`);
        await page.getByRole('heading', { name: 'The learning sequence', exact: true }).waitFor();
        assert.equal(await page.locator('.reader-header h1').innerText(), topic.title);
        assert.equal((await page.locator('.reader-header .topic-status').innerText()).trim().toLowerCase(), 'planned');
        assert.ok(await page.getByRole('button', { name: 'Lesson not yet published' }).isDisabled());
        assert.equal(await page.locator('.syllabus-sequence li').count(), topic.blueprint.sequence.length);
        assert.equal(await page.locator('.syllabus-prerequisites a').count(), topic.prerequisiteIds.length);
        assert.equal(await page.locator('.reader-topic.is-current').count(), 1);
        const details = page.locator('.syllabus-details');
        await details.locator('summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await details.getAttribute('open'), '');
        assert.equal(await page.locator('.syllabus-sources a').count(), topic.blueprint.sources.length);
        for (const href of await page.locator('.syllabus-sources a').evaluateAll(nodes => nodes.map(n => n.href))) assert.match(href, /^https?:\/\//);
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Document overflow at ${width}: ${topic.title}`);
        await page.locator('.planned-lesson').screenshot({ path: `${dir}/${route.id}-${width}.png`, style: '.learn-nav { visibility: hidden !important; }' });
        const order = getPathTopicIds(route);
        const nextId = order[order.indexOf(topic.id) + 1];
        if (nextId) {
          await page.locator('.reader-footer__next').click();
          await page.waitForURL(`**/${nextId}`);
          await page.locator('.reader-topic.is-current').filter({ hasText: topicCatalogue[nextId].title }).waitFor();
          assert.equal(await page.locator('.reader-topic.is-current').count(), 1);
        }
        await page.goto(`http://127.0.0.1:5173/learn/topic/${topic.id}`);
        const target = await page.locator('.syllabus-prerequisites a').first().getAttribute('href');
        await page.locator('.syllabus-prerequisites a').first().click();
        await page.waitForURL(`**${target}`);
        assert.ok(await page.locator('.reader-header h1').innerText());
        results.push({ width, topic: topic.title, route: route.id, checks: 'outline, status, disabled completion, prerequisites, keyboard disclosure, sources, sidebar, next navigation, no overflow' });
      }
      // Original approved lesson and a representative old planned entry remain usable.
      await page.goto('http://127.0.0.1:5173/learn/topic/linux-basics-filesystems-processes');
      await page.locator('.linux-lesson').waitFor();
      assert.equal(await page.locator('.lesson-lab').count(), 4);
      assert.ok(await page.locator('.reader-complete').isEnabled());
      await page.getByRole('button', { name: /Mark as complete/ }).click();
      assert.match(await page.locator('.reader-complete').innerText(), /Completed/);
      await page.locator('.reader-complete').click();
      const oldPlanned = Object.values(topicCatalogue).find(t => !t.blueprint && t.trackId === 'data-structures-algorithms');
      if (oldPlanned) {
        await page.goto(`http://127.0.0.1:5173/learn/topic/${oldPlanned.id}`);
        await page.locator('.planned-lesson').waitFor();
        assert.match(await page.locator('.planned-lesson').innerText(), /full teaching plan and lesson are still to come/);
      }
      const llmIds = new Set(getPathTopicIds(learningPaths.find(p => p.id === 'llm-engineer')));
      const movedBranch = Object.values(topicCatalogue).find(t => t.trackId === 'hardware-systems' && baseline.has(t.id) && !llmIds.has(t.id));
      assert.ok(movedBranch, 'Need an old specialist topic outside the focused LLM route');
      await page.goto(`http://127.0.0.1:5173/learn/path/llm-engineer/${movedBranch.id}`);
      await page.waitForURL(`**/learn/topic/${movedBranch.id}`);
      await page.getByRole('heading', { name: movedBranch.title, exact: true, level: 1 }).waitFor();
      // Module navigation includes its recorded bridges in the same order.
      const moduleIds = getPathTopicIds(['programming-scientific-computing']);
      const fileTopic = Object.values(topicCatalogue).find(t => t.trackId === 'programming-scientific-computing' && t.title.startsWith('Scientific File Formats'));
      assert.ok(fileTopic);
      await page.goto(`http://127.0.0.1:5173/learn/track/programming-scientific-computing/${fileTopic.id}`);
      await page.getByRole('heading', { name: fileTopic.title, exact: true, level: 1 }).waitFor();
      await page.locator('.reader-footer__next').click();
      const moduleNext = moduleIds[moduleIds.indexOf(fileTopic.id) + 1];
      await page.waitForURL(`**/${moduleNext}`);
      await page.locator('.reader-topic.is-current').filter({ hasText: topicCatalogue[moduleNext].title }).waitFor();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(`${dir}/browser-results.json`, JSON.stringify({ results, errors }, null, 2));
    console.log('PASS: 7 paths; both new routes and detailed outlines at 1440/390; prerequisites, sidebar/next navigation, source links, keyboard disclosure, honest completion state, original Linux labs/progress, no page errors or document overflow.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
