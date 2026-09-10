const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const { tracks } = await import('./lib/authoring-curriculum.mjs');
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const sharedTopics = Object.values(topicCatalogue).filter(topic => topic.trackIds.length > 1);
  const uniqueTopicCount = Object.keys(topicCatalogue).length;
  const dir = 'scratch/curriculum-review';
  fs.mkdirSync(dir, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  const results = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto('http://127.0.0.1:5173/learn');
      await page.getByRole('tab', { name: 'Search catalogue', exact: true }).click();
      const moduleFilter = page.getByLabel('Module', { exact: true });
      await moduleFilter.waitFor();
      assert.equal(await moduleFilter.locator('option[value="all"]').innerText(), 'All modules');
      assert.equal(await moduleFilter.locator('option').count(), tracks.length + 1);
      await page.waitForFunction(expected => Number(document.querySelector('.catalogue-result-count').textContent.match(/[\d,]+/)[0].replaceAll(',', '')) === expected, uniqueTopicCount);

      const checkedModules = [];
      for (const track of tracks) {
        const expectedIds = [...new Set(track.topicIds)].sort();
        await moduleFilter.selectOption(track.id);
        await page.waitForFunction(expected => Number(document.querySelector('.catalogue-result-count').textContent.match(/[\d,]+/)[0].replaceAll(',', '')) === expected, expectedIds.length);
        while (await page.locator('.catalogue-more').count()) {
          const count = await page.locator('.topic-result').count();
          await page.locator('.catalogue-more').click();
          await page.waitForFunction(previous => document.querySelectorAll('.topic-result').length > previous, count);
        }
        const actualIds = await page.locator('.topic-result').evaluateAll(nodes => nodes.map(node => node.dataset.topicId).sort());
        assert.deepEqual(actualIds, expectedIds, `Catalogue membership mismatch in ${track.id} at ${width}px`);
        assert.equal(new Set(actualIds).size, actualIds.length, `Duplicate result in ${track.id}`);
        for (const topic of sharedTopics.filter(topic => topic.trackIds.includes(track.id))) {
          assert.equal(await page.locator(`.topic-result[data-topic-id="${topic.id}"]`).count(), 1, `Shared topic missing or duplicated in ${track.id}: ${topic.id}`);
        }
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Document overflow in ${track.id} at ${width}px`);
        checkedModules.push({ id: track.id, topics: actualIds.length });
      }

      // Search should compose with membership filtering, including secondary modules.
      for (const topic of sharedTopics) {
        await moduleFilter.selectOption(topic.trackIds[1]);
        await page.getByRole('searchbox', { name: 'Search the curriculum', exact: true }).fill(topic.title);
        await page.waitForFunction(id => {
          const results = document.querySelectorAll('.topic-result');
          return results.length === 1 && results[0].dataset.topicId === id;
        }, topic.id);
        await page.getByRole('searchbox', { name: 'Search the curriculum', exact: true }).fill('');
      }
      results.push({ width, modules: checkedModules, secondarySharedTopics: sharedTopics.map(topic => topic.id), checks: 'exact module membership counts and IDs, no duplicates, shared-topic search, module terminology, no overflow' });
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(`${dir}/catalogue-membership-browser.json`, JSON.stringify({ results, errors }, null, 2));
    console.log(`PASS: all ${tracks.length} module filters match their complete topic memberships; all ${sharedTopics.length} shared topics appear once in secondary modules; search, terminology and no overflow at 1440/390; no page errors.`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
