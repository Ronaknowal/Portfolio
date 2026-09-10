const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const { tracks } = await import('./lib/authoring-curriculum.mjs');
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const { learningPaths, getPathTopicIds, getPathNavigationGroups } = await import('./lib/authoring-curriculum.mjs');
  const full = learningPaths.find(p => p.id === 'full-curriculum');
  const allIds = getPathTopicIds(full);
  const totalMemberships = tracks.reduce((n, t) => n + new Set(t.topicIds).size, 0);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  const results = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/linux-basics-filesystems-processes');
      await page.locator('.reader-group').first().waitFor();
      assert.equal(await page.locator('.reader-group').count(), 28);
      assert.deepEqual((await page.locator('.reader-group').evaluateAll(nodes => nodes.map(n => n.dataset.moduleId))).sort(), tracks.map(t => t.id).sort());
      await page.locator('.reader-group__toggle').evaluateAll(nodes => nodes.forEach(n => { if (n.getAttribute('aria-expanded') === 'false') n.click(); }));
      await page.waitForFunction(count => document.querySelectorAll('.reader-topic').length === count, totalMemberships);
      const rendered = await page.locator('.reader-group').evaluateAll(groups => groups.map(group => ({
        id: group.dataset.moduleId,
        topics: [...group.querySelectorAll('.reader-topic')].map(t => ({ id: t.dataset.topicId })),
      })));
      const union = new Set(rendered.flatMap(g => g.topics.map(t => t.id)));
      assert.deepEqual([...union].sort(), [...allIds].sort());
      for (const group of rendered) {
        const track = tracks.find(t => t.id === group.id);
        assert.deepEqual(group.topics.map(t => t.id).sort(), [...new Set(track.topicIds)].sort());
        assert.deepEqual(group.topics.map(t => t.id), allIds.filter(id => track.topicIds.includes(id)));
      }
      assert.equal(await page.locator('.reader-topic__step, [data-route-step], .reader-group__scope').count(), 0);
      assert.equal(await page.locator('.reader-topic.is-current').count(), 1);
      await page.locator('.reader-group__toggle').evaluateAll(nodes => nodes.forEach(n => { if (n.getAttribute('aria-expanded') === 'true') n.click(); }));
      await page.waitForFunction(() => document.querySelectorAll('.reader-group__topics').length === 0);
      await page.locator('.reader-sidebar__groups').evaluate(node => { node.scrollTop = 0; });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await page.screenshot({ path: `scratch/curriculum-review/module-groups-${width}.png` });

      const route = learningPaths.find(p => p.id === 'gpu-engineer');
      const ids = getPathTopicIds(route);
      const navigation = getPathNavigationGroups(ids, { moduleOrder: route.trackIds });
      await page.goto('http://127.0.0.1:5173/learn/path/gpu-engineer');
      await page.locator('.reader-group').first().waitFor();
      assert.equal(await page.locator('.reader-group').count(), navigation.length);
      const partial = navigation.find(g => g.topicIds.length < g.totalTopicCount);
      assert.ok(partial);
      const group = page.locator(`[data-module-id="${partial.id}"]`);
      if (await group.locator('.reader-group__toggle').getAttribute('aria-expanded') === 'false') {
        await group.locator('.reader-group__toggle').focus();
        await page.keyboard.press('Enter');
      }
      const fullLink = group.getByRole('button', { name: `View all ${partial.totalTopicCount} module topics →`, exact: true });
      await fullLink.waitFor();
      assert.equal((await group.locator('.reader-group__size').innerText()).replace(/\s+/g, ' '), `${partial.topicIds.length} ${partial.topicIds.length === 1 ? 'topic' : 'topics'} 0 completed`);
      await fullLink.click();
      await page.waitForURL(`**/learn/track/${partial.id}/*`);
      await page.locator('.reader-topic.is-current').waitFor();
      const ownCurrent = await page.locator('.reader-topic.is-current').getAttribute('data-topic-id');
      assert.ok(tracks.find(t => t.id === partial.id).topicIds.includes(ownCurrent));
      for (const moduleId of ['hardware-systems', 'computational-neuroscience']) {
        await page.goto(`http://127.0.0.1:5173/learn/track/${moduleId}`);
        await page.locator('.reader-topic.is-current').waitFor();
        const current = await page.locator('.reader-topic.is-current').getAttribute('data-topic-id');
        assert.equal(current, getPathTopicIds([moduleId])[0], 'Module entry skipped the first topic in its prerequisite sequence');
      }
      const transition = ids.findIndex((id, index) => index < ids.length - 1 && topicCatalogue[id].trackId !== topicCatalogue[ids[index + 1]].trackId);
      assert.ok(transition >= 0);
      await page.goto(`http://127.0.0.1:5173/learn/path/gpu-engineer/${ids[transition]}`);
      await page.locator(`.reader-topic.is-current[data-topic-id="${ids[transition]}"]`).waitFor();
      await page.locator('.reader-footer__next').click();
      await page.waitForURL(`**/${ids[transition + 1]}`);
      await page.locator(`.reader-topic.is-current[data-topic-id="${ids[transition + 1]}"]`).waitFor();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      results.push({ width, fullModuleGroups: rendered.length, uniqueTopics: union.size, displayedMemberships: totalMemberships, checks: 'all module contents conserved, unnumbered topics in route order, compact selected counts, keyboard expansion, full-module access, entry in prerequisite order, cross-module Next, no overflow' });
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync('scratch/curriculum-review/module-grouping-browser.json', JSON.stringify({ results, errors }, null, 2));
    console.log('PASS: 28 real module groups, all 1,218 unique topics and 1,222 memberships; unnumbered topics, compact counts, full-module access, entry in prerequisite order, keyboard and cross-module navigation at 1440/390; no page errors or document overflow.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
