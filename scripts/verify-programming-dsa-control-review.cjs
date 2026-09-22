// Bounded interaction review for the 39 implemented Programming and DSA lessons.
// Checks representative process controls and every initially visible range; it
// does not claim exhaustive coverage of all inputs or hidden practice states.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read = path => JSON.parse(fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const requested = process.argv.find(argument => argument.startsWith('--topics='))?.slice(9).split(',');
const topics = read('docs/teaching/evidence/live-exploration-runtime-scope.json').topics.filter(topic => ['programming-scientific-computing', 'data-structures-algorithms'].includes(topic.module) && (!requested || requested.includes(topic.id)));
assert.equal(topics.length, requested?.length ?? 39);
const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
const receipt = process.env.CONTROL_REVIEW_RECEIPT || `docs/teaching/evidence/programming-dsa-control-review${requested ? '-focused' : ''}.json`;
const evidence = { capturedAt: new Date().toISOString(), base, manifestHash: hash('dist/.vite/manifest.json'), scope: topics.map(topic => topic.id), records: [], errors: [], captures: [], passed: false, sourceHashes: Object.fromEntries([...new Set(topics.flatMap(topic => topic.dependencies))].map(path => [path, hash(path)])) };

function inspectPage() {
  const controls = [...document.querySelectorAll('.reader-article button,.reader-article select,.reader-article input,.reader-article textarea')].filter(node => node.checkVisibility());
  const name = node => node.getAttribute('aria-label') || [...(node.labels || [])].map(label => label.textContent.trim()).join(' ') || node.textContent.trim();
  const unlabeled = controls.filter(node => node.tagName !== 'BUTTON' && !name(node) && !node.hasAttribute('aria-labelledby')).map(node => node.outerHTML.slice(0, 250));
  const escaped = [...document.querySelectorAll('.reader-article *')].filter(node => node.checkVisibility()).filter(node => {
    const rect = node.getBoundingClientRect();
    if (rect.right <= innerWidth + 2 && rect.left >= -2) return false;
    let parent = node.parentElement;
    while (parent && !parent.matches('.reader-article')) {
      if (['auto', 'scroll', 'hidden', 'clip'].includes(getComputedStyle(parent).overflowX)) return false;
      parent = parent.parentElement;
    }
    return true;
  }).slice(0, 10).map(node => ({ tag: node.tagName, text: node.textContent.slice(0, 90) }));
  const checkboxTargets = controls.filter(node => node.type === 'checkbox').map(node => {
    const box = node.getBoundingClientRect(), label = node.labels?.[0], labelBox = label?.getBoundingClientRect();
    return { name: name(node), glyph: [box.width, box.height], label: labelBox ? [labelBox.width, labelBox.height] : null };
  });
  return { controls: controls.length, unlabeled, escaped, checkboxTargets, documentWidth: document.documentElement.scrollWidth, width: innerWidth };
}

function outputSnapshot(control) {
  const root = control.closest('section') || control.closest('figure') || control.parentElement.parentElement;
  const clone = root.cloneNode(true);
  clone.querySelectorAll('input,select,textarea,button,label,summary').forEach(node => node.remove());
  return JSON.stringify({ text: clone.textContent.replace(/\s+/g, ' ').trim(), graphics: [...clone.querySelectorAll('svg')].map(node => node.outerHTML).join('') });
}

const labels = {
  'bash-scripting-command-line-automation': 'Follow the expansion into argument boundaries',
  'arrays-strings-hash-maps': 'Build a byte and watch which 1 disappears at each step.',
  'binary-search-sorting-two-pointer-patterns': 'Choose a rate to compare its total slots with the available budget.',
  'greedy-algorithms-exchange-arguments': 'Compare which individual completion gets later and what happens to maximum lateness.',
  'dynamic-programming-states-transitions-optimization': 'Then reopen one exit and follow the changed route.',
  'segment-trees-fenwick-trees-range-queries': 'For prefix length 7, follow 7→6→4→0.',
  'shortest-paths-spanning-trees-topological-ordering': 'Make A→B cost 0 and follow its extraction.',
  'string-matching-prefix-functions-rolling-hashes': 'Step through a fallback and follow the next shorter candidate.',
  'network-flow-minimum-cuts-bipartite-matching': 'Preview the residual route and inspect its bottleneck, then send that amount.',
  'computational-geometry-robust-predicates-convex-hulls': 'use the coordinate sliders to move one endpoint and watch the classification change.'
};

(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'msedge', headless: true });
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
    await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.abort());
    let index = 0;
    await Promise.all([0, 1].map(async () => {
      const page = await context.newPage();
      page.on('pageerror', error => evidence.errors.push(error.message));
      while (index < topics.length) {
        const topic = topics[index++], record = { id: topic.id, ranges: [], interactions: [] };
        try {
          await page.setViewportSize({ width: 1366, height: 1000 });
          await page.goto(`${base}/learn/path/full-curriculum/${topic.id}?module=${topic.module}`, { waitUntil: 'domcontentloaded' });
          await page.locator('.reader-article h2,.reader-article h3').first().waitFor();
          await page.evaluate(() => document.fonts.ready);
          record.desktop = await page.evaluate(inspectPage);
          if (labels[topic.id]) assert((await page.locator('.reader-article').allTextContents()).join(' ').includes(labels[topic.id]), `${topic.id}: revised exploration wording is rendered`);
          const ranges = page.locator('.reader-article input[type=range]:visible');
          for (let rangeIndex = 0; rangeIndex < await ranges.count(); rangeIndex++) {
            const control = ranges.nth(rangeIndex);
            const name = await control.evaluate(node => node.getAttribute('aria-label') || [...node.labels].map(label => label.textContent).join(' '));
            const initial = await control.inputValue();
            const before = await control.evaluate(outputSnapshot);
            await control.focus();
            await page.keyboard.press('Home');
            const minimum = await control.inputValue();
            await page.keyboard.press('End');
            const maximum = await control.inputValue();
            assert.notEqual(minimum, maximum, `${name}: Home and End reach distinct values`);
            const keyboardChangesOutput = before !== await control.evaluate(outputSnapshot);
            await control.evaluate(node => node.scrollIntoView({ block: 'center', behavior: 'instant' }));
            await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            const rect = await control.boundingBox();
            assert(rect.height >= 24, `${name}: useful pointer hit area`);
            // Use the real rendered range and native pointer events, not fill().
            await page.mouse.move(rect.x + rect.width - 10, rect.y + rect.height / 2);
            await page.mouse.down();
            await page.mouse.move(rect.x + rect.width * 0.3, rect.y + rect.height / 2, { steps: 6 });
            await page.mouse.up();
            const dragged = await control.inputValue();
            assert.notEqual(dragged, maximum, `${name}: actual pointer drag changes the value`);
            record.ranges.push({ name, initial, minimum, maximum, dragged, keyboardChangesOutput });
            await control.fill(initial);
          }
          // Prefer process advancement to re-applying unchanged drafts. Some
          // configuration choices legitimately preserve a current first frame.
          const roots = page.locator('.reader-article section').filter({ has: page.locator('button,select') });
          for (let rootIndex = 0; rootIndex < await roots.count(); rootIndex++) {
            const root = roots.nth(rootIndex);
            if (!await root.isVisible() || await root.locator('section').count()) continue;
            const candidates = root.getByRole('button', { name: /^(Next|Advance|Step|Preview|Take |Swap|Reverse|Toggle)/i }).filter({ visible: true });
            let control;
            for (let buttonIndex = 0; buttonIndex < await candidates.count(); buttonIndex++) if (await candidates.nth(buttonIndex).isEnabled()) { control = candidates.nth(buttonIndex); break; }
            if (!control) continue;
            const name = await control.innerText(), before = await control.evaluate(outputSnapshot);
            await control.click();
            const outputChanged = before !== await control.evaluate(outputSnapshot);
            record.interactions.push({ name, outputChanged });
          }
          if (topic.id === 'disjoint-sets-union-find') {
            const ring = page.getByRole('button', { name: 'Load ring around center', exact: true });
            assert.equal(await ring.count(), 1);
            await ring.click();
            assert((await page.locator('.reader-article').allTextContents()).join(' ').includes('Open that cell and inspect the count; all four neighbors already connect.'));
          }
          if (topic.id === 'external-memory-algorithms-b-trees-i-o-complexity') {
            // These two parameter edits can preserve the initial process frame.
            // Advance their explicitly staged operation before judging feedback.
            const tree = page.getByRole('region', { name: 'B-tree page repairs', exact: true });
            await tree.locator('input[type=range]').fill('10');
            await tree.getByRole('button', { name: 'Search final tree', exact: true }).click();
            assert.match(await tree.locator('p[role="status"]').innerText(), /10 is present/);
            await tree.locator('input[type=range]').fill('99');
            await tree.getByRole('button', { name: 'Search final tree', exact: true }).click();
            assert.match(await tree.locator('p[role="status"]').innerText(), /99 is absent/);
            const range = page.getByRole('region', { name: 'B-plus linked leaf range', exact: true });
            await range.getByLabel('End of inclusive range', { exact: true }).fill('11');
            await range.getByRole('button', { name: 'Finish range page', exact: true }).click();
            assert.equal(await range.locator('.external-facts dd').first().innerText(), '11');
            await range.getByLabel('End of inclusive range', { exact: true }).fill('35');
            await range.getByRole('button', { name: 'Finish range page', exact: true }).click();
            assert.equal(await range.locator('.external-facts dd').first().innerText(), '11, 14, 17, 20, 23, 26, 29, 32, 35');
            record.stagedParameterChecks = 'Final-tree search distinguishes present10/absent99; [10,11] returns11 and [10,35] returns all nine expected records after finishing the page trace.';
          }
          await page.setViewportSize({ width: 320, height: 900 });
          record.phone = await page.evaluate(inspectPage);
          for (const dimensions of ['desktop', 'phone']) {
            assert.deepEqual(record[dimensions].unlabeled, [], `${topic.id}: ${dimensions} control labels`);
            assert.deepEqual(record[dimensions].escaped, [], `${topic.id}: ${dimensions} escaped content`);
            assert(record[dimensions].documentWidth <= record[dimensions].width + 1, `${topic.id}: ${dimensions} page width`);
          }
          if (['backtracking-divide-and-conquer', 'binary-search-sorting-two-pointer-patterns'].includes(topic.id)) {
            const name = topic.id.startsWith('backtracking') ? 'Divide tree and indexed array; scroll horizontally if needed' : 'Closed intervals on a common axis; scroll horizontally to inspect every endpoint';
            const scroller = page.getByRole('region', { name, exact: true });
            await scroller.evaluate(node => { node.scrollLeft = 0; node.scrollIntoView({ block: 'center', behavior: 'instant' }); });
            assert(await scroller.evaluate(node => node.scrollWidth > node.clientWidth), `${name}: intentional local overflow`);
            await scroller.focus();
            await page.keyboard.press('ArrowRight');
            await page.waitForFunction(label => document.querySelector(`[aria-label="${label}"]`).scrollLeft > 0, name);
            if (topic.id.startsWith('backtracking')) {
              const finalLeaf = scroller.getByRole('button', { name: 'Inspect range 5 through 6 exclusive', exact: true });
              await finalLeaf.focus();
              await page.keyboard.press('Enter');
              assert.equal(await finalLeaf.getAttribute('aria-pressed'), 'true');
              assert.match(await page.locator('.bd-status').last().innerText(), /Range \[5, 6\)/);
            }
            record.localScrollerKeyboard = 'Named region pans with ArrowRight; hierarchy final leaf can be focused and activated when present.';
          }
          record.checkboxLabelClicks = 0;
          const checks = page.locator('.reader-article input[type=checkbox]:visible');
          for (let checkboxIndex = 0; checkboxIndex < await checks.count(); checkboxIndex++) {
            const checkbox = checks.nth(checkboxIndex);
            if (!await checkbox.isEnabled()) continue;
            const labelHandle = await checkbox.evaluateHandle(node => node.labels?.[0]);
            const label = labelHandle.asElement();
            assert(label, 'A visible checkbox has a native associated label');
            const before = await checkbox.isChecked();
            await label.click();
            assert.notEqual(await checkbox.isChecked(), before, 'Clicking the full label toggles the checkbox');
            await label.click();
            assert.equal(await checkbox.isChecked(), before, 'A second label click restores the checkbox');
            record.checkboxLabelClicks++;
            await labelHandle.dispose();
          }
          const captureTargets = {
            'computational-geometry-robust-predicates-convex-hulls': () => page.getByRole('region', { name: 'Closed segment intersection', exact: true }),
            'graphs-representations-bfs-dfs': () => page.locator('[data-lab="grid-wavefront"]'),
            'algorithm-correctness-loop-invariants-termination': () => page.locator('.proof-obligations'),
            'backtracking-divide-and-conquer': () => page.getByText(/^A simpler divide-and-conquer version scans outward/),
            'complexity-analysis-recursion': () => page.getByRole('region', { name: 'Recurrence level investigation', exact: true }),
            'binary-search-sorting-two-pointer-patterns': () => page.getByRole('figure', { name: 'Closed interval union from one through five; six through eight is separate', exact: true })
          };
          if (captureTargets[topic.id]) {
            const target = captureTargets[topic.id]();
            const path = `docs/teaching/evidence/screenshots/control-review-${topic.id}-phone.png`;
            await target.scrollIntoViewIfNeeded();
            const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
            await target.screenshot({ path });
            await style.evaluate(node => node.remove());
            evidence.captures.push({ path, sha256: hash(path) });
          }
          await page.setViewportSize({ width: 1366, height: 1000 });
        } catch (error) { record.error = error.message; }
        evidence.records.push(record);
      }
      await page.close();
    }));
    assert.equal(evidence.records.length, topics.length);
    assert.deepEqual(evidence.records.filter(record => record.error), []);
    assert.deepEqual(evidence.errors, []);
    evidence.passed = true;
  } catch (error) { evidence.failure = error.message; process.exitCode = 1; }
  finally {
    await browser?.close();
    evidence.records.sort((a, b) => a.id.localeCompare(b.id));
    if (!process.argv.includes('--no-evidence')) fs.writeFileSync(receipt, JSON.stringify(evidence, null, 2) + '\n');
    console.log(JSON.stringify({ passed: evidence.passed, routes: evidence.records.length, ranges: evidence.records.reduce((sum, record) => sum + record.ranges.length, 0), processActions: evidence.records.reduce((sum, record) => sum + record.interactions.length, 0), failures: evidence.records.filter(record => record.error).map(record => ({ id: record.id, error: record.error })), receipt }, null, 2));
  }
})();
