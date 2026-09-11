const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = 'scratch/abstract-algebra-independent';
const payload = JSON.parse(fs.readFileSync(`${directory}/payload.json`, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
(async () => {
  for (const source of payload.sources) assert.equal(hash(source.path), source.sha256);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440,390,320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/abstract-algebra-groups-symmetry-actions?module=math-foundations');
      await page.locator('.algebra-cayley').waitFor(); await page.evaluate(() => document.fonts.ready);
      const record = { width, captures: [] };
      for (const [selector,name] of [['.algebra-cayley','cayley'],['.algebra-action-graph','action']]) {
        const target = page.locator(selector);
        await target.scrollIntoViewIfNeeded();
        await target.evaluate(node => scrollTo({ top: scrollY+node.getBoundingClientRect().top-100, behavior: 'instant' }));
        if (name === 'cayley') {
          const endpoints = await target.locator('path[d*=" 220H"]').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('d').split('H')[1])));
          assert.deepEqual(endpoints, [105,245,385]);
          assert(endpoints.every((endpoint,index) => endpoint > 70 + index*140 + 26 + 3));
          record.reverseArrowEndpointsOutsideCircles = endpoints;
          const region = target.locator('..'); await region.focus();
          const before = await region.evaluate(node => node.scrollLeft);
          await page.keyboard.press('ArrowRight'); await page.waitForTimeout(250);
          const after = await region.evaluate(node => node.scrollLeft);
          if (width < 600) assert(after > before);
          Object.assign(record, { localScrollBefore: before, localScrollAfter: after });
        }
        await page.waitForTimeout(250);
        const rectangle = await target.evaluate(node => ({ top: node.getBoundingClientRect().top, height: node.getBoundingClientRect().height }));
        assert(rectangle.top >= 55 && rectangle.top < 300);
        const file = `${directory}/final-${name}-paint-${width}.png`;
        await page.screenshot({ path: file }); record.captures.push({ path: file, sha256: hash(file), rectangle });
      }
      records.push(record); await page.close();
    }
    for (const source of payload.sources) assert.equal(hash(source.path), source.sha256);
    fs.writeFileSync(`${directory}/graph-paint-results.json`,JSON.stringify({ checkedAt: new Date().toISOString(), status:'passed', sources:payload.sources, records, scope:'Focused replacement captures after inspecting blank/stale raster snapshots from keyboard End in the original harness. Uses local ArrowRight scrolling and observed stable element bounds; no production source change.' },null,2));
    console.log(JSON.stringify(records,null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
