const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const label = process.argv[2] || 'before';
if (!/^[a-z][a-z0-9-]*$/.test(label)) throw new Error('Use a short alphanumeric output label.');
const base = process.env.PERFORMANCE_BASE_URL || 'http://127.0.0.1:4173';
const repeats = Number(process.env.PERFORMANCE_REPEATS || 3);
const routes = [
  { name: 'portfolio', path: '/', ready: 'h1' },
  { name: 'learn-hub', path: '/learn', ready: '.workspace-hero' },
  { name: 'python', path: '/learn/path/full-curriculum/python-basics-types-control-flow-functions-modules?module=programming-scientific-computing', ready: '.lesson-intro' },
  { name: 'planned-trees', path: '/learn/path/full-curriculum/trees-binary-search-trees?module=data-structures-algorithms', ready: '.planned-lesson' },
  { name: 'dsa-arrays', path: '/learn/path/full-curriculum/arrays-strings-hash-maps?module=data-structures-algorithms', ready: '.lesson-intro' },
  { name: 'older-tokenization', path: '/learn/path/full-curriculum/byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram?module=large-language-models', ready: '.reader-article h2' },
];
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const samples = [];
  const dist = path.resolve('dist');
  const report = {
    label, measuredAt: new Date().toISOString(), base,
    methodology: {
      browser: browser.version(), viewport: { width: 1440, height: 1000 }, repeats,
      cache: 'Fresh browser context for every route/run; CDP HTTP cache disabled; no persisted cookies/storage. One browser process, normal local OS caches, no network/CPU throttling.',
      timing: 'Navigation start through visible route readiness, then a fixed 1000ms observation window. PerformanceNavigationTiming and long-task entries are local lab observations, not field Core Web Vitals.',
      bytes: 'Same-origin JS/CSS resource entries: encoded body, decoded body, and transfer size including HTTP overhead. Vite preview serves the local production build; external font requests are listed separately, not counted in JS/CSS totals.',
      limitations: 'Not cold-device timing, not deployment CDN/compression timing, not an observed-user study. Readiness and long tasks depend on host contention and browser/network conditions. No performance ranking is inferred from a single run.',
    },
    build: fs.existsSync(dist) ? fs.readdirSync(path.join(dist, 'assets')).filter(n => /\.(js|css)$/.test(n)).map(name => ({ name, bytes: fs.statSync(path.join(dist, 'assets', name)).size })).sort((a, b) => b.bytes - a.bytes) : [],
    samples,
  };
  try {
    for (const route of routes) {
      for (let run = 1; run <= repeats; run++) {
        const context = await browser.newContext({ viewport: report.methodology.viewport });
        const page = await context.newPage();
        const cdp = await context.newCDPSession(page);
        await cdp.send('Network.enable');
        await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(() => {
          performance.setResourceTimingBufferSize(5000);
          window.__lessonLongTasks = [];
          new PerformanceObserver(list => {
            window.__lessonLongTasks.push(...list.getEntries().map(entry => ({ startTime: entry.startTime, duration: entry.duration })));
          }).observe({ type: 'longtask', buffered: true });
        });
        await page.goto(base + route.path, { waitUntil: 'domcontentloaded', timeout: 60000 });
        await page.locator(route.ready).first().waitFor({ timeout: 60000 });
        const readyMs = await page.evaluate(() => performance.now());
        await page.waitForTimeout(1000);
        const metrics = await page.evaluate(() => {
          const entries = performance.getEntriesByType('resource').map(entry => ({
            url: entry.name, initiator: entry.initiatorType,
            transferBytes: entry.transferSize, encodedBodyBytes: entry.encodedBodySize,
            decodedBodyBytes: entry.decodedBodySize, durationMs: entry.duration,
            sameOrigin: new URL(entry.name).origin === location.origin,
          }));
          const nav = performance.getEntriesByType('navigation')[0];
          return {
            resources: entries, navigation: nav ? {
              responseEndMs: nav.responseEnd, domContentLoadedEndMs: nav.domContentLoadedEventEnd,
              loadEventEndMs: nav.loadEventEnd, documentTransferBytes: nav.transferSize,
            } : null,
            modulePreloads: [...document.querySelectorAll('link[rel="modulepreload"]')].map(n => n.href),
            observedUntilMs: performance.now(), longTasks: window.__lessonLongTasks,
          };
        });
        const js = metrics.resources.filter(r => r.sameOrigin && /\.js(?:\?|$)/.test(r.url));
        const css = metrics.resources.filter(r => r.sameOrigin && /\.css(?:\?|$)/.test(r.url));
        const total = list => ({ count: list.length, transferBytes: list.reduce((n, r) => n + r.transferBytes, 0), encodedBodyBytes: list.reduce((n, r) => n + r.encodedBodyBytes, 0), decodedBodyBytes: list.reduce((n, r) => n + r.decodedBodyBytes, 0) });
        assert.equal(errors.length, 0, `${route.name} page errors: ${errors.join('; ')}`);
        samples.push({ name: route.name, path: route.path, run, readyMs, js: total(js), css: total(css), longTaskCount: metrics.longTasks.length, longTaskTotalMs: metrics.longTasks.reduce((n, task) => n + task.duration, 0), ...metrics, errors });
        console.log(`${label} ${route.name} #${run}: JS ${total(js).encodedBodyBytes} bytes/${js.length} requests; CSS ${total(css).encodedBodyBytes} bytes; ready ${Math.round(readyMs)}ms; long tasks ${metrics.longTasks.length}`);
        await context.close();
      }
    }
    report.summary = routes.map(route => {
      const items = samples.filter(s => s.name === route.name);
      return { name: route.name, path: route.path, runs: items.length,
        jsEncodedBodyBytes: median(items.map(s => s.js.encodedBodyBytes)),
        jsTransferBytes: median(items.map(s => s.js.transferBytes)), jsRequests: median(items.map(s => s.js.count)),
        cssEncodedBodyBytes: median(items.map(s => s.css.encodedBodyBytes)), cssRequests: median(items.map(s => s.css.count)),
        medianReadyMs: median(items.map(s => s.readyMs)), medianLongTaskCount: median(items.map(s => s.longTaskCount)),
        medianLongTaskTotalMs: median(items.map(s => s.longTaskTotalMs)),
      };
    });
    fs.mkdirSync('scratch/learning-performance', { recursive: true });
    fs.writeFileSync(`scratch/learning-performance/${label}.json`, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report.summary, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
