const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/implementation-depth-remediation-baseline.json'));
const manifest = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json'));
const reviewedPrograms = JSON.parse(fs.readFileSync('docs/teaching/implementation-depth/remediation-links-review.json'));
const expectedPrograms = Object.fromEntries(reviewedPrograms.topics.map(topic => [topic.id, topic.programSources]));
for (const [id, metadata] of [
  ['perceptrons-neurons-activation-functions', 'perceptron'],
  ['backpropagation-automatic-differentiation', 'backprop'],
  ['transfer-learning-fine-tuning-strategies', 'transfer-learning'],
]) {
  const source = fs.readFileSync(`src/learn/data/${metadata}-mechanism-program.js`, 'utf8');
  expectedPrograms[id] = [source.match(/"source":\s*"([^"]+)"/)[1]];
}
const inlineSections = {
  'multivariate-calculus-gradients': '#multivariate-calculus-code-route',
  'second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient': '#second-order-code-route',
  'stochastic-processes-markov-chains-brownian-motion-poisson': '#stochastic-processes-code-route',
};
const inlineCode = {
  'linear-logistic-regression': { heading: "Map the scratch penalty to scikit-learn's C", code: 'C=1 / (len(y) * penalty)' },
  'gaussian-processes-gp': { heading: 'Match this exact posterior to the library', code: 'library.log_marginal_likelihood()' },
};
const fonts = JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json', 'utf8').replace(/^\uFEFF/, ''));
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const requested = process.argv.slice(2);
const ids = requested.length ? requested : baseline.scope;
const directory = 'docs/teaching/evidence/screenshots/depth-remediation';
const receipt = 'docs/teaching/evidence/depth-remediation-browser.json';
fs.mkdirSync(directory, { recursive: true });
const previous = requested.length && fs.existsSync(receipt) ? JSON.parse(fs.readFileSync(receipt)) : { topics: {} };
const report = { ...previous, status: 'running', checkedAt: new Date().toISOString(), manifestHash: hash('dist/.vite/manifest.json') };
report.verifierHash = hash(__filename);
report.expectationsHash = crypto.createHash('sha256').update(JSON.stringify(expectedPrograms)).digest('hex');
// A subset cannot silently bless evidence from a different production build or checker.
for (const topic of Object.values(report.topics)) {
  const sourcesCurrent = Object.entries(topic.sourceHashes || {}).every(([file, digest]) => fs.existsSync(file) && hash(file) === digest);
  if (topic.manifestHash !== report.manifestHash || topic.verifierHash !== report.verifierHash || topic.expectationsHash !== report.expectationsHash || !sourcesCurrent) {
    topic.status = 'stale';
  }
}
const save = () => fs.writeFileSync(receipt, JSON.stringify(report, null, 2) + '\n');
save();

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const id of ids) {
      const item = { status: 'running', manifestHash: report.manifestHash, verifierHash: report.verifierHash, expectationsHash: report.expectationsHash, checks: [], captures: [], errors: [], sourceHashes: {} };
      report.topics[id] = item;
      save();
      const context = await browser.newContext({ viewport: { width: 1366, height: 950 }, reducedMotion: 'reduce' });
      try {
        await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
        await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.abort());
        const page = await context.newPage();
        const requests = [];
        page.on('request', request => requests.push(request.url()));
        page.on('pageerror', error => item.errors.push(error.message));
        await page.goto(`${base}/learn/path/full-curriculum/${id}`, { timeout: 60000 });
        await page.locator('main.reader-content .reader-article').first().waitFor();
        await page.locator('main.reader-content .reader-article h2').first().waitFor();
        await page.evaluate(() => document.fonts.ready);
        const programs = page.locator('.mechanism-program');
        const count = await programs.count();
        assert.ok(expectedPrograms[id], `No reviewed expectation for ${id}`);
        const displayedUrls = await programs.locator('a[download]').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
        assert.deepEqual([...displayedUrls].sort(), [...expectedPrograms[id]].sort(), `${id}: all reviewed code sections must be present`);
        if (!count) {
          if (inlineSections[id]) {
            assert.equal(await page.locator(inlineSections[id]).count(), 1, `${id}: reuse/boundary section`);
            assert.ok((await page.locator(inlineSections[id]).innerText()).length > 300);
          } else {
            assert.ok(inlineCode[id], `${id}: no explicit zero-program expectation`);
            await page.getByRole('heading', { name: inlineCode[id].heading, exact: true }).waitFor();
            assert.ok(await page.locator('.reader-article div[style*="white-space: pre"]').filter({ hasText: inlineCode[id].code }).count(), `${id}: inline library code`);
          }
        }
        for (let index = 0; index < count; index++) {
          const program = programs.nth(index);
          const download = program.locator('a[download]').first();
          const url = await download.getAttribute('href');
          assert.ok(url?.startsWith('/learn-assets/'), `${id}: canonical program URL`);
          assert.equal(requests.some(request => request.split('?')[0].endsWith(url)), false, `${id}: code fetched before opening`);
          const summary = program.locator('summary').first();
          await summary.focus();
          await summary.press('Enter');
          const code = program.locator('pre[aria-label^="Complete "]');
          await code.waitFor();
          const response = await context.request.get(base + url);
          assert.equal(response.status(), 200, url);
          const source = fs.readFileSync('public' + url, 'utf8').replace(/\r\n/g, '\n');
          assert.equal((await response.text()).replace(/\r\n/g, '\n'), source, `${id}: download bytes`);
          assert.equal((await code.textContent()).replace(/\r\n/g, '\n'), source, `${id}: displayed bytes`);
          assert.ok(requests.some(request => request.endsWith(url)), `${id}: requested after opening`);
          assert.equal(await code.getAttribute('tabindex'), '0');
          assert.match(await code.getAttribute('aria-label'), /Complete (Python|C\+\+) program/);
          item.sourceHashes['public' + url] = hash('public' + url);
          await summary.press('Enter');
          await code.waitFor({ state: 'detached' });
          item.checks.push(`${url}: lazy request, keyboard open/close, canonical display/download`);
        }
        for (const width of [1366, 390, 320]) {
          await page.setViewportSize({ width, height: 950 });
          const state = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth, errors: document.querySelectorAll('.katex-error,.lesson-load-error').length }));
          assert.ok(state.scroll <= width + 1, `${id} ${width}: ${JSON.stringify(state)}`);
          assert.equal(state.errors, 0);
          for (let index = 0; index < count; index++) {
            const program = programs.nth(index);
            const summary = program.locator('summary').first();
            await summary.click();
            const code = program.locator('pre[aria-label^="Complete "]');
            await code.waitFor();
            await code.focus();
            assert.equal(await code.evaluate(node => document.activeElement === node), true, `${id}: focused code region`);
            const canScroll = await code.evaluate(node => { node.scrollLeft = 0; return node.scrollWidth > node.clientWidth + 1; });
            if (canScroll) {
              await code.press('ArrowRight');
              await page.waitForFunction(node => node.scrollLeft > 0, await code.elementHandle(), { timeout: 3000 });
              item.checks.push(`${width}px ${index}: ArrowRight moved the focused code region`);
            }
            assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${id}: open code overflow at ${width}`);
            await page.mouse.move(0, 0);
            const style = await program.locator('a[download]').first().evaluate(node => ({ color: getComputedStyle(node).color, overflow: getComputedStyle(node.closest('.mechanism-program').querySelector('pre[aria-label^="Complete "]')).overflowX }));
            assert.equal(style.color, 'rgb(226, 181, 90)', `${id}: theme gold link`);
            assert.equal(style.overflow, 'auto');
            if (width === 320 && index === 0) {
              await summary.scrollIntoViewIfNeeded();
              const file = `${directory}/${id}-320.png`;
              await page.screenshot({ path: file });
              item.captures.push({ file, sha256: hash(file) });
            }
            await summary.click();
            await code.waitFor({ state: 'detached' });
          }
          if (!count && width === 320) {
            const heading = inlineSections[id] ? page.locator(inlineSections[id]) : page.getByRole('heading', { name: inlineCode[id].heading, exact: true });
            if (await heading.count()) await heading.scrollIntoViewIfNeeded();
            const file = `${directory}/${id}-320.png`;
            await page.screenshot({ path: file });
            item.captures.push({ file, sha256: hash(file) });
          }
          item.checks.push(`${width}px: page containment, math/runtime render, all new program scroll regions and links`);
        }
        assert.deepEqual(item.errors, []);
        const body = 'src/learn/data/' + manifest[id].replace(/^\.\//, '');
        item.sourceHashes[body] = hash(body);
        for (const path of ['src/learn/components/lesson-labs/MechanismProgram.jsx', 'src/learn/components/lesson-labs/mechanism-program.css']) item.sourceHashes[path] = hash(path);
        item.status = 'passed';
      } catch (error) {
        item.status = 'failed'; item.failure = error.stack; process.exitCode = 1;
      } finally {
        await context.close(); save();
      }
    }
    report.missingTopics = baseline.scope.filter(id => !report.topics[id]);
    report.status = !report.missingTopics.length && Object.values(report.topics).every(topic => topic.status === 'passed') ? 'passed' : 'failed';
    if (report.status !== 'passed') process.exitCode = 1;
  } catch (error) { report.status = 'failed'; report.failure = error.stack; process.exitCode = 1; }
  finally { await browser.close(); save(); }
  console.log(JSON.stringify({ status: report.status, topics: Object.keys(report.topics).length, failures: Object.entries(report.topics).filter(([, value]) => value.status !== 'passed').map(([id, value]) => ({ id, failure: value.failure })) }));
})();
