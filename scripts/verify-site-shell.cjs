// Focused production checks for site navigation; no lesson numerical reruns.
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const base = process.env.SITE_BASE_URL || 'http://127.0.0.1:4195';
const output = path.resolve('docs/engineering/evidence/site-shell');
const manifest = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
const report = { checkedAt: new Date().toISOString(), base, checks: [], screenshots: [], errors: [] };
const sources = ['index.html', 'src/main.jsx', 'src/index.css', 'src/portfolio/Portfolio.jsx', 'src/shared/layout/SiteHeader.jsx', 'src/app/SiteRoutes.jsx', 'src/home/Home.jsx', 'src/app/navigation.js', 'src/shared/layout/site-shell.css', 'src/home/home.css', 'src/learn/components/LearningNav.jsx', 'src/learn/learning-base.css', 'src/learn/Reader.jsx'];

function dependencyFiles(entry, files = new Set()) {
  const item = manifest[entry];
  assert.ok(item, `Missing manifest entry ${entry}`);
  if (files.has('/' + item.file)) return files;
  files.add('/' + item.file);
  for (const file of item.css || []) files.add('/' + file);
  for (const imported of item.imports || []) dependencyFiles(imported, files);
  return files;
}

async function open(page, route, selector) {
  await page.goto(base + route, { waitUntil: 'domcontentloaded' });
  await page.locator(selector).first().waitFor();
  await page.evaluate(() => document.fonts.ready);
}

async function layout(page, label, entirePage = true) {
  const bounds = await page.evaluate(() => {
    const header = document.querySelector('.site-header');
    const overlapping = [...header.children].slice(1).some((element, index) => {
      const previous = header.children[index].getBoundingClientRect();
      const current = element.getBoundingClientRect();
      return current.top < previous.bottom - 1 && current.left < previous.right - 1;
    });
    return {
      width: innerWidth, scrollWidth: document.documentElement.scrollWidth, overlapping,
      headerOverflow: header.scrollWidth > header.clientWidth + 1,
      clippedControls: [...header.querySelectorAll('button, a')].filter(element => {
        if (element.closest('.portfolio-nav')) return false; // Intentionally scrollable local row.
        const rect = element.getBoundingClientRect();
        return rect.width && (rect.left < 0 || rect.right > innerWidth + 1);
      }).map(element => element.textContent.trim()),
    };
  });
  assert.equal(bounds.overlapping, false, `${label}: header groups overlap`);
  assert.equal(bounds.headerOverflow, false, `${label}: header overflow`);
  assert.deepEqual(bounds.clippedControls, [], `${label}: controls clipped`);
  if (entirePage) assert.ok(bounds.scrollWidth <= bounds.width + 1, `${label}: page overflow`);
  report.checks.push({ case: label, ...bounds });
}

async function capture(page, name, fullPage = false) {
  await page.screenshot({ path: path.join(output, name), fullPage });
  report.screenshots.push(name);
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1040 }, reducedMotion: 'reduce' });
    await require('./lib/lesson-browser-fonts.cjs')(context);
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    const requests = [];
    page.on('request', request => requests.push(new URL(request.url()).pathname));

    await open(page, '/', '.home-destinations');
    assert.equal(await page.locator('h1').innerText(), 'Ronak Sharma.');
    const shortcuts = page.getByRole('navigation', { name: 'Start learning' });
    assert.deepEqual(await shortcuts.locator('a').evaluateAll(links => links.map(link => link.getAttribute('href'))), ['/learn/paths', '/learn/projects', '/learn/catalogue']);
    const allowedHome = dependencyFiles('index.html');
    assert.deepEqual(requests.filter(file => file.startsWith('/assets/') && /\.(js|css)$/.test(file) && !allowedHome.has(file)), [], 'Home loaded an unrelated section');
    assert.ok(![...allowedHome].some(file => /LearningNav|Portfolio|Reader|catalogue/i.test(file)), 'Home imports section content');
    assert.equal(await page.locator('canvas').count(), 0);
    assert.equal(await page.locator('.home-destinations .home-destination--articles').getAttribute('href'), '/articles');
    report.homePayload = await page.evaluate(() => performance.getEntriesByType('resource').filter(item => new URL(item.name).origin === location.origin && /\.(js|css)$/.test(new URL(item.name).pathname)).map(item => ({ file: new URL(item.name).pathname, decodedBytes: item.decodedBodySize, encodedBytes: item.encodedBodySize })));
    report.checks.push({ case: 'Home loads only the shared shell and its own presentation; no curriculum, lesson, lab or portfolio body', passed: true });
    await capture(page, 'home-desktop.png', true);

    await page.keyboard.press('Tab');
    assert.equal(await page.locator(':focus').innerText(), 'Skip to content');
    await page.keyboard.press('Enter');
    assert.equal(await page.locator(':focus').getAttribute('id'), 'site-main');
    await page.locator('.home-destination--portfolio').click();
    await page.locator('#portfolio-main').waitFor();
    assert.equal(new URL(page.url()).pathname, '/portfolio');
    const portfolioSwitch = page.getByRole('button', { name: 'Browse site, current section: Portfolio' });
    await portfolioSwitch.click();
    await page.getByRole('navigation', { name: 'Site sections' }).getByRole('link', { name: 'Learn', exact: false }).click();
    await page.locator('.workspace-hero').waitFor();
    const learnSwitch = page.getByRole('button', { name: 'Browse site, current section: Learn' });
    await learnSwitch.focus();
    await page.keyboard.press('Enter');
    assert.equal(await learnSwitch.getAttribute('aria-expanded'), 'true');
    await page.keyboard.press('Escape');
    assert.equal(await learnSwitch.getAttribute('aria-expanded'), 'false');
    assert.ok(await learnSwitch.evaluate(element => element === document.activeElement));
    await learnSwitch.click();
    await page.locator('.workspace-hero h1').click();
    assert.equal(await learnSwitch.getAttribute('aria-expanded'), 'false');
    await learnSwitch.click();
    await page.getByRole('navigation', { name: 'Learning navigation' }).getByRole('link', { name: 'Paths', exact: true }).focus();
    assert.equal(await learnSwitch.getAttribute('aria-expanded'), 'false');
    report.checks.push({ case: 'Home entrances, cross-section navigation, skip link, keyboard disclosure, Escape/focus return, outside click and focus dismissal', passed: true });

    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      for (const [route, selector] of [['/', '.home-destinations'], ['/learn', '.workspace-hero'], ['/learn/paths', '.workspace-path'], ['/portfolio', '#portfolio-main']]) {
        await open(page, route, selector);
        await layout(page, `${route} at ${width}px`, route !== '/portfolio');
      }
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await open(page, '/', '.home-destinations');
    await capture(page, 'home-mobile.png', true);
    await open(page, '/learn', '.workspace-hero');
    await learnSwitch.click();
    await capture(page, 'section-switcher-mobile.png');

    await page.setViewportSize({ width: 1440, height: 1040 });
    for (const anchor of ['about', 'experience', 'projects', 'architecture', 'journey', 'outputs', 'contact']) {
      await open(page, '/#' + anchor, '#' + anchor);
      assert.equal(new URL(page.url()).pathname, '/portfolio');
      assert.equal(new URL(page.url()).hash, '#' + anchor);
      const top = await page.locator('#' + anchor).evaluate(element => element.getBoundingClientRect().top);
      assert.ok(top >= 56 && top < 350, `Legacy ${anchor} anchor did not land below the header: ${top}`);
    }
    await open(page, '/#explore', '.home-destinations');
    assert.equal(new URL(page.url()).pathname, '/', 'Home anchor was incorrectly redirected');
    report.checks.push({ case: 'All seven old portfolio anchors redirect and scroll correctly; home anchor stays home', passed: true });

    const lessonProgress = JSON.stringify({ 'perceptrons-neurons-activation-functions': true });
    const projectProgress = JSON.stringify({ 'typed-decision-model/define': true });
    await page.evaluate(([lesson, project]) => { localStorage.setItem('kd-progress', lesson); localStorage.setItem('learning-project-progress-v1', project); }, [lessonProgress, projectProgress]);
    await open(page, '/', '.home-destinations');
    for (const title of ['Build a typed decision model', 'What makes a neuron learn?', 'Bloospace']) {
      await page.getByRole('link').filter({ has: page.getByRole('heading', { name: title, exact: true }) }).click();
      if (title === 'Build a typed decision model') await page.locator('.tdp-stage').waitFor();
      else if (title === 'What makes a neuron learn?') {
        await page.locator('.reader-article').last().waitFor();
        assert.ok((await page.title()).includes('Perceptrons'));
      } else await page.locator('#projects').waitFor();
      await page.getByRole('link', { name: 'ronak.ai — Home', exact: true }).click();
      await page.locator('.home-destinations').waitFor();
      assert.equal(await page.evaluate(() => scrollY), 0);
    }
    await open(page, '/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals', '.reader-header h1');
    assert.ok((await page.locator('.reader-header h1').innerText()).includes('Sequence'));
    await layout(page, 'Existing sequence lesson route');
    await open(page, '/learn/projects/typed-decision-model/training', '.tdp-stage');
    await layout(page, 'Existing project training route');
    assert.deepEqual(await page.evaluate(() => [localStorage.getItem('kd-progress'), localStorage.getItem('learning-project-progress-v1')]), [lessonProgress, projectProgress]);
    await open(page, '/does-not-exist', '.site-recovery');
    await page.getByRole('link', { name: 'Back to home', exact: false }).click();
    await page.locator('.home-destinations').waitFor();
    report.checks.push({ case: 'All curated links, existing lesson and project deep links, progress conservation, page scroll reset and unknown-address recovery', passed: true });

    const failureContext = await browser.newContext();
    await require('./lib/lesson-browser-fonts.cjs')(failureContext);
    const failedPage = await failureContext.newPage();
    const portfolioFile = '/' + manifest['src/portfolio/Portfolio.jsx'].file;
    await failedPage.route('**' + portfolioFile, route => route.abort());
    await open(failedPage, '/portfolio', '.site-recovery');
    assert.ok(await failedPage.getByRole('button', { name: 'Reload page', exact: false }).isVisible());
    await failedPage.unroute('**' + portfolioFile);
    await failedPage.getByRole('button', { name: 'Reload page', exact: false }).click();
    await failedPage.locator('#portfolio-main').waitFor();
    await failureContext.close();
    report.checks.push({ case: 'Section import failure provides working reload recovery', passed: true });
    assert.deepEqual(report.errors, [], 'Unexpected browser errors');
    report.sourceHashes = Object.fromEntries(sources.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
    report.passed = true;
    console.log(`PASS: ${report.checks.length} site checks; ${report.screenshots.length} screenshot candidates retained.`);
  } catch (error) {
    report.passed = false;
    report.errors.push(error.stack);
    console.error(error.stack);
    process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'browser-review.json'), JSON.stringify(report, null, 2) + '\n');
  }
})();
