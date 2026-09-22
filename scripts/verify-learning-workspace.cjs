// Scoped production regression for the learning workspace and project reader.
// Run with PLAYWRIGHT_PACKAGE and LEARNING_BASE_URL against a fresh Vite build.
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');

const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const evidenceDirectory = path.resolve(process.env.LEARNING_WORKSPACE_EVIDENCE_DIR || 'docs/teaching/projects/evidence');
const manifest = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
const projectEntry = 'src/learn/data/projects/typed-decision-model/content.jsx';
const report = { checkedAt: new Date().toISOString(), base, checks: [], payloads: [], screenshots: [], errors: [] };

function dependencyClosure(entry, files = new Set()) {
  const item = manifest[entry];
  assert.ok(item, `Missing build entry ${entry}`);
  if (files.has('/' + item.file)) return files;
  files.add('/' + item.file);
  for (const stylesheet of item.css || []) files.add('/' + stylesheet);
  for (const imported of item.imports || []) dependencyClosure(imported, files);
  return files;
}

async function ready(page, pathname, selector) {
  await page.goto(base + pathname, { waitUntil: 'domcontentloaded' });
  await page.locator(selector).first().waitFor({ state: 'visible' });
  await page.evaluate(() => document.fonts.ready);
}

async function checkLayout(page, label) {
  const geometry = await page.evaluate(() => ({
    width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    controls: [...document.querySelectorAll('input,button,select')].filter(element => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && (rect.left < -1 || rect.right > innerWidth + 1);
    }).map(element => element.getAttribute('aria-label') || element.id || element.textContent.slice(0, 80)),
  }));
  assert.ok(geometry.scrollWidth <= geometry.width + 1, `${label}: page overflow`);
  assert.deepEqual(geometry.controls, [], `${label}: clipped controls`);
  report.checks.push({ case: label, ...geometry });
}

async function capture(page, filename) {
  const target = path.join(evidenceDirectory, filename);
  await page.screenshot({ path: target });
  report.screenshots.push(path.relative(process.cwd(), target).replaceAll('\\', '/'));
}

(async () => {
  fs.mkdirSync(evidenceDirectory, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const pageErrors = [];
    const requests = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('request', request => requests.push(new URL(request.url()).pathname));
    const modelFile = '/' + manifest[projectEntry].file;

    await ready(page, '/learn', '.workspace-hero');
    assert.equal(requests.includes(modelFile), false, 'Hub eagerly fetched project teaching');
    const allowedHub = new Set([...dependencyClosure('index.html'), ...dependencyClosure('src/learn/LearnHub.jsx')]);
    const unexpected = requests.filter(file => /\.(js|css)$/.test(file) && !allowedHub.has(file) && file.startsWith('/assets/'));
    assert.deepEqual(unexpected, [], 'Hub downloaded an unrelated lesson or lab');
    report.payloads.push({ route: '/learn', measurements: await page.evaluate(() => performance.getEntriesByType('resource').filter(item => new URL(item.name).origin === location.origin && /\.(js|css)$/.test(new URL(item.name).pathname)).map(item => ({ file: new URL(item.name).pathname, encodedBytes: item.encodedBodySize, decodedBytes: item.decodedBodySize, transferBytes: item.transferSize }))) });
    await capture(page, 'workspace-desktop.png');

    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 950 });
      for (const [route, selector] of [['/learn', '.workspace-hero'], ['/learn/paths', '.workspace-path'], ['/learn/projects', '.project-feature'], ['/learn/modules?group=start', '.track-card'], ['/learn/catalogue', '#topic-search']]) {
        await ready(page, route, selector);
        await checkLayout(page, `${route} at ${width}px`);
      }
    }

    await page.locator('#topic-search').fill('typed decision');
    await page.locator('[data-topic-id="typed-decision-models-calibrated-neural-decision-systems"]').waitFor();
    assert.equal(await page.locator('.topic-results > li').count(), 1);
    await page.locator('[data-topic-id="typed-decision-models-calibrated-neural-decision-systems"]').click();
    await page.locator('.planned-lesson').waitFor();
    assert.ok(await page.getByRole('link', { name: /Build a typed decision model/ }).isVisible());
    await page.getByRole('link', { name: /Build a typed decision model/ }).click();
    await page.locator('.tdp-explorer').waitFor();
    assert.ok(requests.includes(modelFile), 'Selected project was not fetched');
    assert.equal(requests.some(file => /\.py$/.test(file)), false, 'Program eagerly fetched before source disclosure');
    report.checks.push({ case: 'Search, planned companion and reciprocal project link', passed: true });

    await page.setViewportSize({ width: 1366, height: 1000 });
    const lab = page.getByRole('region', { name: 'Explore probabilities and decision costs' });
    const beforeProbability = await lab.locator('.tdp-readouts dd').first().innerText();
    await lab.getByRole('slider', { name: 'Temperature', exact: true }).press('End');
    assert.equal(await lab.getByRole('slider', { name: 'Temperature', exact: true }).inputValue(), '3');
    assert.notEqual(await lab.locator('.tdp-readouts dd').first().innerText(), beforeProbability);
    await lab.getByRole('checkbox', { name: 'Add an “Other” candidate' }).check();
    assert.equal(await lab.locator('.tdp-probability').count(), 4);
    await lab.getByRole('slider', { name: 'Cost of review', exact: true }).press('End');
    assert.match(await lab.locator('.tdp-policy-result > strong').innerText(), /Act/);
    await lab.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await lab.getByRole('slider', { name: 'Temperature', exact: true }).inputValue(), '1');
    assert.equal(await lab.locator('.tdp-probability').count(), 3);
    await lab.scrollIntoViewIfNeeded();
    await capture(page, 'decision-lab-desktop.png');
    await page.setViewportSize({ width: 390, height: 950 });
    await lab.scrollIntoViewIfNeeded();
    await checkLayout(page, 'Project investigation at 390px');
    await capture(page, 'decision-lab-mobile.png');
    report.checks.push({ case: 'Live temperature, candidates, cost policy and reset', passed: true });

    const lessonProgress = await page.evaluate(() => localStorage.getItem('kd-progress'));
    const checkbox = page.getByRole('checkbox', { name: 'I have completed this stage’s deliverable' });
    await checkbox.check();
    await page.reload();
    await page.locator('.tdp-explorer').waitFor();
    assert.equal(await checkbox.isChecked(), true);
    assert.match(await page.locator('.project-progress').innerText(), /1 completed/);
    assert.equal(await page.evaluate(() => localStorage.getItem('kd-progress')), lessonProgress);
    await ready(page, '/learn/projects', '.project-feature');
    const continueLink = page.getByRole('link', { name: 'Continue project', exact: true });
    assert.match(await continueLink.getAttribute('href'), /\/data$/);
    await continueLink.click();
    await page.getByRole('heading', { name: 'Build the evidence', exact: true }).waitFor();
    report.checks.push({ case: 'Milestone persistence, isolation and first-unfinished resume', passed: true });

    const stageIds = ['define', 'data', 'baseline', 'architecture', 'training', 'calibration', 'evaluation', 'serving'];
    for (const width of [1366, 320]) {
      await page.setViewportSize({ width, height: 950 });
      for (const stage of stageIds) {
        await ready(page, `/learn/projects/typed-decision-model/${stage}`, '.tdp-stage');
        assert.ok((await page.locator('.tdp-stage').innerText()).length > 1000, `${stage}: incomplete body`);
        assert.equal(await page.locator('.project-outline [aria-current="step"]').count(), 1);
        assert.ok(await page.getByRole('checkbox', { name: 'I have completed this stage’s deliverable' }).isEnabled());
        await checkLayout(page, `Project ${stage} at ${width}px`);
      }
    }
    await ready(page, '/learn/projects/typed-decision-model/architecture', '.tdp-stage');
    const disclosures = page.locator('.tdp-source');
    for (let index = 0; index < await disclosures.count(); index += 1) {
      const disclosure = disclosures.nth(index);
      await disclosure.locator('summary').click();
      await disclosure.locator('pre').waitFor();
      assert.ok((await disclosure.locator('pre').innerText()).length > 100);
    }
    report.checks.push({ case: 'On-demand canonical source disclosures', passed: true });

    for (const route of ['/learn/projects/no-such-project', '/learn/projects/__proto__', '/learn/projects/typed-decision-model/no-such-stage']) {
      await ready(page, route, 'main h1');
      assert.match(await page.locator('main h1').innerText(), /not found/i);
    }
    await ready(page, '/learn/path/full-curriculum/linear-logistic-regression?module=classical-ml', '.reader-article');
    assert.ok(await page.getByRole('navigation', { name: 'Primary navigation' }).getByRole('link', { name: 'Projects', exact: true }).isVisible());
    await checkLayout(page, 'Existing lesson and shared navigation at 320px');
    assert.deepEqual(pageErrors, []);
    report.checks.push({ case: 'Unknown route recovery and existing lesson compatibility', passed: true });
    await context.close();

    const failureContext = await browser.newContext();
    const failurePage = await failureContext.newPage();
    await failurePage.route('**' + modelFile, route => route.abort());
    await ready(failurePage, '/learn/projects/typed-decision-model', '.project-error');
    assert.equal(await failurePage.getByRole('checkbox', { name: 'I have completed this stage’s deliverable' }).isEnabled(), false);
    await failurePage.unroute('**' + modelFile);
    await failurePage.getByRole('button', { name: 'Reload page', exact: true }).click();
    await failurePage.locator('.tdp-explorer').waitFor();
    assert.ok(await failurePage.getByRole('checkbox', { name: 'I have completed this stage’s deliverable' }).isEnabled());
    await failureContext.close();
    report.checks.push({ case: 'Failed project import, disabled completion and explicit reload recovery', passed: true });

    report.sourceHashes = Object.fromEntries(['src/learn/LearnHub.jsx', 'src/learn/ProjectReader.jsx', 'src/learn/components/LearningNav.jsx', 'src/learn/hooks/useProjectProgress.js', 'src/learn/learning-workspace.css', 'src/learn/project-reader.css', 'src/learn/data/projects/typed-decision-model/content.jsx', 'src/learn/data/projects/typed-decision-model/decision-model.js'].map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
    report.passed = true;
    console.log(`PASS: ${report.checks.length} scoped workspace checks; three reviewed screenshot candidates retained.`);
  } catch (error) {
    report.passed = false;
    report.errors.push(error.stack);
    console.error(error.stack);
    process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(evidenceDirectory, 'workspace-browser-review.json'), JSON.stringify(report, null, 2) + '\n');
  }
})();
