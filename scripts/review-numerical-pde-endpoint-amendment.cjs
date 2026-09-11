const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/numerical-pde-endpoint-amendment/browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

(async () => {
  const modelRecord = JSON.parse(fs.readFileSync('scratch/numerical-pde-endpoint-amendment/model-results.json', 'utf8'));
  const sourceHashes = modelRecord.sourceHashes;
  assert(Object.entries(sourceHashes).every(([file, digest]) => hash(file) === digest));
  const { numericalPdeExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/numerical-pde-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      const record = { width, states: [], captures: [] };
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-pdes-grids-finite-elements-stability?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.npde-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(record.fonts.some(font => font.includes('Space Grotesk')) && record.fonts.some(font => font.includes('JetBrains Mono')));
      await page.addStyleTag({ content: 'html { scroll-behavior:auto!important; }' });
      const served = await page.evaluate(async input => {
        const { poissonProblem } = await import('/src/learn/data/numerical-pde-models.js');
        const actual = poissonProblem(input);
        return { first: actual.nodes[0], last: actual.nodes.at(-1), curveEnd: actual.curve.at(-1), right: actual.right };
      }, modelRecord.reproducer);
      assert.equal(served.first, 0);
      assert.equal(served.last, .7);
      assert.equal(served.curveEnd.x, .7);
      assert.equal(served.curveEnd.numerical, 4.1);
      record.servedModuleRegression = served;
      record.states.push('Actual served model accepts changed physical endpoint; UI remains fixed to the unit rod.');
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const file = name + '-' + width + '.png';
        await page.screenshot({ path: path.join(directory, file) });
        record.captures.push(file);
      }
      const lab = lesson.getByRole('region', { name: 'From a stencil row to a field certificate', exact: true });
      await lab.getByLabel('Manufactured target', { exact: true }).selectOption('linear');
      await lab.getByLabel('Endpoint values', { exact: true }).selectOption('tilted');
      await lab.getByLabel('Solve', { exact: true }).selectOption('direct');
      assert(!(await lab.innerText()).includes('outside numeric range'));
      await shot(lab.locator('figure'), 'unit-rod-boundary');
      record.states.push('Existing unit-rod linear field and nonzero boundary values render.');
      await lab.getByLabel('Solve', { exact: true }).selectOption('jacobi');
      const updates = lab.getByLabel('Jacobi updates', { exact: false });
      await updates.focus();
      const previous = await updates.inputValue();
      await page.keyboard.press('ArrowRight');
      assert.notEqual(await updates.inputValue(), previous);
      await lab.getByRole('button', { name: 'Reset', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await lab.getByLabel('Manufactured target', { exact: true }).inputValue(), 'quartic');
      assert.equal(await lab.getByLabel('Solve', { exact: true }).inputValue(), 'direct');
      record.states.push('Jacobi slider responds to keyboard; keyboard Reset restores the original field.');
      const example = examples.poisson;
      const program = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
      const blocks = program.locator(':scope > div');
      for (const [index, expected] of [[0, example.code], [1, example.expected]]) {
        const actual = await blocks.nth(index).evaluate(node => [...node.childNodes].filter(child => child.nodeType === 3).map(child => child.textContent).join(''));
        assert.equal(normalize(actual), normalize(expected));
      }
      assert.equal(normalize(await program.evaluate(node => node.previousElementSibling.textContent)), normalize('Before running: ' + example.question));
      // Position the viewport at the actual new line, not the program header far above it.
      const line = 'nodes[0], nodes[-1] = 0.0, length';
      const lineRect = await blocks.first().evaluate((node, needle) => {
        const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
        let text;
        while ((text = walker.nextNode())) {
          const start = text.textContent.indexOf(needle);
          if (start < 0) continue;
          const range = document.createRange();
          range.setStart(text, start); range.setEnd(text, start + needle.length);
          const box = range.getBoundingClientRect();
          window.scrollTo({ top: box.top + scrollY - 180, behavior: 'instant' });
          return { width: box.width, content: text.textContent.slice(start, start + needle.length) };
        }
        throw new Error('Amended line not found in displayed complete code.');
      }, line);
      assert.equal(lineRect.content, line);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const codeCapture = 'pinned-endpoint-program-' + width + '.png';
      await page.screenshot({ path: path.join(directory, codeCapture) });
      record.captures.push(codeCapture);
      await shot(blocks.last(), 'unchanged-program-output');
      record.states.push('Amended complete program, visible question and conserved actual output match source.');
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
      assert.deepEqual(errors, []);
      records.push(record);
      await page.close();
    }
    assert(Object.entries(sourceHashes).every(([file, digest]) => hash(file) === digest));
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sourceHashes, records, errors, scope: 'Targeted amended program and endpoint/model paths; original full lesson browser evidence is preserved separately.' }, null, 2) + '\n');
    console.log('Numerical PDE endpoint amendment: original-font three-width served model, program/output and keyboard checks passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
