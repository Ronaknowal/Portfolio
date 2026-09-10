const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');

(async () => {
  const { gitExamples, linuxExamples } = await Promise.all([import("../src/learn/data/git-examples.js"), import("../src/learn/data/linux-command-examples.js")]).then(modules => Object.assign({}, ...modules));
  const { gitNewExamples } = await import("../src/learn/data/git-practice-examples.js");
  const { stagingTrace } = await import('../src/learn/data/git-foundations-model.js');
  const { investigation } = await import('../src/learn/data/linux-practice-examples.js');
  const { permissionModel } = await import('../src/learn/data/system-lesson-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    fs.mkdirSync('scratch/programming-four-browser', { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      for (const [slug, examples] of [['git-github-collaborative-version-control', {...gitExamples,...gitNewExamples}], ['linux-basics-filesystems-processes', { ...linuxExamples, investigation }]]) {
        await page.goto('http://127.0.0.1:5173/learn/topic/' + slug);
        await page.locator('.lesson-intro').waitFor();
        assert.equal(await page.locator('.lesson-guide').count(), 0);
        assert.match(await page.locator('.lesson-intro').innerText(), /Shell examples run in your own environment/);
        const ids = await page.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
        for (const id of ids) assert.equal(await page.locator('[id="' + id + '"]').count(), 1, id);
        for (const details of await page.locator('.lesson-check details').all()) {
          await details.evaluate(n=>{for(let p=n.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
          await details.locator('summary').click();
          assert.equal(await details.getAttribute('open'), '');
        }
        // Linux now has several labs and optional worked examples. Its complete
        // interaction coverage lives in review-linux-lesson.cjs.
        for (const details of await page.locator('.linux-lesson details').all()) {
          if (await details.getAttribute('open') === null) await details.locator(':scope > summary').click();
        }
        const outputs = await page.locator('.terminal-example__output').allTextContents();
        assert.equal(outputs.length, Object.keys(examples).length);
        for (const example of Object.values(examples)) assert.ok(outputs.some(output => output.includes(example.output)), 'Missing output ' + slug);
        assert.ok(await page.locator('.lesson-pilot code').evaluateAll(nodes => nodes.every(node => node.textContent.trim())));
        const lab = page.locator(slug.startsWith('git') ? '[data-investigation="git-staging"]' : '.permission-lab');
        if (slug.startsWith('git')) {
          const previous = lab.getByRole('button', { name: 'Back', exact: true });
          const next = lab.getByRole('button', { name: 'Next step', exact: true });
          assert.ok(await previous.isDisabled());
          const trace=stagingTrace(false);
          for (let i = 0; i < trace.length; i++) {
            if (i) await next.click();
            assert.match(await lab.locator('.nt-stepper').innerText(), new RegExp('Step ' + (i + 1) + ' of 6'));
            assert.deepEqual(await lab.locator('.nt-snapshot pre').allTextContents(), trace[i].versions.map(version => 'version ' + version));
            assert.equal(await lab.locator('.nt-feedback').innerText(), trace[i].note);
          }
          assert.ok(await next.isDisabled());
          await previous.click();
          assert.match(await lab.locator('.nt-stepper').innerText(), /Step 5 of 6/);
          await lab.getByRole('button', { name: 'Reset', exact: true }).focus();
          await page.keyboard.press('Enter');
          assert.ok(await previous.isDisabled());
        } else {
          for (const mode of ['600', '640', '700', '750', '755']) {
            await lab.getByLabel('Permission mode', { exact: true }).selectOption(mode);
            for (const subject of ['owner', 'group', 'other']) {
              await lab.getByLabel('Permission class', { exact: true }).selectOption(subject);
              for (const kind of ['file', 'directory']) {
                await lab.getByLabel('Filesystem object', { exact: true }).selectOption(kind);
                const actual = await lab.locator('tbody tr td:nth-child(2)').allTextContents();
                assert.deepEqual(actual, permissionModel(mode, subject, kind === 'directory').map(row => row[1] ? 'Yes' : 'No'));
              }
            }
          }
          await lab.getByRole('button', { name: 'Reset permissions', exact: true }).focus();
          await page.keyboard.press('Enter');
          assert.equal(await lab.getByLabel('Permission mode', { exact: true }).inputValue(), '640');
          assert.equal(await lab.getByLabel('Permission class', { exact: true }).inputValue(), 'owner');
          assert.equal(await lab.getByLabel('Filesystem object', { exact: true }).inputValue(), 'file');
        }
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), slug + ': document overflow');
        await lab.screenshot({ path: 'scratch/programming-four-browser/' + slug + '-' + width + '.png', style: '.learn-nav { visibility: hidden !important; }' });
        console.log(width + 'px: ' + slug + ': outputs, navigation, practice and explorer checks passed');
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
