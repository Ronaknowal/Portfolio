const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  const dir = 'scratch/linux-lesson-review';
  fs.mkdirSync(dir, { recursive: true });
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto('http://127.0.0.1:5173/learn/topic/linux-basics-filesystems-processes');
      await page.locator('.linux-lesson').waitFor();
      assert.equal(await page.locator('.lesson-lab').count(), 4);
      assert.equal(await page.locator('.lesson-guide').count(), 0);
      for (const id of await page.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)))) {
        assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      }
      const path = page.locator('.linux-path-lab');
      const walk = async () => {
        await path.getByRole('button', { name: 'Start path walk', exact: true }).click();
        for (let i = 0; i < 20 && await path.getByRole('button', { name: 'Next step', exact: true }).count(); i++) {
          await path.getByRole('button', { name: 'Next step', exact: true }).click();
        }
        assert.ok(await path.getByRole('button', { name: 'Walk complete', exact: true }).isDisabled());
      };
      await path.getByLabel('Before stepping, I predict…').selectOption('directory');
      await walk();
      assert.match(await path.locator('.linux-path-step').innerText(), /cd succeeds/);
      assert.match(await path.locator('.linux-path-shell-tag').locator('..').locator('..').innerText(), /reports/);
      await path.screenshot({ path: `${dir}/paths-${width}.png`, style: '.learn-nav { visibility: hidden !important; }' });
      for (const [preset, expected] of [['wrong-start', /No entry named reports/], ['dot-file', /Located the file/], ['file-cd', /directory is required/]]) {
        await path.getByLabel('Path investigation').selectOption(preset);
        await walk();
        assert.match(await path.locator('.linux-path-step').innerText(), expected);
        assert.match(await path.locator('.linux-path-shell-tag').locator('..').locator('..').innerText(), /raw/);
      }
      await path.getByRole('button', { name: 'Reset path lab', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await path.getByLabel('Path value').inputValue(), '../../reports');

      const streams = page.locator('.linux-stream-lab');
      for (const [label, expected] of [['Both to terminal', /both reach the terminal/], ['Save results', /warning still reaches the terminal/], ['Separate files', /prints neither to the terminal/], ['Count result lines', /warning bypasses wc/]]) {
        await streams.getByRole('button', { name: label }).click();
        await streams.getByRole('button', { name: '3. Trace the streams', exact: true }).click();
        assert.match(await streams.locator('.linux-stream-feedback').innerText(), expected);
      }
      await streams.screenshot({ path: `${dir}/streams-${width}.png`, style: '.learn-nav { visibility: hidden !important; }' });
      await streams.getByText('Deeper: why redirection order matters', { exact: true }).click();
      for (const label of ['File, then duplicate', 'Duplicate, then file']) {
        await streams.getByRole('button', { name: label }).click();
        await streams.getByRole('button', { name: '3. Trace the streams', exact: true }).click();
        assert.match(await streams.locator('.linux-stream-output-grid').innerText(), /combined.txt/);
      }
      await streams.getByRole('button', { name: 'Count result lines' }).click();
      await streams.getByText('Deeper: can useful output come from a failed command?', { exact: true }).click();
      await streams.getByLabel('Exit 7 after printing the same messages').check();
      await streams.getByRole('button', { name: '3. Trace the streams', exact: true }).click();
      assert.match(await streams.locator('.linux-stream-status').innerText(), /Program exit status 7/);
      assert.match(await streams.locator('.linux-stream-status').innerText(), /Pipeline status 0/);
      await streams.getByRole('button', { name: 'Reset streams' }).click();

      const permission = page.getByRole('region', { name: 'Linux permission explorer', exact: true });
      assert.equal(await permission.locator('[data-permission-gate="file"]').getAttribute('data-state'), 'not-reached');
      for (let mask = 0; mask < 8; mask++) {
        await permission.getByLabel('Directory read · r', { exact: false }).setChecked(!!(mask & 4));
        await permission.getByLabel('Directory search · x', { exact: false }).setChecked(!!(mask & 2));
        await permission.getByLabel('File read · r', { exact: false }).setChecked(!!(mask & 1));
        assert.equal(await permission.locator('[data-permission-gate="file"]').getAttribute('data-state'), !(mask & 2) ? 'not-reached' : mask & 1 ? 'passed' : 'blocked');
        assert.equal(await permission.locator('.permission-lab__observations output').last().innerText(), (mask & 3) === 3 ? 'latency_ms\n10\n20' : 'Permission denied');
      }
      await permission.getByLabel('Directory read · r', { exact: false }).uncheck();
      await permission.screenshot({ path: `${dir}/permissions-${width}.png`, style: '.learn-nav { visibility: hidden !important; }' });
      await permission.getByText('Deeper: read an ordinary permission mode', { exact: true }).click();
      await permission.getByLabel('Permission mode', { exact: true }).selectOption('047');
      assert.deepEqual(await permission.locator('tbody tr td:nth-child(2)').allTextContents(), ['No', 'No', 'No']);
      await permission.getByLabel('Permission class', { exact: true }).selectOption('group');
      assert.deepEqual(await permission.locator('tbody tr td:nth-child(2)').allTextContents(), ['Yes', 'No', 'No']);

      const processLab = page.getByRole('region', { name: 'Linux process lifecycle explorer', exact: true });
      for (const [action, expected] of [['Start child', 'running'], ['Pause child', 'stopped'], ['Continue child', 'running'], ['End child', 'exited'], ['Collect exit status', 'collected']]) {
        await processLab.getByRole('button', { name: action, exact: true }).click();
        assert.ok(await processLab.locator(`.linux-process__child--${expected}`).count());
        assert.match(await processLab.locator('.linux-process__parent').innerText(), /Bash shell/);
      }
      await processLab.getByRole('button', { name: 'Reset process', exact: true }).click();
      await processLab.getByRole('button', { name: 'Start child', exact: true }).click();
      await processLab.getByRole('button', { name: 'Pause child', exact: true }).click();
      await processLab.screenshot({ path: `${dir}/process-${width}.png`, style: '.learn-nav { visibility: hidden !important; }' });

      for (const details of await page.locator('.linux-lesson details').all()) {
        if (await details.getAttribute('open') === null) await details.locator(':scope > summary').click();
      }
      const { investigation } = await import('../src/learn/data/linux-practice-examples.js');
      assert.equal(await page.locator('.terminal-example__output').count(), 9);
      assert.ok((await page.locator('.linux-mission .terminal-example__output').innerText()).includes(investigation.output));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `page overflow at ${width}`);
      const overflow = await page.locator('.lesson-lab').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => node.className));
      assert.deepEqual(overflow, [], `lab overflow at ${width}`);
      const reset = path.getByRole('button', { name: 'Reset path lab', exact: true });
      await reset.focus();
      await page.keyboard.press('Tab');
      assert.ok(await page.evaluate(() => document.activeElement?.tagName !== 'BODY'));
      console.log(`PASS ${width}px: four labs, routes, practice, permission states, process transitions, navigation and overflow.`);
    }
    assert.deepEqual(errors, []);
    console.log('PASS no browser page errors. Screenshots saved in ' + dir);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
