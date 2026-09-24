const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const directory = path.resolve('scratch/gradient-boosted-trees-verification/resources');
  fs.mkdirSync(directory, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
    await page.goto('https://www.youtube.com/watch?v=3CC4N4z3GJc', { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.getByRole('heading', { name: /Gradient Boost/i }).first().waitFor({ timeout: 25000 }).catch(() => {});
    await page.locator('video').evaluateAll(videos => videos.forEach(video => video.pause()));
    const more = page.getByRole('button', { name: /^(?:\.\.\.)?more\s*$/i });
    if (await more.count()) await more.first().click();
    const transcript = page.getByRole('button', { name: /Show transcript/i });
    if (await transcript.count()) await transcript.first().click();
    await page.locator('ytd-transcript-segment-renderer').first().waitFor({ timeout: 15000 }).catch(() => {});
    const text = await page.locator('body').innerText();
    fs.writeFileSync(path.join(directory, 'statquest-page.txt'), text);
    const buttons = await page.getByRole('button').allTextContents();
    fs.writeFileSync(path.join(directory, 'statquest-buttons.json'), JSON.stringify(buttons, null, 2));
    await page.screenshot({ path: path.join(directory, 'statquest-page.png') });
    console.log(text.slice(0, 24000));
    console.log('Buttons:', buttons.filter(text => /transcript|more|accept|reject/i.test(text)));
    const video = page.locator('video').first();
    for (const seconds of [365, 480, 835]) {
      await video.evaluate((node, time) => new Promise(resolve => {
        node.pause();
        node.addEventListener('seeked', resolve, { once: true });
        node.currentTime = time;
        setTimeout(resolve, 10000);
      }), seconds);
      await video.screenshot({ path: path.join(directory, `statquest-frame-${seconds}.png`) });
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
