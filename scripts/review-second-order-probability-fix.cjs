const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await page.routeWebSocket('**', socket => socket.close());
  await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient?module=mathematical-statistical-foundations');
  const lab = page.getByRole('region', { name: 'Natural gradient probability geometry', exact: true });
  await lab.waitFor();
  const probability = lab.getByRole('slider', { name: 'Starting success probability', exact: true });
  await probability.focus();
  await probability.press('Home');
  const fraction = lab.getByRole('slider', { name: 'Natural step fraction', exact: true });
  await fraction.focus();
  await fraction.press('End');
  const text = await lab.innerText();
  if (!text.includes('p=1 − 2.639e-6') || !text.includes('Numerical readouts are rounded')) throw new Error(text);
  const bar = lab.getByRole('img', { name: /Logit update mapped back: success probability/ });
  const accessibleLabel = await bar.getAttribute('aria-label');
  if (!accessibleLabel.includes('1 − 2.639e-6')) throw new Error(accessibleLabel);
  const folder = path.resolve('scratch/second-order-independent');
  fs.mkdirSync(folder, { recursive: true });
  await lab.screenshot({ path: path.join(folder, 'probability-repair-390.png') });
  const record = { status: 'passed', reviewed_at: new Date().toISOString(), viewport: '390x844',
    fixture: { probability: .05, target: .8, fraction: 1 }, accessibleLabel,
    keyboard: 'Home on starting probability; End on step fraction',
    screenshot: 'scratch/second-order-independent/probability-repair-390.png',
    scope: 'Targeted actual UI regression only; author owns full browser review and final freeze' };
  fs.writeFileSync(path.resolve('docs/teaching/evidence/second-order-probability-display-after.json'), JSON.stringify(record, null, 2) + '\n');
  console.log(JSON.stringify(record, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
