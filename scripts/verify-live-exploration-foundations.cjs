// Review the live-exploration migration without overwriting historical receipts.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read = path => JSON.parse(fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4190';
const production = process.argv.includes('--production');
const survey = !process.argv.includes('--focused-only');
const capture = !process.argv.includes('--no-captures');
const scope = read('docs/teaching/evidence/live-exploration-runtime-scope.json');
const ownTopics = scope.topics.filter(topic => topic.module !== 'classical-ml' || topic.id === 'k-means-hierarchical-clustering');
const sources = [...new Set(ownTopics.flatMap(topic => topic.dependencies))];
const sourceHashes = Object.fromEntries(sources.map(path => [path, hash(path)]));
const fonts = read(process.env.LEARNING_FONT_FIXTURES || 'scratch/kmeans-revision-review/fonts/manifest.json');
const mode = production ? 'production' : 'browser';
const receipt = `docs/teaching/evidence/live-exploration-foundations-${mode}.json`;
const records = [], captures = [], errors = [];
let active = 'start';
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const matches = async (locator, expression) => assert.match(await locator.innerText(), expression);
async function range(page, locator, value) {
  await locator.fill(String(value));
  await settle(page);
}
async function noPredictionForms(page) {
  const failures = await page.locator('main').evaluate(root => [...root.querySelectorAll('input, select, textarea, button')].filter(node => {
    const label = [...(node.labels || [])].map(item => item.textContent).join(' ');
    const text = `${label} ${node.getAttribute('aria-label') || ''} ${node.getAttribute('placeholder') || ''} ${node.tagName === 'BUTTON' ? node.textContent : ''}`;
    return /record.{0,25}prediction|your prediction|predict first|check.{0,15}prediction|choose a prediction|make a prediction|commit prediction/i.test(text);
  }).map(node => node.outerHTML.slice(0,250)));
  assert.deepEqual(failures, [], 'No learner-prediction entry or gate');
}
async function open(page, id) {
  const topic = scope.topics.find(item => item.id === id);
  assert.ok(topic, id);
  await page.goto(`${base}/learn/path/full-curriculum/${id}?module=${topic.module}`, { waitUntil: 'domcontentloaded' });
  await page.locator('main h2, main h3').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
  await noPredictionForms(page);
  return topic;
}
async function shot(page, target, name) {
  if (!capture) return;
  const file = `docs/teaching/evidence/screenshots/live-exploration-${mode}-${name}.png`;
  await target.scrollIntoViewIfNeeded();
  const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
  try { await target.screenshot({ path: file }); } finally { await style.evaluate(node => node.remove()); }
  captures.push({ path:file, sha256:hash(file) });
}
(async () => {
  const browser = await chromium.launch({ channel:'msedge', headless:true });
  const context = await browser.newContext({ viewport:{width:1366,height:1000} });
  await context.route('https://fonts.googleapis.com/**', route => route.fulfill({path:fonts.stylesheet,contentType:'text/css',headers:{'access-control-allow-origin':'*'}}));
  await context.route('https://fonts.gstatic.com/**', route => {
    const file = fonts.files[route.request().url()];
    return file ? route.fulfill({path:file,contentType:'font/ttf',headers:{'access-control-allow-origin':'*'}}) : route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push({ check:active, message:error.message }));
  try {
    active = 'K-Means: five live investigations';
    await open(page,'k-means-hierarchical-clustering');
    assert.equal(await page.evaluate(() => [...document.fonts].some(f=>f.family.includes('Space Grotesk')&&f.status==='loaded')),true);
    const labs = page.locator('.kh-investigation');
    assert.equal(await labs.count(),5);
    const lloyd = labs.nth(0);
    await matches(lloyd.locator('.kh-readout'),/SSE = 19.25/);
    await lloyd.getByRole('button',{name:'Run to fixed point'}).click();
    await matches(lloyd.locator('.kh-readout'),/SSE = 7.6875/);
    await range(page,lloyd.getByLabel('Lloyd process phase'),1);
    await matches(lloyd.locator('.kh-readout'),/SSE = 19.25/);
    await lloyd.getByLabel('Point configuration').selectOption('rectangle');
    await lloyd.getByLabel('Center 1 starts at row').selectOption('1');
    await lloyd.getByRole('button',{name:'Run to fixed point'}).click();
    await matches(lloyd.locator('.kh-readout'),/SSE = 9/);
    const geometry = labs.nth(1);
    await matches(geometry.locator('.kh-readout'),/SSE 1\b/);
    await geometry.getByLabel('Vertical measurement unit').selectOption('10');
    await matches(geometry.locator('.kh-readout'),/SSE 9\b/);
    await range(page,geometry.getByLabel('Weight on squared vertical differences'),.01);
    await matches(geometry.locator('.kh-readout'),/SSE 1\b/);
    assert.equal(await geometry.getByRole('table').count(),2);
    const seeding = labs.nth(2);
    await seeding.getByLabel('Draw position in the cumulative probability line').focus();
    await page.keyboard.press('Home');
    await matches(seeding.locator('.kh-readout'),/selects P1/);
    await page.keyboard.press('End');
    await matches(seeding.locator('.kh-readout'),/selects P5/);
    await seeding.getByRole('button',{name:'Accept this draw and pick the next center'}).click();
    await seeding.getByRole('button',{name:'Undo last draw'}).click();
    await seeding.getByLabel('Seeding data').selectOption('duplicates');
    await matches(seeding.locator('.kh-readout'),/Every D² is zero/);
    const hierarchy = labs.nth(3);
    await matches(hierarchy.locator('.kh-readout'),/3 clusters/);
    await hierarchy.getByLabel('Partition rule').selectOption('count');
    await matches(hierarchy.locator('.kh-readout'),/4 clusters/);
    await hierarchy.getByLabel('Point configuration').selectOption('chain');
    await hierarchy.getByLabel('Linkage definition').selectOption('complete');
    await matches(hierarchy.locator('.kh-caption'),/right pair/);
    const palette = labs.nth(4);
    await palette.getByLabel('Image to quantize').selectOption('sky');
    await range(page,palette.getByLabel('Requested palette size'),8);
    await matches(palette,/teaching cap is not lossless/);
    await shot(page,geometry,'geometry-live-desktop');
    records.push({check:active,assertions:'Initial outputs; slider changes and compensation; process scrubbing; keyboard draws; zero mass; linkage/cut comparison; palette cap.'});
    await page.setViewportSize({width:390,height:1000});
    await shot(page,hierarchy,'hierarchy-live-phone');
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    active = 'Linux paths and streams live edits';
    await open(page,'linux-basics-filesystems-processes');
    const paths = page.locator('.linux-path-lab');
    await paths.getByLabel('Path value').fill('/project/reports');
    await matches(paths.locator('.linux-path-step'),/reports/);
    await paths.getByLabel('Path value').fill('/missing');
    await matches(paths.locator('.linux-path-step'),/not|cannot|missing/i);
    await paths.getByRole('button',{name:'Walk from the start'}).click();
    await matches(paths.locator('.linux-path-step'),/STEP 1/);
    const streams = page.locator('.linux-stream-lab');
    await streams.getByRole('button',{name:/Separate files/}).click();
    await matches(streams.locator('.linux-stream-results'),/errors.txt/);
    await streams.getByRole('button',{name:/Count result lines/}).click();
    await streams.locator('details').filter({hasText:'Deeper: can useful output come from a failed command?'}).locator('summary').click();
    await streams.getByLabel('Exit 7 after printing the same messages').check();
    await matches(streams.locator('.linux-stream-status'),/Program exit status\s*7.*Pipeline status\s*0/s);
    await shot(page,streams,'streams-live-phone');
    records.push({check:active});
    active = 'NumPy input/output correspondence is always available';
    await open(page,'numpy-arrays-broadcasting-vectorization');
    const selection=page.locator('.numpy-selection-lab');
    await selection.getByLabel('Array selection').selectOption('scalar');
    await matches(selection,/scalar · no axes/);
    await selection.getByLabel('Array selection').selectOption('column2d');
    await matches(selection,/\(3, 1\)/);
    const broadcast=page.locator('.numpy-broadcast-lab');
    await broadcast.getByLabel('Broadcast investigation').selectOption('invalid');
    await matches(broadcast.locator('.numpy-feedback'),/ValueError: no compatible broadcast/i);
    await broadcast.getByLabel('Broadcast investigation').selectOption('outer');
    await matches(broadcast,/\(3, 3\)/);
    const reduction=page.locator('.numpy-reduction-lab');
    await reduction.getByLabel('Axis to average over').selectOption('1');
    await matches(reduction.locator('.numpy-feedback'),/19/);
    await shot(page,broadcast,'broadcast-live-phone');
    records.push({check:active});
    active='OOP validation follows the selected input';
    await open(page,'object-oriented-programming-in-python');
    const validation=page.locator('[data-oop-lab="validation"]');
    const choices=await validation.getByLabel('Candidate reading').locator('option').evaluateAll(nodes=>nodes.map(n=>n.value));
    const initial=await validation.locator('.oop-feedback').innerText();
    for(const value of choices)await validation.getByLabel('Candidate reading').selectOption(value);
    assert.notEqual(await validation.locator('.oop-feedback').innerText(),initial);
    records.push({check:active,offeredCases:choices.length});
    active='Constructed diagonal subset and factorial effects stay live';
    await open(page,'sets-logic-relations-proof-techniques');
    const diagonal=page.getByRole('region',{name:'Diagonal missing subset investigation'});
    const original=await diagonal.locator('.sets-logic-constructed').innerText();
    await diagonal.getByRole('button',{name:'Element 0 in subset f(0)',exact:true}).click();
    assert.notEqual(await diagonal.locator('.sets-logic-constructed').innerText(),original);
    await open(page,'sampling-measurement-experimental-design');
    const factorial=page.getByRole('region',{name:'Factorial interaction investigation'});
    const before=await factorial.locator('.sampling-readout').innerText();
    await range(page,factorial.getByLabel('Interaction contrast'),-6);
    assert.notEqual(await factorial.locator('.sampling-readout').innerText(),before);
    records.push({check:active});
    if(survey) {
      for(const topic of ownTopics.filter(t=>t.module!=='classical-ml')) {
        active=`Foundation page survey: ${topic.id}`;
        await open(page,topic.id);
        const h3=await page.locator('main h3').count();
        assert.ok(h3>0,topic.id);
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,topic.id+' page overflow');
        records.push({check:active,headings:h3,width:390});
        if (records.length % 20 === 0) console.log(`Checked ${records.length} live-exploration groups so far.`);
      }
    }
    assert.deepEqual(errors,[],'No uncaught browser errors');
    for(const[file,digest]of Object.entries(sourceHashes))assert.equal(hash(file),digest,'Source changed during checks: '+file);
    fs.writeFileSync(receipt,JSON.stringify({status:'passed',checkedAt:new Date().toISOString(),production,sourceHashes,verifierSha256:hash(__filename),records,captures,fontAssets:Object.fromEntries([fonts.stylesheet,...Object.values(fonts.files)].map(p=>[p,hash(p)])),limitations:'Focused behavioral cases plus page-load/control-copy survey. The survey does not exercise every control. Images need separate actual review; one Chromium-family engine.'},null,2)+'\n');
    console.log(`PASS: ${records.length} live-exploration groups; ${captures.length} captures.`);
  } catch(error) {
    console.error(JSON.stringify({status:'failed',active,error:error.stack,pageErrors:errors},null,2));
    process.exitCode=1;
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
