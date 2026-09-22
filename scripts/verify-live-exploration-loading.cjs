// Bounded production integration for the live-exploration revision.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4191';
const manifest = read('dist/.vite/manifest.json');
const manifestHash = hash('dist/.vite/manifest.json');
const publication = read('src/learn/data/lesson-manifest.json');
const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
const records = [];
const sources = ['src/learn/Reader.jsx', 'src/learn/components/TopicContent.jsx', 'src/learn/components/LessonBoundary.jsx', 'src/learn/hooks/useTopicResource.js', 'src/learn/data/lesson-loader.js'];
const sourceHashes = Object.fromEntries(sources.map(file => [file, hash(file)]));
async function context(browser) {
  const value = await browser.newContext();
  await value.route('https://fonts.googleapis.com/**', route => route.fulfill({path:fonts.stylesheet,contentType:'text/css'}));
  await value.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()]
    ? route.fulfill({path:fonts.files[route.request().url()],contentType:'font/ttf',headers:{'access-control-allow-origin':'*'}}) : route.continue());
  return value;
}
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    assert.deepEqual(publication, read('docs/teaching/evidence/live-exploration-baseline.json').publication);
    const bodies = new Set(Object.values(publication).map(value => manifest[`src/learn/data/${value.replace(/^\.\//,'')}`].file));
    const outlines = new Set(Object.entries(manifest).filter(([key]) => key.includes('/generated/outlines/')).map(([,value]) => value.file));
    for (const [id,module] of [['python-basics-types-control-flow-functions-modules','programming-scientific-computing'], ['k-means-hierarchical-clustering','classical-ml'], ['feature-selection-importance-shap-permutation-mutual-info','classical-ml']]) {
      const isolated = await context(browser), page = await isolated.newPage();
      const requests = [], errors = [];
      page.on('request', request => requests.push(request.url()));
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/${id}?module=${module}`);
      await page.locator('main h2, main h3').first().waitFor();
      const source = `src/learn/data/topics/${id}.jsx`, body = manifest[source].file;
      const allowed = new Set();
      const visit = key => { const chunk=manifest[key]; if(allowed.has(chunk.file)) return; allowed.add(chunk.file); (chunk.imports||[]).forEach(visit); };
      [Object.keys(manifest).find(key=>manifest[key].isEntry&&key.endsWith('index.html')),'src/learn/Reader.jsx',source].forEach(visit);
      const local = [...new Set(requests.filter(url=>url.startsWith(base+'/')).map(url=>new URL(url).pathname.slice(1)))];
      assert.deepEqual(local.filter(file=>bodies.has(file)),[body]);
      assert.deepEqual(local.filter(file=>outlines.has(file)),[]);
      const scripts = local.filter(file=>file.endsWith('.js'));
      assert.deepEqual(scripts.filter(file=>!allowed.has(file)),[]);
      assert.deepEqual(errors,[]);
      records.push({check:'Only selected lesson and shared dependencies load',id,decodedJsBytes:scripts.reduce((n,file)=>n+fs.statSync('dist/'+file).size,0),gzipEstimateBytes:scripts.reduce((n,file)=>n+gzipSync(fs.readFileSync('dist/'+file)).length,0)});
      if(id==='k-means-hierarchical-clustering') {
        await page.locator('.reader-complete').click(); await page.reload(); await page.locator('main h2').first().waitFor();
        assert.equal(await page.evaluate(id=>JSON.parse(localStorage.getItem('kd-progress'))[id],id),true);
        await page.locator('.reader-footer__next').click(); await page.waitForURL('**/pca-dimensionality-reduction?module=classical-ml');
        records.push({check:'Completion persists and Next follows module sequence'});
      }
      await isolated.close();
    }
    const id='k-means-hierarchical-clustering', body=manifest[`src/learn/data/topics/${id}.jsx`].file;
    for(const failure of ['import','render']) {
      const isolated=await context(browser), page=await isolated.newPage(); let attempts=0;
      await page.route(`**/${body}*`,route=>++attempts>1?route.continue():failure==='import'?route.abort('failed'):route.fulfill({status:200,contentType:'text/javascript',body:'export default {content(){throw new Error("Controlled live-exploration render failure")}};'}));
      await page.goto(`${base}/learn/path/full-curriculum/${id}?module=classical-ml`);
      const alert=page.locator('.lesson-load-error'); await alert.waitFor();
      assert.ok(await page.locator('.reader-complete').isDisabled()); assert.equal(await page.locator('.planned-lesson').count(),0);
      await Promise.all([page.waitForEvent('domcontentloaded'),alert.getByRole('button',{name:'Reload page',exact:true}).click()]);
      await page.locator('main h2').first().waitFor(); assert.ok(attempts>1); assert.ok(await page.locator('.reader-complete').isEnabled());
      records.push({check:'Published lesson error recovers after reload',failure,attempts}); await isolated.close();
    }
    assert.equal(hash('dist/.vite/manifest.json'),manifestHash);
    for(const[file,digest]of Object.entries(sourceHashes))assert.equal(hash(file),digest);
    fs.writeFileSync('docs/teaching/evidence/live-exploration-loading.json',JSON.stringify({status:'passed',checkedAt:new Date().toISOString(),base,manifestHash,sourceHashes,verifierSha256:hash(__filename),records,limitations:'Three representative loading closures and two controlled recovery paths; gzip figures are estimates, not latency benchmarks.'},null,2)+'\n');
    console.log(`PASS: ${records.length} production loading/recovery groups.`);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
