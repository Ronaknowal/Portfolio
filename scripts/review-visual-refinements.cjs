const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const dir = 'scratch/visual-refinements';
fs.mkdirSync(dir, {recursive:true});
(async () => {
  const {ringTrace} = await import('../src/learn/data/linked-foundations-model.js');
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const errors = [], results = [];
  try {
    for (const width of [1440,390]) {
      const page = await browser.newPage({viewport:{width,height:1000}});
      page.on('pageerror', e => errors.push(e.message));
      const go = async id => {
        await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173'}/learn/path/full-curriculum/${id}`);
        await page.locator('.lesson-intro').waitFor();
      };
      const inspect = async (target,name) => {
        await target.scrollIntoViewIfNeeded();
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),name+' page overflow');
        const bad = await target.locator('svg text').evaluateAll(nodes => nodes.filter(n => n.getBoundingClientRect().width).filter(n => {
          const a=n.getBoundingClientRect(), b=n.ownerSVGElement.getBoundingClientRect();
          return a.left<b.left-1 || a.right>b.right+1 || a.top<b.top-1 || a.bottom>b.bottom+1;
        }).map(n => n.textContent));
        assert.deepEqual(bad,[],name+' labels outside diagram');
        await page.locator('.learn-nav').evaluate(n => n.style.visibility='hidden');
        try {await target.screenshot({path:`${dir}/${name}-${width}.png`});}
        finally {await page.locator('.learn-nav').evaluate(n => n.style.visibility='');}
        results.push({width,name});
      };
      await go('os-processes-virtual-memory-isolation');
      await inspect(page.locator('.separate-address'),'process-addresses');
      const os = page.locator('[data-investigation="os-translation"]');
      await os.getByRole('button',{name:'Offset 15',exact:true}).focus();
      await page.keyboard.press('Enter');
      assert.equal(await os.getByLabel('Virtual byte address').inputValue(),'31');
      for (let i=0;i<3;i++) await os.getByRole('button',{name:'Next step',exact:true}).click();
      assert.match(await os.locator('.nt-feedback').innerText(),/63/);
      await os.getByLabel('Selected process').selectOption('B');
      for (let i=0;i<3;i++) await os.getByRole('button',{name:'Next step',exact:true}).click();
      assert.match(await os.locator('.nt-feedback').innerText(),/95/);
      await inspect(os,'offset-translation');
      await os.getByRole('button',{name:'Offset 0',exact:true}).click();
      assert.equal(await os.getByLabel('Virtual byte address').inputValue(),'16');
      assert.ok(await os.getByRole('button',{name:'Back',exact:true}).isDisabled());
      await go('arrays-strings-hash-maps');
      await inspect(page.locator('.array-address'),'array-address');
      await go('linked-lists-stacks-queues');
      const ring = page.locator('[data-investigation="circular-queue"]');
      for (const capacity of [3,4]) {
        await ring.getByLabel('Buffer capacity').selectOption(String(capacity));
        const trace=ringTrace(capacity);
        for(let i=0;i<trace.length;i++) {
          const s=trace[i];
          assert.deepEqual(await ring.locator('.queue-ring-view .ring-value').allTextContents(),s.cells.map(v=>v??'·'));
          assert.ok((await ring.locator('.queue-ring-view svg').getAttribute('aria-label')).includes(`head ${s.head}, size ${s.size}`));
          if(i===6)await inspect(ring,`queue-wrap-${capacity}`);
          if(i<trace.length-1)await ring.getByRole('button',{name:'Next step',exact:true}).click();
        }
      }
      await go('hypothesis-testing-confidence-intervals');
      const interval=page.locator('.interval-decision');
      assert.match(await interval.locator('.mechanism-result').innerText(),/-0.323, 4.323/);
      await interval.getByRole('button',{name:'Add 1 ms to each saving'}).focus();
      await page.keyboard.press('Enter');
      assert.match(await interval.locator('.mechanism-result').innerText(),/0.677, 5.323/);
      await inspect(interval,'interval-decision');
      await interval.getByRole('button',{name:'Original five pairs'}).click();
      assert.match(await interval.locator('.mechanism-result').innerText(),/-0.323, 4.323/);
      await go('bayesian-inference-conjugate-priors');
      const beta=page.locator('.beta-update');
      assert.equal(await beta.locator('.beta-success').count(),8);
      assert.equal(await beta.locator('.beta-failure').count(),2);
      await inspect(beta,'beta-update');
      await go('spectral-graph-theory');
      const row=page.locator('.laplacian-row');
      assert.match(await row.locator('.laplacian-terms p').innerText(),/= 0.0/);
      await row.getByRole('button',{name:'Two group values'}).click();
      assert.match(await row.locator('.laplacian-terms p').innerText(),/= 0.4/);
      await inspect(row,'laplacian-row');
      await row.getByRole('button',{name:'Constant signal'}).click();
      assert.match(await row.locator('.laplacian-terms p').innerText(),/= 0.0/);
      await go('linux-basics-filesystems-processes');
      const linux=page.locator('.linux-lesson');
      assert.equal(await linux.locator('.lesson-lab').count(),4);
      await inspect(linux.locator('figure').first(),'linux-retained-figure');
      await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(`${dir}/browser-results.json`,JSON.stringify({results,errors},null,2));
    console.log(`PASS: ${results.length} desktop/mobile visual states, two complete queue traces per viewport, offset translation, keyboard and figure controls; no page errors.`);
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
