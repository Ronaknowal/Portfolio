const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/linear-logistic-independent-review');
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge' });
  const records = [];
  try {
    for (const width of [1440,390,320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 } });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/linear-logistic-regression', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.linear-logistic-lesson');
      await lesson.waitFor({timeout:60000});
      await page.evaluate(() => document.fonts.ready);
      const capture = async (locator, name) => {
        const retained = new Set(['mean-baseline-390', 'changed-fit-scrolled-320', 'changed-gradient-1440', 'wrong-label-390', 'reading-3-320', 'reading-8-390', 'reading-11-320', 'capstone-1440']);
        if (!retained.has(name+'-'+width)) return;
        await locator.first().evaluate(node => scrollTo({top:node.getBoundingClientRect().top+scrollY-85,behavior:'instant'}));
        await page.waitForTimeout(100);
        await page.screenshot({path:path.join(directory,name+'-'+width+'.png')});
      };
      const lab = name => lesson.locator('[data-investigation="'+name+'"]');
      const residuals = lab('regression-residuals');
      await residuals.getByRole('button',{name:'Use mean baseline'}).click();
      const baseline = { slider: await residuals.getByRole('slider',{name:/Intercept/}).inputValue(),
        readout: await residuals.locator('.regression-readout').innerText(),
        label: await residuals.locator('.regression-control').first().innerText() };
      console.log(JSON.stringify({width,baseline}));
      await capture(residuals,'mean-baseline');
      await residuals.getByRole('combobox').selectOption('8');
      await residuals.getByRole('button',{name:'Fit least squares'}).click();
      assert((await residuals.locator('.regression-readout').innerText()).includes('0.100 + 2.100x'));
      const pathPoints = await residuals.locator('polyline.regression-line').getAttribute('points');
      const mapped = pathPoints.split(' ').map(pair => pair.split(',').map(Number));
      assert(Math.abs(mapped[0][1] - (212-(.1+4)/18*190))<1e-10);
      assert(Math.abs(mapped[1][1] - (212-(6.4+4)/18*190))<1e-10);
      const scroll = residuals.locator('.regression-chart-scroll');
      await scroll.focus(); await page.keyboard.press('End'); await page.waitForTimeout(140);
      const scrolling = await scroll.evaluate(n=>({offset:n.scrollLeft,width:n.clientWidth,content:n.scrollWidth}));
      if(width<500) assert(scrolling.offset>0);
      await capture(scroll,'changed-fit-scrolled');
      const gradient = lab('regression-gradient');
      await gradient.getByRole('combobox').selectOption('0.2');
      for(let i=0;i<4;i++) await gradient.getByRole('button',{name:'Next step'}).click();
      const contours = await gradient.locator('polyline.regression-contour').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('points')));
      const losses = contours.map(text=>text.split(' ').map(pair=>{
        const [px,py]=pair.split(',').map(Number); const db=-.5+(px-58)/474*3-.9,dw=2.5-(py-22)/190*3-.9;
        return db*db+3*db*dw+3.5*dw*dw;
      }));
      for(const values of losses) assert(Math.max(...values)-Math.min(...values)<1e-10);
      await capture(gradient,'changed-gradient');
      const score = lab('regression-logistic-score');
      await score.getByRole('slider',{name:/Intercept/}).fill('2');
      await score.getByRole('slider',{name:/Weight/}).fill('-1.5');
      await score.getByRole('slider',{name:/Feature/}).fill('-2');
      assert((await score.locator('.regression-calculation').innerText()).includes('5.000'));
      const pointBefore = await score.locator('circle').getAttribute('cy');
      await score.getByRole('combobox').selectOption('0');
      assert.equal(await score.locator('circle').getAttribute('cy'),pointBefore);
      assert((await score.innerText()).includes('5.007'));
      await capture(score,'wrong-label');
      const threshold = lab('regression-threshold');
      await threshold.getByRole('slider').fill('0.35');
      assert((await threshold.innerText()).includes('Total cost = FP + 4×FN = 6'));
      await threshold.getByRole('slider').fill('1');
      assert((await threshold.innerText()).includes('precision undefined'));
      await capture(threshold.locator('[aria-live]'),'no-warnings');
      const separation = lab('regression-separation');
      await separation.getByRole('combobox').selectOption('0.02');
      await separation.getByRole('slider').fill('6');
      assert((await separation.innerText()).includes('0.362476'));
      await capture(separation,'penalized-loss');
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(n=>n.open=true));
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map((n,i)=>{
        const b=n.getBoundingClientRect();
        const children=[...n.querySelectorAll('.katex-html span')].map(s=>s.getBoundingClientRect()).filter(r=>r.width>0);
        return {i,boxWidth:b.width,content:n.scrollWidth,left:b.left,right:b.right,actualLeft:Math.min(...children.map(r=>r.left)),actualRight:Math.max(...children.map(r=>r.right))};
      }));
      for(const i of [2,7,10]) await capture(lesson.locator('h2').nth(i),'reading-'+(i+1));
      await capture(lesson.locator('figure').last(),'uncertainty');
      await capture(lesson.locator('.lesson-check').last(),'capstone');
      const programs = await lesson.locator('.python-example').count();
      assert.equal(programs,10);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      records.push({width,baseline,scrolling,contourLevels:losses.map(v=>v[0]),equations,errors,programs});
      fs.writeFileSync(path.join(directory,'browser.json'),JSON.stringify({checkedAt:new Date().toISOString(),complete:records.length===3&&records.every(r=>r.errors.length===0),records},null,2));
      assert.deepEqual(errors,[]);
      await page.close();
    }
    fs.writeFileSync(path.join(directory,'browser.json'),JSON.stringify({checkedAt:new Date().toISOString(),records},null,2));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
