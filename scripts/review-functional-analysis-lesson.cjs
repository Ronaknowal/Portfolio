const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/functional-analysis-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const { functionalAnalysisExamples: examples } = await import('../src/learn/data/functional-analysis-examples.js');
  const model = await import('../src/learn/data/functional-analysis-models.js');
  const browser = await chromium.launch({ channel:'msedge', headless:true });
  const records = [];
  try {
    for (const width of [1440,390,320]) {
      const page = await browser.newPage({ viewport:{width,height:1080}, reducedMotion:'reduce' });
      const errors=[];
      page.on('pageerror', error=>errors.push(error.message));
      page.on('console', message=>{if(message.type()==='error') errors.push(message.text());});
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/functional-analysis-rkhs',{waitUntil:'domcontentloaded',timeout:60000});
      const lesson=page.locator('.functional-analysis-lesson');
      await lesson.waitFor({timeout:60000});
      await page.evaluate(()=>document.fonts.ready);
      const fontLoaded = await page.evaluate(()=>document.fonts.check('16px "Space Grotesk"'));
      assert(fontLoaded,'Intended Space Grotesk font');
      const capture = async (locator,name) => {
        await locator.first().evaluate(node=>window.scrollTo({top:node.getBoundingClientRect().top+scrollY-84,behavior:'instant'}));
        await page.waitForTimeout(220);
        await page.screenshot({path:path.join(directory,`${name}-${width}.png`)});
      };
      const setRange=async(label,value)=>{
        const locator=lesson.getByRole('slider',{name:label,exact:true});
        await locator.fill(String(value));
        await locator.dispatchEvent('input');
      };
      assert.equal(await lesson.locator('.functional-lab').count(),7);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const headings=lesson.locator('h2');
      assert.equal(await headings.count(),11);
      for (let i=0;i<await headings.count();i+=1) await capture(headings.nth(i),`reading-${i+1}`);
      const broken=await lesson.locator('nav a').evaluateAll(links=>links.map(a=>a.hash).filter(hash=>!document.getElementById(hash.slice(1))));
      assert.deepEqual(broken,[]);
      // Open a real checkpoint and verify both its question and answer contract.
      const checkpoint=lesson.locator('.lesson-check').filter({hasText:'Does completeness say every bounded sequence converges?'});
      await checkpoint.locator('summary').click();
      assert((await checkpoint.innerText()).includes('bounded but not Cauchy'));
      const anchorCheckpoint=lesson.locator('.lesson-check').filter({hasText:'Why not use derivative energy on all differentiable functions without an anchor?'});
      await anchorCheckpoint.locator('summary').click();
      assert((await anchorCheckpoint.innerText()).includes('nonzero constants'));

      const spike=lesson.getByRole('region',{name:'Spike and evaluation investigation'});
      // Sections expose aria-label but no implicit region role without a heading association in some browsers; fallback is exact DOM.
      const labs=Object.fromEntries(await lesson.locator('.functional-lab').evaluateAll(nodes=>nodes.map(n=>[n.getAttribute('aria-label'),true])));
      const findLab=name=>lesson.locator(`.functional-lab[aria-label="${name}"]`);
      await lesson.getByRole('button',{name:'Narrow spike',exact:true}).click();
      assert((await findLab('Spike and evaluation investigation').innerText()).includes(model.functionalNumber(2*.025/3)));
      await capture(findLab('Spike and evaluation investigation'),'spike');
      await lesson.getByRole('button',{name:'Reset spike'}).click();
      const slider=lesson.getByRole('slider',{name:'Spike half-width'});
      await slider.focus();const before=Number(await slider.inputValue());await page.keyboard.press('ArrowRight');
      assert(Number(await slider.inputValue())>before);
      await lesson.getByRole('button',{name:'Reset spike'}).click();

      const evaluation=findLab('Evaluation kernel investigation');
      await lesson.getByRole('button',{name:'Attain the bound'}).click();
      assert((await evaluation.innerText()).includes('Absolute value 1 ≤ 1'));
      await capture(evaluation,'evaluation-attainment');
      await lesson.getByRole('textbox',{name:'Four slope values'}).fill('2, , 1, 0');
      await lesson.getByRole('button',{name:'Apply slopes'}).click();
      assert.equal(await evaluation.getByRole('alert').count(),1);
      assert((await evaluation.innerText()).includes('Absolute value 1 ≤ 1'));
      await capture(evaluation,'evaluation-error');
      await lesson.getByRole('textbox',{name:'Four slope values'}).fill('-4, 4, -4, 4');
      await lesson.getByRole('button',{name:'Apply slopes'}).click();
      assert.equal(await evaluation.getByRole('alert').count(),0);
      await lesson.getByRole('button',{name:'Read the anchor'}).click();
      assert((await evaluation.innerText()).includes('Absolute value 0 ≤ 0'));
      await lesson.getByRole('button',{name:'Reset evaluation'}).click();

      const validity=findLab('Kernel validity investigation');
      await lesson.getByRole('combobox',{name:'Kernel construction'}).selectOption('invalid');
      assert((await validity.innerText()).includes('-0.6'));
      await capture(validity,'invalid-gram');
      await lesson.getByRole('combobox',{name:'Kernel construction'}).selectOption('bigrams');
      assert((await validity.innerText()).includes('ABAB'));
      await lesson.getByRole('button',{name:'Reset kernel'}).click();

      const representer=findLab('Representer geometry investigation');
      await representer.getByRole('checkbox').check();
      assert((await representer.innerText()).includes('no longer orthogonal'));
      await setRange('Wiggle amplitude',-1);
      await capture(representer,'changed-wiggle');
      await lesson.getByRole('button',{name:'Remove unseen component'}).click();
      assert((await representer.innerText()).includes('Total energy\n4'));
      await lesson.getByRole('button',{name:'Reset wiggle'}).click();

      const ridge=findLab('Kernel ridge investigation');
      assert((await ridge.innerText()).includes(model.functionalNumber(model.kernelRidgeState().prediction)));
      await lesson.getByRole('combobox',{name:'Observed data'}).selectOption('duplicates');
      await setRange('RBF gamma',4);
      assert((await ridge.innerText()).includes('Diagonal shift nλ\n0.2'));
      await capture(ridge,'duplicate-ridge');
      await lesson.getByRole('combobox',{name:'Observed data'}).selectOption('curve');
      await setRange('Average-loss lambda',.0001); await setRange('RBF gamma',.05);
      await capture(ridge,'ridge-extrapolation');
      await lesson.getByRole('button',{name:'Reset ridge'}).click();

      const means=findLab('Kernel distribution witness investigation');
      await capture(means,'distribution-witness');
      await lesson.getByRole('button',{name:'Set Q equal to P'}).click();
      assert((await means.innerText()).includes('Gaussian MMD²\n0'));
      await lesson.getByRole('button',{name:'Reset witness'}).click();

      const quadrature=findLab('Kernel quadrature investigation');
      await lesson.getByRole('button',{name:'Best one-node location'}).click();
      assert((await quadrature.innerText()).includes(model.functionalNumber(1/27)));
      assert(Math.abs(Number(await lesson.getByRole('slider',{name:'Quadrature node',exact:true}).inputValue())-2/3)<1e-12);
      await capture(quadrature,'quadrature-optimum');
      await quadrature.getByRole('checkbox').check(); await setRange('Quadrature weight',-1);
      await capture(quadrature,'quadrature-manual');
      await lesson.getByRole('button',{name:'Observe only the anchor'}).click();
      assert((await quadrature.innerText()).includes('Any weight has the same error'));
      await lesson.getByRole('button',{name:'Reset quadrature'}).click();

      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>node.open=true));
      const mathWidths=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map((node,index)=>({index,width:node.clientWidth,content:node.scrollWidth})));
      assert.deepEqual(mathWidths.filter(item=>item.content>item.width+2),[],'equations fit the ordinary reading width');
      assert.equal(await lesson.locator('.python-example').count(),examples.length);
      for(const example of examples){
        const container=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:example.title,exact:true})});
        const text=(await container.innerText()).replace(/\r\n/g,'\n');
        assert(text.includes(example.code.replace(/\r\n/g,'\n').trim()),'actual code '+example.id);
        assert(text.includes(example.expected.trim()),'actual output '+example.id);
        assert((await lesson.innerText()).includes(example.question),'actual question '+example.id);
      }
      const practice=lesson.locator('section.lesson-check');
      assert.equal(await practice.count(),9);
      for(let i=0;i<9;i++)assert((await practice.nth(i).locator('details').nth(1).innerText()).length>170);
      await capture(practice.nth(6),'practice-quadrature');
      await capture(lesson.locator('.functional-inline').first(),'completion');
      const sequence=lesson.locator('.functional-sequence');
      await sequence.focus(); await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(250);
      if(width<620) assert(await sequence.evaluate(node=>node.scrollLeft)>0,'keyboard access to sequence coordinates');
      const formulas=lesson.locator('.katex-display');
      for(const index of [0,2,3,5,7,11,12,13]) await capture(formulas.nth(index),`equation-${index}`);
      await capture(lesson.locator('.lesson-sources'),'sources');
      const geometry=await page.evaluate(()=>({width:innerWidth,document:document.documentElement.scrollWidth,font:getComputedStyle(document.querySelector('.functional-analysis-lesson p')).fontFamily}));
      assert(geometry.document<=width+1,JSON.stringify(geometry));
      assert.equal(await lesson.locator('.katex-error').count(),0);
      assert.deepEqual(errors,[]);
      records.push({width,geometry,fontLoaded,labCount:Object.keys(labs).length,readingSections:11,examples:examples.length,practice:9,checkpoints:2,keyboardArrow:true,sequenceKeyboardScroll:true,quadratureSliderAgreement:true,mathWidths,invalidStatePreserved:true,errors});
      await page.close();
    }
  }finally{await browser.close();}
  const result={at:new Date().toISOString(),passed:true,records};
  fs.writeFileSync(path.join(directory,'results.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exit(1);});
