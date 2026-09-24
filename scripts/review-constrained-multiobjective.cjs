const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const folder = path.resolve('scratch/constrained-multiobjective-browser');
fs.mkdirSync(folder, { recursive: true });

async function range(lab, label, value) {
  await lab.getByRole('slider', { name: label, exact: true }).evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}
async function state(lab) { return JSON.parse(await lab.getAttribute('data-state')); }
async function capture(page, element, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await element.screenshot({ path: path.join(folder, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function geometry(lesson) {
  const clipped = await lesson.locator('.constrained-plot').evaluateAll(nodes => nodes.flatMap((svg, index) => {
    const border = svg.getBoundingClientRect();
    return [...svg.querySelectorAll('text')].filter(node => {
      const box = node.getBoundingClientRect();
      return box.left < border.left - 1 || box.right > border.right + 1 || box.top < border.top - 1 || box.bottom > border.bottom + 1;
    }).map(node => ({ index, text: node.textContent }));
  }));
  assert.deepEqual(clipped, [], 'SVG text within viewport');
}

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/constrained-multiobjective-models.js')));
  const { constrainedExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/constrained-multiobjective-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/constrained-multi-objective-optimization');
      const lesson = page.locator('.constrained-lesson');
      await lesson.waitFor();
      const counts = { projection: 0, penalty: 0, admm: 0, pareto: 0, continuous: 0 };
      const projection = lesson.locator('[data-lab="coupled-projection"]');
      for (const target of [[2,-0.6],[0.5,3],[-2,4],[4,-2],[-2,-2],[4,4]]) {
        await range(projection, 'Target coordinate 1', target[0]);
        await range(projection, 'Target coordinate 2', target[1]);
        for (const budget of [0.5,1,2,3]) {
          await range(projection, 'Required sum B', budget);
          for (const order of ['orthant-first','line-first']) {
            await projection.getByRole('combobox').selectOption(order);
            const expected = models.coupledProjectionState(target,budget,order);
            assert.equal((await state(projection)).step,0);
            for (let step = 1; step <= 2; step++) {
              await projection.getByRole('button',{name:'Next step',exact:true}).click();
              const actual = await state(projection);
              assert.deepEqual(actual,JSON.parse(JSON.stringify({...expected,step})));
              assert.equal(await projection.locator('dd').nth(1).innerText(),models.constraintNumber(expected.frames[step].nonnegativeViolation));
              counts.projection++;
            }
            assert(await projection.getByRole('button',{name:'Next step',exact:true}).isDisabled());
          }
        }
      }
      await range(projection,'Target coordinate 1',2); await range(projection,'Target coordinate 2',-0.6); await range(projection,'Required sum B',1);
      await projection.getByRole('combobox').selectOption('orthant-first');
      await projection.getByRole('button',{name:'Final step'}).click();
      await capture(page,projection,`projection-${width}.png`);
      await projection.getByRole('button',{name:'Previous step'}).click();
      assert.equal((await state(projection)).step,1);
      await projection.getByRole('button',{name:'Reset trace'}).click();
      await projection.getByRole('button',{name:'Next step',exact:true}).focus(); await page.keyboard.press('Space');
      assert.equal((await state(projection)).step,1);

      const penalty = lesson.locator('[data-lab="constraint-penalty"]');
      for (const mode of ['quadratic','hinge','barrier']) {
        await penalty.getByRole('combobox').selectOption(mode);
        const values = mode === 'quadratic' ? [0,2,19,20] : mode === 'hinge' ? [0,1.9,2,2.1,5] : [0.01,0.1,0.5,4];
        for (const value of values) {
          await range(penalty,mode === 'barrier' ? 'Barrier weight τ' : 'Penalty strength',value);
          const expected = models.constraintPenaltyState(mode,value);
          assert.deepEqual(await state(penalty),expected);
          assert.equal(await penalty.locator('dd').first().innerText(),models.constraintNumber(expected.optimum));
          counts.penalty++;
        }
        await capture(page,penalty,`penalty-${mode}-${width}.png`);
      }
      const admm = lesson.locator('[data-lab="consensus-admm"]');
      for (const target of [[2,-0.6],[1.5,-0.5],[-2,4],[4,4]]) {
        await range(admm,'ADMM target coordinate 1',target[0]); await range(admm,'ADMM target coordinate 2',target[1]);
        for (const budget of [0.5,1,3]) {
          await range(admm,'ADMM required sum',budget);
          for (const rho of [0.1,0.3,1,3,10]) {
            await admm.getByRole('combobox').selectOption(String(rho));
            assert.equal((await state(admm)).step,0);
            const expected = models.consensusAdmmState(target,budget,rho,80);
            for (const step of [1,20,80]) {
              await range(admm,'ADMM iteration',step);
              assert.deepEqual(await state(admm),JSON.parse(JSON.stringify({...expected,step})));
              const frame=expected.frames[step];
              assert.equal(await admm.locator('dd').nth(3).innerText(),`${models.constraintNumber(frame.primalNorm)} / ${models.constraintNumber(frame.primalTolerance)}`);
              counts.admm++;
            }
          }
        }
      }
      await range(admm,'ADMM target coordinate 1',2);await range(admm,'ADMM target coordinate 2',-0.6);await range(admm,'ADMM required sum',1);await admm.getByRole('combobox').selectOption('1');
      assert.match(await admm.locator('[role=status]').innerText(),/Initial primal residual is zero/);
      await admm.getByRole('button',{name:'Next step',exact:true}).click();
      await capture(page,admm,`admm-first-${width}.png`);
      await admm.getByRole('button',{name:'Final step'}).click();
      assert.match(await admm.locator('[role=status]').innerText(),/Both numerical stopping tests pass/);
      await capture(page,admm,`admm-final-${width}.png`);
      await admm.getByRole('button',{name:'Previous step'}).click();assert.equal((await state(admm)).step,79);
      await admm.getByRole('button',{name:'Reset trace'}).click();assert.equal((await state(admm)).step,0);

      const pareto=lesson.locator('[data-lab="pareto-decision"]');
      for(const latency of [0,8,15,25,80]) for(const memory of [0,32,48,96,400]) {
        await range(pareto,'Maximum latency · ms',latency);await range(pareto,'Maximum memory · MB',memory);
        for(const method of ['error','latency','weighted']) {
          await pareto.getByRole('combobox').selectOption(method);
          if(method==='weighted') await range(pareto,'Price · error percentage points / ms',0.4);
          const actual=await state(pareto),expected=models.paretoDecisionState(latency,memory,actual.price,method);
          assert.deepEqual(actual,expected);
          assert.equal(await pareto.locator('dd').first().innerText(),expected.selected ? expected.candidates.find(item=>item.id===expected.selected).name : 'No feasible candidate');
          counts.pareto++;
        }
      }
      await range(pareto,'Maximum latency · ms',15);await range(pareto,'Maximum memory · MB',400);await pareto.getByRole('combobox').selectOption('error');
      assert.equal((await state(pareto)).selected,'C');await capture(page,pareto,`pareto-compact-${width}.png`);
      await range(pareto,'Maximum memory · MB',0);assert.match(await pareto.locator('[role=status]').innerText(),/no feasible choice/);await capture(page,pareto,`pareto-empty-${width}.png`);
      await range(pareto,'Maximum memory · MB',400);
      const continuous=lesson.locator('[data-lab="continuous-tradeoff"]');
      for(const method of ['weighted','epsilon']) {
        await continuous.getByRole('combobox').selectOption(method);
        for(const value of method==='weighted'?[0,0.25,0.5,0.75,1]:[0,0.01,0.25,1,2.25,4]) {
          await range(continuous,method==='weighted'?'First-objective weight α':'Second-objective bound ε',value);
          const actual=await state(continuous),expected=models.continuousTradeoffState(method,actual.alpha,actual.epsilon);
          assert.deepEqual(actual,expected);assert.equal(await continuous.locator('dd').first().innerText(),models.constraintNumber(expected.optimum));counts.continuous++;
        }
      }
      await range(continuous,'Second-objective bound ε',0.25);await capture(page,continuous,`continuous-${width}.png`);
      await geometry(lesson);
      const anchors=await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>node.hash.slice(1)));
      assert.equal(anchors.length,9);
      for(const anchor of anchors) {assert.equal(await page.locator(`[id="${anchor}"]`).count(),1,anchor);await lesson.locator(`.lesson-intro a[href="#${anchor}"]`).click();await page.waitForFunction(id=>{const r=document.getElementById(id).getBoundingClientRect();return r.top>=-2&&r.top<innerHeight;},anchor);}
      assert.equal(await lesson.locator('.constrained-practice').count(),7);
      await lesson.locator('.constrained-practice summary').first().focus();await page.keyboard.press('Enter');assert.equal(await lesson.locator('.constrained-practice details[open]').count(),1);
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>node.open=true));
      const text=await lesson.innerText();
      for(const [key,example] of Object.entries(examples)){assert(text.includes(example.code),key+' full code');assert(text.includes(example.expected),key+' full output');}
      assert.equal(await lesson.locator('.katex-error').count(),0);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      assert.deepEqual(errors,[]);
      results.push({width,counts,anchors:9,fullPrograms:11,practiceGroups:7,sources:await lesson.locator('.lesson-sources a').count(),errors,pageOverflow:false});
      await page.close();
    }
  } finally {await browser.close();}
  fs.writeFileSync(path.join(folder,'results.json'),JSON.stringify({at:new Date().toISOString(),status:'passed',results},null,2));console.log(JSON.stringify(results,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
