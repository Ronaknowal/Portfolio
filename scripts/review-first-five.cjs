const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const { learningPaths, getPathTopicIds } = await import('./lib/authoring-curriculum.mjs');
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const { measurementCases, publicationTrace, joinModel, transactionTrace } = await Promise.all([import("../src/learn/data/scientific-file-models.js"), import("../src/learn/data/sql-models.js")]).then(modules => Object.assign({}, ...modules));
  const ids = getPathTopicIds(learningPaths.find(p=>p.id === 'full-curriculum'));
  const expected = ['python-basics-types-control-flow-functions-modules','numpy-arrays-broadcasting-vectorization','scientific-file-formats-schemas-reliable-data-i-o','sql-relational-data-transactions-for-ml','object-oriented-programming-in-python'];
  expected.forEach(id=>assert.ok(ids.includes(id)));
  const counts = [3,4,2,2,4];
  const dir = 'scratch/first-five-review';
  fs.mkdirSync(dir, { recursive:true });
  const browser = await chromium.launch({ channel:'msedge', headless:true });
  const errors = [], results = [];
  const base = 'http://127.0.0.1:5173';
  try {
    for (const width of [1440,390]) {
      const page = await browser.newPage({ viewport:{width,height:1000} });
      const captureLab = async (lab, name) => {
        // Tall element captures can place fixed navigation across the diagram.
        // Suppress it only in these isolated captures, restoring it for checks.
        await page.locator('.learn-nav').evaluate(node => { node.style.visibility = 'hidden'; });
        try { await lab.screenshot({path:dir+'/'+name+'-'+width+'.png'}); }
        finally { await page.locator('.learn-nav').evaluate(node => { node.style.visibility = ''; }); }
      };
      page.on('pageerror', error=>errors.push(error.message));
      for (const [index,id] of expected.entries()) {
        await page.goto(base+'/learn/path/full-curriculum/'+id);
        await page.locator('.reader-article .lesson-intro').waitFor();
        assert.equal(await page.locator('.reader-header h1').innerText(),topicCatalogue[id].title);
        assert.equal((await page.locator('.reader-header .topic-status').innerText()).toLowerCase(),'published');
        assert.equal(await page.locator('.reader-article .lesson-lab, .reader-article .oop-lab').count(),counts[index],id);
        assert.ok(await page.locator('.reader-complete').isEnabled());
        const links = await page.locator('.reader-article a[href^="#"]').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('href').slice(1)));
        for(const anchor of links) assert.ok(await page.evaluate(id=>Boolean(document.getElementById(id)),anchor),'Missing lesson anchor: '+anchor);
        const topicLinks = await page.locator('.reader-article a[href^="/learn/topic/"]').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('href').split('/').pop()));
        topicLinks.forEach(target=>assert.ok(topicCatalogue[target],'Unknown linked topic: '+target));
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Document overflow: '+id+' at '+width);
        const nextId=ids[ids.indexOf(id)+1];
        assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[nextId].title));
        await page.screenshot({path:dir+'/lesson-'+(index+1)+'-'+width+'.png'});
        await page.locator('.reader-footer__next').click();
        await page.waitForURL(url=>url.pathname.endsWith('/'+nextId));
        await page.locator('.reader-header h1').filter({hasText:topicCatalogue[nextId].title}).waitFor();
      }
      await page.goto(base+'/learn/topic/'+expected[2]);
      const schema = page.locator('[data-lab="file-schema"]');
      await schema.waitFor();
      for (const scenario of measurementCases) {
        await schema.getByLabel('File variation').selectOption(scenario.id);
        await schema.getByRole('button',{name:'Next step',exact:true}).focus();
        await page.keyboard.press('Enter');
        await schema.getByRole('button',{name:'Next step',exact:true}).click();
        await schema.getByRole('button',{name:'Next step',exact:true}).click();
        if(scenario.id === 'valid') {
          assert.match(await schema.locator('.data-verdict').innerText(),/3 records, 2 measured values, 1 missing value/);
        } else {
          assert.match(await schema.locator('.data-verdict').innerText(),/Stop. Do not publish/);
          assert.ok(await schema.locator('.is-blocked').count()>0);
        }
        await schema.getByRole('button',{name:'Back',exact:true}).click();
        await schema.getByRole('button',{name:'Reset',exact:true}).click();
        assert.ok(await schema.getByRole('button',{name:'Back',exact:true}).isDisabled());
      }
      await schema.getByLabel('File variation').selectOption('valid');
      await schema.getByRole('button',{name:'Next step',exact:true}).click();
      await schema.getByRole('button',{name:'Next step',exact:true}).click();
      await captureLab(schema,'schema');
      const publication = page.locator('[data-lab="file-publication"]');
      for(const strategy of ['direct','replace']) for(const fail of [true,false]) {
        await publication.getByLabel('Write strategy').selectOption(strategy);
        await publication.getByLabel('Writer outcome').selectOption(fail?'fail':'success');
        const trace=publicationTrace(strategy,fail);
        for(const [index,state] of trace.entries()) {
          const files=await publication.locator('.data-state pre').allTextContents();
          assert.equal(files[0],state.destination||'(empty file)');
          assert.equal(files[1],state.temporary===null?'(no staging file)':state.temporary||'(empty staging file)');
          if(index<trace.length-1) await publication.getByRole('button',{name:'Next step',exact:true}).click();
        }
        await publication.getByRole('button',{name:'Reset',exact:true}).click();
      }
      await publication.getByRole('button',{name:'Next step',exact:true}).click();
      await publication.getByRole('button',{name:'Next step',exact:true}).click();
      await captureLab(publication,'publication');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));

      await page.goto(base+'/learn/topic/'+expected[3]);
      const joinLab=page.locator('[data-lab="sql-join"]');
      await joinLab.waitFor();
      for(const join of ['left','inner']) for(const duplicate of [false,true]) for(const cutoff of [0,10]) for(const placement of ['on','where']) {
        await joinLab.getByLabel('Join type').selectOption(join);
        await joinLab.getByLabel('Sensor keys').selectOption(duplicate?'duplicate':'unique');
        await joinLab.getByLabel('Include readings through minute').selectOption(String(cutoff));
        await joinLab.getByLabel('Time condition lives in').selectOption(placement);
        const model=joinModel({join,duplicate,cutoff,placement});
        assert.equal(await joinLab.locator('.data-lineage button').count(),model.pairs.length);
        for(let index=0;index<model.pairs.length;index++) {
          await joinLab.locator('.data-lineage button').nth(index).click();
          assert.equal(await joinLab.locator('.data-source-rows .is-linked').count(),model.pairs[index].reading?2:1);
        }
        const actual=await joinLab.locator('tbody tr').evaluateAll(rows=>rows.map(row=>[...row.cells].map(cell=>cell.textContent)));
        assert.deepEqual(actual,model.grouped.map(g=>[g.id,g.rows,g.readings,g.measured,g.mean??'NULL'].map(String)));
      }
      await joinLab.getByRole('button',{name:'Reset join',exact:true}).click();
      assert.ok(await joinLab.locator('table').evaluate(table=>table.getBoundingClientRect().width<=table.parentElement.clientWidth+1),'Aggregate columns must fit together');
      await captureLab(joinLab,'join');
      const transaction=page.locator('[data-lab="sql-transaction"]');
      for(const atomic of [true,false]) for(const fail of [true,false]) {
        await transaction.getByLabel('Transaction boundary').selectOption(atomic?'atomic':'separate');
        await transaction.getByLabel('Second statement').selectOption(fail?'fail':'success');
        const trace=transactionTrace({atomic,fail});
        for(const [index,state] of trace.entries()) {
          const values=await transaction.locator('.data-credit strong').allTextContents();
          assert.deepEqual(values,[...state.writer,...state.committed].map(String));
          assert.ok((await transaction.locator('.data-verdict').innerText()).includes(state.label));
          if(index<trace.length-1) await transaction.getByRole('button',{name:'Next step',exact:true}).click();
        }
        await transaction.getByRole('button',{name:'Reset',exact:true}).click();
      }
      await transaction.getByLabel('Transaction boundary').selectOption('atomic');
      await transaction.getByRole('button',{name:'Next step',exact:true}).click();
      await transaction.getByRole('button',{name:'Next step',exact:true}).click();
      await captureLab(transaction,'transaction');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      const hint=page.getByText('Hint: work sensor by sensor',{exact:true});
      await hint.focus();
      await page.keyboard.press('Enter');
      assert.equal(await hint.locator('..').getAttribute('open'),'');
      results.push({width,topics:5,labs:counts.reduce((a,b)=>a+b,0),checks:'five published sequential topics, named next links, anchors and related links, full schema/publication/join/transaction controls against models, keyboard, hints, reset/back and no document overflow'});
      await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(dir+'/browser-results.json',JSON.stringify({results,errors},null,2));
    console.log('PASS: all five published lessons and 15 focused labs; exact sequence, links, File/SQL interactions and keyboard at 1440/390; no page errors or document overflow.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
