const assert = require('node:assert/strict');
module.exports = async function reviewFamilies(page, lesson, width, directory) {
  const { matrixChainPlan, treeBoundaryPlan, treeBoundaryWitness, treePresetEdges, createDigitCounter } = await import('../src/learn/data/dp-state-families-models.js');
  const { dpStateFamiliesExamples } = await import('../src/learn/data/dp-state-families-examples.js');
  const counts = { interval: 0, tree: 0, digit: 0, rejectedDrafts: 0 };
  async function capture(target, name) {
    await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 90, behavior: 'instant' }));
    await page.screenshot({ path: `${directory}/${name}-${width}.png` });
  }
  async function reset(region, name) {
    await region.getByRole('button', { name, exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await region.getByRole('alert').count(), 0);
  }
  const interval = lesson.getByRole('region', { name: 'Interval split investigation', exact: true });
  for (const dimensions of [[8,2,12,3,6], [3,7,2,5], [2,2,2,2], [3,7], [2,3,4,5,6,7,8]]) {
    await interval.getByLabel('Matrix-chain dimensions', { exact: true }).fill(dimensions.join(','));
    await interval.getByRole('button', { name: 'Apply dimensions', exact: true }).click();
    const model = matrixChainPlan(dimensions);
    assert.equal(await interval.locator('.dpf-expression').innerText(), model.expression);
    for (const cell of model.cells) {
      await interval.getByRole('button', { name: `Inspect interval ${cell.left} to ${cell.right}`, exact: true }).click();
      if (!cell.candidates.length) {
        assert.match(await interval.locator('[data-dpf-status]').innerText(), /cost 0, no multiplication/);
        counts.interval++;
      }
      for (const candidate of cell.candidates) {
        await interval.getByLabel('Candidate final split', { exact: true }).selectOption(String(candidate.split));
        assert.equal(await interval.locator('[data-dpf-status]').innerText(), `Left ${candidate.first} + right ${candidate.second} + final ${candidate.merge} = ${candidate.total}. Best interval cost: ${cell.cost}.`);
        const tree = interval.getByRole('region', { name: 'Candidate expression tree', exact: true });
        assert.equal(await tree.locator('g').count(), 2 * (cell.right-cell.left)-1);
        assert.equal(await tree.locator('path').count(), 2 * (cell.right-cell.left)-2);
        assert.equal(await interval.locator('.dpf-expression').innerText(), model.expression);
        counts.interval++;
      }
    }
    for (let index = 0; index < model.operations.length; index++) {
      assert.equal(await interval.locator('.dpf-operation-list [aria-current="step"]').count(), 1);
      const active = await interval.locator('.dpf-operation-list [aria-current="step"]').innerText();
      assert(active.includes(`${model.operations[index].cost} multiplications`));
      if (index+1 < model.operations.length) await interval.getByRole('button', { name: 'Next multiplication', exact: true }).click();
    }
  }
  const retainedExpression = await interval.locator('.dpf-expression').innerText();
  for (const invalid of ['3,,4','1,0,2','1.5,2','21,2','1,2,3,4,5,6,7,8','']) {
    await interval.getByLabel('Matrix-chain dimensions', { exact: true }).fill(invalid);
    await interval.getByRole('button', { name: 'Apply dimensions', exact: true }).click();
    assert(await interval.getByRole('alert').isVisible());
    assert.equal(await interval.locator('.dpf-expression').innerText(), retainedExpression);
    counts.rejectedDrafts++;
  }
  await reset(interval, 'Reset interval lab');
  await capture(interval.getByRole('heading', { name: 'Inspect [0,4) · result shape 8×6', exact: true }), 'interval-default-reading');
  await capture(interval.locator('.dpf-expression'), 'interval-reconstruction-reading');
  await interval.getByLabel('Candidate final split', { exact: true }).focus();
  await page.keyboard.press('End');
  await page.keyboard.press('Enter');
  assert.equal(await interval.getByLabel('Candidate final split', { exact: true }).inputValue(), '3');
  await capture(interval.getByRole('heading', { name: 'Inspect [0,4) · result shape 8×6', exact: true }), 'interval-changed-split-reading');
  await reset(interval, 'Reset interval lab');

  const tree = lesson.getByRole('region', { name: 'Tree parent boundary investigation', exact: true });
  for (const weights of [[5,9,2,4,1,6,3], [4,3,3,3], [-5,-2,0], [7], [], [2,1,3,4,5,6,7,8,9]]) {
    await tree.getByLabel('Node weights in ID order', { exact: true }).fill(weights.join(','));
    await tree.getByRole('button', { name: 'Apply tree weights', exact: true }).click();
    for (const shape of ['binary','chain','star']) {
      await tree.getByLabel('Tree shape', { exact: true }).selectOption(shape);
      const plan = treeBoundaryPlan(weights, treePresetEdges(weights.length, shape));
      if (!weights.length) {
        assert.match(await tree.locator('[data-dpf-status]').innerText(), /Empty tree: value 0/);
        counts.tree++;
      }
      for (let node=0; node<weights.length; node++) for (const parent of [false,true]) {
        await tree.getByLabel('Subtree root', { exact: true }).selectOption(String(node));
        await tree.getByLabel('External parent selected', { exact: true }).setChecked(parent);
        const witness = treeBoundaryWitness(plan,node,parent);
        assert.equal(await tree.locator('[data-dpf-status]').innerText(), `Requested F(${node},${Number(parent)}) = ${witness.result}; selected IDs: ${witness.selected.join(', ') || 'none'}.`);
        assert.equal(await tree.locator('.dpf-selected-node').count(), witness.selected.length);
        assert.equal(await tree.locator('svg g').count(), weights.length);
        assert.equal(await tree.locator('svg path').count(), Math.max(0,weights.length-1));
        counts.tree++;
      }
    }
  }
  const treeRetained = await tree.locator('[data-dpf-status]').innerText();
  for (const invalid of ['3,,4','-10,2','1,21','1.5','NaN','1,2,3,4,5,6,7,8,9,10']) {
    await tree.getByLabel('Node weights in ID order', { exact: true }).fill(invalid);
    await tree.getByRole('button', { name: 'Apply tree weights', exact: true }).click();
    assert(await tree.getByRole('alert').isVisible());
    assert.equal(await tree.locator('[data-dpf-status]').innerText(),treeRetained);
    counts.rejectedDrafts++;
  }
  await reset(tree, 'Reset tree lab');
  await capture(tree.getByRole('region', { name: 'Weighted tree and conditional witness', exact: true }), 'tree-free-reading');
  await tree.getByLabel('External parent selected', { exact: true }).focus();
  await page.keyboard.press('Space');
  assert.equal(await tree.getByLabel('External parent selected', { exact: true }).isChecked(),true);
  assert.match(await tree.locator('[data-dpf-status]').innerText(), /F\(1,1\) = 5/);
  await capture(tree.getByRole('region', { name: 'Weighted tree and conditional witness', exact: true }), 'tree-parent-selected-reading');
  await tree.getByRole('button', { name: 'Previous postorder node', exact: true }).click();
  await tree.getByRole('button', { name: 'Next postorder node', exact: true }).click();
  assert.equal(await tree.getByLabel('Subtree root', { exact: true }).inputValue(),'1');
  await capture(tree.getByRole('heading', { name: 'Inspect the completed table in child-before-parent order', exact: true }), 'tree-postorder-reading');
  await reset(tree, 'Reset tree lab');

  const digit = lesson.getByRole('region', { name: 'Digit prefix investigation', exact: true });
  for (const bound of [0,9,99,100,102,213,999999]) {
    await digit.getByLabel('Inclusive upper bound', { exact: true }).fill(String(bound));
    await digit.getByRole('button', { name: 'Apply digit bound', exact: true }).click();
    const counter = createDigitCounter(bound);
    const prefixes = new Set(['']);
    for (const value of [0,7,10,12,21,90,102,213,987654]) {
      const word = String(value).padStart(String(bound).length,'0');
      if (word.length !== String(bound).length) continue;
      for (let length=1; length<=word.length; length++) {
        const prefix=word.slice(0,length);
        try { counter.inspect(prefix); prefixes.add(prefix); } catch {}
      }
    }
    for (const prefix of prefixes) {
      if (!(await digit.getByRole('button', { name: 'Return to empty prefix', exact: true }).isDisabled())) await digit.getByRole('button', { name: 'Return to empty prefix', exact: true }).click();
      for (const next of prefix) await digit.getByRole('button', { name: new RegExp(`^Append digit ${next}:`) }).click();
      const state = counter.inspect(prefix);
      assert.equal(await digit.locator('[data-dpf-status]').innerText(), `Prefix ${prefix || '(empty)'} has ${state.remaining} valid positive completions.${state.completion !== null ? ` One is ${state.completion}.` : ' No positive integer completes this prefix.'}`);
      assert.equal(await digit.locator('.dpf-used').count(),state.state.used.toString(2).replaceAll('0','').length);
      for (const branch of state.branches) {
        const button=digit.getByRole('button', { name: new RegExp(`^Append digit ${branch.digit}:`) });
        assert.equal(await button.isDisabled(),!branch.allowed);
        if (branch.allowed) assert((await button.getAttribute('aria-label')).includes(`${branch.count} completions`));
      }
      counts.digit++;
    }
  }
  const digitRetained=await digit.locator('[data-dpf-status]').innerText();
  for (const invalid of ['-1','1000000','1e3','001','1.5','']) {
    await digit.getByLabel('Inclusive upper bound', { exact: true }).fill(invalid);
    await digit.getByRole('button', { name: 'Apply digit bound', exact: true }).click();
    assert(await digit.getByRole('alert').isVisible());
    assert.equal(await digit.locator('[data-dpf-status]').innerText(),digitRetained);
    counts.rejectedDrafts++;
  }
  await reset(digit,'Reset digit lab');
  await digit.getByRole('button', { name:'Inspect prefix 12', exact:true }).click();
  await capture(digit.locator('.dpf-prefix-strip'),'digit-loose-reading');
  await digit.getByRole('button', { name:'Inspect prefix 21', exact:true }).focus();
  await page.keyboard.press('Enter');
  assert.match(await digit.locator('[data-dpf-status]').innerText(), /Prefix 21 has 2 /);
  await capture(digit.locator('.dpf-prefix-strip'),'digit-tight-reading');
  await digit.getByRole('button',{name:/^Append digit 0:/}).click();
  assert.match(await digit.innerText(),/End of spelling: 210 is one valid number/);
  await digit.getByRole('button',{name:'Return to empty prefix',exact:true}).click();
  for (let index=0;index<3;index++) await digit.getByRole('button',{name:/^Append digit 0:/}).click();
  assert.match(await digit.innerText(),/represents 0, which is excluded/);
  await capture(digit.locator('.dpf-prefix-strip'),'digit-zero-excluded-reading');
  await reset(digit,'Reset digit lab');
  for (const [index,figure] of (await lesson.locator('.dpf-inline').all()).entries()) await capture(figure,`family-inline-${index}`);
  const rows = await lesson.locator('.dpf-balloon-row,.dpf-shape-plan>div').evaluateAll(nodes => nodes.map(node => {
    const rectangles=[...node.children].map(child=>child.getBoundingClientRect());
    return {kind:node.className,centers:rectangles.map(rectangle=>rectangle.top+rectangle.height/2),left:rectangles[0].left,right:rectangles.at(-1).right,container:node.getBoundingClientRect().toJSON()};
  }));
  for (const row of rows) {
    assert(Math.max(...row.centers)-Math.min(...row.centers)<2,'Adjacent entities must remain on one geometric row');
    assert(row.left>=row.container.left-1&&row.right<=row.container.right+1,'Small fixed examples should fit their frame');
  }
  for (const example of Object.values(dpStateFamiliesExamples)) {
    const question=lesson.locator('p').filter({ hasText: example.question });
    assert.equal(await question.count(),1);
    const program=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:example.title,exact:true})});
    assert.equal(await program.count(),1);
    assert(await question.evaluate((node, title)=>{
      const programs=[...document.querySelectorAll('.python-example')];
      const target=programs.find(program=>program.querySelector('h3')?.textContent===title);
      return Boolean(node.compareDocumentPosition(target)&Node.DOCUMENT_POSITION_FOLLOWING);
    },example.title));
  }
  for (const id of ['interval-states-split-the-last-operation','tree-states-remember-the-parent-boundary','digit-states-count-constrained-continuations']) await capture(lesson.locator(`#${id}`),`section-${id}`);
  const familyPractice=lesson.locator('.dsa-practice__stage').filter({hasText:'Deeper state families · identify what crosses the boundary'});
  assert.equal(await familyPractice.locator('a[href^="https://leetcode.com/problems/"]').count(),3);
  await capture(familyPractice,'family-practice-reading');
  const changed=lesson.locator('.lesson-check').filter({hasText:'Derive the changed recurrence and its endpoint rule'});
  await changed.locator('summary').focus();
  await page.keyboard.press('Enter');
  assert.match(await changed.innerText(),/totaling 19/);
  await capture(changed,'digit-changed-practice-reading');
  await changed.locator('summary').click();
  const geometry=await lesson.locator('.dpf-scroll svg').evaluateAll(nodes=>nodes.map(svg=>{
    const bounds=svg.getBoundingClientRect();
    return {width:bounds.width,viewport:svg.parentElement.clientWidth,text:[...svg.querySelectorAll('text')].map(text=>({size:parseFloat(getComputedStyle(text).fontSize),box:text.getBoundingClientRect().toJSON()})),bounds:bounds.toJSON()};
  }));
  for (const svg of geometry) for (const text of svg.text) {
    assert(text.size>=12);
    assert(text.box.left>=svg.bounds.left-1 && text.box.right<=svg.bounds.right+1);
  }
  if(width===320) assert(geometry.every(svg=>svg.width<=svg.viewport+1),'Default small trees should fit without hidden leaves');
  for (const lab of [interval,tree,digit]) assert.equal(await lab.evaluate(node=>node.scrollWidth>node.clientWidth+1),false);
  return {counts,geometry,rows};
};
