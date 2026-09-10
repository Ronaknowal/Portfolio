const {
  chromium
} = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs'),
  path = require('path'),
  assert = require('assert/strict');
const output = path.resolve('scratch/entropy-browser');
fs.mkdirSync(output, {
  recursive: true
});
(async () => {
  const model = await import('../src/learn/data/entropy-information-models.js');
  const {
    entropyInformationExamples: examples
  } = await import('../src/learn/data/entropy-information-examples.js');
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true
  });
  const results = [];
  try {
    for (const viewportWidth of [1440, 390, 320]) {
      const page = await browser.newPage({
        viewport: {
          width: viewportWidth,
          height: 1050
        },
        reducedMotion: 'reduce'
      });
      const errors = [],
        warnings = [],
        failedRequests = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', e => {
        if (e.type() === 'error') errors.push(e.text());
        if (e.type() === 'warning') warnings.push(e.text());
      });
      page.on('requestfailed', r => failedRequests.push({
        url: r.url(),
        failure: r.failure()
      }));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/entropy-cross-entropy-kl-divergence');
      const lesson = page.locator('.entropy-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const lab = name => lesson.locator('[data-entropy-lab="' + name + '"]');
      const metric = (region, label) => region.locator('.entropy-metrics>div').filter({
        has: page.locator('dt', {
          hasText: label
        })
      }).locator('dd');
      async function equal(region, label, value) {
        assert.equal(await metric(region, label).innerText(), typeof value === 'number' ? model.entropyNumber(value) : value);
      }
      async function slider(region, label, value) {
        await region.getByRole('slider', {
          name: label,
          exact: true
        }).fill(String(value));
      }
      async function shot(region, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await region.screenshot({
          path: path.join(output, name + '-' + viewportWidth + '.png')
        });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      if (viewportWidth !== 320 && !process.env.ENTROPY_READING_ONLY) {
        const surprise = lab('surprise');
        for (const p of [0, .01, .1, .5, .9, .99, 1]) {
          await slider(surprise, 'Probability of heads p', p);
          await equal(surprise, 'Average entropy (bits / draw)', model.binaryEntropyState(p).entropy);
          states++;
        }
        await slider(surprise, 'Probability of heads p', .99);
        await shot(surprise, 'surprise-rare');
        await surprise.getByRole('button', {
          name: 'Reset to p=0.9',
          exact: true
        }).click();
        const coding = lab('coding');
        for (const codebook of Object.keys(model.CODEBOOKS)) {
          await coding.getByRole('combobox', {
            name: 'Codebook'
          }).selectOption(codebook);
          const full = model.prefixCodeState('AAAABBCD', codebook);
          for (let consumed = 0; consumed <= full.bits.length; consumed++) {
            const state = model.prefixCodeState('AAAABBCD', codebook, consumed);
            await equal(coding, 'Complete decoded symbols', state.decoded || '(none yet)');
            await equal(coding, 'Unfinished prefix', state.buffer || '(empty: at a word boundary)');
            if (consumed < full.bits.length) await coding.getByRole('button', {
              name: 'Read next bit',
              exact: true
            }).click();
            states++;
          }
          await coding.getByRole('button', {
            name: 'Back one bit'
          }).click();
          assert.equal(await metric(coding, 'Bits read').innerText(), `${full.bits.length - 1} / ${full.bits.length}`);
          await coding.getByRole('button', {
            name: 'Restart decoder'
          }).click();
        }
        for (const message of ['CCCC', 'ABCDABCDABCDABCD', 'CABD']) {
          await coding.getByRole('textbox', {
            name: 'Message',
            exact: true
          }).fill(message);
          await coding.getByRole('button', {
            name: 'Encode message'
          }).click();
          await coding.getByRole('button', {
            name: 'Decode all bits'
          }).click();
          await equal(coding, 'Complete decoded symbols', message);
          states++;
        }
        const saved = await coding.locator('.entropy-metrics').innerText();
        for (const invalid of ['', 'XYZ', 'A'.repeat(17)]) {
          await coding.getByRole('textbox', {
            name: 'Message',
            exact: true
          }).fill(invalid);
          await coding.getByRole('button', {
            name: 'Encode message'
          }).click();
          assert(await coding.getByRole('alert').isVisible());
          assert.equal(await coding.locator('.entropy-metrics').innerText(), saved);
          states++;
        }
        await coding.getByRole('button', {
          name: 'Reset coding'
        }).click();
        await coding.getByRole('button', {
          name: 'Read next bit'
        }).click();
        await shot(coding, 'coding-step');
        await coding.getByRole('button', {
          name: 'Reset coding'
        }).click();
        const mismatch = lab('mismatch');
        for (const [p, q] of [[[4, 2, 1, 1], [1, 1, 2, 4]], [[0, 2, 1, 1], [0, 2, 1, 1]], [[4, 2, 1, 1], [0, 2, 1, 1]], [[3, 1, 0, 0], [1, 1, 0, 0]], [[1, 1, 1, 1], [1.0000001, .9999999, 1, 1]]]) {
          for (const unit of ['bits', 'nats']) {
            await mismatch.getByRole('textbox', {
              name: 'Source weights P'
            }).fill(p.join(','));
            await mismatch.getByRole('textbox', {
              name: 'Model weights Q'
            }).fill(q.join(','));
            await mismatch.getByRole('button', {
              name: 'Apply distributions'
            }).click();
            await mismatch.getByRole('combobox', {
              name: 'Information units'
            }).selectOption(unit);
            const state = model.informationState(p, q, unit === 'bits' ? 2 : Math.E);
            await equal(mismatch, 'Entropy H(P), ' + unit, state.entropy);
            await equal(mismatch, 'Cross-entropy H(P,Q), ' + unit, state.crossEntropy);
            await equal(mismatch, 'KL(P ∥ Q), ' + unit, state.kl);
            states++;
          }
        }
        await mismatch.getByRole('combobox', {
          name: 'Information units'
        }).selectOption('bits');
        await mismatch.getByRole('button', {
          name: 'Reset mismatch'
        }).click();
        const active = await mismatch.locator('.entropy-metrics').innerText();
        for (const invalid of ['0,0,0,0', '1,-1,2,3', '1,NaN,2,3', '1,2,3', '101,1,1,1']) {
          await mismatch.getByRole('textbox', {
            name: 'Model weights Q'
          }).fill(invalid);
          await mismatch.getByRole('button', {
            name: 'Apply distributions'
          }).click();
          assert(await mismatch.getByRole('alert').isVisible());
          assert.equal(await mismatch.locator('.entropy-metrics').innerText(), active);
          states++;
        }
        await mismatch.getByRole('button', {
          name: 'Reset mismatch'
        }).click();
        await shot(mismatch, 'mismatch-signed');
        await mismatch.getByRole('button', {
          name: 'Match Q to P'
        }).click();
        await equal(mismatch, 'KL(P ∥ Q), bits', 0);
        await mismatch.getByRole('button', {
          name: 'Exclude possible A'
        }).click();
        await equal(mismatch, 'KL(P ∥ Q), bits', Infinity);
        await shot(mismatch, 'mismatch-support');
        await mismatch.getByRole('button', {
          name: 'Exclude A in both'
        }).click();
        await equal(mismatch, 'KL(P ∥ Q), bits', 0);
        await mismatch.getByRole('button', {
          name: 'Reset mismatch'
        }).click();
        const conditional = lab('conditional');
        for (const e of [0, .01, .1, .2, .5]) for (const trust of [0, .5, .8, .9, 1]) {
          await slider(conditional, 'Label-flip probability e', e);
          await slider(conditional, 'Model probability Q(Y=X | X)', trust);
          const state = model.conditionalLossState(e, trust);
          await equal(conditional, 'Conditional entropy H(Y | X), bits', state.conditionalEntropy);
          await equal(conditional, 'Model conditional cross-entropy, bits', state.modelLoss);
          await equal(conditional, 'Expected conditional KL, bits', state.excess);
          states++;
        }
        await conditional.getByRole('button', {
          name: 'Reset conditional'
        }).click();
        await conditional.getByRole('button', {
          name: 'Ignore the context'
        }).click();
        await equal(conditional, 'Model conditional cross-entropy, bits', 1);
        await conditional.getByRole('button', {
          name: 'Use the true conditional'
        }).click();
        await equal(conditional, 'Expected conditional KL, bits', model.conditionalLossState(.1, .9).excess);
        await shot(conditional, 'conditional-context');
        const logits = lab('logits');
        for (const gap of [0, 2, 50, 1000]) for (const offset of [-1000, 0, 1000]) for (const target of [0, 1, 2]) {
          await slider(logits, 'Score gap g', gap);
          await slider(logits, 'Common score offset', offset);
          await logits.getByRole('combobox', {
            name: 'Observed class'
          }).selectOption(String(target));
          await equal(logits, 'Observed-class log loss (nats)', model.logitsState(gap, offset, target).loss);
          states++;
        }
        await shot(logits, 'logits-underflow');
        await logits.getByRole('button', {
          name: 'Reset logits'
        }).click();
        const continuous = lab('continuous');
        for (const width of [.125, .25, 4]) for (const scale of [1, 10, 100]) for (const bins of [2, 4, 16]) {
          await slider(continuous, 'Width w in metres', width);
          await continuous.getByRole('combobox', {
            name: 'New coordinate unit'
          }).selectOption(String(scale));
          await slider(continuous, 'Equal quantization bins', bins);
          const state = model.continuousEntropyState(width, scale, bins);
          await equal(continuous, 'h(X), bits relative to metre coordinate', state.original.entropy);
          await equal(continuous, 'h(new coordinate), bits', state.transformed.entropy);
          await equal(continuous, 'Entropy of bin label, bits', state.quantizedEntropy);
          states++;
        }
        await continuous.getByRole('button', {
          name: 'Reset units'
        }).click();
        await shot(continuous, 'continuous-units');
        const maximum = lab('maximum');
        for (const mean of [0, .5, 1, 10 / 7, 1.8, 2]) for (const fraction of [0, .25, .5, .75, 1]) {
          await maximum.getByRole('combobox', {
            name: 'Required mean m'
          }).selectOption(String(mean));
          await slider(maximum, 'Position along feasible slice', fraction);
          const state = model.maxEntropyState(mean, fraction);
          await equal(maximum, 'Selected entropy H(q), bits', state.entropy);
          await equal(maximum, 'Maximum entropy H(p*), bits', state.maximumEntropy);
          await equal(maximum, 'KL(q ∥ p*), bits', state.gap);
          states++;
        }
        await shot(maximum, 'maximum-boundary');
        await maximum.getByRole('combobox', {
          name: 'Required mean m'
        }).selectOption('0.5');
        await maximum.getByRole('button', {
          name: 'Select the maximum'
        }).click();
        assert(Number.parseFloat(await metric(maximum, 'KL(q ∥ p*), bits').innerText()) < 1e-20);
        await shot(maximum, 'maximum-interior');
        await maximum.getByRole('button', {
          name: 'Reset mean constraint'
        }).click();
      }
      // Every local checkpoint and practice reveal must contain its actual question/answer.
      assert.equal(await lesson.locator('.lesson-check').count(), 2);
      for (const checkpoint of await lesson.locator('.lesson-check').all()) {
        assert((await checkpoint.locator('p').first().innerText()).length > 80);
        await checkpoint.locator('summary').focus();
        await page.keyboard.press('Enter');
        assert((await checkpoint.locator('details').innerText()).length > 150);
        await checkpoint.locator('summary').press('Enter');
      }
      assert.equal(await lesson.locator('.entropy-practice').count(), 9);
      for (const practice of await lesson.locator('.entropy-practice').all()) for (const summary of await practice.locator('summary').all()) {
        await summary.focus();
        await page.keyboard.press('Enter');
        assert((await summary.locator('..').innerText()).length > 90);
        await summary.press('Enter');
      }
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const rendered = await lesson.locator('.python-example').all();
      assert.equal(rendered.length, 11);
      for (const example of Object.values(examples)) {
        const block = lesson.locator('.python-example').filter({
          has: page.getByRole('heading', {
            name: example.title,
            exact: true
          })
        });
        assert.equal(await block.count(), 1);
        const normalize = text => text.replace(/\r\n/g, '\n');
        assert(normalize(await block.innerText()).includes(normalize(example.code)));
        assert(normalize(await block.innerText()).includes(normalize(example.expected)));
        assert((await lesson.innerText()).includes(example.question));
      }
      const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({
        width: node.clientWidth,
        scrollWidth: node.scrollWidth,
        text: node.textContent.slice(0, 60)
      })));
      assert(math.length >= 10);
      assert.deepEqual(math.filter(item => item.scrollWidth > item.width + 1), []);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const overflow = await page.evaluate(() => ({
        viewport: document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
        paragraphs: [...document.querySelectorAll('.entropy-lesson p,.entropy-lesson li,.entropy-lesson h2,.entropy-lesson h3')].filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent.slice(0, 80))
      }));
      assert(overflow.documentWidth <= overflow.viewport + 1, JSON.stringify(overflow));
      assert.deepEqual(overflow.paragraphs, []);
      const svgGeometry = await lesson.locator('svg').evaluateAll(nodes => nodes.map(svg => {
        const view = svg.viewBox.baseVal;
        return [...svg.querySelectorAll('text')].map(text => {
          const box = text.getBBox();
          return {
            text: text.textContent,
            x: box.x,
            y: box.y,
            right: box.x + box.width,
            bottom: box.y + box.height,
            maxX: view.width,
            maxY: view.height
          };
        });
      }));
      assert.deepEqual(svgGeometry.flat().filter(text => text.x < -1 || text.y < -1 || text.right > text.maxX + 1 || text.bottom > text.maxY + 1), []);
      const controls = await lesson.locator('.entropy-controls button:not(:disabled),.entropy-controls input,.entropy-controls select').all();
      let keyboardControls = 0;
      for (const control of controls) {
        await control.focus();
        assert(await control.evaluate(node => document.activeElement === node));
        const style = await control.evaluate(node => ({
          height: node.getBoundingClientRect().height,
          outline: getComputedStyle(node).outlineStyle
        }));
        assert(style.height >= 44);
        await page.keyboard.press('Tab');
        keyboardControls++;
      }
      const sliderControl = lab('surprise').getByRole('slider', {
        name: 'Probability of heads p'
      });
      await sliderControl.focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowRight');
      assert.equal(await sliderControl.inputValue(), '0.01');
      await lab('surprise').getByRole('button', {
        name: 'Reset to p=0.9'
      }).click();
      let scrollRegions = 0;
      for (const table of await lesson.locator('.lesson-table-wrap').all()) {
        if (await table.evaluate(n => n.scrollWidth > n.clientWidth + 1)) {
          await table.evaluate(n => n.scrollLeft = 0);
          await table.focus();
          assert(await table.evaluate(n => document.activeElement === n));
          await page.keyboard.press('ArrowRight');
          await page.waitForFunction(n => n.scrollLeft > 0, await table.elementHandle(), {
            timeout: 2000
          });
          scrollRegions++;
        }
      }
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      const navLinks = await lesson.getByRole('navigation', {
        name: 'In this lesson'
      }).getByRole('link').all();
      assert.equal(navLinks.length, 10);
      for (let i = 0; i < navLinks.length; i++) {
        const href = await navLinks[i].getAttribute('href');
        assert.equal(await page.locator('[id="' + href.slice(1) + '"]').count(), 1);
        await navLinks[i].click();
        assert.equal(new URL(page.url()).hash, href);
        await page.locator('[id="' + href.slice(1) + '"]').evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 90));
        await page.screenshot({
          path: path.join(output, 'reading-' + (i + 1) + '-' + viewportWidth + '.png')
        });
      }
      for (const [selector, name] of [['.entropy-tree', 'prefix-tree'], ['.entropy-case-row', 'classifier-loss']]) {
        const region = selector === '.entropy-case-row' ? lesson.locator('.entropy-inline').last() : lesson.locator(selector).locator('..');
        await shot(region, name);
      }
      await shot(lesson.locator('.lesson-sources'), 'sources');
      await shot(lesson.locator('.entropy-practice').last(), 'practice-maximum');
      await page.locator('[id="6-predict-with-the-information-you-have"]').evaluate(n => window.scrollTo(0, window.scrollY + n.getBoundingClientRect().top - 90));
      await page.screenshot({
        path: path.join(output, 'conditional-reading-' + viewportWidth + '.png')
      });
      assert.deepEqual(errors, []);
      assert.deepEqual(warnings, []);
      assert.deepEqual(failedRequests, []);
      results.push({
        viewportWidth,
        states,
        programs: 11,
        checkpoints: 2,
        practice: 9,
        anchors: 10,
        math: math.length,
        keyboardControls,
        keyboardScrollRegions: scrollRegions,
        svgGeometry,
        overflow,
        errors,
        warnings,
        failedRequests
      });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(output, process.env.ENTROPY_READING_ONLY ? 'reading-results.json' : 'results.json'), JSON.stringify({
    passed: true,
    timestamp: new Date().toISOString(),
    results
  }, null, 2));
  console.log(results.map(({
    svgGeometry,
    ...result
  }) => result));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
