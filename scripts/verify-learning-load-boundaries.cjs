const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { parse } = require('@babel/parser');
const base = process.env.PERFORMANCE_BASE_URL || 'http://127.0.0.1:4173';
const build = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
const lessons = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
const python = 'python-basics-types-control-flow-functions-modules';
const oop = 'object-oriented-programming-in-python';
const mathematics = 'eigenvalues-eigenvectors';
const dsaOwnership = {
  'arrays-strings-hash-maps': ['array-map-foundations-examples.js', 'array-map-foundations-model.js', 'bitwise-foundations-examples.js', 'bitwise-foundations-models.js'],
  'linked-lists-stacks-queues': ['linked-foundations-examples.js', 'linked-foundations-model.js', 'linked-traversal-examples.js', 'linked-traversal-models.js', 'monotonic-stack-examples.js', 'monotonic-stack-models.js'],
  'trees-binary-search-trees': ['tree-examples.js', 'tree-models.js'],
  'heaps-priority-queues-tries': ['heap-trie-examples.js', 'heap-trie-models.js'],
  'graphs-representations-bfs-dfs': ['graph-traversal-examples.js', 'graph-traversal-models.js'],
  'disjoint-sets-union-find': ['union-find-examples.js', 'union-find-models.js'],
  'complexity-analysis-recursion': ['complexity-recursion-examples.js', 'complexity-recursion-models.js'],
  'binary-search-sorting-two-pointer-patterns': ['ordered-pattern-examples.js', 'ordered-pattern-models.js'],
  'backtracking-divide-and-conquer': ['backtracking-divide-examples.js', 'backtracking-divide-models.js'],
  'greedy-algorithms-exchange-arguments': ['greedy-exchange-examples.js', 'greedy-exchange-models.js'],
  'dynamic-programming-states-transitions-optimization': ['dynamic-programming-examples.js', 'dynamic-programming-models.js', 'dp-state-families-examples.js', 'dp-state-families-models.js'],
  'segment-trees-fenwick-trees-range-queries': ['range-query-examples.js', 'range-query-models.js', 'range-deque-models.js'],
  'algorithm-correctness-loop-invariants-termination': ['algorithm-correctness-examples.js', 'algorithm-correctness-models.js'],
  'hashing-collision-resolution-amortized-analysis': ['hashing-amortized-examples.js', 'hashing-amortized-models.js'],
  'shortest-paths-spanning-trees-topological-ordering': ['weighted-graph-examples.js', 'weighted-graph-models.js'],
  'string-matching-prefix-functions-rolling-hashes': ['string-matching-examples.js', 'string-matching-models.js'],
  'reductions-p-np-computational-intractability': ['intractability-examples.js', 'intractability-models.js'],
  'randomized-algorithms-sampling-error-guarantees': ['randomized-algorithm-examples.js', 'randomized-algorithm-models.js'],
  'network-flow-minimum-cuts-bipartite-matching': ['network-flow-examples.js', 'network-flow-models.js'],
  'computational-geometry-robust-predicates-convex-hulls': ['computational-geometry-examples.js', 'computational-geometry-models.js'],
  'persistent-data-structures-structural-sharing-versioned-queries': ['persistent-structures-examples.js', 'persistent-structures-models.js'],
  'external-memory-algorithms-b-trees-i-o-complexity': ['external-memory-examples.js', 'external-memory-models.js'],
};
const mathematicsLessons = [
  'vectors-matrices-tensor-operations',
  'matrix-decompositions-svd-qr-cholesky-lu',
  'eigenvalues-eigenvectors',
  'matrix-calculus-jacobians',
  'tensor-algebra-einsum-notation',
  'randomized-linear-algebra',
  'multivariate-calculus-gradients',
  'convex-optimization',
  'gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars',
  'learning-rate-schedules-cosine-warmup-onecyclelr',
  'convex-duality-lagrangian-methods-kkt-conditions',
  'second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient',
  'non-convex-optimization-landscape',
  'constrained-multi-objective-optimization',
  'probability-distributions-bayes-theorem',
  'maximum-likelihood-map-estimation',
  'hypothesis-testing-confidence-intervals',
  'bayesian-inference-conjugate-priors',
  'concentration-inequalities-hoeffding-bernstein-chernoff',
  'monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts',
  'variational-inference',
  'exponential-families-sufficient-statistics',
  'measure-theory-probability-spaces',
  'optimal-transport-wasserstein-distance-sinkhorn',
  'causal-inference-do-calculus',
  'entropy-cross-entropy-kl-divergence',
  'mutual-information-information-bottleneck',
  'rate-distortion-theory',
  'f-divergences-integral-probability-metrics',
  'graph-fundamentals-adjacency-laplacian-connectivity',
  'spectral-graph-theory',
  'combinatorial-optimization-approximation-algorithms',
  'stochastic-processes-markov-chains-brownian-motion-poisson',
  'random-matrix-theory',
  'queueing-theory-m-m-1-m-g-1-little-s-law',
  'dynamical-systems-theory-chaos',
  'it-calculus-stochastic-differential-equations',
  'numerical-methods-finite-differences-quadrature-root-finding',
  'functional-analysis-rkhs',
  'topology-topological-data-analysis-tda',
  'category-theory-emerging-use-in-ml',
  'differential-geometry-riemannian-manifolds',
  'algebra-functions-exponentials-logarithms',
  'sets-logic-relations-proof-techniques',
  'geometry-trigonometry-coordinate-reasoning',
  'counting-combinatorics-mathematical-induction',
  'single-variable-calculus-limits-derivatives-integrals',
  'random-variables-expectation-covariance',
  'sampling-measurement-experimental-design',
  'ordinary-differential-equations-linear-systems',
  'complex-numbers-fourier-laplace-transforms',
  'conditioning-stability-numerical-analysis',
  'decision-theory-risk-cost-sensitive-decisions',
  'real-analysis-sequences-modes-of-convergence',
  'abstract-algebra-groups-symmetry-actions',
  'partial-differential-equations-conservation-boundary-conditions',
  'numerical-pdes-grids-finite-elements-stability',
];
const modulePath = id => `src/learn/data/${lessons[id].replace(/^\.\//, '')}`;
const outlinePath = id => `src/learn/data/generated/outlines/${id}.json`;
const entryFile = source => {
  assert.ok(build[source], `Build manifest has no entry for ${source}`);
  return build[source].file;
};
const bodyFiles = new Set(Object.keys(lessons).map(id => entryFile(modulePath(id))));
const bodySources = new Set(Object.keys(lessons).map(modulePath));
assert.equal(bodyFiles.size, bodySources.size, 'Distinct lesson modules were combined into a shared lesson-body entry chunk');
for (const source of bodySources) assert.equal(build[source].isDynamicEntry, true, `${source} is no longer a lazy entry`);
assert.deepEqual(Object.keys(build).filter(source => source.startsWith('src/learn/data/topics/') && build[source].isDynamicEntry && !bodySources.has(source)), [], 'Unregistered draft lesson included in the dynamic production entry graph');
const outlineFiles = new Set(Object.keys(build).filter(key => key.startsWith('src/learn/data/generated/outlines/')).map(key => build[key].file));
// Check actual engine/style content, not only hashed chunk names: a barrel can
// accidentally fold KaTeX into an otherwise innocently named shared chunk.
const assetNames = fs.readdirSync('dist/assets');
const mathRendererFiles = new Set(assetNames.filter(name => name.endsWith('.js') && fs.readFileSync(`dist/assets/${name}`, 'utf8').includes('__renderToDomTree')).map(name => `assets/${name}`));
const mathFontStyleFiles = new Set(assetNames.filter(name => name.endsWith('.css') && /font-family:KaTeX_/.test(fs.readFileSync(`dist/assets/${name}`, 'utf8'))).map(name => `assets/${name}`));
assert.ok(mathRendererFiles.size > 0, 'Cannot identify the built KaTeX renderer for isolation checks');
assert.ok(mathFontStyleFiles.size > 0, 'Cannot identify the built KaTeX font styles for isolation checks');
const closure = roots => {
  const visited = new Set(), files = new Set();
  const visit = key => {
    if (visited.has(key)) return;
    visited.add(key);
    assert.ok(build[key], `Missing manifest import ${key}`);
    files.add(build[key].file);
    for (const key2 of build[key].imports || []) visit(key2);
  };
  roots.forEach(visit);
  return files;
};
const main = Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html'));
assert.ok(main, 'Missing application entry manifest');
const jsRequests = requests => new Set(requests.filter(url => url.startsWith(base) && /\.js(?:\?|$)/.test(url)).map(url => new URL(url).pathname.slice(1)));
const intersection = (a, b) => [...a].filter(value => b.has(value));
function checkMathRequests(requests, expected) {
  const loaded = new Set(requests.filter(url => url.startsWith(base)).map(url => new URL(url).pathname.slice(1)));
  const rendererJs = intersection(loaded, mathRendererFiles), fontCss = intersection(loaded, mathFontStyleFiles);
  assert.equal(rendererJs.length > 0, expected, expected ? 'Math lesson did not request its renderer' : 'Non-math route requested the KaTeX renderer');
  assert.equal(fontCss.length > 0, expected, expected ? 'Math lesson did not request KaTeX styles' : 'Non-math route requested KaTeX font styles');
  return { rendererJs, fontCss };
}
function checkRequests(requests, ids, outlineIds = []) {
  const loaded = jsRequests(requests), expectedBodies = ids.map(id => entryFile(modulePath(id)));
  const expectedOutlines = outlineIds.map(id => entryFile(outlinePath(id)));
  assert.deepEqual(intersection(loaded, bodyFiles).sort(), [...new Set(expectedBodies)].sort(), 'Unexpected or missing lesson body requests');
  assert.deepEqual(intersection(loaded, outlineFiles).sort(), [...new Set(expectedOutlines)].sort(), 'Unexpected or missing outline requests');
  return { jsFiles: [...loaded], lessonBodies: expectedBodies, outlines: expectedOutlines };
}
function checkClosure(requests, roots) {
  const allowed = closure([main, ...roots]);
  assert.deepEqual([...jsRequests(requests)].filter(file => !allowed.has(file)), [], 'Requested JavaScript outside the selected static dependency closures');
}
const route = id => `${base}/learn/path/full-curriculum/${id}`;

// A production chunk can legitimately contain several tiny shared controls.
// Verify ownership from parsed source imports, then verify actual production
// requests against manifest closures; do not infer source ownership from a hash.
function lessonSourceClosure(source) {
  const repositoryRoot = path.resolve('.');
  const visited = new Set();
  const visit = filename => {
    const relative = path.relative(repositoryRoot, filename).replaceAll('\\', '/');
    assert.ok(!relative.startsWith('../'), 'Lesson import escapes the repository');
    if (visited.has(relative)) return;
    visited.add(relative);
    if (!/\.(?:js|jsx)$/.test(filename)) return;
    const ast = parse(fs.readFileSync(filename, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
    for (const declaration of ast.program.body) {
      if (!['ImportDeclaration', 'ExportNamedDeclaration', 'ExportAllDeclaration'].includes(declaration.type)) continue;
      const specifier = declaration.source?.value;
      if (!specifier?.startsWith('.')) continue;
      const basePath = path.resolve(path.dirname(filename), specifier);
      const resolved = [basePath, ...['.js', '.jsx', '.json', '.css'].map(extension => basePath + extension), path.join(basePath, 'index.js'), path.join(basePath, 'index.jsx')]
        .find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
      assert.ok(resolved, `Cannot resolve ${specifier} from ${relative}`);
      visit(resolved);
    }
  };
  visit(path.resolve(source));
  return visited;
}
function checkDsaSourceOwnership(id) {
  const sources = lessonSourceClosure(modulePath(id));
  assert.deepEqual([...sources].filter(source => source.startsWith('src/learn/data/topics/')), [modulePath(id)], `${id} imports an unrelated authored lesson`);
  const practiceSources = [...sources].filter(source => source.startsWith('src/learn/data/practice/'));
  assert.deepEqual(practiceSources, [`src/learn/data/practice/${id}.js`], `${id} must import only its own practice dataset`);
  const dataSources = [...sources].filter(source => source.startsWith('src/learn/data/') && !source.startsWith('src/learn/data/topics/') && !source.startsWith('src/learn/data/practice/'));
  assert.deepEqual(dataSources.slice().sort(), dsaOwnership[id].map(filename => `src/learn/data/${filename}`).sort(), `${id} pulled unrelated examples/models or lost its own`);
  assert.equal([...sources].some(source => source.endsWith('/content/Math.jsx')), false, `${id} imports the math renderer`);
  return { practiceSources, dataSources, sourceFiles: [...sources] };
}

(async () => {
  const { allTopicsOrdered } = await import('../src/learn/data/generated/navigation.js');
  const outlined = allTopicsOrdered.find(topic => topic.status === 'planned' && topic.hasOutline);
  const withoutOutline = allTopicsOrdered.find(topic => topic.status === 'planned' && !topic.hasOutline);
  assert.ok(outlined, 'Need an existing planned topic with an outline for this fixture');
  assert.ok(withoutOutline, 'Need an existing planned topic without a brief for this fixture');
  const outlinedTopic = outlined.id, noOutlineTopic = withoutOutline.id;
  assert.equal(lessons[outlinedTopic], undefined);
  assert.equal(lessons[noOutlineTopic], undefined);
  for (const id of [...Object.keys(dsaOwnership), ...mathematicsLessons]) assert.ok(lessons[id], `Publish and rebuild ${id} before running its isolation review`);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const fresh = async () => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage(), requests = [], errors = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    const cdp = await context.newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
    return { context, page, requests, errors };
  };
  const settled = async page => page.waitForTimeout(350);
  try {
    for (const id of Object.keys(dsaOwnership)) {
      const ownership = checkDsaSourceOwnership(id);
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(id), { waitUntil: 'domcontentloaded' });
      await page.locator(`[data-practice-topic="${id}"]`).waitFor(); await settled(page);
      const network = checkRequests(requests, [id]);
      checkClosure(requests, ['src/learn/Reader.jsx', modulePath(id)]);
      checkMathRequests(requests, false);
      assert.equal(await page.locator('[data-practice-topic]').count(), 1, 'Another topic practice set was rendered');
      assert.ok(await page.locator('.reader-complete').isEnabled());
      assert.deepEqual(errors, []);
      results.push({ case: 'DSA lesson loads its own body/practice/examples/models, allowed shared controls and no math renderer', topicId: id, ...ownership, ...network });
      await context.close();
    }
    for (const id of mathematicsLessons) {
      const sources = lessonSourceClosure(modulePath(id));
      assert.deepEqual([...sources].filter(source => source.startsWith('src/learn/data/topics/')), [modulePath(id)], `${id} imports another authored lesson`);
      assert.equal([...sources].some(source => source.startsWith('src/learn/data/practice/')), false, `${id} imports unrelated DSA practice`);
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(id), { waitUntil: 'domcontentloaded' });
      await page.locator('.reader-article .katex').first().waitFor();
      await settled(page);
      const network = checkRequests(requests, [id]);
      checkClosure(requests, ['src/learn/Reader.jsx', modulePath(id)]);
      const mathDependencies = checkMathRequests(requests, true);
      assert.equal(await page.locator('.katex-error').count(), 0);
      assert.ok(await page.locator('.reader-complete').isEnabled());
      assert.deepEqual(errors, []);
      results.push({ case: 'reviewed mathematics lesson loads only its selected body and actual dependencies', topicId: id, sourceFiles: [...sources], ...network, ...mathDependencies });
      await context.close();
    }
    {
      const { context, page, requests, errors } = await fresh();
      await page.goto(`${base}/learn`, { waitUntil: 'domcontentloaded' });
      await page.locator('.workspace-hero').waitFor(); await settled(page);
      const network = checkRequests(requests, []);
      checkClosure(requests, ['src/learn/LearnHub.jsx']);
      checkMathRequests(requests, false);
      assert.equal(await page.locator('.workspace-availability').count(), 1);
      assert.deepEqual(errors, []);
      results.push({ case: 'hub loads metadata and no lesson bodies or outlines', ...network });
      await context.close();
    }
    {
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(python), { waitUntil: 'domcontentloaded' });
      await page.locator('[data-pyf-lab="references"]').waitFor(); await settled(page);
      checkRequests(requests, [python]);
      checkClosure(requests, ['src/learn/Reader.jsx', modulePath(python)]);
      checkMathRequests(requests, false);
      assert.ok(await page.locator('.reader-complete').isEnabled());
      const initial = jsRequests(requests), start = requests.length;
      await page.locator('.reader-footer__next').click();
      await page.waitForURL(url => url.pathname.endsWith('/' + oop));
      await page.locator('[data-oop-lab="binding"]').waitFor(); await settled(page);
      const destination = jsRequests(requests.slice(start));
      assert.ok(destination.has(entryFile(modulePath(oop))), 'Navigation did not fetch the destination lesson');
      const permitted = closure(['src/learn/Reader.jsx', modulePath(oop)]);
      assert.deepEqual([...destination].filter(file => !permitted.has(file)), [], 'Navigation fetched unrelated JavaScript');
      checkRequests(requests, [python, oop]);
      checkMathRequests(requests, false);
      const beforeBack = requests.length;
      await page.locator('.reader-footer__previous').click();
      await page.locator('[data-pyf-lab="references"]').waitFor(); await settled(page);
      assert.equal(jsRequests(requests.slice(beforeBack)).size, 0, 'Returning to an imported lesson fetched JavaScript again');
      const beforeHistoryBack = requests.length;
      await page.goBack();
      await page.waitForURL(url => url.pathname.endsWith('/' + oop));
      await page.locator('[data-oop-lab="binding"]').waitFor(); await settled(page);
      assert.equal(jsRequests(requests.slice(beforeHistoryBack)).size, 0, 'Browser history fetched an already imported lesson again');
      assert.deepEqual(errors, []);
      results.push({ case: 'selected lesson and next destination only; footer previous and browser Back reuse imports', initialJs: [...initial], destinationJs: [...destination] });
      await context.close();
    }
    {
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(mathematics), { waitUntil: 'domcontentloaded' });
      await page.locator('.reader-article .katex').first().waitFor(); await settled(page);
      const network = checkRequests(requests, [mathematics]);
      checkClosure(requests, ['src/learn/Reader.jsx', modulePath(mathematics)]);
      const mathDependencies = checkMathRequests(requests, true);
      assert.equal(await page.locator('.katex-error').count(), 0, 'Mathematical notation rendered a KaTeX parse error');
      assert.ok(await page.locator('.reader-complete').isEnabled());
      assert.deepEqual(errors, []);
      results.push({ case: 'mathematical lesson loads its renderer and styles and renders notation', ...network, ...mathDependencies });
      await context.close();
    }
    {
      const { context, page, requests, errors } = await fresh();
      let releaseOutline;
      const waitingOutline = new Promise(resolve => { releaseOutline = resolve; });
      await page.route(`**/${entryFile(outlinePath(outlinedTopic))}`, async intercepted => { await waitingOutline; await intercepted.continue(); });
      await page.goto(route(outlinedTopic), { waitUntil: 'domcontentloaded' });
      await page.locator('.lesson-loading').waitFor();
      assert.ok((await page.locator('.lesson-loading').innerText()).includes('Loading syllabus outline'));
      assert.equal(await page.locator('.reader-header h1').innerText(), outlined.title);
      assert.ok(await page.locator('.reader-complete').isDisabled());
      releaseOutline();
      await page.locator('.planned-lesson').waitFor(); await settled(page);
      const network = checkRequests(requests, [], [outlinedTopic]);
      checkClosure(requests, ['src/learn/Reader.jsx', outlinePath(outlinedTopic)]);
      checkMathRequests(requests, false);
      assert.ok(await page.locator('.reader-complete').isDisabled());
      assert.deepEqual(errors, []);
      results.push({ case: 'planned topic fetches only its outline and stays uncompletable', topicId: outlinedTopic, ...network });
      await context.close();
    }
    {
      const { context, page, requests, errors } = await fresh();
      await page.goto(route(noOutlineTopic), { waitUntil: 'domcontentloaded' });
      await page.locator('.planned-lesson').waitFor(); await settled(page);
      const network = checkRequests(requests, []);
      checkClosure(requests, ['src/learn/Reader.jsx']);
      checkMathRequests(requests, false);
      assert.ok(await page.locator('.reader-complete').isDisabled());
      assert.deepEqual(errors, []);
      results.push({ case: 'planned entry without an individual outline makes no nonexistent-resource request', topicId: noOutlineTopic, ...network });
      await context.close();
    }
    {
      const { context, page, errors } = await fresh();
      let release;
      const pending = new Promise(resolve => { release = resolve; });
      await page.route(`**/${entryFile(modulePath(python))}`, async intercepted => { await pending; await intercepted.continue(); });
      await page.goto(route(python), { waitUntil: 'domcontentloaded' });
      await page.locator('.lesson-loading').waitFor();
      assert.equal(await page.locator('.lesson-loading').getAttribute('role'), 'status');
      assert.ok((await page.locator('.reader-header h1').innerText()).startsWith('Python Basics'));
      assert.ok(await page.locator('.reader-complete').isDisabled());
      await page.locator('.reader-footer__next').click();
      await page.locator('[data-oop-lab="binding"]').waitFor();
      release(); await settled(page);
      assert.ok((await page.locator('.reader-header h1').innerText()).includes('Object-Oriented'));
      assert.equal(await page.locator('[data-pyf-lab="references"]').count(), 0, 'Stale loaded lesson replaced the selected topic');
      assert.equal(await page.locator('[data-oop-lab="binding"]').count(), 1);
      assert.deepEqual(errors, []);
      results.push({ case: 'shell stays available while loading; stale completion cannot replace destination' });
      await context.close();
    }
    {
      const { context, page, errors } = await fresh();
      let attempts = 0;
      await page.route(`**/${entryFile(modulePath(python))}*`, async intercepted => {
        attempts += 1;
        if (attempts === 1) await intercepted.abort('failed');
        else await intercepted.continue();
      });
      await page.goto(route(python), { waitUntil: 'domcontentloaded' });
      const error = page.locator('.lesson-load-error');
      await error.waitFor(); assert.equal(await error.getAttribute('role'), 'alert');
      assert.ok((await page.locator('.reader-header h1').innerText()).startsWith('Python Basics'));
      assert.ok(await page.locator('.reader-complete').isDisabled());
      assert.equal(await page.locator('.planned-lesson').count(), 0, 'Failed published content was mislabeled as planned');
      assert.equal(await error.getByRole('button', { name: 'Reload page', exact: true }).count(), 1);
      await error.getByRole('button', { name: 'Try again', exact: true }).focus(); await page.keyboard.press('Enter');
      await settled(page);
      await page.waitForFunction(() => document.querySelector('[data-pyf-lab="references"]') || document.querySelector('.lesson-load-error'), null, { timeout: 15000 });
      const retryRecovered = await page.locator('[data-pyf-lab="references"]').count() > 0;
      if (!retryRecovered) {
        assert.ok(await page.locator('.reader-complete').isDisabled());
        await error.getByRole('button', { name: 'Reload page', exact: true }).click();
        await page.locator('[data-pyf-lab="references"]').waitFor({ timeout: 15000 });
      }
      assert.ok(attempts > 1, 'Recovery did not actually request the failed chunk again');
      assert.ok(await page.locator('.reader-complete').isEnabled());
      assert.equal(await error.count(), 0);
      assert.deepEqual(errors, []);
      results.push({ case: 'failed lesson stays distinct from planned; keyboard retry and explicit reload provide recovery', retryRecovered, reloadRequired: !retryRecovered, requestsForFailedChunk: attempts });
      await context.close();
    }
    {
      const { context, page, errors } = await fresh();
      let attempts = 0;
      await page.route(`**/${entryFile(modulePath(python))}*`, async intercepted => {
        attempts += 1;
        if (attempts === 1) await intercepted.fulfill({
          status: 200, contentType: 'text/javascript',
          body: 'export default { content() { throw new Error("Controlled lesson render failure"); } };',
        });
        else await intercepted.continue();
      });
      await page.goto(route(python), { waitUntil: 'domcontentloaded' });
      const error = page.locator('.lesson-load-error');
      await error.waitFor(); await settled(page);
      assert.ok(await page.locator('.reader-complete').isDisabled(), 'Render failure left completion enabled despite absent usable content');
      assert.ok((await page.locator('.reader-header h1').innerText()).startsWith('Python Basics'));
      assert.equal(await page.locator('.planned-lesson').count(), 0);
      await error.getByRole('button', { name: 'Reload page', exact: true }).click();
      await page.locator('[data-pyf-lab="references"]').waitFor();
      assert.ok(await page.locator('.reader-complete').isEnabled());
      assert.deepEqual(errors.filter(message => !message.includes('Controlled lesson render failure')), []);
      results.push({ case: 'controlled render failure preserves shell, prevents completion and recovers after reload', requestsForLessonChunk: attempts, intentionalFixture: 'Browser-only one-response replacement; no application or lesson source mutation.' });
      await context.close();
    }
    fs.mkdirSync('scratch/learning-performance', { recursive: true });
    fs.writeFileSync('scratch/learning-performance/load-boundaries.json', JSON.stringify({ measuredAt: new Date().toISOString(), browser: browser.version(), base, results }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
