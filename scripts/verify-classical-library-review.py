"""Independent complementary checks; does not overwrite author evidence.

Use lesson-tools python for `inline`, manifold-learning-runtime for `manifold`,
and classical-depth-runtime for `crf` / `active`.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/teaching/evidence/classical-library-independent-native.json'


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inline():
    # Read the actual JSX expression strings with Babel, not a manually copied
    # equivalent. Resolve the original complete programs from their real modules.
    extraction = r'''
import fs from 'node:fs';
import {parse} from '@babel/parser';
import {gaussianProcessExamples} from './src/learn/data/gaussian-process-examples.js';
import {linearLogisticExamples} from './src/learn/data/linear-logistic-examples.js';
const result=[];
for(const [id,prefix,original] of [
 ['gaussian-processes-gp','from sklearn.gaussian_process import GaussianProcessRegressor',gaussianProcessExamples[0].code],
 ['linear-logistic-regression','from sklearn.linear_model import LogisticRegression',linearLogisticExamples[3].code]]) {
 const file='src/learn/data/topics/'+id+'.jsx';
 const tree=parse(fs.readFileSync(file,'utf8'),{sourceType:'module',plugins:['jsx']});
 const found=[];
 function walk(node) {
  if(!node||typeof node!=='object')return;
  if(node.type==='JSXElement'&&node.openingElement.name.name==='CodeBlock') {
   for(const child of node.children)if(child.type==='JSXExpressionContainer'&&child.expression.type==='StringLiteral'&&child.expression.value.startsWith(prefix))found.push(child.expression.value);
  }
  for(const value of Object.values(node))if(Array.isArray(value))value.forEach(walk);else if(value&&typeof value==='object')walk(value);
 }
 walk(tree);if(found.length!==1)throw Error(id+' expected exactly one appended bridge, got '+found.length);
 result.push({id,file,original,appended:found[0]});
}
console.log(JSON.stringify(result));
'''
    examples = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', extraction], cwd=ROOT, text=True, encoding='utf-8'))
    checks = []
    for example in examples:
        namespace = {}
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(example['original'] + '\n' + example['appended'], example['file'], 'exec'), namespace)
        checks.append(dict(id=example['id'], result='actual original + appended JSX code passed',
                           appendedSha256=hashlib.sha256(example['appended'].encode()).hexdigest(),
                           originalSha256=hashlib.sha256(example['original'].encode()).hexdigest(),
                           originalOutput=capture.getvalue()))
    return checks, [row['file'] for row in examples] + ['src/learn/data/gaussian-process-examples.js', 'src/learn/data/linear-logistic-examples.js']


def manifold():
    source = 'public/learn-assets/manifold-learning/umap-mechanism.py'
    module = load(source)
    x = np.array([[0.], [.7], [1.9], [4.2], [8.6], [11.3]])
    graphs = {}
    for k in (3, 4):
        graph = module.fuzzy_graph(*module.exact_neighbors(x, k))[0]
        scaled = module.fuzzy_graph(*module.exact_neighbors(x * 5, k))[0]
        np.testing.assert_allclose(graph.toarray(), scaled.toarray(), atol=5e-5)
        graphs[k] = graph
    assert not np.allclose(graphs[3].toarray(), graphs[4].toarray())
    points = np.array([[0.], [2.], [.09]])
    changed = module.sampled_edge_step(points, 0, 1, [2], .1)
    # Head reaches .08 through attraction, then negative repulsion clips to -4.
    np.testing.assert_allclose(changed, [[-.32], [1.92], [.09]], atol=1e-14)
    np.testing.assert_array_equal(points, [[0.], [2.], [.09]])
    self_negative = module.sampled_edge_step(points, 0, 1, [0], .1)
    np.testing.assert_allclose(self_negative, [[.08], [1.92], [.09]], atol=1e-14)
    for points in (np.array([[1e308], [-1e308], [0.]]),
                   np.array([[1e45], [-1e45], [0.]]),
                   np.array([[0.], [1e-50], [3e-50]])):
        try:
            module.exact_neighbors(points, 2)
        except ValueError as error:
            assert 'rescale' in str(error).lower()
        else:
            raise AssertionError('Unrepresentable distance accepted')
    return ['k=3/4 graph changes; uniform distance scaling preserves graph within search precision',
            'hand-derived clipped negative update; only positive tail and head move; input copy preserved',
            'self-negative ID skipped',
            'squared-distance overflow, float32 conversion overflow and lost positive distance reject with rescaling guidance'], [source, 'scratch/manifold-learning-runtime/Lib/site-packages/umap/layouts.py']


def crf():
    source = 'public/learn-assets/crf/crfsuite-bridge.py'
    module = load(source)
    model = module.fit()
    labels, emissions, transitions = module.chain_arrays(model, module.features(['Taylor', 'enjoys', 'running']))
    original_z, original_nodes, original_path = module.infer(emissions, transitions)
    shift = np.array([1000., -500., 3.])
    logz, nodes, path = module.infer(emissions + shift[:, None], transitions)
    np.testing.assert_allclose(nodes, original_nodes, atol=2e-13)
    np.testing.assert_allclose(logz - original_z, shift.sum(), atol=2e-13)
    assert path == original_path
    np.testing.assert_allclose(nodes.sum(axis=1), 1, atol=2e-13)
    return ['large common per-token score offsets preserve normalized marginals/Viterbi and shift logZ by their sum',
            'finite log-space inference and normalized rows under those offsets'], [source]


def active():
    source = 'public/learn-assets/active-learning/query-library-bridge.py'
    module = load(source)
    x = np.array([-3., -2., -1., .1, 1.1, 1.8, 3., 4.])[:, None]
    known = np.array([0., 0., np.nan, np.nan, np.nan, np.nan, 1., 1.])
    model = module.SklearnClassifier(module.LogisticRegression(max_iter=300, tol=1e-10), classes=[0, 1], missing_label=np.nan)
    model.fit(x, known)
    strategy = module.UncertaintySampling(method='entropy', missing_label=np.nan, random_state=19)
    calls=[]
    def oracle(row):
        assert np.isnan(known[row]) and row not in calls
        calls.append(row)
        return int(x[row, 0] >= .5)
    for budget in (3, 2, 1):
        before = known.copy()
        updated, remaining, event = module.acquire_one(x, known, model, strategy, oracle, budget)
        assert remaining == budget - 1 and event[0] == calls[-1]
        np.testing.assert_array_equal(before, known)
        assert np.isfinite(updated).sum() == np.isfinite(known).sum() + 1
        independent = module.SklearnClassifier(module.LogisticRegression(max_iter=300, tol=1e-10), classes=[0, 1], missing_label=np.nan).fit(x, updated)
        np.testing.assert_allclose(model.predict_proba(x), independent.predict_proba(x), atol=1e-13)
        known = updated
    return ['oracle sees exactly one still-unlabeled original ID per acquisition; IDs never repeated',
            'input label vector unchanged; returned vector has exactly one new label',
            'actual post-query classifier equals independent fit on acquired labels, including final acquisition'], [source]


selection = sys.argv[1]
checks, sources = {'inline': inline, 'manifold': manifold, 'crf': crf, 'active': active}[selection]()
receipt = json.loads(EVIDENCE.read_text()) if EVIDENCE.exists() else {'date': '2026-09-22', 'records': {}}
sources.append('scripts/verify-classical-library-review.py')
receipt['records'][selection] = {'status': 'passed', 'checks': checks,
    'sourceHashes': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
EVIDENCE.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(selection, 'independent checks passed')
