"""Independent graph-library, linear algebra, exact-practice and native checks."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import io
import json
import random
import subprocess
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/graph-fundamentals-native'
OUT.mkdir(parents=True, exist_ok=True)
RNG = random.Random(3010)


def node_json(code, data=None):
    result = subprocess.run(['node', '--input-type=module', '-e', code], cwd=ROOT, input=json.dumps(data) if data is not None else None, capture_output=True, text=True, encoding='utf-8', check=True)
    return json.loads(result.stdout)


def close(actual, expected, tolerance=2e-10):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=tolerance, atol=tolerance)


def groups(value):
    return sorted(sorted(component) for component in value)


def main():
    examples = node_json("import {graphFundamentalsExamples} from './src/learn/data/graph-fundamentals-examples.js'; console.log(JSON.stringify(graphFundamentalsExamples));")
    namespaces = {}
    for key, example in examples.items():
        script = OUT / 'programs' / f'{key}.py'
        script.parent.mkdir(exist_ok=True)
        script.write_text(example['code'] + '\n', encoding='utf-8')
        run = subprocess.run([sys.executable, '-X', 'utf8', '-I', str(script)], capture_output=True, text=True, encoding='utf-8', check=True)
        assert run.stdout.rstrip() == example['expected'].rstrip(), key
        assert not run.stderr, (key, run.stderr)
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(example['code'], key, 'exec'), namespace)
        namespaces[key] = namespace
    archive = json.loads((ROOT / 'docs/teaching/evidence/graph-fundamentals-original-content.json').read_text(encoding='utf-8'))
    assert examples['original']['code'] == archive['program'].strip()
    assert examples['original']['expected'] == archive['output'].strip()

    cases = []
    for n in range(7):
        for trial in range(20):
            edges = []
            for u in range(n):
                for v in range(u, n):
                    if RNG.random() < .28:
                        edges.append([u, v, RNG.choice([0, .25, .5, 1, 2, 3])])
            if edges and trial % 3 == 0:
                edges.append(edges[0][:])
            cases.append({'n': n, 'edges': edges, 'signal': [RNG.randint(-6, 8) for _ in range(n)]})
    directed_cases = []
    for mask in range(64):
        edges = [[u, v, 1 + ((u + v) % 2)] for bit, (u, v) in enumerate((pair for pair in product(range(3), repeat=2) if pair[0] != pair[1])) if mask & (1 << bit)]
        directed_cases.append({'n': 4, 'edges': edges})
    harmonic_cases = []
    for case in cases:
        for policy in range(3):
            anchors = {str(i): RNG.randint(-8, 8) for i in range(case['n']) if (i + policy) % 3 == 0}
            harmonic_cases.append({'n': case['n'], 'edges': case['edges'], 'anchors': anchors})

    payload = {'graphs': cases, 'directed': directed_cases, 'harmonic': harmonic_cases}
    js = r'''
import fs from 'node:fs';
import * as model from './src/learn/data/graph-fundamentals-models.js';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const result={
 graphs:input.graphs.map(c=>({energy:model.graphEnergy(c.n,c.edges,c.signal), normalized:model.graphNormalizations(c.n,c.edges)})),
 directed:input.directed.map(c=>{const g=model.graphMatrices(c.n,c.edges,true);return {...g,...model.graphConnectivity(g.adjacency)}}),
 harmonic:input.harmonic.map(c=>model.graphHarmonicInterpolation(c.n,c.edges,c.anchors)),
 walks:[], averaging:[]
};
const adjacency=[[0,2,1],[2,0,3],[1,3,0]];
for(let source=0;source<3;source++)for(let target=0;target<3;target++)for(let steps=0;steps<=4;steps++)result.walks.push({source,target,steps,...model.graphWalks(adjacency,source,target,steps)});
for(const method of ['exchange','neighbor','lazy'])for(const tau of [.25,.5,.75])for(const steps of [0,1,2,8,24])result.averaging.push(model.graphAveragingTrace(method,steps,tau));
let invalid=0;
for(const fn of [()=>model.graphMatrices(-1,[]),()=>model.graphMatrices(3,[[0,3,1]]),()=>model.graphMatrices(3,[[0,1,-1]]),()=>model.graphMatrices(3,[[0,1,NaN]]),()=>model.graphMatrices(3,[[0,1,Infinity]]),()=>model.graphEnergy(3,[],[1,2]),()=>model.graphEnergy(3,[],[1,2,NaN]),()=>model.graphWalks(adjacency,0,2,-1),()=>model.graphWalks(adjacency,0,2,5),()=>model.graphHarmonicInterpolation(3,[],{'01':2}),()=>model.graphHarmonicInterpolation(3,[],{3:2}),()=>model.graphHarmonicInterpolation(3,[],{0:Infinity}),()=>model.graphAveragingTrace('wrong'),()=>model.graphAveragingTrace('exchange',25),()=>model.graphAveragingTrace('exchange',2,-.1)]){try{fn();throw Error('invalid contract accepted')}catch(error){if(!(error instanceof RangeError))throw error;invalid++}}
result.invalidContracts=invalid;
result.normalizationRangeRejections = [1e-310, 5e-309, Number.MIN_VALUE].map(weight => {
  try {
    model.graphNormalizations(2, [[0, 1, weight]]);
    throw new Error('Non-finite reciprocal accepted');
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    return weight;
  }
});
result.smallFiniteNormalization = model.graphNormalizations(2, [[0, 1, 1e-308]]);
result.tinyFormat=model.formatGraphNumber(1e-12);
console.log(JSON.stringify(result));
'''
    result = node_json(js, payload)
    max_energy_error = 0
    for case, actual in zip(cases, result['graphs']):
        n = case['n']
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(n))
        graph.add_weighted_edges_from(edge for edge in case['edges'] if edge[2] > 0)
        adjacency = nx.to_numpy_array(graph, nodelist=list(range(n)))
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        if n:
            close(actual['energy']['adjacency'], adjacency)
            close(actual['energy']['laplacian'], laplacian)
            close(actual['energy']['action'], laplacian @ case['signal'])
            close(actual['normalized']['symmetric'], nx.normalized_laplacian_matrix(graph, nodelist=list(range(n))).toarray())
            components = list(nx.connected_components(graph))
            assert n - np.linalg.matrix_rank(laplacian, tol=1e-9) == len(components)
            assert groups(actual['energy']['components']) == groups(components)
            inverse = np.divide(1, adjacency.sum(axis=1), out=np.zeros(n), where=adjacency.sum(axis=1)>0)
            transition = inverse[:, None] * adjacency + np.diag(adjacency.sum(axis=1) == 0)
            close(actual['normalized']['transition'], transition)
            close(np.array(actual['normalized']['randomWalk']) + transition, np.eye(n))
            close(transition.sum(axis=1), np.ones(n))
            c = np.asarray(actual['energy']['incidence']).reshape((-1, n))
            weights = np.array([edge[2] for edge in graph.edges(data='weight')])
            # Independently build oriented incidence in source-record order.
            active = [edge for edge in case['edges'] if edge[2] > 0]
            independent_c = np.zeros((len(active), n))
            for i, (u, v, weight) in enumerate(active):
                independent_c[i, u] += 1
                independent_c[i, v] -= 1
            close(c, independent_c)
            close(c.T @ np.diag([edge[2] for edge in active]) @ c, laplacian)
        else:
            assert actual['energy']['adjacency'] == [] and actual['energy']['components'] == []
        exact_energy = sum(Fraction(weight) * (Fraction(case['signal'][u]) - Fraction(case['signal'][v])) ** 2 for u, v, weight in case['edges'])
        close(actual['energy']['edgeEnergy'], float(exact_energy))
        close(actual['energy']['quadraticEnergy'], float(exact_energy))
        max_energy_error = max(max_energy_error, abs(actual['energy']['quadraticEnergy'] - float(exact_energy)))

    for case, actual in zip(directed_cases, result['directed']):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(case['n']))
        graph.add_weighted_edges_from(case['edges'])
        assert groups(actual['strongComponents']) == groups(nx.strongly_connected_components(graph))
        assert groups(actual['weakComponents']) == groups(nx.weakly_connected_components(graph))
        assert actual['reachable'] == [[i == j or nx.has_path(graph, i, j) for j in range(case['n'])] for i in range(case['n'])]

    a = [[0, 2, 1], [2, 0, 3], [1, 3, 0]]
    for actual in result['walks']:
        total = 0
        count = 0
        if actual['steps'] == 0:
            total = count = int(actual['source'] == actual['target'])
        else:
            for intermediate in product(range(3), repeat=actual['steps'] - 1):
                walk = (actual['source'], *intermediate, actual['target'])
                weight = 1
                for u, v in zip(walk, walk[1:]):
                    weight *= a[u][v]
                total += weight
                count += int(weight > 0)
        assert actual['weightSum'] == total
        assert len(actual['walks']) == count
        close(actual['power'], np.linalg.matrix_power(a, actual['steps']))

    maximum_residual = 0
    for case, actual in zip(harmonic_cases, result['harmonic']):
        n = case['n']
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(n))
        graph.add_weighted_edges_from(edge for edge in case['edges'] if edge[2] > 0)
        adjacency = nx.to_numpy_array(graph, nodelist=list(range(n)))
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        anchors = {int(key): value for key, value in case['anchors'].items()}
        unanchored = []
        for component in nx.connected_components(graph):
            fixed = sorted(component.intersection(anchors))
            if not fixed:
                unanchored.append(component)
                assert all(actual['values'][vertex] is None for vertex in component)
                continue
            unknown = sorted(component.difference(anchors))
            expected = np.linalg.solve(laplacian[np.ix_(unknown, unknown)], -laplacian[np.ix_(unknown, fixed)] @ [anchors[i] for i in fixed]) if unknown else []
            close([actual['values'][i] for i in unknown], expected)
            close([actual['values'][i] for i in fixed], [anchors[i] for i in fixed])
            for vertex in unknown:
                assert min(anchors[i] for i in fixed) - 1e-9 <= actual['values'][vertex] <= max(anchors[i] for i in fixed) + 1e-9
                maximum_residual = max(maximum_residual, abs(actual['residuals'][vertex]))
        assert groups(actual['unanchored']) == groups(unanchored)
        assert actual['unique'] == (not unanchored)

    base_l = np.array([[1,-1,0,0],[-1,2,-1,0],[0,-1,1,0],[0,0,0,0]], dtype=float)
    p = np.array([[0,1,0,0],[.5,0,.5,0],[0,1,0,0],[0,0,0,1]])
    for actual in result['averaging']:
        update = np.eye(4) - actual['stepSize'] * base_l if actual['method'] == 'exchange' else p if actual['method'] == 'neighbor' else (np.eye(4) + p) / 2
        for step, state in enumerate(actual['states']):
            close(state, np.linalg.matrix_power(update, step) @ [6,0,0,4])
            close(state[3], 4)
            if actual['method'] == 'exchange':
                close(sum(state[:3]), 6)
            else:
                close(state[0] + 2*state[1] + state[2], 6)

    # Changed tasks invoke the actual learner functions with independent expected results.
    directed = namespaces['representations']['adjacency_matrix'](['W','U','Q','V'], [('U','V',3),('V','W',2)], True)
    close(np.array(directed) @ [5,1,9,4], [0,12,0,10])
    close(np.array(directed).T @ [5,1,9,4], [8,0,0,3])
    assert namespaces['walks']['matrix_power'](a,2)[0][2] == 6
    assert namespaces['walks']['matrix_power'](a,2)[0][0] == 5
    calculation = namespaces['incidence']['edge_calculation'](3,[(0,1,1),(1,2,3)],[8,2,0])
    assert calculation[3] == [6,0,-6] and calculation[-1] == 48
    average = namespaces['averaging']
    close(average['iterate'](average['exchange'],[0,4,0],1),[1,2,1])
    close(average['iterate'](average['lazy'],[0,4,0],1),[2,2,2])
    close(average['iterate'](average['exchange'],[0,4,0],100),[4/3]*3)
    harmonic = namespaces['interpolation']['harmonic_values']
    assert harmonic(4,[(0,1,1),(1,2,2),(2,3,1)],{0:0,3:12}) == [0,Fraction(24,5),Fraction(36,5),12]
    try:
        harmonic(5,[(0,1,1),(1,2,2),(2,3,1)],{0:0,3:12})
        raise AssertionError('Unanchored isolated value was accepted')
    except ValueError:
        pass
    close(np.linalg.eigvals([[1,-1,0],[0,1,-1],[0,0,0]]),[1,1,0])
    assert result['invalidContracts'] == 15
    assert len(result['normalizationRangeRejections']) == 3
    native_normalized = namespaces['normalization']['normalized_operators']
    for weight in [1e-310, 5e-309, float.fromhex('0x0.0000000000001p-1022')]:
        try:
            native_normalized([[0, weight], [weight, 0]])
            raise AssertionError('Native non-finite reciprocal accepted')
        except ValueError:
            pass
    native_small = native_normalized([[0, 1e-308], [1e-308, 0]])
    close(native_small[2], [[1,-1],[-1,1]])
    close(native_small[4], [[0,1],[1,0]])
    close(result['smallFiniteNormalization']['symmetric'], [[1,-1],[-1,1]])
    close(result['smallFiniteNormalization']['randomWalk'], [[1,-1],[-1,1]])
    close(result['smallFiniteNormalization']['transition'], [[0,1],[1,0]])
    assert result['tinyFormat'] != '0'
    report = {'checkedAt':datetime.now(timezone.utc).isoformat(),'allPassed':True,'python':sys.version,'numpy':np.__version__,'networkx':nx.__version__,'nativePrograms':len(examples),'originalProgramAndOutputPreserved':True,'undirectedWeightedGraphs':len(cases),'directedGraphs':len(directed_cases),'walkCases':len(result['walks']),'harmonicCases':len(harmonic_cases),'averagingCases':len(result['averaging']),'invalidContracts':result['invalidContracts'],'maxEnergyError':max_energy_error,'maximumHarmonicResidual':maximum_residual,'changedPractice':'Actual learner functions: ordering, matrix walks, signed energy, averaging, rational harmonic solving and rejected unanchored isolate; independent directed eigenvalue counterexample.','sourceHashes':{str(file.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(file.read_bytes()).hexdigest() for file in [ROOT/'src/learn/data/graph-fundamentals-models.js',ROOT/'src/learn/data/graph-fundamentals-examples.js']}}
    report['normalizationRangeRegression'] = {
        'rejectedNonfiniteReciprocalWeights': result['normalizationRangeRejections'],
        'acceptedFiniteReciprocalWeight': 1e-308,
        'finiteNormalizedMatricesPassed': True,
        'actualNativeHelperRejectionsAndFiniteMatricesPassed': True,
    }
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
