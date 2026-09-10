"""Independent dense/library and finite combinatorial oracles for spectral lessons."""
import contextlib
from datetime import datetime, timezone
import io
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'scratch/spectral-verification'
data = json.loads((DIR/'cases.json').read_text(encoding='utf-8'))
counts = {}
def close(actual, expected, atol=3e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-8)

def spectrum_check(matrix, state):
    M = np.asarray(matrix)
    values = np.asarray(state['values'])
    U = np.array(state['vectors']).T
    reference = np.linalg.eigvalsh(M)
    close(values, reference)
    close(U.T@U, np.eye(len(M)))
    close(M@U, U*values)
    close((U*values)@U.T, M)

for row in data['spectra']:
    state = row['state']
    spectrum_check(state['laplacian'], state)
    expected_components = 2 if row['weight']==0 else 1
    assert len(state['components']) == expected_components
counts['bridge_spectra'] = len(data['spectra'])
for row in data['arbitrary']:
    graph = row['graph']
    A = np.array(graph['adjacency'])
    closure = A>0
    np.fill_diagonal(closure,True)
    for k in range(len(A)):
        closure |= closure[:,k,None] & closure[None,k,:]
    expected = len({tuple(v) for v in closure})
    assert len(graph['components']) == expected
    spectrum_check(graph['laplacian'],row['spectrum'])
    spectrum_check(graph['normalized'],row['normalized'])
counts['arbitrary_weighted_graphs'] = len(data['arbitrary'])

def cut_oracle(A,S):
    indicator = np.array([i in S for i in range(len(A))])
    degree = A.sum(axis=1)
    # Independently count edges that disagree with a Boolean node labeling.
    cut = sum(A[i,j] for i in range(len(A)) for j in range(i+1,len(A)) if indicator[i]!=indicator[j])
    volumes = [float(degree[indicator].sum()),float(degree[~indicator].sum())]
    sizes = [int(indicator.sum()),int((~indicator).sum())]
    return cut,volumes,cut/min(volumes),cut*sum(1/v for v in volumes),cut*sum(1/s for s in sizes)

for row in data['cuts']:
    state = row['state']; A=np.array(state['adjacency'])
    spectrum_check(state['normalized'],state['spectrum'])
    values = np.array(state['spectrum']['values'])
    y=np.array(state['coordinates']); d=A.sum(axis=1)
    if row['weight'] > 0:
        close(y@d,0)
    close(np.array(state['laplacian'])@y, values[1]*d*y)
    for candidate in state['candidates']:
        actualS={i for i,value in enumerate(y) if value<candidate['threshold']}
        assert actualS==set(candidate['nodes'])
        cut,volumes,phi,ncut,ratio = cut_oracle(A,actualS)
        close([candidate['cut'],candidate['volume'],candidate['otherVolume'],candidate['conductance'],candidate['normalizedCut'],candidate['ratioCut']],[cut,*volumes,phi,ncut,ratio])
    opt=min(cut_oracle(A,{i for i in range(6) if mask&(1<<i)})[2] for mask in range(1,63))
    assert opt+1e-8>=values[1]/2
    assert state['best']['conductance'] <= np.sqrt(max(0,2*values[1]))+1e-7
counts['normalized_sweep_graphs'] = len(data['cuts'])
counts['exhaustive_cut_partitions'] = len(data['cuts'])*62

for row in data['clusters']:
    state=row['state']; N=np.array(state['normalized'])
    spectrum_check(N,state['spectrum'])
    raw=np.array(state['raw']); points=np.array(state['points'])
    close(raw/np.linalg.norm(raw,axis=1)[:,None], points)
    close(np.linalg.norm(points,axis=1),np.ones(9))
    previous=None; lastloss=float('inf')
    for frame in state['frames']:
        centers=np.array(frame['centroids']); labels=frame['labels']
        if labels is None: continue
        labels=np.array(labels)
        loss=np.sum((points-centers[labels])**2)
        close(loss,frame['loss'])
        assert loss<=lastloss+1e-9
        lastloss=loss
        if frame['phase'] in ['Assign nearest center','Assignments stable']:
            distances=np.sum((points[:,None,:]-centers[None,:,:])**2,axis=2)
            chosen=distances[np.arange(9),labels]
            close(chosen,distances.min(axis=1))
        elif frame['phase']=='Move to means':
            for group in range(3):
                if np.any(labels==group): close(centers[group],points[labels==group].mean(axis=0))
        previous=frame
counts['clustering_runs'] = len(data['clusters'])

signals={'groups':[1,1,1,-1,-1,-1],'noisy':[1.6,.4,1,-1.6,-.4,-1],'spike':[1,0,0,0,0,0],'constant':[2]*6}
graphs={row['weight']:np.array(row['state']['laplacian']) for row in data['spectra']}
cache={}
for row in data['filters']:
    L=graphs[row['weight']]; x=np.array(signals[row['signal']]); amount=row['amount']
    key=(row['weight'],row['filter'],amount)
    if row['filter']!='cutoff':
        if key not in cache:
            cache[key]=expm(-amount*L) if row['filter']=='heat' else np.linalg.solve(np.eye(6)+amount*L,np.eye(6))
        expected=cache[key]@x
        close(row['output'],expected)
        close(row['discarded'],np.sum((expected-x)**2))
        close(row['energyAfter'],expected@L@expected)
    else:
        state=row['state']; U=np.array(state['vectors']).T
        projector=U[:,:amount]@U[:,:amount].T
        output=np.array(state['output'])
        close(projector@x,output)
        close(projector@projector,projector)
        close(L@projector,projector@L)
        if not state['repeatedBoundary']:
            _,V=np.linalg.eigh(L)
            close(projector,V[:,:amount]@V[:,:amount].T)
    output=np.array(row['output'] if 'output' in row else row['state']['output'])
    assert float(output@L@output)<=float(x@L@x)+1e-8
counts['filter_states'] = len(data['filters'])
counts['independent_matrix_functions'] = len(cache)

namespaces={}
for example in data['examples']:
    namespace={'__name__':'__main__'}
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer): exec(compile(example['code'],example['id'],'exec'),namespace)
    assert buffer.getvalue().rstrip()==example['expected'],example['id']
    namespaces[example['id']]=namespace
counts['displayed_native_programs']=len(namespaces)

# Actual displayed helpers: changed graphs and grounded versus series-path resistance.
resistance=namespaces['resistance']['effective_resistance']
grounded=namespaces['resistance']['grounded_resistance']
rng=np.random.default_rng(35); native_circuits=0
for n in range(2,10):
    for _ in range(12):
        conductances=rng.uniform(.2,4,n-1)
        A=np.diag(conductances,1)+np.diag(conductances,-1)
        L=np.diag(A.sum(axis=1))-A
        for a,b in [(0,n-1),(0,1),(n//2,n-1)]:
            expected=sum(1/conductances[i] for i in range(a,b))
            close(resistance(L,a,b),expected)
            close(grounded(L,a,b),expected)
            native_circuits+=1
counts['changed_native_series_circuits']=native_circuits

rows_helper=namespaces['clusters']['spectral_rows']
lloyd=namespaces['clusters']['lloyd']
for n in range(3,10):
    A=np.ones((n,n))-np.eye(n)
    _,points=rows_helper(A,min(3,n))
    close(np.linalg.norm(points,axis=1),np.ones(n))
    for _ in range(8):
        result=lloyd(points,min(3,n),rng.choice(n,min(3,n),replace=False))
        if result is not None:
            labels,centers,loss=result
            close(centers,np.array([points[labels==j].mean(axis=0) for j in range(min(3,n))]))
            distances=np.sum((points[:,None,:]-centers[None,:,:])**2,axis=2)
            close(distances[np.arange(n),labels],distances.min(axis=1))
            close(loss,np.sum((points-centers[labels])**2))
counts['changed_native_clustering_trials']=56

counts['invalid_model_inputs']=data['invalid']
result={'verifiedAt':datetime.now(timezone.utc).isoformat(),'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__},'counts':counts,'originalProgramPreserved':True,'limits':'Finite bounded graphs and fixtures, independent library/finite oracles; not empirical clustering quality or a proof that floating-point thresholds attain exact theorem bounds in arbitrary ranges.'}
(DIR/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
