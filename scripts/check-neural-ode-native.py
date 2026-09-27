"""Execute the published ODE arithmetic/library programs and replay fitted checkpoints."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import numpy as np
import torch
import torchdiffeq
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
ID='neural-ode-continuous-depth-models'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
torch.set_num_threads(2)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ode-native-',dir=ROOT/'scratch') as folder:
        folder=Path(folder)
        for name in ['neural_ode_study.py','ode_calculations.py','iris.csv','fitted-models.json']:
            shutil.copyfile(PACKET/name,folder/name)
        sys.path.insert(0,str(folder))
        study=load('neural_ode_study',folder/'neural_ode_study.py')
        calculations=load('ode_calculations',folder/'ode_calculations.py')
        text=io.StringIO()
        with contextlib.redirect_stdout(text):calculations.main()
        fixture=json.loads((folder/'calculated-inputs.json').read_text())
        fixture['matrix_cases']=[]
        for matrix in [[[0,-1],[1,0]], [[1,1],[0,1]], [[-2,3],[0,-2]], [[.2,2],[.4,-.1]], [[0,0],[0,0]], [[1,0],[0,-1]], [[2,1e-6],[-1e-6,2]]]:
            for time in [.1,.7,2]:fixture['matrix_cases'].append({'matrix':matrix,'time':time,'exponential':expm(np.array(matrix)*time).tolist()})
        x,y,roles,metadata=study.load_data()
        snapshots=json.loads((PACKET/'fitted-models.json').read_text())
        report=json.loads((PACKET/'study-results.json').read_text())
        assert metadata==report['data']
        replay=[]
        fixture['fresh_models']=[]
        for snapshot,run in zip(snapshots,report['runs'],strict=True):
            assert(snapshot['kind'],snapshot['seed'])==(run['kind'],run['seed'])
            model=study.DepthClassifier(snapshot['kind']);model.load_state_dict({k:torch.tensor(v) for k,v in snapshot['state'].items()})
            selected=min(run['curves'],key=lambda row:(row['validation']['cross_entropy'],row['step']))
            assert selected['step']==snapshot['selected_step']==run['selected_step']
            for role,indices in roles.items():
                actual=study.evaluate(model,x[indices],y[indices]);expected=run[role]
                assert actual['correct']==expected['correct'] and actual['count']==expected['count']
                assert abs(actual['cross_entropy']-expected['cross_entropy'])<1e-12
            replay.append({'kind':snapshot['kind'],'seed':snapshot['seed'],'selectedStep':selected['step'],'assessment':actual})
            if 'ode' in snapshot['kind']:
                for raw,steps,method in [([6.2,2.6,4.9,1.8],4,'rk4'),([5.4,3.6,2.2,.8],16,'euler')]:
                    standardized=(np.array(raw)-metadata['mean'])/metadata['scale']
                    tensor=torch.tensor(standardized[None,:],requires_grad=True)
                    logits,trace=model(tensor,steps,method,return_trace=True)
                    derivative=torch.autograd.grad(logits.softmax(-1)[0,1],tensor)[0][0].detach().numpy()/metadata['scale']
                    fixture['fresh_models'].append({'kind':snapshot['kind'],'seed':snapshot['seed'],'raw':raw,'steps':steps,'method':method,'trace':trace[:,0].detach().tolist(),'logits':logits[0].detach().tolist(),'probabilities':logits[0].softmax(-1).detach().tolist(),'gradient_per_cm':derivative.tolist()})
        fixture['metadata']=metadata
        # Extra independently differentiated fixed-step cases include the Euler zero multiplier.
        fixture['extra_gradients']=[calculations.scalar_gradient(float(rate),float(z),float(target),float(time),n,method) for rate,z,target,time,n,method in [(-4,1,.2,1,4,'euler'),(-1.2,-.8,.3,1.7,3,'rk4'),(.9,.4,-.5,1.8,4,'euler')]]
        sys.path.pop(0)
    bridge=load('ode_library_bridge',PACKET/'solver_library_bridge.py')
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):bridge.main()
    (OUT/'native-fixtures.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    receipt={'passed':True,'torch':torch.__version__,'torchdiffeq':torchdiffeq.__version__,'maximumThreads':2,'checks':['Complete unchanged published ode_calculations.py executed in isolated temporary directory','All12 selected checkpoints replay all149 unique rows by declared fit/validation/assessment roles','Validation-only selected checkpoint identity','Actual torchdiffeq Euler state/gradient and dopri5 direct/adjoint analytic checks','21 SciPy matrix exponentials covering repeated, complex, zero and real eigenvalues','Additional native gradient fixtures including exact-zero Euler amplification'],'libraryOutput':stream.getvalue(),'checkpointReplay':replay,'limitations':['Existing12 fits reused; no duplicate fit campaign.','No JAX/Diffrax or GPU execution.']}
    (OUT/'native-checks.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'checkpoints':len(replay),'torchdiffeq':torchdiffeq.__version__,'libraryOutput':stream.getvalue()}))

if __name__=='__main__':main()
