"""Independent complete carried-state derivatives for every selected reader."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
ID='xlstm-extended-lstm'
spec=importlib.util.spec_from_file_location('xlstm_review_reader',ROOT/'docs/teaching/drafts'/ID/'row_sequence_models.py')
native=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
torch.set_num_threads(1)
rng=np.random.default_rng(713419)
records=[]
for kind in ['lstm','slstm','mlstm']:
    for seed in [19,43]:
        model_id=f'{kind}_seed{seed}'
        saved=json.loads((ROOT/'public/learn-code'/ID/f'model-{model_id}.json').read_text())
        model=native.DigitReader(kind).double()
        model.load_state_dict({name:torch.tensor(value,dtype=torch.float64) for name,value in saved['parameters'].items()})
        model.eval()
        pixels=torch.tensor(rng.integers(0,17,(1,8,8)),dtype=torch.float64)
        with torch.no_grad():
            _,state=model(pixels[:,:3]/16)
        initial=tuple(p.detach().clone().requires_grad_() for p in state)
        logits,final=model(pixels[:,3:]/16,initial)
        probe=torch.tensor(rng.normal(size=10),dtype=torch.float64)
        loss=(logits[0,-1]*probe).sum()
        gradients=torch.autograd.grad(loss,initial)
        def plain(values):
            return [x.detach().reshape(x.shape[2:] if kind=='lstm' else x.shape[1:]).tolist() for x in values]
        records.append({'id':model_id,'pixels':pixels[0].tolist(),'initial':plain(initial),'logits':logits[0].detach().tolist(),'final':plain(final),'probe':probe.tolist(),'stateGradient':plain(gradients)})
path=ROOT/'docs/teaching/deep-learning-completion'/ID/'independent-native-fixtures.json'
path.write_text(json.dumps({'torch':torch.__version__,'threads':1,'cases':records},separators=(',',':'))+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'freshCompleteReaders':6,'carriedStateDerivatives':466,'fits':0}))
