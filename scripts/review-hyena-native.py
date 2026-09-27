"""Fresh full-sequence oracles and complete embedding/filter-state gradient probes."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
ID='hyena-long-convolution-models'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
sys.path.insert(0,str(PACKET))
from splice_models import SpliceReader,ALPHABET
torch.set_num_threads(1)
rng=np.random.default_rng(671033)
saved=np.load(PACKET/'splice-fits.npz',allow_pickle=False)
cases=[]
for kind,seed in [('gated',29),('ungated',29),('gated',71)]:
    prefix=f'{kind}_{seed}__state__'
    state={key[len(prefix):]:torch.tensor(saved[key]) for key in saved.files if key.startswith(prefix)}
    model=SpliceReader(kind);model.load_state_dict(state);model=model.double().eval()
    for length in [17,60]:
        tokens=torch.tensor([list(range(8))+rng.integers(0,8,length-8).tolist()])
        probe=torch.tensor(rng.normal(size=3),dtype=torch.float64)
        logits=model(tokens)[0]
        hidden=model(tokens,return_hidden=True)[0]
        gradients=[]
        if length==60:
            objective=logits@probe
            names=['embedding.weight']+[f'blocks.{b}.filter.{p}' for b in range(2) for p in ['first.bias','decay']]
            parameters=dict(model.named_parameters())
            derivatives=torch.autograd.grad(objective,[parameters[name] for name in names])
            for name,derivative in zip(names,derivatives):
                gradients.append({'name':name,'values':derivative.detach().tolist()})
        cases.append({'modelId':f'{kind}_{seed}','sequence':''.join(ALPHABET[t] for t in tokens[0]),'probe':probe.tolist(),'logits':logits.detach().tolist(),'hidden':hidden.detach().tolist(),'gradients':gradients})
(OUT/'independent-native-fixtures.json').write_text(json.dumps({'cases':cases,'environment':{'torch':torch.__version__,'numpy':np.__version__,'dtype':'float64','threads':1,'device':'CPU'}},separators=(',',':'))+'\n')
print(json.dumps({'freshModels':3,'fullSequenceCases':6,'completeParameterDerivativeCoordinates':672,'scope':'All embedding entries and both filter-network first biases/decay vectors; no refitting.'}))
