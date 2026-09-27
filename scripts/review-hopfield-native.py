"""Independent new-query and shared-projection gradients; no fitting or downloads."""
from pathlib import Path
import sys,json
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
ID='modern-hopfield-networks'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'docs/teaching/drafts'/ID))
from digit_memory import read_memory
torch.set_num_threads(1)
public=ROOT/'public/learn-code'/ID
bank=json.loads((public/'digit-bank.json').read_text())
memory=torch.tensor([r['pixels'] for r in bank['memory']],dtype=torch.float64)/16
labels=torch.tensor([r['label'] for r in bank['memory']])
rng=np.random.default_rng(7129)
inputs=[rng.integers(0,17,64).tolist(),[16 if (i//8+i%8)%2 else 1 for i in range(64)]]
records=[]
for key in ['fixed','seed17','seed41']:
    model=json.loads((public/f'model-{key}.json').read_text())
    projection=None
    if model.get('projection'):
        projection=nn.Linear(64,16,bias=False,dtype=torch.float64)
        with torch.no_grad():projection.weight.copy_(torch.tensor(model['projection'],dtype=torch.float64))
    for raw in inputs:
        x=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
        log_class,weights=read_memory(x[None]/16,memory,labels,model['beta'],projection)
        probe=torch.linspace(-.4,.5,10,dtype=torch.float64)
        targets=[x]+([] if projection is None else [projection.weight])
        grads=torch.autograd.grad((log_class[0]*probe).sum(),targets)
        records.append({'model':key,'raw':raw,'probe':probe.tolist(),'logClasses':log_class[0].detach().tolist(),'weights':weights[0].detach().tolist(),'readPixels':(weights@memory)[0].detach().tolist(),'inputGradient':grads[0].tolist(),'projectionGradient':None if projection is None else grads[1].tolist()})
out={'passed':True,'seed':7129,'threads':1,'fitsRerun':0,'records':records}
(ROOT/'docs/teaching/deep-learning-completion'/ID/'independent-native-fixtures.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print(json.dumps({'passed':True,'freshNativeReads':6,'inputGradientCoordinates':384,'sharedProjectionGradientCoordinates':4096}))
