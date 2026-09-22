"""Reviewer-owned native oracle; no author model/helper imports, no training."""
import json
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
path = root/'docs/teaching/evidence/depthwise-independent-native.json'
path.write_text('{"passed":false,"status":"running"}', encoding='utf-8')
source = root/'public/learn-code/depthwise-separable-dilated-convolutions/digit-inference.json'
saved = json.loads(source.read_text())
fixtures = []
for run in saved['runs']:
    dense = {key:torch.tensor(value,dtype=torch.float64) for key,value in run['dense_state'].items()}
    images = [torch.linspace(0,1,64,dtype=torch.float64).reshape(8,8), (torch.arange(64).reshape(8,8)%2).double(), torch.zeros(8,8,dtype=torch.float64)]
    images[2][7,0] = .75
    for image_index,image in enumerate(images):
        stem = F.relu(F.conv2d(image[None,None],dense['stem.weight'],dense['stem.bias'],padding=1))
        for rank in [None,1,2,4,9]:
            if rank is None:
                weight,bias = dense['spatial.weight'],dense['spatial.bias']
            else:
                factors=next(f for f in run['factorizations'] if f['multiplier']==rank)['factorized_spatial_state']
                d=torch.tensor(factors['0.weight'],dtype=torch.float64).reshape(8,rank,3,3)
                p=torch.tensor(factors['1.weight'],dtype=torch.float64).reshape(12,8,rank)
                # Use effective dense kernels, unlike the browser's grouped two-pass route.
                weight=torch.einsum('ocr,crij->ocij',p,d)
                bias=torch.tensor(factors['1.bias'],dtype=torch.float64)
            features=F.relu(F.conv2d(stem,weight,bias,padding=run['dilation'],dilation=run['dilation']))
            features=F.avg_pool2d(features,2).flatten(1)
            logits=F.linear(features,dense['head.weight'],dense['head.bias'])[0]
            fixtures.append({'dilation':run['dilation'],'kind':['ramp','checkerboard','corner-impulse'][image_index],'rank':rank,'image':image.tolist(),'logits':logits.tolist()})
gradients=[]
for seed in [2,19,41]:
    torch.manual_seed(seed)
    x=torch.rand(2,3,dtype=torch.float64)*4-2
    d=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    p=torch.randn(2,dtype=torch.float64,requires_grad=True)
    target=float(torch.randn(1));rate=.025
    spatial=F.conv1d(x[None],d[:,None,:],groups=2)
    outputs=F.conv1d(spatial,p.reshape(1,2,1))[0,0]
    loss=(outputs[0]-target).square()/2
    loss.backward()
    with torch.no_grad():
        new_d=d-rate*d.grad;new_p=p-rate*p.grad
        after=F.conv1d(F.conv1d(x[None],new_d[:,None,:],groups=2),new_p.reshape(1,2,1))[0,0,0]
    gradients.append({'input':x.tolist(),'filters':d.detach().tolist(),'mixing':p.detach().tolist(),'target':target,'rate':rate,'outputs':outputs.detach().tolist(),'filter_gradient':d.grad.tolist(),'mixing_gradient':p.grad.tolist(),'after':float(after)})
report={'passed':True,'torch':torch.__version__,'fixtures':fixtures,'gradients':gradients,'sourceHash':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'Independent torch float64 dense effective-kernel inference on three new image patterns, plus autograd convolution gradients; no author helper imports or fits'}
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'freshScoreVectors':len(fixtures),'gradientStates':len(gradients)}))
