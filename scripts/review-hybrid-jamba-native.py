"""Independent fresh-input/chunk-partition oracles; no fitting or downloads."""
from pathlib import Path
import copy,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
ID='hybrid-ssm-transformer-architectures-jamba'
SOURCE=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
sys.dont_write_bytecode=True
sys.path.insert(0,str(SOURCE))
from stroke_models import load_fit
torch.set_num_threads(1)
rng=np.random.default_rng(9371)
coordinates=[rng.integers(0,101,size=(8,2)).tolist(),[[0,100],[100,0],[0,0],[100,100],[37,82],[82,37],[50,0],[0,50]]]
records=[]
for key in ['MMM-37','AAA-37','MAM-37','AMM-37','MAM-73']:
    model=load_fit(key).double()
    for input_index,raw in enumerate(coordinates):
        tensor=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
        points=(tensor/50-1)[None]
        full=model(points,True)
        probe=torch.linspace(-.7,.8,10,dtype=torch.float64)
        gradient=torch.autograd.grad((full[0,-1]*probe).sum(),tensor)[0]
        with torch.no_grad():
            pieces=[];cache=None;offset=0
            for count in [2,1,2,3]:
                part,cache=model.stream(points[:,offset:offset+count],cache,offset=offset)
                pieces.append(part);offset+=count
            chunked=torch.cat(pieces,1)
            torch.testing.assert_close(chunked,full,atol=1e-10,rtol=1e-10)
            branches=[]
            for boundary in [1,4,7]:
                prefix,original=model.stream(points[:,:boundary])
                for mode in ['carry','recurrent-reset','kv-reset','convolution-reset','position-reset']:
                    cache=copy.deepcopy(original)
                    for i,kind in enumerate(model.pattern):
                        if (mode=='recurrent-reset' and kind=='M') or (mode=='kv-reset' and kind=='A'):cache[i]=None
                        if mode=='convolution-reset' and kind=='M':cache[i]=(cache[i][0],torch.zeros_like(cache[i][1]))
                    suffix,_=model.stream(points[:,boundary:],cache,offset=0 if mode=='position-reset' else boundary)
                    branches.append({'boundary':boundary,'mode':mode,'logits':torch.cat([prefix,suffix],1)[0].tolist()})
        records.append({'key':key,'input':input_index,'coordinates':raw,'full':full[0].detach().tolist(),'chunked':chunked[0].tolist(),'probe':probe.tolist(),'gradient':gradient.tolist(),'branches':branches})
linear=load_fit('linear-37').double()
linear_cases=[]
for raw in coordinates:
    with torch.no_grad():y=linear((torch.tensor(raw,dtype=torch.float64)/50-1)[None])[0]
    linear_cases.append({'coordinates':raw,'logits':y.tolist()})
result={'passed':True,'seed':9371,'threads':1,'fitsRerun':0,'records':records,'linear':{'pattern':'linear','state':{k:v.tolist() for k,v in linear.state_dict().items()},'cases':linear_cases}}
(OUT/'independent-native-fixtures.json').write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'freshFullInputs':len(records),'freshFaultBranches':sum(len(r['branches']) for r in records),'gradientCoordinates':160,'chunkPartition':[2,1,2,3]}))
