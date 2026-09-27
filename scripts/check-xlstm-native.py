"""Native replay and lossless assets for the three declared recurrent readers."""
from pathlib import Path
import contextlib
import copy
import gc
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
ID='xlstm-extended-lstm'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
PUBLIC=ROOT/'public/learn-code'/ID
torch.set_num_threads(2)
checks=[]

def check(name,condition):
    assert condition,name
    checks.append({'name':name,'passed':True})

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def compare(a,b,tol=3e-6):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k],tol)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y,tol)
    elif isinstance(a,(float,int)):
        assert abs(a-b)<=tol+1e-6*abs(b),(a,b)
    else:assert a==b

def plain_state(state,kind):
    return [x.detach().cpu().numpy().reshape(x.shape[2:] if kind=='lstm' else x.shape[1:]).tolist() for x in state]

def traced(model,image,boundary=3,mode='carry'):
    states=[];logits=[];stages=[];state=None
    names=['input_projection','pre_norm','post_norm','expand','gate','contract','classifier']
    if model.kind=='mlstm':names+=['sequence_model.queries','sequence_model.keys','sequence_model.values','sequence_model.gates','sequence_model.output_gate','sequence_model.read_norm']
    current={}
    def capture(name):
        def hook(module,args,output):current[name]=output.detach().reshape(-1).tolist()
        return hook
    handles=[dict(model.named_modules())[name].register_forward_hook(capture(name)) for name in names]
    try:
        for t in range(image.shape[1]):
            if t==boundary:
                if mode=='reset':state=None
                elif mode=='reset-normalizer' and model.kind!='lstm':
                    state=list(state);state[2 if model.kind=='slstm' else 1]=torch.zeros_like(state[2 if model.kind=='slstm' else 1]);state=tuple(state)
            current={};out,state=model(image[:,t:t+1],state)
            logits.append(out[0,0].detach().tolist());states.append(plain_state(state,model.kind));stages.append(current)
    finally:
        for handle in handles:handle.remove()
    return {'logits':logits,'states':states,'stages':stages}

def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xlstm-check-',dir=ROOT/'scratch') as temporary:
        temp=Path(temporary)
        for file in ['memory_mechanisms.py','row_sequence_models.py','author_calculations.py','row-sequence-fits.npz','optdigits.tra','optdigits.tes','optdigits.names']:
            shutil.copyfile(PACKET/file,temp/file)
        sys.path.insert(0,str(temp))
        operators=load('memory_mechanisms',temp/'memory_mechanisms.py')
        native=load('row_sequence_models',temp/'row_sequence_models.py')
        with contextlib.redirect_stdout(io.StringIO()):
            operators.main()
            calculations=load('xlstm_author_calculations',temp/'author_calculations.py');calculations.main()
        for file in ['mechanism-results.json','investigation-results.json']:
            compare(json.loads((temp/file).read_text()),json.loads((PACKET/file).read_text()))
            check('Complete canonical replay '+file,True)
        source=(PACKET/'lesson.md').read_text(encoding='utf-8')
        inline=re.findall(r'```python\n(.*?)```',source,re.S)
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):exec(compile(inline[0],'xlstm-inline','exec'),{})
        check('Displayed standalone scalar program executed exactly',captured.getvalue()=='0.150000\n-0.364286\n0.443023\n')
        roles,metadata=native.load_data();report=json.loads((PACKET/'row-sequence-results.json').read_text())
        check('All original source bytes and role IDs identical',metadata==report['data'])
        check('Disjoint1000/300 fit/validation roles',len(set(metadata['training_source_ids']['fit'])|set(metadata['training_source_ids']['validation']))==1300)
        with np.load(PACKET/'row-sequence-fits.npz') as archive:
            check('All saved validation images/labels equal exact source role',np.array_equal(archive['validation_images'],roles['validation'][0].numpy()) and np.array_equal(archive['validation_labels'],roles['validation'][1].numpy()))
            port={'validation':[],'fresh':[],'mechanisms':json.loads((temp/'mechanism-results.json').read_text()),'investigations':json.loads((temp/'investigation-results.json').read_text())}
            for selected in report['models']:
                kind,seed=selected['kind'],selected['seed'];model_id=f'{kind}_seed{seed}';prefix=model_id+'__'
                weights={name[len(prefix):]:torch.from_numpy(archive[name].copy()) for name in archive.files if name.startswith(prefix)}
                model=native.DigitReader(kind);model.load_state_dict(weights);model.eval()
                check(model_id+' exact parameter count',sum(x.numel() for x in model.parameters())==selected['parameters'])
                check(model_id+' clean-validation-only selection',min(selected['training_curve'],key=lambda r:r['validation_loss_after_update'])['epoch']==selected['selected_epoch'])
                with torch.no_grad():
                    for role,(images,truth) in roles.items():
                        for condition in ['clean','reversed']:
                            inputs=images if condition=='clean' else images.flip(1)
                            logits,state=model(inputs)
                            compare(native.metrics(logits[:,-1],truth),selected[role+'_'+condition])
                            check(model_id+' '+role+' '+condition+' selected-fit complete metrics',True)
                            if role=='validation':
                                assert np.max(np.abs(logits.numpy()-archive[model_id+'_'+condition+'_logits']))<1e-6
                                port['validation'].append({'id':model_id,'condition':condition,'logits':logits.tolist()})
                    # Every prefix and all carried components at each valid split.
                    x=roles['validation'][0][:3]
                    whole,whole_state=model(x)
                    for split in range(1,8):
                        left,state=model(x[:,:split]);right,final=model(x[:,split:],state)
                        assert torch.allclose(whole,torch.cat([left,right],1),atol=1e-5,rtol=1e-5)
                        assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(final,whole_state))
                    check(model_id+' all seven complete-state carry splits',True)
                save(PUBLIC/('model-'+model_id+'.json'),{'id':model_id,'kind':kind,'seed':seed,'selectedEpoch':selected['selected_epoch'],'parameters':{key:value.tolist() for key,value in weights.items()},'trainingCurve':selected['training_curve'],'metrics':{k:v for k,v in selected.items() if k not in ['training_curve','kind','seed']}})
                # New float64 full-network inputs, all layer outputs, states and gradients.
                double=copy.deepcopy(model).double()
                cases=[('source187',roles['validation'][0][142].double()),('fresh-pattern',torch.tensor([[(r*5+c*3+1)%17/16 for c in range(8)] for r in range(8)],dtype=torch.float64)),('blank',torch.zeros(8,8,dtype=torch.float64))]
                for case,image in cases:
                    for condition in ['clean','reversed','bottom-zero']:
                        pixels=image.clone()
                        if condition=='reversed':pixels=pixels.flip(0)
                        if condition=='bottom-zero':pixels[5:]=0
                        with torch.no_grad():
                            trace=traced(double,pixels[None])
                            reset=traced(double,pixels[None],boundary=3,mode='reset')
                            reset_n=None if kind=='lstm' else traced(double,pixels[None],boundary=3,mode='reset-normalizer')
                        x=pixels[None].clone().requires_grad_();out,_=double(x);loss=F.cross_entropy(out[:,-1],torch.tensor([4]));loss.backward()
                        port['fresh'].append({'id':model_id,'case':case,'condition':condition,'pixels':(pixels*16).tolist(),'trace':trace,'reset':reset,'resetNormalizer':reset_n,'loss':float(loss.detach()),'gradientPerIntensity':(x.grad[0]/16).tolist()})
                check(model_id+' nine fresh full-network state/layer/autograd cases',True)
                # A bounded actual library training step verifies the advertised trainable route.
                before={name:p.detach().clone() for name,p in model.named_parameters()}
                optimizer=torch.optim.Adam(model.parameters(),lr=.003);optimizer.zero_grad()
                logits,_=model(roles['fit'][0]);loss=F.cross_entropy(logits[:,-1],roles['fit'][1]);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                check(model_id+' finite actual clipped training gradient',bool(torch.isfinite(gradient) and gradient>0))
                optimizer.step();check(model_id+' actual Adam update changes weights',any(not torch.equal(before[name],p) for name,p in model.named_parameters()))
            rows=[{'sourceId':source_id,'label':int(label),'pixels':(image.numpy()*16).astype(int).tolist()} for source_id,image,label in zip(metadata['training_source_ids']['validation'],*roles['validation'])]
            save(PUBLIC/'digit-rows.json',{'rows':rows,'source':'UCI Optical Recognition of Handwritten Digits; E.Alpaydin/C.Kaynak; CC BY4.0','normalizationDivisor':16})
            save(OUT/'native-port-fixtures.json',port)
        sys.path.pop(0);gc.collect()
    save(OUT/'native-checks.json',{'topicId':ID,'passed':True,'checks':checks,'environment':{'torch':torch.__version__,'numpy':np.__version__,'maximumThreads':2},'validationReads':3600,'freshNetworkCases':54,'limits':['Six original150-epoch fits replayed from selected checkpoints,not refitted.','No official pretrained xLSTM model/GPU kernel or browser executed.']})
    print(json.dumps({'passed':True,'checks':len(checks),'validationReads':3600,'freshCases':54}))

if __name__=='__main__':main()
