"""Replay saved fits, execute all four manuscript programs and export lossless readers."""
from pathlib import Path
import ast, contextlib, copy, importlib.util, io, json, re, shutil, sys, tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import oaconvolve
from scipy.special import erf

ROOT=Path(__file__).resolve().parents[1]
ID='hyena-long-convolution-models'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
PUBLIC=ROOT/'public/learn-code'/ID
torch.set_num_threads(2)
checks=[]
def check(name,value=True):
    assert value,name
    checks.append({'name':name,'passed':True})
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec)
    sys.modules[name]=loaded
    spec.loader.exec_module(loaded)
    return loaded
def compare(actual,expected,tolerance=4e-6):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:compare(actual[key],expected[key],tolerance)
    elif isinstance(actual,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):compare(a,b,tolerance)
    elif isinstance(actual,(float,int)):assert abs(actual-expected)<=tolerance+1e-6*abs(expected),(actual,expected)
    else:assert actual==expected
def capture_forward(model,tokens,options):
    length=tokens.shape[1];stages={}
    names=['head'] if model.kind=='linear' else ['embedding']+[f'blocks.{b}.{name}' for b in range(2) for name in ['norm','project','short','filter.first','filter.last','filter','output','feed_norm','feed.0','feed.1','feed.2']]+['blocks.0','blocks.1','head']
    def hook(name):
        def capture(layer,inputs,output):
            value=output.detach()
            if name.endswith('.short'):value=value.transpose(1,2)[:,:length]
            if value.ndim>1 and value.shape[0]==1:value=value[0]
            stages[name]=value.tolist()
        return capture
    handles=[dict(model.named_modules())[name].register_forward_hook(hook(name)) for name in names]
    try:logits=model(tokens,**options)[0]
    finally:
        for handle in handles:handle.remove()
    return logits,stages

def main():
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hyena-native-',dir=ROOT/'scratch') as temporary:
        tmp=Path(temporary)
        for name in ['convolution_mechanisms.py','blocked_convolution.py','splice_models.py','author_calculations.py','splice.data','splice.names','splice-fits.npz']:
            shutil.copyfile(PACKET/name,tmp/name)
        sys.path.insert(0,str(tmp))
        mechanisms=module('convolution_mechanisms',tmp/'convolution_mechanisms.py')
        blocked=module('blocked_convolution',tmp/'blocked_convolution.py')
        models=module('splice_models',tmp/'splice_models.py')
        author=module('author_calculations',tmp/'author_calculations.py')
        for program,filename in [(mechanisms,'mechanism-results.json'),(author,'author-results.json')]:
            with contextlib.redirect_stdout(io.StringIO()):program.main()
            compare(json.loads((tmp/filename).read_text()),json.loads((PACKET/filename).read_text()))
            check('Complete canonical program replay '+filename)
        with contextlib.redirect_stdout(io.StringIO()):blocked.main()
        check('Complete blocked FFT and SciPy overlap-add bridge executed')
        manuscript=(PACKET/'lesson.md').read_text(encoding='utf-8')
        programs=re.findall(r'```python\n(.*?)```',manuscript,re.S)
        check('Four complete embedded Python blocks',len(programs)==4)
        for i,program in enumerate(programs):
            namespace={'torch':torch,'__name__':'__main__'}
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(program,f'manuscript-{i}','exec'),namespace)
            if i==0:np.testing.assert_allclose(namespace['fast'],[1,2.5,4.25,6.125],atol=1e-12)
            elif i==1:
                result=namespace['causal_convolution'](torch.tensor([[[1.],[2.],[3.],[4.]]],dtype=torch.float64),torch.tensor([[1.],[.5],[.25],[.125]],dtype=torch.float64))
                np.testing.assert_allclose(result.flatten(),[1,2.5,4.25,6.125],atol=1e-12)
            elif i==2:check('Embedded block program equals complete downloadable source',program.strip()==(PACKET/'blocked_convolution.py').read_text(encoding='utf-8').strip())
            else:np.testing.assert_allclose(namespace['outputs'],[1,-1.8,.275,2.81875,-.4109375,2.299609375],atol=1e-12)
            check('Executed displayed Python block '+str(i+1))
        rng=np.random.default_rng(473)
        blocked_cases=[]
        for n,k,b in [(1,1,1),(5,3,2),(10,5,3),(3,9,2),(12,1,20),(12,12,1),(7,4,8),(10,8,3)]:
            values=rng.normal(size=n);kernel=rng.normal(size=k)
            expected=np.array([sum(kernel[t-j]*values[j] for j in range(t+1) if t-j<k) for t in range(n)])
            actual=blocked.overlap_add_fft(values,kernel,b)
            np.testing.assert_allclose(actual,expected,atol=3e-14)
            np.testing.assert_allclose(oaconvolve(values,kernel,mode='full')[:n],expected,atol=3e-14)
            blocked_cases.append({'values':values.tolist(),'kernel':kernel.tolist(),'blockSize':b,'output':expected.tolist(),'same':oaconvolve(values,kernel,mode='same').tolist()})
        check('Eight independent literal convolution/block/SciPy cases including short final and long kernel')
        tokens,labels,roles,data=models.load_data()
        results=json.loads((PACKET/'splice-results.json').read_text())
        compare(data,results['data'],0)
        check('Complete raw deduplication,conflicts,prefix unions,role identities and class counts replay')
        raw=[tuple(part.strip() for part in line.split(',')) for line in (PACKET/'splice.data').read_text().splitlines()]
        validation=[{'sourceId':data['source_ids'][index],'label':models.CLASSES[int(labels[index])],'sequence':raw[data['source_ids'][index]-1][2]} for index in roles['validation']]
        save(PUBLIC/'validation-sequences.json',{'rows':validation,'alphabet':models.ALPHABET,'classes':models.CLASSES,'attribution':'Towell,Noordewier,Shavlik; UCI Splice-Junction dataset69,CC BY4.0. See data-provenance.md.'})
        port={'mechanisms':json.loads((tmp/'mechanism-results.json').read_text()),'author':json.loads((tmp/'author-results.json').read_text()),'blocked':blocked_cases,'validation':{},'fresh':[],'erf':[[float(x),float(erf(x))] for x in np.linspace(-12,12,1201)]}
        with np.load(tmp/'splice-fits.npz',allow_pickle=False) as saved:
            for fit in results['fits']:
                kind,seed=fit['kind'],fit['seed'];key=f'{kind}_{seed}';prefix=key+'__state__'
                state={name.removeprefix(prefix):torch.tensor(saved[name]) for name in saved.files if name.startswith(prefix)}
                model=models.SpliceReader(kind);model.load_state_dict(state);model.eval()
                check(key+' parameter count and minimum-validation selection',sum(p.numel() for p in model.parameters())==fit['parameters'] and min(fit['history'],key=lambda row:row[1])[0]==fit['selected_epoch'])
                with torch.no_grad():
                    for role,ids in roles.items():
                        logits=torch.cat([model(tokens[part]) for part in np.array_split(ids,max(1,len(ids)//128))])
                        compare(models.metrics(logits,labels[ids]),fit[role],4e-6)
                        np.testing.assert_allclose(logits.numpy(),saved[key+'__'+role+'_logits'],atol=2e-5,rtol=2e-6)
                        check(key+' full selected-checkpoint '+role+' metrics/logits')
                    if kind=='gated':
                        for label,options in [('lag_0_to_4',{'kernel_limit':5}),('gates_off',{'gates_off':True})]:
                            compare(models.metrics(model(tokens[roles['assessment']],**options),labels[roles['assessment']]),fit[label])
                            check(key+' reported post-fit '+label+' intervention')
                port['validation'][key]=saved[key+'__validation_logits'].tolist()
                save(PUBLIC/f'model-{key}.json',{'id':key,'kind':kind,'seed':seed,'selectedEpoch':fit['selected_epoch'],'parameters':{name:value.tolist() for name,value in state.items()},'history':fit['history'],'metrics':{name:value for name,value in fit.items() if isinstance(value,dict)}})
                training=copy.deepcopy(model);training.train();optimizer=torch.optim.Adam(training.parameters(),lr=.003)
                loss=F.cross_entropy(training(tokens[roles['fit'][:128]]),labels[roles['fit'][:128]]);loss.backward();norm=nn.utils.clip_grad_norm_(training.parameters(),1.)
                before=next(training.parameters()).detach().clone();optimizer.step()
                check(key+' real128-example gradient and Adam update',torch.isfinite(norm).item() and not torch.equal(before,next(training.parameters())))
                double=copy.deepcopy(model).double()
                # The canonical linear branch explicitly constructs float32 one-hot
                # inputs. Cast that exact representation at its linear boundary for
                # this separate float64 arithmetic/gradient comparison.
                if kind=='linear':double.head.register_forward_pre_hook(lambda layer,args:(args[0].double(),))
                cases=[('source4',validation[1]['sequence']),('allN','N'*60),('independent','ACGTDNRS'*7+'TGCA')]
                for case,sequence in cases:
                    token=torch.tensor([[models.ALPHABET.index(char) for char in sequence]])
                    options_list=[{}] if kind=='linear' else [{},{'kernel_limit':5},{'gates_off':True}]
                    for options in options_list:
                        logits,stages=capture_forward(double,token,options)
                        loss=logits.square().sum()/2
                        double.zero_grad();loss.backward()
                        probes=[]
                        chosen=[('head.weight',(0,0)),('head.bias',(1,))] if kind=='linear' else [('embedding.weight',(0,2)),('blocks.0.project.weight',(0,3)),('blocks.0.project.weight',(16,3)),('blocks.0.project.weight',(32,3)),('blocks.0.short.weight',(17,0,0)),('blocks.0.filter.first.weight',(2,0)),('blocks.0.filter.decay',(3,)),('blocks.0.skip',(4,)),('blocks.1.feed.0.weight',(2,3)),('head.weight',(0,3))]
                        named=dict(double.named_parameters())
                        for name,index in chosen:probes.append({'name':name,'index':list(index),'gradient':float(named[name].grad[index])})
                        port['fresh'].append({'modelId':key,'case':case,'sequence':sequence,'options':options,'logits':logits.detach().tolist(),'stages':stages,'parameterGradients':probes})
                if kind!='linear':
                    with torch.no_grad():
                        sample=tokens[roles['validation'][:4]];whole=double(sample,return_hidden=True);prefix_hidden=double(sample[:,:31],return_hidden=True);edited=sample.clone();edited[:,31:]=models.ALPHABET.index('N')
                        assert (whole[:,:31]-prefix_hidden).abs().max()<1e-11
                        assert (whole[:,:31]-double(edited,return_hidden=True)[:,:31]).abs().max()<1e-11
                    check(key+' fresh float64 prefix/future invariant with unchanged saved position buffers')
                check(key+' fresh complete native layers and parameter gradients')
        save(OUT/'native-port-fixtures.json',port)
        save(OUT/'native-checks.json',{'topicId':ID,'passed':True,'checks':checks,'environment':{'torch':torch.__version__,'numpy':np.__version__,'threads':2},'selectedFits':4,'validationSequences':1840,'freshCases':len(port['fresh']),'limits':['Original four80-epoch fitting campaigns replayed from saved selected fits,not refitted.','No pretrained HyenaDNA/StripedHyena,GPU benchmark or browser execution.']})
        sys.path.remove(str(tmp))
    print(json.dumps({'passed':True,'checks':len(checks),'freshCases':len(port['fresh'])}))
if __name__=='__main__':main()
