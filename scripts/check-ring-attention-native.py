"""Execute the complete packet and real CPU/Gloo transport, without training."""
from pathlib import Path
import contextlib, hashlib, importlib.util, io, json, os, re, shutil, socket, subprocess, sys, tempfile, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
ID='ring-attention-sequence-parallelism'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
def write(name,data): (OUT/name).write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def module(path):
    spec=importlib.util.spec_from_file_location(path.stem.replace('-','_'),path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def compare(a,b,where='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),where
        return sum(compare(a[k],b[k],where+'/'+k) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b),where
        return sum(compare(x,y,where) for x,y in zip(a,b))
    assert a==b,(where,a,b)
    return int(isinstance(a,(int,float)))
log=[];checks=[]
with tempfile.TemporaryDirectory(prefix='ring-attention-native-',dir=ROOT/'scratch') as temporary:
    work=Path(temporary)
    for name in ['ring-attention-reference.py','attention-partition-study.py','systems-calculations.py','distributed_ring.py','movement-attention-model.json','movement_libras.data']:shutil.copyfile(PACKET/name,work/name)
    for name in ['ring-attention-reference.py','attention-partition-study.py','systems-calculations.py']:
        result=subprocess.run([sys.executable,str(work/name)],capture_output=True,text=True,timeout=90,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        log.append({'program':name,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr});assert result.returncode==0,result.stderr
    replay={name:compare(load(PACKET/name),load(work/name)) for name in ['partition-results.json','systems-results.json']}
    checks.append({'name':'Complete reference, full partition study and systems arithmetic execute; all original records exactly reproduce','passed':True,'numericFields':replay})
    source=(work/'distributed_ring.py').read_text(encoding='utf-8')
    process_runs=[]
    for world,length in [(1,None),(2,None),(3,None),(3,8)]:
        path=work/'distributed_probe.py';path.write_text(source if length is None else source.replace('2*world+1, 2, 3',f'{length}, 2, 3'),encoding='utf-8')
        with socket.socket() as server:server.bind(('127.0.0.1',0));port=server.getsockname()[1]
        processes=[]
        try:
            for rank in range(world):
                env={**os.environ,'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port),'WORLD_SIZE':str(world),'RANK':str(rank),'LOCAL_RANK':str(rank),'USE_LIBUV':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                processes.append(subprocess.Popen([sys.executable,str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
            rows=[]
            for rank,p in enumerate(processes):
                stdout,stderr=p.communicate(timeout=90);rows.append({'rank':rank,'returncode':p.returncode,'stdout':stdout,'stderr':stderr});assert p.returncode==0,stderr
            process_runs.append({'ranks':world,'length':length or 2*world+1,'processes':rows})
            print('Actual Gloo passed:',world,'ranks,',length or 2*world+1,'positions',flush=True)
        finally:
            for p in processes:
                if p.poll() is None:p.kill();p.wait()
    checks.append({'name':'Actual CPU/Gloo separate-process forward and all three owner gradients match native SDPA/autograd for P1/P2/P3 and fresh uneven L8','passed':True})
    write('distributed-checks.json',{'passed':True,'backend':'gloo','torch':torch.__version__,'threadsPerRank':1,'launch':'Independent child Python processes receive standard RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT environment; USE_LIBUV=0 for this Windows build. Canonical ring routines unchanged.','runs':process_runs,'limits':['Synchronous correctness only; no overlap, NCCL, GPU, multi-node or throughput claim.']})
    ref=module(work/'ring-attention-reference.py');rng=np.random.default_rng(91)
    arrays=[rng.normal(size=(2,7,3)) for _ in range(4)]
    write('constructed-inputs.json',dict(zip(['query','key','value','upstream'],[a.tolist() for a in arrays])))
    fixtures=[]
    for length,ranks,width,vwidth in [(7,3,3,3),(8,3,2,4),(5,1,4,2),(9,4,3,3)]:
        q=rng.normal(size=(2,length,width));k=rng.normal(size=q.shape);v=rng.normal(size=(2,length,vwidth));up=rng.normal(size=v.shape)
        owners=[a.tolist() for a in np.array_split(np.arange(length),ranks)]
        for kind in ['dense','causal','packed','empty']:
            docs=np.arange(length)//3;mask=np.ones((length,length),bool) if kind=='dense' else np.arange(length)[None,:]<=np.arange(length)[:,None]
            if kind in ['packed','empty']:mask &= docs[:,None]==docs[None,:]
            if kind=='empty':mask[1]=False
            ts=[torch.tensor(a,dtype=torch.float64,requires_grad=True) for a in (q,k,v)]
            expected=torch.nn.functional.scaled_dot_product_attention(*ts,attn_mask=torch.tensor(mask),dropout_p=0.)
            grads=torch.autograd.grad((expected*torch.tensor(up)).sum(),ts)
            result,lse,_=ref.ring_attention(q,k,v,[np.array(x) for x in owners],mask)
            gradient=ref.blockwise_backward(q,k,v,[np.array(x) for x in owners],mask,result,lse,up)
            np.testing.assert_allclose(result,expected.detach(),atol=1e-12,rtol=1e-12)
            for a,b in zip(gradient,grads):np.testing.assert_allclose(a,b.detach(),atol=1e-12,rtol=1e-12)
            fixtures.append({'query':q.tolist(),'key':k.tolist(),'value':v.tolist(),'upstream':up.tolist(),'ownership':owners,'allowed':mask.tolist(),'output':expected.detach().tolist(),'gradients':[g.detach().tolist() for g in grads],'mask':kind})
    write('native-fixtures.json',{'attention':fixtures})
    checks.append({'name':'Sixteen new SDPA/autograd cases include unequal key/value widths, dense/causal/packed/empty rows and uneven owners','passed':True})
write('native-checks.json',{'passed':True,'torch':torch.__version__,'numpy':np.__version__,'checks':checks,'threads':1,'noTraining':True})
(OUT/'native-output.json').write_text(json.dumps(log,indent=2)+'\n',encoding='utf-8')
print('Packet replay and native transport verification complete.',flush=True)
