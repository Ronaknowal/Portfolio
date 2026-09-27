"""Expected-failure probe: omit only the final gradient return rotation."""
from pathlib import Path
import json,os,socket,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];ID='ring-attention-sequence-parallelism'
source=(ROOT/'docs/teaching/drafts'/ID/'distributed_ring.py').read_text(encoding='utf-8')
needle='        packet = rotate(packet)\n    return query_gradient'
assert source.count(needle)==1
broken=source.replace(needle,'        if step+1 < world:\n            packet = rotate(packet)\n    return query_gradient')
processes=[];records=[]
with tempfile.TemporaryDirectory(prefix='ring-gradient-return-',dir=ROOT/'scratch') as directory:
    path=Path(directory)/'missing_return.py';path.write_text(broken,encoding='utf-8')
    with socket.socket() as server:server.bind(('127.0.0.1',0));port=server.getsockname()[1]
    try:
        for rank in range(3):
            env={**os.environ,'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port),'WORLD_SIZE':'3','RANK':str(rank),'LOCAL_RANK':str(rank),'USE_LIBUV':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            processes.append(subprocess.Popen([sys.executable,str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        for rank,p in enumerate(processes):
            out,err=p.communicate(timeout=90);assert p.returncode!=0 and 'Tensor-likes are not close' in err,(out,err)
            records.append({'rank':rank,'expectedFailure':True,'returncode':p.returncode,'stdout':out,'stderr':err})
    finally:
        for p in processes:
            if p.poll() is None:p.kill();p.wait()
(ROOT/'docs/teaching/deep-learning-completion'/ID/'gradient-return-checks.json').write_text(json.dumps({'passed':True,'claim':'All three actual process validators reject omitted final backward rotation after forward parity; canonical program unchanged.','runs':records},indent=2)+'\n',encoding='utf-8')
print('All three actual Gloo processes rejected the missing final gradient return as expected.')
