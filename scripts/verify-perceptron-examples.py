"""Execute displayed programs and generate a dense, independent PyTorch oracle.

Run with scratch/lesson-tools/Scripts/python.exe from the repository root.
--dense-only reuses the already executed source-bound native outputs.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
import torch
from torch.nn import functional as F

root=Path.cwd()
parser=argparse.ArgumentParser();parser.add_argument('--dense-only',action='store_true');args=parser.parse_args()
torch.set_num_threads(1)
evidence=root/'docs/teaching/evidence'
if not args.dense_only:
    source='import {perceptronExamples} from "./src/learn/data/perceptron-examples.js";console.log(JSON.stringify(perceptronExamples))'
    examples=json.loads(subprocess.check_output(['node','--input-type=module','-e',source],text=True,encoding='utf-8'))
    directory=root/'scratch/deep-learning-core-implementation/perceptron-native';directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'public/learn-assets/perceptrons/digits-400.csv',directory/'digits-400.csv')
    checks=[]
    for example in examples:
        path=directory/example['file'];path.write_text(example['code']+'\n',encoding='utf-8')
        run=subprocess.run([sys.executable,str(path)],capture_output=True,text=True,check=True)
        assert run.stdout.strip()==example['expected'],example['file']
        checks.append({'file':example['file'],'stdout':run.stdout.strip(),'codeSha256':hashlib.sha256(example['code'].encode()).hexdigest()})
    (evidence/'perceptron-native.json').write_text(json.dumps({'status':'passed','runtime':sys.executable,'programs':checks},indent=2)+'\n')
z=torch.linspace(-6,6,481,dtype=torch.float64,requires_grad=True)
functions={'sigmoid':torch.sigmoid,'tanh':torch.tanh,'relu':F.relu,'leaky_relu':lambda x:F.leaky_relu(x,.1),'gelu':lambda x:F.gelu(x,approximate='none'),'gelu_tanh':lambda x:F.gelu(x,approximate='tanh'),'silu':F.silu,'elu':F.elu,'mish':F.mish}
oracle={'torch':torch.__version__,'z':z.detach().tolist(),'functions':{}}
for name,function in functions.items():
    values=function(z);derivatives=torch.autograd.grad(values.sum(),z)[0]
    oracle['functions'][name]={'values':values.detach().tolist(),'slopes':derivatives.tolist()}
(evidence/'perceptron-activation-oracle.json').write_text(json.dumps(oracle,separators=(',',':'))+'\n')
print('Dense oracle:',len(functions),'activations ×',len(z),'values and slopes; displayed programs', 'reused' if args.dense_only else 'passed')
