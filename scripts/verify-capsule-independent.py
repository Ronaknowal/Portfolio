"""Independent implementation review: changed shapes and held-out input interventions."""
from pathlib import Path
import hashlib
import json
import runpy
import torch

root = Path(__file__).resolve().parents[1]
assets = root/'public/learn-assets/capsule-networks'
learning = runpy.run_path(str(assets/'capsule-learning.py'))
torch.set_num_threads(1)
torch.manual_seed(127)
def reference(votes, rounds, temperature):
    logits = torch.zeros(votes.shape[:-1], dtype=votes.dtype)
    trace = []
    for step in range(rounds):
        probability = torch.softmax(logits/temperature, dim=1)
        total = torch.einsum('ij,ijd->jd', probability, votes)
        length = torch.linalg.vector_norm(total, dim=1)
        output = total * (length/(1+length.square()))[:, None]
        agreement = torch.einsum('ijd,jd->ij', votes, output)
        trace.append({'coupling': probability.tolist(), 'sums': total.tolist(), 'outputs': output.tolist()})
        logits = logits + agreement
    return trace
fixtures=[]
for parents in [1, 2, 4]:
    for dimensions in [2, 5]:
        votes = torch.randn(4,parents,dimensions,dtype=torch.float64)
        for rounds in [1, 3, 8]:
            for temperature in [.31, 1.7]:
                fixtures.append({'votes': votes.tolist(), 'rounds': rounds, 'temperature': temperature, 'trace': reference(votes,rounds,temperature)})
saved=json.loads((assets/'calculated-inputs.json').read_text())
model=learning['TinyCapsules']().double().eval()
model.load_state_dict({name:torch.tensor(value,dtype=torch.float64) for name,value in saved['saved_models']['3'].items()})
images=[torch.full((1,1,8,8),.37,dtype=torch.float64), (torch.arange(64).reshape(1,1,8,8)%2).double(), torch.linspace(0,1,64,dtype=torch.float64).reshape(1,1,8,8)]
frozen=[]
with torch.no_grad():
    for image in images:
        capsules,_=model.encode(image)
        scores=capsules.norm(dim=-1)
        predicted=scores.argmax(1)
        frozen.append({'image':image.flatten().tolist(),'capsules':capsules[0].tolist(),'predicted':int(predicted[0]),'reconstruction':model.reconstruct(capsules,predicted).flatten().tolist()})
paths=[assets/'capsule-learning.py',assets/'calculated-inputs.json',root/'scripts/verify-capsule-independent.py']
report={'passed':True,'torch':torch.__version__,'fixtures':fixtures,'frozen':frozen,'sourceHashes':{str(path.relative_to(root)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},'limits':'Constructed reviewer interventions on fixed weights; no additional empirical training outcome.'}
(root/'docs/teaching/evidence/capsule-independent-native.json').write_text(json.dumps(report,separators=(',',':'))+'\n',encoding='utf8')
print(json.dumps({'passed':True,'routingCases':len(fixtures),'frozenCases':len(frozen)}))
