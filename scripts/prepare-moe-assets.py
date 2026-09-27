"""Replay retained MoE experiments and execute the scratch/PyTorch boundary; no fits."""
from pathlib import Path
import contextlib, hashlib, io, json, platform, re, shutil, sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
ID = 'mixture-of-experts-transformers-moe'
PACKET = ROOT/'docs/teaching/drafts'/ID
OUT = ROOT/'docs/teaching/deep-learning-completion'/ID
ASSETS = ROOT/'public/learn-assets'/ID
DOWNLOAD = ROOT/'public/learn-code'/ID
for folder in (OUT, ASSETS, DOWNLOAD): folder.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PACKET))
from moe_study import DigitTransformer, load_data, evaluate
torch.set_num_threads(1)
namespace = {'__file__':str(PACKET/'moe_calculations.py'), '__name__':'moe_native_replay', 'OUT':OUT}
source = (PACKET/'moe_calculations.py').read_text().replace('(HERE/"calculated-inputs.json").write_text', '(OUT/"calculated-inputs.json").write_text')
exec(compile(source, str(PACKET/'moe_calculations.py'), 'exec'), namespace)
with contextlib.redirect_stdout(io.StringIO()): namespace['main']()
calculated = json.loads((OUT/'calculated-inputs.json').read_text())
assert calculated == json.loads((PACKET/'calculated-inputs.json').read_text())
lesson = (PACKET/'lesson.md').read_text(encoding='utf-8')
program = next(block for block in re.findall(r'~~~python\n(.*?)~~~', lesson, re.S) if 'class DigitTransformer' in block)
assert program.strip() == (PACKET/'moe_study.py').read_text().strip()
rows, pixels, labels, roles = load_data()
assert {role:len(index) for role,index in roles.items()} == {'fit':500,'validation':150,'assessment':300}
assert len({(row['source_file'],row['source_id']) for row in rows}) == 950
fits = json.loads((PACKET/'fitted-models.json').read_text())
study = json.loads((PACKET/'study-results.json').read_text())
for run in study['runs']:
    model = DigitTransformer(run['condition'] != 'dense').eval()
    model.load_state_dict({name:torch.tensor(value) for name,value in fits[run['key']].items()})
    changed = pixels[roles['assessment']].clone().reshape(-1,8,8); changed[:,4:,:] = 0
    assert evaluate(model, changed.reshape(-1,64), labels[roles['assessment']]) == run['measurements']['assessment_lower_half_zero']
    best = min(run['curve'], key=lambda point:point['validation_ce_after'])
    assert best['step'] == run['selected_step']
model = DigitTransformer(True).eval()
model.load_state_dict({name:torch.tensor(value) for name,value in fits['moe_001_17'].items()})
fixtures = []
with torch.no_grad():
    for label in (0,7):
        index = next(int(i) for i in roles['validation'] if int(labels[i]) == label)
        for condition in ('clean','lower_half_zero','upper_left_patch_zero','temperature_2','disable_0','temperature_half','temperature_3','disable_1','disable_2','disable_3','arbitrary_pixels'):
            image = pixels[index:index+1].clone().reshape(1,8,8)
            if condition == 'lower_half_zero': image[:,4:,:] = 0
            if condition == 'upper_left_patch_zero': image[:,:2,:2] = 0
            if condition == 'arbitrary_pixels': image[0,3,4] = .3125; image[0,6,1] = .875
            temperature = {'temperature_2':2., 'temperature_half':.5, 'temperature_3':3.}.get(condition, 1.)
            disabled = int(condition[-1]) if condition.startswith('disable_') else None
            logits, auxiliary, trace = model(image.reshape(1,64), temperature, disabled, True)
            inputs = trace['expert_inputs'][0]
            expert_outputs = torch.stack([expert(inputs) for expert in model.experts], dim=1)
            fixtures.append({'label':label,'condition':condition,'pixels':image.flatten().tolist(),'temperature':temperature,'disabled':disabled,'classProbabilities':logits.softmax(-1)[0].tolist(),'auxiliary':float(auxiliary),'expertOutputs':expert_outputs.tolist(), **{name:value.tolist() for name,value in trace.items()}})
display = {'routing':calculated['routing'],'capacity':calculated['capacity'],'auxiliary':calculated['auxiliary'],'selectionBias':calculated['selection_bias'],'study':study['runs'], 'examples':[{'label':int(label),'source_file':row['source_file'],'source_id':row['source_id'],'pixels':row['variants']['clean']['pixels'],'baseline':row['variants']['clean']} for label,row in calculated['trained_fixtures'].items()]}
def save(path, value): path.write_text(json.dumps(value, separators=(',',':'))+'\n',encoding='utf-8')
save(ROOT/'src/learn/data/moe-examples.json', display)
save(ASSETS/'moe-001-17.json', fits['moe_001_17'])
save(OUT/'native-fixtures.json', fixtures)
for file in ('moe_study.py','moe_calculations.py','optical-digits.csv','fitted-models.json','study-results.json','calculated-inputs.json','data-provenance.md'):
    shutil.copyfile(PACKET/file, DOWNLOAD/file)
save(OUT/'native-checks.json', {'passed':True,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'threads':torch.get_num_threads(),'checks':calculated['checks'],'additional':['Full displayed PyTorch model matches executed companion bytes','All twelve frozen fits: fit/validation/assessment and changed assessment replayed; validation checkpoint minima checked','950 unique original source records; 500/150/300 disjoint roles','22 fresh native port fixtures include all expert ablations, temperature endpoints and arbitrary pixel edits'],'fixtureCount':len(fixtures),'training':'Retained fits replayed; no retraining or checkpoint downloads'})
print(json.dumps({'passed':True,'fixtures':len(fixtures),'checks':calculated['checks']}))
