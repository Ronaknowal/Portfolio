"""Retained vision operators and frozen fits; no training/checkpoint downloads."""
from pathlib import Path
import contextlib
import io
import json
import platform
import re
import runpy
import shutil
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOPIC = 'vision-transformers-vit-deit-swin-dinov2'
PACKET = ROOT/'docs/teaching/drafts'/TOPIC
OUT = ROOT/'docs/teaching/deep-learning-completion'/TOPIC
PUBLIC = ROOT/'public/learn-assets'/TOPIC
DOWNLOAD = ROOT/'public/learn-code'/TOPIC
for folder in (OUT, PUBLIC, DOWNLOAD): folder.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
study = runpy.run_path(str(PACKET/'vision-study.py'))
mechanisms = runpy.run_path(str(PACKET/'vision-mechanisms.py'))
author = runpy.run_path(str(PACKET/'author-checks.py'))
saved = json.loads((PACKET/'vision-models.json').read_text())
prior = json.loads((PACKET/'author-results.json').read_text())
expected = json.loads((PACKET/'study-results.json').read_text())
data, _ = study['load_data']()
models = study['reload_models']()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    mechanisms['main'].__globals__['ROOT'] = OUT
    mechanisms['main']()
    displayed = re.findall(r'```python\n(.*?)```', (PACKET/'lesson.md').read_text(encoding='utf-8'), re.S)[0]
    exec(compile(displayed, 'vision-displayed-patch-program', 'exec'))
mechanism_results = json.loads((OUT/'mechanism-results.json').read_text())
assert mechanism_results == json.loads((PACKET/'mechanism-results.json').read_text())
fixtures = []
numpy_error = 0.
with torch.no_grad():
    for name, model in models.items():
        actual = study['evaluate'](model, *data['test'])
        assert actual['predictions'] == expected[name]['metrics']['test']['predictions']
    model = models['vit']
    for source in range(3):
        for condition in ('original', 'pixel24', 'pixel23', 'content_swap', 'joint_swap'):
            image = data['test'][0][source:source+1].clone()
            order = torch.arange(16)
            if condition.startswith('pixel'):
                column = int(condition[-1]); image[0,0,2,column] = 1-image[0,0,2,column]
            if condition.endswith('swap'): order[5], order[6] = 6, 5
            features, attention = model.encode(image, order, condition == 'joint_swap', capture=True)
            logits = model.head(features[:,0])
            fixtures.append({'source': source+1, 'condition': condition, 'image': image[0,0].tolist(), 'order': order.tolist(), 'movePositions': condition == 'joint_swap', 'logits': logits[0].tolist(), 'probabilities': logits.softmax(-1)[0].tolist(), 'features': features[0].tolist(), 'attentionRows': [[head[[0,6]].tolist() for head in layer[0]] for layer in attention]})
            if condition == 'original':
                independent = author['numpy_vit'](image.numpy(), saved['vit'])[0]
                numpy_error = max(numpy_error, float(np.abs(independent-logits.numpy()).max()))
                assert numpy_error < 1e-4
    # A fresh arbitrary input, deliberately different from published edit presets.
    image = data['test'][0][2:3].clone()
    image[0,0,6,1] = .3; image[0,0,1,5] = .9
    features, attention = model.encode(image, capture=True)
    logits = model.head(features[:,0])
    fixtures.append({'source':3,'condition':'two_arbitrary_pixels','image':image[0,0].tolist(),'order':list(range(16)),'movePositions':False,'logits':logits[0].tolist(),'probabilities':logits.softmax(-1)[0].tolist(),'features':features[0].tolist(),'attentionRows':[[head[[0,6]].tolist() for head in layer[0]] for layer in attention]})

def save(path,value): path.write_text(json.dumps(value,separators=(',',':')),encoding='utf-8')
save(PUBLIC/'plain-vit.json', {'state':saved['vit'],'pca':prior['pca']})
with torch.no_grad():
    model = models['vit']; image = data['test'][0][:1]
    tokens = torch.cat([(model.cls + model.cls_position), model.patch(image).flatten(2).transpose(1,2) + model.patch_position], dim=1)
    tokens = model.blocks[0](tokens)
    qkv = model.blocks[1].qkv(model.blocks[1].norm1(tokens)).reshape(1,17,3,4,8).permute(2,0,3,1,4)
    q,k,v = qkv.unbind(0)
    weights = (q @ k.transpose(-1,-2) / 8**.5).softmax(-1)
    read = {'weights':weights[0,0,0].tolist(),'values':v[0,0,:,0].tolist(),'products':(weights[0,0,0]*v[0,0,:,0]).tolist(),'output':float((weights @ v)[0,0,0,0])}
display = {'mechanisms':mechanism_results, 'clsRead':read, 'examples':[{key:e[key] for key in ('source_id','label','image','logits','probabilities','patch_pca','last_head0_cls_attention')} for e in prior['examples']], 'pca':{key:prior['pca'][key] for key in ('training_min','training_max','explained_variance_ratio')}, 'matching':prior['patch_matching'], 'study':{name:{key:expected[name][key] for key in ('chosen_step','parameters','trace')} for name in ('cnn','vit','distilled')}}
save(ROOT/'src/learn/data/vision-transformer-examples.json', display)
save(OUT/'native-fixtures.json',fixtures)
save(OUT/'native-checks.json',{'passed':True,'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'threads':2},'fixtures':len(fixtures),'allSavedTestPredictionsReproduced':True,'maximumIndependentNumpyError':numpy_error,'checks':['All three saved models reproduce all1797 test argmaxes','Complete constructed mechanism file reproduced exactly, including padded Swin reference and bias gradients','Displayed patch/Conv2d code executed','15 original/edit/permutation fixtures and one arbitrary two-pixel probe exported','No fit was repeated'],'stdout':captured.getvalue(),'optionalCheckpoint':'Not executed: local DINOv2 checkpoint and images not supplied'})
downloads=['vision-study.py','vision-mechanisms.py','vision_library_bridge.py','author-checks.py','inspect-pretrained-features.py','vision-models.json','study-results.json','author-results.json','mechanism-results.json','optdigits.tra','optdigits.tes','optdigits.names','provenance.md','experiment-contract.md']
for file in downloads: shutil.copyfile(PACKET/file,DOWNLOAD/file)
print(json.dumps({'passed':True,'fixtures':len(fixtures),'numpyError':numpy_error,'downloads':len(downloads)}))
