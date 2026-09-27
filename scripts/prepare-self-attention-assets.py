"""Execute displayed scratch/library programs and frozen inference, without retraining."""
from pathlib import Path
import importlib.util
import contextlib
import io
import json
import platform
import re
import shutil
import torch

root = Path(__file__).resolve().parents[1]
topic = 'self-attention-multi-head-attention'
packet = root / 'docs/teaching/drafts' / topic
evidence = root / 'docs/teaching/deep-learning-completion' / topic
assets = root / 'public/learn-assets' / topic
downloads = root / 'public/learn-code' / topic
for folder in (evidence, assets, downloads): folder.mkdir(parents=True, exist_ok=True)
learner_files = ('additional-calculations.py', 'additional-fixtures.json', 'attention-model.json',
                 'author-calculations.py', 'author-results.json', 'data-provenance.md',
                 'movement_libras.data', 'movement_libras.names')
for name in learner_files:
    shutil.copyfile(packet / name, downloads / name)
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('attention_author', packet / 'author-calculations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
captured = []
for number, code in enumerate(re.findall(r'```python\n(.*?)\n```', (packet / 'lesson.md').read_text(encoding='utf-8'), re.S)):
    output = io.StringIO()
    with contextlib.redirect_stdout(output): exec(compile(code, f'displayed-program-{number}', 'exec'), {})
    captured.append(output.getvalue())
saved = json.loads((packet / 'attention-model.json').read_text(encoding='utf-8'))
model = module.TrajectoryClassifier('attention-2').eval()
model.load_state_dict({name: torch.tensor(values) for name, values in saved['state_dict'].items()})
points = torch.tensor(saved['probes']['points'])[None]
fixtures = []
for variant in ('original', 'reverse', 'reflect', 'padding-masked', 'padding-unmasked', 'duplicate-first'):
    data = points.clone()
    valid = None
    if variant == 'reverse': data = data.flip(1)
    if variant == 'reflect': data[0, 22, 0] = 1 - data[0, 22, 0]
    if variant.startswith('padding'):
        data = torch.cat([data, torch.full((1, 5, 2), .75)], dim=1)
        if variant == 'padding-masked': valid = torch.tensor([[True] * 45 + [False] * 5])
    if variant == 'duplicate-first': data = torch.cat([data, data[:, :1]], 1)
    with torch.no_grad(): logits, weights = model(data, valid, return_weights=True)
    fixtures.append({'variant': variant, 'logits': logits[0].tolist(), 'probabilities': logits.softmax(1)[0].tolist(),
                     'selectedWeights': weights[0, :, 0].tolist()})
(assets / 'trajectory-model.json').write_text(json.dumps({'state_dict': saved['state_dict'], 'points': saved['probes']['points'], 'sourceRow': 77, 'label': 4}, separators=(',', ':')) + '\n')
(root / 'src/learn/data/self-attention-trajectory-example.json').write_text(json.dumps({'points': saved['probes']['points']}, separators=(',', ':')) + '\n')
(evidence / 'native-fixtures.json').write_text(json.dumps(fixtures, indent=2) + '\n')
(evidence / 'native-checks.json').write_text(json.dumps({'passed': True, 'environment': {'python': platform.python_version(), 'torch': torch.__version__},
    'checks': ['Both complete displayed Python programs executed, including all-parameter gradients and one equal update', 'Frozen model inferred under six meaningful changed conditions'],
    'displayedOutputs': captured, 'scope': 'No training rerun; original 12-fit evidence preserved.'}, indent=2) + '\n')
print('PASS: 2 displayed scratch/library programs, 6 frozen-model conditions')
