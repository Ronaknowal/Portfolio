"""Execute retained operators and saved fits; never retrain or download weights."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import platform
import re
import shutil
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOPIC = 'sparse-linear-attention-variants'
PACKET = ROOT / 'docs/teaching/drafts' / TOPIC
OUT = ROOT / 'docs/teaching/deep-learning-completion' / TOPIC
PUBLIC = ROOT / 'public/learn-assets' / TOPIC
DOWNLOAD = ROOT / 'public/learn-code' / TOPIC
for folder in (OUT, PUBLIC, DOWNLOAD):
    folder.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), PACKET / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')

study = load('author-calculations')
mechanism = load('mechanism-calculations')
bridge = load('attention_compression_bridges')
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    bridge.main()
    manuscript = (PACKET / 'lesson.md').read_text(encoding='utf-8')
    displayed = re.findall(r'```python\n(.*?)```', manuscript, re.S)[0]
    exec(compile(displayed, 'sparse-displayed-program', 'exec'))

points, _, _, _ = study.load_records()
saved = json.loads((PACKET / 'forecast-models.json').read_text())
fixtures = []
max_saved = max_incremental = 0.
gradient = None
for mode in ('dense', 'window', 'kernel'):
    model = study.AttentionForecaster(mode)
    model.load_state_dict({name: torch.tensor(value) for name, value in saved[mode]['weights'].items()})
    model.eval()
    save(PUBLIC / f'{mode}-forecast.json', {'weights': saved[mode]['weights']})
    with torch.no_grad():
        for length, edit, axis in ((27, -1, 0), (27, 19, 1), (27, 25, 1), (8, 4, 0), (44, 31, 1)):
            current = points[76:77, :length].clone()
            if edit >= 0:
                current[:, edit, axis] *= -1
            prediction, _, trace = model(current, trace=True)
            cache = None
            streamed = []
            for position in range(length):
                step, cache = model(current[:, position:position + 1], positions=torch.tensor([position]), cache=cache)
                streamed.append(step)
            error = float((torch.cat(streamed, 1) - prediction).abs().max())
            max_incremental = max(max_incremental, error)
            assert error < 5e-6
            fixture = {'mode': mode, 'length': length, 'edit': edit, 'axis': axis,
                       'points': ((current[0] + 1) / 2).tolist(),
                       'predictions': ((prediction[0] + 1) / 2).tolist()}
            if mode == 'kernel':
                fixture['matrix'] = trace['prefix_matrix'][0, :, -1].tolist()
                fixture['normalizer'] = trace['prefix_normalizer'][0, :, -1].tolist()
                reference = model(current, reference=True)[0]
                assert float((reference - prediction).abs().max()) < 5e-6
            else:
                fixture['last_weights'] = trace['weights'][0, :, -1].tolist()
            fixtures.append(fixture)
            if length == 27 and edit == -1:
                discrepancy = float((((prediction[0, -1] + 1) / 2) - torch.tensor(saved[mode]['fresh_gated_example']['forecast'])).abs().max())
                max_saved = max(max_saved, discrepancy)
                assert discrepancy < 1e-6
    if mode == 'kernel':
        gradient = study.gradient_check(model, points[76:78, :7])

# Execute the manual mechanisms in evidence storage, retaining original packet bytes.
mechanism.ROOT = OUT
with contextlib.redirect_stdout(captured):
    mechanism.main()
computed = json.loads((OUT / 'mechanism-results.json').read_text())
original = json.loads((PACKET / 'mechanism-results.json').read_text())
assert computed == original
summary = json.loads((PACKET / 'author-results.json').read_text())
randoms = json.loads((PACKET / 'random-feature-results.json').read_text())
display = {name: original[name] for name in ('removed_mass_worked', 'feature_worked', 'projection_worked', 'nystrom_worked', 'random_features_fresh', 'random_features')}
display['trials'] = [{key: trial[key] for key in ('seed', 'features', 'relative_output_l2_error')} for trial in randoms['trials']]
display['sourcePoints'] = ((points[76] + 1) / 2).tolist()
display['workedForecasts'] = [{'label': mode, 'point': saved[mode]['worked_example']['forecast']} for mode in ('dense', 'window', 'kernel')]
display['results'] = {'baselines': summary['baselines'], 'models': {name: {'metrics': value['metrics_rmse_original'], 'selectedUpdate': value['selected_update']} for name, value in summary['models'].items()}}
save(ROOT / 'src/learn/data/sparse-attention-examples.json', display)
save(OUT / 'native-fixtures.json', fixtures)
save(OUT / 'native-checks.json', {'passed': True, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'threads': 2}, 'savedModelCount': 3, 'portFixtures': len(fixtures), 'maximumSavedForecastError': max_saved, 'maximumNativeIncrementalError': max_incremental, 'kernelGradientComparison': gradient, 'checks': ['Displayed NumPy recurrence executed', 'Linformer forward and all five gradients agree with SDPA', 'Gathered-window agrees with dense SDPA oracle', 'All-landmark Nyström agrees with dense attention', 'All original manual fixture values reproduced', 'Frozen models reproduced without fitting; causal kernel reference and incremental execution agree'], 'stdout': captured.getvalue()})
downloads = ['author-calculations.py', 'author-checks.py', 'attention_compression_bridges.py', 'mechanism-calculations.py', 'author-results.json', 'forecast-models.json', 'fresh-controls.json', 'mechanism-results.json', 'random-feature-results.json', 'provenance.md', 'movement_libras.data', 'movement_libras.names']
for name in downloads:
    shutil.copyfile(PACKET / name, DOWNLOAD / name)
print(json.dumps({'passed': True, 'fixtures': len(fixtures), 'maximumNativeIncrementalError': max_incremental, 'downloads': len(downloads)}))
