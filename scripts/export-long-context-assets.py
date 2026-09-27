"""Lossless, deterministic export of the retained lesson fits; never retrains them."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ID = 'long-context-sequence-models-transformer-xl-griffin-perceiver'
SOURCE = ROOT / 'docs/teaching/drafts' / ID
DEST = ROOT / 'public/learn-code' / ID
DEST.mkdir(parents=True, exist_ok=True)
for name in ['sequence_mechanisms.py', 'latent_trajectory_classifier.py', 'memory_library_bridge.py',
             'movement_libras.data', 'movement_libras.names', 'data-provenance.md',
             'checked-results.json', 'trajectory-results.json', 'small-fits.npz', 'investigation-checks.json']:
    shutil.copyfile(SOURCE / name, DEST / name)

# Preserve the research protocol without publishing its obsolete phase-two to-do.
provenance = (SOURCE / 'data-provenance.md').read_text(encoding='utf-8')
provenance = provenance.replace(
    'Source files and generated retained arrays are necessary for the pending content-first handoff. Runtime export, licensing presentation, exact forward parity, desktop/mobile/accessibility/performance checks and independent lesson review remain phase two.',
    'The lesson retains these source files and fixed fitted arrays so its results can be reproduced. The interactive workbench exposes only the 50 predeclared validation trajectories and performs inference with four frozen models; it does not train or tune models in the browser.'
)
(DEST / 'data-provenance.md').write_text(provenance, encoding='utf-8')

with np.load(SOURCE / 'small-fits.npz', allow_pickle=False) as stored:
    report = json.loads((SOURCE / 'trajectory-results.json').read_text())
    raw = np.loadtxt(SOURCE / 'movement_libras.data', delimiter=',')
    validation = report['data_roles']['validation_source_rows']
    payload = {
        'attribution': 'Libras Movement: Dias, Peres and Biscaro (2009), UCI DOI 10.24432/C5GC82, CC BY 4.0.',
        'dataSha256': hashlib.sha256((SOURCE / 'movement_libras.data').read_bytes()).hexdigest(),
        'role': 'validation',
        'specimens': [{'sourceRow': i, 'label': int(raw[i-1, -1]), 'records': [
            {'id': point + 1, 'x': float(raw[i-1, 2*point]), 'y': float(raw[i-1, 2*point+1]),
             'position': float(np.linspace(-1, 1, 45, dtype=np.float32)[point]), 'valid': True}
            for point in range(45)]} for i in validation],
        'models': {},
        'mean': {'coef': stored['mean_coef'].tolist(), 'intercept': stored['mean_intercept'].tolist()},
    }
    for count in [1, 4]:
        for seed in [11, 29]:
            key = f'latents{count}_seed{seed}'
            payload['models'][key] = {name.split('::')[1]: stored[name].tolist()
                                     for name in stored.files if name.startswith(key+'::')
                                     and not name.endswith(('::logits', '::read_weights_first_validation'))}
    (DEST / 'trajectory-models.json').write_text(json.dumps(payload, separators=(',', ':'))+'\n', encoding='utf-8')
    # Verification oracle stays author-only, outside the deployed public tree.
    oracle = {key: stored[key+'::logits'][np.array(validation)-1].tolist() for key in payload['models']}
    evidence = ROOT / 'docs/teaching/evidence/long-context'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'native-validation-logits.json').write_text(json.dumps(oracle)+'\n', encoding='utf-8')
print(json.dumps({'validationTrajectories': len(validation), 'models': len(payload['models']),
                  'runtimeBytes': (DEST / 'trajectory-models.json').stat().st_size}))
