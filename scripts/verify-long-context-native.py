"""Run canonical scratch/library operators and verify saved fitted states without tuning."""
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ID = 'long-context-sequence-models-transformer-xl-griffin-perceiver'
SOURCE = ROOT / 'public/learn-code' / ID
EVIDENCE = ROOT / 'docs/teaching/evidence/long-context'
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / 'native-checks.json').write_text(json.dumps({'status': 'incomplete', 'note': 'A run started; only a subsequent passed report is successful evidence.'}, indent=2) + '\n')
torch.set_num_threads(2)
bridge = runpy.run_path(str(SOURCE / 'memory_library_bridge.py'))
bridge['check_segmented_attention']()
bridge['check_latent_read']()

scratch = runpy.run_path(str(SOURCE / 'sequence_mechanisms.py'))
for length in [1, 3, 11]:
    keys = np.arange(length, dtype=float)[:, None] / max(1, length)
    values = np.cos(keys)
    for segment in [1, 4]:
        for memory in [0, 2, 16]:
            result = scratch['segmented_attention'](keys, values, segment, memory)
            assert len(result['outputs']) == length
            for trace in result['segments']:
                for query, weights in zip(trace['queries'], trace['weights']):
                    assert all(w == 0 for position, w in zip(trace['keys'], weights) if position > query)
                    np.testing.assert_allclose(sum(weights), 1, atol=1e-12)

namespace = runpy.run_path(str(SOURCE / 'latent_trajectory_classifier.py'))
data, frames, valid, pooled, ordered, fit, validation, test = namespace['prepare'](SOURCE)
assert (len(fit), len(validation), len(test)) == (220, 50, 60)
assert len(set(fit) | set(validation) | set(test)) == 330
saved = np.load(SOURCE / 'small-fits.npz', allow_pickle=False)
max_error = 0.0
max_null = 0.0
cases = 0
for count in [1, 4]:
    for seed in [11, 29]:
        name = f'latents{count}_seed{seed}'
        model = namespace['LatentClassifier'](count)
        model.load_state_dict({key.split('::')[1]: torch.from_numpy(saved[key]) for key in saved.files
                               if key.startswith(name+'::') and key.split('::')[1] in model.state_dict()})
        model.eval()
        with torch.no_grad():
            logits, weights = model(frames, valid)
            error = float(np.max(np.abs(logits.numpy()-saved[name+'::logits'])))
            max_error = max(max_error, error)
            assert error < 1e-5
            reversed_logits, _ = model(frames.flip(1), valid.flip(1))
            padded, _ = model(torch.cat([frames, torch.full((360,5,3),1000.)],1),
                              torch.cat([valid,torch.zeros((360,5),dtype=torch.bool)],1))
            null_error = max(float((logits-reversed_logits).abs().max()), float((logits-padded).abs().max()))
            max_null = max(max_null, null_error)
            assert null_error < 1e-4
            if name == 'latents4_seed29':
                mask = torch.zeros((1,45),dtype=torch.bool); mask[0,0] = True
                point_only, _ = model(frames[6:7],mask)
                assert int(logits[6].argmax())+1 == 1 and int(point_only.argmax())+1 == 10
        cases += 360

# Explicit reset reference: later documents must not depend on the earlier stream.
torch.manual_seed(21)
inputs = torch.randn(2,7,8,requires_grad=True)
positions = torch.tensor([[0,1,2,0,1,2,3],[0,1,2,3,4,5,6]])
parameters = [torch.randn(2,4,4)*.1,torch.zeros(2,4),torch.randn(2,4,4)*.1,torch.zeros(2,4),torch.zeros(8)]
whole, _ = bridge['rglru_reference'](inputs,positions,*parameters)
changed = inputs.clone(); changed[0,:3] += 2
other, _ = bridge['rglru_reference'](changed,positions,*parameters)
torch.testing.assert_close(whole[0,3:],other[0,3:])
grad = torch.autograd.grad(whole[0,3:].square().sum(),inputs)[0]
torch.testing.assert_close(grad[0,:3],torch.zeros_like(grad[0,:3]))

files = [SOURCE/name for name in ['sequence_mechanisms.py','memory_library_bridge.py',
                                 'latent_trajectory_classifier.py','movement_libras.data','small-fits.npz']]
report = {'status':'passed','versions':{'numpy':np.__version__,'torch':torch.__version__},
          'frozenRowModelCases':cases,'maxSavedLogitError':max_error,'maxPermutationPaddingError':max_null,
          'checks':['trailing-segment scratch/SDPA parity','latent projected output and all gradient parity',
                    'cache masks and short lengths','all four saved models on all 360 source rows',
                    'paired record and masked padding invariance','first-point contrast','RG-LRU reset and upstream gradient isolation'],
          'optionalRecurrentGemma':'not executed; explicit optional package boundary retained',
          'training':'retained original selected fits reused; no new tuning or retraining',
          'reviewedFiles':{str(file.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
EVIDENCE.mkdir(parents=True,exist_ok=True)
(EVIDENCE/'native-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
