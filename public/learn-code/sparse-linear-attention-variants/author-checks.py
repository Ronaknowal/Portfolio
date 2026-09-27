"""Bounded saved-weight and displayed-program checks; does not retrain."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location('attention_study', ROOT/'author-calculations.py')
study = importlib.util.module_from_spec(specification)
specification.loader.exec_module(study)
torch.set_num_threads(1)
points, _, _, _ = study.load_records()
assets = json.loads((ROOT/'forecast-models.json').read_text(encoding='utf-8'))
checks = {}
with torch.no_grad():
    for mode in ('dense', 'window', 'kernel'):
        model = study.AttentionForecaster(mode)
        model.load_state_dict({name: torch.tensor(value) for name, value in assets[mode]['weights'].items()})
        model.eval()
        prefix = points[76:77, :27]
        original = model(prefix)[0]
        changed = prefix.clone()
        changed[:, 25, 1] *= -1
        modified = model(changed)[0]
        checks[mode] = {
            'source_row': 77, 'prefix_length': 27, 'near_edit_position': 25, 'coordinate': 1,
            'baseline_forecast': ((original[0, -1]+1)/2).tolist(),
            'near_edited_forecast': ((modified[0, -1]+1)/2).tolist(),
            'strict_earlier_error': float((original[:, :25]-modified[:, :25]).abs().max()),
            'reload_saved_error': float((((original[0, -1]+1)/2)-torch.tensor(
                assets[mode]['fresh_gated_example']['forecast'])).abs().max())}
        assert checks[mode]['strict_earlier_error'] == checks[mode]['reload_saved_error'] == 0
        assert (original[0, -1]-modified[0, -1]).abs().max() > 1e-5
manual = json.loads((ROOT/'mechanism-results.json').read_text(encoding='utf-8'))
random_case = manual['random_features_fresh']
baseline = np.asarray(random_case['m8'])
changed = np.asarray(random_case['changed_m8'])
assert np.array_equal(baseline[0], changed[0])
assert np.max(np.abs(baseline[-1]-changed[-1])) > 1e-5
assert np.allclose(random_case['constant_value_null'], np.tile([2., -1.], (4, 1)))
reference = np.asarray(random_case['reference_output'])
random_errors = {name: float(np.linalg.norm(np.asarray(random_case[name])-reference)/np.linalg.norm(reference))
                 for name in ('m8', 'm64')}
sparse_weights = np.asarray(manual['removed_mass_fresh']['weights'])
constant_values = np.tile([2., -1.], (4, 1))
keep = [0, 2, 3]
assert np.allclose(sparse_weights @ constant_values,
                   sparse_weights[keep] @ constant_values[keep]/sparse_weights[keep].sum())
coefficients = np.array([.25, .5, 0., .25])
values = np.array([4., -2., 3., 8.])
inside = values.copy(); inside[1] = 6.
zeroed = coefficients.copy(); zeroed[3] = 0.
future = values.copy(); future[3] = -4.
assert zeroed @ values == zeroed @ future
assert coefficients[:2] @ inside[:2] != coefficients[:2] @ values[:2]
checks['manual_controls'] = {
    'random_feature_errors': random_errors,
    'random_first_query_change': float(np.max(np.abs(baseline[0]-changed[0]))),
    'random_last_query_change': float(np.max(np.abs(baseline[-1]-changed[-1]))),
    'sparse_constant_output': (sparse_weights @ constant_values).tolist(),
    'projection_zero_coefficient_output': float(zeroed @ values),
    'projection_inside_prefix_before': float(coefficients[:2] @ values[:2]),
    'projection_inside_prefix_after': float(coefficients[:2] @ inside[:2])}
(ROOT/'fresh-controls.json').write_text(json.dumps(checks, separators=(',', ':')), encoding='utf-8')
manuscript = (ROOT/'lesson.md').read_text(encoding='utf-8')
displayed_program = manuscript.split('```python\n')[1].split('```')[0]
exec(compile(displayed_program, 'lesson-displayed-example', 'exec'))
print('PASS: saved-model fresh and near controls; sparse/kernel/random/projection contrasts and nulls; displayed NumPy program executed.')
