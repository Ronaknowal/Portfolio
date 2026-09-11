"""Execute complete SVM programs and save their actual output plus bounded fitted visual data."""
import contextlib
import hashlib
import io
import json
import platform
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / 'scripts/support-vector-machines-programs'
OUT = ROOT / 'scratch/support-vector-machines-verification'
OUT.mkdir(parents=True, exist_ok=True)
specifications = [
    ('projection', 'margin-projection', 'The score changes; the perpendicular distance does not', 'If every weight and the bias are multiplied by three, which printed quantities should change?'),
    ('support', 'support-certificates', 'Check a moved point and a duplicate coefficient family', 'Which move invalidates the original margin constraint, and can two different alpha vectors describe the same classifier?'),
    ('soft', 'soft-margin-bias', 'Calculate the loss, dual value and admissible bias', 'Can all observations be support vectors in both a perfectly separated fit and a conflicting-label fit?'),
    ('kernels', 'kernel-xor', 'Compare explicit features with kernel scores', 'Which kernel can distinguish the XOR labels, and does a zero Gram eigenvalue mean the kernel is invalid?'),
    ('pair', 'pair-move', 'Move two coefficients while preserving label balance', 'Why does a same-label pair follow a different diagonal, and what should happen when pair curvature is zero?'),
    ('solver', 'pair-coordinate-solver', 'Fit the original Gaussian and XOR examples with a checked gap', 'Will the solver pass its objective-gap criterion on both datasets and on opposite labels at an identical input?'),
    ('selection', 'moons-validation', 'Select C and gamma using observations outside training', 'Will the model with the highest training accuracy also be the first winner on validation data?'),
    ('classification', 'heldout-classification', 'Fit and compare complete classification pipelines', 'Where is each scaler fitted during cross-validation, and how does the selected model compare with a majority baseline?'),
    ('calibration', 'calibrated-pipeline', 'Calibrate the entire pipeline on out-of-fold scores', 'Which observations fit each scaler and classifier, and what does a held-out Brier score establish?'),
    ('multiclass', 'multiclass-and-gram', 'Inspect multiclass scores and precomputed kernel shapes', 'How many binary models are trained for four classes, and does switching output presentation retrain them?'),
    ('tube', 'regression-tube-units', 'Convert the regression target units without changing the fit', 'If targets are divided by ten, how must C and epsilon change for the predictions to agree after converting back?'),
    ('regression', 'heldout-regression', 'Fit a regression pipeline in training-estimated target units', 'What is the epsilon tolerance in original target units, and how does held-out error compare with predicting the training mean?'),
    ('sequence', 'sequence-kernel', 'Turn overlapping sequence pieces into a valid kernel', 'Can two different sequences become indistinguishable after counting their two-character pieces?'),
    ('approximation', 'kernel-approximations', 'Compare finite kernel features with the original kernel', 'Does a smaller Gram approximation error necessarily imply a better held-out classification result in this one run?')
]
examples = {}
records = []
moons = None
for key, stem, title, question in specifications:
    source = PROGRAMS / f'{stem}.py'
    code = source.read_text(encoding='utf8').strip()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        namespace = runpy.run_path(str(source), run_name='__main__')
    output = stream.getvalue().strip()
    if not output:
        raise AssertionError(f'No output from {stem}')
    examples[key] = {'title': title, 'question': question, 'code': code, 'expected': output,
                     'language': 'python'}
    records.append({'key': key, 'path': str(source.relative_to(ROOT)).replace('\\', '/'),
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'stdout': output})
    if key == 'selection':
        moons = namespace
    print(f'{key}: passed')

data = {
    'provenance': {'program': 'scripts/support-vector-machines-programs/moons-validation.py',
                   'sklearn': sklearn.__version__, 'generator': 'make_moons(200,noise=.2,random_state=42)',
                   'split': 'test20% stratified seed17; validation25% of remaining seed19; scaler fits120 training rows only',
                   'boundary': 'Browser evaluates saved support coefficients in standardized coordinates; field colors are a finite sampled rendering, not an exact contour proof.'},
    'scaler': {'mean': moons['scaler'].mean_.tolist(), 'scale': moons['scaler'].scale_.tolist()},
    'train': [{'id': int(index), 'x': moons['x'][index].tolist(), 'y': int(moons['y'][index])} for index in moons['train']],
    'validation': [{'id': int(index), 'x': moons['x'][index].tolist(), 'y': int(moons['y'][index])} for index in moons['valid']],
    'models': []
}
for c, gamma, model, training, validation in moons['fitted']:
    data['models'].append({'c': c, 'gamma': gamma, 'training': training, 'validation': validation,
                           'support': model.support_vectors_.tolist(), 'coefficients': model.dual_coef_[0].tolist(),
                           'bias': float(model.intercept_[0]), 'supportCount': len(model.support_)})

header = '// Generated by scripts/build-support-vector-machines-examples.py from executed complete programs.\n'
(ROOT / 'src/learn/data/support-vector-machines-examples.js').write_text(
    header + 'export const svmExamples = ' + json.dumps(examples, indent=2, ensure_ascii=False) + ';\n', encoding='utf8')
(ROOT / 'src/learn/data/svm-validation-fixtures.js').write_text(
    header + 'export default ' + json.dumps(data, indent=2, ensure_ascii=False) + ';\n', encoding='utf8')
record = {'executedAt': datetime.now(timezone.utc).isoformat(),
          'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
          'programCount': len(records), 'programs': records,
          'fixturePath': 'src/learn/data/svm-validation-fixtures.js'}
(OUT / 'program-results.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
print(json.dumps({'programs': len(records), 'models': len(data['models']), 'executedAt': record['executedAt']}))
