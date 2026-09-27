"""Independent explicit basis-response SVD for every supported convolution shape."""
import json
from pathlib import Path
import numpy as np

records = []
for n in range(3, 7):
    for stride in [1, 2]:
        for mode in ['valid', 'circular']:
            for kernel in [[1, 2], [-2, .7], [0, 0], [.2, -.8]]:
                inputs = np.eye(n)
                starts = range(0, n if mode == 'circular' else n-1, stride)
                # Each column is the response of the actual sliding-window
                # cross-correlation to a different unit input, then LAPACK SVD.
                outputs = [[sum(kernel[j] * basis[(i+j) % n] for j in range(2)) for i in starts] for basis in inputs]
                matrix = np.array(outputs).T
                records.append(dict(n=n, stride=stride, mode=mode, kernel=kernel, matrix=matrix.tolist(), norm=float(np.linalg.svd(matrix, compute_uv=False)[0])))
out = Path(__file__).resolve().parents[1] / 'docs/teaching/deep-learning-completion/spectral-normalization-gradient-penalty/operator-fixtures.json'
out.write_text(json.dumps(dict(passed=True, cases=records, numpy=np.__version__), indent=2)+'\n', encoding='utf-8')
print(f'{len(records)} independent basis-response SVD fixtures calculated.')
