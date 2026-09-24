import numpy as np

for conflicting, c in [(False, .25), (False, 1.0), (True, 1.0)]:
    x = np.array([0., 0.]) if conflicting else np.array([-1., 1.])
    y = np.array([-1., 1.])
    alpha = c if conflicting else min(c, .5)
    w = np.sum(alpha * y * x)
    limit = max(0., 1 - abs(x[0]) * w)
    b = limit / 2
    hinge = np.maximum(0., 1 - y * (w * x + b))
    primal = .5 * w*w + c * hinge.sum()
    dual = 2 * alpha - .5 * w*w
    print(f'conflicting={conflicting} C={c:.2f} w={w:.2f} b interval=[{-limit:.2f},{limit:.2f}]')
    print('hinge:', hinge.round(3).tolist(), f'primal={primal:.3f} dual={dual:.3f}')

# Dividing the sum-loss objective by C*n gives lambda/2 ||w||² + mean hinge.
n, c, copies = 8, 2.0, 3
print(f'lambda={1/(n*c):.4f}; duplicated-data C={c/copies:.6f}')
