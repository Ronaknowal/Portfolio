import math

def central_difference(function, point, step):
    return (function(point + step) - function(point - step)) / (2 * step)

for step in (1e-1, 1e-3, 1e-5, 1e-9, 1e-13, 1e-15):
    estimate = central_difference(math.sin, 1.0, step)
    error = abs(estimate - math.cos(1.0))
    offset_estimate = central_difference(lambda x: 1e12 + x, 1.0, step)
    print(f"{step:.0e}", f"{error:.3e}", f"{offset_estimate:.9f}")
