"""Euler-Maruyama, scalar Milstein, exact GBM, and a same-noise sdeint bridge.

Run: python sde-library-bridge.py
Dependencies: numpy==2.3.5, sdeint==0.3.0.
Fixed uniform grid, explicit supplied increments dW[step,driver]. Drift and
diffusion callbacks return [state] and [state,driver]. Time-stepping is sequential;
state/driver arithmetic is vectorized. This is not an adaptive SDE solver.
"""
import numpy as np
import sdeint


def euler_maruyama(drift, diffusion, initial, times, increments):
    initial, times, increments = map(lambda a: np.asarray(a, float), (initial, times, increments))
    if (initial.ndim != 1 or not initial.size or times.ndim != 1 or len(times) < 2
            or increments.ndim != 2 or not increments.shape[1] or increments.shape[0] != len(times)-1
            or not np.isfinite(initial).all() or not np.isfinite(times).all()
            or not np.isfinite(increments).all()):
        raise ValueError("Finite state, time vector and [steps,drivers] increments required")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-12, atol=0):
        raise ValueError("This comparison uses a strictly increasing uniform grid")
    path = np.empty((len(times), len(initial))); path[0] = initial
    for k, dt in enumerate(steps):
        f, g = np.asarray(drift(path[k], times[k])), np.asarray(diffusion(path[k], times[k]))
        if f.shape != initial.shape or g.shape != (len(initial), increments.shape[1]):
            raise ValueError("Callback shapes must match state and driver dimensions")
        path[k+1] = path[k] + f*dt + g @ increments[k]
    if not np.isfinite(path).all():
        raise FloatingPointError("The explicit path diverged or coefficients overflowed")
    return path


def scalar_milstein(drift, diffusion, diffusion_derivative, initial, dt, increments):
    """One scalar state and one noise driver; no multidimensional Levy areas."""
    increments = np.asarray(increments, float)
    if (increments.ndim != 1 or not np.isfinite(increments).all()
            or not np.isfinite(initial) or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Finite scalar increments and positive step size required")
    path = np.empty(len(increments)+1); path[0] = initial
    for k, dw in enumerate(increments):
        x, t = path[k], k*dt
        g = diffusion(x, t)
        path[k+1] = x + drift(x, t)*dt + g*dw + .5*g*diffusion_derivative(x, t)*(dw*dw-dt)
    if not np.isfinite(path).all():
        raise FloatingPointError("The explicit path diverged or coefficients overflowed")
    return path


def main():
    drift_rate, volatility = .1, .4
    drift = lambda x, t: drift_rate*x
    diffusion = lambda x, t: np.diag(volatility*x)
    times = np.linspace(0, 1, 65)
    increments = np.random.default_rng(53).normal(0, np.sqrt(1/64), (64, 1))
    manual = euler_maruyama(drift, diffusion, [1.], times, increments)
    library = sdeint.itoEuler(drift, diffusion, [1.], times, dW=increments)
    np.testing.assert_allclose(manual, library, atol=1e-14)
    exact = np.exp((drift_rate-volatility**2/2)*times + volatility*np.r_[0., increments[:, 0].cumsum()])
    milstein = scalar_milstein(lambda x,t: drift_rate*x, lambda x,t: volatility*x,
                               lambda x,t: volatility, 1., 1/64, increments[:, 0])
    coarse = increments.reshape(16, 4, 1).sum(axis=1)
    coarse_path = euler_maruyama(drift, diffusion, [1.], times[::4], coarse)
    print("Euler library max_error", f"{np.max(abs(manual-library)):.2e}")
    print("terminal exact/Euler/Milstein/coarse Euler", np.round([exact[-1], manual[-1, 0], milstein[-1], coarse_path[-1, 0]], 9).tolist())
    print("coarse/fine Brownian endpoint agreement", bool(np.isclose(coarse.sum(), increments.sum())))
    # A non-diagonal additive diffusion checks the orientation of independent drivers.
    matrix = np.array([[1., .3], [-.2, .7]])
    vector_noise = np.random.default_rng(54).normal(0, .1, (10, 2))
    vector_times = np.linspace(0, .1, 11)
    zero = lambda x,t: np.zeros(2)
    constant = lambda x,t: matrix
    path = euler_maruyama(zero, constant, [0., 0.], vector_times, vector_noise)
    reference = sdeint.itoEuler(zero, constant, [0., 0.], vector_times, dW=vector_noise)
    np.testing.assert_allclose(path, reference, atol=1e-14)
    np.testing.assert_allclose(path[-1], matrix @ vector_noise.sum(axis=0), atol=1e-14)
    print("two-state additive terminal", np.round(path[-1], 9).tolist())


if __name__ == "__main__":
    main()
