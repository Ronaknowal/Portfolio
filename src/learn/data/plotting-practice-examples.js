export const plottingPracticeExamples={
  residual:{code:`import numpy as np
import matplotlib.pyplot as plt

x = np.array([0., 1., 2., 3., 4.])
observed = np.array([1., 1.5, 3., 4.5, 7.])
predicted = 1 + x
residual = observed - predicted
assert x.shape == observed.shape == predicted.shape
assert np.isfinite(observed).all()
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 5), layout="constrained")
    axes[0].scatter(x, observed, marker="o", label="Observed")
    axes[0].plot(x, predicted, "--", label="Proposed straight-line model")
    axes[0].set(ylabel="Response (units)")
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, color="0.7", linewidth=1)
    axes[1].scatter(x, residual, marker="s", color="#e1b85c")
    axes[1].set(xlabel="Input setting", ylabel="Observed − predicted (units)")
    fig.savefig("residual-diagnostic.svg")
    plt.close(fig)
print("residuals:", residual.tolist())
print("mean residual:", round(residual.mean(), 2))
print("saved residual-diagnostic.svg")`,output:`residuals: [0.0, -0.5, 0.0, 0.5, 2.0]
mean residual: 0.4
saved residual-diagnostic.svg`,artifact:'residual-diagnostic.svg'},
  transfer:{code:`import numpy as np
import matplotlib.pyplot as plt

distance = np.array([1., 2., 3., 4.])
measured = np.array([2., 4., 7., 8.])
predicted = 2 * distance
residual = measured - predicted
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 5), layout="constrained")
    axes[0].scatter(distance, measured, label="Measured", marker="o")
    axes[0].plot(distance, predicted, "--", label="Prediction: 2 × distance")
    axes[0].set(ylabel="Time (ms)")
    axes[0].legend()
    axes[1].axhline(0, color="0.7", linewidth=1)
    axes[1].scatter(distance, residual, marker="s")
    axes[1].set(xlabel="Distance (m)", ylabel="Residual (ms)")
    fig.savefig("residual-practice.svg")
    plt.close(fig)
assert residual.tolist() == [0., 0., 1., 0.]
print("largest residual at distance:", distance[np.argmax(np.abs(residual))])
print("saved residual-practice.svg")`,output:`largest residual at distance: 3.0
saved residual-practice.svg`,artifact:'residual-practice.svg'},
};
