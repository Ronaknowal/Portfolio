const plot = (body, name, output) => ({
  code: `import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
${body}
fig.savefig("${name}.svg", facecolor=fig.get_facecolor())
plt.close(fig)
print("saved ${name}.svg")`,
  output: `${output}\nsaved ${name}.svg`,
  artifact: `${name}.svg`,
});

export const plottingExamples = {
  curves: plot(`epoch = np.array([1, 2, 3, 4])
train = np.array([1.10, 0.72, 0.48, 0.39])
valid = np.array([1.20, 0.78, 0.61, 0.66])
if epoch.shape != train.shape or train.shape != valid.shape:
    raise ValueError("one loss per epoch is required")
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(epoch, train, "o-", color="#e2b55a", label="Training")
ax.plot(epoch, valid, "s--", color="#8fcfe0", label="Validation")
best = int(np.argmin(valid))
ax.annotate("Lowest validation loss", xy=(epoch[best], valid[best]),
            xytext=(1.3, 0.43), arrowprops={"arrowstyle": "->", "color": "white"})
ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Does validation improve with training?")
ax.set_xticks(epoch)
ax.legend()
ax.grid(axis="y", alpha=0.2)
print("best epoch:", int(epoch[best]), "validation loss:", float(valid[best]))`, "training-curves", "best epoch: 3 validation loss: 0.61"),
  bars: plot(`region = ["North", "South"]
revenue = [10, 20]
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
bars = ax.bar(region, revenue, color="#e2b55a", width=0.6)
ax.bar_label(bars, fmt="%.0f", padding=4)
ax.set(ylim=(0, 24), ylabel="Accepted paid revenue (cents)",
       title="South has twice North's accepted revenue")
print("total cents:", sum(revenue))`, "regional-revenue", "total cents: 30"),
  scatter: plot(`size = np.array([1, 2, 3, 4, 5])
latency = np.array([9, 13, 12, 20, 25])
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.scatter(size, latency, s=65, alpha=0.8, color="#8fcfe0", edgecolors="white")
ax.set(xlabel="Payload size (MB)", ylabel="Latency (ms)",
       title="Larger payloads tend to take longer in this sample")
ax.grid(alpha=0.2)
print("paired observations:", len(size))`, "payload-latency", "paired observations: 5"),
  histogram: plot(`latency = np.array([1, 2, 2, 3, 7, 9])
edges = np.array([0, 3, 6, 10])
fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
counts, _, _ = axes[0].hist(latency, bins=edges, color="#e2b55a", edgecolor="black")
density, _, _ = axes[1].hist(latency, bins=edges, density=True, color="#8fcfe0", edgecolor="black")
for ax in axes:
    ax.set(xlabel="Latency (ms)", xticks=edges)
axes[0].set(ylabel="Requests", title="Count per bin")
axes[1].set(ylabel="Density (1/ms)", title="Area, not height, totals one")
print("counts:", counts.astype(int).tolist())
print("densities:", np.round(density, 4).tolist())
print("density area:", round(float(np.sum(density * np.diff(edges))), 4))`, "histogram-density", "counts: [3, 1, 2]\ndensities: [0.1667, 0.0556, 0.0833]\ndensity area: 1.0"),
  uncertainty: plot(`# Rows are three runs; columns are two configurations.
runs = np.array([[10, 13], [12, 14], [14, 15]], dtype=float)
mean = runs.mean(axis=0)
sd = runs.std(axis=0, ddof=1)
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
for row in runs:
    ax.plot([0, 1], row, "o", color="#8fcfe0", alpha=0.65)
ax.errorbar([0, 1], mean, yerr=sd, fmt="D", color="#e2b55a",
            capsize=6, label="Mean ± sample SD")
ax.set_xticks([0, 1], ["A", "B"])
ax.set(ylabel="Latency (ms)", title="Three run means per configuration")
ax.legend()
print("means:", mean.tolist())
print("sample SD:", sd.tolist())`, "run-variation", "means: [12.0, 14.0]\nsample SD: [2.0, 1.0]"),
  heatmap: plot(`counts = np.array([[8, 2], [1, 9]])
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
image = ax.imshow(counts, cmap="viridis", vmin=0, vmax=10,
                  origin="upper", interpolation="nearest")
ax.set_xticks([0, 1], ["Negative", "Positive"])
ax.set_yticks([0, 1], ["Negative", "Positive"])
ax.set(xlabel="Predicted class", ylabel="True class", title="Counts on a fixed 0–10 colour scale")
for (row, col), value in np.ndenumerate(counts):
    ax.text(col, row, str(value), ha="center", va="center",
            color="black" if value >= 6 else "white")
fig.colorbar(image, ax=ax, label="Observations")
print("correct / total:", int(np.trace(counts)), "/", int(counts.sum()))`, "confusion-counts", "correct / total: 17 / 20"),
  scales: plot(`step = [0, 1, 2, 3]
error = [1, 0.1, 0.01, 0.001]
fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
for ax in axes:
    ax.plot(step, error, "o-", color="#e2b55a")
    ax.set(xlabel="Iteration", ylabel="Absolute error", xticks=step)
    ax.grid(alpha=0.2)
axes[0].set_title("Linear: equal differences")
axes[1].set_yscale("log")
axes[1].set_title("Log: equal ratios")
print("ratio per step:", [round(error[i + 1] / error[i], 2) for i in range(3)])`, "linear-log", "ratio per step: [0.1, 0.1, 0.1]"),
};
