from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared

data = np.genfromtxt(
    Path(__file__).with_name("mauna-loa-monthly.csv"),
    delimiter=",", names=True,
)
x = ((data["year"] - 1990) + (data["month"] - 0.5) / 12)[:, None]
y = data["co2_ppm"]
training = np.arange(72)
development = np.arange(72, 96)
test = np.arange(96, 120)
noise_variance = 0.09
families = {
    "rbf": 4.0 * RBF(1.0, (0.2, 20.0)),
    "trend_periodic": (
        DotProduct(1.0, sigma_0_bounds="fixed")
        + 4.0 * ExpSineSquared(
            1.0, 1.0, length_scale_bounds=(0.2, 5.0),
            periodicity_bounds="fixed",
        )
        + RBF(3.0, (0.5, 20.0))
    ),
}

def fit(kernel, rows):
    center = y[rows].mean()
    model = GaussianProcessRegressor(
        kernel=clone(kernel), alpha=noise_variance,
        normalize_y=False, n_restarts_optimizer=1, random_state=23,
    )
    model.fit(x[rows], y[rows] - center)
    return model, center

def evaluate(model, center, rows):
    mean, latent_sd = model.predict(x[rows], return_std=True)
    mean = mean + center
    observation_sd = np.sqrt(latent_sd**2 + noise_variance)
    errors = y[rows] - mean
    mae = np.abs(errors).mean()
    rmse = np.sqrt(np.mean(errors**2))
    covered = np.count_nonzero(np.abs(errors) <= 1.959963984540054 * observation_sd)
    return float(mae), float(rmse), int(covered)

development_scores = {}
for name, kernel in families.items():
    model, center = fit(kernel, training)
    scores = evaluate(model, center, development)
    development_scores[name] = scores[0]
    print(name, "development", tuple(round(v, 6) for v in scores))

selected = min(development_scores, key=development_scores.get)
model, center = fit(families[selected], np.arange(96))
print("selected", selected)
print("test", tuple(round(v, 6) for v in evaluate(model, center, test)))
seasonal_naive = y[84 + np.arange(24) % 12]
print("seasonal_naive MAE", round(float(np.abs(y[test] - seasonal_naive).mean()), 6))
