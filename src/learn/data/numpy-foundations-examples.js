// Each displayed example is a complete independent Python program.
export const numpyFoundationsExamples = {
  create: {
    code: `import numpy as np

# Rows: times 0, 1, 2. Columns: sensors A, B. All values are in degrees C.
X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
print("shape:", X.shape, "axes:", X.ndim, "values:", X.size)
print("dtype:", X.dtype, "bytes per value:", X.itemsize)
print("time 1, sensor A:", X[1, 0])
print("list repetition:", [18, 20] * 2)
print("array multiplication:", (X[0] * 2).tolist())`,
    output: `shape: (3, 2) axes: 2 values: 6
dtype: float64 bytes per value: 8
time 1, sensor A: 24.0
list repetition: [18, 20, 18, 20]
array multiplication: [36.0, 40.0]`,
  },
  select: {
    code: `import numpy as np

X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
print("column:", X[:, 0].tolist(), X[:, 0].shape)
print("column with axis:", X[:, :1].tolist(), X[:, :1].shape)
row_mask = X[:, 0] >= 24
print("row mask:", row_mask.tolist())
print("whole rows:", X[row_mask].tolist(), X[row_mask].shape)
print("cells:", X[X >= 24].tolist(), X[X >= 24].shape)
both = (X[:, 0] >= 24) & (X[:, 1] < 30)
print("both conditions:", X[both].tolist())`,
    output: `column: [18.0, 24.0, 30.0] (3,)
column with axis: [[18.0], [24.0], [30.0]] (3, 1)
row mask: [False, True, True]
whole rows: [[24.0, 26.0], [30.0, 32.0]] (2, 2)
cells: [24.0, 26.0, 30.0, 32.0] (4,)
both conditions: [[24.0, 26.0]]`,
  },
  memory: {
    code: `import numpy as np

X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
view = X[:, 0]
copied = X[:, 0].copy()
gathered = X[[0, 1, 2], 0]
print("shared:", np.shares_memory(X, view), np.shares_memory(X, copied),
      np.shares_memory(X, gathered))
view[1] = 99
copied[1] = -1
print("raw after writes:", X.tolist())
print("copied:", copied.tolist(), "gathered:", gathered.tolist())
print("X strides:", X.strides, "view strides:", view.strides)

# Direct indexed assignment targets X, even when that selection would copy on read.
X[[0, 2], 0] = 7
print("direct assignment:", X[:, 0].tolist())`,
    output: `shared: True False False
raw after writes: [[18.0, 20.0], [99.0, 26.0], [30.0, 32.0]]
copied: [18.0, -1.0, 30.0] gathered: [18.0, 24.0, 30.0]
X strides: (16, 8) view strides: (16,)
direct assignment: [7.0, 99.0, 7.0]`,
  },
  broadcast: {
    code: `import numpy as np

X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
offsets = np.array([2, 4])
print("per sensor:", (X - offsets).tolist())
time_offsets = np.array([1, 2, 3])
try:
    X - time_offsets
except ValueError:
    print("(3, 2) and (3,) do not broadcast")
print("per time:", (X - time_offsets[:, None]).tolist())
v = np.array([18, 24, 30])
print("pairwise:", (v[:, None] - v).tolist())`,
    output: `per sensor: [[16.0, 16.0], [22.0, 22.0], [28.0, 28.0]]
(3, 2) and (3,) do not broadcast
per time: [[17.0, 19.0], [22.0, 24.0], [27.0, 29.0]]
pairwise: [[0, -6, -12], [6, 0, -6], [12, 6, 0]]`,
  },
  reduce: {
    code: `import numpy as np

X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
sensor_mean = X.mean(axis=0)
time_mean = X.mean(axis=1, keepdims=True)
print("per sensor:", sensor_mean.tolist(), sensor_mean.shape)
print("per time:", time_mean.tolist(), time_mean.shape)
print("relative to each time:", (X - time_mean).tolist())
print("all readings:", X.mean().item())
print("sensor A maximum at time:", X[:, 0].argmax().item())`,
    output: `per sensor: [24.0, 26.0] (2,)
per time: [[19.0], [25.0], [31.0]] (3, 1)
relative to each time: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
all readings: 25.0
sensor A maximum at time: 2`,
  },
  loopReference: {
    filename: 'numpy_loop_reference.py',
    code: `import numpy as np

# Nonempty rectangular readings; one finite offset per sensor.
readings = [[18.0, 20.0], [24.0, 26.0], [30.0, 32.0]]
offsets = [2.0, 4.0]
times, sensors = len(readings), len(offsets)
corrected = []
for time in range(times):
    row = []
    for sensor in range(sensors):
        row.append(readings[time][sensor] - offsets[sensor])
    corrected.append(row)

means = []
for sensor in range(sensors):
    total = 0.0
    for time in range(times):
        total += corrected[time][sensor]
    means.append(total / times)

def calibrate_sensor_means(readings, offsets):
    array = np.asarray(readings, dtype=np.float64)
    offset = np.asarray(offsets, dtype=np.float64)
    if (array.ndim != 2 or 0 in array.shape or offset.ndim != 1
            or offset.size != array.shape[1]):
        raise ValueError("expected a nonempty table and one offset per sensor")
    if not (np.isfinite(array).all() and np.isfinite(offset).all()):
        raise ValueError("readings and offsets must be finite")
    calibrated = array - offset
    return calibrated, calibrated.mean(axis=0)

vectorized, array_means = calibrate_sensor_means(readings, offsets)
np.testing.assert_allclose(vectorized, corrected, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(array_means, means, rtol=1e-12, atol=1e-12)
print("loop corrected:", corrected)
print("loop sensor means:", means)
print("same shape:", vectorized.shape == (times, sensors))
print("array sensor means:", array_means.tolist())`,
    output: `loop corrected: [[16.0, 16.0], [22.0, 22.0], [28.0, 28.0]]
loop sensor means: [22.0, 22.0]
same shape: True
array sensor means: [22.0, 22.0]`,
  },
  shapes: {
    code: `import numpy as np

X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
print("transpose:", X.T.tolist())
print("reshape:", X.reshape(2, 3).tolist())
v = X[:, 0]
print("v and v.T:", v.shape, v.T.shape)
print("new column axis:", v[:, None].shape)
extra = np.array([[22, 24]])
print("another time:", np.concatenate([X, extra], axis=0).shape)
print("two runs:", np.stack([X, X], axis=0).shape)
print("regular samples:", np.linspace(0, 1, 3).tolist())`,
    output: `transpose: [[18.0, 24.0, 30.0], [20.0, 26.0, 32.0]]
reshape: [[18.0, 20.0, 24.0], [26.0, 30.0, 32.0]]
v and v.T: (3,) (3,)
new column axis: (3, 1)
another time: (4, 2)
two runs: (2, 3, 2)
regular samples: [0.0, 0.5, 1.0]`,
  },
  weighted: {
    code: `import numpy as np

X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
w = np.array([0.25, 0.75])
contributions = X * w
weighted = X @ w
print("contributions:", contributions.tolist(), contributions.shape)
print("one result per time:", weighted.tolist(), weighted.shape)
assert weighted.shape == (3,)
np.testing.assert_allclose(weighted, contributions.sum(axis=1),
                           rtol=1e-12, atol=1e-12)
print("two formulations agree")`,
    output: `contributions: [[4.5, 15.0], [6.0, 19.5], [7.5, 24.0]] (3, 2)
one result per time: [19.5, 25.5, 31.5] (3,)
two formulations agree`,
  },
  numerical: {
    code: `import numpy as np

small = np.array([120], dtype=np.int8)
print("overflow:", (small + small).tolist())
wide = small.astype(np.int16)
print("convert BEFORE adding:", (wide + wide).tolist())
integer_readings = np.array([18, 20], dtype=np.int64)
try:
    integer_readings /= 2
except TypeError:
    print("in-place float division cannot write into this integer array")
print("new floating result:", (integer_readings / 2).tolist())
print("decimal equality:", bool(np.float64(0.1) + np.float64(0.2) == 0.3))
print("within tolerance:", bool(np.isclose(0.1 + 0.2, 0.3, rtol=0, atol=1e-15)))
values = np.array([1.0, np.nan, np.inf])
print("finite:", np.isfinite(values).tolist())
print("is NaN:", np.isnan(values).tolist())`,
    output: `overflow: [-16]
convert BEFORE adding: [240]
in-place float division cannot write into this integer array
new floating result: [9.0, 10.0]
decimal equality: False
within tolerance: True
finite: [True, False, False]
is NaN: [False, True, False]`,
  },
  workflow: {
    filename: 'numpy_sensor_report.py',
    code: `import numpy as np

# Invented readings in degrees C; one row per time, columns A and B.
raw = np.array([[18, 20], [24, np.nan], [30, 32], [22, 24]],
               dtype=np.float64)
offsets = np.array([2.0, 4.0])  # Known calibration offsets in degrees C.
original = raw.copy()
if raw.ndim != 2 or raw.shape[1] != offsets.size:
    raise ValueError("expected a table with one column per calibration offset")
valid = np.isfinite(raw).all(axis=1)
if not valid.any():
    raise ValueError("no complete finite observations")
times = np.arange(raw.shape[0])[valid]
clean = raw[valid].copy()
calibrated = clean - offsets
per_sensor = calibrated.mean(axis=0)
per_time = calibrated.mean(axis=1)
warm = per_time > 23
print("kept times:", times.tolist())
print("calibrated:", calibrated.tolist())
print("sensor means:", np.round(per_sensor, 3).tolist())
print("warm times:", times[warm].tolist())
assert calibrated.shape == (3, 2)
assert np.array_equal(raw, original, equal_nan=True)
np.testing.assert_allclose(per_sensor, [64 / 3, 64 / 3],
                           rtol=1e-12, atol=1e-12)
assert np.array_equal(times[warm], [2])
print("shape, original data and calculated results checked")`,
    output: `kept times: [0, 2, 3]
calibrated: [[16.0, 16.0], [28.0, 28.0], [20.0, 20.0]]
sensor means: [21.333, 21.333]
warm times: [2]
shape, original data and calculated results checked`,
  },
  transfer: {
    filename: 'numpy_energy_report.py',
    code: `import numpy as np

# Rows are days; columns are meters A, B, C. Readings are in kWh.
raw = np.array([[10, 20, 30], [20, 10, 20],
                [30, np.nan, 10], [40, 20, 10]], dtype=np.float64)
gains = np.array([1.0, 0.5, 2.0])  # Dimensionless calibration multipliers.
original = raw.copy()
valid = np.isfinite(raw).all(axis=1)
days = np.arange(raw.shape[0])[valid]
corrected = raw[valid] * gains
daily_total = corrected.sum(axis=1)
per_meter = corrected.mean(axis=0, keepdims=True)
deviation = corrected - per_meter
print("kept days:", days.tolist())
print("corrected:", corrected.tolist())
print("daily kWh:", daily_total.tolist())
print("days above 65 kWh:", days[daily_total > 65].tolist())
print("mean shape:", per_meter.shape, "deviation shape:", deviation.shape)
assert np.array_equal(raw, original, equal_nan=True)
assert corrected.shape == (3, 3)
assert np.array_equal(daily_total, [80, 65, 70])
np.testing.assert_allclose(deviation.mean(axis=0), [0, 0, 0],
                           rtol=0, atol=1e-12)
print("original preserved; each meter's deviations average to zero")`,
    output: `kept days: [0, 1, 3]
corrected: [[10.0, 10.0, 60.0], [20.0, 5.0, 40.0], [40.0, 10.0, 20.0]]
daily kWh: [80.0, 65.0, 70.0]
days above 65 kWh: [0, 3]
mean shape: (1, 3) deviation shape: (3, 3)
original preserved; each meter's deviations average to zero`,
  },
};
