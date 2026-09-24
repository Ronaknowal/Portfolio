"""Optional hmmlearn 0.3.3 examples; dependency unavailable during authoring.

Run after installing compatible numpy and hmmlearn in a separate environment.
No stdout or native results from this file are claimed until it is executed.
"""
import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM

activities = np.array([[0], [1], [0], [2]], dtype=int)
fixed = CategoricalHMM(n_components=2, n_features=3, init_params="", params="")
fixed.startprob_ = np.array([0.6, 0.4])
fixed.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
fixed.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
print("Observation log probability:", fixed.score(activities))
joint_log_score, path = fixed.decode(activities, algorithm="viterbi")
print("Best joint log score and path:", joint_log_score, path)
print("Smoothed marginals:", fixed.predict_proba(activities))
expected_correct_count, pointwise = fixed.decode(activities, algorithm="map")
print("Pointwise expected correct count and modes:", expected_correct_count, pointwise)
assert np.isclose(fixed.score(activities), np.log(0.00933936))
assert np.array_equal(path, [1, 1, 1, 0])

# Store independent recordings together, but retain their boundaries.
recordings = [np.array([[0], [1]]), np.array([[0], [2]])]
joined = np.concatenate(recordings)
lengths = [len(recording) for recording in recordings]
learned = CategoricalHMM(n_components=2, n_features=3, n_iter=30,
                         tol=1e-6, random_state=7)
learned.fit(joined, lengths)
print("Fitted independent-sequence score:", learned.score(joined, lengths))
print("Training history (check budget and objective):", list(learned.monitor_.history))

# A constructed scalar sensor: continuous emissions, two independent recordings.
rng = np.random.default_rng(8)
sensor_recordings = []
for _ in range(2):
    hidden = 0
    readings = []
    for time in range(60):
        if time and rng.random() > 0.9:
            hidden = 1 - hidden
        readings.append([rng.normal(-1.5 if hidden == 0 else 1.5, 0.4)])
    sensor_recordings.append(np.asarray(readings))
sensor = np.concatenate(sensor_recordings)
sensor_lengths = [len(recording) for recording in sensor_recordings]
gaussian = GaussianHMM(n_components=2, covariance_type="diag", n_iter=50,
                       tol=1e-5, random_state=9)
gaussian.fit(sensor, sensor_lengths)
print("Sensor log density:", gaussian.score(sensor, sensor_lengths))
print("Means (state indices have no fixed meaning):", gaussian.means_.ravel())
print("Variances:", gaussian.covars_)
print("First recording smoothed probabilities:",
      gaussian.predict_proba(sensor_recordings[0])[:5])

