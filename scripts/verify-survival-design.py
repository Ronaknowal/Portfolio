"""Check proposed teaching fixtures before committing the lesson design."""
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import inspect
import json
import math
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from sksurv.nonparametric import cumulative_incidence_competing_risks
from sksurv.util import Surv
from scipy.special import logsumexp
import pandas as pd

root = Path(__file__).resolve().parents[1]
directory = root / 'scratch/survival'
directory.mkdir(parents=True, exist_ok=True)
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'provisional design fixtures; not author verification'}

times = np.array([2, 3, 4, 4, 6, 7, 8, 9], dtype=float)
events = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
survival = Fraction(1)
greenwood = Fraction(0)
rows = []
for time in sorted(set(times[events])):
    risk = int(np.count_nonzero(times >= time))
    failures = int(np.count_nonzero((times == time) & events))
    survival *= Fraction(risk - failures, risk)
    greenwood += Fraction(failures, risk * (risk - failures))
    rows.append({'time': time, 'risk': risk, 'events': failures, 'survival': str(survival), 'greenwood': str(greenwood)})
km = KaplanMeierFitter().fit(times, events)
assert np.allclose(km.predict([2, 4, 6, 8]), [7/8, 35/48, 35/64, 35/128])
rmst = Fraction(2) + 2 * Fraction(7, 8) + 2 * Fraction(35, 48) + 2 * Fraction(35, 64) + Fraction(35, 128)
result['km'] = {'rows': rows, 'rmst9': str(rmst), 'median': float(km.median_survival_time_), 'pointwise95at4': km.confidence_interval_.loc[4].tolist(), 'exponentialRateMLE': float(events.sum()/times.sum()), 'completedOnlyMean': float(times[events].mean()), 'allObservedMean': float(times.mean())}

x = np.array([-1, 0, 1, -1, 0, 1], dtype=float)
cox_time = np.array([1, 2, 2, 3, 4, 5], dtype=float)
cox_event = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
beta = math.log(2)
weights = np.exp(beta * x)
risk_weights = weights[cox_time >= 2]
breslow = 2 / float(risk_weights.sum() ** 2)
efron = 2 / float(risk_weights.sum() * (risk_weights.sum() - 1.5))
unordered = 2 / sum(a*b for a, b in combinations(risk_weights, 2))
assert np.allclose([breslow, efron, unordered], [8/169, 4/65, 1/8])
result['coxTiedContribution'] = {'riskWeights': risk_weights.tolist(), 'breslow': breslow, 'efron': efron, 'discreteUnorderedConditional': unordered, 'warning': 'The unordered conditional likelihood has a different normalization. These raw contributions do not establish a likelihood ranking across different conventions.'}

ph_time = 6
assert math.isclose(.1 * ph_time, .05 * 4 + .2 * (ph_time-4))
def mixed_hazard(time):
    return (.1 * math.exp(-.1*time) + .4 * math.exp(-.4*time)) / (math.exp(-.1*time) + math.exp(-.4*time))
result['ph'] = {'switch': 4, 'survivalCrossing': 6, 'commonSurvivalAtCrossing': math.exp(-.6), 'mixtureMarginalHRat5': .5 * mixed_hazard(2.5)/mixed_hazard(5), 'weibullScale12Shape2Median': 12*math.sqrt(math.log(2)), 'aftTimeRatio2HazardRatio': .25}

competing_time = np.array([1, 2, 3, 4, 5, 6], dtype=float)
status = np.array([2, 1, 0, 2, 1, 0])
grid, cif = cumulative_incidence_competing_risks(status, competing_time)
assert np.allclose(cif[1:, -1], [7/18, 7/18])
result['competing'] = {'lastTime': float(grid[-1]), 'F1': float(cif[1, -1]), 'F2': float(cif[2, -1]), 'eventFree': 2/9, 'naiveOneMinusKM': .6, 'constantHazardsF1at5': .1/.4 * -math.expm1(-.4*5), 'netFailureAt5': -math.expm1(-.1*5)}

truth = {2: .4, 8: .6}
censor = {3: .5, 10: .5}
predicted_survival = .6
weighted = 0
for lifetime, lifetime_mass in truth.items():
    for followup, followup_mass in censor.items():
        observed = min(lifetime, followup)
        event = lifetime <= followup
        if event and observed <= 5:
            contribution = predicted_survival**2 / sum(m for c, m in censor.items() if c >= observed)
        elif observed > 5:
            contribution = (1-predicted_survival)**2 / sum(m for c, m in censor.items() if c > 5)
        else:
            contribution = 0
        weighted += lifetime_mass * followup_mass * contribution
assert math.isclose(weighted, .24)
result['ipcw'] = {'knownGWeightedScore': weighted, 'fullDataScore': .24, 'completeCaseScore': (.4*.36+.3*.16)/.7, 'censorAsFailureScore': .7*.36+.3*.16, 'G5': .5, 'noSharedEventCensorTimes': True}

pair_time = np.array([1., 2., 2., 4.])
pair_event = np.array([True, False, True, True])
pair_risk = np.array([3., 2., 2., 0.])
result['pairs'] = list(map(float, concordance_index_censored(pair_event, pair_time, pair_risk)))
assert result['pairs'] == [0.9, 4., 0., 1., 1.]

rng = np.random.default_rng(73)
features = np.column_stack([rng.normal(size=900), rng.integers(0, 2, 900), rng.normal(size=900)])
score = .6*features[:, 0]-.7*features[:, 1]+.5*(features[:, 2]**2-1)
actual = 12*(rng.exponential(size=900)/np.exp(score))**(1/1.5)
followup = rng.uniform(8, 25, size=900)
observed = np.minimum(actual, followup)
indicator = actual <= followup
train, validation, test = np.arange(600), np.arange(600, 750), np.arange(750, 900)
target = Surv.from_arrays(indicator, observed)
def administratively_capped(indices):
    return Surv.from_arrays(indicator[indices] & (observed[indices] <= 12), np.minimum(observed[indices], 12))
validation_target = administratively_capped(validation)
test_target = administratively_capped(test)
evaluation_grid = np.array([2., 4., 6., 8., 10.])
assert observed[train].max() > 12
expanded = np.column_stack([features, features[:, 2]**2])
fits = []
train_km = KaplanMeierFitter().fit(observed[train], indicator[train])
baseline_validation = np.tile(train_km.predict(evaluation_grid).to_numpy(), (len(validation), 1))
fits.append({'name': 'training KM', 'validationIBS': float(integrated_brier_score(target[train], validation_target, baseline_validation, evaluation_grid))})
for name, matrix, estimator in [('linear Cox', features, CoxPHSurvivalAnalysis(alpha=.1, ties='efron')), ('quadratic Cox', expanded, CoxPHSurvivalAnalysis(alpha=.1, ties='efron')), ('forest leaf10', features, RandomSurvivalForest(n_estimators=60, min_samples_leaf=10, max_features=1., random_state=73, n_jobs=1)), ('forest leaf25', features, RandomSurvivalForest(n_estimators=60, min_samples_leaf=25, max_features=1., random_state=73, n_jobs=1))]:
    estimator.fit(matrix[train], target[train])
    validation_prediction = np.array([fn(evaluation_grid) for fn in estimator.predict_survival_function(matrix[validation])])
    test_prediction = np.array([fn(evaluation_grid) for fn in estimator.predict_survival_function(matrix[test])])
    fits.append({'name': name, 'validationIBS': float(integrated_brier_score(target[train], validation_target, validation_prediction, evaluation_grid)), 'testIBS': float(integrated_brier_score(target[train], test_target, test_prediction, evaluation_grid)), 'testC': float(concordance_index_censored(test_target['event'], test_target['time'], estimator.predict(matrix[test]))[0])})
result['provisionalReport'] = {'generatorSeed': 73, 'trainValidationTest': [600, 150, 150], 'trainEvents': int(indicator[train].sum()), 'grid': evaluation_grid.tolist(), 'models': fits, 'note': 'Design-time fixture exploration. Final learner program will select on validation only and expose the test result only for frozen chosen models and baseline; these design results are not external performance claims.'}
(directory / 'design-fixtures.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
