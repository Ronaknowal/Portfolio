"""Create complete, executed Survival lesson programs; use the isolated runtime."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import subprocess
import textwrap
import black

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'scratch/survival-tools/Scripts/python.exe'
DEST = ROOT / 'scratch/survival/native'
DEST.mkdir(parents=True, exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
examples = {}
selected = {name for name in os.environ.get('SURVIVAL_EXAMPLES', '').split(',') if name}
existing_path = ROOT / 'src/learn/data/survival-examples.js'
existing = json.loads(existing_path.read_text(encoding='utf8').split('export const survivalExamples = ', 1)[1].strip().removesuffix(';')) if selected else {}
executed = []


def add(key, title, question, code):
    code = textwrap.dedent(code).strip() + '\n'
    formatted = black.format_str(code, mode=black.Mode(line_length=88))
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(formatted))
    if selected and key not in selected:
        assert existing[key]['code'] == formatted.rstrip(), f'Unselected changed code: {key}'
        assert existing[key]['title'] == title and existing[key]['question'] == question
        examples[key] = existing[key]
        return
    path = DEST / f'{key}.py'
    path.write_text(formatted, encoding='utf8')
    result = subprocess.run([str(PYTHON), '-X', 'utf8', '-I', str(path)], capture_output=True,
                            text=True, encoding='utf8', timeout=150)
    if result.returncode:
        raise RuntimeError(f'{key}\n{result.stdout}\n{result.stderr}')
    examples[key] = dict(title=title, question=question, code=formatted.rstrip(),
                         expected=result.stdout.rstrip(), language='python')
    executed.append(key)


PUMPS = '''
import numpy as np
times = np.array([2, 3, 4, 4, 6, 7, 8, 9], dtype=float)
events = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
'''
SYNTHETIC = '''
import numpy as np
# Preserve the original lesson's RandomState stream, not merely its seed number.
rng = np.random.RandomState(42)
x = rng.binomial(1, 0.5, 200)
true_beta = -0.7
lifetime = rng.exponential(1 / (0.1 * np.exp(true_beta * x)))
followup = rng.uniform(5, 20, 200)
time = np.minimum(lifetime, followup)
event = lifetime <= followup
'''

add('observation', 'Encode what was observed, without inventing a failure',
    'Pump B was still operating at day 3. Which stored field says that its failure time is not known?', PUMPS + '''
names = list("ABCDEFGH")
for name, observed, failed in zip(names, times, events):
    statement = f"T = {observed:g}" if failed else f"T > {observed:g}"
    print(name, f"Y={observed:g}", f"event={int(failed)}", statement)
print("observed-time mean:", times.mean())
print("completed-only mean:", times[events].mean())
print("Neither is an ordinary estimate of unrestricted mean lifetime here.")
cutoff = 5.0
print("administrative cutoff 5:", np.minimum(times, cutoff).tolist())
print("events after cutoff are now censored:", (events & (times <= cutoff)).astype(int).tolist())
''')

KM_HELPER = '''
from fractions import Fraction
def km_table(times, events):
    survival = Fraction(1)
    greenwood = Fraction(0)
    rows = []
    for at in sorted(set(times)):
        risk = sum(value >= at for value in times)
        failures = sum(value == at and failed for value, failed in zip(times, events))
        censored = sum(value == at and not failed for value, failed in zip(times, events))
        survival *= Fraction(risk - failures, risk)
        if failures == risk:
            greenwood = None
        elif greenwood is not None:
            greenwood += Fraction(failures, risk * (risk - failures))
        rows.append((at, risk, failures, censored, survival, greenwood))
    return rows

def step_value(rows, at):
    if at < 0 or at > rows[-1][0]:
        raise ValueError("Requested time is outside the observed support.")
    return next((row[4] for row in reversed(rows) if row[0] <= at), Fraction(1))

def restricted_mean(rows, horizon):
    if not 0 <= horizon <= rows[-1][0]:
        raise ValueError("Choose a supported horizon.")
    left, height, area = 0, Fraction(1), Fraction(0)
    for row in rows:
        right = min(row[0], horizon)
        area += (right - left) * height
        if right == horizon:
            break
        left, height = right, row[4]
    return area
'''
add('kaplanMeier', 'Build exact steps and calculate the supported area',
    'At day 4, does the pump censored at day 4 belong in the denominator? Calculate the step before running.', KM_HELPER + '''
from math import exp, log, sqrt
times = [2, 3, 4, 4, 6, 7, 8, 9]
events = [True, False, True, False, True, False, True, False]
rows = km_table(times, events)
print("day risk failures censored survival Greenwood")
for row in rows:
    print(*row)
survival, greenwood = rows[2][4:]
z = 1.959963984540054
transformed = log(-log(float(survival)))
se = sqrt(float(greenwood)) / abs(log(float(survival)))
limits = [exp(-exp(transformed + z * se)), exp(-exp(transformed - z * se))]
print("approximate pointwise 95% limits at 4:", [round(v, 6) for v in limits])
print("S(5), held constant after 4:", step_value(rows, 5))
print("median:", next((r[0] for r in rows if r[4] <= Fraction(1, 2)), "not reached"))
print("RMST through day 9:", restricted_mean(rows, 9))
''')

add('timeouts', 'A timeout changes the question to a restricted mean',
    'Two search methods both finish by the same timeout half the time. Can their expected compute use still differ?', KM_HELPER + '''
methods = {
    "A": ([1, 2, 5, 5], [True, True, False, False]),
    "B": ([3, 4, 5, 5], [True, True, False, False]),
}
for name, (times, events) in methods.items():
    rows = km_table(times, events)
    print(name, "success by timeout:", 1 - step_value(rows, 5),
          "restricted compute:", restricted_mean(rows, 5))
print("An unfinished run may finish at 6 or at 600; these observations cannot decide.")
from math import exp
for shape in [1, 2]:
    scale, age, extra = 12, 4, 3
    integrated = ((age + extra) / scale) ** shape - (age / scale) ** shape
    print("shape", shape, "failure in next 3 days given running at 4:", round(1-exp(-integrated), 6))
''')

add('exponential', 'Every observed running day contributes exposure',
    'If the censored pumps are observed for one more event-free day each, should the fitted constant rate increase or decrease?', PUMPS + '''
from scipy.optimize import minimize_scalar
failures = int(events.sum())
exposure = float(times.sum())
rate = failures / exposure
def negative_log_likelihood(log_rate):
    return np.exp(log_rate) * exposure - failures * log_rate
fit = minimize_scalar(negative_log_likelihood, bracket=(-4, -1))
print("events / exposure:", failures, exposure)
print("rate per day:", round(rate, 8))
print("numerical likelihood fit:", round(np.exp(fit.x), 8))
print("rate with four extra censored running days:", round(failures / (exposure + 4), 8))
print("S(10):", round(np.exp(-rate * 10), 6))
print("rate per hour:", round(rate / 24, 8))
print("No events: likelihood increases as the positive rate tends to zero.")
''')

add('syntheticComparison', 'Reproduce the original experiment with true KM steps',
    'The generator gives one group a lower hazard. Which groups should have larger survival at days 5 and 10?', SYNTHETIC + '''
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
print("subjects / events / censored:", len(time), int(event.sum()), int((~event).sum()))
for group in [0, 1]:
    member = x == group
    fit = KaplanMeierFitter().fit(time[member], event[member])
    print("group", group, "S(5), S(10):", np.round(fit.predict([5, 10]).to_numpy(), 6).tolist())
test = logrank_test(time[x == 0], time[x == 1], event[x == 0], event[x == 1])
print("log-rank chi-square / asymptotic p:", round(test.test_statistic, 6), f"{test.p_value:.6g}")
print("These are simulated group associations under the stated generator.")
''')

COX_HELPER = '''
from scipy.special import logsumexp
def cox_terms(beta, time, event, x, ties="efron"):
    log_likelihood = score = information = 0.0
    for at in np.unique(time[event]):
        risk = time >= at
        failed = (time == at) & event
        count = int(failed.sum())
        shift = np.max(beta * x[risk])
        weight = np.exp(beta * x - shift)
        risk_moments = [np.sum(weight[risk] * x[risk]**power) for power in range(3)]
        event_moments = [np.sum(weight[failed] * x[failed]**power) for power in range(3)]
        log_likelihood += beta * x[failed].sum()
        score += x[failed].sum()
        for step in range(count):
            fraction = step / count if ties == "efron" else 0
            total, first, second = [r - fraction*e for r, e in zip(risk_moments, event_moments)]
            mean = first / total
            log_likelihood -= shift + np.log(total)
            score -= mean
            information += second / total - mean**2
    return np.array([log_likelihood, score, information])

def fit_cox(time, event, x, max_steps=50, tolerance=1e-8):
    beta = 0.0
    for iteration in range(max_steps):
        objective, score, information = cox_terms(beta, time, event, x)
        if abs(score) <= tolerance:
            return beta, iteration, "score tolerance reached"
        if information <= 1e-12:
            return beta, iteration, "insufficient curvature; finite fit not certified"
        direction = np.clip(score / information, -2, 2)
        step = 1.0
        while step > 2**-30:
            trial = beta + step * direction
            if cox_terms(trial, time, event, x)[0] >= objective + 1e-4 * step * score * direction:
                beta = trial
                break
            step /= 2
        else:
            return beta, iteration, "line search exhausted"
    return beta, max_steps, "iteration budget exhausted"
'''
add('coxFit', 'Fit the original binary-feature Cox model from risk sets',
    'Why can an individual censored late affect several earlier denominators even though it never supplies an event numerator?', SYNTHETIC + COX_HELPER + '''
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
beta, steps, status = fit_cox(time, event, x)
library = CoxPHSurvivalAnalysis(ties="efron").fit(x[:, None], Surv.from_arrays(event, time))
print("beta / HR:", round(beta, 6), round(np.exp(beta), 6))
print("iterations / status:", steps, status)
print("log likelihood / score / information:", np.round(cox_terms(beta, time, event, x), 8).tolist())
print("library beta:", round(library.coef_[0], 6))
flat = cox_terms(0, time, event, np.zeros(len(x)))
print("constant-feature information:", flat[2])
print("A finite score tolerance alone does not rule out separation; inspect information and coefficient growth.")
''')

add('ties', 'Separate tie approximations from baseline estimation',
    'At beta=log(2), calculate the risk-weight sum at day 2. Why does Efron subtract 1.5 in its second denominator?', '''
import numpy as np
from itertools import combinations
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
import pandas as pd
x = np.array([-1, 0, 1, -1, 0, 1], dtype=float)
time = np.array([1, 2, 2, 3, 4, 5], dtype=float)
event = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
weight = np.exp(np.log(2) * x[time >= 2])
print("day 2 risk weights:", weight.tolist())
print("Breslow / Efron contributions:", round(2/weight.sum()**2, 8), round(2/(weight.sum()*(weight.sum()-1.5)), 8))
print("discrete unordered subset probability:", round(2/sum(a*b for a,b in combinations(weight, 2)), 8))
frame = pd.DataFrame({"x": x, "time": time, "event": event})
fit = CoxPHFitter(baseline_estimation_method="breslow").fit(frame, "time", "event")
other = CoxPHSurvivalAnalysis(ties="efron").fit(x[:,None], Surv.from_arrays(event, time))
print("Efron fitted beta, lifelines / sksurv:", round(fit.params_["x"], 6), round(other.coef_[0], 6))
beta = fit.params_["x"]
center = frame.x.mean()
baseline = 0.0
for at in sorted(set(time[event])):
    baseline += np.sum((time == at) & event) / np.exp(beta * (x[time >= at] - center)).sum()
    actual = fit.baseline_cumulative_hazard_.loc[at].iloc[0]
    assert np.isclose(baseline, actual)
print("centered baseline cumulative hazard at 4:", round(baseline, 6))
print("predicted S(4) for x=-1,0,1:", np.round(fit.predict_survival_function(pd.DataFrame({"x": [-1.,0.,1.]}), times=[4]).iloc[0].to_numpy(), 6).tolist())
''')

add('rossi', 'Run the retained historical example with actual diagnostic output',
    'The financial-aid coefficient is negative. Does that by itself identify a causal effect, or prove proportional hazards for every feature?', '''
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_rossi
from lifelines.statistics import proportional_hazard_test
data = load_rossi()
fit = CoxPHFitter().fit(data, "week", "arrest")
km = KaplanMeierFitter().fit(data.week, data.arrest)
print("rows / events:", len(data), int(data.arrest.sum()))
print("KM S(26), median:", round(float(km.predict(26)), 6), km.median_survival_time_)
print(fit.summary[["coef", "exp(coef)", "coef lower 95%", "coef upper 95%"]].round(6).to_string())
test = proportional_hazard_test(fit, data, time_transform="rank")
print("rank-time PH diagnostic:")
print(test.summary[["test_statistic", "p"]].round(6).to_string())
print("first event record's Schoenfeld residual:")
print(fit.compute_residuals(data, kind="schoenfeld").sort_index().iloc[0].round(6).to_dict())
print("training concordance:", round(fit.concordance_index_, 6))
print("Historical randomized-aid dataset; this fit alone is not a complete causal analysis or held-out prediction assessment.")
''')

add('aft', 'Translate an accelerated time scale into a survival prediction',
    'With a common Weibull shape of 2, doubling the time scale multiplies the hazard by what?', '''
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from lifelines.datasets import load_rossi
shape, scale, time_ratio = 2, 12, 2
print("S at scale:", round(np.exp(-1), 6), "failed:", round(1-np.exp(-1), 6))
print("baseline / doubled median:", round(scale*np.log(2)**(1/shape), 6), round(time_ratio*scale*np.log(2)**(1/shape), 6))
print("hazard ratio:", time_ratio**(-shape))
data = load_rossi()
fit = WeibullAFTFitter().fit(data, "week", "arrest")
rho = np.exp(fit.params_.loc[("rho_", "Intercept")])
coefficient = fit.params_.loc[("lambda_", "fin")]
print("Rossi shape / fin time ratio:", round(rho, 6), round(np.exp(coefficient), 6))
print("training concordance:", round(fit.concordance_index_, 6))
profiles = data.drop(columns=["week", "arrest"]).iloc[[0]].copy()
print("first profile S(26):", round(float(fit.predict_survival_function(profiles, times=[26]).iloc[0,0]), 6))
print("If shape also depends on features, one common time multiplier no longer describes the change.")
''')

add('metrics', 'Inspect comparable pairs and inverse observation weights',
    'Which tied-time pair is comparable here? Does changing a survival probability affect the ordering of the supplied risk scores?', '''
from itertools import combinations, product
import numpy as np
from sksurv.metrics import concordance_index_censored
times = np.array([1, 2, 2, 4], dtype=float)
events = np.array([True, False, True, True])
scores = np.array([3, 2, 2, 0], dtype=float)
for i,j in combinations(range(4), 2):
    early, late = sorted([i,j], key=lambda k: (times[k], not events[k]))
    comparable = events[early] and (times[early] < times[late] or not events[late])
    credit = (0.5 if scores[early] == scores[late] else float(scores[early] > scores[late])) if comparable else None
    print((i,j), "credit:", credit)
print("library C, concordant, discordant, risk ties, time ties:", concordance_index_censored(events, times, scores))
prediction, horizon = .6, 5
weighted = complete = known_mass = naive = full = 0.0
for (lifetime, mass), (censor, censor_mass) in product([(2,.4),(8,.6)], [(3,.5),(10,.5)]):
    probability = mass*censor_mass
    observed, event = min(lifetime,censor), lifetime <= censor
    loss = (float(lifetime > horizon)-prediction)**2
    known = event or observed > horizon
    observation_probability = 1 if lifetime == 2 else .5
    weighted += probability * (loss/observation_probability if known else 0)
    complete += probability*loss if known else 0
    known_mass += probability if known else 0
    full += probability*loss
    naive += probability*(float(observed > horizon)-prediction)**2
print("full / known-G IPCW:", round(full, 6), round(weighted, 6))
print("complete-case / censor-as-failure:", round(complete/known_mass, 6), round(naive, 6))
''')

add('survivalTree', 'See what a survival tree stores in each leaf',
    'After routing by temperature, can two pumps in the same leaf receive different survival curves from this one tree?', '''
import numpy as np
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
temperature = np.array([0,1,2,3,4,5,6,7], dtype=float)
time = np.array([8,9,7,6,4,3,2,5], dtype=float)
event = np.array([1,0,1,0,1,1,1,0], dtype=bool)
target = Surv.from_arrays(event,time)
for threshold in [2.5,3.5,4.5]:
    low = temperature <= threshold
    test = logrank_test(time[low],time[~low],event[low],event[~low])
    print("threshold / log-rank chi-square:", threshold, round(test.test_statistic,6))
tree = SurvivalTree(max_depth=1,min_samples_leaf=2,random_state=4).fit(temperature[:,None],target)
threshold = tree.tree_.threshold[0]
print("chosen threshold:", threshold)
for profile in [1.,6.]:
    survival = tree.predict_survival_function([[profile]])[0]
    print("profile", profile, "S(4), S(5):", np.round(survival([4,5]),6).tolist())
forest = RandomSurvivalForest(n_estimators=20,max_depth=2,min_samples_leaf=2,random_state=4,n_jobs=1).fit(temperature[:,None],target)
print("forest S(4) for both profiles:", [round(fn(4),6) for fn in forest.predict_survival_function([[1.],[6.]])])
print("This tiny training illustration is not a held-out model comparison.")
''')

add('competing', 'Allocate first-event probability from one common pool',
    'A second failure cause removes pumps from the event-free pool. Why does censoring it produce a different answer from actual cause-1 incidence?', '''
from fractions import Fraction
import numpy as np
from sksurv.nonparametric import cumulative_incidence_competing_risks
from lifelines import KaplanMeierFitter
times = np.array([1,2,3,4,5,6], dtype=float)
status = np.array([2,1,0,2,1,0])
survival, first, second = Fraction(1), Fraction(0), Fraction(0)
for at in sorted(set(times)):
    risk = int(np.sum(times >= at))
    d1 = int(np.sum((times == at)&(status == 1)))
    d2 = int(np.sum((times == at)&(status == 2)))
    first += survival*Fraction(d1,risk)
    second += survival*Fraction(d2,risk)
    survival *= Fraction(risk-d1-d2,risk)
    assert first+second+survival == 1
    print("day / event-free / cause1 / cause2:", int(at),survival,first,second)
grid,cif = cumulative_incidence_competing_risks(status,times)
print("library final CIF:", np.round(cif[1:,-1],6).tolist())
net = KaplanMeierFitter().fit(times,status == 1)
print("naive one-minus-KM cause1:", round(1-float(net.predict(6)),6))
hazard1,hazard2,horizon = .1,.3,5
actual = hazard1/(hazard1+hazard2)*(-np.expm1(-(hazard1+hazard2)*horizon))
print("constant-hazard actual CIF / net quantity:", round(actual,6), round(-np.expm1(-hazard1*horizon),6))
''')

add('history', 'Keep a subject on one clock and use only known history',
    'The maintenance indicator changes at day 3. Which value belongs to the interval ending in a day-5 failure?', '''
import pandas as pd
from lifelines import KaplanMeierFitter
entry = [0,1,2,0]
exit_time = [3,4,5,6]
event = [True,False,True,False]
fit = KaplanMeierFitter().fit(exit_time,event,entry=entry)
print("delayed-entry S(3), S(5):", [round(float(fit.predict(t)),6) for t in [3,5]])
histories = pd.DataFrame([
    {"id":"A","start":0.,"stop":3.,"maintained":0,"event":0},
    {"id":"A","start":3.,"stop":5.,"maintained":1,"event":1},
    {"id":"B","start":1.,"stop":6.,"maintained":0,"event":0},
])
print(histories.to_string(index=False))
for subject, rows in histories.groupby("id",sort=False):
    rows = rows.sort_values("start")
    assert (rows.start < rows.stop).all()
    assert all(rows.start.iloc[k] >= rows.stop.iloc[k-1] for k in range(1,len(rows)))
    assert rows.event.iloc[:-1].sum() == 0
for time in [3.,5.]:
    risk = histories[(histories.start < time)&(time <= histories.stop)]
    print("risk set at", time, risk[["id","maintained","event"]].to_dict("records"))
print("All rows from A stay in one split. A future 'ever maintained' flag is not a baseline feature.")
''')

add('discrete', 'Turn a first-event sequence into observed conditional trials',
    'A subject is censored after interval 2. Should there be an invented third no-event row?', '''
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(17)
features = rng.binomial(1,.5,500)
intercepts = np.array([-2.,-1.5,-1.,-.5])
subject_rows, interval_rows, x_rows, labels = [],[],[],[]
for subject,x in enumerate(features):
    observed_intervals = int(rng.choice([2,4],p=[.3,.7]))
    for interval in range(observed_intervals):
        failed = int(rng.random() < expit(intercepts[interval] + .8*x))
        subject_rows.append(subject); interval_rows.append(interval); x_rows.append(x); labels.append(failed)
        if failed:
            break
subject_rows = np.array(subject_rows)
design = np.column_stack([np.eye(4)[interval_rows],x_rows])
training_rows = subject_rows < 400
fit = LogisticRegression(fit_intercept=False,C=100,max_iter=1000).fit(design[training_rows],np.array(labels)[training_rows])
print("subjects / person-period rows:",len(features),len(labels))
print("fitted interval intercepts and x coefficient:",np.round(fit.coef_[0],5).tolist())
for x in [0,1]:
    probability = fit.predict_proba(np.column_stack([np.eye(4),np.full(4,x)]))[:,1]
    print("x",x,"hazards:",np.round(probability,5).tolist(),"survival:",np.round(np.cumprod(1-probability),5).tolist())
q = np.array([.2,.3,.4])
print("event in interval3 likelihood:", round((1-q[0])*(1-q[1])*q[2],6))
print("censored after interval2 likelihood:", round((1-q[0])*(1-q[1]),6))
print("Conditional likelihood factorization does not make one subject's rows independent subjects.")
''')

add('intervalLikelihood', 'Keep an interval-censored event as an interval',
    'A pump is running at day 2 and failed at its next inspection on day 5. Which likelihood contribution uses both facts?', '''
import numpy as np
from scipy.optimize import minimize_scalar
# Bounds (L,R] for failures; R=infinity for right censoring.
bounds = [(0.,2.),(2.,5.),(3.,7.),(4.,np.inf),(6.,np.inf)]
def negative_log_likelihood(log_rate):
    rate = np.exp(log_rate)
    total = 0.0
    for left,right in bounds:
        if np.isinf(right):
            total -= rate*left
        else:
            total += -rate*left + np.log(-np.expm1(-rate*(right-left)))
    return -total
fit = minimize_scalar(negative_log_likelihood,bracket=(-4,-1))
rate = np.exp(fit.x)
print("fitted constant rate:",round(rate,8))
print("P(2<T<=5):",round(np.exp(-2*rate)-np.exp(-5*rate),6))
print("right-censored contribution at6:",round(np.exp(-6*rate),6))
print("Interval inspection is not an exact failure timestamp or delayed study entry.")
''')

REPORT = '''
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

def report(seed=73, shape=1.5, nonlinear=True):
    rng = np.random.default_rng(seed)
    features = np.column_stack([rng.normal(size=900),rng.binomial(1,.5,900),rng.normal(size=900)])
    score = .6*features[:,0]-.7*features[:,1]
    if nonlinear:
        score += .5*(features[:,2]**2-1)
    lifetime = 12*(rng.exponential(size=900)/np.exp(score))**(1/shape)
    followup = rng.uniform(8,25,900)
    time, event = np.minimum(lifetime,followup),lifetime<=followup
    target = Surv.from_arrays(event,time)
    train,validation,test = np.arange(600),np.arange(600,750),np.arange(750,900)
    def cap(indices):
        return Surv.from_arrays(event[indices]&(time[indices]<=12),np.minimum(time[indices],12))
    valid_y,test_y = cap(validation),cap(test)
    grid = np.array([2.,4.,6.,8.,10.])
    censor = CensoringDistributionEstimator().fit(target[train])
    assert time[train].max()>12
    assert np.all(censor.predict_proba(np.r_[grid,valid_y['time'],test_y['time']])>0)
    expanded = np.column_stack([features,features[:,2]**2])
    km = KaplanMeierFitter().fit(time[train],event[train])
    baseline = km.predict(grid).to_numpy()
    def curves(model,matrix,indices):
        return np.array([fn(grid) for fn in model.predict_survival_function(matrix[indices])])
    candidates = [
        ('linear Cox',features,make_pipeline(StandardScaler(),CoxPHSurvivalAnalysis(alpha=.1,ties='efron'))),
        ('quadratic Cox',expanded,make_pipeline(StandardScaler(),CoxPHSurvivalAnalysis(alpha=.1,ties='efron'))),
        ('forest leaf10',features,RandomSurvivalForest(n_estimators=60,min_samples_leaf=10,max_features=1.,random_state=seed,n_jobs=1)),
        ('forest leaf25',features,RandomSurvivalForest(n_estimators=60,min_samples_leaf=25,max_features=1.,random_state=seed,n_jobs=1)),
    ]
    # Pipeline does not forward survival-specific methods; transform explicitly.
    def survival_curves(model,matrix,indices):
        if hasattr(model,'steps'):
            transformed = model[:-1].transform(matrix[indices])
            return np.array([fn(grid) for fn in model[-1].predict_survival_function(transformed)])
        return curves(model,matrix,indices)
    fitted = []
    for name,matrix,model in candidates:
        model.fit(matrix[train],target[train])
        value = integrated_brier_score(target[train],valid_y,survival_curves(model,matrix,validation),grid)
        fitted.append((value,name,matrix,model))
        print('validation IBS',name,round(value,6))
    value,name,matrix,model = min(fitted,key=lambda row:row[0])
    predicted = survival_curves(model,matrix,test)
    baseline_predictions = np.tile(baseline,(len(test),1))
    risk = model.predict(matrix[test])
    print('selected:',name)
    print('test subjects/events:',len(test),int(test_y['event'].sum()))
    print('minimum training censor survival through12:',round(float(censor.predict_proba([12])[0]),6))
    print('test IBS model/KM:',round(integrated_brier_score(target[train],test_y,predicted,grid),6),round(integrated_brier_score(target[train],test_y,baseline_predictions,grid),6))
    print('test Harrell C / Uno C truncated at10:',round(concordance_index_censored(test_y['event'],test_y['time'],risk)[0],6),round(concordance_index_ipcw(target[train],test_y,risk,tau=10)[0],6))
    auc,mean_auc = cumulative_dynamic_auc(target[train],test_y,risk,grid)
    print('dynamic AUC at2,4,6,8,10:',np.round(auc,6).tolist())
    true_survival = np.exp(-np.exp(score[test,None])*(grid[None,:]/12)**shape)
    print('synthetic oracle probability MSE:',round(float(np.mean((predicted-true_survival)**2)),6))
    print('This is one generated split, not an industrial or clinical benchmark.')
    return {'selected':name,'probabilities':predicted,'baseline':baseline_predictions,'test':test_y}
'''
add('reliabilityReport', 'Produce one frozen, supported held-out reliability report',
    'Choose candidates and validation IBS before examining the test split. Does the more flexible candidate necessarily win?', REPORT + '''
result = report()
''')
add('changedReport', 'Repeat the declared protocol on a different mechanism',
    'Before running, predict how removing the quadratic effect and changing the Weibull shape could change selection. Keep the evaluation protocol fixed.', REPORT + '''
result = report(seed=91,shape=2.,nonlinear=False)
''')

output = ROOT / 'src/learn/data/survival-examples.js'
output.write_text('// Complete programs executed in the documented isolated Survival environment.\nexport const survivalExamples = ' + json.dumps(examples,ensure_ascii=False,indent=2) + ';\n',encoding='utf8')
previous_record = json.loads((DEST.parent/'program-execution.json').read_text(encoding='utf8')) if selected else None
record = {'executedAt':datetime.now(timezone.utc).isoformat(),'runtime':str(PYTHON.relative_to(ROOT)), 'executedPrograms':executed,
          'programs':len(examples),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'outputs':{key:value['expected'] for key,value in examples.items()}}
if selected:
    record['unchangedProgramExecution'] = { 'executedAt': previous_record['executedAt'], 'sha256': previous_record['sha256'], 'reusedPrograms': [key for key in examples if key not in selected] }
(DEST.parent/'program-execution.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'programs':len(examples),'executedAt':record['executedAt'],'sha256':record['sha256']},indent=2))
