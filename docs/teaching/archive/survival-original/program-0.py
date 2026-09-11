import numpy as np

np.random.seed(42)

def gen_survival_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.binomial(1, 0.5, n)               # binary treatment
    lam = 0.1 * np.exp(-0.7 * x)              # true beta = -0.7
    T_event = rng.exponential(1.0 / lam)      # exponential event times
    C = rng.uniform(5, 20, n)                 # uniform censoring
    T_obs = np.minimum(T_event, C)
    E = (T_event <= C).astype(int)            # 1 = event, 0 = censored
    return T_obs, E, x

T, E, x = gen_survival_data()
print(f"n={len(T)}, events={E.sum()}, censored={len(T) - E.sum()}")
# n=200, events=113, censored=87
print(f"event rate: {E.mean():.2f},  median observed time: {np.median(T):.2f}")
# event rate: 0.56,  median observed time: 7.23

def kaplan_meier(T, E):
    idx = np.argsort(T)
    T_s, E_s = T[idx], E[idx]
    event_times = np.unique(T_s[E_s == 1])
    S = 1.0
    times, surv = [0.0], [1.0]
    for t in event_times:
        n_at_risk = np.sum(T_s >= t)
        d = np.sum((T_s == t) & (E_s == 1))
        S *= (1 - d / n_at_risk)
        times.append(t)
        surv.append(S)
    return np.array(times), np.array(surv)

T0, E0 = T[x == 0], E[x == 0]
T1, E1 = T[x == 1], E[x == 1]
times0, surv0 = kaplan_meier(T0, E0)
times1, surv1 = kaplan_meier(T1, E1)

print(f"KM control:   S(5.0)={np.interp(5.0, times0, surv0):.3f},  S(10.0)={np.interp(10.0, times0, surv0):.3f}")
# KM control:   S(5.0)=0.603,  S(10.0)=0.361
print(f"KM treatment: S(5.0)={np.interp(5.0, times1, surv1):.3f},  S(10.0)={np.interp(10.0, times1, surv1):.3f}")
# KM treatment: S(5.0)=0.780,  S(10.0)=0.613