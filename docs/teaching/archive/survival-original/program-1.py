def log_rank_test(T0, E0, T1, E1):
    """
    Log-rank statistic: chi-squared test of H0: S0(t) = S1(t) for all t.
    At each event time, compare observed vs. expected events under H0.
    """
    all_T = np.concatenate([T0, T1])
    all_E = np.concatenate([E0, E1])
    event_times = np.unique(all_T[all_E == 1])
    O_minus_E = 0.0
    V = 0.0
    for t in event_times:
        n0 = np.sum(T0 >= t)
        n1 = np.sum(T1 >= t)
        d0 = np.sum((T0 == t) & (E0 == 1))
        d1 = np.sum((T1 == t) & (E1 == 1))
        n = n0 + n1
        d = d0 + d1
        if n < 2:
            continue
        e0 = d * n0 / n                           # expected events in group 0
        O_minus_E += (d0 - e0)
        v = d * n0 * n1 * (n - d) / (n**2 * (n - 1))
        V += v
    chi2 = O_minus_E**2 / V
    return chi2, O_minus_E

chi2, ome = log_rank_test(T0, E0, T1, E1)
print(f"Log-rank chi2={chi2:.3f},  O-E={ome:.3f}")
# Log-rank chi2=19.744,  O-E=23.460
print(f"Critical value at alpha=0.05 (df=1): 3.841  ->  {'reject H0' if chi2 > 3.841 else 'fail to reject'}")
# Critical value at alpha=0.05 (df=1): 3.841  ->  reject H0