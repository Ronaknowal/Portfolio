# pip install lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_rossi

rossi = load_rossi()
# Shape: (432, 9)
# Columns: week, arrest, fin, age, race, wexp, mar, paro, prio
# week = time to re-arrest or end of follow-up
# arrest = 1 if arrested, 0 if censored

# --- Kaplan-Meier ---
kmf = KaplanMeierFitter()
kmf.fit(rossi["week"], event_observed=rossi["arrest"])
print(f"KM S(26 weeks): {kmf.survival_function_at_times(26).values[0]:.4f}")
# KM S(26 weeks): 0.8750
print(f"KM median survival time: {kmf.median_survival_time_}")
# KM median survival time: inf     (most subjects were not re-arrested)

# --- Cox PH ---
cph = CoxPHFitter()
cph.fit(rossi, duration_col="week", event_col="arrest")
cph.print_summary(decimals=4, columns=["coef", "exp(coef)", "p"])
# ---
# covariate    coef    exp(coef)     p
# fin        -0.3794    0.6843    0.0474  (financial aid -> lower hazard)
# age        -0.0574    0.9442    0.0090  (older -> lower hazard)
# race        0.3139    1.3688    0.3081
# wexp       -0.1498    0.8609    0.4803
# mar        -0.4337    0.6481    0.2561
# paro       -0.0849    0.9186    0.6646
# prio        0.0915    1.0958    0.0014  (prior convictions -> higher hazard)
# ---
print(f"Concordance index: {cph.concordance_index_:.4f}")
# Concordance index: 0.6403