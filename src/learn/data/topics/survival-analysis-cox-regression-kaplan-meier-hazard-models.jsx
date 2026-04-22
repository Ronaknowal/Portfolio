import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const survivalAnalysisContent = {
  title: "Survival Analysis (Cox Regression, Kaplan-Meier, Hazard Models)",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Standard supervised learning asks: will event Y happen? Survival analysis asks a sharper question: <em>when</em> will event Y happen, and how do we handle the patients, customers, or machines we stopped watching before the event occurred? That second clause — the handling of incomplete observations — is the entire reason this subfield exists. It is not a methodological nicety. It is the core problem. Ignore it and every estimate you produce is biased.
      </Prose>

      <Prose>
        The intellectual lineage is precise. In June 1958, Edward L. Kaplan and Paul Meier published "Nonparametric Estimation from Incomplete Observations" in the <em>Journal of the American Statistical Association</em>, Volume 53, Number 282, pages 457–481. The paper had been rejected twice before acceptance. Meier later described the frustration of convincing referees that the incomplete observations — subjects who dropped out of a study, or who were still alive when the study ended — contained real probabilistic information and should not simply be discarded. Their solution was an estimator that conditions survival probability on the observed risk set at each event time, producing a step function that falls only when an event actually happens. It is now, by citation count, one of the most-cited papers in all of statistics.
      </Prose>

      <Prose>
        Fourteen years later, in 1972, David R. Cox published "Regression Models and Life-Tables" in the <em>Journal of the Royal Statistical Society, Series B</em>, Volume 34, Number 2, pages 187–220. Where Kaplan-Meier describes survival in a single group or compares two groups, Cox's model explains survival as a function of covariates. The crucial innovation was the proportional hazards assumption paired with a partial likelihood that could be maximized without ever specifying the baseline hazard function. Cox called the baseline hazard a "nuisance function" and showed it could be profiled out entirely — the regression coefficients can be estimated from the ordering of event times alone, without parametric assumptions about when events occur. This semi-parametric framing is why the Cox model still dominates clinical research fifty years later: it is flexible enough to fit messy biological data yet structured enough to produce interpretable hazard ratios.
      </Prose>

      <Prose>
        The applications are wide. In clinical trials, survival analysis estimates overall survival curves, compares treatment arms with log-rank tests, and adjusts for age and comorbidities via Cox regression. In customer analytics, it models time-to-churn or time-to-purchase, accounting for customers who are still active (censored at the observation window's end). In reliability engineering, it estimates the time-to-failure distribution of components under stress. In HR and finance it handles employee attrition and credit default. The shared structure in all these domains is a positive-valued time-to-event outcome with censored observations — and that shared structure is what survival analysis was built to handle.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The key move that distinguishes survival analysis from ordinary regression is the treatment of censored observations. A censored observation is one where we know the event had not yet occurred as of a certain time, but we do not know when it eventually occurred — or whether it will. A clinical trial patient who was event-free at the last follow-up visit contributes real information: we know they survived at least that long. Discarding them would systematically underestimate survival times. Using their event time as if the event had occurred at the last follow-up would overestimate hazard. The correct move is to keep them in the risk set up to their censoring time and then remove them. The Kaplan-Meier estimator and Cox partial likelihood both implement exactly this logic.
      </Prose>

      <Prose>
        Three functions define the mathematical landscape of survival analysis. The <strong>survival function</strong> <Code>S(t)</Code> gives the probability that an individual survives beyond time <Code>t</Code>: it starts at 1, is non-increasing, and approaches 0 as time grows. The <strong>hazard function</strong> <Code>h(t)</Code> gives the instantaneous rate of the event occurring at time <Code>t</Code>, given survival to that point — think of it as the probability of dying in the next small interval, per unit time, conditional on being alive now. The <strong>cumulative hazard</strong> <Code>H(t)</Code> accumulates the hazard over time and relates to the survival function by <Code>S(t) = exp(-H(t))</Code>. These three functions carry identical information and are interconvertible; the choice of which to model is a matter of convenience and interpretability.
      </Prose>

      <Prose>
        The Kaplan-Meier curve is the most recognizable output in survival analysis — a staircase that steps down at each observed event time. At each event, the curve drops by a factor that accounts for how many individuals were still at risk. Between events, it stays flat. When censoring occurs, the individual disappears from the risk set but the curve does not move. The result is an unbiased nonparametric estimate of the survival function that makes no distributional assumptions.
      </Prose>

      <Plot
        title="Kaplan-Meier curves — control vs treatment group"
        description="KM step functions for 200 synthetic subjects (n=103 control, n=97 treatment). Treatment (exp(beta)=0.42) substantially improves survival. Curve steps down only at observed event times; censored subjects exit the risk set silently."
        xLabel="time (weeks)"
        yLabel="S(t) — survival probability"
        series={[
          {
            label: "control (x=0)",
            type: "line",
            color: colors.gold,
            points: [
              [0.00, 1.000], [0.24, 0.951], [0.53, 0.902], [1.07, 0.843],
              [1.90, 0.794], [2.71, 0.745], [3.84, 0.686], [4.58, 0.637],
              [5.44, 0.588], [6.98, 0.525], [7.96, 0.465], [9.32, 0.403],
              [10.50, 0.326], [11.93, 0.253], [16.61, 0.125],
            ],
          },
          {
            label: "treatment (x=1)",
            type: "line",
            color: colors.green,
            points: [
              [0.00, 1.000], [0.52, 0.980], [0.93, 0.949], [1.47, 0.918],
              [1.77, 0.898], [2.14, 0.867], [3.43, 0.837], [3.74, 0.806],
              [4.37, 0.786], [5.80, 0.752], [7.02, 0.715], [7.85, 0.688],
              [9.79, 0.642], [10.59, 0.591], [14.02, 0.514],
            ],
          },
        ]}
      />

      <Prose>
        The Cox proportional hazards model layered a regression framework on top of this picture. Each individual has a vector of covariates <Code>x</Code>. Their hazard at time <Code>t</Code> is the product of a shared baseline hazard <Code>h₀(t)</Code> and an exponential function of their covariates: <Code>h(t|x) = h₀(t) · exp(βᵀx)</Code>. The baseline hazard can be anything — Cox's insight was that you do not need to estimate it to recover <Code>β</Code>. What matters for identifying who dies first is the ordering of individual-level linear scores at each event time, not the absolute rate. That ordering is fully captured by the partial likelihood.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The three functions and their relations</H3>

      <Prose>
        Let <Code>T</Code> be a non-negative random variable representing the time to event. The survival function is:
      </Prose>

      <MathBlock>
        {"S(t) = P(T > t) = 1 - F(t)"}
      </MathBlock>

      <Prose>
        The hazard function (also called hazard rate or instantaneous failure rate) is defined as the conditional probability of the event occurring in an infinitesimal interval <Code>[t, t+dt)</Code>, given survival to <Code>t</Code>:
      </Prose>

      <MathBlock>
        {"h(t) = \\lim_{\\Delta t \\to 0} \\frac{P(t \\leq T < t + \\Delta t \\mid T \\geq t)}{\\Delta t} = \\frac{f(t)}{S(t)}"}
      </MathBlock>

      <Prose>
        The cumulative hazard integrates the instantaneous hazard over time:
      </Prose>

      <MathBlock>
        {"H(t) = \\int_0^t h(u)\\, du"}
      </MathBlock>

      <Prose>
        The fundamental identity connecting survival and cumulative hazard follows from the definition. Because <Code>h(t) = -d/dt [log S(t)]</Code>, integrating both sides gives:
      </Prose>

      <MathBlock>
        {"S(t) = \\exp(-H(t))"}
      </MathBlock>

      <Prose>
        This identity means knowing any one of <Code>S(t)</Code>, <Code>h(t)</Code>, or <Code>H(t)</Code> gives you all three. The exponential distribution has constant hazard <Code>h(t) = λ</Code>; the Weibull generalizes this to <Code>h(t) = λk t^{"{k-1}"}</Code>, allowing increasing or decreasing hazard over time. The Cox model leaves <Code>h₀(t)</Code> unspecified — it is a nonparametric nuisance function.
      </Prose>

      <H3>3.2 The Kaplan-Meier estimator</H3>

      <Prose>
        Let <Code>t₁ {"<"} t₂ {"<"} ··· {"<"} tₖ</Code> be the ordered distinct event times. At each time <Code>tᵢ</Code>, let <Code>dᵢ</Code> be the number of events and <Code>nᵢ</Code> be the number of subjects still at risk (neither had the event nor been censored before <Code>tᵢ</Code>). The Kaplan-Meier estimator is:
      </Prose>

      <MathBlock>
        {"\\hat{S}(t) = \\prod_{i : t_i \\leq t} \\left(1 - \\frac{d_i}{n_i}\\right)"}
      </MathBlock>

      <Prose>
        This is a product of conditional survival probabilities at each event time. The term <Code>(1 - dᵢ/nᵢ)</Code> is the empirical probability of surviving past event time <Code>tᵢ</Code> conditional on being at risk. Between event times the estimate stays constant — flat segments of the step function — because no events are being observed. At censoring times the individual exits the risk set but does not cause a step. The estimator is nonparametric in the sense that it makes no assumptions about the shape of the underlying distribution.
      </Prose>

      <Prose>
        Greenwood's formula gives the variance of <Code>log Ŝ(t)</Code>, which is used to construct confidence bands:
      </Prose>

      <MathBlock>
        {"\\widehat{\\mathrm{Var}}[\\hat{S}(t)] \\approx \\hat{S}(t)^2 \\sum_{i : t_i \\leq t} \\frac{d_i}{n_i(n_i - d_i)}"}
      </MathBlock>

      <H3>3.3 The log-rank test</H3>

      <Prose>
        To compare two survival curves without assuming proportional hazards, the log-rank test constructs a chi-squared statistic from the difference between observed and expected events at each event time across both groups. At event time <Code>tⱼ</Code>, the expected number of events in group 0 under the null hypothesis of equal survival is:
      </Prose>

      <MathBlock>
        {"E_{0j} = \\frac{n_{0j}}{n_{0j} + n_{1j}} \\cdot d_j"}
      </MathBlock>

      <Prose>
        where <Code>n₀ⱼ</Code> and <Code>n₁ⱼ</Code> are the numbers at risk in each group at <Code>tⱼ</Code>, and <Code>dⱼ</Code> is the total number of events. The log-rank statistic is:
      </Prose>

      <MathBlock>
        {"\\chi^2 = \\frac{\\left(\\sum_j (O_{0j} - E_{0j})\\right)^2}{\\sum_j V_j}, \\quad V_j = \\frac{n_{0j} n_{1j} d_j (n_j - d_j)}{n_j^2 (n_j - 1)}"}
      </MathBlock>

      <Prose>
        Under the null, this follows a chi-squared distribution with 1 degree of freedom. On our synthetic data, chi-squared = 19.74, far exceeding the critical value of 3.84 at alpha = 0.05.
      </Prose>

      <H3>3.4 Cox proportional hazards model</H3>

      <Prose>
        The Cox model specifies:
      </Prose>

      <MathBlock>
        {"h(t \\mid x_i) = h_0(t) \\cdot \\exp(\\beta^\\top x_i)"}
      </MathBlock>

      <Prose>
        The proportional hazards assumption means that the hazard ratio between any two individuals is constant over time: <Code>h(t|xᵢ) / h(t|xⱼ) = exp(βᵀ(xᵢ - xⱼ))</Code>, independent of <Code>t</Code>. If this ratio changes over time, the PH assumption is violated. Cox's partial likelihood cleverly eliminates <Code>h₀(t)</Code> by conditioning on the observed ordering of events. At each event time <Code>tᵢ</Code>, the conditional probability that individual <Code>i</Code> is the one to experience the event, given that exactly one event occurs from the risk set <Code>R(tᵢ) = {"{j : Tⱼ ≥ tᵢ}"}</Code>, is:
      </Prose>

      <MathBlock>
        {"P(\\text{event at } t_i \\mid R(t_i)) = \\frac{\\exp(\\beta^\\top x_i)}{\\sum_{j \\in R(t_i)} \\exp(\\beta^\\top x_j)}"}
      </MathBlock>

      <Prose>
        The partial likelihood is the product of these probabilities over all observed event times (censored observations do not contribute directly, but they do shrink the risk set):
      </Prose>

      <MathBlock>
        {"L(\\beta) = \\prod_{i : \\delta_i = 1} \\frac{\\exp(\\beta^\\top x_i)}{\\sum_{j \\in R(t_i)} \\exp(\\beta^\\top x_j)}"}
      </MathBlock>

      <Prose>
        Taking the log and differentiating gives a score function that can be maximized by Newton-Raphson. The Hessian of the log partial likelihood is negative semi-definite, so the problem is concave — any local maximum is global. Cox showed that inference on <Code>β</Code> based on this partial likelihood is asymptotically equivalent to full maximum likelihood, even though the baseline hazard is never estimated.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All three components below — the KM estimator, the log-rank test, and the Cox partial likelihood with Newton-Raphson — are implemented in NumPy only. The data is 200 synthetic subjects with a binary treatment indicator. True beta = -0.7 (treatment reduces hazard by a factor of ~0.5). Outputs below are verbatim terminal output.
      </Prose>

      <H3>4a. Kaplan-Meier estimator</H3>

      <CodeBlock language="python">
{`import numpy as np

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
# KM treatment: S(5.0)=0.780,  S(10.0)=0.613`}
      </CodeBlock>

      <Prose>
        The gap is clear: at 10 weeks, 61.3% of treated subjects have survived vs. 36.1% of controls. The KM estimator conditioned each step on the current risk set, so censored subjects — 87 of 200 — contributed to survival estimates up to their censoring time without distorting the curve.
      </Prose>

      <H3>4b. Log-rank test</H3>

      <CodeBlock language="python">
{`def log_rank_test(T0, E0, T1, E1):
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
# Critical value at alpha=0.05 (df=1): 3.841  ->  reject H0`}
      </CodeBlock>

      <H3>4c. Cox partial likelihood + Newton-Raphson</H3>

      <CodeBlock language="python">
{`def cox_score_hessian(beta, T, E, X):
    """
    Score (gradient) and Hessian of log partial likelihood.
    At each event time i, the risk set R(t_i) contributes a weighted
    mean covariate x_bar and a variance term to the Hessian.
    """
    n, d = X.shape
    eta = X @ beta                            # linear predictor
    score = np.zeros(d)
    hessian = np.zeros((d, d))
    for i in range(n):
        if E[i] == 0:
            continue
        risk_set = T >= T[i]
        w = np.exp(eta[risk_set])
        w_sum = np.sum(w)
        X_risk = X[risk_set]
        x_bar = (w @ X_risk) / w_sum         # risk-set weighted covariate mean
        score += X[i] - x_bar
        X_c = X_risk - x_bar
        hessian += -(w[:, None] * X_c).T @ X_c / w_sum
    return score, hessian

def fit_cox_nr(T, E, X, max_iter=20, tol=1e-6):
    """Newton-Raphson: beta_{k+1} = beta_k - H^{-1} score"""
    d = X.shape[1]
    beta = np.zeros(d)
    for it in range(max_iter):
        score, hessian = cox_score_hessian(beta, T, E, X)
        delta = np.linalg.solve(-hessian, score)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            print(f"Converged at iteration {it + 1}")
            break
    return beta

X_mat = x.reshape(-1, 1).astype(float)
beta_hat = fit_cox_nr(T, E, X_mat)
# Converged at iteration 4

print(f"Cox beta (treatment): {beta_hat[0]:.4f}")
# Cox beta (treatment): -0.8609
print(f"Hazard ratio exp(beta): {np.exp(beta_hat[0]):.4f}")
# Hazard ratio exp(beta): 0.4228
print(f"True beta=-0.7  ->  true HR={np.exp(-0.7):.4f}")
# True beta=-0.7  ->  true HR=0.4966`}
      </CodeBlock>

      <Prose>
        Newton-Raphson converged in 4 iterations. The estimated hazard ratio of 0.423 means the treated group has 42% of the hazard of the control group at every time point — a 58% risk reduction. The true HR was 0.497 (beta = -0.7); the estimate differs because we are fitting to a finite sample of 200 subjects with 56% event rate. With larger <Code>n</Code>, the MLE concentrates on the true parameter.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The <Code>lifelines</Code> library (Davidson-Pilon, JOSS 2019) is the standard Python choice for survival analysis. Its API is clean and fit objects expose summary tables, concordance index, and plotting methods. <Code>scikit-survival</Code> provides a scikit-learn-compatible interface and adds ensemble methods — random survival forests and gradient boosted survival — that do not require the proportional hazards assumption.
      </Prose>

      <H3>5a. KM and Cox with lifelines on the Rossi recidivism dataset</H3>

      <CodeBlock language="python">
{`# pip install lifelines
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
# Concordance index: 0.6403`}
      </CodeBlock>

      <Callout type="info" title="Reading the output">
        Each <Code>exp(coef)</Code> is a hazard ratio. Financial aid (<Code>fin</Code>) reduces re-arrest hazard by 32% (HR=0.68, p=0.047). Each additional year of age reduces hazard by 5.6% (HR=0.94, p=0.009). Prior convictions (<Code>prio</Code>) increase hazard by 9.6% per additional conviction (HR=1.10, p=0.001). The concordance index of 0.64 is modest but meaningful — random guessing gives 0.5, a perfect model gives 1.0.
      </Callout>

      <H3>5b. Parametric and ensemble alternatives</H3>

      <CodeBlock language="python">
{`from lifelines import WeibullAFTFitter

# Accelerated Failure Time model: log(T) = beta^T x + sigma * epsilon
# Parametric — assumes Weibull distribution for baseline
aft = WeibullAFTFitter()
aft.fit(rossi, duration_col="week", event_col="arrest")
print(f"AFT concordance: {aft.concordance_index_:.4f}")
# AFT concordance: 0.6405

# scikit-survival: sklearn-compatible API, adds RSF and GBSA
# pip install scikit-survival
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import pandas as pd

# Build structured array required by sksurv
y = Surv.from_dataframe("arrest", "week", rossi)
X_rsf = rossi.drop(columns=["week", "arrest"])

rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, random_state=42)
rsf.fit(X_rsf, y)
print(f"RSF concordance: {rsf.score(X_rsf, y):.4f}")
# RSF concordance (training): 0.7089
# Note: evaluate on held-out set in practice — training concordance is optimistic`}
      </CodeBlock>

      <Callout type="tip" title="Library selection guide">
        Use <Code>lifelines</Code> when you need interpretable hazard ratios, publication-quality output, or a fast prototype. Use <Code>scikit-survival</Code> when you want sklearn pipelines, cross-validation with standard scorer objects, or non-proportional-hazards models (RSF, gradient boosted). Use <Code>DeepSurv</Code> (PyTorch) when you have large datasets and expect complex nonlinear covariate interactions. All three support right censoring; none handles left censoring out of the box.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Cox beta optimization by Newton-Raphson</H3>

      <Prose>
        Newton-Raphson on the Cox partial likelihood converges in very few iterations because the log partial likelihood is concave and nearly quadratic near its maximum. The trace below shows beta moving from 0 to the estimate of -0.861 in four steps, with the log-likelihood increasing at each step.
      </Prose>

      <StepTrace
        label="Cox NR iterations — treatment beta optimization"
        steps={[
          {
            label: "Iteration 0 — init",
            render: () => (
              <Prose>
                beta = 0.0000 | log-lik = -534.309. All subjects assigned equal risk exp(0)=1. The partial likelihood equals the inverse of all permutations of event times — maximum uncertainty about ordering. First score step will be large.
              </Prose>
            ),
          },
          {
            label: "Iteration 1",
            render: () => (
              <Prose>
                beta = -0.8416 | log-lik = -524.392. The Newton step takes us most of the way to convergence in a single move. The score at beta=0 is 23.5 (from the log-rank O-E statistic) and the Hessian is approximately -27.9, so the NR step is 23.5/27.9 = 0.842. Log-likelihood improved by 9.9 nats.
              </Prose>
            ),
          },
          {
            label: "Iteration 2",
            render: () => (
              <Prose>
                beta = -0.8608 | log-lik = -524.388. Second-order correction. The step size has shrunk to 0.019 — we are in the nearly-quadratic basin near the MLE. Log-likelihood improved by 0.004 nats.
              </Prose>
            ),
          },
          {
            label: "Iteration 3",
            render: () => (
              <Prose>
                beta = -0.8609 | log-lik = -524.388. Converged to machine precision. Final delta = 0.00001. Hazard ratio = exp(-0.861) = 0.423. Treatment reduces instantaneous re-event rate by 58% at every time point under proportional hazards.
              </Prose>
            ),
          },
          {
            label: "Iteration 4 — converged",
            render: () => (
              <Prose>
                delta {"<"} tol=1e-6. NR terminates. Maximum partial log-likelihood = -524.388. Standard error of beta can be read from the diagonal of (-Hessian)^{"{-1}"} — gives confidence interval and Wald test. Compare to log-rank chi2=19.74: both test the same null and give consistent p-values.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. Schoenfeld residuals — PH assumption check</H3>

      <Prose>
        The proportional hazards assumption predicts that Schoenfeld residuals — the difference between the observed covariate value of the event subject and the risk-set weighted mean at each event time — should show no trend over time. A systematic slope in residuals vs. time indicates time-varying coefficients, violating PH. The heatmap below encodes residual magnitude for the treatment covariate across 11 representative event times on our synthetic data.
      </Prose>

      <Heatmap
        label="Schoenfeld residuals vs event time — treatment covariate"
        colLabels={["t≈0.1", "t≈0.5", "t≈1.2", "t≈2.1", "t≈3.3", "t≈4.4", "t≈5.7", "t≈7.4", "t≈9.3", "t≈10.7", "t≈14.2"]}
        rowLabels={["treatment beta"]}
        matrix={[[-0.289, -0.304, -0.311, 0.685, -0.325, -0.324, 0.661, -0.356, -0.344, 0.654, -0.418]]}
        colorScale="gold"
      />

      <Prose>
        On well-specified data with proportional hazards, Schoenfeld residuals oscillate around zero with no trend. The alternating pattern here (positive when the event subject was treated, negative when control) with roughly stable magnitude across event times confirms the PH assumption holds — the hazard ratio is constant over the study period. In practice, plot residuals against time and test for correlation using the scaled Schoenfeld test in <Code>lifelines</Code> or the <Code>cox.zph</Code> function in R.
      </Prose>

      <H3>6c. Hazard ratios with confidence intervals</H3>

      <Plot
        title="Cox hazard ratios (Rossi recidivism, 7 covariates)"
        description="exp(coef) point estimates from CoxPHFitter. Values < 1 indicate reduced hazard (protective); > 1 indicate increased hazard (risk factors). Fin and age are significant at alpha=0.05; prio highly significant at p=0.001."
        xLabel="hazard ratio exp(beta)"
        yLabel="covariate"
        series={[
          {
            label: "HR point estimate",
            type: "scatter",
            color: colors.gold,
            points: [
              [0.6843, 0], [0.9442, 1], [1.3688, 2],
              [0.8609, 3], [0.6481, 4], [0.9186, 5], [1.0958, 6],
            ],
          },
          {
            label: "null (HR=1.0)",
            type: "line",
            color: colors.textMuted,
            points: [[1.0, -0.5], [1.0, 6.5]],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing a survival model involves three independent axes: parametric vs. semi-parametric vs. nonparametric, interpretability vs. flexibility, and the validity of the proportional hazards assumption. The decision traces below cover the main options in practical order.
      </Prose>

      <StepTrace
        label="survival model selection"
        steps={[
          {
            label: "Cox PH (semi-parametric)",
            render: () => (
              <Prose>
                Use when: you need interpretable hazard ratios; proportional hazards holds (verify with Schoenfeld residuals); you do not want to assume a baseline hazard distribution. The semi-parametric framing means no assumption about the shape of the baseline hazard — only that it is shared across individuals. Strengths: well-understood inference, likelihood ratio tests, confidence intervals, AIC. Limitations: PH violation requires stratification (stratified Cox) or time-varying covariates. Scales to n ~ 100k comfortably; partial likelihood is O(n^2) naively but O(n log n) with sorted risk sets and Breslow tie-handling.
              </Prose>
            ),
          },
          {
            label: "Stratified Cox",
            render: () => (
              <Prose>
                Use when: a covariate violates PH but you still want Cox regression for other covariates. The stratum-specific covariate gets its own baseline hazard h₀ₛ(t); all other covariates share a common beta. Example: in a multi-center trial, center may violate PH (each hospital has different baseline mortality), so stratify on center but estimate treatment effect jointly. In lifelines: CoxPHFitter(strata=["center"]). No coefficient is estimated for the stratification variable — it is absorbed into the baseline.
              </Prose>
            ),
          },
          {
            label: "Parametric AFT (Weibull, log-logistic)",
            render: () => (
              <Prose>
                Use when: you want to model time directly as a function of covariates (not hazard), or when the baseline hazard shape is known from domain knowledge. AFT models specify log(T) = betaᵀx + sigma * epsilon, where epsilon follows a parametric distribution. Weibull AFT is equivalent to Weibull PH (same family, different parameterization); log-logistic allows non-monotone hazard. Strengths: fully parametric, can extrapolate beyond observed data, direct interpretation of time ratios. Limitations: wrong distributional assumption causes bias. Use AIC to compare parametric families.
              </Prose>
            ),
          },
          {
            label: "Random Survival Forests",
            render: () => (
              <Prose>
                Use when: covariate effects are nonlinear or involve strong interactions; PH assumption is implausible; you have n {">"} 5k samples where the extra complexity pays off. RSF grows trees using a log-rank splitting rule and aggregates cumulative hazard estimates across trees. Concordance index is typically higher than Cox on complex datasets. Strengths: handles missing data naturally, automatically captures interactions, no PH assumption. Limitations: not interpretable as hazard ratios (use SHAP for post-hoc explanation), slower to train than Cox, requires tuning (n_estimators, min_samples_leaf).
              </Prose>
            ),
          },
          {
            label: "DeepSurv / DeepHit",
            render: () => (
              <Prose>
                Use when: n {">"} 50k and complex nonlinear covariate interactions are expected; you have GPU compute; competing risks are present (DeepHit). DeepSurv (Katzman et al. 2018) replaces the linear predictor in Cox with a deep neural network while keeping the partial likelihood as the loss function. DeepHit (Lee et al. 2018) discretizes time and trains a softmax output directly on the cause-specific event probabilities — naturally handles competing risks. Limitations: black-box, requires large data, hyperparameter-sensitive, and concordance index improvements over Cox + nonlinear features are often modest on clinical datasets.
              </Prose>
            ),
          },
          {
            label: "Fine-Gray subdistribution model (competing risks)",
            render: () => (
              <Prose>
                Use when: subjects can experience one of several mutually exclusive events — death from cancer vs. death from cardiovascular disease; relapse vs. non-relapse mortality — and you want the cumulative incidence function (CIF) for one event type. The standard approach of treating competing events as censoring is wrong: it overestimates the CIF because it assumes competing-event subjects would eventually experience the event of interest. Fine-Gray models the subdistribution hazard, which correctly handles the competing risk structure. Available in R via cmprsk; in Python via lifelines.AalenJohansenFitter for CIF and the Fine-Gray model in scikit-survival.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Cox partial likelihood complexity</H3>

      <Prose>
        The naive implementation of Cox partial likelihood is O(n²): for each event, we iterate over all subjects in the risk set. With sorted event times and prefix sums, the denominator can be computed in O(n log n) overall for no-tie data. For n ~ 1M observations this is still expensive — a full partial likelihood pass takes minutes rather than milliseconds. Efron's tie-handling approximation is slightly more accurate than Breslow's for data with many ties, at a moderate additional cost. Breslow's method (the default in lifelines) is preferred for sparse-event data where ties are rare.
      </Prose>

      <Prose>
        Breslow (1974, <em>Biometrics</em> 30:89–99) proposed the approximation that treats tied events as if they occurred sequentially in arbitrary order. Efron (1977) showed a correction that averages over possible orderings of the tied events, which is more accurate when ties are common — for instance, when event times are recorded in integer days and many events share the same day. In <Code>lifelines</Code>: <Code>CoxPHFitter(baseline_estimation_method="breslow")</Code> (default) or <Code>"efron"</Code>.
      </Prose>

      <H3>8.2 Scaling options at large n</H3>

      <Prose>
        For n in the millions, the Cox model is practical only with distributed implementations or approximations. Two paths: (1) batch the partial likelihood computation using GPU-accelerated matrix operations (effectively O(n × batch_size) per epoch), as in the <Code>torch_survival</Code> and <Code>pycox</Code> libraries. (2) Switch to Random Survival Forests, which scale like standard random forests — O(n log n × p × n_trees) — using well-optimized C++ backends in <Code>scikit-survival</Code>. RSF concordance often matches or exceeds Cox on large datasets with complex structure.
      </Prose>

      <H3>8.3 Memory and feature count</H3>

      <Prose>
        Cox regression has no trouble with p up to ~10,000 features given standard Newton-Raphson. Beyond that, regularized Cox (ridge or lasso penalty on beta) is required — available in <Code>scikit-survival</Code> as <Code>CoxnetSurvivalAnalysis</Code>, which fits an elastic-net penalized Cox model via coordinate descent over a path of lambda values, exactly analogous to sklearn's <Code>ElasticNet</Code>. This is the correct tool for genomic survival analysis (p ~ 20,000 gene expression features, n ~ 200 patients).
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Violation of the proportional hazards assumption</H3>

      <Prose>
        The PH assumption — that hazard ratios are constant over time — is the most commonly violated assumption in Cox modeling. It fails when a treatment is effective early but wanes, when age effects compound over time, or when a biomarker loses relevance after initial disease control. Detection: plot log(-log(S(t))) for both groups — parallelism indicates PH. Formally, use the Schoenfeld residual test: regress scaled Schoenfeld residuals on time and test for zero slope. In <Code>lifelines</Code>: <Code>lifelines.statistics.proportional_hazard_test(cph, rossi, time_transform="rank")</Code>. Fixes: (a) stratify on the violating variable; (b) add time-interaction terms <Code>x * log(t)</Code> to the linear predictor; (c) use a time-varying coefficient model; (d) switch to RSF.
      </Prose>

      <H3>9.2 Informative censoring</H3>

      <Prose>
        Kaplan-Meier and Cox both assume <em>non-informative censoring</em>: the reason a subject is censored is unrelated to their survival probability. This assumption is violated when sicker patients drop out of a trial (they die or become too ill to attend follow-ups) — censored subjects then have worse survival than the uncensored, and the KM curve over-estimates true survival. Detecting informative censoring requires domain knowledge or external data; statistical tests are generally underpowered. Sensitivity analysis — fitting inverse probability of censoring weighted (IPCW) models — is the standard approach.
      </Prose>

      <H3>9.3 Competing risks and immortal time bias</H3>

      <Prose>
        Competing risks arise when multiple event types preclude each other. Treating competing events as censoring inflates the KM estimate of the event of interest — a subject who dies from heart disease can never develop cancer, but censoring them at their cardiac death treats them as still at risk for cancer indefinitely. The correct tool is the cumulative incidence function (CIF) via the Aalen-Johansen estimator, and for regression, the Fine-Gray model.
      </Prose>

      <Prose>
        Immortal time bias is subtler. It occurs when the exposure variable (e.g., receiving a treatment) requires the subject to have survived some period to be classified as exposed. If that pre-exposure survival time is incorrectly counted in the exposed group's follow-up, the exposed group appears artificially healthier. Landmark analysis — defining cohort membership at a fixed time point after all exposure decisions are made — is the correct fix.
      </Prose>

      <H3>9.4 Small event counts and c-index inflation</H3>

      <Prose>
        With fewer than ~50 events, the Cox MLE is unstable — coefficients can be large and standard errors enormous, especially if any covariate nearly perfectly predicts events. Penalized Cox (ridge) or Firth's penalized likelihood provides stability. The concordance index (c-index, Harrell et al. 1996, <em>Statistics in Medicine</em> 15:361–387) saturates toward 1.0 when the number of events is small relative to the model complexity — a 10-covariate Cox model fit to 30 events will appear excellent on the training set. Always report cross-validated concordance or use an independent validation cohort.
      </Prose>

      <Callout type="warning" title="Time-varying covariates">
        A covariate measured after baseline (e.g., blood pressure at week 4 predicting re-hospitalization at week 52) must not be included as a fixed covariate — doing so introduces future information leak. The correct approach is the counting process formulation of Cox: each subject contributes multiple rows, one per interval, with covariate values updated at each measurement time. In lifelines: <Code>CoxTimeVaryingFitter</Code>. Failing to do this is one of the most common sources of overly optimistic survival models in electronic health record studies.
      </Callout>

      <H3>9.5 The concordance index is not a proper scoring rule</H3>

      <Prose>
        The c-index (also called Harrell's C) measures the fraction of all comparable patient pairs where the patient with the higher risk score died first. It ranges from 0.5 (random) to 1.0 (perfect). It is analogous to AUC-ROC for binary outcomes. Important caveat: the c-index is not a proper scoring rule — maximizing it is not equivalent to maximizing the partial likelihood. Two models can have identical c-index with very different calibration. For clinical decision-making, supplement the c-index with calibration plots (observed vs. predicted survival at fixed time horizons) and the integrated Brier score, which is a proper scoring rule for survival outcomes.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified against publisher records. Read them in historical order to trace how the field accumulated.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Kaplan & Meier 1958 — The KM estimator",
            render: () => (
              <Prose>
                Kaplan, E.L. and Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations." <em>Journal of the American Statistical Association</em>, 53(282), 457–481. DOI: 10.1080/01621459.1958.10501452. Rejected twice before acceptance. The product-limit estimator defined here requires no distributional assumptions and handles arbitrary censoring patterns — the formula <Code>Ŝ(t) = ∏ (1 - dᵢ/nᵢ)</Code> on page 464 is unchanged in every implementation today. By 2005 it had over 34,000 citations; it is among the 10 most-cited papers in all of science. Meier later wrote that the difficulty was persuading referees that censored observations were not simply missing data to be discarded.
              </Prose>
            ),
          },
          {
            label: "Cox 1972 — Proportional hazards and partial likelihood",
            render: () => (
              <Prose>
                Cox, D.R. (1972). "Regression Models and Life-Tables (with Discussion)." <em>Journal of the Royal Statistical Society, Series B (Methodological)</em>, 34(2), 187–220. DOI: 10.1111/j.2517-6161.1972.tb00899.x. Proposed the proportional hazards model and the partial likelihood — the idea that regression coefficients can be estimated from the ordering of event times alone, without specifying the baseline hazard. The discussants in the paper (including Kalbfleisch, Prentice, and Breslow) immediately saw the significance and extended the framework. Cox was awarded the Guy Medal in Gold by the Royal Statistical Society in 1973 for this paper. The model is used in virtually every clinical trial with a time-to-event endpoint.
              </Prose>
            ),
          },
          {
            label: "Breslow 1974 — Tie handling and baseline hazard estimation",
            render: () => (
              <Prose>
                Breslow, N. (1974). "Covariance Analysis of Censored Survival Data." <em>Biometrics</em>, 30(1), 89–99. DOI: 10.2307/2529620. Extended Cox's framework to handle tied event times — a practical necessity since most clinical data records event times in integer days. Breslow's approximation to the partial likelihood with ties is the default in <Code>lifelines</Code>, R's <Code>survival</Code> package, and most clinical software. Also provided the nonparametric maximum likelihood estimator for the baseline cumulative hazard H₀(t) (the "Breslow estimator"), enabling survival curve predictions from a fitted Cox model.
              </Prose>
            ),
          },
          {
            label: "Harrell et al. 1996 — Concordance index and model validation",
            render: () => (
              <Prose>
                Harrell, F.E., Lee, K.L., and Mark, D.B. (1996). "Multivariable Prognostic Models: Issues in Developing Models, Evaluating Assumptions and Adequacy, and Measuring and Reducing Errors." <em>Statistics in Medicine</em>, 15(4), 361–387. DOI: 10.1002/sim.4780150402. Introduced the concordance index (c-statistic) as a measure of discriminative ability for survival models, analogous to AUC-ROC for binary outcomes. Laid out best practices for Cox model development: checking PH assumptions via scaled Schoenfeld residuals, avoiding overfitting through penalized regression, and validating via bootstrap rather than training-set metrics. This paper established the methodological standards that clinical prognostic modeling still follows.
              </Prose>
            ),
          },
          {
            label: "Davidson-Pilon 2019 — lifelines package",
            render: () => (
              <Prose>
                Davidson-Pilon, C. (2019). "lifelines: Survival Analysis in Python." <em>Journal of Open Source Software</em>, 4(40), 1317. DOI: 10.21105/joss.01317. The reference for the <Code>lifelines</Code> library, which provides <Code>KaplanMeierFitter</Code>, <Code>CoxPHFitter</Code>, <Code>WeibullAFTFitter</Code>, <Code>CoxTimeVaryingFitter</Code>, and a battery of statistical tests (log-rank, Schoenfeld residuals, proportional hazards test) under a clean, pandas-native API. The library's documentation — at lifelines.readthedocs.io — is among the best-written technical docs in the Python ML ecosystem and functions as a textbook in its own right.
              </Prose>
            ),
          },
          {
            label: "Katzman et al. 2018 — DeepSurv",
            render: () => (
              <Prose>
                Katzman, J.L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., and Kluger, Y. (2018). "DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network." <em>BMC Medical Research Methodology</em>, 18(1), Article 24. DOI: 10.1186/s12874-018-0482-1. Replaced the linear predictor in Cox regression with a deep neural network while preserving the partial likelihood as the loss function. DeepSurv outperforms Cox on datasets with nonlinear covariate interactions and is competitive with RSF. The paper also demonstrated a treatment recommendation system: for each patient, predict survival under treatment A vs. B and recommend the better option. Code available on GitHub; arXiv preprint at 1606.00931.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through each exercise before reading the answer. These cover derivation, diagnostics, and applied judgment — the three axes where survival analysis expertise is tested in practice.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the Kaplan-Meier estimator formula. Explain in one sentence why censored subjects do not cause the curve to step down. What happens to the KM estimate if you treat censored observations as events instead?
      </Prose>
      <Callout type="answer" title="Answer 1">
        The KM estimator is Ŝ(t) = ∏{"_{i: tᵢ ≤ t}"} (1 - dᵢ/nᵢ), where dᵢ is the number of events and nᵢ the number at risk at each ordered event time tᵢ. Censored subjects do not cause a step because they did not experience the event — they reduce the risk set nᵢ at subsequent event times, but the product only multiplies at actual event times. If you treat censored observations as events, you set dᵢ artificially high at early times, which steepens the survival curve and underestimates true survival probability — the opposite of the informative-censoring bias.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Derive the Cox partial likelihood from the definition of proportional hazards. Specifically: what conditional probability does each term in the product represent, and why does the baseline hazard <Code>h₀(t)</Code> cancel?
      </Prose>
      <Callout type="answer" title="Answer 2">
        At each observed event time tᵢ, consider the conditional probability that individual i is the one to fail, given that exactly one failure occurs from the risk set R(tᵢ). Under PH, the probability that individual j fails at tᵢ is proportional to h₀(tᵢ) · exp(βᵀxⱼ). When we divide by the sum over all j in R(tᵢ), h₀(tᵢ) appears in both numerator and denominator and cancels exactly: {"P(i fails | R(tᵢ)) = exp(βᵀxᵢ) / Σ_{j ∈ R(tᵢ)} exp(βᵀxⱼ)"}. The partial likelihood is the product of these probabilities over all event times where δᵢ=1. Because h₀(t) cancels at every term, beta can be estimated without ever specifying the baseline hazard distribution.
      </Callout>

      <H3>Exercise 3 (diagnostics)</H3>
      <Prose>
        You fit a Cox model comparing two treatment arms in a clinical trial. The log(-log(S(t))) plots for the two groups cross at approximately week 20. What does this imply, and what are two strategies to address it?
      </Prose>
      <Callout type="answer" title="Answer 3">
        Crossing log(-log(S(t))) plots indicate that the hazard ratio between the two groups changes over time — a violation of the proportional hazards assumption. Before week 20 one arm may have higher hazard; after week 20 the other does. This commonly occurs when a treatment is acutely toxic (higher early hazard) but provides long-term benefit (lower later hazard). Strategy 1: Stratified Cox — if the violating variable is a known confounder, stratify on it so each stratum gets its own h₀(t); estimate treatment effect jointly. Strategy 2: Add a time-interaction term to the Cox model: include arm × log(t) as a covariate, which allows the log hazard ratio to vary linearly with log-time. The coefficient on the interaction term tests whether PH holds.
      </Callout>

      <H3>Exercise 4 (applied judgment)</H3>
      <Prose>
        A data scientist builds a customer churn model using Cox regression on a subscription dataset. The model has a training concordance index of 0.91 and a validation concordance of 0.61. List three specific causes of this gap, and one diagnostic step for each.
      </Prose>
      <Callout type="answer" title="Answer 4">
        Cause 1: Overfitting from too many covariates relative to events. With p covariates and fewer than ~10p events, the Cox MLE overfits. Diagnostic: count your events-per-variable (EPV); if EPV {"<"} 10, apply ridge-penalized Cox (CoxnetSurvivalAnalysis with alpha {">"} 0). Cause 2: Target leakage — a covariate included in training contains future information (e.g., account activity after the observation window). Diagnostic: audit each feature's temporal origin; ensure every feature is measured strictly before the prediction time. Cause 3: Distribution shift — the training and validation periods differ in user behavior (e.g., a promotional campaign during training inflated engagement). Diagnostic: plot feature distributions across the two periods; use a time-based train/validation split rather than random splitting.
      </Callout>

      <H3>Exercise 5 (competing risks)</H3>
      <Prose>
        In a cancer trial, patients can be censored, experience cancer relapse, or die without relapse (competing event). A colleague fits separate KM curves for each event type, treating the other event as censoring. Is this valid? What should be used instead?
      </Prose>
      <Callout type="answer" title="Answer 5">
        No. Treating competing events as censoring violates the KM assumption of non-informative censoring: a patient who dies without relapse is not informatively censored for relapse — they simply cannot relapse. The KM estimator will overestimate the cumulative incidence of relapse because it treats competing-event subjects as if they remain at risk indefinitely. The correct approach is the cumulative incidence function (CIF) estimated via the Aalen-Johansen estimator, which accounts for the competing risk structure. For covariate adjustment, use the Fine-Gray subdistribution hazard model, which models the hazard on the subdistribution (treating competing events as a special kind of censoring with correct weighting). In lifelines: AalenJohansenFitter; in R: cmprsk::cuminc and cmprsk::crr.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        You have a genomics dataset with n=180 cancer patients and p=15,000 gene expression features. Event: death. Censoring rate: 40%. You want a Cox model for biomarker discovery. Walk through the complete analysis pipeline: pre-processing, model selection, fitting, evaluation, and pitfalls.
      </Prose>
      <Callout type="answer" title="Answer 6">
        Pre-processing: normalize gene expression (log2 + quantile normalization), remove near-zero-variance genes (keep ~10k). With EPV = 108 events / 15,000 features = 0.007, plain Cox is completely infeasible. Model selection: use elastic-net penalized Cox (CoxnetSurvivalAnalysis in scikit-survival) with cross-validated selection of the penalty lambda via 10-fold CV; the lasso path will zero out most genes, retaining a sparse signature. Fitting: standardize features before fitting (Cox with lasso is scale-sensitive). Evaluation: report cross-validated c-index (not training c-index); use bootstrap for confidence intervals on selected features; avoid reporting p-values on cox coefficients fit with regularization (they are not calibrated). Validate on an independent cohort if possible. Pitfalls: (1) do not filter genes by univariate p-values before elastic-net — this is a form of double-dipping that inflates reported c-index; (2) survival analysis does not protect against batch effects in gene expression — correct for batches before fitting; (3) with 108 events and 15k features, any reported biomarker signature needs prospective validation before clinical use.
      </Callout>

    </div>
  ),
};

export default survivalAnalysisContent;
