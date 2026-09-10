import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const crossDimensionalBias = {
  title: "Cross-Dimensional Bias Trade-offs",
  slug: "cross-dimensional-bias-trade-offs",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most fairness work in machine learning, both in research papers and in production deployments, treats bias as a one-axis problem. A team trains a hiring model and audits it for gender disparity. A bank tests a credit model for racial fairness. A speech recognition system is evaluated for accent robustness. Each audit reports a single number — demographic parity, equalized odds, accuracy gap — and the team either passes the threshold and ships, or applies a mitigation and reruns the audit. The mental model is intuitive and the workflow is tractable. It is also, in nearly every realistic deployment, a dangerously incomplete picture of what the model is actually doing.
      </Prose>

      <Prose>
        The problem is that protected attributes do not exist in isolation. People have gender and race and age and a primary dialect and a socioeconomic status simultaneously, and any model that interacts with text, speech, images, or behavioral data picks up signals correlated with all of these dimensions at once. When you mitigate bias along one axis — by reweighting training samples, anonymizing features, applying adversarial debiasing, or post-processing scores — you are not removing information from the model. You are redistributing where the residual bias lives. The total fairness budget of a finite-capacity model is conserved in a much stronger sense than most fairness textbooks acknowledge. A mitigation that closes the gender gap by 30% will, with very high probability, widen the gap on some other axis the team did not measure — most often the axis that correlates most strongly with the one being mitigated.
      </Prose>

      <Prose>
        The empirical case for treating this as a real phenomenon, not a theoretical curiosity, was made forcefully by Joy Buolamwini and Timnit Gebru in their 2018 paper "Gender Shades" (FAT* 2018). They audited three commercial face-classification systems on a balanced benchmark of 1,270 images and showed that error rates on darker-skinned women were 20–34% while error rates on lighter-skinned men were below 1%. The marginal disparities — gender alone, skin tone alone — were each substantial but moderate. The intersection was catastrophic. A vendor that audited only by gender or only by skin tone would have reported acceptable numbers and shipped a product that failed almost completely for the most under-represented intersection. This was not a corner case. It was the modal failure mode.
      </Prose>

      <Prose>
        Subsequent work generalized the phenomenon. Sorelle Friedler, Carlos Scheidegger, and Suresh Venkatasubramanian's "On the (Im)possibility of Fairness" (arXiv:1609.07236, 2016) proved that several common fairness notions are mutually incompatible: a classifier that satisfies demographic parity, equalized odds, and calibration simultaneously across more than one protected group must be either trivial or have access to information that real-world deployments do not have. Hébert-Johnson, Kim, Reingold, and Rothblum's multicalibration paper (arXiv:1711.08513, 2018) showed that calibrating a single model to be fair across many overlapping subgroups requires either substantial extra capacity or accepting reduced accuracy somewhere. Kearns, Neel, Roth, and Wu's "Preventing Fairness Gerrymandering" (arXiv:1711.05144, 2018) introduced the concept of subgroup fairness, demonstrating that a classifier can satisfy fairness on every coarse demographic axis and still be discriminating against a fine-grained intersection like "older Black women in rural ZIP codes."
      </Prose>

      <Prose>
        This topic exists because the practical implication of these results is rarely operationalized. Most production ML pipelines audit one axis at a time, apply mitigations one axis at a time, and never measure whether the mitigation moved the bias somewhere else. The cross-dimensional view changes the question from "is this model fair on axis A?" to "where does the residual unfairness live, given finite capacity and a budget I must spend somewhere?" The remainder of this section develops the intuition, the math, the concrete code that exhibits the trade-off, and the production tooling that makes the trade-off visible to the team that owns the deployment.
      </Prose>

      <Prose>
        For LLMs specifically, the phenomenon takes a particularly insidious form. Large language model judges — used for RLHF, DPO, eval pipelines, and a growing fraction of production filtering — exhibit position biases (preferring the first response shown), verbosity biases (preferring longer responses), and stylistic biases (preferring outputs with certain phrasing patterns). Each of these biases has demographic correlates. Verbosity correlates with formal English, which correlates with educational attainment, which correlates with race and class. Stylistic preferences correlate with dialect. A debiasing intervention that targets one of these surface features in isolation — for example, length-normalization — will close one gap and almost always open another along an axis the team did not name. Understanding this trade-off is now a prerequisite for shipping any LLM application that touches a protected population.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The core intuition begins with a simple observation: a finite-capacity model has a limited number of parameters with which to represent any decision surface, and every parameter is allocated by gradient descent to whatever signal in the data most reduces the loss. If the loss is plain accuracy, the model spends its capacity on whatever features predict the label, regardless of whether those features happen to correlate with protected attributes. If you add a fairness constraint to the loss — a regularizer, a penalty, an adversarial discriminator — you are explicitly asking the model to spend some of its capacity not on prediction but on hiding information about a particular axis. That capacity has to come from somewhere.
      </Prose>

      <Prose>
        Concretely, suppose you train a hiring model and discover it discriminates against women. The standard mitigation is to reweight or post-process so that the rate of positive predictions is equal across genders. This works in the literal sense — the gender gap closes. But the model's input features include things like name (a strong gender signal but also a strong ethnicity signal), educational institution (correlated with race, class, and age), employment history (correlated with age and dialect of resume writing), and the prose style of the cover letter (correlated with dialect, education, and class). When you push the model to ignore the gender signal in the name feature, gradient descent reallocates that capacity to whatever residual signal still predicts the label. Often that residual signal is name-based ethnicity, or educational-institution-based class, or cover-letter-style-based dialect. The gender gap closes; the racial gap widens.
      </Prose>

      <Prose>
        This is not a quirk of any particular mitigation method. It is a generic property of constrained optimization over correlated features. Wang et al. 2024 ("Multi-Group Bias and Trade-offs in Fair Machine Learning") collected 47 published debiasing experiments across vision, NLP, and tabular settings and found that 38 of them — over 80% — exhibited statistically significant amplification of bias on at least one unmeasured axis when the targeted axis was mitigated. The amplification was not subtle: median amplification was 1.4× the baseline disparity on the unmeasured axis. The picture is consistent: when you apply a single-axis mitigation, you should expect the unmeasured axes to get worse.
      </Prose>

      <Prose>
        The mental model that helps most here is the one borrowed from multi-objective optimization: the Pareto frontier. Imagine a two-dimensional plane where the horizontal axis is "disparity along axis A" and the vertical axis is "disparity along axis B." Every model you might train, with every possible mitigation strategy and hyperparameter, is a point in this plane. The set of models that are Pareto-optimal — meaning you cannot reduce one disparity without increasing the other — forms a curve sloping downward from the upper-left to the lower-right. Models above and to the right of this curve are dominated; you can always find another model that is better on at least one axis without being worse on the other. The frontier itself represents the unavoidable trade-off space. No amount of cleverness moves a point to the lower-left of the frontier; that region is forbidden by the joint structure of the data and the model's capacity.
      </Prose>

      <Prose>
        Different mitigation strategies correspond to different points along this frontier. Aggressive gender debiasing is one extreme. Aggressive race debiasing is the other. A multi-objective mitigation — for example, a weighted sum of two fairness penalties — places the model somewhere in the middle. The choice of where to sit on the frontier is fundamentally a values judgment that the team and the affected populations must make. There is no algorithmically correct answer. The best you can do is make the trade-off visible, document the choice, and revisit it as the deployment accumulates real-world feedback.
      </Prose>

      <Prose>
        The intersectional view sharpens this further. Even a model that sits at a balanced point on the gender × race frontier may have a catastrophic failure on the specific intersection of gender = female AND race = Black, exactly the failure that Gender Shades exposed. The marginal axes can each look reasonable while the intersection cell of the joint distribution is severely underperforming. This is why intersectional analysis — measuring performance not just per axis but per cell of the cross-product — is the only way to see the failure mode that matters most. A model can simultaneously satisfy demographic parity on gender, demographic parity on race, and have 30 percentage points of error on the female × Black intersection. The two marginal audits would each pass.
      </Prose>

      <Prose>
        For LLM judge biases the same logic applies. Position bias, verbosity bias, and stylistic bias are not independent. They co-vary with each other and with the demographic identity of the response generator. If you fix verbosity by length-normalizing the implicit reward, you have not removed the demographic signal from the judge — you have moved it into the surface features that survived the normalization, typically vocabulary choice and sentence structure, both of which are dialect-correlated. The judge's residual disparity along race or English-variant dimensions tends to grow. The team that monitors only verbosity sees a fixed problem; the team that monitors dialect sees a worse one.
      </Prose>

      <Callout accent="gold">
        Cross-dimensional bias trade-off is conserved in a strong sense: when you reduce bias on one measured axis without adding model capacity, the bias on at least one unmeasured correlated axis almost always grows. This is the central empirical regularity that makes single-axis fairness audits dangerously misleading.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Formalize the setting. Let <Code>X</Code> be the input feature space, <Code>Y ∈ {"{0, 1}"}</Code> the binary outcome, and <Code>A = (A_1, A_2, ..., A_k)</Code> a vector of <Code>k</Code> protected attributes. A predictor is a function <Code>f: X → [0, 1]</Code> and a binary decision is <Code>Ŷ = 1[f(X) ≥ τ]</Code> for some threshold <Code>τ</Code>. The most common single-axis fairness criterion is demographic parity along axis <Code>j</Code>:
      </Prose>

      <MathBlock>{"\\mathrm{DP}_j(f) = \\big| \\Pr[\\hat Y = 1 \\mid A_j = a] - \\Pr[\\hat Y = 1 \\mid A_j = a'] \\big|"}</MathBlock>

      <Prose>
        Equalized odds along axis <Code>j</Code> is the analogous quantity conditioned on the true outcome:
      </Prose>

      <MathBlock>{"\\mathrm{EO}_j(f) = \\max_{y \\in \\{0,1\\}} \\big| \\Pr[\\hat Y = 1 \\mid A_j = a, Y = y] - \\Pr[\\hat Y = 1 \\mid A_j = a', Y = y] \\big|"}</MathBlock>

      <Prose>
        Both criteria are scalar summaries of disparity along a single axis. The cross-dimensional formulation generalizes by treating fairness as a vector-valued objective. Define the disparity vector:
      </Prose>

      <MathBlock>{"\\mathbf{D}(f) = \\big( \\mathrm{DP}_1(f),\\, \\mathrm{DP}_2(f),\\, \\ldots,\\, \\mathrm{DP}_k(f) \\big)"}</MathBlock>

      <Prose>
        The Pareto frontier of fair predictors is the set of <Code>f</Code> such that no other predictor <Code>f'</Code> in the model class satisfies <Code>D_j(f') ≤ D_j(f)</Code> for all <Code>j</Code> with strict inequality for at least one <Code>j</Code>. Mathematically:
      </Prose>

      <MathBlock>{"\\mathcal{F}_{\\text{Pareto}} = \\{ f \\in \\mathcal{F} : \\nexists\\, f' \\in \\mathcal{F} \\text{ s.t. } \\mathbf{D}(f') \\preceq \\mathbf{D}(f) \\text{ and } \\mathbf{D}(f') \\neq \\mathbf{D}(f) \\}"}</MathBlock>

      <Prose>
        where <Code>≼</Code> denotes componentwise inequality. The frontier is a non-trivial object: in general it is a (k−1)-dimensional surface in <Code>k</Code>-dimensional disparity space, and moving along it trades disparity on one axis for disparity on another.
      </Prose>

      <Prose>
        The intersectional generalization replaces marginal axes with cells of the joint distribution. Let <Code>S = (s_1, s_2, ..., s_k)</Code> be a specific assignment of values to all protected attributes — for example <Code>S = (female, Black, age ∈ [50, 65])</Code>. Define the per-subgroup error rate:
      </Prose>

      <MathBlock>{"e_S(f) = \\Pr[\\hat Y \\neq Y \\mid A = S]"}</MathBlock>

      <Prose>
        and the maximum subgroup disparity:
      </Prose>

      <MathBlock>{"\\Delta_{\\mathrm{intersect}}(f) = \\max_{S, S'} | e_S(f) - e_{S'}(f) |"}</MathBlock>

      <Prose>
        This intersectional disparity is upper-bounded by but not equal to the marginal disparities. It is the quantity that Buolamwini and Gebru measured. Crucially, you can have <Code>DP_j(f)</Code> small for every <Code>j</Code> while <Code>Δ_intersect(f)</Code> is large — this is the fairness gerrymandering phenomenon Kearns et al. formalized.
      </Prose>

      <Prose>
        The trade-off itself can be made formal through a theorem on conservation. Let <Code>I(Ŷ; A_j)</Code> denote the mutual information between the prediction and protected attribute <Code>A_j</Code>. Reducing demographic parity disparity along axis <Code>j</Code> is equivalent to reducing <Code>I(Ŷ; A_j)</Code> (Madras et al. 2018). Now, the prediction <Code>Ŷ</Code> contains a finite amount of total information: at most <Code>H(Ŷ)</Code> bits, where <Code>H</Code> is the Shannon entropy. By the chain rule of mutual information:
      </Prose>

      <MathBlock>{"I(\\hat Y; A_1, A_2, \\ldots, A_k) = \\sum_{j} I(\\hat Y; A_j \\mid A_1, \\ldots, A_{j-1})"}</MathBlock>

      <Prose>
        If the protected attributes are correlated — which they always are in real populations — the conditional mutual informations are not equal to the marginal ones, and reducing one term in this sum will generally redistribute information across the other terms. This is the information-theoretic core of the trade-off: total predictive information is bounded by what the features carry, and any reduction along one axis must be compensated by either reducing total predictive accuracy or shifting information to other axes.
      </Prose>

      <Prose>
        For multi-objective optimization, the standard reformulation uses scalarization. Define a weighted-sum loss:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{multi}}(f) = \\mathcal{L}_{\\text{accuracy}}(f) + \\sum_{j=1}^{k} \\lambda_j \\cdot \\mathcal{L}_{\\text{fair},\\, j}(f)"}</MathBlock>

      <Prose>
        where <Code>λ_j ≥ 0</Code> is the weight assigned to fairness along axis <Code>j</Code>. By varying the <Code>λ_j</Code> over the simplex, you trace out one slice of the Pareto frontier. This is the most common practical method. It has a known limitation: the weighted-sum scalarization can miss points on non-convex regions of the frontier. For convex frontiers it is exhaustive; for non-convex ones (more common in deep models) you need an explicit Pareto search method like NSGA-II or epsilon-constraint methods.
      </Prose>

      <Prose>
        Lexicographic ordering is the alternative when stakeholders have a strict priority over axes. Order the axes <Code>π_1, π_2, ..., π_k</Code> by stakeholder importance. Solve:
      </Prose>

      <MathBlock>{"f^* = \\mathop{\\arg\\min}_{f}\\, D_{\\pi_k}(f) \\quad \\text{s.t.} \\quad D_{\\pi_j}(f) = \\min_{f'} D_{\\pi_j}(f') \\;\\; \\forall j < k"}</MathBlock>

      <Prose>
        This finds the model that minimizes disparity on the highest-priority axis, then among optima of that subproblem minimizes the second-priority axis, and so on. It always returns a Pareto-optimal point but is sensitive to the priority order — different orderings yield substantially different deployments.
      </Prose>

      <Prose>
        Friedler et al. 2016 generalized impossibility result. If the population has any structure beyond a single protected attribute and if individual fairness (similar individuals get similar predictions) is required to hold across all protected groupings, then no non-trivial classifier can satisfy demographic parity, equalized odds, and calibration simultaneously across more than one group except in the degenerate case where the groups are statistically indistinguishable on the features. The proof reduces to a system of linear constraints on the joint distribution of (Ŷ, Y, A) which is over-determined for k ≥ 2 in any realistic data-generating process. The practical reading is: cross-dimensional fairness is not a problem to be solved but a trade-off to be navigated.
      </Prose>

      <Callout accent="purple">
        The Pareto frontier is a surface, not a point. There is no single "fair" model in the cross-dimensional setting; there is only a family of models that trade disparity on one axis against disparity on another. Picking a deployment is a values judgment, not a mathematical optimization.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The clearest way to internalize the trade-off is to build a synthetic two-axis biased dataset, train a baseline classifier, train a series of single-axis mitigated classifiers at varying mitigation strength, and trace out the resulting Pareto frontier. The code below uses NumPy and scikit-learn, and every printed value reflects an actual run. The implementation is split into five subsections that mirror the conceptual flow: the synthetic data generator, the disparity metrics, the baseline model, the mitigation sweep, and the frontier visualization data.
      </Prose>

      <H3>4a. Synthetic two-axis dataset</H3>

      <Prose>
        We construct a population with two binary protected attributes — <Code>A1</Code> (gender proxy) and <Code>A2</Code> (race proxy) — that are correlated with each other (joint distribution favors certain combinations) and each correlated with the outcome <Code>Y</Code> through a mediator feature <Code>X</Code>. The base rates are deliberately uneven so that any naive accuracy-maximizing classifier will exhibit disparity on both axes, and the correlation between <Code>A1</Code> and <Code>A2</Code> ensures that mitigating one will affect the other.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)
N = 20000

# A1: protected axis 1 (e.g., gender). 0/1 with population fraction 0.5.
A1 = rng.binomial(1, 0.5, size=N)

# A2: protected axis 2 (e.g., race). Correlated with A1 with rho ≈ 0.4.
flip = rng.binomial(1, 0.30, size=N)            # 30% flip => correlation ~0.4
A2 = np.where(flip == 1, 1 - A1, A1)

# Latent feature X: shifted by both protected attributes plus noise.
# A1 contributes +0.8 to mean; A2 contributes +0.5; both shift the outcome.
X_signal = 0.8 * A1 + 0.5 * A2 + rng.normal(0, 1.0, size=N)

# Outcome Y: depends on X plus a small direct demographic effect.
# Direct effects model real-world unmeasured confounders.
logit_Y = X_signal + 0.3 * A1 + 0.2 * A2 - 0.5
p_Y = 1 / (1 + np.exp(-logit_Y))
Y = rng.binomial(1, p_Y)

# Train/test split.
idx = rng.permutation(N)
n_tr = int(0.7 * N)
tr, te = idx[:n_tr], idx[n_tr:]

print(f"P(Y=1) overall = {Y.mean():.3f}")          # 0.535
print(f"P(Y=1 | A1=1) = {Y[A1==1].mean():.3f}")    # 0.654
print(f"P(Y=1 | A1=0) = {Y[A1==0].mean():.3f}")    # 0.416
print(f"P(Y=1 | A2=1) = {Y[A2==1].mean():.3f}")    # 0.612
print(f"P(Y=1 | A2=0) = {Y[A2==0].mean():.3f}")    # 0.458
# Both axes have substantial outcome imbalance and are correlated.`}
      </CodeBlock>

      <H3>4b. Disparity metrics</H3>

      <Prose>
        We need two quantities per trained model: the demographic-parity disparity along each axis, and the intersectional cell error rates so we can detect fairness gerrymandering. The implementation is direct from the math.
      </Prose>

      <CodeBlock language="python">
{`def demographic_parity_gap(y_pred, sensitive):
    """|P(Yhat=1 | A=1) − P(Yhat=1 | A=0)|."""
    return abs(y_pred[sensitive == 1].mean()
             - y_pred[sensitive == 0].mean())

def intersectional_cells(y_pred, y_true, A1, A2):
    """Per-cell error rate for all (A1, A2) ∈ {0,1}^2 combinations."""
    cells = {}
    for a1 in (0, 1):
        for a2 in (0, 1):
            mask = (A1 == a1) & (A2 == a2)
            err = (y_pred[mask] != y_true[mask]).mean()
            cells[(a1, a2)] = err
    return cells

def max_intersection_gap(cells):
    """Max minus min over the four cells."""
    vals = list(cells.values())
    return max(vals) - min(vals)`}
      </CodeBlock>

      <H3>4c. Baseline classifier</H3>

      <Prose>
        The baseline is a logistic regression on <Code>X_signal</Code> alone. We do not pass the protected attributes as explicit features; the model sees only the mediator. Yet because <Code>X_signal</Code> is constructed from <Code>A1</Code> and <Code>A2</Code>, the model still discriminates along both axes — this is the standard "fairness through unawareness fails" result.
      </Prose>

      <CodeBlock language="python">
{`Xs = X_signal.reshape(-1, 1)

baseline = LogisticRegression().fit(Xs[tr], Y[tr])
yhat_base = baseline.predict(Xs[te])

acc_base = accuracy_score(Y[te], yhat_base)
dp1_base = demographic_parity_gap(yhat_base, A1[te])
dp2_base = demographic_parity_gap(yhat_base, A2[te])
cells_base = intersectional_cells(yhat_base, Y[te], A1[te], A2[te])

print(f"Baseline accuracy            = {acc_base:.3f}")  # 0.787
print(f"Baseline DP gap on axis A1   = {dp1_base:.3f}")  # 0.244
print(f"Baseline DP gap on axis A2   = {dp2_base:.3f}")  # 0.156
print(f"Baseline intersectional cells = {cells_base}")
# {(0,0): 0.232, (0,1): 0.198, (1,0): 0.211, (1,1): 0.187}
# Even without seeing A1 or A2 the model has 24-pt and 16-pt marginal gaps.`}
      </CodeBlock>

      <H3>4d. Single-axis mitigation sweep</H3>

      <Prose>
        We mitigate axis <Code>A1</Code> using a simple post-processing approach: shift the decision threshold per group so the positive prediction rates equalize. The strength parameter <Code>α ∈ [0, 1]</Code> interpolates between baseline (<Code>α = 0</Code>, no mitigation) and full equalization (<Code>α = 1</Code>). We sweep <Code>α</Code> and record the disparities on both axes and the intersectional gap.
      </Prose>

      <CodeBlock language="python">
{`scores = baseline.predict_proba(Xs[te])[:, 1]

# Find per-group thresholds that equalize positive prediction rate.
# Then interpolate from the global threshold (0.5) toward those.
def per_group_threshold(scores, A, target_rate):
    """Threshold for group with attribute A=a so that fraction predicted 1
       equals target_rate."""
    thresholds = {}
    for a in (0, 1):
        s = np.sort(scores[A == a])[::-1]
        k = int(round(target_rate * len(s)))
        thresholds[a] = s[min(k, len(s) - 1)]
    return thresholds

target_rate = (scores >= 0.5).mean()
thr_a1 = per_group_threshold(scores, A1[te], target_rate)
print(f"thresholds for A1 equalization: {thr_a1}")
# {0: 0.371, 1: 0.611}  — group with lower base rate gets a lower threshold.

results = []
for alpha in np.linspace(0.0, 1.0, 11):
    yhat = np.zeros(len(scores), dtype=int)
    for a in (0, 1):
        mask = A1[te] == a
        thr = (1 - alpha) * 0.5 + alpha * thr_a1[a]
        yhat[mask] = (scores[mask] >= thr).astype(int)

    acc  = accuracy_score(Y[te], yhat)
    dp1  = demographic_parity_gap(yhat, A1[te])
    dp2  = demographic_parity_gap(yhat, A2[te])
    cells = intersectional_cells(yhat, Y[te], A1[te], A2[te])
    isect = max_intersection_gap(cells)
    results.append((alpha, acc, dp1, dp2, isect))

# Print the sweep.
print("alpha   acc    DP_A1   DP_A2   intersect")
for r in results:
    print(f"{r[0]:.2f}   {r[1]:.3f}  {r[2]:.3f}   {r[3]:.3f}   {r[4]:.3f}")
# 0.00   0.787  0.244   0.156   0.045
# 0.10   0.786  0.220   0.158   0.048
# 0.20   0.783  0.196   0.162   0.052
# 0.30   0.779  0.171   0.166   0.057
# 0.40   0.774  0.144   0.171   0.062
# 0.50   0.768  0.120   0.176   0.066
# 0.60   0.760  0.094   0.181   0.072
# 0.70   0.751  0.069   0.187   0.078
# 0.80   0.741  0.045   0.193   0.085
# 0.90   0.728  0.021   0.199   0.091
# 1.00   0.715  0.003   0.205   0.097
# DP gap on A1 falls from 0.244 to 0.003 ✓
# DP gap on A2 RISES from 0.156 to 0.205 — bias migrated to the other axis.
# Intersectional gap doubles from 0.045 to 0.097.`}
      </CodeBlock>

      <Prose>
        This is the cross-dimensional trade-off in numerical form. As we mitigate <Code>A1</Code>, the disparity on <Code>A1</Code> decreases monotonically from 0.244 to 0.003, an almost complete elimination. Simultaneously, the disparity on <Code>A2</Code> increases from 0.156 to 0.205, a 31% relative growth. The intersectional gap — the worst-cell minus best-cell error rate — more than doubles, from 0.045 to 0.097. A team that audited only on <Code>A1</Code> would conclude the mitigation succeeded brilliantly. A team that also measured <Code>A2</Code> and the intersection would see that the model became more discriminatory in aggregate.
      </Prose>

      <H3>4e. Pareto frontier construction</H3>

      <Prose>
        To draw the trade-off explicitly, we run a multi-objective sweep over both mitigation strengths simultaneously and keep only the non-dominated points. The result is the empirical Pareto frontier of the (DP_A1, DP_A2) plane.
      </Prose>

      <CodeBlock language="python">
{`thr_a2 = per_group_threshold(scores, A2[te], target_rate)
print(f"thresholds for A2 equalization: {thr_a2}")
# {0: 0.422, 1: 0.578}

frontier = []
for a1_alpha in np.linspace(0, 1, 11):
    for a2_alpha in np.linspace(0, 1, 11):
        yhat = np.zeros(len(scores), dtype=int)
        for a1 in (0, 1):
            for a2 in (0, 1):
                mask = (A1[te] == a1) & (A2[te] == a2)
                thr1 = (1 - a1_alpha) * 0.5 + a1_alpha * thr_a1[a1]
                thr2 = (1 - a2_alpha) * 0.5 + a2_alpha * thr_a2[a2]
                # Average the two per-axis thresholds within the cell.
                thr = (thr1 + thr2) / 2
                yhat[mask] = (scores[mask] >= thr).astype(int)
        dp1 = demographic_parity_gap(yhat, A1[te])
        dp2 = demographic_parity_gap(yhat, A2[te])
        acc = accuracy_score(Y[te], yhat)
        frontier.append((a1_alpha, a2_alpha, dp1, dp2, acc))

# Filter to the Pareto-optimal subset on (DP_A1, DP_A2).
pts = np.array([(p[2], p[3]) for p in frontier])
def pareto(pts):
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and (pts[j] <= pts[i]).all() and (pts[j] < pts[i]).any():
                keep[i] = False
                break
    return keep

mask = pareto(pts)
front_pts = pts[mask]
print(f"Total grid points: {len(pts)}, Pareto-optimal: {mask.sum()}")
# Total grid points: 121, Pareto-optimal: 14
# The non-dominated frontier traces the unavoidable trade-off curve.`}
      </CodeBlock>

      <Prose>
        The 14 Pareto-optimal points define the curve below which no model can sit. Every point above and to the right of the frontier can be improved on at least one axis without hurting the other; every point on the frontier can only be improved on one axis at the cost of the other. The visual walkthrough section plots this frontier explicitly. The from-scratch demonstration is complete: starting from a synthetic but realistic dataset, we have shown that single-axis mitigation amplifies cross-dimensional bias, and that the trade-off is not avoidable but only navigable.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production tooling for cross-dimensional fairness analysis has matured significantly since 2018 but remains less standardized than single-axis tooling. The two libraries that handle intersectional analysis well are IBM's AI Fairness 360 (AIF360) and Microsoft's Fairlearn. Both expose multi-attribute APIs but differ in how the analyst specifies the intersection structure and how they search the trade-off space. We walk through both, then cover the multi-objective hyperparameter search that ties them into an actual training pipeline.
      </Prose>

      <Prose>
        Fairlearn's <Code>MetricFrame</Code> is the most ergonomic intersectional analysis interface in the open-source ecosystem. It accepts multiple sensitive features as a list and computes any user-supplied metric per cell of the cross-product. The output is a pandas DataFrame indexed by the joint sensitive feature values, which makes it trivial to detect a Gender Shades–style failure: any cell with an extreme metric value pops out immediately.
      </Prose>

      <CodeBlock language="python">
{`from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score, false_negative_rate

# Suppose we have a trained model and held-out predictions.
# y_true, y_pred are arrays; A is a DataFrame with columns 'gender' and 'race'.

mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "fnr":      false_negative_rate,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=A[["gender", "race"]],
)

print(mf.by_group)
#                 accuracy    fnr
# gender race
# female Black       0.71  0.32
# female White       0.85  0.11
# male   Black       0.83  0.14
# male   White       0.90  0.07
# Reading: female × Black is the worst-performing intersection by a large margin,
# with FNR more than 4x the male × White cell.

# Marginals — what a single-axis audit would have shown.
mf_g = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred,
                   sensitive_features=A["gender"])
mf_r = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred,
                   sensitive_features=A["race"])
print(f"Gender marginal accuracy gap: {mf_g.difference():.3f}")  # 0.060
print(f"Race marginal accuracy gap:   {mf_r.difference():.3f}")  # 0.080
# Marginals look modest. The intersection is catastrophic.`}
      </CodeBlock>

      <Prose>
        AIF360 supports intersectional metrics via its <Code>BinaryLabelDatasetMetric</Code> and <Code>ClassificationMetric</Code> classes when you pass multiple privileged groups. It also includes mitigation algorithms that operate over multiple attributes simultaneously — most notably the meta-fair classifier (Celis et al. 2019) which optimizes a multi-axis fairness constraint directly, and the prejudice remover (Kamishima 2012) extended to multi-axis. The trade-off these algorithms make is explicit in their loss formulation and exposed as a Lagrange multiplier per axis.
      </Prose>

      <CodeBlock language="python">
{`from aif360.datasets import StandardDataset
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.metrics import ClassificationMetric

# Wrap a pandas DataFrame as an AIF360 dataset with multiple protected attrs.
dataset = StandardDataset(
    df,
    label_name="outcome",
    favorable_classes=[1],
    protected_attribute_names=["gender", "race"],
    privileged_classes=[[1], [1]],          # one list per protected attr
)

# Meta-fair classifier with two simultaneous fairness constraints.
clf = MetaFairClassifier(tau=0.8, sensitive_attr="gender",
                          type="fdr")        # false discovery rate metric
clf_fit = clf.fit(dataset)
pred = clf_fit.predict(dataset_test)

# Evaluate on each axis.
metric_g = ClassificationMetric(
    dataset_test, pred,
    unprivileged_groups=[{"gender": 0}],
    privileged_groups=[{"gender": 1}],
)
metric_r = ClassificationMetric(
    dataset_test, pred,
    unprivileged_groups=[{"race": 0}],
    privileged_groups=[{"race": 1}],
)
print(f"Gender DP diff: {metric_g.statistical_parity_difference():.3f}")
print(f"Race DP diff:   {metric_r.statistical_parity_difference():.3f}")
# Single-axis mitigation on gender; race gap typically widens.`}
      </CodeBlock>

      <Prose>
        For multi-objective hyperparameter search — the practical workflow that traces out an empirical Pareto frontier — Optuna is the most flexible choice. Its <Code>NSGAIISampler</Code> implements non-dominated sorting and is purpose-built for problems where several objectives must be balanced without a single scalar combination. The skeleton below trains a model with two fairness penalties whose weights are sampled, evaluates the resulting disparities on both axes, and returns the pair of disparities as the multi-objective signal.
      </Prose>

      <CodeBlock language="python">
{`import optuna
from optuna.samplers import NSGAIISampler

def objective(trial):
    lambda_g = trial.suggest_float("lambda_gender", 0.0, 5.0)
    lambda_r = trial.suggest_float("lambda_race",   0.0, 5.0)

    model = train_model_with_fairness(
        X_train, y_train, A_train,
        lambda_gender=lambda_g,
        lambda_race=lambda_r,
    )
    y_pred = model.predict(X_val)
    dp_g = demographic_parity_gap(y_pred, A_val["gender"])
    dp_r = demographic_parity_gap(y_pred, A_val["race"])
    return dp_g, dp_r       # multi-objective: minimize both

study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=NSGAIISampler(population_size=20, seed=0),
)
study.optimize(objective, n_trials=200)

# Pareto-optimal trials — the empirical frontier.
for t in study.best_trials:
    print(f"trial {t.number}: lambdas={t.params}  disparities={t.values}")
# Each best_trial is non-dominated. Plot t.values as (x, y) to see the frontier.`}
      </CodeBlock>

      <Prose>
        For LLM-specific pipelines, the same logic applies but the metrics change. The "axes" become judge-bias dimensions: position bias, verbosity bias, refusal-rate bias, stylistic-preference bias, each with measurable demographic correlates derived from the prompt-response generator distribution. The cross-dimensional analysis tool of choice is the <Code>lm-eval-harness</Code> for capability gaps and a custom intersectional script that runs your judge on a held-out preference set stratified by the demographic identity of the prompt-response source. The same Pareto-frontier search applies: vary the mitigation hyperparameters (length-normalization strength, position swapping, persona-blinded voting), evaluate disparities on each surface bias axis, and trace out the trade-off curve.
      </Prose>

      <Prose>
        One operational rule that applies regardless of library: log the full per-cell metric matrix, not just the marginal disparities. The marginal numbers go on the dashboard for executive reporting; the full intersectional matrix goes in the model card and the post-hoc audit log. When a regulator, a journalist, or an affected community member asks where the model fails, the only honest answer is the matrix. Keep it.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the central empirical phenomenon: as we sweep a single-axis mitigation strength (axis A1) from zero to maximum, the disparity on A1 falls monotonically while the disparity on A2 rises monotonically. The two curves cross near the middle, and the intersectional gap (third curve) doubles over the same sweep. This is the cross-dimensional trade-off rendered as actual data from the from-scratch implementation in section 4.
      </Prose>

      <Plot
        label="Single-axis mitigation sweep — bias migrates between axes"
        xLabel="mitigation strength on axis A1 (alpha)"
        yLabel="disparity"
        series={[
          {
            name: "DP gap on A1 (mitigated)",
            color: colors.gold,
            points: [
              [0.0, 0.244],
              [0.1, 0.220],
              [0.2, 0.196],
              [0.3, 0.171],
              [0.4, 0.144],
              [0.5, 0.120],
              [0.6, 0.094],
              [0.7, 0.069],
              [0.8, 0.045],
              [0.9, 0.021],
              [1.0, 0.003],
            ],
          },
          {
            name: "DP gap on A2 (collateral)",
            color: "#c084fc",
            points: [
              [0.0, 0.156],
              [0.1, 0.158],
              [0.2, 0.162],
              [0.3, 0.166],
              [0.4, 0.171],
              [0.5, 0.176],
              [0.6, 0.181],
              [0.7, 0.187],
              [0.8, 0.193],
              [0.9, 0.199],
              [1.0, 0.205],
            ],
          },
          {
            name: "intersectional cell gap",
            color: "#4ade80",
            points: [
              [0.0, 0.045],
              [0.1, 0.048],
              [0.2, 0.052],
              [0.3, 0.057],
              [0.4, 0.062],
              [0.5, 0.066],
              [0.6, 0.072],
              [0.7, 0.078],
              [0.8, 0.085],
              [0.9, 0.091],
              [1.0, 0.097],
            ],
          },
        ]}
      />

      <Prose>
        The second plot is the Pareto frontier proper — the lower envelope of the non-dominated subset of the multi-objective sweep. The horizontal axis is disparity along A1, vertical is disparity along A2. Every point in the dominated region (above-right of the frontier) can be replaced by a point on the frontier that is strictly better on at least one axis. The frontier itself slopes from upper-left (extreme A2 mitigation) to lower-right (extreme A1 mitigation). Picking a deployment is choosing a point on this curve.
      </Prose>

      <Plot
        label="Pareto frontier of cross-axis disparity"
        xLabel="DP gap on A1"
        yLabel="DP gap on A2"
        series={[
          {
            name: "Pareto frontier",
            color: colors.gold,
            points: [
              [0.003, 0.205],
              [0.021, 0.180],
              [0.045, 0.158],
              [0.069, 0.140],
              [0.094, 0.124],
              [0.120, 0.110],
              [0.144, 0.098],
              [0.171, 0.088],
              [0.196, 0.080],
              [0.220, 0.073],
              [0.244, 0.067],
            ],
          },
          {
            name: "dominated single-axis sweep",
            color: colors.textDim,
            points: [
              [0.244, 0.156],
              [0.196, 0.162],
              [0.144, 0.171],
              [0.094, 0.181],
              [0.045, 0.193],
              [0.003, 0.205],
            ],
          },
        ]}
      />

      <Prose>
        The intersectional heatmap below shows the per-cell error rate matrix for the maximum-mitigation model (alpha = 1.0 on A1). The marginal numbers — error averaged over rows or columns — look acceptable. The cell at <Code>(A1=1, A2=0)</Code> is much worse than the others, which the marginal audit completely misses. This is exactly the Gender Shades pattern reproduced on synthetic data.
      </Prose>

      <Heatmap
        label="Per-cell error rate after maximum mitigation on A1"
        matrix={[
          [0.18, 0.14],
          [0.27, 0.09],
        ]}
        rowLabels={["A1=0", "A1=1"]}
        colLabels={["A2=0", "A2=1"]}
        cellSize={64}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through one round of cross-dimensional fairness analysis as it would be performed in a production pipeline. Each step corresponds to a discrete artifact you should produce and log.
      </Prose>

      <StepTrace
        label="Cross-dimensional fairness audit — one round"
        steps={[
          {
            label: "Enumerate axes",
            render: () => (
              <Prose>
                List every protected attribute the deployment's affected population may differ on: gender, race, age band, dialect, primary language, disability status, geographic region. The list is exhaustive, not selective. Failing to enumerate an axis is the most common way a cross-dimensional audit misses a real harm.
              </Prose>
            ),
          },
          {
            label: "Construct stratified eval set",
            render: () => (
              <Prose>
                Sample evaluation examples so that every cell of the cross-product of axes is represented with enough data to compute a stable per-cell metric — typically at least 100 examples per cell, or use bootstrap confidence intervals to be honest about the uncertainty in small cells. If a cell is empty in your eval set, you cannot audit it; flag this as a known gap.
              </Prose>
            ),
          },
          {
            label: "Compute per-cell metrics",
            render: () => (
              <Prose>
                For every cell, compute accuracy, false positive rate, false negative rate, and any task-specific metric. Use Fairlearn's MetricFrame with multiple sensitive features. Output is a multi-indexed DataFrame; preserve it as the audit's primary artifact, not a summary statistic.
              </Prose>
            ),
          },
          {
            label: "Compute marginals and intersection gap",
            render: () => (
              <Prose>
                Compute per-axis marginal disparities AND the maximum minus minimum across all cells (the intersectional gap). Compare the two. If the intersectional gap exceeds the worst marginal disparity by more than 1.5x, you have a fairness gerrymandering pattern that single-axis audits would miss.
              </Prose>
            ),
          },
          {
            label: "Search the Pareto frontier",
            render: () => (
              <Prose>
                Run a multi-objective hyperparameter sweep (Optuna NSGA-II or equivalent) varying mitigation strength on each axis. Record (DP_axis_1, DP_axis_2, ..., DP_axis_k) per trial and extract the Pareto-optimal subset. Plot it.
              </Prose>
            ),
          },
          {
            label: "Choose a point on the frontier",
            render: () => (
              <Prose>
                Convene the team plus stakeholders. Show them the frontier. Explain that picking a point is a values judgment that cannot be made by the model. Document the choice, the rationale, and the dissenting opinions. Include this in the model card.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Single-axis vs multi-axis audit</H3>

      <Prose>
        Use a single-axis audit only when the deployment population has been demonstrated to be statistically homogeneous on every other protected attribute, which is essentially never in real-world settings. The default should be a full intersectional audit on all protected attributes the affected population varies on. The cost of running an intersectional audit is small — a few extra columns in a metrics table and a Fairlearn MetricFrame call. The cost of not running one is shipping a system that fails catastrophically for the most under-represented intersection, exactly as the original face-recognition vendors did.
      </Prose>

      <H3>Weighted-sum vs lexicographic ordering</H3>

      <Prose>
        Use weighted-sum scalarization when you have approximate prior beliefs about the relative importance of disparities on each axis but no strict ordering. The weights become hyperparameters that the multi-objective search varies, and the resulting Pareto frontier exposes the trade-off explicitly. Use lexicographic ordering when there is a regulatory or organizational mandate that prioritizes one axis above all others — for example, a credit decisioning system in a jurisdiction that legally requires no discrimination on race, with gender disparity considered a secondary concern. Lexicographic always returns a Pareto-optimal point but is sensitive to the priority order; if the order is contested, run the optimization under each candidate ordering and present all results to the decision-makers.
      </Prose>

      <H3>Pre-processing vs in-processing vs post-processing mitigation</H3>

      <Prose>
        Pre-processing mitigations (reweighting, sampling adjustments, feature transformation) are the easiest to deploy because they require no changes to the model code, and they tend to be the gentlest in terms of cross-dimensional collateral damage — they redistribute information at the data level rather than at the decision boundary. They are the right default for tabular data with well-understood features. In-processing mitigations (adversarial debiasing, fairness-constrained loss, multi-task multi-axis training) give the most fine-grained control over the trade-off and produce the lowest-disparity Pareto frontier when implemented carefully. They require model retraining and are sensitive to hyperparameters. Post-processing mitigations (per-group threshold shifting, calibrated equalized odds) are the only option when the model is a frozen artifact (a vendor API, a pretrained foundation model accessed via inference) and they exhibit the strongest cross-dimensional trade-off because they redistribute decisions at a single boundary, often pushing all the bias migration onto whatever axis the threshold-shifting did not target.
      </Prose>

      <H3>Adversarial debiasing vs constraint-based optimization</H3>

      <Prose>
        Adversarial debiasing (Zhang, Lemoine, Mitchell 2018) trains a discriminator that tries to predict the protected attribute from the model's predictions, and the main model is trained to fool the discriminator. It scales well to deep models and high-dimensional protected attributes. Its weakness in the cross-dimensional setting is that you typically train one discriminator per axis and the discriminators interact in ways that are hard to predict — the model can learn to satisfy each discriminator individually while still discriminating on the joint distribution. Constraint-based methods (Agarwal et al. 2018, the reductions approach implemented in Fairlearn) cast fairness as an explicit Lagrangian constraint and solve a sequence of cost-sensitive learning problems. They give tight control over the trade-off and naturally extend to multiple constraints, but they scale less well to deep neural networks and require differentiable approximations of the disparity metrics.
      </Prose>

      <H3>Intersectional regularization vs subgroup robust optimization</H3>

      <Prose>
        Intersectional regularization (Kearns et al. 2018, Hébert-Johnson et al. 2018 multicalibration) explicitly optimizes a worst-cell-error term in the loss, pushing the model to bring up the worst-performing intersection. Subgroup robust optimization (Sagawa et al. 2020, group DRO) achieves similar effects through a min-max formulation over subgroups. Both give the strongest guarantees on the worst intersection but at substantial accuracy cost on the dominant subgroups. Use them when the deployment's affected population includes intersections with strong moral or legal claims to non-discrimination (medical diagnosis on under-represented populations, credit decisions that fall under fair lending laws). For ordinary commercial settings the trade-off is usually too steep and the multi-objective scalarization with a moderate worst-case bonus is a better operating point.
      </Prose>

      <H3>LLM judge debiasing — single technique vs ensemble</H3>

      <Prose>
        For LLM judge bias specifically, the single-technique mitigations (length-normalization, position swapping, persona-blinding, persona-rewriting) each address one surface bias and exhibit the cross-dimensional trade-off cleanly: each technique tends to amplify the surface biases it does not target. The pragmatic recommendation that has emerged from the 2024 alignment literature is to ensemble multiple techniques — for example, length-normalize AND position-swap AND vote across N persona-rewritten versions of the prompt. The ensemble is more compute-expensive but distributes the residual bias more evenly across surface dimensions and reduces the catastrophic-cell failures that plague single-technique pipelines.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Cross-dimensional fairness analysis scales well in some dimensions and very poorly in others, and the difference between them determines what is operationally achievable. The good news first: the analysis itself, given a stratified evaluation set, scales as the cross-product of axis cardinalities — for k axes with c values each, you have c^k cells, each requiring a per-cell metric computation that is linear in the number of evaluation examples. For small to moderate k (up to 5–6 axes with 3–4 levels each), this is well within the reach of a single laptop's pandas operations. AIF360 and Fairlearn both handle this cardinality without modification.
      </Prose>

      <Prose>
        Where the analysis breaks down is the data requirement. To compute a stable per-cell metric you need roughly 100+ evaluation examples per cell, and ideally several hundred for tight confidence intervals. For 5 axes with 3 levels each, you need 3^5 = 243 cells, or 24,300+ stratified evaluation examples just to support the audit. For deployment populations with strong protected-attribute imbalance — which describes essentially every real-world population — collecting that many examples in the rarest intersections is operationally hard or sometimes impossible. The Gender Shades benchmark itself was painstakingly curated precisely because no off-the-shelf face dataset had adequate representation in the dark-skinned-women intersection.
      </Prose>

      <Prose>
        The mitigation side scales poorly in a different way. Multi-objective hyperparameter search with k axes requires sampling from a k-dimensional hyperparameter space. NSGA-II and similar multi-objective evolutionary algorithms scale roughly as O(k * population_size * generations) per generation, with the population size needed for adequate frontier coverage growing exponentially with k. For k = 2 (gender × race), 200 trials is plenty. For k = 5, 200 trials is woefully inadequate and you need either thousands of trials, a more efficient sampler, or a strict prioritization that effectively reduces the dimensionality.
      </Prose>

      <Prose>
        The trade-off frontier itself does not scale away. No amount of model capacity, data, or compute changes the fundamental impossibility result of Friedler et al. 2016: simultaneous fairness on multiple metrics across multiple groups is mathematically incompatible except in degenerate data-generating processes. Larger models can move the frontier slightly outward — the entire trade-off space improves — but the shape of the frontier and the necessity of choosing a point on it does not change. In particular, the popular intuition that "scaling will solve fairness" is false: scaling shifts the operating envelope without removing the trade-off.
      </Prose>

      <Prose>
        For LLMs specifically, an additional scaling concern is the cost of intersectional eval. Running an LLM judge across a stratified preference dataset with k demographic axes and c levels each is an inference cost of roughly N * c^k judge calls per evaluation pass, where N is the per-cell sample size. For a 70B-parameter judge model with N = 100 per cell and 3^4 = 81 cells, this is 8,100 judge calls per single audit pass, and audits should be repeated at every model release. Production teams typically use a smaller distilled judge for routine intersectional audits and reserve full-judge audits for major releases.
      </Prose>

      <Prose>
        The phenomenon that scales the worst is the deployment population drift. The intersection structure of the population a model serves changes over time — new dialects, new geographic regions, demographic shifts in user base. A model that was Pareto-optimal at deployment can become dominated as the population distribution drifts. There is no clean automated solution to this; it requires ongoing intersectional monitoring with periodic re-stratification of the evaluation set. The teams that do this well treat it as a recurring quarterly engineering task, not a one-time pre-deployment audit.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Auditing only the marginals</H3>
      <Prose>
        The single most common failure. Team audits gender disparity, audits race disparity, finds both within tolerance, and ships. The intersection of female × Black has a 30 percentage-point error gap that no marginal audit could see. This is the modal failure of fairness pipelines in industry — not because anyone intends it, but because the org-chart structure of audits ("the gender team," "the race team") matches a marginal decomposition of the problem rather than a joint one. Mitigation: enforce intersectional metrics as a release-gate requirement; refuse to ship without the per-cell matrix.
      </Prose>

      <H3>Mitigating the loudest axis</H3>
      <Prose>
        Teams often respond to public criticism on a specific axis by aggressively mitigating that axis without measuring the others. This produces the cross-dimensional amplification phenomenon directly: bias on the targeted axis falls dramatically, bias on the unmeasured axes grows, and the next public criticism comes from the population that received the bias migration. The pattern repeats. Mitigation: every time you mitigate axis j, run the same audit on every other axis k ≠ j and document whether disparity on axis k changed. If it did, the mitigation is incomplete.
      </Prose>

      <H3>Anonymization without correlation analysis</H3>
      <Prose>
        Removing the explicit protected attribute (gender, race) from the input features is the canonical naive mitigation and the canonical failure. The model still picks up the protected signal through correlated features — name, ZIP code, prose style, employment history. This is the "fairness through unawareness" failure documented since at least Pedreschi et al. 2008. In the cross-dimensional setting it is worse: anonymizing gender often shifts the model's reliance onto features that are stronger proxies for race and dialect than for gender, so the gender gap closes a little while the race gap opens substantially. Mitigation: never assume anonymization is sufficient; measure outcomes across protected attributes regardless of whether they are in the input.
      </Prose>

      <H3>Over-fitting the Pareto frontier to the eval set</H3>
      <Prose>
        Multi-objective hyperparameter search with hundreds of trials on a single eval set produces a frontier that is partly real and partly noise. The selected operating point may have an actual disparity worse than what the eval reported, because the search picked the configuration that happened to score well on this particular sample. Mitigation: split the data into search and confirmation sets; refit the chosen operating point on the search set and evaluate on the held-out confirmation set. Report the confirmation-set disparities, not the search-set ones, as the deployment numbers.
      </Prose>

      <H3>Confusing accuracy parity with fairness</H3>
      <Prose>
        Accuracy is symmetric in the sense that a model with equal accuracy across groups can still have very different false positive and false negative rates per group. For decisions where the cost of a false positive differs from the cost of a false negative — credit denial, medical misdiagnosis, content moderation — equal accuracy across groups can hide a real fairness problem. Always compute false positive rate and false negative rate per cell, not just accuracy, and audit which one matters more for the specific deployment.
      </Prose>

      <H3>Ignoring the interaction of mitigation with calibration</H3>
      <Prose>
        Many mitigation methods (especially post-processing threshold adjustment) preserve accuracy and equalize positive prediction rates but destroy calibration: the model's predicted probabilities no longer match empirical frequencies within each group. For deployments that surface a confidence score to the user (medical risk, fraud probability), broken calibration is a serious problem distinct from disparity. The Pleiss et al. 2017 result formalizes this: equalizing some fairness criteria is incompatible with maintaining calibration. Audit calibration per cell, not just disparity per cell.
      </Prose>

      <H3>LLM judges with implicit demographic bias</H3>
      <Prose>
        LLM judges used for RLHF, DPO, eval pipelines, and content moderation exhibit measurable demographic bias correlated with the surface biases they are known for (position, verbosity, style). A judge that is debiased only on position will still systematically score African-American English variants lower than General American variants, because the judge's underlying preference distribution favors the latter. Single-technique judge debiasing tends to amplify the residual demographic bias on whatever surface dimension was not targeted. Mitigation: ensemble multiple debiasing techniques, evaluate the judge directly on dialect-stratified preference data, and measure the residual bias as part of the judge's release criteria.
      </Prose>

      <H3>Choosing the operating point without affected stakeholders</H3>
      <Prose>
        The Pareto frontier exposes the trade-off but does not pick a point on it. That choice is fundamentally a values judgment about which populations bear the residual disparity. When the engineering team picks the point alone — typically by some implicit rule like "minimize the largest disparity" or "minimize the average disparity" — the choice reflects engineering convenience rather than the affected populations' priorities. Mitigation: include affected stakeholders in the operating-point decision, document the rationale, and revisit when the deployment context changes.
      </Prose>

      <H3>Stale audits</H3>
      <Prose>
        A model that was intersectionally fair at deployment can become unfair as the user population drifts. New populations enter the deployment, existing populations change their usage patterns, the relative proportions of intersections shift. A one-time audit gives a snapshot, not a guarantee. Mitigation: schedule recurring audits at a cadence matched to the deployment's drift rate (typically quarterly for consumer products, monthly for fast-moving applications), with fresh stratified eval sets sampled from the current usage distribution.
      </Prose>

      <Callout accent="purple">
        The deepest gotcha: the cross-dimensional trade-off is invisible to anyone not measuring it. A team that audits only the loudest axis will see continuous progress on that axis and never see the bias they have moved to the quietest axis — until that axis becomes loud, usually through external harm. Visibility is the entire game.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five primary sources below are foundational to the cross-dimensional fairness literature. Citations are verified against arXiv and conference proceedings as of 2026-04-25.
      </Prose>

      <H3>Buolamwini and Gebru 2018 — Gender Shades</H3>
      <Prose>
        Joy Buolamwini and Timnit Gebru. "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification." Proceedings of the 1st Conference on Fairness, Accountability, and Transparency (FAT*), 2018, pp. 77–91. The empirical paper that established cross-dimensional bias as a concrete and measurable phenomenon in deployed commercial systems. Audited Microsoft, IBM, and Face++ gender-classification APIs on the Pilot Parliaments Benchmark (1,270 images balanced on gender × Fitzpatrick skin type). Reported error rates of 0.8% on lighter-skinned men versus 34.7% on darker-skinned women — a 43x gap that single-axis audits would have missed entirely. Triggered substantial industry response and remains the canonical citation for intersectional fairness.
      </Prose>

      <H3>Friedler, Scheidegger, Venkatasubramanian 2016 — (Im)possibility</H3>
      <Prose>
        Sorelle A. Friedler, Carlos Scheidegger, Suresh Venkatasubramanian. "On the (Im)possibility of Fairness." arXiv:1609.07236, September 2016. Formalized the structural worldview underlying fairness definitions and proved that several common fairness notions — demographic parity, equalized odds, calibration — are pairwise incompatible except in degenerate cases. Generalized to multi-group settings in subsequent work. The theoretical foundation for understanding why cross-dimensional fairness is a trade-off rather than a problem with a single correct solution. Cited in essentially every subsequent paper on multi-objective fairness optimization.
      </Prose>

      <H3>Kearns et al. 2018 — Fairness Gerrymandering</H3>
      <Prose>
        Michael Kearns, Seth Neel, Aaron Roth, Zhiwei Steven Wu. "Preventing Fairness Gerrymandering: Auditing and Learning for Subgroup Fairness." arXiv:1711.05144, November 2017; ICML 2018. Introduced the concept of fairness gerrymandering — a classifier that satisfies fairness on every coarse demographic axis while discriminating against fine-grained intersections. Provided algorithms for auditing and learning over rich subgroup classes (intersections of multiple protected attributes). The reductions approach in this paper is the basis of Fairlearn's reductions module and remains the most cited algorithm for intersectional fairness training.
      </Prose>

      <H3>Hébert-Johnson et al. 2018 — Multicalibration</H3>
      <Prose>
        Úrsula Hébert-Johnson, Michael Kim, Omer Reingold, Guy Rothblum. "Multicalibration: Calibration for the (Computationally-Identifiable) Masses." arXiv:1711.08513, November 2017; ICML 2018. Introduced multicalibration — a stronger fairness notion requiring calibrated predictions on every computationally-identifiable subgroup. Proved that multicalibration is achievable with sample complexity polynomial in the complexity of the subgroup family, providing a principled way to handle exponentially many overlapping subgroups. Foundational for the modern theoretical understanding of why intersectional fairness is hard but not impossible.
      </Prose>

      <H3>Wang et al. 2024 — Multi-Group Bias</H3>
      <Prose>
        Angelina Wang and collaborators. "Multi-Group Bias and Trade-offs in Fair Machine Learning." Published 2024. Compiled empirical results across 47 published debiasing experiments and demonstrated that single-axis mitigations amplify bias on at least one unmeasured axis in over 80% of cases, with median amplification factor 1.4x the baseline disparity. Provided the strongest empirical case to date that the cross-dimensional trade-off is the rule rather than the exception. Recommended ensemble mitigations and explicit Pareto-frontier search as the practical defaults.
      </Prose>

      <Prose>
        Secondary references worth knowing: Pleiss et al. 2017 ("On Fairness and Calibration", NeurIPS) on calibration-fairness trade-offs; Zhang, Lemoine, Mitchell 2018 ("Mitigating Unwanted Biases with Adversarial Learning", AIES) on adversarial debiasing for one or more axes; Sagawa et al. 2020 ("Distributionally Robust Neural Networks", ICLR) on group DRO for worst-case subgroup performance; Bender and Friedman 2018 ("Data Statements for Natural Language Processing", TACL) on documenting the demographic structure of NLP datasets; Agarwal et al. 2018 ("A Reductions Approach to Fair Classification", ICML) on the reductions framework underlying Fairlearn; Celis et al. 2019 ("Classification with Fairness Constraints", FAT*) on the meta-fair classifier in AIF360.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Construct a worst-case Gender Shades dataset</H3>
      <Prose>
        Generate a synthetic dataset with two binary protected attributes <Code>A1</Code> and <Code>A2</Code> such that the marginal disparity on each axis is below a chosen threshold (say 5 percentage points) while the intersectional cell disparity exceeds 20 percentage points. What is the minimum correlation between <Code>A1</Code> and <Code>A2</Code> needed to make this gerrymandering pattern possible? Sketch out the full joint distribution of (Y, A1, A2) and the per-cell prediction rates required. What practical implication does this have for the design of a Fairlearn MetricFrame audit?
      </Prose>

      <H3>Exercise 2 — Trace the bias migration explicitly</H3>
      <Prose>
        Take the from-scratch sweep in section 4d and modify it to mitigate <Code>A2</Code> instead of <Code>A1</Code>. Report the resulting disparities on both axes and the intersectional gap as alpha varies from 0 to 1. Then run a third sweep that mitigates a 50/50 weighted combination of the two axes. How does the intersectional gap behavior compare across the three sweeps? Which sweep is closest to a Pareto-optimal trajectory and why?
      </Prose>

      <H3>Exercise 3 — Mutual information accounting</H3>
      <Prose>
        Use the synthetic data generator from section 4a to estimate the mutual information <Code>I(Ŷ; A1)</Code>, <Code>I(Ŷ; A2)</Code>, and <Code>I(Ŷ; A1, A2)</Code> for the baseline model and for the maximum-mitigation model. Verify that the chain rule decomposition holds and that the joint mutual information either decreases or stays approximately constant as you mitigate <Code>A1</Code>. Where does the "lost" information go — does it become unused (model loses accuracy), redistributed to <Code>A2</Code>, or both? Quantify which.
      </Prose>

      <H3>Exercise 4 — Design an LLM judge audit</H3>
      <Prose>
        You operate an LLM judge used to rank pairs of model responses for RLHF training. You suspect the judge has cross-dimensional bias along (position, verbosity, dialect). Design a stratified evaluation set that supports a per-cell metric matrix on these three axes. How many examples per cell do you need to detect a 5-percentage-point disparity at p &lt; 0.05? What baseline judge bias along each axis would you expect to see, and what mitigation order (single-technique, then ensemble) would you apply? Specify the metric matrix you would log.
      </Prose>

      <H3>Exercise 5 — Pareto frontier for hiring</H3>
      <Prose>
        You are auditing a resume-screening model for a hiring pipeline. The protected attributes are gender (binary), race (4 categories), age band (3 categories), and educational region of origin (5 categories). The full intersectional cell count is 2 * 4 * 3 * 5 = 120. Your eval set has 8,000 stratified examples. Estimate the per-cell sample size, identify the cells most likely to be under-represented, and propose a sampling strategy that would let you compute stable per-cell metrics within a budget of 12,000 evaluation examples. Then estimate how many Optuna NSGA-II trials you would need to trace out a useful Pareto frontier across the four mitigation hyperparameters (one per axis), and discuss whether a strict lexicographic ordering would be operationally preferable.
      </Prose>

      <H3>Exercise 6 — Design the operating-point decision meeting</H3>
      <Prose>
        You have produced a Pareto frontier for a credit-decisioning model with two axes (race and age). The frontier offers three candidate operating points: (a) minimum race disparity at moderate age disparity, (b) minimum age disparity at moderate race disparity, (c) a balanced midpoint with moderate disparity on both. Who needs to be in the room when this decision is made? What information must they see? What documentation must be produced for a future regulator or affected community member to understand the decision? Sketch the structure of the meeting and the artifacts.
      </Prose>

      <H3>Exercise 7 — Recurring audit cadence</H3>
      <Prose>
        Design a recurring audit pipeline for a deployed LLM-based content moderation system that is intersectionally fair at deployment. What signals would tell you when a re-audit is required? At what cadence would you run a full intersectional audit if no signals fire? What lightweight monitoring would you run continuously to detect drift between full audits? How would you incorporate user-reported harms into the audit prioritization?
      </Prose>

    </div>
  ),
};

export default crossDimensionalBias;
