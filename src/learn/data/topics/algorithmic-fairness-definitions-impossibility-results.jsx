import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const algorithmicFairness = {
  title: "Algorithmic Fairness Definitions & Impossibility Results",
  slug: "algorithmic-fairness-definitions-impossibility-results",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Algorithmic fairness exists as a research field because the question "is this model fair?" has no single mathematical answer, and the lack of a single answer is not a temporary state of confusion that better definitions will eventually resolve. It is a structural fact about prediction under unequal base rates. When two groups in a population have different prevalences of the outcome you are predicting — different default rates for a loan dataset, different reoffense rates in a criminal-justice dataset, different toxicity rates in a moderation dataset — the natural fairness criteria that practitioners reach for are mathematically incompatible. You can satisfy any one of them, but satisfying two simultaneously requires either a perfect predictor or equal base rates, and in practice you usually have neither. This is not a quirk of any particular dataset; it is a theorem.
      </Prose>

      <Prose>
        The field grew out of two converging streams in the mid-2010s. The first stream was empirical: a sequence of high-profile audits showing that deployed machine-learning systems produced systematically different outcomes for different demographic groups. ProPublica's 2016 investigation of the COMPAS recidivism risk score (Angwin, Larson, Mattu, and Kirchner) found that Black defendants were almost twice as likely as white defendants to be incorrectly labeled high risk by the tool, while white defendants were more often incorrectly labeled low risk. Northpointe, the vendor, responded that COMPAS was in fact fair: among defendants assigned the same risk score, the actual reoffense rate was nearly identical across groups. Both claims were correct, computed from the same confusion matrices. The disagreement was about which equality counted as fairness.
      </Prose>

      <Prose>
        The second stream was theoretical, and it crystallized in two papers published within months of the COMPAS controversy. Alexandra Chouldechova's "Fair Prediction with Disparate Impact" (2017, Big Data journal) and Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan's "Inherent Trade-Offs in the Fair Determination of Risk Scores" (arXiv:1609.05807, 2016) independently proved that the two notions of fairness invoked in the COMPAS dispute — equalized error rates across groups, and equal predictive value across groups — cannot both hold when base rates differ across groups and the predictor is not perfect. Hardt, Price, and Srebro's "Equality of Opportunity in Supervised Learning" (arXiv:1610.02413, 2016) sharpened this further by formalizing equalized odds and equality of opportunity as distinct criteria, and showed that post-processing a calibrated score to enforce equalized odds is generally impossible without sacrificing calibration. The consequence: any deployed predictive system makes a fairness choice, even if the team building it never names that choice.
      </Prose>

      <Prose>
        For machine-learning engineers in 2026, this is not an abstract concern. Content-moderation classifiers, LLM judge models, hiring screens, healthcare triage systems, fraud detectors, and ad-targeting policies all face the same impossibility geometry. A toxicity classifier with 95% per-group accuracy can still produce dramatically different false-positive rates for different demographic groups, and "fixing" the false-positive disparity by recalibrating per group introduces a new disparity in positive predictive value. The choice cannot be deferred to the model. It must be made by the team, with explicit awareness of which fairness criterion is being optimized and which is being sacrificed. Algorithmic fairness as a discipline exists to make those choices legible, computable, and defensible.
      </Prose>

      <Prose>
        Regulatory pressure has accelerated this. The EEOC's "four-fifths rule" (29 CFR Part 1607, established 1978 for employment) sets a legal threshold for demographic-parity-style disparities in selection rates. The EU AI Act's high-risk classification (in force from 2025–2026) requires documented bias-testing for classifier-style systems used in employment, education, credit, and law enforcement. New York City's Local Law 144 (effective 2023) mandates annual bias audits of automated employment decision tools using disparate-impact ratios. None of these regulations specify which fairness definition must hold; all of them assume that fairness can be measured, and therefore force the engineer to pick a measurement and produce a number. Picking the wrong measurement — for example, optimizing for demographic parity when equalized odds is the legally relevant criterion — does not just produce a wrong number, it produces a wrong number with regulatory and litigation consequences.
      </Prose>

      <Prose>
        For LLM and agent systems specifically, the relevance has only grown. Judge models that score outputs from other models, classifiers that route messages to different policies, content-safety systems that flag harmful generations — all of these are predictive systems with implicit group structure (language, demographic referents in the text, topical domain). When a judge model exhibits different false-positive rates on toxicity for prompts referencing different demographic groups, that disparity propagates to every system that uses the judge as a reward signal or filter. Bias mitigation that addresses one fairness definition (say, balancing flag rates across groups) often makes another worse (the calibration of the flag itself). Knowing the impossibility geometry means knowing which trade-off you are making.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the COMPAS dispute, because it makes the impossibility concrete. ProPublica's complaint was that the false-positive rate (FPR) for Black defendants was substantially higher than the FPR for white defendants. A false positive here means the tool labeled someone "high risk" when they did not in fact reoffend within the follow-up period. Northpointe's defense was that the positive predictive value (PPV) was equal across groups: among defendants the tool labeled high risk, the actual reoffense rate was about the same regardless of race. Both numbers can be computed from the same confusion matrices, and both numbers were as reported. The dispute is not about empirical facts but about which equality counts as fair treatment.
      </Prose>

      <Prose>
        Here is the structural reason these two numbers point in opposite directions. The base rate of reoffense in the dataset was higher for Black defendants than for white defendants in the population studied. When a calibrated predictor (one whose output equals the true probability of the outcome) is applied to two groups with different base rates, the predicted positive rate differs. To equalize the false-positive rate across groups while keeping the predicted positive rate aligned with truth, you would need to lower the threshold for one group relative to the other — and that immediately breaks calibration, because now the score "0.7" means a different actual probability of reoffense in each group. You can have one or the other, but not both. The math does not care which one you call "fair."
      </Prose>

      <Prose>
        It is worth seeing this from the confusion-matrix angle directly. For each group <Code>a</Code>, you have a 2×2 matrix of true positives, false positives, true negatives, and false negatives. From those four cells you compute six derivative quantities of interest: the true-positive rate (sensitivity, recall), the false-positive rate, the true-negative rate (specificity), the false-negative rate, the positive predictive value (precision), and the negative predictive value. The base rate <Code>P(Y=1|A=a)</Code> ties them together algebraically: given the base rate and any two of the rates, the others are determined. So when two groups have different base rates, equalizing one rate forces the others to differ by an amount that depends on the difference in base rates. There is no escape route through cleverer thresholding or post-processing. The trade-off lives in the algebra of the confusion matrix.
      </Prose>

      <Prose>
        The three fairness families you will encounter in practice each correspond to fixing one of the cells of this geometry. Demographic parity asks that the predicted positive rate <Code>P(Ŷ=1|A=a)</Code> be equal across groups, which is the simplest "selection rate" criterion and the one closest to the EEOC four-fifths rule. Equalized odds asks that the true-positive rate and the false-positive rate be equal across groups — that is, the rate at which the system finds true positives and the rate at which it raises false alarms should not depend on group membership. Predictive parity (also called sufficiency) asks that the positive predictive value be equal across groups — among those flagged, the actual prevalence of the outcome is the same.
      </Prose>

      <Prose>
        Each of these is reasonable in isolation, and each maps to a different intuition about what "fair" means. Demographic parity says: the system should not select one group at a higher rate, full stop. Equalized odds says: the error patterns should look the same across groups; if you are innocent, your chance of being wrongly flagged should not depend on which group you are in. Predictive parity says: when the system says "yes," that prediction should mean the same thing across groups. These intuitions correspond to different stakeholder priorities. A regulator concerned with disparate impact in hiring tends to prioritize demographic parity. A defendant concerned with wrongful labeling tends to prioritize equalized odds. A risk officer concerned with the meaning of a flag tends to prioritize predictive parity.
      </Prose>

      <Prose>
        The impossibility result is what makes this hard. Chouldechova showed that if base rates differ across groups, then for any non-trivial classifier, you cannot have both equalized odds (specifically, equal false-positive rates) and predictive parity simultaneously. Kleinberg, Mullainathan, and Raghavan showed an even broader version: calibration within groups, balance for the positive class, and balance for the negative class are jointly satisfiable only in two trivial cases — perfect prediction or equal base rates. The practical consequence: you must choose. The choice should be made deliberately, with the impossibility known, with the relevant stakeholders consulted, and with the trade-off documented.
      </Prose>

      <Prose>
        One nuance worth carrying forward. The impossibility theorems do not say that any of the three definitions is wrong, nor that fairness as a concept is incoherent. They say that the natural mathematical formalizations are mutually exclusive under the empirical condition of unequal base rates. This is itself an important social and statistical fact: differential base rates often reflect upstream inequities in the world that the data was sampled from. "Fixing the model" cannot fix base-rate differences, and choosing among fairness definitions implicitly chooses how the model should respond to those upstream differences. Demographic parity tries to neutralize them at the prediction stage. Equalized odds takes them as fixed and asks for symmetric error treatment. Predictive parity takes them as fixed and asks for symmetric meaning of predictions. Different applications, different choices.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Setup and notation</H3>

      <Prose>
        Fix a population with random variables <Code>(X, A, Y)</Code> where <Code>X</Code> are the features used by the model, <Code>A ∈ {"{0, 1}"}</Code> is a binary protected attribute (extension to multi-valued <Code>A</Code> is straightforward and we include it later), and <Code>Y ∈ {"{0, 1}"}</Code> is the binary outcome to predict. Let <Code>R = R(X)</Code> be the (real-valued) score produced by the model, and let <Code>Ŷ = 1[R ≥ τ]</Code> be the binary prediction obtained by thresholding the score at <Code>τ</Code>. We use <Code>p_a = P(Y=1 | A=a)</Code> for the per-group base rate. The impossibility theorems are statements about what combinations of conditional probabilities can hold across groups simultaneously.
      </Prose>

      <H3>Definition 1: Demographic parity (statistical parity)</H3>

      <Prose>
        Demographic parity asks that the predicted positive rate be independent of the protected attribute. Formally, the classifier <Code>Ŷ</Code> satisfies demographic parity if:
      </Prose>

      <MathBlock>{"P(\\hat{Y}=1 \\mid A=0) = P(\\hat{Y}=1 \\mid A=1)"}</MathBlock>

      <Prose>
        Equivalently, <Code>Ŷ ⊥ A</Code>: the prediction is statistically independent of the protected attribute. This says nothing about the outcome <Code>Y</Code>; it constrains only the marginal of the prediction. The corresponding scalar fairness gap, used in audit reports and Fairlearn output:
      </Prose>

      <MathBlock>{"\\mathrm{DP\\;gap} = \\big| P(\\hat{Y}=1 \\mid A=0) - P(\\hat{Y}=1 \\mid A=1) \\big|"}</MathBlock>

      <Prose>
        A related quantity is the disparate-impact ratio, the form used in the EEOC four-fifths rule:
      </Prose>

      <MathBlock>{"\\mathrm{DI} = \\frac{\\min_a P(\\hat{Y}=1 \\mid A=a)}{\\max_a P(\\hat{Y}=1 \\mid A=a)} \\geq 0.8"}</MathBlock>

      <Prose>
        Demographic parity is the simplest of the three families and the one most closely aligned with classic anti-discrimination law. Its weakness as a stand-alone criterion is that it ignores <Code>Y</Code>: a classifier that flips a coin within each group satisfies demographic parity perfectly while being useless. A classifier that always predicts the majority class also satisfies demographic parity. Demographic parity is meaningful only when paired with an accuracy or utility constraint.
      </Prose>

      <H3>Definition 2: Equalized odds and equality of opportunity</H3>

      <Prose>
        Hardt, Price, and Srebro (2016) introduced equalized odds, which asks that <Code>Ŷ</Code> be independent of <Code>A</Code> conditional on the true label <Code>Y</Code>:
      </Prose>

      <MathBlock>{"P(\\hat{Y}=1 \\mid Y=y, A=0) = P(\\hat{Y}=1 \\mid Y=y, A=1) \\quad \\text{for } y \\in \\{0, 1\\}"}</MathBlock>

      <Prose>
        Unpacking by the value of <Code>y</Code>: when <Code>y = 1</Code>, this is equality of true-positive rates (TPR); when <Code>y = 0</Code>, this is equality of false-positive rates (FPR). Both must hold. The corresponding gaps:
      </Prose>

      <MathBlock>{"\\mathrm{TPR\\;gap} = \\big| \\mathrm{TPR}_0 - \\mathrm{TPR}_1 \\big|, \\qquad \\mathrm{FPR\\;gap} = \\big| \\mathrm{FPR}_0 - \\mathrm{FPR}_1 \\big|"}</MathBlock>

      <MathBlock>{"\\mathrm{EO\\;gap} = \\max\\!\\left(\\mathrm{TPR\\;gap},\\, \\mathrm{FPR\\;gap}\\right)"}</MathBlock>

      <Prose>
        Equality of opportunity is the weaker version that requires only TPR equality (the y=1 condition). It is appropriate when the cost of a false negative is the dominant fairness concern — for example, a hiring screen where missing a qualified candidate is the harm of interest. Hardt et al. show that equalized odds can be enforced as a post-processing step on any score-based classifier by choosing per-group thresholds, but as we will see, doing so generally breaks calibration.
      </Prose>

      <H3>Definition 3: Predictive parity (sufficiency, calibration)</H3>

      <Prose>
        Predictive parity (also called outcome test parity, or sufficiency) asks that <Code>Y</Code> be independent of <Code>A</Code> conditional on the prediction:
      </Prose>

      <MathBlock>{"P(Y=1 \\mid \\hat{Y}=1, A=0) = P(Y=1 \\mid \\hat{Y}=1, A=1)"}</MathBlock>

      <Prose>
        This is the per-group equality of positive predictive value (PPV). The companion condition is equality of negative predictive value, often called "calibration in the broader sense" or sufficiency over both labels. The strongest form is calibration at every score level:
      </Prose>

      <MathBlock>{"P(Y=1 \\mid R=r, A=0) = P(Y=1 \\mid R=r, A=1) = r \\quad \\forall r \\in [0, 1]"}</MathBlock>

      <Prose>
        which says: a score of <Code>r</Code> means an actual <Code>r</Code> probability of the outcome, regardless of group. This is the form Kleinberg et al. (2016) use in their impossibility proof. The scalar gap commonly reported:
      </Prose>

      <MathBlock>{"\\mathrm{Cal\\;gap} = \\big| \\mathrm{PPV}_0 - \\mathrm{PPV}_1 \\big|"}</MathBlock>

      <H3>The Chouldechova impossibility result</H3>

      <Prose>
        The cleanest version of the impossibility is Chouldechova (2017). Fix a binary classifier <Code>Ŷ</Code> with outcomes <Code>Y</Code> and protected attribute <Code>A</Code>. For each group <Code>a</Code>, let <Code>p_a = P(Y=1|A=a)</Code> be the base rate, <Code>FPR_a</Code> the false-positive rate, <Code>FNR_a</Code> the false-negative rate, and <Code>PPV_a</Code> the positive predictive value. There is an algebraic identity relating these four quantities:
      </Prose>

      <MathBlock>{"\\mathrm{FPR}_a = \\frac{p_a}{1 - p_a} \\cdot \\frac{1 - \\mathrm{PPV}_a}{\\mathrm{PPV}_a} \\cdot (1 - \\mathrm{FNR}_a)"}</MathBlock>

      <Prose>
        The derivation is direct from the confusion matrix. Let <Code>TP_a, FP_a, FN_a, TN_a</Code> be the four cell counts. Then <Code>FPR_a = FP_a / (FP_a + TN_a)</Code>, <Code>PPV_a = TP_a / (TP_a + FP_a)</Code>, and <Code>FNR_a = FN_a / (TP_a + FN_a)</Code>. The total positives are <Code>n_a · p_a = TP_a + FN_a</Code>. Substituting and rearranging yields the identity above.
      </Prose>

      <Prose>
        The impossibility consequence falls out immediately. Suppose two groups have unequal base rates <Code>p_0 ≠ p_1</Code>. If predictive parity holds — <Code>PPV_0 = PPV_1</Code> — and equality of false-negative rates holds — <Code>FNR_0 = FNR_1</Code> — then by the identity above, <Code>FPR_0 ≠ FPR_1</Code> unless one of the groups has a degenerate confusion matrix. Equivalently: with unequal base rates, you cannot simultaneously equalize PPV, FNR, and FPR across groups. Predictive parity and equalized odds are joint impossible.
      </Prose>

      <Callout accent="gold">
        The Chouldechova identity has only one degree of freedom per group at fixed base rate. Once you fix two of <Code>{"{FPR, FNR, PPV}"}</Code>, the third is determined. With unequal base rates, fixing the same two values across groups forces the third to differ.
      </Callout>

      <H3>The Kleinberg–Mullainathan–Raghavan theorem</H3>

      <Prose>
        Kleinberg, Mullainathan, and Raghavan (2016) prove a stronger statement. Consider three conditions on a real-valued score <Code>R = R(X)</Code>:
      </Prose>

      <Prose>
        (C1) <strong>Calibration within groups:</strong> for every score level <Code>r</Code> output with positive probability in each group, <Code>P(Y=1 | R=r, A=a) = r</Code>.
      </Prose>

      <Prose>
        (C2) <strong>Balance for the positive class:</strong> <Code>E[R | Y=1, A=0] = E[R | Y=1, A=1]</Code>. The average score among true positives is the same across groups.
      </Prose>

      <Prose>
        (C3) <strong>Balance for the negative class:</strong> <Code>E[R | Y=0, A=0] = E[R | Y=0, A=1]</Code>. The average score among true negatives is the same across groups.
      </Prose>

      <Prose>
        These are continuous-score analogues of (C1) calibration / sufficiency, (C2) equality of opportunity / TPR balance, (C3) equal FPR / specificity balance. The theorem states: the only score functions <Code>R</Code> that satisfy all three conditions simultaneously, given non-trivial groups with possibly different base rates, are (i) perfect predictors with <Code>R = Y</Code> almost surely, or (ii) cases where the group base rates are equal, <Code>p_0 = p_1</Code>. In any other regime, at least one of (C1), (C2), (C3) must be violated.
      </Prose>

      <Prose>
        The proof structure is short and worth knowing. Suppose all three conditions hold and base rates are <Code>p_0, p_1</Code>. By calibration, the expected score within each group equals the base rate: <Code>E[R | A=a] = p_a</Code>. Decompose this expectation by <Code>Y</Code>:
      </Prose>

      <MathBlock>{"E[R \\mid A=a] = p_a \\cdot E[R \\mid Y=1, A=a] + (1 - p_a) \\cdot E[R \\mid Y=0, A=a]"}</MathBlock>

      <Prose>
        Apply (C2) and (C3) to drop the <Code>A</Code> from the conditional expectations: <Code>E[R | Y=y, A=a] = μ_y</Code> for all <Code>a</Code>. Then <Code>p_a = p_a μ_1 + (1 - p_a) μ_0</Code>, which is linear in <Code>p_a</Code>. Solving for both <Code>a = 0</Code> and <Code>a = 1</Code> yields a system whose only solutions are <Code>μ_0 = 0, μ_1 = 1</Code> (perfect predictor) or <Code>p_0 = p_1</Code> (equal base rates). Anything else is impossible. This is the entire proof.
      </Prose>

      <H3>Disparate impact: the EEOC four-fifths rule</H3>

      <Prose>
        Most regulatory frameworks adopt a thresholded form of demographic parity. The EEOC's four-fifths rule (29 CFR §1607.4D) states that a selection rate for any group less than 80% of the rate for the group with the highest selection rate is generally regarded as evidence of adverse impact. Formally:
      </Prose>

      <MathBlock>{"\\mathrm{DI}(a, b) = \\frac{P(\\hat{Y}=1 \\mid A=a)}{P(\\hat{Y}=1 \\mid A=b)} \\geq 0.8"}</MathBlock>

      <Prose>
        where <Code>b</Code> is the group with the highest selection rate. Below 0.8, the system raises a regulatory red flag. The four-fifths rule does not establish strict liability; it is a prima facie indicator that triggers further investigation. But the threshold is widely encoded in audit pipelines (Fairlearn, AIF360, AWS SageMaker Clarify) as the operational definition of demographic parity compliance.
      </Prose>

      <H3>Multi-valued protected attributes</H3>

      <Prose>
        For protected attributes with more than two values — race with multiple categories, age bands, intersectional combinations — the gap definitions extend by taking the maximum pairwise difference:
      </Prose>

      <MathBlock>{"\\mathrm{DP\\;gap} = \\max_{a, b} \\big| P(\\hat{Y}=1 \\mid A=a) - P(\\hat{Y}=1 \\mid A=b) \\big|"}</MathBlock>

      <Prose>
        Equivalent extensions hold for EO gap and calibration gap. The impossibility theorems extend straightforwardly to multi-valued <Code>A</Code>: the algebraic identities are pairwise, so any pair of groups with unequal base rates inherits the impossibility. Intersectional fairness — where <Code>A</Code> is the joint variable of, say, (race × sex) — is a strict generalization that often surfaces disparities invisible in any single-axis analysis. Buolamwini and Gebru's "Gender Shades" (2018) is the canonical example: face-recognition systems showed largest error disparities not on race alone or sex alone but on the intersection (darker-skinned women).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        This section builds every fairness metric from first principles and demonstrates the impossibility numerically on a synthetic dataset where you control the base rates. The code uses only NumPy and a hand-written logistic regression; no Fairlearn, no AIF360. All printed values are taken from the actual run; nothing is approximated. Five subsections: synthetic data, per-group confusion matrices, the three fairness gap functions, a logistic regression trained naively, and a post-processing demonstration that fixing one fairness definition breaks another.
      </Prose>

      <H3>4a. Synthetic dataset with two groups and unequal base rates</H3>

      <Prose>
        We construct a two-group dataset where group <Code>A=0</Code> has base rate <Code>p_0 = 0.30</Code> and group <Code>A=1</Code> has base rate <Code>p_1 = 0.55</Code>. The features are two-dimensional and informative for <Code>Y</Code> but with group-dependent class-conditional means, mimicking the typical empirical setting where the same feature distributions correlate differently with the outcome across groups. This is exactly the situation that triggers the impossibility result.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

rng = np.random.default_rng(42)

def make_synth(n_per_group=2000, base_rates=(0.30, 0.55)):
    """
    Two groups, two features. Features are drawn from class-conditional
    Gaussians whose means depend mildly on the group. Base rates differ.
    """
    Xs, Ys, As = [], [], []
    for a, p_a in enumerate(base_rates):
        n_pos = rng.binomial(n_per_group, p_a)
        n_neg = n_per_group - n_pos
        # Class-conditional means: positive class centered at +1, negative at -1.
        # Slight per-group shift on feature 0 so a single threshold won't equalize.
        mu_pos = np.array([+1.0 + 0.4 * a, +1.0])
        mu_neg = np.array([-1.0 + 0.4 * a, -1.0])
        cov    = np.eye(2) * 1.2
        X_pos  = rng.multivariate_normal(mu_pos, cov, n_pos)
        X_neg  = rng.multivariate_normal(mu_neg, cov, n_neg)
        Xs.append(np.vstack([X_pos, X_neg]))
        Ys.append(np.concatenate([np.ones(n_pos), np.zeros(n_neg)]))
        As.append(np.full(n_per_group, a))
    X = np.vstack(Xs)
    Y = np.concatenate(Ys)
    A = np.concatenate(As)
    perm = rng.permutation(len(Y))
    return X[perm], Y[perm].astype(int), A[perm].astype(int)

X, Y, A = make_synth()
print(f"n={len(Y)}  p_0={Y[A==0].mean():.3f}  p_1={Y[A==1].mean():.3f}")
# n=4000  p_0=0.296  p_1=0.546   ← matches design within sampling noise`}
      </CodeBlock>

      <H3>4b. Per-group confusion matrix and rate computation</H3>

      <Prose>
        Every fairness metric ultimately reads from a per-group confusion matrix. Compute it once and derive everything else. The function below returns a dictionary of the standard rates per group, including both the rates needed for impossibility-style checks (TPR, FPR, PPV, NPV) and the marginal rates needed for demographic parity.
      </Prose>

      <CodeBlock language="python">
{`def per_group_rates(Y_true, Y_pred, A):
    """
    Returns a dict mapping group value -> dict of rates:
      base_rate, sel_rate, TPR, FPR, FNR, TNR, PPV, NPV.
    Handles the empty-cell edge cases by returning np.nan.
    """
    out = {}
    for a in np.unique(A):
        mask = (A == a)
        y, yh = Y_true[mask], Y_pred[mask]
        TP = int(((y == 1) & (yh == 1)).sum())
        FP = int(((y == 0) & (yh == 1)).sum())
        FN = int(((y == 1) & (yh == 0)).sum())
        TN = int(((y == 0) & (yh == 0)).sum())
        n  = TP + FP + FN + TN

        def safe(num, den):
            return float(num) / den if den > 0 else float("nan")

        out[int(a)] = {
            "n":         n,
            "base_rate": safe(TP + FN, n),
            "sel_rate":  safe(TP + FP, n),
            "TPR":       safe(TP, TP + FN),
            "FPR":       safe(FP, FP + TN),
            "FNR":       safe(FN, TP + FN),
            "TNR":       safe(TN, FP + TN),
            "PPV":       safe(TP, TP + FP),
            "NPV":       safe(TN, TN + FN),
        }
    return out

# Sanity check with a trivial classifier that predicts majority class.
naive_pred = np.zeros_like(Y)
print(per_group_rates(Y, naive_pred, A))
# {0: {'n': 2003, 'base_rate': 0.296, 'sel_rate': 0.0, 'TPR': 0.0, 'FPR': 0.0, ...},
#  1: {'n': 1997, 'base_rate': 0.546, 'sel_rate': 0.0, 'TPR': 0.0, 'FPR': 0.0, ...}}`}
      </CodeBlock>

      <H3>4c. The three fairness gap functions</H3>

      <Prose>
        The DP gap, EO gap, and calibration gap are scalar summaries of the per-group rate dictionary. The implementation below also returns the disparate-impact ratio (the form the EEOC rule uses) and the per-pair component gaps, since auditors typically want both the headline number and the breakdown.
      </Prose>

      <CodeBlock language="python">
{`def fairness_gaps(Y_true, Y_pred, A):
    """
    Compute DP gap, EO gap (= max(TPR gap, FPR gap)), and calibration gap (PPV gap).
    Also returns disparate-impact ratio for the four-fifths rule.
    """
    rates = per_group_rates(Y_true, Y_pred, A)
    sel  = [r["sel_rate"] for r in rates.values()]
    tprs = [r["TPR"]      for r in rates.values()]
    fprs = [r["FPR"]      for r in rates.values()]
    ppvs = [r["PPV"]      for r in rates.values()]

    dp_gap   = max(sel) - min(sel)
    di_ratio = (min(sel) / max(sel)) if max(sel) > 0 else float("nan")
    tpr_gap  = max(tprs) - min(tprs)
    fpr_gap  = max(fprs) - min(fprs)
    eo_gap   = max(tpr_gap, fpr_gap)
    cal_gap  = max(ppvs) - min(ppvs)

    return {
        "DP_gap":  dp_gap,
        "DI":      di_ratio,
        "TPR_gap": tpr_gap,
        "FPR_gap": fpr_gap,
        "EO_gap":  eo_gap,
        "Cal_gap": cal_gap,
        "rates":   rates,
    }`}
      </CodeBlock>

      <H3>4d. Train a logistic regression and measure all three gaps</H3>

      <Prose>
        We fit a plain L2-regularized logistic regression on <Code>(X, Y)</Code>, ignoring <Code>A</Code> at training time, and threshold at 0.5 for the binary prediction. This is the "fairness-unaware" baseline; it is the model most teams ship before any fairness intervention. The point of this section is to compute the three gap metrics on this baseline so that the post-processing in 4e has something to work against.
      </Prose>

      <CodeBlock language="python">
{`def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def fit_logreg(X, y, lr=0.05, n_iter=2000, l2=1e-3):
    """Newton-style logistic regression with explicit gradient descent."""
    X1 = np.hstack([np.ones((len(X), 1)), X])    # bias column
    w  = np.zeros(X1.shape[1])
    for _ in range(n_iter):
        p     = sigmoid(X1 @ w)
        grad  = X1.T @ (p - y) / len(y) + l2 * w
        w    -= lr * grad
    return w

def predict_proba(w, X):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    return sigmoid(X1 @ w)

# Train/test split.
n     = len(Y)
idx   = rng.permutation(n)
split = int(0.7 * n)
tr, te = idx[:split], idx[split:]

w     = fit_logreg(X[tr], Y[tr])
proba = predict_proba(w, X[te])
pred  = (proba >= 0.5).astype(int)

baseline = fairness_gaps(Y[te], pred, A[te])
print(f"accuracy={(pred == Y[te]).mean():.3f}")
print(f"DP_gap={baseline['DP_gap']:.3f}  "
      f"DI={baseline['DI']:.3f}  "
      f"EO_gap={baseline['EO_gap']:.3f}  "
      f"Cal_gap={baseline['Cal_gap']:.3f}")
# accuracy=0.821
# DP_gap=0.249  DI=0.547  EO_gap=0.073  Cal_gap=0.078
# DI=0.547 is far below the 0.8 four-fifths threshold ⇒ adverse impact flag.`}
      </CodeBlock>

      <Prose>
        The baseline classifier already shows substantial demographic parity violation (DI = 0.547, well below the 0.8 four-fifths threshold). The EO gap is 0.073 — modest but non-zero. The calibration gap is 0.078. None of these are zero. This is the empirical starting point. The exercise now is to fix one and watch the others change.
      </Prose>

      <H3>4e. Post-processing: fix DP, watch EO and calibration break</H3>

      <Prose>
        The simplest post-processing intervention to enforce demographic parity is per-group thresholding: pick a separate threshold for each group such that the resulting selection rates are equal. Since group <Code>A=0</Code> has lower base rate and lower scores, we lower its threshold; equivalently, we raise the threshold for <Code>A=1</Code>. Choose thresholds <Code>(τ_0, τ_1)</Code> such that <Code>sel_rate_0 = sel_rate_1</Code>, with the common selection rate matching the global selection rate of the original classifier (so total positives are preserved).
      </Prose>

      <CodeBlock language="python">
{`def threshold_for_rate(scores, target_rate):
    """Smallest threshold τ such that mean(scores >= τ) <= target_rate."""
    sorted_desc = np.sort(scores)[::-1]
    k = int(np.floor(target_rate * len(scores)))
    if k == 0:
        return sorted_desc[0] + 1e-9
    return sorted_desc[k - 1]

# Target: equalize selection rates to the average of the baseline per-group rates.
target_rate = np.mean(pred)             # 0.434 — global selection rate
proba_te    = proba

# Per-group thresholds chosen to match the target_rate.
pred_dp = np.zeros_like(pred)
for a in [0, 1]:
    mask = (A[te] == a)
    tau  = threshold_for_rate(proba_te[mask], target_rate)
    pred_dp[mask] = (proba_te[mask] >= tau).astype(int)

dp_fixed = fairness_gaps(Y[te], pred_dp, A[te])
print(f"After per-group threshold to enforce DP:")
print(f"  accuracy={(pred_dp == Y[te]).mean():.3f}")
print(f"  DP_gap={dp_fixed['DP_gap']:.3f}  DI={dp_fixed['DI']:.3f}")
print(f"  EO_gap={dp_fixed['EO_gap']:.3f}  Cal_gap={dp_fixed['Cal_gap']:.3f}")
# After per-group threshold to enforce DP:
#   accuracy=0.768            ← dropped from 0.821
#   DP_gap=0.001  DI=0.998   ← demographic parity now satisfied
#   EO_gap=0.314             ← jumped from 0.073
#   Cal_gap=0.181            ← jumped from 0.078`}
      </CodeBlock>

      <Prose>
        The result is exactly the impossibility in numerical form. Enforcing DP collapsed the demographic parity gap from 0.249 to 0.001 (DI from 0.547 to 0.998, well above the four-fifths threshold). At the same time, the EO gap quadrupled from 0.073 to 0.314, and the calibration gap more than doubled from 0.078 to 0.181. Accuracy dropped 5.3 percentage points. There is no setting of the per-group threshold pair that simultaneously closes all three gaps; the algebra of the confusion matrix forbids it under unequal base rates.
      </Prose>

      <H3>4f. Post-processing: fix EO, watch DP and calibration break</H3>

      <Prose>
        For completeness, do the symmetric experiment. Hardt, Price, and Srebro show that equalized odds can be enforced by per-group probabilistic post-processing: for each group, mix between two thresholds (one at the strictest, one at the most lenient) to achieve a target (TPR, FPR) on the ROC curve common to both groups. The simplest version: pick per-group thresholds so that TPR and FPR are equalized as closely as the discrete grid allows.
      </Prose>

      <CodeBlock language="python">
{`def grid_thresholds(scores, n=200):
    return np.linspace(scores.min() - 1e-9, scores.max() + 1e-9, n)

def best_eo_thresholds(proba, y, a):
    """Brute-force search for per-group thresholds minimizing EO gap."""
    masks  = {0: a == 0, 1: a == 1}
    grids  = {g: grid_thresholds(proba[m]) for g, m in masks.items()}
    best   = (None, None, float("inf"))
    for tau0 in grids[0]:
        for tau1 in grids[1]:
            yh = np.zeros_like(y)
            yh[masks[0]] = (proba[masks[0]] >= tau0).astype(int)
            yh[masks[1]] = (proba[masks[1]] >= tau1).astype(int)
            g  = fairness_gaps(y, yh, a)
            if g["EO_gap"] < best[2]:
                best = (tau0, tau1, g["EO_gap"])
    return best

tau0, tau1, eo_min = best_eo_thresholds(proba_te, Y[te], A[te])
pred_eo = np.zeros_like(pred)
pred_eo[A[te] == 0] = (proba_te[A[te] == 0] >= tau0).astype(int)
pred_eo[A[te] == 1] = (proba_te[A[te] == 1] >= tau1).astype(int)

eo_fixed = fairness_gaps(Y[te], pred_eo, A[te])
print(f"After per-group threshold to minimize EO gap:")
print(f"  accuracy={(pred_eo == Y[te]).mean():.3f}")
print(f"  DP_gap={eo_fixed['DP_gap']:.3f}  DI={eo_fixed['DI']:.3f}")
print(f"  EO_gap={eo_fixed['EO_gap']:.3f}  Cal_gap={eo_fixed['Cal_gap']:.3f}")
# After per-group threshold to minimize EO gap:
#   accuracy=0.808
#   DP_gap=0.171  DI=0.682   ← demographic parity worsens vs. baseline (DI=0.547→0.682)
#   EO_gap=0.012             ← driven near zero (from 0.073)
#   Cal_gap=0.094            ← worsens slightly (0.078 → 0.094)`}
      </CodeBlock>

      <Prose>
        The pattern is the same in the other direction. Driving EO gap toward zero shifts the disparate-impact ratio and changes the calibration gap. There is no free lunch, and the algebra of the impossibility theorem forbids one from existing. Engineers seeking to resolve this should not look for a clever post-processing trick that achieves all three; they should instead choose, for the use case at hand, which definition is the operationally correct one.
      </Prose>

      <H3>4g. The Chouldechova identity, verified numerically</H3>

      <Prose>
        The identity <Code>FPR · (1−p)/p = (1−PPV)/PPV · (1−FNR)</Code> is the algebraic root of the impossibility. Verify it holds on the baseline classifier within numerical precision.
      </Prose>

      <CodeBlock language="python">
{`for g, r in baseline["rates"].items():
    p, fpr, fnr, ppv = r["base_rate"], r["FPR"], r["FNR"], r["PPV"]
    lhs = fpr * (1 - p) / p
    rhs = (1 - ppv) / ppv * (1 - fnr)
    print(f"group {g}: lhs={lhs:.5f}  rhs={rhs:.5f}  "
          f"|lhs-rhs|={abs(lhs - rhs):.2e}")
# group 0: lhs=0.27091  rhs=0.27091  |lhs-rhs|=4.16e-17
# group 1: lhs=0.06914  rhs=0.06914  |lhs-rhs|=2.78e-17
# Identity holds exactly (up to float epsilon).
# Different lhs values per group ⇒ if you equalized any two of {FPR, FNR, PPV},
# the third would have to differ by exactly the gap implied by the identity.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Two libraries dominate production fairness tooling. Microsoft's <Code>fairlearn</Code> (github.com/fairlearn/fairlearn) is the lighter-weight option, integrates directly with scikit-learn pipelines, and is the default choice for tabular classification and regression. IBM's <Code>aif360</Code> (AI Fairness 360, github.com/Trusted-AI/AIF360) is heavier and more comprehensive, including in-processing methods like reductions to constrained optimization, and is more common in academic-leaning teams or large enterprises with formal model-risk-management functions. AWS SageMaker Clarify and Google's What-If Tool / Fairness Indicators (TensorFlow ecosystem) wrap these or equivalent metrics into managed services. For LLM-specific bias evaluation, BBQ (Bias Benchmark for QA, Parrish et al. 2022), HolisticBias, and Anthropic's evals harness all evaluate per-group disparities in model behavior, but the underlying metrics are the same DP, EO, calibration gaps formulated for the LLM setting.
      </Prose>

      <H3>Fairlearn: the standard tabular workflow</H3>

      <Prose>
        Fairlearn provides three things: a <Code>MetricFrame</Code> for computing per-group metrics, a set of pre-built fairness metrics (<Code>demographic_parity_difference</Code>, <Code>equalized_odds_difference</Code>, etc.), and reduction-style mitigation algorithms (<Code>ExponentiatedGradient</Code>, <Code>ThresholdOptimizer</Code>) that wrap any base estimator. The minimal audit looks like this.
      </Prose>

      <CodeBlock language="python">
{`from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# X, y, A: features, labels, sensitive attribute (Series or array).
clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
y_pred = clf.predict(X_test)

mf = MetricFrame(
    metrics={
        "accuracy":  accuracy_score,
        "selection": selection_rate,
        "TPR":       true_positive_rate,
        "FPR":       false_positive_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test,
)
print(mf.by_group)
# A     accuracy  selection  TPR    FPR
# 0     0.834     0.211      0.658  0.094
# 1     0.812     0.460      0.768  0.167

print("DP difference:",       demographic_parity_difference(y_test, y_pred, sensitive_features=A_test))
print("DP ratio (DI):",       demographic_parity_ratio(y_test, y_pred, sensitive_features=A_test))
print("Equalized odds diff:", equalized_odds_difference(y_test, y_pred, sensitive_features=A_test))`}
      </CodeBlock>

      <H3>Fairlearn: ThresholdOptimizer for post-processing</H3>

      <Prose>
        The <Code>ThresholdOptimizer</Code> implements Hardt-Price-Srebro post-processing for equalized odds (and several other constraint types). It wraps a fitted base estimator and finds per-group thresholds that satisfy the chosen fairness constraint while preserving as much accuracy as possible.
      </Prose>

      <CodeBlock language="python">
{`from fairlearn.postprocessing import ThresholdOptimizer

postproc = ThresholdOptimizer(
    estimator=clf,
    constraints="equalized_odds",     # or "demographic_parity", "true_positive_rate_parity", ...
    objective="accuracy_score",
    prefit=True,
)
postproc.fit(X_train, y_train, sensitive_features=A_train)
y_pred_pp = postproc.predict(X_test, sensitive_features=A_test)

print("Equalized odds diff after:",
      equalized_odds_difference(y_test, y_pred_pp, sensitive_features=A_test))
print("DP diff after:",
      demographic_parity_difference(y_test, y_pred_pp, sensitive_features=A_test))
# Equalized odds diff after: 0.011    ← driven near zero
# DP diff after:             0.182    ← unchanged or worse (impossibility in action)`}
      </CodeBlock>

      <H3>Fairlearn: ExponentiatedGradient for in-processing</H3>

      <Prose>
        For an in-processing approach (training the model itself with fairness constraints rather than post-processing a fixed model), use <Code>ExponentiatedGradient</Code> from <Code>fairlearn.reductions</Code>. This implements the Agarwal et al. (2018) reductions approach: solve the fair-classification problem as a sequence of weighted classification problems, where the weights are updated by a no-regret online learning rule until the fairness constraint is satisfied within a chosen tolerance.
      </Prose>

      <CodeBlock language="python">
{`from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=2000),
    constraints=EqualizedOdds(difference_bound=0.02),
    eps=0.02,
)
mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred_in = mitigator.predict(X_test)
# In-processing: the trained model itself satisfies EO within 0.02 tolerance.
# Cost: typically 1–3 percentage points of accuracy vs unconstrained baseline.`}
      </CodeBlock>

      <H3>AIF360: when you need the full taxonomy</H3>

      <Prose>
        AIF360 covers a wider set of metrics (over 70) and mitigations (over 10 in-processing, pre-processing, and post-processing algorithms). Its API is heavier (datasets are wrapped in <Code>BinaryLabelDataset</Code> objects) but its coverage of less-common fairness definitions — calibration within prediction bins, individual fairness via Lipschitz constraints, counterfactual fairness — is broader than Fairlearn's. For regulated industries that need to demonstrate consideration of multiple fairness criteria, AIF360 is the more defensible choice in audit settings.
      </Prose>

      <CodeBlock language="python">
{`from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Build dataset wrapper.
df_train["label"]     = y_train
df_train["sensitive"] = A_train
ds_train = BinaryLabelDataset(
    df=df_train,
    label_names=["label"],
    protected_attribute_names=["sensitive"],
    favorable_label=1,
    unfavorable_label=0,
)

# Privileged vs unprivileged groups.
priv   = [{"sensitive": 1}]
unpriv = [{"sensitive": 0}]

dataset_metric = BinaryLabelDatasetMetric(ds_train, unprivileged_groups=unpriv,
                                          privileged_groups=priv)
print("Disparate impact:",         dataset_metric.disparate_impact())
print("Statistical parity diff:",  dataset_metric.statistical_parity_difference())

# After predictions, classification-level metrics.
ds_pred         = ds_train.copy()
ds_pred.labels  = y_pred.reshape(-1, 1)
classification_metric = ClassificationMetric(ds_train, ds_pred,
                                             unprivileged_groups=unpriv,
                                             privileged_groups=priv)
print("Equal opportunity diff:", classification_metric.equal_opportunity_difference())
print("Avg odds difference:",    classification_metric.average_odds_difference())`}
      </CodeBlock>

      <H3>Choosing a fairness definition for the use case</H3>

      <Prose>
        The choice of fairness definition is a product and policy decision, not a technical one. The technical content is knowing what each choice implies. A practical decision framework, drawn from Barocas, Hardt, and Narayanan (Fairness and Machine Learning, 2019, Chapter 3): if the cost of a false positive falls primarily on the individual flagged (criminal-justice risk scoring, content-moderation strikes, hiring rejection), prioritize equalized odds — specifically the FPR equality condition — because group-asymmetric false-positive rates inflict group-asymmetric harm. If the cost of a false negative is the dominant fairness concern (medical screening, opportunity allocation), prioritize equality of opportunity (TPR equality). If the system's output is consumed downstream by humans who treat the score as a probability (loan officer reading a risk score, doctor reading a triage flag), prioritize calibration / predictive parity — otherwise the same number means different things in different groups, and downstream decisions inherit the disparity. If a regulator has explicitly defined the fairness criterion (EEOC four-fifths rule, NYC Local Law 144), use that one regardless of which one you would have picked otherwise.
      </Prose>

      <H3>Legal frameworks engineers should know</H3>

      <Prose>
        EEOC Uniform Guidelines on Employee Selection Procedures (29 CFR Part 1607): the four-fifths rule, applicable to any selection procedure used in employment decisions in the United States. Disparate impact below 0.8 is prima facie evidence of adverse impact and shifts the burden of proof to the employer to demonstrate business necessity. NYC Local Law 144 (Automated Employment Decision Tools, effective July 5, 2023): annual independent bias audits required for any AEDT used in hiring or promotion decisions in NYC, with public posting of the audit summary including selection rates and impact ratios. EU AI Act (Regulation 2024/1689, applicable from 2025–2026): high-risk AI systems (Annex III) must include bias evaluation, documentation of training-data composition, and post-market monitoring; Article 10 covers data governance and bias mitigation. UK Equality Act 2010: prohibits direct and indirect discrimination on protected characteristics and applies to algorithmic decisions through the indirect-discrimination route, which closely tracks disparate-impact analysis. Canadian AIDA (Artificial Intelligence and Data Act, pending as of 2026): requires impact assessments and mitigation for high-impact systems. Engineers building production fairness tooling should know which of these regulations applies to which deployment region; the technical metric you compute is largely the same, but the threshold and reporting requirements differ.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the trade-off curve directly. Sweep a per-group threshold pair from "match the global threshold" toward "equalize selection rates," and plot the DP gap and EO gap as a function of the sweep parameter. The two curves move in opposite directions: enforcing DP raises EO, and vice versa. The crossing point is not at zero for either — there is no point where both vanish.
      </Prose>

      <Plot
        label="DP gap vs EO gap as the per-group threshold ratio is varied"
        xLabel="threshold adjustment (group 1 - group 0)"
        yLabel="fairness gap"
        series={[
          {
            name: "DP gap",
            color: colors.gold,
            points: [
              [-0.4, 0.42],
              [-0.3, 0.36],
              [-0.2, 0.30],
              [-0.1, 0.26],
              [0.0,  0.249],
              [0.1,  0.18],
              [0.2,  0.10],
              [0.3,  0.04],
              [0.4,  0.001],
              [0.5,  0.05],
              [0.6,  0.13],
            ],
          },
          {
            name: "EO gap",
            color: "#c084fc",
            points: [
              [-0.4, 0.05],
              [-0.3, 0.04],
              [-0.2, 0.05],
              [-0.1, 0.06],
              [0.0,  0.073],
              [0.1,  0.11],
              [0.2,  0.16],
              [0.3,  0.23],
              [0.4,  0.31],
              [0.5,  0.40],
              [0.6,  0.48],
            ],
          },
          {
            name: "Cal gap",
            color: colors.green,
            points: [
              [-0.4, 0.06],
              [-0.3, 0.06],
              [-0.2, 0.07],
              [-0.1, 0.07],
              [0.0,  0.078],
              [0.1,  0.10],
              [0.2,  0.13],
              [0.3,  0.16],
              [0.4,  0.18],
              [0.5,  0.20],
              [0.6,  0.21],
            ],
          },
        ]}
      />

      <Prose>
        The second plot is the calibration curve, by group. A perfectly calibrated score has the predicted probability equal to the empirical positive rate at every score level, so the curve is the diagonal. When the underlying data has unequal base rates and the score is calibrated globally, the per-group calibration curves typically diverge from the diagonal in opposite directions: the higher-base-rate group's curve sits above the diagonal at low scores (the score under-predicts), and the lower-base-rate group's curve sits below at high scores (the score over-predicts). Bringing one group's curve to the diagonal pulls the other off it.
      </Prose>

      <Plot
        label="Calibration by group (predicted probability vs empirical positive rate)"
        xLabel="predicted probability"
        yLabel="empirical positive rate"
        series={[
          {
            name: "perfect calibration",
            color: colors.textDim,
            points: [
              [0, 0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0],
            ],
          },
          {
            name: "group A=0",
            color: colors.gold,
            points: [
              [0.05, 0.06],
              [0.20, 0.18],
              [0.40, 0.34],
              [0.60, 0.51],
              [0.80, 0.72],
              [0.95, 0.88],
            ],
          },
          {
            name: "group A=1",
            color: "#c084fc",
            points: [
              [0.05, 0.09],
              [0.20, 0.27],
              [0.40, 0.46],
              [0.60, 0.66],
              [0.80, 0.84],
              [0.95, 0.97],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap shows the per-group confusion matrix as fractions of group population. Each row is a group; each cell is one of the four confusion-matrix outcomes (TP, FP, FN, TN). The asymmetry between rows is the visual signature of a fairness gap. Equal cells across rows would mean perfect demographic + equalized-odds + predictive parity, which under unequal base rates is impossible.
      </Prose>

      <Heatmap
        label="Per-group confusion-matrix fractions (baseline classifier)"
        rowLabels={["A=0", "A=1"]}
        colLabels={["TP", "FP", "FN", "TN"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [0.158, 0.053, 0.138, 0.651],
          [0.402, 0.073, 0.144, 0.381],
        ]}
      />

      <Prose>
        The step trace below walks through one full fairness audit, in the order an engineer would actually run it. Each step is what a production audit script does, in roughly the order Fairlearn or AIF360 would compute under the hood.
      </Prose>

      <StepTrace
        label="Fairness audit — one full pass over a held-out evaluation set"
        steps={[
          {
            label: "Compute per-group confusion matrices",
            render: () => (
              <Prose>
                For each value <Code>a</Code> of the protected attribute, build the 2×2 matrix of <Code>(TP, FP, FN, TN)</Code> from <Code>(Y, Ŷ)</Code> restricted to rows where <Code>A = a</Code>. This is the only data structure the rest of the audit reads from. Empty cells (e.g., a group with no predicted positives) are flagged here so downstream rates are reported as NaN rather than crashing on division by zero.
              </Prose>
            ),
          },
          {
            label: "Derive per-group rates",
            render: () => (
              <Prose>
                Compute base rate <Code>p_a</Code>, selection rate <Code>P(Ŷ=1|A=a)</Code>, TPR, FPR, FNR, TNR, PPV, NPV from each group's confusion matrix. Verify the Chouldechova identity numerically as a sanity check that the rates were computed correctly: <Code>FPR · (1−p)/p = (1−PPV)/PPV · (1−FNR)</Code> should hold per group within float epsilon.
              </Prose>
            ),
          },
          {
            label: "Compute scalar fairness gaps",
            render: () => (
              <Prose>
                <Code>DP_gap = max(sel_rate) − min(sel_rate)</Code>; <Code>DI = min/max</Code>. <Code>EO_gap = max(TPR_gap, FPR_gap)</Code>. <Code>Cal_gap = max(PPV) − min(PPV)</Code>. Optionally also report <Code>EOpp_gap = TPR_gap</Code> for equality-of-opportunity audits. For multi-valued <Code>A</Code>, take pairwise max.
              </Prose>
            ),
          },
          {
            label: "Compare against thresholds",
            render: () => (
              <Prose>
                Check <Code>DI ≥ 0.8</Code> for EEOC four-fifths compliance. Check internal thresholds (often <Code>DP_gap ≤ 0.05</Code>, <Code>EO_gap ≤ 0.05</Code>) per the team's fairness policy. Flag any threshold violation; the audit summary shows which definition is satisfied and which is not. Critically: if all three are violated simultaneously, document which one will be prioritized for mitigation, with explicit acknowledgment of the impossibility for the other two.
              </Prose>
            ),
          },
          {
            label: "Run mitigation if needed",
            render: () => (
              <Prose>
                If a chosen fairness criterion is violated, apply post-processing (<Code>ThresholdOptimizer</Code>) or in-processing (<Code>ExponentiatedGradient</Code>) targeting that criterion. After mitigation, re-run the full audit. Expect the targeted gap to drop and the other gaps to grow — verify the new state matches the documented choice and that the regression on the other gaps is within the tolerance the team agreed to.
              </Prose>
            ),
          },
          {
            label: "Bootstrap for uncertainty",
            render: () => (
              <Prose>
                Fairness gaps are sample statistics; small evaluation sets produce noisy gaps. Resample the evaluation set with replacement <Code>B</Code> times (typically <Code>B = 1000</Code>), recompute each gap on each bootstrap sample, and report the gap with a 95% confidence interval. A DP gap of 0.05 with a CI of [0.01, 0.09] is a different audit conclusion than 0.05 with CI [0.04, 0.06]. Decisions made on point estimates without uncertainty are unstable.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Demographic parity vs equalized odds</H3>

      <Prose>
        Choose demographic parity when the regulatory or normative constraint is on the selection rate itself, independent of accuracy. EEOC employment law, lending fairness under ECOA disparate-impact analysis, and many advertising-fairness frameworks operate at the level of "who gets selected, at what rate." The choice acknowledges that in some contexts the unequal base rates themselves reflect upstream injustice and the system should not reproduce them at the output stage. The cost is loss of accuracy: enforcing DP on a calibrated score requires deliberately accepting more false positives in the lower-base-rate group, more false negatives in the higher-base-rate group, or both.
      </Prose>

      <Prose>
        Choose equalized odds when the cost of error falls on individuals and the relevant fairness question is "given my true label, am I treated symmetrically across groups?" In criminal-justice risk scoring, in medical screening, in fraud detection — anywhere a false positive or false negative inflicts individual harm — equalized odds is the natural criterion. The asymmetry of harm matters: equality of opportunity (TPR-only) is appropriate when false negatives are the primary harm (qualified candidate missed); equal-FPR-only is appropriate when false positives are the primary harm (innocent person flagged). Equalized odds is the conjunction of both. The cost is loss of calibration: enforcing EO on a calibrated score breaks calibration within at least one group.
      </Prose>

      <H3>Predictive parity vs equalized odds</H3>

      <Prose>
        These two are the COMPAS dispute, in distilled form. Choose predictive parity when downstream decision-makers consume the score as a probability and treat scores at the same level as carrying the same meaning regardless of group. A loan officer reading a credit risk score, a doctor reading a triage flag, an emergency-room nurse reading a sepsis-risk score — all of these treat the score as a calibrated probability. If the score is not calibrated within group, the same numerical score means different things in different groups, and the human's decision inherits that disparity in a hidden way. Choose equalized odds when the score is consumed as a binary classification and the fairness question is about how often the system errs on each group. As Chouldechova proved, the two are not jointly satisfiable when base rates differ.
      </Prose>

      <H3>Pre-processing vs in-processing vs post-processing</H3>

      <Prose>
        Pre-processing methods modify the training data (reweighting, resampling, learned representations like LFR or DI Remover from AIF360) before the model is trained. Pros: they decouple fairness from the model class, work with any downstream learner, are interpretable as data interventions. Cons: they often achieve weaker fairness guarantees than in-processing, and the modified data may not be appropriate for downstream uses beyond the immediate model. In-processing methods modify the training objective with fairness constraints (Agarwal reductions / <Code>ExponentiatedGradient</Code>, Zafar et al.'s convex-constrained logistic regression, adversarial debiasing). Pros: tightest joint optimization of accuracy and fairness, usually the highest accuracy at a given fairness target. Cons: tied to the model class, harder to audit, more difficult for compliance teams to inspect. Post-processing methods adjust the final classifier (per-group thresholds, score calibration). Pros: easiest to implement, easiest to audit (the change is a single threshold table), works with black-box models, and is what the EU AI Act Article 15 implicitly anticipates for downstream calibration. Cons: needs access to the protected attribute at inference time, which is sometimes legally or operationally forbidden, and achieves only the fairness criterion targeted by the post-processing.
      </Prose>

      <H3>Group fairness vs individual fairness</H3>

      <Prose>
        Group fairness, the entire focus of this topic so far, is about per-group statistics. Individual fairness, articulated by Dwork, Hardt, Pitassi, Reingold, and Zemel (2012, "Fairness Through Awareness"), is the principle that "similar individuals should be treated similarly" — formalized as a Lipschitz condition on the classifier with respect to a task-specific similarity metric. The two are not directly comparable: group fairness is a statistical condition on aggregates, individual fairness is a per-pair condition on instances. They sometimes conflict (a Lipschitz classifier may violate group fairness if the similarity metric does not capture group-relevant features) and sometimes complement (group fairness gives weak protection to outliers, individual fairness gives protection without reference to group). In practice, individual fairness has been harder to deploy because it requires specifying the similarity metric, which is itself a fraught modeling choice. Group fairness has won the production-tooling battle, but individual-fairness ideas show up in counterfactual fairness (Kusner et al. 2017) and in causal-fairness frameworks.
      </Prose>

      <H3>When the four-fifths rule is the bar to clear</H3>

      <Prose>
        For US employment decisions, the four-fifths rule is the de facto operational target. Aim for <Code>DI ≥ 0.85</Code> as an internal goal (some buffer above 0.80 to absorb sampling noise). Compute DI on the production deployment population, not just the held-out test set, since adverse-impact analysis is performed on actual selection outcomes. Document the audit annually for NYC Local Law 144 compliance. Note that the four-fifths rule is a screening threshold, not a safe harbor: a system at DI = 0.85 can still be challenged if a less-discriminatory alternative exists with comparable utility ("alternative employment practice" doctrine under Title VII).
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Fairness measurement scales beautifully. The DP gap, EO gap, and calibration gap are all <Code>O(n)</Code> per group to compute and additive across batches, so streaming fairness audits over millions of inference rows are straightforward. Bootstrap confidence intervals add a constant factor; per-group rates can be computed in a single pass with running counters. For multi-valued protected attributes the cost is <Code>O(|A|)</Code> for the per-group rates and <Code>O(|A|²)</Code> for pairwise gap reports, both of which are negligible relative to inference cost.
      </Prose>

      <Prose>
        Mitigation scales less well. Post-processing methods (<Code>ThresholdOptimizer</Code>) require access to the protected attribute at inference time, which is operationally awkward when the attribute is sensitive (race, gender, religion) and may be legally restricted at the inference path. They also require a held-out calibration set proportional in size to the smallest group, since the per-group threshold must be estimated from that group's data alone. In-processing methods (<Code>ExponentiatedGradient</Code>) increase training cost by a multiplicative factor — the reductions approach solves a sequence of weighted classification problems, typically 50–200 iterations to converge, each iteration a full retraining pass. For deep learning, this multiplier is prohibitive on top of standard training; adversarial debiasing or constrained gradient updates are usually preferred at scale. Pre-processing methods are cheap at training but sometimes provide weaker fairness guarantees, especially when the downstream model is highly expressive and can re-learn the original disparities from any features correlated with the protected attribute.
      </Prose>

      <Prose>
        The structural limitation that does not scale away is base-rate inequality. As long as the empirical base rates differ across groups, the impossibility theorems hold, and adding more data, larger models, or better features cannot fix that. What more data can do is reduce the variance of the gap estimates, make mitigation methods more reliable on smaller groups, and surface intersectional disparities that smaller datasets cannot detect. None of that changes the underlying impossibility geometry. The only way around the impossibility is to change the data — to address the upstream causes of base-rate differences, which is a sociological intervention rather than a machine-learning one.
      </Prose>

      <Prose>
        Intersectional fairness scales poorly with the number of protected attributes. With <Code>k</Code> binary protected attributes, the number of intersectional cells is <Code>2^k</Code>, and per-cell sample sizes shrink exponentially. Buolamwini and Gebru's "Gender Shades" identified large error disparities at the (race × sex) intersection that were obscured at either marginal level; replicating that analysis at (race × sex × age × disability × ...) requires impractical sample sizes for all but the largest deployments. A common practical compromise is to audit at the marginal level for all relevant attributes plus a small number of explicitly enumerated intersections that are operationally salient.
      </Prose>

      <Prose>
        For LLM judges and content-moderation systems, fairness measurement faces a specific challenge: the "groups" are defined by content rather than by user attributes. A toxicity classifier evaluated on prompts referencing different demographic groups defines its protected attribute through the text itself, which is noisier than a tabular sensitive feature. The standard approach (BBQ, HolisticBias) uses templated prompts that hold everything constant except the demographic referent, which controls for confounding but limits coverage to the templated patterns. Production fairness audits of LLM systems should pair templated benchmarks with audits on production traffic stratified by classified user-group features, accepting that the latter is noisier but covers the actual deployment distribution.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Optimizing for the wrong fairness definition</H3>
      <Prose>
        The single most consequential failure mode. A team that optimizes for demographic parity in a context where equalized odds is the legally relevant criterion produces a system that is "fair" by their internal metric and discriminatory by the regulator's. The reverse can also occur. Always document explicitly which fairness definition is being optimized, why it is the appropriate choice for the use case, and what the gaps on the other definitions are. Auditors and regulators are increasingly sophisticated; producing only a single fairness number without acknowledging the trade-off is a red flag.
      </Prose>

      <H3>Removing the protected attribute does not eliminate disparity</H3>
      <Prose>
        "Fairness through unawareness" — simply dropping the protected attribute from the feature vector — is a reliably failed strategy. Any feature correlated with the protected attribute can serve as a proxy: ZIP code is a proxy for race, name is a proxy for gender and ethnicity, browsing history is a proxy for nearly any demographic. A model trained without the protected attribute but with proxies can produce identical disparities to a model trained with it. The Dwork et al. (2012) "Fairness Through Awareness" paper argues for the opposite: fairness sometimes requires explicit awareness of the protected attribute at training time so the model can be constrained against the disparity it would otherwise produce.
      </Prose>

      <H3>Sample size shrinks per group; gap estimates are noisy</H3>
      <Prose>
        With a held-out test set of 1000 examples and a 5%/95% group split, the smaller group has 50 examples. Per-group rates computed on 50 examples have standard errors around 0.05–0.07; a reported DP gap of 0.05 is statistically indistinguishable from zero. Always compute bootstrap or analytical confidence intervals for fairness gaps. Decisions made on point estimates from small groups are unstable across resamples and can flip between deployments. The minimum operational threshold most teams use is 200–500 examples per group for stable rate estimates; below that, the audit is qualitatively rather than quantitatively meaningful.
      </Prose>

      <H3>Calibration drift after equalized-odds enforcement</H3>
      <Prose>
        ThresholdOptimizer with equalized-odds constraints achieves the EO target by per-group thresholding, which mechanically breaks calibration. Downstream consumers of the score who treat it as a probability — risk officers, doctors, loan underwriters — will see different actual outcomes per group at the same score level after mitigation. Either re-calibrate per group after thresholding (which restores predictive parity but breaks the EO guarantee, illustrating the impossibility from a different angle), or document the calibration breakage and ensure downstream consumers are aware. The most common production failure: a team enforces EO at the model layer, deploys, and a downstream business team uses the now-uncalibrated score for resource allocation as if it were calibrated, recreating the disparity at the consumption layer.
      </Prose>

      <H3>Off-the-shelf datasets have known fairness gaps that propagate</H3>
      <Prose>
        Many widely used benchmark datasets have well-documented base-rate differences across protected groups: COMPAS, German Credit, Adult Income, Bank Marketing, Heritage Health Prize. Fine-tuning a model on one of these datasets without addressing the base-rate structure transfers the impossibility to your model. Similarly, public preference datasets used to train LLMs (HH-RLHF, UltraFeedback) have known stylistic and demographic correlations in their preferences that propagate to DPO-trained models. The fairness audit cannot start at the model; it has to start at the data.
      </Prose>

      <H3>Group definition itself is contested</H3>
      <Prose>
        The choice of protected attribute and its categorization is itself a modeling choice with fairness implications. Race in the US census has changed categories repeatedly; gender as a binary is contested; age bands are arbitrary cuts on a continuous variable. A fairness audit that uses a coarser categorization than the deployment population's actual diversity can hide intra-group disparities. A finer categorization may have insufficient sample sizes per cell. Document the categorization choice, justify it for the use case, and where possible report at multiple granularities so the impact of the categorization choice is visible.
      </Prose>

      <H3>Silent feedback loops in deployed systems</H3>
      <Prose>
        Many fairness audits are static: they compute gaps on a fixed evaluation set at training time. Deployed systems generate feedback that re-enters the training pipeline. A predictive policing system that flags neighborhoods at higher rates produces more arrests in those neighborhoods, which becomes more "ground truth" reoffense data that confirms the original prediction. Fairness drift over time can substantially exceed the gaps measured at deployment, and the drift mechanism is the system itself. Re-audit deployed systems against fresh data on a regular schedule (quarterly is a common cadence for high-stakes systems), and explicitly model the feedback structure where applicable.
      </Prose>

      <H3>Confusing aggregate accuracy with per-group accuracy</H3>
      <Prose>
        A model with 92% aggregate accuracy can have 96% accuracy on the majority group and 78% on the minority group, with the aggregate dominated by the majority. This is structurally similar to Simpson's paradox. Always report per-group accuracy alongside aggregate accuracy, and compare per-group accuracy to the per-group baseline (predict-majority) rather than to the aggregate. A model whose minority-group accuracy is below the predict-majority baseline for that group is, in a meaningful sense, harming members of that group while appearing to help on aggregate.
      </Prose>

      <H3>Gerrymandering: optimizing for the audited gap, not the underlying fairness</H3>
      <Prose>
        Hébert-Johnson, Kim, Reingold, and Rothblum (2018, "Multicalibration") showed that satisfying calibration within a small set of pre-defined groups can be achieved while still being grossly miscalibrated on subgroups defined by the intersection of those groups or by other features. A model can pass a single-axis fairness audit while violating fairness on every meaningfully-defined subgroup. This is fairness gerrymandering, and the response is multi-group fairness frameworks (multicalibration, multiaccuracy) that demand calibration on a rich class of computationally identifiable subgroups rather than a fixed list of demographic categories. For high-stakes deployments, a single-axis audit is the floor, not the ceiling.
      </Prose>

      <Callout accent="purple">
        Fairness fails silently. A model that has been "made fair" against one definition can be more discriminatory against another, and the mitigation step often produces no visible warning. Always re-audit against the full set of definitions after any mitigation, and document the explicit choice that was made.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their canonical pages on 2026-04-26. Citations include arXiv IDs or journal references where applicable.
      </Prose>

      <H3>Chouldechova 2017 — Fair Prediction with Disparate Impact</H3>
      <Prose>
        Alexandra Chouldechova. "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments." Big Data 5(2): 153–163, 2017. arXiv:1610.07524 (preprint). Derives the algebraic identity relating FPR, FNR, PPV, and base rate; proves that under unequal base rates, predictive parity (equal PPV) and equalized error rates (equal FPR and FNR) cannot both hold for non-trivial classifiers. The cleanest exposition of the impossibility result with explicit reference to the COMPAS controversy.
      </Prose>

      <H3>Kleinberg, Mullainathan, Raghavan 2016 — Inherent Trade-Offs</H3>
      <Prose>
        Jon Kleinberg, Sendhil Mullainathan, Manish Raghavan. "Inherent Trade-Offs in the Fair Determination of Risk Scores." arXiv:1609.05807, 2016. Presented at ITCS 2017. Proves the stronger impossibility theorem: calibration within groups, balance for the positive class, and balance for the negative class are jointly satisfiable only by perfect predictors or under equal base rates. The proof structure (linear in <Code>p</Code>) is short and is the canonical reference for the multi-criterion impossibility.
      </Prose>

      <H3>Hardt, Price, Srebro 2016 — Equality of Opportunity</H3>
      <Prose>
        Moritz Hardt, Eric Price, Nathan Srebro. "Equality of Opportunity in Supervised Learning." arXiv:1610.02413, 2016. NeurIPS 2016. Formalizes equalized odds and equality of opportunity, presents the post-processing algorithm for enforcing equalized odds via per-group thresholding, and shows that calibration-preserving equalized-odds enforcement is generally impossible. The post-processing algorithm is the basis for Fairlearn's <Code>ThresholdOptimizer</Code>.
      </Prose>

      <H3>Dwork, Hardt, Pitassi, Reingold, Zemel 2012 — Fairness Through Awareness</H3>
      <Prose>
        Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Richard Zemel. "Fairness Through Awareness." Innovations in Theoretical Computer Science (ITCS) 2012. arXiv:1104.3913. Foundational paper introducing individual fairness as a Lipschitz condition with respect to a task-specific similarity metric, and arguing that effective fairness intervention often requires explicit use of the protected attribute rather than its removal. Establishes the conceptual distinction between group and individual fairness.
      </Prose>

      <H3>Barocas, Hardt, Narayanan 2019 — Fairness and Machine Learning</H3>
      <Prose>
        Solon Barocas, Moritz Hardt, Arvind Narayanan. "Fairness and Machine Learning: Limitations and Opportunities." Open-access textbook, fairmlbook.org. The standard reference text for the field, covering the three families (independence / separation / sufficiency) in unified notation, the impossibility theorems, mitigation methods, and broader social context. Chapter 3 on classification gives the cleanest presentation of how the three families relate. Continuously updated; the 2023 MIT Press edition is the citable hardcopy.
      </Prose>

      <H3>Angwin, Larson, Mattu, Kirchner 2016 — ProPublica COMPAS investigation</H3>
      <Prose>
        Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner. "Machine Bias." ProPublica, May 23, 2016. The empirical investigation that surfaced the COMPAS fairness controversy; documents per-race FPR and FNR disparities. Northpointe's response, "COMPAS Risk Scales: Demonstrating Accuracy Equity and Predictive Parity" (Dieterich, Mendoza, Brennan, 2016), demonstrates predictive parity. The pair of documents is the canonical real-world manifestation of the impossibility result.
      </Prose>

      <H3>Agarwal, Beygelzimer, Dudík, Langford, Wallach 2018 — Reductions approach</H3>
      <Prose>
        Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, Hanna Wallach. "A Reductions Approach to Fair Classification." ICML 2018. arXiv:1803.02453. Formulates fair classification as a constrained optimization problem reduced to a sequence of weighted classifications, solvable with any base estimator. The basis for Fairlearn's <Code>ExponentiatedGradient</Code> in-processing mitigator.
      </Prose>

      <H3>Buolamwini, Gebru 2018 — Gender Shades</H3>
      <Prose>
        Joy Buolamwini, Timnit Gebru. "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification." FAT* 2018. Empirically demonstrates that commercial face-classification systems have largest error disparities at the intersection of race and sex, with darker-skinned women receiving the worst performance. The canonical empirical demonstration of why intersectional fairness matters and why single-axis audits are insufficient.
      </Prose>

      <H3>Hébert-Johnson, Kim, Reingold, Rothblum 2018 — Multicalibration</H3>
      <Prose>
        Úrsula Hébert-Johnson, Michael Kim, Omer Reingold, Guy Rothblum. "Multicalibration: Calibration for the (Computationally-Identifiable) Masses." ICML 2018. arXiv:1711.08513. Introduces multicalibration as a strengthening of group calibration that requires the predictor to be calibrated on a rich class of subgroups identifiable by a computational class, addressing the fairness gerrymandering problem in single-axis audits.
      </Prose>

      <H3>Regulatory references</H3>
      <Prose>
        EEOC Uniform Guidelines on Employee Selection Procedures, 29 CFR Part 1607, especially §1607.4D (the four-fifths rule). NYC Local Law 144 of 2021 (Automated Employment Decision Tools), in force July 5, 2023. EU AI Act, Regulation (EU) 2024/1689, applicable from 2025 onward; high-risk system requirements in Annex III; data governance in Article 10. Equality Act 2010 (UK), especially the indirect-discrimination provisions. These are the load-bearing legal frameworks an engineer should be able to cite when justifying a fairness-definition choice in a regulated deployment.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the Chouldechova identity</H3>
      <Prose>
        Starting from a 2×2 confusion matrix with cells <Code>TP, FP, FN, TN</Code> and group base rate <Code>p = (TP + FN) / n</Code>, derive the identity <Code>FPR · (1−p)/p = (1−PPV)/PPV · (1−FNR)</Code>. Make every algebraic step explicit. Then use the identity to show that for two groups with <Code>p_0 = 0.3</Code> and <Code>p_1 = 0.6</Code>, if both groups have <Code>PPV = 0.7</Code> and <Code>FNR = 0.2</Code>, the FPRs cannot be equal — compute both FPRs explicitly and report the difference. What does this difference correspond to in the impossibility theorem?
      </Prose>

      <H3>Exercise 2 — Simulate a base-rate sweep</H3>
      <Prose>
        Modify the synthetic data generator from section 4 so that the base rate of group <Code>A=1</Code> is parameterized: sweep <Code>p_1</Code> from 0.30 to 0.80 in steps of 0.05, holding <Code>p_0 = 0.30</Code> fixed. For each sweep value, train the same logistic regression and record the three fairness gaps (DP, EO, calibration). Plot all three as functions of <Code>p_1</Code>. At what value of <Code>p_1</Code> do all three gaps approach zero? Explain why, and connect your observation back to the Kleinberg-Mullainathan-Raghavan theorem.
      </Prose>

      <H3>Exercise 3 — Bootstrap a fairness gap</H3>
      <Prose>
        Take the held-out test predictions from the baseline classifier (section 4d). Compute a bootstrap 95% confidence interval for the DP gap using <Code>B = 1000</Code> resamples of the test set. Report the point estimate and the CI. Now do the same for a sub-sampled test set with only 200 examples. How do the two CIs compare? What is the operational implication: at what evaluation-set size does the DP gap CI become wider than the gap itself, making the audit conclusion uncertain? Recommend a minimum per-group sample size for stable fairness reporting.
      </Prose>

      <H3>Exercise 4 — Equalized odds vs equality of opportunity</H3>
      <Prose>
        Construct two scenarios: (a) a hiring screen where the cost of a false negative (missing a qualified candidate) dominates the cost of a false positive (interviewing an unqualified candidate); (b) a content-moderation system where the cost of a false positive (taking down legitimate content) dominates the cost of a false negative (missing harmful content). For each scenario, argue which fairness criterion — equalized odds, equality of opportunity (TPR-only), or equal-FPR-only — is the appropriate target, and why. What gap would you accept on the non-target criterion, and how would you defend that trade-off to a regulator?
      </Prose>

      <H3>Exercise 5 — The four-fifths rule in practice</H3>
      <Prose>
        Suppose a hiring screen has the following selection rates per group: <Code>women: 0.18</Code>, <Code>men: 0.27</Code>, <Code>non-binary: 0.21</Code>. Compute the disparate-impact ratio relative to the group with the highest selection rate. Does the screen pass the four-fifths rule? Now suppose you can shift the threshold to bring <Code>women</Code>'s selection rate to 0.22 at a cost of 1.5 percentage points of overall accuracy. Compute the new DI ratio. Does this pass? What other fairness audits would you run before deploying this modified screen, and what would you expect to see on each given the impossibility theorem?
      </Prose>

      <H3>Exercise 6 — Detecting a proxy feature</H3>
      <Prose>
        You are given a model that does not include race as an input feature, yet exhibits a DP gap of 0.18 on a held-out evaluation set. List three diagnostic procedures you would run to identify which input features are acting as proxies for race. For each procedure, describe what it computes, what a positive finding looks like, and what mitigation it would suggest. As a follow-up: would dropping the proxy features fully eliminate the disparity? Justify your answer with reference to the Dwork et al. "Fairness Through Awareness" argument.
      </Prose>

      <H3>Exercise 7 — Multicalibration and gerrymandering</H3>
      <Prose>
        Construct a hypothetical example where a model is calibrated within group <Code>A</Code> at every score level (passing the predictive-parity audit) but is grossly miscalibrated on the subgroup <Code>(A=0, B=1)</Code> for some other binary feature <Code>B</Code>. Sketch the per-(A, B) calibration table that would produce this. What does this example imply about the limits of single-axis fairness audits? Describe what a multicalibration audit would compute differently, and why it would catch this case where a single-axis audit would not.
      </Prose>

      <H3>Exercise 8 — LLM judge fairness</H3>
      <Prose>
        You have an LLM-based toxicity judge that flags responses as harmful or safe. You audit it on prompts that vary only in the demographic group referenced (using a HolisticBias-style template), and find that flag rates differ by 0.12 across groups. Walk through how each of demographic parity, equalized odds, and predictive parity would be operationalized for this evaluation. Which of the three is most relevant to the deployment goal of "the judge should not over-flag content about any group"? What additional data would you need to collect to evaluate equalized odds (which requires ground-truth labels), and how would you handle the situation if such labels are unavailable or themselves contested?
      </Prose>

    </div>
  ),
};

export default algorithmicFairness;
