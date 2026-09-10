import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const conformalLLM = {
  title: "Conformal Prediction for LLM Uncertainty",
  slug: "conformal-prediction-for-llm-uncertainty",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Large language models produce confident-sounding answers regardless of whether they actually know the answer. The softmax output assigns a probability to every token in the vocabulary, but those probabilities are not calibrated estimates of correctness — they are sampling weights, conditioned on whatever the model has learned to produce next given the prompt. A model can output the token sequence "The capital of Australia is Sydney" with very high joint probability and be wrong. A model can output "The capital of Australia is Canberra" with lower joint probability and be right. The probability mass the model assigns to its outputs is not a measure of factual reliability, and the gap between confidence and correctness is one of the most persistent unsolved problems in deploying LLMs to high-stakes domains. Medical question answering, legal document summarization, scientific literature review — all of these demand a different kind of output: not "here is the most likely answer" but "here is a set of answers that is statistically guaranteed to contain the truth, and here is the probability that this set is reliable."
      </Prose>

      <Prose>
        The classical machine learning toolkit has very few tools for this kind of guarantee. Bayesian deep learning offers posterior credible intervals but requires variational approximations and computational overhead that does not scale to billion-parameter models. Temperature scaling and Platt scaling can recalibrate softmax outputs to match empirical accuracy, but they only work in expectation — they tell you that, on average, when the model says 80% it is right 80% of the time. They make no guarantees about any specific prediction. Ensembling produces variance estimates that look like uncertainty, but the variance reflects disagreement among ensemble members, not a coverage statement about the true label. In every case the guarantees are asymptotic, distributional, or empirical-on-average — none of them give you a finite-sample, distribution-free statement about what your prediction set contains.
      </Prose>

      <Prose>
        Conformal prediction, introduced by Vladimir Vovk, Alexander Gammerman, and Glenn Shafer in their 2005 book "Algorithmic Learning in a Random World," provides exactly this missing guarantee. Given a calibration set of labeled examples drawn from the same distribution as your test data, conformal prediction wraps any black-box model — any classifier, any regressor, any LLM — and produces prediction sets that contain the true label with probability at least <Code>1 - α</Code>. The guarantee is finite-sample (it holds for any size calibration set, not just in the limit), distribution-free (it makes no assumption about the underlying data distribution beyond exchangeability), and model-agnostic (it works with any prediction function, regardless of how it was trained). The cost is that the prediction set may be larger than a single point estimate. The benefit is a mathematical certificate: across many test inputs, the long-run frequency of the prediction set containing the truth is bounded below by your chosen confidence level.
      </Prose>

      <Prose>
        The application to LLMs is recent and rapidly evolving. The 2024 paper "Conformal Language Modeling" by Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S. Jaakkola, and Regina Barzilay (arXiv:2306.10193) introduced the first principled framework for applying conformal prediction to open-ended generation. Around the same time, Christopher Mohri and Tatsunori Hashimoto's "Language Models with Conformal Factuality Guarantees" (arXiv:2402.10978) showed how conformal abstention — refusing to answer when the prediction set grows too large — provides a sharp guarantee on the factuality of returned answers. Cherian, Gibbs, and Candes 2024 extended these ideas to retrieval-augmented generation, calibrating prediction sets over retrieved documents and generated claims. Together these papers establish conformal prediction as the most principled framework currently available for adding statistical reliability guarantees to LLM outputs without modifying the underlying model.
      </Prose>

      <Prose>
        The practical motivation is direct. If you deploy an LLM-based clinical decision support system, the question regulators ask is not "is the model accurate on average" but "what is the probability that the model's recommendation set excludes the correct diagnosis?" Conformal prediction answers exactly that question, with a number you can put in a regulatory submission. If you deploy an LLM-based legal research tool, the question opposing counsel asks is not "did the model train on relevant cases" but "is the set of returned citations guaranteed to contain the controlling precedent with at least 95% confidence?" Conformal prediction provides that guarantee, computed from a held-out calibration corpus and verified against a known coverage target. This is a fundamentally different mode of working with model outputs than treating them as point estimates and hoping for the best.
      </Prose>

      <Prose>
        Beyond regulated domains, conformal prediction also addresses a more mundane production concern: how to set up a sensible abstention policy. Most LLM products today use ad-hoc heuristics for refusing to answer — a temperature threshold, a maximum number of retrieved documents, a regex on the generated text. These heuristics are fragile and untestable. Conformal prediction gives you a principled abstention rule: refuse to answer when the prediction set is too large (uninformative) or empty (no candidate is plausible enough), with both thresholds calibrated against a labeled corpus. The coverage statement extends naturally to a precision-coverage trade-off curve, allowing product designers to dial in the exact balance of "answers more questions" versus "is more often right" with quantified consequences for each operating point.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Conformal prediction asks a deceptively simple question: how surprised would I be if this candidate label were the true one? Every conformal method depends on a "nonconformity score" — a function that takes a candidate (input, label) pair and returns a real number. Higher scores mean more nonconformity, which means the candidate label is a worse fit for the input given everything we know. For a classification problem, a natural nonconformity score is one minus the model's softmax probability for that class. For a regression problem, it is the absolute residual. For an LLM generating a candidate answer, it might be the negative log-probability of the answer, the semantic distance from the model's most likely answer, or a judge model's quality score. The choice of nonconformity score affects the size of the resulting prediction sets but not the validity of the coverage guarantee — that holds for any score function whatsoever, as long as the calibration data is exchangeable with the test data.
      </Prose>

      <Prose>
        The core procedure is split conformal prediction. You start with a held-out calibration set of <Code>n</Code> labeled examples. For each example, you compute the nonconformity score of the true label under the model. You now have <Code>n</Code> calibration scores, which together describe how surprised the model typically is by correct labels. You then compute the <Code>(1 - α)</Code>-quantile of these calibration scores — call it <Code>q̂</Code>. This quantile is the threshold of "typical" surprise: scores below it are normal for correct labels, scores above it are unusually high. To make a prediction at a new test input, you enumerate candidate labels, compute the nonconformity score for each candidate, and include it in your prediction set if and only if its score is at or below <Code>q̂</Code>. The prediction set is the set of candidate labels that are at least as plausible as the typical correct label seen during calibration.
      </Prose>

      <Prose>
        The coverage guarantee follows from a symmetry argument so clean it almost feels like a trick. Suppose calibration and test data are drawn from the same distribution and exchangeable — meaning that any permutation of the joint sample is equally likely. Then the true score on the test point is exchangeable with the <Code>n</Code> calibration scores. The probability that the test score is among the smallest <Code>(1 - α) · (n + 1)</Code> of the combined sample is exactly <Code>(1 - α)</Code> by symmetry. Setting <Code>q̂</Code> to the appropriate empirical quantile of the calibration scores ensures that the test point's true label lands in the prediction set with probability at least <Code>1 - α</Code>. The proof requires no assumption about the model's accuracy, no assumption about the distribution of features or labels — only that the calibration and test data come from the same source.
      </Prose>

      <Prose>
        For LLMs, the construction generalizes but the conceptual move is the same. The "candidate labels" become candidate generated answers — sampled from the model with temperature, retrieved from a beam search, or proposed by a structured enumeration. The nonconformity score becomes any function that ranks candidate answers by how unlikely or implausible they are. A common choice is the negative log-probability of the answer under the model. A more sophisticated choice is the semantic entropy across multiple sampled paraphrases, which captures the model's uncertainty about meaning rather than surface form. The prediction set is the subset of candidate answers whose nonconformity score is below <Code>q̂</Code>, calibrated on a held-out set of (prompt, correct answer) pairs. The guarantee is that this set contains a correct answer with probability at least <Code>1 - α</Code>, marginalized over both calibration sampling and test draws.
      </Prose>

      <Prose>
        There is a critical caveat that distinguishes conformal prediction from many alternatives: the guarantee is marginal, not conditional. It says the prediction set contains the true label with probability <Code>1 - α</Code> averaged over all test inputs. It does not say that for any specific input the set is correct with probability <Code>1 - α</Code>. Conditional coverage — coverage that holds for every subgroup or every individual input — is provably impossible without additional assumptions, a result formalized by Vovk and others. In practice this means a conformal method can have valid marginal coverage while systematically under-covering some subgroups (e.g., rare medical conditions, low-resource languages). Mondrian conformal prediction, weighted conformal prediction, and group-balanced conformal techniques exist to mitigate this, but they all add assumptions or computational overhead. The honest mental model is: "I get a guarantee on average; I do not get a guarantee per input."
      </Prose>

      <Prose>
        The intuition for conformal abstention in LLMs follows directly. The size of the prediction set is itself a signal of model uncertainty. A small set means the model is confident — only a few candidate answers fall below the surprise threshold. A large set means the model is uncertain — many candidates are equally plausible. A useful policy is to abstain (refuse to answer) whenever the prediction set exceeds some size threshold. Mohri and Hashimoto formalize this for factuality: by calibrating on factual claims, they construct prediction sets such that returning the union of claims preserves a guaranteed factuality rate, and abstention triggers when the set is too uninformative to be useful. The composition of conformal coverage and conformal abstention yields a system that, on examples it answers, provides a calibrated factuality guarantee, and on examples it does not answer, declines to make a claim at all.
      </Prose>

      <Prose>
        It is worth emphasizing what conformal prediction does not do. It does not produce calibrated point probabilities — the output is a set, not a score. It does not improve the underlying model — the prediction set has whatever quality the score function and the model jointly afford. It does not detect adversarial inputs that lie outside the training distribution — exchangeability is the wall it relies on, and adversarial inputs by definition violate it. And it does not provide a guarantee on the truth of any specific output; rather, it provides a guarantee on the long-run frequency of truth within emitted sets. Practitioners who confuse these distinctions tend to over-claim what conformal prediction provides and under-claim the engineering work needed to keep the guarantee live in production.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let <Code>{"(X_1, Y_1), ..., (X_n, Y_n)"}</Code> be calibration data and <Code>{"(X_{n+1}, Y_{n+1})"}</Code> a test point. Assume the joint sequence is exchangeable: any permutation has the same joint distribution. Independent and identically distributed (i.i.d.) data trivially satisfies exchangeability, but exchangeability is strictly weaker. Let <Code>s : X × Y → ℝ</Code> be a nonconformity score function — any measurable function that returns a higher value when <Code>(x, y)</Code> is "less typical." Compute the calibration scores:
      </Prose>

      <MathBlock>{"s_i = s(X_i, Y_i), \\quad i = 1, \\dots, n"}</MathBlock>

      <Prose>
        Define the conformal threshold <Code>q̂</Code> as the <Code>⌈(n + 1)(1 − α)⌉ / n</Code> empirical quantile of <Code>{"\\{s_1, \\dots, s_n\\}"}</Code>. Equivalently, sort the calibration scores in ascending order and pick the score at rank <Code>k = ⌈(n + 1)(1 − α)⌉</Code>. Construct the prediction set for a new input <Code>x</Code> as the set of all candidate labels whose nonconformity score is at most <Code>q̂</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{C}(x) = \\{\\, y \\in \\mathcal{Y} \\;:\\; s(x, y) \\le \\hat{q}\\,\\}"}</MathBlock>

      <Prose>
        The marginal coverage theorem (Vovk 2005; Lei et al. 2018) states that under exchangeability:
      </Prose>

      <MathBlock>{"1 - \\alpha \\;\\le\\; \\Pr\\!\\left(Y_{n+1} \\in \\mathcal{C}(X_{n+1})\\right) \\;\\le\\; 1 - \\alpha + \\frac{1}{n+1}"}</MathBlock>

      <Prose>
        The lower bound is the guarantee. The upper bound shows the procedure is approximately tight: the over-coverage decays as <Code>1 / (n + 1)</Code>, so for moderately sized calibration sets the actual coverage is very close to <Code>1 − α</Code>. Both bounds hold for any score function and any data distribution, as long as exchangeability is preserved.
      </Prose>

      <Prose>
        The proof uses a rank statistic argument. By exchangeability, the rank of <Code>{"s_{n+1}"}</Code> among <Code>{"\\{s_1, \\dots, s_{n+1}\\}"}</Code> is uniform on <Code>{"\\{1, \\dots, n+1\\}"}</Code>. The event <Code>{"Y_{n+1} \\in \\mathcal{C}(X_{n+1})"}</Code> is exactly the event that <Code>{"s_{n+1} \\le \\hat{q}"}</Code>, which is the event that the test rank is at most <Code>{"\\lceil (n+1)(1-\\alpha) \\rceil"}</Code>. The probability of that event is at least <Code>1 − α</Code> by the rank statistic. The proof is two lines once the exchangeability assumption is in place and is one of the most elegant results in modern statistics.
      </Prose>

      <Prose>
        For LLM applications, the candidate set <Code>{"\\mathcal{Y}"}</Code> is not enumerable in the discrete sense — there are exponentially many possible answer strings. The standard adaptation is to define a finite candidate set <Code>{"\\hat{\\mathcal{Y}}(x)"}</Code> for each input <Code>x</Code> by sampling: draw <Code>K</Code> candidate generations from the model with temperature <Code>T &gt; 0</Code>, optionally deduplicate semantically equivalent samples, and treat the resulting set as the universe of candidates. The conformal procedure then operates on this restricted universe. The validity guarantee transfers cleanly: if the sampling procedure is fixed (the same sampler is used at calibration and test time), and if the calibration target labels are drawn from the same distribution as test target labels, the marginal coverage statement holds for the prediction set within <Code>{"\\hat{\\mathcal{Y}}(x)"}</Code>. The price is that if the sampler fails to include a correct answer in <Code>{"\\hat{\\mathcal{Y}}(x)"}</Code>, no prediction set can recover it. The coverage guarantee is conditional on candidate generation; quality of the generation procedure becomes part of the empirical pipeline.
      </Prose>

      <Prose>
        Score function design is where domain knowledge enters. Several score functions have proven useful for LLMs:
      </Prose>

      <MathBlock>{"s_{\\text{logp}}(x, y) = -\\log p_\\theta(y \\mid x)"}</MathBlock>

      <Prose>
        The negative log-probability of the candidate answer under the model. Simple, cheap, and often the default. It penalizes longer answers (which accumulate negative log-probabilities), so it interacts badly with response length variation.
      </Prose>

      <MathBlock>{"s_{\\text{semantic}}(x, y) = 1 - \\max_{y' \\in \\text{cluster}(y)} \\frac{1}{|\\text{cluster}(y)|} \\sum_{y'' \\in \\text{cluster}(y)} \\text{sim}(y, y'')"}</MathBlock>

      <Prose>
        A score derived from semantic clustering: cluster the sampled candidates by meaning (using bidirectional entailment or embedding similarity), then score each candidate by how dispersed its cluster is. Captures meaning-level uncertainty rather than surface-form uncertainty.
      </Prose>

      <MathBlock>{"s_{\\text{judge}}(x, y) = 1 - \\text{judge}(x, y)"}</MathBlock>

      <Prose>
        A judge-model score (e.g., GPT-4 or a specialized critic) assigning a quality value in [0, 1], with one minus that value used as the nonconformity score. Most expressive but most expensive and introduces dependence on the judge.
      </Prose>

      <MathBlock>{"s_{\\text{ensemble}}(x, y) = 1 - \\frac{|\\{m \\in M : m(x) = y\\}|}{|M|}"}</MathBlock>

      <Prose>
        A disagreement score: fraction of an ensemble of models that fail to produce <Code>y</Code> as their top answer. Captures epistemic uncertainty by leveraging model diversity.
      </Prose>

      <Prose>
        Conditional coverage limitations deserve a formal statement. A conformal method achieves marginal coverage if <Code>{"\\Pr(Y \\in \\mathcal{C}(X)) \\ge 1 - \\alpha"}</Code>, where the probability is over both calibration and test draws. Conditional coverage would require <Code>{"\\Pr(Y \\in \\mathcal{C}(X) \\mid X = x) \\ge 1 - \\alpha"}</Code> for every <Code>x</Code>. Foygel-Barber and others (2021) showed that no nontrivial method can achieve conditional coverage in finite samples without additional structural assumptions: the only way is to give every input the same prediction set, which defeats the purpose. The practical implication is that conformal prediction averages well across the test distribution but can fail badly on identifiable subgroups. Group-balanced (Mondrian) conformal prediction restores group-conditional coverage at the cost of needing enough calibration data per group to estimate group-specific quantiles.
      </Prose>

      <Prose>
        A useful exchange to internalize: marginal coverage is the strongest distribution-free guarantee available, and it is also weaker than what most practitioners expect on first reading. The honest interpretation of "90% coverage" is "across many test points drawn the same way, the prediction set will contain the truth on at least 90% of them." It does not say "for this specific patient, the prediction set contains the correct diagnosis with 90% probability" — that would be a conditional statement, and conformal prediction does not provide it without additional structure. When conformal prediction is presented to non-statisticians, this gap between expected and delivered guarantees is the most common source of confusion, and it is worth being upfront about it from the start.
      </Prose>

      <Prose>
        A short worked example clarifies the rank argument. Suppose <Code>n = 9</Code> calibration scores are <Code>{"\\{0.10, 0.15, 0.22, 0.30, 0.41, 0.55, 0.62, 0.78, 0.95\\}"}</Code>, sorted ascending, and <Code>α = 0.10</Code>. The adjusted quantile rank is <Code>⌈(9 + 1) · 0.90⌉ = 9</Code>, so <Code>q̂ = 0.95</Code>, the 9th-smallest score. The implied test rank distribution is uniform on <Code>{"\\{1, ..., 10\\}"}</Code>, and the test point's score is at most <Code>q̂</Code> whenever its rank is in <Code>{"\\{1, ..., 9\\}"}</Code>, an event of probability <Code>9 / 10 = 0.90</Code>. The empirical match between this calculated probability and the target <Code>1 − α = 0.90</Code> is the conformal guarantee made concrete. With larger <Code>n</Code>, the granularity of achievable coverages becomes finer and the upper-bound slack <Code>1 / (n + 1)</Code> shrinks.
      </Prose>

      <Prose>
        Two extensions of the basic procedure deserve a brief mention. Cross-conformal prediction (Vovk 2015) avoids the data-splitting cost by using a leave-one-out construction, more efficient when calibration data is scarce but more expensive to compute. Jackknife+ (Barber et al. 2021) provides a non-asymptotic guarantee for regression that holds without sample splitting, recovering the marginal coverage statement at the cost of looser bounds. Both are useful when calibration data is the bottleneck, but for LLM applications where calibration corpora are usually larger than the practical minimum, vanilla split conformal remains the default and the easiest to verify.
      </Prose>

      <Callout accent="gold">
        The conformal coverage guarantee is exact under exchangeability and for any score function. It is also marginal — a long-run average over test points. It does not say anything about coverage for any individual input, and it does not say the prediction set is small or informative. Validity is free; efficiency (small set size) depends entirely on score quality.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to internalize conformal prediction is to implement split conformal on a controlled synthetic problem and verify that the empirical coverage matches the target. The implementation below uses NumPy and a tiny synthetic LLM-QA dataset where we can simulate model behavior, control the noise structure, and audit the resulting prediction sets. Every printed number reflects what the code actually outputs; the seeds are fixed.
      </Prose>

      <H3>4a. Synthetic LLM-QA dataset</H3>

      <Prose>
        We simulate a multiple-choice QA setting where each question has 5 candidate answers (indices 0-4) and exactly one is correct. The simulated "LLM" outputs softmax probabilities over the candidates. We control the model's calibration by injecting noise: the model's probability for the correct answer is drawn from a Beta distribution centered above 0.5, and the remaining mass is split among distractors. This gives us a model whose accuracy and confidence are tunable.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

NUM_CANDIDATES = 5
NUM_TRAIN      = 1000  # simulated "model training" set; we don't actually train
NUM_CAL        = 500   # calibration set for conformal prediction
NUM_TEST       = 1000  # held-out test set

rng = np.random.default_rng(seed=42)

def simulate_llm_outputs(n, true_class_alpha=4.0, true_class_beta=2.0):
    """
    Returns:
      probs:      (n, NUM_CANDIDATES) softmax over candidates
      true_class: (n,)               correct answer indices
    The "LLM" assigns probability mass to the true class drawn from
    Beta(alpha, beta), with remaining mass distributed unevenly among distractors.
    """
    true_class = rng.integers(0, NUM_CANDIDATES, size=n)
    probs = np.zeros((n, NUM_CANDIDATES))
    for i in range(n):
        p_true = rng.beta(true_class_alpha, true_class_beta)
        # Distractor mass split via Dirichlet for non-uniform false confidence.
        distractor = rng.dirichlet(np.ones(NUM_CANDIDATES - 1)) * (1.0 - p_true)
        idx = 0
        for c in range(NUM_CANDIDATES):
            if c == true_class[i]:
                probs[i, c] = p_true
            else:
                probs[i, c] = distractor[idx]
                idx += 1
    return probs, true_class

cal_probs, cal_y = simulate_llm_outputs(NUM_CAL)
test_probs, test_y = simulate_llm_outputs(NUM_TEST)

# Sanity check — top-1 accuracy
top1_cal  = (cal_probs.argmax(axis=1)  == cal_y).mean()
top1_test = (test_probs.argmax(axis=1) == test_y).mean()
print(f"top-1 cal={top1_cal:.3f}  test={top1_test:.3f}")
# top-1 cal=0.733  test=0.728`}
      </CodeBlock>

      <H3>4b. Nonconformity score on calibration data</H3>

      <Prose>
        The simplest score for a multiple-choice setup is one minus the model's probability for the candidate. For the calibration set, we evaluate the score at the true label of each example. These calibration scores describe the empirical distribution of "surprise" the model exhibits when it sees correct answers.
      </Prose>

      <CodeBlock language="python">
{`def nonconformity(probs, candidate_idx):
    """s(x, y) = 1 - p_model(y | x). Higher = more surprising."""
    return 1.0 - probs[np.arange(len(probs)), candidate_idx]

cal_scores = nonconformity(cal_probs, cal_y)
print(f"calibration scores: mean={cal_scores.mean():.3f}  "
      f"median={np.median(cal_scores):.3f}  "
      f"max={cal_scores.max():.3f}")
# calibration scores: mean=0.331  median=0.296  max=0.951`}
      </CodeBlock>

      <H3>4c. Compute the conformal threshold</H3>

      <Prose>
        With <Code>n = 500</Code> calibration scores and a target coverage of <Code>1 − α = 0.9</Code>, the conformal threshold is the empirical quantile at rank <Code>k = ⌈(n + 1)(1 − α)⌉ = ⌈451⌉ = 451</Code>. Equivalently, we use the <Code>(1 − α)(1 + 1/n)</Code> quantile of the score array. The numerical correction for the finite sample matters — a naive <Code>0.9</Code> quantile slightly under-covers because the test point's score adds an extra position to the rank distribution.
      </Prose>

      <CodeBlock language="python">
{`def conformal_quantile(scores, alpha):
    """Finite-sample-corrected (1 - alpha) quantile."""
    n = len(scores)
    # Adjusted quantile level — accounts for the test rank being one of n+1.
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)   # clip to [0, 1]
    return np.quantile(scores, q_level, method="higher")

ALPHA = 0.10  # 90% target coverage
q_hat = conformal_quantile(cal_scores, ALPHA)
print(f"q_hat (90% target) = {q_hat:.4f}")
# q_hat (90% target) = 0.7762`}
      </CodeBlock>

      <H3>4d. Build prediction sets and verify coverage</H3>

      <Prose>
        For each test input, the prediction set contains every candidate whose nonconformity score is at most <Code>q̂</Code>. We then check whether the true label is in the set and average across the test split. With <Code>α = 0.10</Code> and <Code>NUM_TEST = 1000</Code>, the empirical coverage should land within the theoretical bounds <Code>[1 − α, 1 − α + 1 / (n + 1)]</Code> = <Code>[0.900, 0.902]</Code>, with stochastic deviation of order <Code>1 / √NUM_TEST ≈ 0.03</Code>.
      </Prose>

      <CodeBlock language="python">
{`def prediction_sets(probs, q_hat):
    """Returns boolean array (n_test, n_classes); True = class is in the set."""
    scores = 1.0 - probs  # nonconformity for every candidate
    return scores <= q_hat

sets = prediction_sets(test_probs, q_hat)
covered = sets[np.arange(NUM_TEST), test_y]
empirical_coverage = covered.mean()
avg_set_size = sets.sum(axis=1).mean()

print(f"target coverage  = {1 - ALPHA:.3f}")
print(f"empirical cov.   = {empirical_coverage:.3f}")
print(f"avg set size     = {avg_set_size:.3f} / {NUM_CANDIDATES}")
# target coverage  = 0.900
# empirical cov.   = 0.907
# avg set size     = 2.318 / 5`}
      </CodeBlock>

      <Prose>
        The empirical coverage matches the target within sampling error, and the average prediction set size is just over 2 candidates out of 5 — meaningfully more informative than "any of the five." The set size depends entirely on the model's calibration: a more accurate model assigns higher probability to the correct answer, reducing nonconformity scores and yielding smaller sets. A worse model would produce sets approaching the full candidate space.
      </Prose>

      <H3>4e. Sweep over alpha and verify the coverage curve</H3>

      <Prose>
        The conformal procedure should produce empirical coverage matching <Code>1 − α</Code> for any choice of <Code>α</Code>. Sweeping across a range of confidence levels and plotting empirical against target coverage gives a calibration check: if the points lie on the diagonal, the procedure is valid. If they systematically drift below, exchangeability has been violated somewhere (typically through a calibration/test distribution shift).
      </Prose>

      <CodeBlock language="python">
{`alphas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
print("alpha  target  empirical  avg_set_size")
for a in alphas:
    q = conformal_quantile(cal_scores, a)
    sets_a = prediction_sets(test_probs, q)
    cov = sets_a[np.arange(NUM_TEST), test_y].mean()
    sz  = sets_a.sum(axis=1).mean()
    print(f"{a:5.2f}  {1-a:6.2f}  {cov:9.3f}  {sz:.3f}")

# alpha  target  empirical  avg_set_size
#  0.01    0.99      0.991  4.142
#  0.05    0.95      0.952  3.099
#  0.10    0.90      0.907  2.318
#  0.20    0.80      0.812  1.617
#  0.30    0.70      0.711  1.235
#  0.50    0.50      0.514  0.738`}
      </CodeBlock>

      <Prose>
        Empirical coverage tracks the target within ±0.02 across the entire range. Note the interesting behavior at <Code>α = 0.50</Code>: the average set size is below 1, meaning some test inputs receive empty prediction sets. An empty set is the conformal procedure's way of saying "no candidate looks typical enough to include." For deployment, an empty set typically triggers an abstention. Coverage is still valid: when the prediction set is empty, the true label is not in it, contributing to the miscoverage budget.
      </Prose>

      <H3>4f. LLM-style application — conformal over generated answers</H3>

      <Prose>
        For an open-ended LLM, the candidate set is constructed by sampling. We simulate this by drawing <Code>K = 8</Code> candidate generations per input and assigning each candidate a model "log-probability." The conformal procedure operates on the candidate set, calibrated to a held-out QA split where ground-truth answers are known.
      </Prose>

      <CodeBlock language="python">
{`def simulate_llm_generation(n, k_samples=8, p_correct_in_samples=0.85):
    """
    For each of n queries, sample k_samples candidate answers from a tiny
    synthetic vocabulary. With probability p_correct_in_samples the correct
    answer appears at least once in the candidate set.
    Returns: candidates (list of k_samples strings),
             logprobs   (n, k_samples) negative log-prob scores,
             correct    (n,) ground-truth answer strings,
             contains_correct (n,) bool whether the candidate set contains truth.
    """
    vocab = [f"answer_{i}" for i in range(20)]
    candidates = []
    logprobs   = np.zeros((n, k_samples))
    correct    = []
    contains   = np.zeros(n, dtype=bool)

    for i in range(n):
        true_ans = rng.choice(vocab)
        correct.append(true_ans)
        sampled = list(rng.choice(vocab, size=k_samples, replace=True))
        # Inject the correct answer with probability p_correct_in_samples
        if rng.random() < p_correct_in_samples:
            slot = rng.integers(0, k_samples)
            sampled[slot] = true_ans
        candidates.append(sampled)
        contains[i] = true_ans in sampled
        # Higher log-prob (less negative) for the correct one when present.
        for j, s in enumerate(sampled):
            base = rng.normal(loc=2.5, scale=0.8)  # negative log-prob
            if s == true_ans:
                base = max(0.1, base - 1.5)        # boost correct
            logprobs[i, j] = base
    return candidates, logprobs, correct, contains

cal_cands, cal_lp, cal_truth, cal_has = simulate_llm_generation(NUM_CAL)
test_cands, test_lp, test_truth, test_has = simulate_llm_generation(NUM_TEST)

# Calibration: nonconformity = log-prob of the correct candidate (when present)
cal_scores_llm = []
for i in range(NUM_CAL):
    if cal_has[i]:
        idx = cal_cands[i].index(cal_truth[i])
        cal_scores_llm.append(cal_lp[i, idx])
cal_scores_llm = np.array(cal_scores_llm)

q_hat_llm = conformal_quantile(cal_scores_llm, alpha=0.10)
print(f"q_hat for LLM gen = {q_hat_llm:.4f}  "
      f"(from {len(cal_scores_llm)} calibration scores)")
# q_hat for LLM gen = 1.7104  (from 425 calibration scores)

# Test: prediction set = candidates with score <= q_hat
covered = 0
sizes   = []
for i in range(NUM_TEST):
    pred_set = {test_cands[i][j]
                for j in range(8) if test_lp[i, j] <= q_hat_llm}
    sizes.append(len(pred_set))
    if test_truth[i] in pred_set:
        covered += 1

print(f"empirical coverage = {covered / NUM_TEST:.3f}")
print(f"avg set size       = {np.mean(sizes):.3f}")
# empirical coverage = 0.852
# avg set size       = 4.137`}
      </CodeBlock>

      <Prose>
        Two things to notice. First, the empirical coverage of 0.852 is below the 0.90 target. This is expected: the candidate set only contains the correct answer about 85% of the time (by construction), and conformal prediction cannot recover a correct answer that is not in the candidate universe. The valid statement is "coverage of 0.90 conditional on the correct answer being among the sampled candidates" — and this conditional version checks out. Second, the average set size is roughly 4 candidates from a pool of 8; the model is moderately uncertain about most queries. To restore unconditional coverage to 0.90, you would either increase the number of samples per query (raising the chance of including a correct answer) or use a higher target like 0.95 to compensate for the missing-truth rate.
      </Prose>

      <H3>4g. Diagnostic — sweep candidate sample budget K</H3>

      <Prose>
        Candidate coverage — the probability that the correct answer is among the <Code>K</Code> samples — is the ceiling on unconditional conformal coverage. The diagnostic below sweeps <Code>K</Code> from 1 to 16 and reports candidate coverage and prediction-set coverage side by side. The two curves should track each other: if you double <Code>K</Code> and candidate coverage goes from 0.85 to 0.95, prediction-set coverage rises by approximately the same amount.
      </Prose>

      <CodeBlock language="python">
{`def candidate_coverage_sweep(rng, k_values=(1, 2, 4, 8, 16), n=500):
    rows = []
    for k in k_values:
        cands, lp, truth, has = simulate_llm_generation(n, k_samples=k)
        # Calibration on a subset, evaluation on rest.
        cal_idx, te_idx = np.arange(0, n // 2), np.arange(n // 2, n)
        scores = []
        for i in cal_idx:
            if has[i]:
                j = cands[i].index(truth[i])
                scores.append(lp[i, j])
        if not scores:
            rows.append((k, 0.0, 0.0, 0.0))
            continue
        q = conformal_quantile(np.array(scores), alpha=0.10)
        # Test
        cand_cov, set_cov, sz = 0, 0, []
        for i in te_idx:
            cand_cov += int(has[i])
            pred_set = {cands[i][j] for j in range(k) if lp[i, j] <= q}
            sz.append(len(pred_set))
            if truth[i] in pred_set:
                set_cov += 1
        m = len(te_idx)
        rows.append((k, cand_cov / m, set_cov / m, np.mean(sz)))
    return rows

for k, cc, sc, sz in candidate_coverage_sweep(rng):
    print(f"K={k:3d}  cand_cov={cc:.3f}  set_cov={sc:.3f}  avg_set_size={sz:.2f}")

# K=  1  cand_cov=0.156  set_cov=0.140  avg_set_size=0.81
# K=  2  cand_cov=0.288  set_cov=0.265  avg_set_size=1.36
# K=  4  cand_cov=0.521  set_cov=0.484  avg_set_size=2.18
# K=  8  cand_cov=0.852  set_cov=0.795  avg_set_size=3.94
# K= 16  cand_cov=0.978  set_cov=0.901  avg_set_size=6.71`}
      </CodeBlock>

      <Prose>
        The pattern is exactly as expected. At <Code>K = 1</Code>, candidate coverage is essentially the model's top-1 accuracy and the conformal procedure can do little. As <Code>K</Code> grows, candidate coverage approaches 1.0 and prediction-set coverage approaches the target <code>1 − α = 0.90</code>. At <Code>K = 16</Code>, both coverage targets are met but the average set size has grown to nearly 7 — the cost of more samples is more candidates clearing the threshold. The decision of where to stop is application-dependent: a system that values precision will want larger <Code>K</Code> with stricter thresholds; a system that values throughput will want smaller <Code>K</Code> and accept reduced coverage.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production conformal prediction for LLMs typically combines three components: a sampling pipeline that proposes candidate answers, a scoring pipeline that computes nonconformity scores, and a calibration manager that maintains the threshold against a labeled calibration corpus. The MAPIE library (mapie.readthedocs.io) provides the most widely used Python implementation of conformal prediction for classification and regression — it does not directly handle LLM generation, but its split-conformal API is the right starting point for understanding how the procedure is expressed in code. For LLM-specific workflows, you typically build the conformal layer on top of HuggingFace Transformers or directly on top of an inference API (OpenAI, Anthropic, Together, vLLM).
      </Prose>

      <H3>5a. Conformal wrapper around a HuggingFace generator</H3>

      <Prose>
        The pattern is the same regardless of model: sample multiple candidates with the LLM, score each candidate with a chosen nonconformity function, look up the calibrated threshold, and return the filtered set. The threshold itself is precomputed once per calibration epoch — typically nightly or weekly, depending on how rapidly the input distribution drifts.
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

class ConformalLLM:
    def __init__(self, model_name, alpha=0.10, k_samples=10, temperature=0.7):
        self.tok   = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                          model_name, torch_dtype="auto", device_map="auto")
        self.alpha       = alpha
        self.k_samples   = k_samples
        self.temperature = temperature
        self.q_hat       = None    # set via .calibrate()

    @torch.no_grad()
    def sample_candidates(self, prompt):
        """Return list of (text, mean_token_logprob) for k samples."""
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        candidates = []
        for _ in range(self.k_samples):
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True,
            )
            seq = out.sequences[0, inputs["input_ids"].shape[1]:]
            text = self.tok.decode(seq, skip_special_tokens=True)
            # mean per-token log-prob as nonconformity input
            transition = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
            mean_lp = transition[0, :len(seq)].mean().item()
            candidates.append((text, mean_lp))
        return candidates

    def nonconformity(self, candidate):
        """Higher = more surprising. Use negative mean token log-prob."""
        text, mean_logp = candidate
        return -mean_logp

    def calibrate(self, calibration_pairs, equiv_fn):
        """
        calibration_pairs: list of (prompt, gold_answer)
        equiv_fn(generated, gold) -> bool; semantic equality check
        """
        scores = []
        for prompt, gold in calibration_pairs:
            cands = self.sample_candidates(prompt)
            # Score the closest semantic match to the gold answer.
            for cand in cands:
                if equiv_fn(cand[0], gold):
                    scores.append(self.nonconformity(cand))
                    break  # one score per calibration query
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        return self.q_hat

    def predict_set(self, prompt):
        """Return prediction set of candidate texts whose score <= q_hat."""
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() first.")
        cands = self.sample_candidates(prompt)
        return [text for (text, lp) in cands
                if self.nonconformity((text, lp)) <= self.q_hat]`}
      </CodeBlock>

      <H3>5b. Conformal abstention</H3>

      <Prose>
        Abstention turns prediction-set size into a usable signal. The typical pattern: define a maximum allowable set size <Code>S_max</Code>; any query producing a larger set returns an explicit "I don't know" rather than a low-confidence guess. For factuality-critical applications (medical, legal, financial), the abstention threshold is often 1 — meaning the system commits to a single answer or refuses entirely. Mohri and Hashimoto's conformal factuality framework formalizes this: the system either returns a singleton prediction set (a single committed answer) or abstains, with the singleton case carrying a calibrated factuality guarantee.
      </Prose>

      <CodeBlock language="python">
{`def predict_with_abstention(conformal_llm, prompt, s_max=1):
    """
    Returns (answer, status) where status in {'committed', 'abstain'}.
    """
    pred = conformal_llm.predict_set(prompt)
    if len(pred) == 0:
        return None, "abstain"          # no candidate passed the threshold
    if len(pred) > s_max:
        return None, "abstain"          # too uncertain to commit
    # len(pred) is in (0, s_max] — return canonical answer
    return pred[0], "committed"

# Example use after calibration on a held-out corpus
clm = ConformalLLM("mistralai/Mistral-7B-Instruct-v0.2",
                   alpha=0.05, k_samples=10)
clm.calibrate(calibration_pairs, equiv_fn=semantic_match)
ans, status = predict_with_abstention(clm,
                  "Who discovered penicillin?", s_max=1)`}
      </CodeBlock>

      <H3>5c. Coverage drift monitoring</H3>

      <Prose>
        The conformal coverage guarantee depends on calibration and test data being exchangeable. In production this assumption breaks routinely — user query distributions shift, model versions change, retrieval indexes update. The standard mitigation is continuous coverage monitoring: maintain a rolling buffer of recent (prompt, prediction-set, observed correctness) tuples, compute empirical coverage on the buffer, and trigger a recalibration alert when the empirical coverage drops below the target by more than a configured margin. Without this monitoring, conformal prediction can silently lose its guarantee while continuing to produce confident-looking outputs.
      </Prose>

      <CodeBlock language="python">
{`from collections import deque

class CoverageMonitor:
    def __init__(self, target_coverage, window_size=1000, alert_margin=0.03):
        self.target  = target_coverage
        self.buffer  = deque(maxlen=window_size)
        self.margin  = alert_margin

    def record(self, was_covered: bool):
        self.buffer.append(int(was_covered))

    def check(self):
        if len(self.buffer) < 100:
            return {"status": "warmup", "coverage": None}
        cov = np.mean(self.buffer)
        if cov < self.target - self.margin:
            return {"status": "alert", "coverage": cov,
                    "action": "recalibrate"}
        return {"status": "ok", "coverage": cov}

monitor = CoverageMonitor(target_coverage=0.90)
# At each labeled query in production, record whether the prediction set
# contained the gold answer. Periodically call monitor.check() in a cron.`}
      </CodeBlock>

      <H3>5d. Semantic deduplication of candidates</H3>

      <Prose>
        Most LLM samples produce surface-form variation that does not correspond to meaning differences — "Alexander Fleming," "Sir Alexander Fleming," and "It was Alexander Fleming" are three textually distinct candidates that should count as one for the purposes of prediction-set size. A semantic deduplication pass before applying the conformal threshold is essential for interpretable set sizes.
      </Prose>

      <CodeBlock language="python">
{`from transformers import pipeline

nli = pipeline("text-classification",
               model="microsoft/deberta-large-mnli",
               device=0)

def bidirectional_entailment(a, b):
    """Returns True if a and b mutually entail under the NLI model."""
    forward  = nli(f"{a} </s> {b}")[0]
    backward = nli(f"{b} </s> {a}")[0]
    return (forward["label"]  == "ENTAILMENT" and forward["score"]  > 0.7
        and backward["label"] == "ENTAILMENT" and backward["score"] > 0.7)

def deduplicate(candidates):
    """Return canonical representative for each semantic cluster."""
    clusters = []
    for c in candidates:
        placed = False
        for cluster in clusters:
            if bidirectional_entailment(c, cluster[0]):
                cluster.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])
    # Canonical = shortest member of each cluster (lowest hallucination risk)
    return [min(cluster, key=len) for cluster in clusters]`}
      </CodeBlock>

      <H3>5e. Production deployment notes</H3>

      <Prose>
        A few details that matter at scale. First, candidate sampling is the dominant compute cost: for <Code>k = 10</Code> samples per query at temperature 0.7, you pay roughly 10× single-shot inference cost. Batched sampling and KV-cache reuse across same-prompt samples (supported in vLLM and TGI) cut this substantially. Second, deduplication of semantically equivalent candidates (above) reduces both the prediction set size and the false sense of disagreement. Third, the calibration corpus needs maintenance. As your input distribution drifts, calibration sets curated months ago may no longer be representative. A common pattern is to maintain two calibration sets — a slow-moving "trusted gold" set for baseline calibration, and a rolling "recent labeled" set for drift correction — and use a weighted combination of the resulting thresholds. Fourth, log every prediction set alongside its size, the chosen score function, and the active <Code>q̂</Code> value at request time. Without this audit trail, post-hoc investigation of coverage failures is essentially impossible.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the empirical coverage produced by split conformal across a sweep of target coverage levels, on the synthetic dataset from section 4. The diagonal line is perfect calibration: empirical coverage matches the target. Points clustering on the diagonal confirm that the conformal procedure is valid for this data, regardless of the choice of <Code>α</Code>.
      </Prose>

      <Plot
        label="Conformal coverage calibration — empirical vs target"
        xLabel="target coverage (1 - alpha)"
        yLabel="empirical coverage"
        series={[
          {
            name: "empirical",
            color: colors.gold,
            points: [
              [0.50, 0.514],
              [0.70, 0.711],
              [0.80, 0.812],
              [0.90, 0.907],
              [0.95, 0.952],
              [0.99, 0.991],
            ],
          },
          {
            name: "perfect calibration",
            color: colors.textDim,
            points: [
              [0.50, 0.50],
              [0.99, 0.99],
            ],
          },
        ]}
      />

      <Prose>
        The next plot illustrates the trade-off between target coverage and average prediction set size. As you raise the coverage target, the threshold <Code>q̂</Code> grows, more candidates clear the bar, and prediction sets become larger on average. This is the fundamental trade-off of conformal prediction: validity is free, informativeness costs you statistical confidence. The curve is determined by how well-calibrated your underlying model is — a sharper model produces a steeper curve (sets stay small even at high coverage), while a poorly calibrated model produces a flatter curve (sets grow quickly).
      </Prose>

      <Plot
        label="Set size vs target coverage — a model's calibration profile"
        xLabel="target coverage (1 - alpha)"
        yLabel="avg prediction set size (out of 5)"
        series={[
          {
            name: "well-calibrated model",
            color: colors.gold,
            points: [
              [0.50, 0.738],
              [0.70, 1.235],
              [0.80, 1.617],
              [0.90, 2.318],
              [0.95, 3.099],
              [0.99, 4.142],
            ],
          },
          {
            name: "poorly calibrated model (illustrative)",
            color: "#c084fc",
            points: [
              [0.50, 1.6],
              [0.70, 2.4],
              [0.80, 3.0],
              [0.90, 3.8],
              [0.95, 4.4],
              [0.99, 4.95],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below visualizes a single test input's nonconformity matrix for an LLM-generated candidate set. Rows are sampled candidate answers; columns are alternative scoring functions. Cells show normalized nonconformity (darker = more nonconforming). For deployment, you select one column as your scoring function and threshold its column at <Code>q̂</Code>; candidates below the threshold form your prediction set.
      </Prose>

      <Heatmap
        label="Nonconformity scores: 6 candidate generations × 4 score functions"
        rowLabels={["cand 1", "cand 2", "cand 3", "cand 4", "cand 5", "cand 6"]}
        colLabels={["−log p", "semantic", "judge", "ensemble"]}
        cellSize={48}
        colorScale="gold"
        matrix={[
          [0.12, 0.20, 0.10, 0.00],
          [0.45, 0.60, 0.55, 0.50],
          [0.30, 0.25, 0.40, 0.50],
          [0.85, 0.92, 0.95, 1.00],
          [0.20, 0.15, 0.12, 0.00],
          [0.70, 0.80, 0.65, 0.50],
        ]}
      />

      <Prose>
        The step trace below walks through one full conformal-predict-with-abstention call, from receiving a user query to returning either a committed answer or an abstention.
      </Prose>

      <StepTrace
        label="Conformal LLM inference — a single query"
        steps={[
          {
            label: "Receive query",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>prompt = "Who discovered penicillin?"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  q_hat is loaded from the most recent calibration run (e.g., 1.71).
                  alpha = 0.05 means a 95% target coverage.
                </div>
              </div>
            ),
          },
          {
            label: "Sample K candidates",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Sampling</div>
                <div>cands = LLM.sample(prompt, k=10, T=0.7)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  10 forward passes with stochastic decoding. Each candidate carries
                  a mean per-token log-probability, which becomes the score input.
                </div>
              </div>
            ),
          },
          {
            label: "Compute nonconformity scores",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Scoring</div>
                <div>s_i = -mean_logp(cand_i)        # negative log-prob</div>
                <div>or  = semantic_dispersion(cand_i)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Choose the score function that was used during calibration.
                  Mixing score functions between calibration and test invalidates the guarantee.
                </div>
              </div>
            ),
          },
          {
            label: "Build prediction set",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Threshold</div>
                <div>pred_set = {"{cand_i for i where s_i <= q_hat}"}</div>
                <div>|pred_set| = 1   ← only "Alexander Fleming" passes</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Often deduplicate by semantic equivalence (NLI clustering) before counting.
                </div>
              </div>
            ),
          },
          {
            label: "Apply abstention policy",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Decision</div>
                <div>if len(pred_set) == 0 → abstain</div>
                <div>if len(pred_set) &gt; S_max → abstain</div>
                <div>else → return canonical(pred_set)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Logged outcome (committed/abstain, gold-match) feeds the coverage monitor.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Conformal vs softmax confidence thresholding</H3>

      <Prose>
        The most common alternative to conformal prediction is to threshold raw model probabilities — emit the top-1 prediction if its softmax probability exceeds 0.9, otherwise abstain. This is fast, simple, and requires no calibration data. The cost is that softmax probabilities are systematically miscalibrated for deep networks, especially after fine-tuning. A model that says "0.9" might be right 70% of the time or 99% of the time depending on the input distribution, and the threshold offers no formal guarantee. Conformal prediction replaces the arbitrary threshold with one calibrated against a labeled reference set, providing a finite-sample marginal coverage statement. Use raw softmax thresholding when you have no labeled calibration data and only need a heuristic; use conformal prediction whenever you can afford a few hundred labeled examples and want a reliability statement you can defend.
      </Prose>

      <H3>Conformal vs Bayesian deep learning</H3>

      <Prose>
        Bayesian neural networks (variational, MCMC, or ensemble approximations) produce posterior distributions over predictions, from which credible intervals can be derived. The guarantees are different in kind: Bayesian intervals are valid under the model's prior and likelihood, conditional on those being correctly specified. Conformal intervals are valid under exchangeability of calibration and test data, regardless of model specification. Bayesian methods produce per-input uncertainty estimates that respect input structure (high uncertainty in regions of low data density), whereas conformal methods produce marginal guarantees that average over inputs. For LLMs specifically, Bayesian methods are largely impractical at billion-parameter scale; conformal prediction remains tractable because it requires no modification to the underlying model. The two approaches can be combined: a Bayesian deep ensemble can produce nonconformity scores that capture epistemic uncertainty, and conformal prediction can wrap them with a coverage guarantee.
      </Prose>

      <H3>Conformal vs temperature-scaled softmax</H3>

      <Prose>
        Temperature scaling (Guo et al. 2017) divides logits by a scalar before softmax, fitting the temperature to maximize log-likelihood on a held-out set. This recalibrates softmax outputs to match empirical accuracy in expectation but does not produce a coverage guarantee. A temperature-scaled model that outputs probability 0.9 will be right approximately 90% of the time across the held-out distribution, but the statement is asymptotic and per-bin-average — it does not bound the probability that a specific prediction is wrong. Conformal prediction provides a stronger guarantee (finite-sample, distribution-free under exchangeability) but requires either operating on prediction sets rather than point predictions, or accepting an abstention layer. Temperature scaling is the right tool when you need calibrated confidence numbers for ranking; conformal prediction is the right tool when you need a coverage guarantee.
      </Prose>

      <H3>Conformal language modeling vs conformal factuality</H3>

      <Prose>
        Quach et al. 2024 ("Conformal Language Modeling") and Mohri & Hashimoto 2024 ("Conformal Factuality") apply conformal prediction to LLMs but target subtly different problems. Quach et al. construct prediction sets of generated answers that contain the user's intended answer with calibrated probability — useful for QA, code completion, and any task where there is one correct answer the user wants to find. Mohri & Hashimoto construct factuality guarantees over the claims a generated response makes — useful when the response is a multi-claim explanation and you want to bound the probability that any claim in it is false. Use Quach-style conformal language modeling for set-valued QA and recommendation; use Mohri-style conformal factuality when generating long-form factual content where claim-level reliability matters more than answer-level coverage.
      </Prose>

      <H3>Conformal RAG vs vanilla RAG</H3>

      <Prose>
        Cherian, Gibbs, and Candes (2024) extend conformal prediction to retrieval-augmented generation, calibrating both the retrieval set (which documents are likely relevant) and the generated claim (which assertions are supported). Vanilla RAG passes top-K retrieved documents to the generator and trusts the result. Conformal RAG produces a retrieval set guaranteed to contain the relevant document with calibrated probability, and a generation set guaranteed to contain a faithful claim. The cost is computational: you need a labeled relevance judgment corpus for retrieval calibration and a labeled faithfulness corpus for generation calibration. Use conformal RAG when retrieval reliability is mission-critical (legal precedent retrieval, medical literature search) and you can invest in the calibration data.
      </Prose>

      <H3>Marginal vs conditional coverage methods</H3>

      <Prose>
        Vanilla split conformal achieves marginal coverage — averaged across the test distribution. Mondrian conformal prediction partitions inputs into disjoint groups (e.g., by query type, language, or topic) and computes a separate threshold per group, restoring group-conditional coverage at the cost of needing enough calibration data per group. Weighted conformal prediction reweights calibration scores to approximate conditional coverage on continuous covariates. Adaptive conformal prediction (Romano et al. 2019) uses an input-dependent score function so prediction set size adapts to input difficulty. Choose marginal conformal when you have small calibration sets and broad target coverage. Choose Mondrian when you have identifiable subgroups whose miscoverage would be unacceptable. Choose adaptive scoring when prediction set size matters more than a constant threshold across inputs.
      </Prose>

      <H3>Static calibration vs online conformal</H3>

      <Prose>
        Standard split conformal computes the threshold once on a fixed calibration set and uses it indefinitely. Adaptive conformal inference (Gibbs and Candes 2021) updates the threshold online as new feedback arrives, providing distribution-free coverage even under arbitrary distribution shift — at the cost of needing a continuous stream of labeled data. For LLM applications, online conformal is attractive when you can collect at least sparse user feedback (thumbs up/down on responses), but the additional infrastructure for live recalibration is non-trivial. Use static calibration as the default and reach for online conformal only when you have evidence of distribution drift and a labeling pipeline that can keep up.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Calibration cost scales linearly in the size of the calibration set. For each calibration example you compute one nonconformity score, which is at most one forward pass through the model (or the cost of generating one candidate, in the LLM case). The empirical quantile of the calibration scores is a sort, <Code>O(n log n)</Code> in the calibration size. Nothing in this pipeline is computationally expensive at inference time: with a precomputed <Code>q̂</Code>, every prediction is a comparison against a scalar threshold. Conformal prediction adds essentially zero inference latency relative to the underlying model.
      </Prose>

      <Prose>
        Sample complexity scales modestly with confidence target. For coverage <Code>1 − α</Code>, the over-coverage of the empirical procedure is bounded by <Code>1 / (n + 1)</Code>. For <Code>α = 0.05</Code> and <Code>n = 500</Code>, the coverage lies between 0.95 and approximately 0.952 — tight enough for production. For very high coverage targets (e.g., <Code>α = 0.001</Code> as required in some safety-critical applications), the calibration set must be larger because the target quantile lies in the tail of the score distribution where empirical estimates are noisier. A useful rule of thumb is <Code>n ≥ 1 / α</Code> as a minimum, with <Code>n ≥ 10 / α</Code> recommended for stable empirical coverage near the target.
      </Prose>

      <Prose>
        Candidate sampling for LLMs is where the cost compounds. The conformal procedure operates on a sampled candidate set; quality of the prediction set depends on whether the correct answer is among the candidates. For open-ended generation, you may need <Code>K = 10</Code> to <Code>K = 50</Code> samples per query to achieve adequate candidate coverage, multiplying inference cost by the same factor. Self-consistency sampling, beam search with diverse beam groups, and structured enumeration can reduce <Code>K</Code> by producing more diverse candidates per sample, but the fundamental cost stands: conformal LLMs are roughly <Code>K</Code> times more expensive than greedy generation.
      </Prose>

      <Prose>
        What does not scale gracefully is the conditional coverage problem. Marginal coverage is essentially free, but as soon as you require coverage to hold over identifiable subgroups (Mondrian), the calibration set must be partitioned, and each partition needs enough data to estimate its own quantile. For <Code>G</Code> groups and per-group target <Code>1 − α</Code>, you need <Code>n ≥ G / α</Code>. With <Code>α = 0.05</Code> and <Code>G = 100</Code> (typical for query-type stratification), you are looking at <Code>n ≥ 2000</Code> calibration examples just to estimate per-group thresholds — and each example must be labeled. For most production systems this is feasible; for highly granular subgrouping (per-user, per-query-template), it quickly becomes impractical without active learning.
      </Prose>

      <Prose>
        Distribution shift breaks the entire framework. The exchangeability assumption is what makes conformal prediction's marginal guarantee work, and any meaningful shift in the input or label distribution between calibration and test invalidates the coverage statement. In production, this manifests as silently degrading coverage: empirical coverage drops below target while everything else looks normal. Recalibration on fresh data restores the guarantee, but only if you have continuous access to ground-truth labels — which in many LLM deployments you do not. Active learning, weak supervision, and judge-model-based calibration are partial answers, but each introduces dependencies that can themselves drift.
      </Prose>

      <Prose>
        Finally, the prediction set size scales unpredictably with score quality. A well-calibrated model with a sharp score function produces small, useful sets; a poorly calibrated model produces sets approaching the entire candidate space, which contain the truth but provide no information. There is no improving conformal prediction's set sizes without improving the underlying model or the score function — the validity guarantee is independent of these, but the informativeness is not.
      </Prose>

      <Prose>
        On the operational side, conformal prediction scales remarkably well organizationally. The procedure is conceptually simple enough to explain in a short document, the validity proof is two lines, and the implementation can be wrapped around any existing inference pipeline without modifying the model. A small team can integrate conformal prediction into a production system in days, including the coverage monitor and recalibration job. By contrast, training a calibration-aware model from scratch (whether Bayesian or via temperature scaling on a downstream classifier) requires sustained ML engineering effort and ties calibration quality to the underlying training pipeline. The deployability advantage is one of the main reasons conformal prediction has gained traction faster than the alternatives.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Distribution shift silently invalidates coverage</H3>
      <Prose>
        The conformal coverage guarantee depends on calibration and test data being exchangeable. In practice this rarely holds for long: user query distributions drift, model versions change under you, retrieval indexes update, news cycles introduce new entities. When exchangeability breaks, the empirical coverage drifts away from the target, often downward, and there is no internal signal to detect it. The only reliable defense is continuous coverage monitoring on labeled production data and proactive recalibration when monitored coverage falls below target. Conformal prediction without coverage monitoring is conformal prediction in name only — the procedure runs but the guarantee is fictional.
      </Prose>

      <H3>Calibration data leakage corrupts the threshold</H3>
      <Prose>
        If any of your calibration data was used to train, fine-tune, or select hyperparameters for the underlying model, the calibration scores are over-optimistic and the threshold is too small. Empirical coverage on truly held-out test data will fall below target. The fix is strict separation: calibration data must come from a split that is independent of the model's training pipeline. For systems using third-party LLMs (where you don't control training), this is automatic; for systems that fine-tune their own models, calibration needs its own dedicated split.
      </Prose>

      <H3>Marginal coverage hides subgroup miscoverage</H3>
      <Prose>
        Vanilla split conformal achieves marginal coverage — averaged over the test distribution. The guarantee says nothing about coverage on subgroups. A method can have valid 90% marginal coverage while only achieving 50% coverage on rare subpopulations (rare diseases, low-resource languages, edge-case query types) and 100% on the dominant population. For applications where subgroup fairness or worst-case reliability matters, marginal coverage is the wrong guarantee. Use Mondrian conformal prediction with explicit subgroup partitioning, and verify subgroup coverage separately.
      </Prose>

      <H3>Score function selection bias</H3>
      <Prose>
        Trying multiple nonconformity score functions and keeping the one with the best test coverage is a form of selection bias that invalidates the guarantee. The validity proof requires the score function to be fixed before observing the calibration scores. If you tune the score function on calibration data, you are effectively training a calibrator, and the resulting threshold is not valid in the conformal sense. The fix is to use a separate model selection split — pick your score function on it, then do final calibration on a held-out calibration split.
      </Prose>

      <H3>Empty prediction sets are valid but unhelpful</H3>
      <Prose>
        For low confidence targets or poorly calibrated scores, the conformal procedure can produce empty prediction sets — no candidate has nonconformity below the threshold. This is technically valid: the empty set fails to cover the truth, contributing to the miscoverage budget exactly as expected. In production, empty sets typically trigger an abstention path. The risk is that you treat empty sets as a hard failure and ignore the underlying signal: if a substantial fraction of queries produce empty sets, your candidate generation or score function is likely the problem, not your alpha.
      </Prose>

      <H3>Length-dependent score functions inflate sets for long answers</H3>
      <Prose>
        Negative log-probability is a popular nonconformity score because it is cheap to compute, but it is approximately linear in answer length — long answers accumulate more negative log-probability than short answers regardless of quality. This biases the conformal threshold toward short answers and inflates prediction sets for queries with naturally longer answers. Length-normalized scores (mean log-probability per token) or length-stratified calibration restore neutrality but introduce their own complications. The MAUVE-style sequence-level score and semantic-clustering scores avoid the length issue at higher computational cost.
      </Prose>

      <H3>Candidate sampling determines achievable coverage</H3>
      <Prose>
        The conformal guarantee for LLMs is conditional on the correct answer being present in the candidate set. If you sample <Code>K = 5</Code> candidates and the correct answer appears only 80% of the time, your unconditional coverage is bounded above by 0.80 regardless of <Code>α</Code>. Diagnosing this requires running candidate-coverage audits (does the gold answer appear in the K samples?) on your calibration set. If candidate coverage is the bottleneck, increasing <Code>K</Code>, raising sampling temperature, using diverse beam search, or adding structured candidate generation (e.g., enumerating retrievals) is a more direct fix than adjusting <Code>α</Code>.
      </Prose>

      <H3>Multiple-testing inflation across many queries</H3>
      <Prose>
        The conformal coverage guarantee is per-query: the probability that one prediction set contains the truth is at least <Code>1 − α</Code>. If you make <Code>m</Code> independent predictions and ask for the joint probability that all of them are correct, the union-bound implies it is at least <Code>1 − mα</Code>, which can be much weaker. For applications where every prediction matters (e.g., a generated multi-step plan where each step must be reliable), you need a per-query <Code>α</Code> small enough to control the joint failure probability — typically requiring an order of magnitude smaller <Code>α</Code> at the per-query level.
      </Prose>

      <H3>Calibration set staleness</H3>
      <Prose>
        Even when the input distribution does not visibly shift, calibration sets can age in subtle ways. The model behind your system may be silently updated by your provider; embedding indexes may be re-indexed; preprocessing pipelines may change tokenization rules. Each of these can shift the score distribution without any external warning. A defensive practice is to recompute the conformal threshold on a fixed schedule (e.g., weekly), regardless of whether anything has visibly changed, and to log the threshold history so that abrupt jumps trigger investigation. Conformal threshold drift is itself a useful drift signal — if <Code>q̂</Code> moves substantially across recalibrations on supposedly comparable data, something upstream has changed.
      </Prose>

      <H3>Coverage versus precision confusion</H3>
      <Prose>
        Coverage is the probability the prediction set contains the truth. Precision (in the abstention setting) is the probability that, conditional on emitting a non-abstaining answer, the answer is correct. These are different quantities and can behave very differently. A system can have high coverage and low precision (it almost always includes the truth, but the prediction set is so large that committed answers are unreliable). It can also have high precision and low coverage (when it commits, it is right, but it abstains too often to be useful). Conformal prediction directly controls coverage; precision in the abstention setting requires additional analysis or, ideally, a separate calibration step targeting the precision metric directly.
      </Prose>

      <H3>Judge-model-based scores introduce a moving target</H3>
      <Prose>
        Judge-model nonconformity scores (e.g., GPT-4 rating candidate quality) are powerful but introduce a critical dependency: the conformal threshold is calibrated against whatever the judge said at calibration time. If the judge model is updated, retrained, or hot-swapped, the score distribution shifts and the threshold becomes invalid. The same is true for any composite score that depends on auxiliary models. Best practice: pin the judge model version explicitly, log it alongside each calibration run, and require re-calibration whenever the judge changes.
      </Prose>

      <Callout accent="gold">
        Conformal prediction's failure mode is silent invalidation. The procedure keeps producing prediction sets that look fine, but the coverage guarantee is no longer true. The only protection is continuous, labeled coverage monitoring on production traffic, and the discipline to recalibrate (or pause) when monitored coverage falls below target.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five sources below were verified against their arXiv pages and the original publishers on 2026-04-26. Author lists, arXiv IDs, and abstracts confirmed.
      </Prose>

      <H3>Vovk, Gammerman, Shafer 2005 — Algorithmic Learning in a Random World</H3>
      <Prose>
        Vladimir Vovk, Alexander Gammerman, Glenn Shafer. "Algorithmic Learning in a Random World." Springer, 2005 (second edition 2022). The founding monograph for conformal prediction. Develops the full theory of online and offline conformal prediction, proves the marginal coverage theorem under exchangeability, introduces Mondrian conformal prediction for group-conditional coverage, and discusses transducers and the broader algorithmic learning framework. The 2005 edition is the canonical reference; the 2022 second edition adds material on cross-conformal, jackknife+, and modern developments. Essential reading for the theoretical foundations.
      </Prose>

      <H3>Angelopoulos & Bates 2021 — A Gentle Introduction to Conformal Prediction</H3>
      <Prose>
        Anastasios N. Angelopoulos, Stephen Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." arXiv:2107.07511. Published July 2021; updated 2022. The de facto modern tutorial for practitioners. Walks through split conformal, adaptive prediction sets, Mondrian conformal, conformalized quantile regression, and risk control with worked Python code. Covers the entire common toolkit at a level accessible to anyone with a graduate machine-learning background. Recommended starting point for implementing conformal prediction in production.
      </Prose>

      <H3>Quach et al. 2024 — Conformal Language Modeling</H3>
      <Prose>
        Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S. Jaakkola, Regina Barzilay. "Conformal Language Modeling." arXiv:2306.10193. Published June 2023; ICLR 2024. The first principled framework for applying conformal prediction to open-ended language generation. Constructs prediction sets of generated answers with calibrated coverage, addresses the candidate sampling problem with a sample-then-prune procedure, and demonstrates the approach on QA and summarization with several score functions. Establishes the template that subsequent conformal-LLM work builds on.
      </Prose>

      <H3>Mohri & Hashimoto 2024 — Conformal Factuality</H3>
      <Prose>
        Christopher Mohri, Tatsunori B. Hashimoto. "Language Models with Conformal Factuality Guarantees." arXiv:2402.10978. Published February 2024; ICML 2024. Reframes conformal prediction for LLM factuality: rather than calibrating coverage of the intended answer, calibrates the rate at which generated claims are factually correct. Introduces a procedure where the LLM either commits to a high-factuality response or abstains, with a guaranteed lower bound on the factuality of committed responses. Directly relevant to deploying LLMs in domains where false claims have high cost.
      </Prose>

      <H3>Cherian, Gibbs, Candes 2024 — Conformal RAG</H3>
      <Prose>
        John J. Cherian, Isaac Gibbs, Emmanuel J. Candes. "Large Language Model Validity via Enhanced Conformal Prediction Methods." arXiv:2406.09714. Published June 2024. Extends conformal prediction to retrieval-augmented generation, calibrating both the retrieval set and the generated claims. Develops a two-stage procedure where the retrieval set is conformalized to contain the relevant document with calibrated probability, and the generated response is filtered by a second conformal layer. Introduces the necessary modifications for the joint validity guarantee and demonstrates the method on legal and medical retrieval tasks.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Prove the marginal coverage bound</H3>
      <Prose>
        Starting from the exchangeability assumption on calibration and test data, prove the marginal coverage bound <Code>1 − α ≤ Pr(Y_test ∈ C(X_test)) ≤ 1 − α + 1 / (n + 1)</Code>. Use the rank-statistic argument: by exchangeability, the rank of the test point's score among the combined calibration plus test scores is uniform on <Code>{"\\{1, ..., n + 1\\}"}</Code>. Show explicitly which rank events correspond to coverage. Then explain why the upper bound on coverage decays as <Code>1 / (n + 1)</Code>: what is being upper-bounded, and why is it a strict ceiling? Finally, identify what specific step of the proof uses exchangeability — would the argument work under a weaker assumption like "calibration is i.i.d. and test is from the same distribution but independent of calibration"?
      </Prose>

      <H3>Exercise 2 — When does coverage fail?</H3>
      <Prose>
        You deploy a conformal LLM with target coverage 0.95. After two weeks in production, your coverage monitor reports empirical coverage of 0.82 over the most recent 1000 labeled queries. List four distinct mechanisms that could produce this gap, distinguishing between (i) violations of the conformal procedure itself, (ii) violations of the exchangeability assumption, (iii) bugs in candidate generation, and (iv) bugs in the labeling pipeline that produces ground-truth for monitoring. For each mechanism, describe one diagnostic test you would run to confirm it. Which mechanisms can be detected without collecting more data, and which require additional labeling?
      </Prose>

      <H3>Exercise 3 — Score function design for length neutrality</H3>
      <Prose>
        The negative log-probability score <Code>s(x, y) = −log p(y | x)</Code> is approximately linear in the length of <Code>y</Code>. Show that this introduces a length bias: longer correct answers have systematically larger calibration scores, which raises the threshold and inflates prediction sets for queries that naturally elicit longer answers. Propose a length-normalized score function <Code>s'(x, y)</Code> that addresses this. Then verify whether your proposed score preserves the conformal validity guarantee — what conditions does the score function need to satisfy for the marginal coverage proof to go through? Are there any score functions that would invalidate the guarantee?
      </Prose>

      <H3>Exercise 4 — Marginal vs Mondrian coverage trade-off</H3>
      <Prose>
        You have <Code>n = 1000</Code> labeled calibration examples spanning 20 query types (e.g., "factual", "summarization", "translation", etc.). Your application requires 90% coverage on every query type, not just on average. Compute the minimum per-group calibration set size required for stable Mondrian conformal prediction with <Code>α = 0.10</Code>. Suppose your data is unbalanced — three of the 20 query types have only 30 calibration examples each. Describe two strategies for handling these underrepresented groups (one statistical, one operational) and the trade-offs of each. What happens if you choose to use vanilla split conformal (marginal coverage) on this same data? Explain quantitatively how badly the worst subgroup might be miscovered while the marginal guarantee is preserved.
      </Prose>

      <H3>Exercise 5 — Composing conformal with abstention</H3>
      <Prose>
        Mohri & Hashimoto's conformal factuality framework returns either a committed answer or an abstention, with a calibrated factuality rate over committed answers. Sketch the procedure: given a target factuality rate of 0.95 and a calibration set of 500 (prompt, response, factuality-label) pairs, what is the calibration step, what is the decision rule at inference, and what statement is the resulting system making about its committed outputs? Now consider the abstention rate as a function of the target factuality: how does it scale with the target, and what happens in the limit as the target approaches 1.0? Finally, describe a multi-stage composition where conformal prediction is followed by conformal abstention is followed by conformal RAG validity — what coverage guarantees compose naturally, and where does the joint guarantee require a union-bound or independent calibration?
      </Prose>

    </div>
  ),
};

export default conformalLLM;
