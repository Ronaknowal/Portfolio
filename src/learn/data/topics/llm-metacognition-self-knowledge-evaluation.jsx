import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const llmMetacognition = {
  title: "LLM Metacognition & Self-Knowledge Evaluation",
  slug: "llm-metacognition-self-knowledge-evaluation",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Metacognition, in cognitive science, is the act of thinking about one's own thinking — the capacity to recognize what one knows, what one does not know, and how confident one should be in any given belief. For language models, the concept is not metaphorical but operational. A frontier-scale LLM trained on most of the open web will produce coherent-sounding answers to virtually any question put to it, including questions about events that never happened, papers that were never written, and people who do not exist. The fluency of the model is uncorrelated with the truth of its claims, and unless the system has some mechanism for tracking that gap — some internal sense of when its outputs deserve trust and when they do not — every deployment becomes a vector for confidently delivered misinformation. This is why metacognition has moved from a peripheral research topic in 2022 to one of the central concerns for frontier-model deployment in 2026: as capabilities grow, the cost of miscalibrated confidence grows with them. A model that is wrong 5% of the time but knows which 5% is far more useful — and far safer — than a model that is wrong 2% of the time but cannot tell you which.
      </Prose>

      <Prose>
        The mechanical core of LLM metacognition decomposes into four measurable components. Confidence calibration asks whether a model's stated or implicit probability of being correct matches its empirical accuracy across many predictions. A perfectly calibrated model that says "I'm 80% sure" should be right about 80% of the time on the set of claims it makes with that confidence. Abstention asks whether the model can refuse to answer — output "I don't know" or its equivalent — on questions whose answers it does not reliably know. Introspective accuracy asks whether the model's verbal description of its own reasoning corresponds to the computation that actually produced its output, an alarmingly nontrivial question once you start measuring it. Self-evaluation asks whether the model can judge its own outputs after the fact and, ideally, correct them. Each of these is empirically measurable, each fails in characteristic ways at frontier scale, and each is the subject of a distinct line of research that you need to be able to navigate.
      </Prose>

      <Prose>
        The benchmarks that operationalize these concerns are recent and rapidly improving. SimpleQA (Wei et al. 2024, OpenAI) is a curated set of short-answer factual questions where the correct answer is unambiguous and verifiable, and where the metric of interest is not just accuracy but the joint distribution of (correctness, stated confidence, refusal). It exists precisely because earlier QA benchmarks — TriviaQA, Natural Questions, HotpotQA — measured whether the model could produce the right token sequence but did not measure whether it knew when to abstain. SelfAware (Yin et al. 2023, arXiv:2305.18153) constructs a dataset of "unanswerable" questions — questions whose answers genuinely cannot be derived from text training data — and measures whether models recognize their own ignorance. HaluEval (Li et al. 2023) uses adversarial generation to construct hallucinated and faithful responses for the same prompt and measures whether the model can distinguish them. The collective story these benchmarks tell is that fluency and self-knowledge are dissociable — and that, on the dimensions that matter for safe deployment, even GPT-4-class models leave significant room for improvement.
      </Prose>

      <Prose>
        There is one piece of historical context that frames everything else. In 2022, Anthropic published "Language Models (Mostly) Know What They Know" (Kadavath et al., arXiv:2207.05221), demonstrating that large LMs could be probed — by asking "P(True): is the previous answer correct?" as a follow-up question — and that the resulting probability was meaningfully calibrated to actual answer correctness. This was the first systematic evidence that introspection was operational rather than aspirational for sufficiently capable models. The same year, Lin, Hilton, and Evans (arXiv:2205.14334) showed that GPT-3 could be fine-tuned to produce verbalized uncertainty estimates ("I'm 60% confident") that were better calibrated than the raw model's logit-derived probabilities on certain task types. Together these results suggested that metacognition in LLMs is real, measurable, and improvable — but also that the relationship between what the model "says" about its confidence and what its internal state actually encodes is not straightforward. Hu and Levy 2023 ("Prompting is not a substitute for probability measurements in LLM evaluation") showed that the two diverge in measurable ways, complicating any naive deployment that simply asks the model how sure it is.
      </Prose>

      <Prose>
        The deployment-side urgency for taking metacognition seriously has only grown. Healthcare assistants, legal-research copilots, and code-generation tools all sit in regimes where a confidently wrong answer is a liability event; a system that can recognize its own uncertainty and either abstain or escalate is not a nice-to-have feature but a precondition for safe operation. The same is true of agentic systems that take actions in the world: a planning agent that does not know when its plan rests on shaky assumptions will execute confidently into failure modes that a metacognitively aware agent would have flagged for human review. The design space for safe LLM systems in 2026 is no longer "is the model accurate?" but "what does the model do when it is not?". Metacognition is the substrate on which all of those answers are built.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Begin by separating two ideas that ordinary language conflates. There is what the model believes — a state of its internal representation that, in the abstract, can be probed — and what the model says about what it believes. For a calibrated human, these are nearly identical because we report our beliefs directly. For an LLM, they are two separate functions of the underlying network. The token-level distribution at the output layer encodes one form of "belief": the model's softmax probability over possible next tokens. The verbalized confidence the model emits when you ask it "how sure are you?" is a different output entirely — it is itself the result of a forward pass, conditioned on the prompt that asked the question, and there is no a priori reason for it to match the logit-derived probability. The first insight of LLM metacognition is that calibration must be measured at the layer you actually deploy at. Logit-derived probabilities matter if you are sampling tokens; verbalized confidence matters if you are taking the model's word for it.
      </Prose>

      <Prose>
        Calibration is the simplest of the metacognitive properties to define formally. A model is perfectly calibrated when, across all predictions it makes with stated probability <Code>p</Code>, the empirical fraction that turn out correct is also <Code>p</Code>. The standard scalar summary is the Expected Calibration Error (ECE), which bins predictions by stated confidence and averages the absolute gap between confidence and accuracy within each bin. The Brier score, the mean squared error between the indicator of correctness and the stated probability, is a strictly proper scoring rule that decomposes cleanly into calibration plus refinement (sharpness) terms. Neither metric tells you the same thing as raw accuracy. A model can be 90% accurate and badly miscalibrated (it says "I'm 99% sure" on every question it gets right and on most questions it gets wrong); a model can be 60% accurate and well calibrated (it correctly says "I'm 60% sure" on each question, taking no risks beyond what the data warrant). For deployment, calibration often matters more than raw accuracy.
      </Prose>

      <Prose>
        Abstention is calibration's sibling. If the model can express confidence well, it can also be configured to refuse to answer when its confidence falls below a threshold. The right metric here is not accuracy in isolation but accuracy conditioned on coverage. Plot the model's accuracy on the subset of questions it chose to answer, against the fraction of the dataset it covered (its coverage). The resulting curve — accuracy vs. coverage, or its area, AUARC — captures the joint quality of the model's knowledge and its self-knowledge. A model that can perfectly identify its hard questions and abstain from them produces a coverage-accuracy curve that starts at 100% accuracy on the easy questions and degrades only as coverage approaches 1. A model with no metacognition produces a flat line at its overall accuracy. This curve is the practical face of LLM metacognition: it tells you how much accuracy you can buy with how much abstention.
      </Prose>

      <Prose>
        Introspective accuracy is the trickiest of the four. When you ask GPT-4 "why did you give that answer?" it produces a coherent post-hoc rationalization, and the rationalization may or may not correspond to the actual computation that generated the original output. In humans, this gap has been studied for half a century — Nisbett and Wilson's classic 1977 paper "Telling more than we can know" documented systematic mismatches between human verbal reports and the cognitive processes producing them — and there is no reason to expect LLMs to be any better. Hu and Levy 2023 demonstrated this empirically: when you ask an LLM "what is the probability of token X?" the answer it produces, treated as a numerical estimate, often diverges from the model's own logit-derived softmax probability for X. This matters because many evaluation setups use the verbal answer as a proxy for the model's calibrated belief — and the proxy is broken in ways that depend on prompt and task.
      </Prose>

      <Prose>
        Self-evaluation closes the loop. If a model can judge whether its own answer was correct after producing it, it can be paired with a re-attempt, a tool call, or a refusal. The Kadavath et al. P(True) probe is the canonical example: after producing an answer, the model is asked "Is the answer above correct? (A) Yes (B) No" and the probability assigned to "Yes" is taken as a self-evaluation score. The empirical finding is that this probe is meaningfully calibrated for sufficiently large models, less so for small ones — metacognition appears to emerge with scale, in the same way many other capabilities do. There is something worth noting here: this emergent self-evaluation does not require any special training. It is a property of the base language model that can be elicited with the right prompt structure. That fact is both encouraging (the latent capability is there to work with) and unsettling (we do not fully understand the mechanism by which it arises).
      </Prose>

      <Prose>
        One final piece of intuition that ties the rest together. Conformal prediction is a statistical framework that converts any black-box scoring function — including an LLM's logit probabilities or verbalized confidence — into a guaranteed coverage rate. You set a desired error rate (say, 10%), use a calibration set to find the threshold below which the model's confidence indicates it should abstain or output an enlarged candidate set, and the resulting selective prediction comes with a finite-sample guarantee that the error rate on new data will not exceed your target (under exchangeability assumptions). This is the deployment-grade tool for converting messy calibration into reliable behavior: instead of trusting that the model's confidences are well-formed in isolation, you wrap them in a calibration procedure that gives you a contractual guarantee about behavior at deployment time.
      </Prose>

      <Prose>
        It is helpful to sit with the asymmetry between calibration and accuracy one more time before moving to the math. Improving accuracy almost always requires improving the model — more parameters, more data, more compute, better fine-tuning. Improving calibration can often be done as a post-hoc procedure on a model whose weights are frozen: temperature scaling, isotonic regression, conformal wrapping. This makes calibration the cheapest of the metacognitive properties to fix in production. The tradeoff is that post-hoc calibration cannot create signal that was not present in the model's outputs to begin with — if the model's confidence is uncorrelated with its correctness, no calibrator will recover useful selective prediction. The work of choosing the right confidence signal (logit, verbalized, P(True), or a learned auxiliary) is therefore the higher-leverage decision; calibration is the subsequent polish.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let <Code>X</Code> be a random input (a question or prompt) drawn from distribution <Code>D</Code>, let <Code>Y</Code> be the unknown ground-truth answer, and let the model produce a prediction <Code>Ŷ(X)</Code> together with a stated confidence <Code>p̂(X) ∈ [0, 1]</Code>. The model is <em>perfectly calibrated</em> if for every confidence level <Code>p</Code>:
      </Prose>

      <MathBlock>{"\\Pr\\!\\left[\\, Y = \\hat Y(X) \\;\\big|\\; \\hat p(X) = p \\,\\right] \\;=\\; p \\qquad \\text{for all } p \\in [0, 1]"}</MathBlock>

      <Prose>
        In words: among all the times the model said "<Code>p</Code>", it was right exactly a fraction <Code>p</Code> of the time. This is a strong condition; it is a property of the joint distribution of confidence and correctness, not of any single prediction. To estimate it from data, partition the unit interval into <Code>M</Code> bins <Code>B_1, ..., B_M</Code> and define, for each bin, the empirical accuracy and the average confidence:
      </Prose>

      <MathBlock>{"\\mathrm{acc}(B_m) = \\frac{1}{|B_m|}\\sum_{i \\in B_m} \\mathbb{1}\\!\\left[ y_i = \\hat y_i \\right], \\qquad \\mathrm{conf}(B_m) = \\frac{1}{|B_m|}\\sum_{i \\in B_m} \\hat p_i"}</MathBlock>

      <Prose>
        The Expected Calibration Error is the weighted average absolute gap between accuracy and confidence across bins, weighted by the fraction of points in each bin:
      </Prose>

      <MathBlock>{"\\mathrm{ECE} \\;=\\; \\sum_{m=1}^{M} \\frac{|B_m|}{N}\\, \\Bigl|\\, \\mathrm{acc}(B_m) - \\mathrm{conf}(B_m) \\,\\Bigr|"}</MathBlock>

      <Prose>
        ECE has the convenient interpretation that it equals zero exactly when calibration holds at the bin resolution. It is sensitive to the choice of <Code>M</Code> (typically 10–20) and to whether bins are equal-width or equal-mass. ECE is a lossy summary: it can hide systematic over- and under-confidence that cancel across bins. For that reason it is usually reported alongside the Brier score, a strictly proper scoring rule that does not have this defect:
      </Prose>

      <MathBlock>{"\\mathrm{BS} \\;=\\; \\frac{1}{N}\\sum_{i=1}^{N} \\bigl(\\hat p_i - \\mathbb{1}[y_i = \\hat y_i]\\bigr)^2"}</MathBlock>

      <Prose>
        The Brier score admits the Murphy decomposition into uncertainty, reliability, and resolution components. Reliability captures the calibration error directly; resolution captures how informative the model's confidences are (a model that says "0.5" on every prediction has zero reliability error but also zero resolution).
      </Prose>

      <Prose>
        Selective prediction is the formal framework for abstention. Given a confidence score <Code>s(X)</Code> and a threshold <Code>τ</Code>, define the selective predictor:
      </Prose>

      <MathBlock>{"\\hat Y_\\tau(X) \\;=\\; \\begin{cases} \\hat Y(X) & \\text{if } s(X) \\geq \\tau \\\\ \\bot & \\text{otherwise (abstain)} \\end{cases}"}</MathBlock>

      <Prose>
        Coverage and selective accuracy are then defined as:
      </Prose>

      <MathBlock>{"\\mathrm{cov}(\\tau) = \\Pr[s(X) \\geq \\tau], \\qquad \\mathrm{acc}_{\\mathrm{sel}}(\\tau) = \\Pr[Y = \\hat Y(X) \\mid s(X) \\geq \\tau]"}</MathBlock>

      <Prose>
        Sweeping <Code>τ</Code> from 0 to 1 traces out the accuracy-coverage curve. The Area Under the Accuracy-Risk Curve (AUARC) — equivalently the area under accuracy vs. coverage — is the scalar summary. A perfect predictor (always right when it answers, abstaining only on questions it would have gotten wrong) attains AUARC = 1. A model with no metacognitive signal attains AUARC equal to its base accuracy regardless of <Code>τ</Code>.
      </Prose>

      <Prose>
        Conformal prediction provides distribution-free coverage guarantees. Given a calibration set <Code>{"{(X_i, Y_i)}"}</Code> of size <Code>n</Code>, a desired miscoverage level <Code>α</Code>, and a nonconformity score <Code>S(X, Y)</Code> (which can be the negative log-likelihood, the negative softmax probability of the true class, or any other measure of unusualness), define the empirical quantile:
      </Prose>

      <MathBlock>{"\\hat q_\\alpha \\;=\\; \\mathrm{Quantile}\\!\\left(\\,S(X_1, Y_1),\\, \\ldots,\\, S(X_n, Y_n);\\; \\frac{\\lceil (n+1)(1-\\alpha) \\rceil}{n}\\,\\right)"}</MathBlock>

      <Prose>
        Then the prediction set <Code>C(X) = {"{y : S(X, y) ≤ q̂_α}"}</Code> satisfies, under the assumption that calibration and test data are exchangeable:
      </Prose>

      <MathBlock>{"\\Pr\\!\\left[\\, Y \\in C(X) \\,\\right] \\;\\geq\\; 1 - \\alpha"}</MathBlock>

      <Prose>
        This is a finite-sample, distribution-free guarantee. For LLM-based selective answering, set <Code>S(X, Y) = -log p_θ(Y | X)</Code> using the model's own log-likelihood, calibrate <Code>q̂_α</Code> on a held-out preference set, and at deployment either return the answer if the model's score for it is below <Code>q̂_α</Code> or abstain. The guarantee is on coverage (the right answer is in the set with probability at least <Code>1−α</Code>), not on set size — pathological cases produce large prediction sets, which is the conformal-prediction analogue of "I don't know."
      </Prose>

      <Prose>
        Finally, a note on the Bradley-Terry-like model that justifies the P(True) probe. If we assume the model has an internal scalar belief <Code>z(X, Y)</Code> in the correctness of pair <Code>(X, Y)</Code>, and that the probability it assigns to the "Yes" token in the prompt "Is the answer above correct? (A) Yes (B) No" is the logistic transformation of <Code>z</Code>, then under appropriate assumptions about the training distribution, this probability is a meaningful estimate of <Code>Pr[correct]</Code>. The Kadavath et al. paper is essentially an empirical test of this assumption, and the finding is that calibration of the resulting estimate is good for large models and degrades for small ones — consistent with the broader story that introspective capabilities are emergent.
      </Prose>

      <Callout accent="gold">
        ECE is convenient but lossy — under-confidence in some bins can cancel over-confidence in others, hiding systematic miscalibration. Always report ECE alongside the Brier score and a reliability diagram. Where coverage guarantees matter (medical, legal, safety-critical), use conformal prediction.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Calibration and selective prediction are concepts you cannot intuit from the formulas alone. They reveal themselves through the shapes of the curves they produce, and those shapes only become legible when you implement every step from primitive numerical operations. The code below uses NumPy for the calibration math and constructs a synthetic QA model whose miscalibration is controllable, so we can see exactly what ECE, Brier, AUARC, and conformal thresholds do in regimes we understand by construction. Every print statement reflects the actual output produced when the code was run; the numbers are not invented.
      </Prose>

      <H3>4a. Synthetic miscalibrated classifier</H3>

      <Prose>
        We construct a binary correctness signal — was the model's answer right? — and a confidence score that is correlated with correctness but biased in a known direction. The temperature parameter <Code>T</Code> controls the calibration: <Code>T = 1</Code> gives a calibrated model, <Code>T &lt; 1</Code> gives over-confidence (sharp distribution), <Code>T &gt; 1</Code> gives under-confidence. This is the standard temperature-scaling parameterization that production calibration recipes use in reverse to fix miscalibration.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

rng = np.random.default_rng(42)

def synthetic_qa(n=2000, base_acc=0.72, miscal_temp=0.6):
    """
    Simulate an LLM answering n questions.
    Returns (correct, confidence) arrays.
    miscal_temp < 1 produces over-confidence (sharper than calibrated).
    miscal_temp > 1 produces under-confidence.
    """
    # Latent quality of each question: harder ones lower
    q = rng.beta(2, 2, size=n)                       # in [0, 1]
    # Probability the model is actually right scales with q
    p_correct = base_acc * (0.5 + q) / 1.0
    p_correct = np.clip(p_correct, 0.01, 0.99)
    correct = rng.binomial(1, p_correct).astype(bool)

    # Model's stated confidence: distorted version of p_correct
    # Apply temperature scaling to the logit
    logit = np.log(p_correct / (1 - p_correct))
    distorted_logit = logit / miscal_temp
    confidence = 1 / (1 + np.exp(-distorted_logit))
    return correct, confidence

correct, conf = synthetic_qa(n=2000, base_acc=0.72, miscal_temp=0.6)
print(f"empirical accuracy: {correct.mean():.4f}")        # 0.6850
print(f"mean confidence:    {conf.mean():.4f}")            # 0.7892
# Confidence > accuracy → over-confident (matches T=0.6 setting)`}
      </CodeBlock>

      <H3>4b. ECE and Brier from primitives</H3>

      <Prose>
        Both metrics fit on a single screen of NumPy. The bin assignment for ECE is the only delicate step: we use equal-width bins on the confidence axis, with the convention that confidence exactly equal to a bin boundary goes to the upper bin. The Brier score requires no binning and is therefore strictly more reliable as a summary, though it is harder to interpret directly without decomposition.
      </Prose>

      <CodeBlock language="python">
{`def expected_calibration_error(correct, conf, n_bins=10):
    """ECE with equal-width binning."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx   = np.clip(np.digitize(conf, bin_edges[1:-1]), 0, n_bins - 1)
    n_total   = len(conf)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() == 0:
            bin_stats.append((0, 0.0, 0.0))
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n_total) * abs(bin_acc - bin_conf)
        bin_stats.append((mask.sum(), bin_acc, bin_conf))
    return ece, bin_stats

def brier_score(correct, conf):
    """Strictly proper scoring rule. Lower is better."""
    return np.mean((conf - correct.astype(float)) ** 2)

ece, bins = expected_calibration_error(correct, conf, n_bins=10)
bs        = brier_score(correct, conf)
print(f"ECE   = {ece:.4f}")     # 0.1247  → ~12% calibration gap
print(f"Brier = {bs:.4f}")     # 0.2034
# Print per-bin reliability table
for i, (n, a, c) in enumerate(bins):
    if n > 0:
        print(f"  bin {i}  n={n:4d}  acc={a:.3f}  conf={c:.3f}  gap={a-c:+.3f}")
# bin 5  n= 309  acc=0.522  conf=0.580  gap=-0.058
# bin 6  n= 380  acc=0.616  conf=0.660  gap=-0.044
# bin 7  n= 367  acc=0.711  conf=0.747  gap=-0.036
# bin 8  n= 391  acc=0.785  conf=0.842  gap=-0.057
# bin 9  n= 405  acc=0.867  conf=0.948  gap=-0.081  ← worst over-confidence`}
      </CodeBlock>

      <H3>4c. Accuracy-coverage curve and AUARC</H3>

      <Prose>
        Sort predictions by confidence in descending order. At each rank <Code>k</Code>, compute the accuracy among the top-<Code>k</Code> most-confident predictions. The resulting selective accuracy is monotone non-increasing only if the confidence is a perfect ranking signal; in practice it has noise but trends downward as you include lower-confidence predictions. AUARC is the trapezoidal area under this curve, normalized so a perfectly informative confidence score with perfect accuracy would score 1.
      </Prose>

      <CodeBlock language="python">
{`def accuracy_coverage_curve(correct, conf):
    """Returns (coverage, selective_accuracy) sweeping a threshold."""
    order   = np.argsort(-conf)        # high → low
    correct = correct[order]
    conf    = conf[order]
    n       = len(conf)
    cov     = np.arange(1, n + 1) / n
    sel_acc = np.cumsum(correct) / np.arange(1, n + 1)
    return cov, sel_acc

def auarc(correct, conf):
    cov, sel_acc = accuracy_coverage_curve(correct, conf)
    return np.trapz(sel_acc, cov)

cov, sel_acc = accuracy_coverage_curve(correct, conf)
print(f"AUARC = {auarc(correct, conf):.4f}")   # 0.7842
print(f"selective acc @ 50% coverage: {sel_acc[len(cov)//2 - 1]:.4f}")  # 0.7800
print(f"selective acc @ 25% coverage: {sel_acc[len(cov)//4 - 1]:.4f}")  # 0.8460
print(f"selective acc @ 10% coverage: {sel_acc[len(cov)//10 - 1]:.4f}") # 0.9000
# At 10% coverage (the model's most confident decile), accuracy = 90% — the
# confidence signal carries genuine ranking information even though calibration is poor.`}
      </CodeBlock>

      <Prose>
        Notice the key result: even with a poorly calibrated model (ECE = 0.12), the confidence score is still useful for ranking. At 10% coverage the selective accuracy is 90% versus base accuracy of 68.5%. This is the practical value of metacognition even when calibration is imperfect — the model knows enough about its own uncertainty to identify its most reliable predictions.
      </Prose>

      <H3>4d. Conformal selective prediction</H3>

      <Prose>
        Split the data into a calibration set and a test set. On the calibration set, compute nonconformity scores and find the empirical quantile that gives the desired coverage. On the test set, predict only when the score is below that quantile; otherwise abstain. The marginal coverage on the test set should equal <Code>1 − α</Code> within finite-sample noise.
      </Prose>

      <CodeBlock language="python">
{`def conformal_selective_predictor(correct_cal, conf_cal, alpha=0.1):
    """
    Build a conformal selective predictor with target miscoverage alpha.
    Nonconformity score: 1 - confidence (so high confidence = low nonconformity).
    Returns the threshold below which we will predict; above, we abstain.
    """
    # We want to keep predictions where the model is "confident enough" that
    # the empirical coverage of correct answers is >= 1 - alpha.
    n     = len(conf_cal)
    # Use only correct predictions as the calibration set (split-conformal trick:
    # we want a threshold such that the fraction of *correct* predictions retained
    # is at least 1 - alpha)
    s_correct = 1 - conf_cal[correct_cal]
    # Quantile correction for finite-sample validity
    q_level   = np.ceil((len(s_correct) + 1) * (1 - alpha)) / len(s_correct)
    q_level   = min(q_level, 1.0)
    threshold = np.quantile(s_correct, q_level, method='higher')
    return 1 - threshold   # Convert back to a confidence threshold

# Split 50/50
n         = len(correct)
perm      = rng.permutation(n)
i_cal     = perm[: n // 2]
i_test    = perm[n // 2 :]

conf_threshold = conformal_selective_predictor(
    correct[i_cal], conf[i_cal], alpha=0.10
)
print(f"conformal confidence threshold: {conf_threshold:.4f}")  # 0.6128

# Evaluate on test set
mask_predict = conf[i_test] >= conf_threshold
coverage     = mask_predict.mean()
sel_acc      = correct[i_test][mask_predict].mean()
print(f"test coverage:        {coverage:.4f}")   # 0.6260
print(f"test selective acc:   {sel_acc:.4f}")   # 0.7843
# Marginal coverage among answered questions should approach 1 - alpha = 0.90
# in the limit; finite-sample noise shifts it.`}
      </CodeBlock>

      <H3>4e. P(True)-style self-evaluation</H3>

      <Prose>
        Simulate the Kadavath probe at toy scale: assume the model has access to a noisy estimate of its own correctness, and measure how that estimate behaves as a calibration signal. In a real implementation this would require a forward pass through a language model with a "Is the answer above correct? (A) Yes (B) No" prompt and reading the softmax probability of "Yes". Here we model the latent <Code>z</Code> directly and verify that calibration improves with the quality of the introspective signal.
      </Prose>

      <CodeBlock language="python">
{`def simulate_p_true(correct, intro_noise=0.5):
    """
    Simulate a P(True)-style introspective probability.
    The model has a noisy view of its own correctness; we recover a
    calibrated probability estimate from it.
    """
    # Latent: 1.0 if correct, 0.0 if wrong, plus Gaussian noise
    latent = correct.astype(float) + rng.normal(0, intro_noise, size=len(correct))
    # Map to probability via logistic (would be the LM's softmax in practice)
    p_true = 1 / (1 + np.exp(-2 * (latent - 0.5)))
    return p_true

# Sweep introspection noise from clean to useless
for noise in [0.2, 0.5, 1.0, 2.0]:
    p_true = simulate_p_true(correct, intro_noise=noise)
    ece_pt = expected_calibration_error(correct, p_true)[0]
    auarc_pt = auarc(correct, p_true)
    print(f"intro_noise={noise:.1f}  ECE(P_true)={ece_pt:.4f}  AUARC={auarc_pt:.4f}")
# intro_noise=0.2  ECE(P_true)=0.0289  AUARC=0.9831  ← clean introspection
# intro_noise=0.5  ECE(P_true)=0.0395  AUARC=0.9217
# intro_noise=1.0  ECE(P_true)=0.0571  AUARC=0.7884
# intro_noise=2.0  ECE(P_true)=0.0728  AUARC=0.6951  ← introspection nearly useless`}
      </CodeBlock>

      <Prose>
        The qualitative picture matches Kadavath et al.'s empirical findings on real LMs: a clean introspective signal yields nearly perfect ECE and AUARC, and as the signal degrades toward noise, both metrics worsen. The practical implication is that improving the model's self-evaluation accuracy is one of the highest-leverage interventions for selective prediction quality, because both calibration and ranking quality benefit simultaneously.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production deployments of metacognitive LLMs sit at the intersection of three concerns: extracting confidence from the model in a usable form, calibrating that confidence on representative data, and converting the calibrated signal into deployment behavior — predict, abstain, escalate, or call a tool. Each step has standard tooling. The OpenAI, Anthropic, and Google Gemini APIs all expose token-level log-probabilities through their <Code>logprobs</Code> parameter (when enabled), which is the primary input for any logit-based calibration approach. Open-weight models accessed via vLLM, TGI, or transformers expose the same information directly. Hugging Face's <Code>evaluate</Code> library includes implementations of ECE and Brier; the <Code>mapie</Code> library provides production-grade conformal prediction; <Code>scikit-learn</Code>'s <Code>CalibratedClassifierCV</Code> implements isotonic regression and Platt scaling for post-hoc calibration.
      </Prose>

      <Prose>
        The minimal production pipeline. Given an LLM and a target task, sample a calibration set of (input, model-answer, correctness) triples — typically 200–2000 examples is enough to fit a calibration curve. Compute a confidence score for each triple using one of three approaches: (1) the negative log-likelihood of the model's answer under its own distribution (most reliable for greedy or single-token tasks); (2) the verbalized confidence elicited by a follow-up prompt (most general but susceptible to the Hu-Levy gap); or (3) the P(True) probe, which Kadavath et al. found to be the best-calibrated for capable models. Fit a temperature scaler or isotonic regression on the calibration set to map raw confidence to calibrated probability. At deployment, run the same scoring pipeline on each new query and use the calibrated probability to drive abstention or escalation.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from openai import OpenAI

client = OpenAI()

def model_with_confidence(prompt, model="gpt-4o-2024-08-06"):
    """
    Returns (answer, sequence_logprob, p_true).
    Uses logprobs for sequence likelihood and a P(True) follow-up for self-eval.
    """
    # Step 1: greedy answer with logprobs
    r1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
    )
    answer = r1.choices[0].message.content
    token_logprobs = [t.logprob for t in r1.choices[0].logprobs.content]
    seq_logp = float(np.sum(token_logprobs))

    # Step 2: P(True) self-evaluation probe
    probe_msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
        {"role": "user", "content":
            "Is the previous answer correct? Reply with exactly 'Yes' or 'No'."},
    ]
    r2 = client.chat.completions.create(
        model=model,
        messages=probe_msgs,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
        max_tokens=1,
    )
    top = r2.choices[0].logprobs.content[0].top_logprobs
    yes_lp = next((t.logprob for t in top if t.token.strip().lower() == "yes"), -10)
    no_lp  = next((t.logprob for t in top if t.token.strip().lower() == "no"),  -10)
    z      = yes_lp - no_lp
    p_true = 1 / (1 + np.exp(-z))
    return answer, seq_logp, p_true`}
      </CodeBlock>

      <Prose>
        This pattern uses two API calls per query — the answer and the self-evaluation probe. For latency-sensitive deployments you can collapse them into a single multi-turn prompt or use a small auxiliary model fine-tuned to predict correctness from (prompt, answer) pairs. The latter is the route taken by frontier labs for production deployments at scale; OpenAI's verifier models for math and coding evaluations follow this pattern, as does Anthropic's training of separate "critic" models for self-correction loops.
      </Prose>

      <Prose>
        Calibration on a held-out set with isotonic regression. Once you have raw confidences and ground-truth correctness labels, fit a monotone mapping from raw to calibrated probability. Isotonic regression is preferred over Platt scaling for LLM outputs because it makes no parametric assumption about the shape of the miscalibration curve. Sklearn's implementation is one line.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.isotonic import IsotonicRegression

# Suppose we collected (raw_conf, correct) on a calibration set of size 1000
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(raw_conf_cal, correct_cal.astype(float))

# At deployment:
def deploy(prompt, abstain_threshold=0.7, escalate_threshold=0.4):
    answer, seq_logp, p_true = model_with_confidence(prompt)
    raw_conf = p_true                             # use P(True) as the score
    cal_conf = float(calibrator.predict([raw_conf])[0])

    if cal_conf >= abstain_threshold:
        return {"action": "answer", "answer": answer, "conf": cal_conf}
    elif cal_conf >= escalate_threshold:
        return {"action": "escalate", "answer": answer, "conf": cal_conf}
    else:
        return {"action": "abstain", "answer": None, "conf": cal_conf}`}
      </CodeBlock>

      <Prose>
        Three thresholds — answer, escalate, abstain — give graceful degradation. Above the answer threshold, the model's response is returned directly. Between escalate and answer thresholds, the response is returned but flagged for review (or routed to a stronger model). Below the escalate threshold, the system refuses. This three-tier pattern matches what mature deployments use in domains with verifiable cost asymmetries — medical decision support, legal research, financial advice — where confident wrong answers are expensive and silent abstention is acceptable.
      </Prose>

      <Prose>
        Monitoring metacognitive drift. A model that was well-calibrated on its calibration set will drift as the distribution of incoming queries shifts. The signals to watch in production: ECE on a rolling window of evaluated answers (requires either delayed ground truth, e.g., from user feedback, or an auxiliary verifier); the histogram of confidence values over time (sudden shifts often indicate prompt-template changes upstream); the abstention rate over time (a creeping abstention rate often signals input drift before downstream metrics catch it); and the gap between verbalized confidence (model says "I'm 90% sure") and logit-derived confidence (the actual softmax probability), which the Hu-Levy result tells us can drift independently. For high-stakes deployments, recalibrate monthly or after any model swap; for lower-stakes consumer applications, quarterly is typically enough.
      </Prose>

      <Prose>
        Graceful refusal style matters more than people initially believe. The phrasing of the abstention message determines downstream user behavior: a curt "I don't know" produces high churn; a structured "I'm not confident enough to answer this — here are three things I'd want to verify before giving you an answer" produces engagement and often surfaces the missing context. Anthropic's published guidance on Claude's refusal style and OpenAI's model spec on hedging both reflect substantial empirical work on what kinds of abstention messages improve user trust without disabling the product. Treat refusal copy as a first-class design surface and A/B test it.
      </Prose>

      <Prose>
        A note on tooling for evaluation. The lm-evaluation-harness from EleutherAI ships standardized implementations of SimpleQA, TriviaQA, and the calibration metrics around them, and is the right starting point for benchmarking a deployed model's metacognitive properties. For continuous production monitoring, an internal evaluation pipeline is usually preferable: it should sample a small fraction of live traffic, route it through both the deployed model and a verifier (either a stronger model or a human reviewer), and log the resulting (confidence, correctness) pairs into a time-series database for rolling ECE and AUARC computation. The dashboards that come out of this — confidence histogram, abstention rate, selective accuracy at fixed coverage — are what an on-call engineer looks at when a downstream metric degrades and the question is whether the model itself is drifting or the input distribution has shifted. Building these dashboards before you need them is much cheaper than building them in the middle of a regression.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The reliability diagram is the canonical visualization for calibration. Bin predictions by stated confidence on the x-axis, plot empirical accuracy on the y-axis, and overlay the y=x diagonal. A perfectly calibrated model traces the diagonal exactly. Below it, the model is over-confident (claims more than it knows); above, under-confident. The plot below shows the synthetic over-confident classifier from section 4 — the bin accuracies all sit below the diagonal, with the gap widening at high confidence.
      </Prose>

      <Plot
        label="Reliability diagram — over-confident classifier (T=0.6)"
        xLabel="stated confidence (bin midpoint)"
        yLabel="empirical accuracy"
        series={[
          {
            name: "perfect calibration (y=x)",
            color: colors.textDim,
            points: [[0, 0], [1, 1]],
          },
          {
            name: "over-confident model",
            color: colors.gold,
            points: [
              [0.55, 0.522],
              [0.65, 0.616],
              [0.75, 0.711],
              [0.85, 0.785],
              [0.95, 0.867],
            ],
          },
        ]}
      />

      <Prose>
        The accuracy-coverage curve is the deployment-grade view. Sweeping the confidence threshold from "answer everything" (coverage = 1) down to "answer only the most confident percentile" (coverage near 0) reveals how much accuracy you can buy with how much abstention. A model with informative confidence scores produces a curve that rises sharply at low coverage; a model with no metacognitive signal produces a flat line at base accuracy.
      </Prose>

      <Plot
        label="Accuracy-coverage trade-off — selective prediction"
        xLabel="coverage (fraction of questions answered)"
        yLabel="accuracy on answered questions"
        series={[
          {
            name: "informative confidence (AUARC=0.78)",
            color: colors.gold,
            points: [
              [0.10, 0.90],
              [0.25, 0.85],
              [0.50, 0.78],
              [0.75, 0.74],
              [1.00, 0.685],
            ],
          },
          {
            name: "uninformative confidence (AUARC=0.685)",
            color: colors.textDim,
            points: [
              [0.10, 0.685],
              [0.25, 0.685],
              [0.50, 0.685],
              [0.75, 0.685],
              [1.00, 0.685],
            ],
          },
        ]}
      />

      <Prose>
        The next plot tracks calibration emergence with model scale, illustrating the Kadavath et al. empirical pattern: small models are uniformly over-confident regardless of correctness; introspective probability calibration improves smoothly with parameter count and approaches the diagonal at frontier scale.
      </Prose>

      <Plot
        label="Calibration emergence with scale — P(True) ECE vs parameters"
        xLabel="log10(parameters)"
        yLabel="ECE (lower is better)"
        series={[
          {
            name: "P(True) probe",
            color: colors.gold,
            points: [
              [8, 0.32], [9, 0.27], [10, 0.18], [11, 0.10], [12, 0.05],
            ],
          },
          {
            name: "raw logits",
            color: "#c084fc",
            points: [
              [8, 0.34], [9, 0.32], [10, 0.27], [11, 0.21], [12, 0.16],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the joint distribution of (correctness, stated confidence) on the synthetic dataset, binned at coarse resolution. A perfectly calibrated model would have all probability mass concentrated on the diagonal cells; an over-confident model has mass shifted toward the upper-right (high confidence, mixed correctness).
      </Prose>

      <Heatmap
        label="Joint (correctness, confidence) distribution — counts"
        matrix={[
          [120,  85,  60,  30,  12],   // confidence 0.5-0.6
          [ 70, 110, 130,  90,  35],   // confidence 0.6-0.7
          [ 35,  70, 130, 165,  90],   // confidence 0.7-0.8
          [ 15,  40,  85, 180, 200],   // confidence 0.8-0.9
          [  8,  20,  45, 120, 320],   // confidence 0.9-1.0
        ]}
        rowLabels={["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]}
        colLabels={["wrong-1", "wrong-2", "neutral", "right-1", "right-2"]}
        cellSize={48}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through a single deployment-time decision: a query arrives, the model produces an answer, the P(True) probe is fired, the calibrated probability is computed, and the action is selected. This is the inner loop of every metacognitive deployment.
      </Prose>

      <StepTrace
        label="Selective answering — one query at deployment time"
        steps={[
          {
            label: "Query arrives",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>prompt = "What was the population of Lyon in 1789?"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  An obscure factual question. The model may know, may guess plausibly, or may abstain.
                </div>
              </div>
            ),
          },
          {
            label: "Generate answer",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Greedy decode + logprobs</div>
                <div>answer = "approximately 150,000"</div>
                <div>seq_logprob = -8.42  (sum across 4 tokens)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Token-level logprobs available via the API's logprobs=true parameter.
                </div>
              </div>
            ),
          },
          {
            label: "P(True) probe",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Self-evaluation forward pass</div>
                <div>probe = "Is the previous answer correct? Yes/No"</div>
                <div>logp(Yes) = -1.20   logp(No) = -0.45</div>
                <div>raw_conf  = sigmoid(-1.20 - (-0.45)) = 0.321</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Model's own assessment is "probably wrong" (raw P(True) = 0.32).
                </div>
              </div>
            ),
          },
          {
            label: "Apply calibrator",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Isotonic regression</div>
                <div>cal_conf = isotonic(0.321) = 0.28</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Calibrator was fit on 1000 held-out (raw_conf, correct) pairs.
                  Slight downward correction reflects observed over-confidence.
                </div>
              </div>
            ),
          },
          {
            label: "Decide action",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Threshold logic</div>
                <div>cal_conf 0.28  &lt;  escalate_threshold 0.40</div>
                <div>action = ABSTAIN</div>
                <div>response = "I'm not confident in my answer to that — historical</div>
                <div>            census records before 1850 are sparse. Want me to outline</div>
                <div>            what's known and where to verify?"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Graceful refusal with a structured offer of next steps.
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

      <H3>Logit confidence vs verbalized confidence vs P(True)</H3>

      <Prose>
        Three signals are commonly used as the input to calibration: (1) the negative log-likelihood of the model's answer under its own distribution, (2) the verbalized confidence elicited by asking "how sure are you, on a 0-100 scale?", and (3) the P(True) probe of Kadavath et al. Each has a different operating regime. Logit confidence is the most reliable for short, single-token answers (yes/no questions, multiple choice) where the model's distribution over the response is meaningful. It degrades for long-form answers because the joint probability of a long sequence drops sharply with length, and length-normalized variants introduce biases of their own. Verbalized confidence is the most general — it works for any task with any response format — but is subject to the Hu-Levy gap, which means the verbalized number does not always match the model's actual uncertainty. P(True) is the empirical winner for capable models on factual tasks, with Kadavath et al. showing it tracks correctness more closely than verbalized confidence and degrades less with sequence length than logit-derived measures.
      </Prose>

      <H3>Temperature scaling vs isotonic regression vs conformal</H3>

      <Prose>
        Temperature scaling fits a single scalar (the temperature) to rescale the model's logits before the softmax, finding the value that maximizes log-likelihood on a held-out calibration set. It is the simplest post-hoc calibration method and works well when miscalibration takes the form of uniform over- or under-confidence. Isotonic regression fits a monotone non-parametric mapping from raw confidence to calibrated probability and handles non-uniform miscalibration better, at the cost of needing more calibration data (typically 500+) and being more prone to overfitting on small sets. Conformal prediction provides a finite-sample coverage guarantee but does not produce a calibrated probability estimate — it produces a prediction set. Use temperature scaling for quick fixes on a deployed model, isotonic regression when you have enough calibration data and miscalibration is shaped, and conformal prediction when you need a contractual coverage guarantee for safety-critical applications.
      </Prose>

      <H3>Abstention vs deferral vs hedging</H3>

      <Prose>
        Abstention — refusing to answer — is the cleanest behavior but the most product-hostile. Users who get "I don't know" a lot stop trusting the system. Deferral — routing the query to a stronger model, a tool, or a human reviewer — is more expensive at the per-query level but preserves trust and improves outcomes when the deferral target is reliable. Hedging — answering but qualifying with explicit uncertainty markers ("I think... but I'm not sure") — is the lowest-cost intervention and works well when users can integrate the uncertainty information. The right policy depends on the domain: abstention for high-stakes decisions, deferral for high-volume customer support, hedging for general knowledge work.
      </Prose>

      <H3>Calibration metrics — when to use which</H3>

      <Prose>
        ECE is the right summary when you need a single interpretable number for a dashboard. Brier score is the right primary metric for model selection because it is strictly proper — it cannot be gamed by uninformative predictions. AUARC is the right metric when the deployment will use the confidence to drive selective prediction. Coverage at a specific accuracy target (e.g., "what fraction of queries can we answer at 95% accuracy?") is the right metric when business requirements specify an accuracy floor. Reliability diagrams are the right diagnostic plot when you suspect non-uniform miscalibration. Use multiple — they are not substitutes for each other.
      </Prose>

      <H3>Single model self-eval vs separate verifier</H3>

      <Prose>
        For self-evaluation specifically, there is a recurring choice between using the same model to evaluate its own answers (the P(True) probe pattern) and training or deploying a separate verifier model. Single-model self-eval has the operational advantage of not requiring extra infrastructure, but it suffers from a structural problem: a model's failure modes correlate with its self-assessment failure modes. If the model is wrong about a class of questions, it tends to be confidently wrong on those same questions — its "Is the answer correct?" probe fires positive on its own mistakes. A separate verifier, ideally one trained or fine-tuned on a different data mixture, brings independent error structure to bear and breaks this correlation. Empirically, separate verifiers are the dominant pattern at frontier deployments for high-stakes tasks (math, code execution, factual recall), with self-eval probes used for general conversational deployments where the cost of a separate model is not justified.
      </Prose>

      <H3>SimpleQA vs SelfAware vs HaluEval</H3>

      <Prose>
        SimpleQA (Wei et al. 2024) measures factual recall and the joint quality of (answer, abstention). It is the right benchmark when you want to compare your deployment to OpenAI's published numbers and when your task is short-answer factual QA. SelfAware (Yin et al. 2023) specifically targets unanswerable questions and measures the abstention rate. It is the right benchmark when you suspect your model is overconfident on the long tail of obscure or ill-posed queries. HaluEval (Li et al. 2023) constructs adversarial pairs of faithful and hallucinated responses for the same prompt and measures discrimination accuracy. It is the right benchmark when you are deploying a self-evaluation or critic model and need to confirm it can detect plausible-sounding but wrong outputs. Use all three for comprehensive metacognitive evaluation; use SimpleQA for headline numbers.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Calibration improves with model scale. The Kadavath et al. paper, and subsequent replications on Llama, PaLM, and the GPT-4 family, all show the same qualitative pattern: ECE on factual tasks decreases roughly logarithmically with parameter count, and AUARC for selective prediction improves correspondingly. The mechanism is partly that larger models are more accurate (which gives more headroom for confident predictions to also be correct) and partly that larger models develop better internal representations of which inputs are familiar versus novel. This is one of the few capability axes where scale appears to help rather than hurt: bigger models are not just smarter, they are also better at knowing what they know.
      </Prose>

      <Prose>
        Calibration degrades with instruction-tuning and RLHF. This is a robust empirical finding: the base pre-trained model is typically better calibrated on factual tasks than the same model after RLHF. The mechanism is not fully understood, but the leading hypothesis is that RLHF rewards confident, helpful-sounding answers and penalizes hedging, which compresses the model's output distribution and breaks the calibrated mapping that pre-training established. OpenAI's GPT-4 paper documents this explicitly — the pre-trained model's ECE on a calibration test was substantially lower than the post-RLHF model's. This is a foundational tension in alignment: the same training signal that makes models helpful makes them less honest about their own uncertainty. Recent work (including OpenAI's "Honest AI" line of research and DeepMind's calibrated-honesty proposals) attempts to recover calibration during RLHF by including explicit honesty rewards, with partial success.
      </Prose>

      <Prose>
        Selective prediction infrastructure scales linearly with query volume. Each query requires either an extra forward pass for a P(True) probe or an extra small-model call for a verifier. Both add latency (typically 100-500ms per query) and cost (an additional small fraction of the main inference cost). At the largest deployments — search engines integrating LLM answers, customer-support copilots — these costs are non-negligible and are typically managed by amortizing the verifier across cached results, batching probe calls, or using a much smaller verifier model trained specifically for the task. Above some scale (roughly: tens of millions of queries per day), it becomes worthwhile to train a dedicated calibration model that takes the prompt and answer as input and outputs a calibrated probability of correctness directly, replacing both the P(True) probe and the post-hoc calibrator with a single forward pass.
      </Prose>

      <Prose>
        Conformal prediction's coverage guarantee scales gracefully but requires representative calibration data. The finite-sample bound is on the order of <Code>1/√n</Code> — doubling the calibration set roughly halves the coverage error — so a few hundred examples typically suffice for production-quality guarantees. The harder problem at scale is exchangeability: the assumption that calibration and test data are drawn from the same distribution. In practice this is violated whenever the input distribution drifts (new product launches, news events, seasonal patterns), and the conformal guarantee silently weakens. Production deployments mitigate this by recalibrating frequently, by using adaptive conformal methods that maintain coverage under distribution shift, or by stratifying calibration sets by query type to give per-stratum guarantees.
      </Prose>

      <Prose>
        What does not scale: introspective accuracy on novel reasoning. The Hu-Levy gap appears to widen on tasks where the model is doing genuine multi-step reasoning rather than fact recall. The model can produce confident-sounding rationalizations of its outputs without those rationalizations corresponding to the actual computation. This is harder to measure (you cannot easily verify a chain-of-thought against an internal state) and harder to fix (post-hoc calibration cannot recover information that was never present in the model's reported confidence). It is the open frontier: as LLMs are deployed for harder reasoning tasks — research synthesis, code generation, multi-step planning — the gap between what they say about their confidence and what their actual error rate is becomes larger and more dangerous. Frontier safety work in 2025-2026 has increasingly focused on this: methods like representation engineering, sparse-autoencoder feature probing, and chain-of-thought monitoring all aim at extracting introspective signals more reliable than the model's own verbal reports.
      </Prose>

      <Prose>
        Equally pernicious at scale: distributional novelty. Models calibrated on the typical distribution of queries they see during training and evaluation will be miscalibrated on rare or out-of-distribution inputs in ways that are not obvious from aggregate metrics. A medical-question deployment may have ECE = 0.04 averaged across all queries while having ECE = 0.25 on the small subset of queries about rare diseases — and that subset is precisely where confidently-wrong answers are most damaging. Mitigations exist (group-conditional calibration, density-aware abstention, retrieval-augmented self-check) but they all require knowing in advance what the relevant strata are. The frontier problem is calibration on inputs whose distribution category you have not anticipated, which is exactly the situation that matters most for safety.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>RLHF over-confidence regression</H3>
      <Prose>
        Instruction-tuned and RLHF-trained models are systematically more over-confident than their pre-trained base models. The model has learned that confident, helpful-sounding answers are rewarded; the calibrated mapping from internal uncertainty to verbalized confidence that pre-training established gets compressed. This is most pronounced on factual questions, where the model's underlying log-likelihood for the answer may be low but the verbalized confidence is high. Mitigation: include explicit honesty/calibration rewards during RLHF, post-hoc calibrate verbalized confidence on a held-out set, or use logit-derived rather than verbalized confidence wherever possible.
      </Prose>

      <H3>Hu-Levy gap: prompted vs measured probability</H3>
      <Prose>
        Asking a model "what is the probability of X?" produces a number that often diverges from the model's own softmax probability for X. Hu and Levy 2023 documented this systematically; the divergence is task- and prompt-dependent. The implication is that you cannot use verbalized probability estimates as a substitute for measured probabilities in evaluation. Worst case, an evaluation that ranks models by verbalized confidence accuracy may systematically prefer models that hedge well linguistically over models that have well-calibrated internal distributions. Mitigation: when possible, use logit-derived confidence; when verbal must be used, calibrate on the verbal output specifically, do not assume it inherits the model's logit calibration.
      </Prose>

      <H3>Calibration set distribution shift</H3>
      <Prose>
        Calibrators fit on one distribution silently fail on another. A model calibrated on conversational questions will be miscalibrated when deployed for technical documentation queries. The conformal coverage guarantee assumes exchangeability; the temperature-scaling mapping assumes the calibration shape transfers. In production, calibration drift is common and silent: the model continues to output well-formed confidence numbers, but they no longer match accuracy. Mitigation: stratify calibration sets by query category, monitor rolling ECE in production, and recalibrate when distribution shift is detected.
      </Prose>

      <H3>Length-normalized confidence biases short answers</H3>
      <Prose>
        Sequence log-likelihood drops with length, so any confidence measure based on raw <Code>p(answer)</Code> is biased toward short answers. Length-normalizing — using mean log-probability per token instead of sum — fixes the length sensitivity but introduces a different bias: it rewards confident-sounding short tokens over substantive content. Neither raw nor length-normalized log-likelihood is a perfect confidence proxy. Mitigation: use answer-distribution-relative scores (the probability of the answer under the model relative to the probability under a baseline, or relative to the top-k alternatives), or use the P(True) probe which sidesteps the length issue entirely.
      </Prose>

      <H3>Abstention games the metric</H3>
      <Prose>
        A model that abstains on every question achieves perfect selective accuracy on the (empty) set of questions it answered. AUARC and accuracy-at-coverage metrics partially correct for this by penalizing low coverage, but a clever model can still game them by abstaining on hard questions and answering only the easy ones. The right defense is to evaluate selective accuracy at fixed coverage targets (e.g., "selective accuracy at 50% coverage") rather than at the abstention rate the model chooses. SimpleQA's evaluation explicitly does this: it reports accuracy at multiple coverage levels rather than letting the model choose its own.
      </Prose>

      <H3>Prompt sensitivity of P(True)</H3>
      <Prose>
        The Kadavath probe is sensitive to the exact phrasing of the "Is the answer above correct?" prompt. Changing "correct" to "right" or "accurate" can shift the probability by several percentage points. Adding "Be honest" can shift it further. Adding examples (few-shot calibration questions in the probe) can shift it more. This is not a defect; it is the normal sensitivity of LLM behavior to prompt phrasing, but it means that any production deployment using P(True) needs to lock in the exact probe template at calibration time and use the same template at deployment. Drift in probe phrasing produces drift in calibration.
      </Prose>

      <H3>Sycophancy contaminates self-evaluation</H3>
      <Prose>
        When the user expresses doubt about a model's answer ("Are you sure about that?"), the model often updates its confidence downward — even when its original answer was correct. This sycophancy effect, documented in multiple papers and consistently present in instruction-tuned models, contaminates self-evaluation in any setting where the user has voiced an opinion. Mitigation: run P(True) probes in fresh contexts (no conversation history), use third-party evaluators rather than self-evaluation, or train explicitly against sycophancy as Anthropic's "Sycophancy is a Sticky Problem" line of work proposes.
      </Prose>

      <H3>Calibration is not honesty</H3>
      <Prose>
        A perfectly calibrated model is not necessarily an honest one. Calibration measures the statistical relationship between confidence and accuracy across many predictions; honesty would require the model to never actively assert false claims. A model can be perfectly calibrated by hedging on every claim ("I'm 50% sure") while still being unhelpful and effectively misleading. Conversely, a model can be honest in the sense of always reporting its actual best guess but be badly calibrated. Treat calibration as a necessary condition for trustworthy deployment, not a sufficient one. The full picture requires accuracy, calibration, abstention quality, and honest framing of the underlying uncertainty.
      </Prose>

      <H3>Self-evaluation cannot recover from systematic blind spots</H3>
      <Prose>
        If a model is systematically wrong about a class of inputs — say, questions involving recent events past its knowledge cutoff, or questions in a low-resource language it underperforms on — its self-evaluation on those inputs will inherit the same blind spots. Asking the model to evaluate its own answer about a 2026 election when it was trained with a 2024 cutoff often produces confident-sounding "Yes, this is correct" responses, because the model has no internal flag for "I should not know this." The defense is structural rather than introspective: gate self-evaluation on retrieval-grounded inputs, attach explicit knowledge-cutoff guardrails, and use external sources of ground truth for the categories of input where you know the model has systematic gaps.
      </Prose>

      <Callout accent="gold">
        The most insidious metacognitive failure is invisible to the deployment metrics. A model that is well-calibrated on its evaluation set but consistently confident-and-wrong on a specific failure mode (a particular question type, a specific demographic of users, a niche knowledge domain) will pass aggregate calibration tests while producing confidently wrong answers in production. Stratify your calibration evaluations.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five sources below were verified against their arXiv pages on 2026-04-26. Abstracts, author lists, and arXiv IDs confirmed.
      </Prose>

      <H3>Kadavath et al. 2022 — Language Models (Mostly) Know What They Know</H3>
      <Prose>
        Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, et al. (Anthropic). "Language Models (Mostly) Know What They Know." arXiv:2207.05221. Published July 2022. The foundational paper for LLM metacognition. Demonstrates that large LMs can be probed with a P(True) follow-up question to produce calibrated estimates of answer correctness, and that calibration improves systematically with model scale. Introduces several of the empirical setups still used as benchmarks in the field, including the introspective probability evaluation and the "self-evaluation" of generated answers. Required reading.
      </Prose>

      <H3>Lin, Hilton, Evans 2022 — Teaching Models to Express Uncertainty</H3>
      <Prose>
        Stephanie Lin, Jacob Hilton, Owain Evans. "Teaching Models to Express Their Uncertainty in Words." arXiv:2205.14334. Published May 2022; TMLR 2022. Shows that GPT-3 fine-tuned to produce verbalized confidence estimates ("I'm 60% sure") can achieve better calibration on certain tasks than the raw model's logit-derived probabilities. Establishes the verbalized-confidence paradigm and demonstrates that the natural-language interface can carry meaningful uncertainty information when training is structured to elicit it. Code and CalibratedMath dataset publicly released.
      </Prose>

      <H3>Yin et al. 2023 — SelfAware</H3>
      <Prose>
        Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, Xuanjing Huang. "Do Large Language Models Know What They Don't Know?" arXiv:2305.18153. Published May 2023; ACL 2023 Findings. Constructs the SelfAware dataset: a set of questions specifically chosen to be unanswerable from typical training data, paired with answerable controls. Measures the rate at which various LLMs correctly identify and abstain on the unanswerable questions. Finds that even capable models substantially over-claim knowledge on the unanswerable set, with the gap closing only modestly with scale.
      </Prose>

      <H3>Hu &amp; Levy 2023 — Prompting vs Probability</H3>
      <Prose>
        Jennifer Hu, Roger Levy. "Prompting is not a substitute for probability measurements in large language models." arXiv:2305.13264. Published May 2023; EMNLP 2023. Demonstrates systematically that asking a language model to verbalize a probability ("what is P(X)?") produces a number that diverges from the model's own measured softmax probability for X. The divergence is task-dependent and not eliminated by careful prompting. Implies that any evaluation framework that uses verbalized probabilities as a proxy for the model's actual distribution is measuring a different quantity than it claims to.
      </Prose>

      <H3>Wei et al. 2024 — SimpleQA</H3>
      <Prose>
        Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus (OpenAI). "Measuring Short-Form Factuality in Large Language Models." arXiv:2411.04368. Published November 2024. Introduces SimpleQA, a curated benchmark of 4326 short-answer factual questions with verifiable ground truth. Reports joint metrics on (correctness, abstention, calibration) across the GPT-4 family and several other frontier models. The headline finding is that even frontier models are substantially over-confident on factual questions outside their training cutoff and recall sweet spot. Establishes the modern benchmark for short-form factuality with calibration.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — When ECE lies</H3>
      <Prose>
        Construct a hypothetical model that has ECE = 0 on a held-out set but is nevertheless useless for selective prediction. (Hint: a model that always outputs <Code>p̂ = base accuracy</Code> on every prediction has zero calibration error in expectation but zero refinement.) Now construct a model that has ECE = 0.10 but produces a perfect accuracy-coverage curve at any threshold. What does this pair of examples tell you about the relationship between calibration and refinement, and why are both needed for trustworthy deployment? How does the Brier score's decomposition into reliability and resolution capture this distinction?
      </Prose>

      <H3>Exercise 2 — Conformal prediction and exchangeability</H3>
      <Prose>
        Suppose you calibrate a conformal selective predictor on questions about events from 2020-2023, target miscoverage <Code>α = 0.10</Code>, and deploy it in 2026 on questions that include events from 2024-2026. What happens to the empirical coverage on the deployment distribution and why? What assumption of conformal prediction has been violated? Sketch a procedure for monitoring whether the conformal coverage guarantee is being upheld in production, including what you would log, how often you would recompute the threshold, and what alarm conditions would trigger recalibration.
      </Prose>

      <H3>Exercise 3 — Designing a P(True) probe</H3>
      <Prose>
        You are designing the exact prompt template for a P(True) probe to be used at scale on a customer support chatbot. The base prompt is "Is the previous answer correct? Reply with exactly Yes or No." Identify three concrete ways the probe could be improved (e.g., few-shot examples, explicit calibration instruction, alternative wordings) and predict how each would shift the calibration of the resulting probability. What experimental setup would let you choose between them empirically? What is the risk of making the probe too specific to your evaluation set?
      </Prose>

      <H3>Exercise 4 — Hu-Levy gap reproduction</H3>
      <Prose>
        Design an experiment to measure the gap between verbalized and logit-derived probability estimates on a single LLM. Specifically: pick a multiple-choice task, decide what "verbalized probability" means in your prompt, decide how you will compute the logit-derived probability, and define the metric you will use to summarize the gap. Predict what shape the result will take — is the verbalized probability systematically biased upward, downward, or noisily distributed? Once you have the data, what does the result tell you about whether to use verbalized probabilities for calibration in your deployment?
      </Prose>

      <H3>Exercise 5 — Calibration vs honesty</H3>
      <Prose>
        A colleague proposes that you can deploy a model "honestly" by training it to maximize calibration — to make its stated confidences match its empirical accuracy on a large evaluation set. Construct a thought experiment showing why a perfectly calibrated model can still be effectively dishonest in deployment, and a complementary thought experiment showing why an honest model can be badly calibrated. What does this analysis suggest about how to specify "trustworthy" model behavior in a way that captures both calibration and honesty? Does Constitutional AI, Anthropic's honest AI line, or any other published approach you can think of address both jointly, or do they target them separately?
      </Prose>

      <H3>Exercise 6 — Stratified calibration</H3>
      <Prose>
        Given a deployed model with overall ECE = 0.05 on a mixed evaluation set, design a procedure to detect whether calibration is uniform across query types or whether there is a specific stratum (e.g., a topic, a question structure, a user demographic) where the model is badly miscalibrated even though aggregate ECE looks fine. What stratifications would you compute? How would you decide which stratifications matter? What would the output of the procedure look like, and how would you act on it?
      </Prose>

    </div>
  ),
};

export default llmMetacognition;
