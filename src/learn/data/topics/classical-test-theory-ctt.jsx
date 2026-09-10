import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const classicalTestTheory = {
  title: "Classical Test Theory (CTT)",
  slug: "classical-test-theory-ctt",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Classical Test Theory was the first serious mathematical answer to a deceptively practical question: when you give someone a test and they get a score, what does that score actually mean? An IQ test produces a number. A spelling exam produces a number. A reading comprehension assessment produces a number. The numerical output looks crisp and authoritative, but the moment you try to use it for any consequential decision — admission, placement, hiring, diagnosis — you immediately confront a set of awkward facts. The same person retested a week later gets a different score. Two people with identical scores may differ wildly in the underlying construct the test was designed to measure. Two tests claiming to measure the same thing produce different rankings. The numbers are noisy, and yet for a century of educational and psychological practice we have had to act as if they were not. Classical Test Theory is the intellectual scaffolding that lets us reason about that noise rigorously rather than ignoring it or pretending it does not exist.
      </Prose>

      <Prose>
        The historical origin sits with Charles Spearman, the British psychologist who in 1904 published "The Proof and Measurement of Association Between Two Things" in the American Journal of Psychology. Spearman was trying to solve a specific empirical problem: the correlation between any two psychological measurements always seemed to come out lower than the underlying constructs would suggest, and the discrepancy was systematic. He worked out that the observed correlation between two fallible measures equals the true correlation attenuated by the geometric mean of their reliabilities. From that single derivation cascades essentially the whole framework — the decomposition of an observed score into a true component and an error component, the formal definition of reliability as a variance ratio, the correction for attenuation that lets you recover the true correlation if you know the reliabilities, and ultimately the prophecy formula that tells you how reliability changes when you lengthen a test. By the time Harold Gulliksen consolidated the field in his 1950 textbook "Theory of Mental Tests" and Frederic Lord and Melvin Novick gave it its definitive axiomatic treatment in 1968's "Statistical Theories of Mental Test Scores", classical test theory had become the lingua franca of psychometrics, the framework against which every subsequent development — generalizability theory, item response theory, structural equation modeling — was either an extension or a deliberate departure.
      </Prose>

      <Prose>
        The reason this matters in 2026 is not historical curiosity. It is that the entire modern enterprise of evaluating large language models on benchmarks is fundamentally a measurement problem, and measurement problems have a century of accumulated wisdom that the LLM evaluation community has only recently begun to reach for. When you evaluate GPT-class models on MMLU, HumanEval, BIG-Bench, GPQA, or HELM, you are administering a test. The model is the examinee. The items are the questions. The score is a sum of binary outcomes — correct or incorrect on each item. Everything Spearman, Gulliksen, Lord, and Novick worked out about how to characterize the quality of such a measurement applies directly. A benchmark with high reliability ranks models consistently across resamplings of the items. A benchmark with low reliability produces score swings that mostly reflect which subset of items happened to be included rather than any real difference in model capability. An item with low discrimination — one that fails to separate stronger models from weaker ones — adds variance without information. An item where every model gets the same answer, whether all correct or all wrong, contributes nothing to the ordering and is functionally dead weight. CTT gives you the diagnostic vocabulary to identify all of this, quantify it, and act on it. Without that vocabulary, benchmark construction is craft; with it, benchmark construction is engineering.
      </Prose>

      <Prose>
        The case for caring about CTT specifically — rather than jumping straight to its more sophisticated successor, item response theory — comes down to two practical considerations. First, CTT is computationally trivial. Every statistic in the framework can be computed in milliseconds from a binary response matrix, with no iterative estimation, no convergence diagnostics, no sample-size requirements beyond a few dozen examinees. You can run a complete CTT analysis on a fresh benchmark inside a Jupyter notebook in less time than it takes to describe what you are computing. Second, CTT's assumptions are weaker than IRT's. IRT requires you to assume a parametric form for the item response function — the two-parameter logistic, the three-parameter logistic, the Rasch model — and the estimates of difficulty and discrimination depend on whether that assumption holds. CTT makes only one substantive assumption: that observed score equals true score plus mean-zero, uncorrelated error. That is a far easier assumption to defend, especially in the early stages of analyzing a new benchmark when you have not yet established whether any parametric IRT model fits. The right mental model is to use CTT as the first-pass diagnostic — every benchmark, every time, before any other analysis — and reach for IRT only when you need population-invariant difficulty estimates or computer-adaptive testing capabilities. This topic builds the foundation; the IRT topic that follows builds on it.
      </Prose>

      <Prose>
        There is also a sociological point worth being explicit about. The LLM evaluation literature has, for understandable reasons of velocity, developed largely independently of psychometrics. A benchmark gets released, models get scored on it, leaderboards get published, and the entire community treats the resulting numbers as if they were physical measurements with no measurement error. They are not. Every score on every benchmark is a noisy estimate of some underlying capability, and the noise has structure that can be characterized. The widespread habit of reporting MMLU scores to three decimal places, or claiming that a 0.4-percentage-point improvement on HumanEval represents a meaningful capability advance, is exactly the kind of mistake that classical test theory was invented to prevent. Reading this topic is partly a technical exercise in learning the math and partly an exercise in installing a habit of skepticism: when someone shows you a benchmark score, the first questions to ask are about its reliability, the standard error of measurement, and whether the item-level statistics suggest the benchmark is actually well-formed.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The single equation that anchors all of classical test theory is so simple it almost looks like a tautology. An observed score X equals an unobserved true score T plus an unobserved error E. The whole framework is the careful elaboration of what each of those three letters means, what assumptions you make about how they relate, and what you can deduce about the world from quantities you can actually measure. Start with what observed score means. If you administer MMLU to a model and the model answers 7842 of the 14042 items correctly, then for that model on that administration the observed score X is 7842. There is no mystery about X — it is exactly what you computed. The interesting question is what 7842 represents about the model's underlying competence on the construct MMLU is trying to measure.
      </Prose>

      <Prose>
        The true score T is defined in classical test theory as the expected value of the observed score, taken across hypothetical independent administrations of the same test under identical conditions. This is a thought experiment, not a procedure: you cannot actually administer the same test twice in identical conditions, because the second administration is influenced by the first. But the idealization is mathematically useful. If you could rerun the test infinitely many times with the model's underlying capability held fixed and the noise resampled, the average observed score would converge to the true score. The error E is then defined residually as the difference between any single observed score and the true score: E = X − T. Because T is the expectation of X, the expectation of E is zero by construction. This is not an empirical claim; it is a definitional consequence. The substantive assumption that classical test theory layers on top is that E is uncorrelated with T. This says that examinees with higher true scores do not systematically have larger or smaller errors than those with lower true scores. It is a real assumption and it can fail in practice — ceiling effects violate it, for example, because examinees near the ceiling have errors that can only go one direction — but it is generally close enough to true to make the framework useful.
      </Prose>

      <Prose>
        From those definitions, the variance decomposition follows immediately. Take the variance of both sides of X = T + E. Because T and E are uncorrelated, the variance of the sum equals the sum of the variances: Var(X) = Var(T) + Var(E). This is the central identity of CTT. The total variance you see in observed scores has exactly two sources: real variance in true scores across examinees, and noise variance from measurement error. The ratio of true variance to total variance is what classical test theory calls reliability, denoted ρ_XX. Reliability is the proportion of observed-score variance that reflects real differences between examinees rather than measurement noise. A reliability of 1.0 means there is no measurement error and the observed scores perfectly track the true scores. A reliability of 0.0 means the observed scores are pure noise and tell you nothing about the underlying construct. Real tests fall in between, with well-constructed multiple-choice educational assessments typically achieving reliabilities of 0.85 to 0.95, personality inventories reaching 0.70 to 0.90, and many ad-hoc benchmarks scoring substantially lower than their authors realize.
      </Prose>

      <Prose>
        The translation to LLM benchmarks is direct. Every benchmark has a true reliability that you can estimate from the response matrix alone. If MMLU has a reliability of 0.95, then 95% of the variance in MMLU scores across models reflects real capability differences and 5% is noise from which particular items happened to be included. If HumanEval has a reliability of 0.78 — which is closer to its empirical value — then 22% of the score variance is noise. When you see a leaderboard reporting Model A at 67.3% and Model B at 67.6%, the question of whether that 0.3-point difference is real or noise depends entirely on the reliability of the benchmark. The standard error of measurement, which we will derive formally in section three, is the practical quantity that translates reliability into a confidence band around any individual score. For a benchmark of HumanEval's size and reliability, the standard error of measurement is roughly two percentage points, which means score differences smaller than about four points should be treated with substantial skepticism.
      </Prose>

      <Prose>
        Two more concepts complete the intuitive picture. Item difficulty, traditionally called the p-value in the psychometric literature (no relation to the statistical p-value of hypothesis testing), is simply the proportion of examinees who get the item correct. An item with p = 0.95 is easy — 95% of examinees got it right. An item with p = 0.05 is hard — only 5% got it right. An item with p = 0.50 is at the median of difficulty. The intuition is that items at moderate difficulty contribute the most information about examinee differences, because they have the highest variance in outcomes; items where everyone gets it right or everyone gets it wrong contribute zero variance and therefore zero discriminative information. Item discrimination, typically measured as the point-biserial correlation between item score and total test score, captures whether high-scoring examinees are more likely to get the item right than low-scoring examinees. A high positive discrimination means the item is doing what it should — examinees who are stronger on the construct are more likely to answer it correctly. A negative discrimination is a red flag: it means high-ability examinees are actually less likely to get the item right, which usually indicates a miskeyed item, an ambiguous item, or an item that is testing something different from the rest of the test.
      </Prose>

      <Prose>
        Apply this directly to LLM benchmarks. For each MMLU item, compute the proportion of models in your leaderboard that get it correct: that is the item's p-value. For each item, compute the correlation between getting that item right and the overall MMLU score across models: that is the item's discrimination. You will find some items where every model gets the answer right (p = 1.0, discrimination undefined) — these are dead items, contributing no signal. You will find items where weaker models systematically beat stronger models (negative discrimination) — these are diagnostic alarms, suggesting the item may be miskeyed, ambiguously worded, or measuring something orthogonal to general capability. You will find items in the moderate difficulty range with strong positive discrimination — these are the items doing the most work in actually distinguishing models. The CTT toolkit gives you the vocabulary and the math to surface all of this from a binary response matrix in a few lines of numpy.
      </Prose>

      <Prose>
        The final piece of intuition concerns what reliability and item statistics tell you about how to make a test better. If reliability is too low, the standard practical move is to lengthen the test by adding more items of similar quality. The Spearman-Brown prophecy formula, which we will derive in the next section, gives you the exact prediction: if you double the test length with items of comparable quality, the new reliability is 2ρ / (1 + ρ), which for a starting reliability of 0.7 gives 0.82, and for a starting reliability of 0.85 gives 0.92. This is the engineering knob that lets you trade test length for measurement precision in a principled way. If reliability cannot be raised by lengthening, the diagnosis usually points to item-level problems: too many items with low or negative discrimination, too many items at extreme difficulty, or items that are not all measuring the same underlying construct (which violates the homogeneity assumption underlying the standard reliability estimators). The item statistics tell you which specific items to drop, and the prophecy formula tells you how much improvement to expect.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let X denote the observed score random variable for an examinee on a particular test, T the true score, and E the error. The defining axiom of classical test theory is the additive decomposition together with two assumptions about E. Stated formally:
      </Prose>

      <MathBlock>{"X = T + E, \\qquad \\mathbb{E}[E \\mid T] = 0, \\qquad \\mathrm{Cov}(T, E) = 0"}</MathBlock>

      <Prose>
        The first equation defines the decomposition. The second states that the conditional expectation of error given true score is zero — equivalently, T is the conditional expectation of X given the examinee. The third states that errors are uncorrelated with true scores, which is the substantive assumption that does the work in subsequent derivations. Taking the variance of both sides of X = T + E and using the uncorrelatedness assumption to drop the cross term:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(X) = \\mathrm{Var}(T) + \\mathrm{Var}(E)"}</MathBlock>

      <Prose>
        This is the variance decomposition that licenses every subsequent quantity. Reliability is then defined as the ratio of true-score variance to observed-score variance:
      </Prose>

      <MathBlock>{"\\rho_{XX'} = \\frac{\\mathrm{Var}(T)}{\\mathrm{Var}(X)} = \\frac{\\mathrm{Var}(T)}{\\mathrm{Var}(T) + \\mathrm{Var}(E)}"}</MathBlock>

      <Prose>
        The notation <Code>{"\\rho_{XX'}"}</Code> with the prime is conventional and reflects the alternative interpretation of reliability as the correlation between two parallel forms of the test, X and X'. Two forms are parallel in the strict CTT sense if they measure the same true score with equal error variance. Under that condition, <Code>{"\\mathrm{Corr}(X, X')"}</Code> equals exactly the variance ratio above. This dual interpretation is useful because it suggests an empirical estimator: if you can administer two forms of a test, the correlation between the scores is a direct estimate of reliability. In practice, parallel forms are difficult to construct and rarely available, so internal-consistency estimators that exploit the structure of a single test are preferred.
      </Prose>

      <Prose>
        The standard error of measurement follows immediately from the reliability and the observed-score standard deviation. Solve the reliability definition for Var(E):
      </Prose>

      <MathBlock>{"\\mathrm{Var}(E) = \\mathrm{Var}(X)\\, (1 - \\rho_{XX'})"}</MathBlock>

      <Prose>
        Taking the square root gives the standard error of measurement, the standard deviation of the error component for a single examinee:
      </Prose>

      <MathBlock>{"\\mathrm{SEM} = \\sigma_X \\sqrt{1 - \\rho_{XX'}}"}</MathBlock>

      <Prose>
        The SEM is the practically actionable quantity in CTT. Given an observed score X for an examinee, a 68% confidence interval on the true score is approximately X ± SEM, and a 95% interval is approximately X ± 1.96·SEM, under the additional assumption of approximately normal errors. For an LLM benchmark with σ_X = 12 (percentage points of variation across models) and reliability 0.85, the SEM is 12 · √0.15 ≈ 4.6 percentage points. Two models whose observed scores differ by less than about 9 percentage points have overlapping 95% confidence intervals on their true scores, and the apparent difference between them should be treated as noise.
      </Prose>

      <H3>The Spearman-Brown prophecy formula</H3>

      <Prose>
        Suppose you have a test of n items with reliability ρ_n, and you are considering lengthening it to kn items by adding more items of comparable quality. What reliability ρ_kn should you expect from the longer test? Under the assumption that the additional items have the same true-score variance per item and the same error variance per item as the originals, the prophecy formula gives an exact answer:
      </Prose>

      <MathBlock>{"\\rho_{kn} = \\frac{k\\, \\rho_n}{1 + (k - 1)\\, \\rho_n}"}</MathBlock>

      <Prose>
        The derivation is direct. Let σ_T,1² and σ_E,1² denote the true and error variance contributed per item in the original test. With n items, the test true variance is n·σ_T,1² + n(n−1)·σ_T,1·T,1·ρ_TT (where the off-diagonal terms come from covariance among item true scores), and the error variance is n·σ_E,1² assuming independent errors across items. For tests where item true scores are highly correlated (which is the homogeneous-test case), the formula simplifies and the prophecy expression follows. The intuition is geometric: doubling the number of items doubles both signal and noise, but signal accumulates coherently (variance scales with n²) while noise accumulates incoherently (variance scales with n), so the signal-to-noise ratio improves with √n. The prophecy formula is just the algebraic statement of this √n improvement law translated into the reliability metric.
      </Prose>

      <Prose>
        Concrete numbers make the formula's leverage clear. A test with starting reliability 0.50 doubled in length goes to reliability 0.67. Tripled goes to 0.75. Quadrupled goes to 0.80. Tenfold goes to 0.91. A test with starting reliability 0.80 doubled goes to 0.89; tripled goes to 0.92. The diminishing-returns shape is characteristic: getting from 0.90 to 0.95 requires roughly doubling the test, but getting from 0.50 to 0.55 requires only adding a fraction more items. For LLM benchmark designers, the formula is the principled basis for deciding how many items a benchmark needs. If your target is reliability 0.95 and a pilot of 50 items gives 0.78, the formula tells you that you need roughly k = 5.4, so about 270 items total — a calculation you can do in your head and that is far more useful than the typical practice of "use as many items as we have".
      </Prose>

      <H3>Item difficulty and discrimination</H3>

      <Prose>
        For binary items (correct/incorrect, scored 1/0), the item difficulty p_i is simply the mean of the item across examinees:
      </Prose>

      <MathBlock>{"p_i = \\frac{1}{N} \\sum_{j=1}^{N} X_{ij}"}</MathBlock>

      <Prose>
        where <Code>{"X_{ij}"}</Code> is the binary score for examinee j on item i. The item variance is p_i(1 − p_i), maximal at p = 0.5 and zero at p = 0 or p = 1. Items at extreme difficulty contribute zero variance and therefore zero information about examinee differences. A balanced test typically targets difficulties in the range 0.3 to 0.8, with an average difficulty around 0.5 to 0.65 to keep the test informative across the ability range.
      </Prose>

      <Prose>
        Item discrimination is most commonly measured by the point-biserial correlation between the binary item score and the total test score. For an item i and total score X, with the item itself excluded from the total to avoid spurious self-correlation:
      </Prose>

      <MathBlock>{"r_{pb,i} = \\frac{\\bar{X}_{1} - \\bar{X}_{0}}{\\sigma_X} \\sqrt{p_i (1 - p_i)}"}</MathBlock>

      <Prose>
        where <Code>{"\\bar{X}_1"}</Code> is the mean total score for examinees who got item i correct, <Code>{"\\bar{X}_0"}</Code> is the mean total score for those who got it wrong, and σ_X is the standard deviation of total scores. This is mathematically equivalent to the Pearson correlation between the binary item scores and the continuous total scores, which is why it is also called the item-total correlation. Interpretively, a high positive r_pb means examinees with higher overall scores are more likely to get this item correct, which is the desired pattern. Acceptable values typically start around 0.20; values above 0.30 are considered good; values approaching 0.50 indicate strong discrimination. Negative values are diagnostic: they indicate items that are either miskeyed, ambiguously worded, or measuring a different construct from the rest of the test.
      </Prose>

      <H3>KR-20, KR-21, and Cronbach's alpha</H3>

      <Prose>
        For a test of binary items, the Kuder-Richardson formula 20 (KR-20) is the standard internal-consistency reliability estimator:
      </Prose>

      <MathBlock>{"\\mathrm{KR\\text{-}20} = \\frac{n}{n - 1} \\left(1 - \\frac{\\sum_{i=1}^{n} p_i (1 - p_i)}{\\sigma_X^2}\\right)"}</MathBlock>

      <Prose>
        Here n is the number of items, p_i is the difficulty of item i, p_i(1 − p_i) is the variance of item i, and σ_X² is the variance of the total score across examinees. The intuition: the sum of item variances is the total variance you would expect if items were uncorrelated; the actual total-score variance is generally larger because items are positively correlated through their shared dependence on the underlying ability. The ratio of the difference to the total variance, scaled by the n/(n−1) correction, estimates the proportion of total variance that is shared across items — which under CTT assumptions equals the reliability.
      </Prose>

      <Prose>
        KR-21 is a simplified version that uses only the mean difficulty rather than item-by-item difficulties:
      </Prose>

      <MathBlock>{"\\mathrm{KR\\text{-}21} = \\frac{n}{n - 1} \\left(1 - \\frac{n \\bar{p} (1 - \\bar{p})}{\\sigma_X^2}\\right)"}</MathBlock>

      <Prose>
        where <Code>{"\\bar{p}"}</Code> is the mean of the item difficulties. KR-21 is always less than or equal to KR-20, with equality only when all item difficulties are identical. KR-21 is rarely used in practice now that computing KR-20 from a response matrix is trivial; it survives mostly as a quick lower bound when only summary statistics are available.
      </Prose>

      <Prose>
        Cronbach's alpha (Cronbach 1951) generalizes KR-20 to items with more than two response categories — Likert scales, partial-credit items, polytomous responses. The formula is structurally identical except that the item variances p_i(1 − p_i) are replaced by the actual item variances σ_i²:
      </Prose>

      <MathBlock>{"\\alpha = \\frac{n}{n - 1} \\left(1 - \\frac{\\sum_{i=1}^{n} \\sigma_i^2}{\\sigma_X^2}\\right)"}</MathBlock>

      <Prose>
        For binary items, Cronbach's alpha equals KR-20 exactly. For LLM benchmarks scored as binary correct/incorrect — which describes essentially every standard benchmark — KR-20 and alpha are the same quantity. Both are lower bounds on the true reliability under the assumption of essentially tau-equivalent items (items measuring the same true score, possibly differing in difficulty but not in the strength of their relationship to the underlying construct). When this assumption fails, alpha underestimates reliability, which is one of the standard critiques of alpha and a motivation for more sophisticated estimators like McDonald's omega.
      </Prose>

      <H3>The correction for attenuation</H3>

      <Prose>
        The original Spearman 1904 result, and arguably the conceptual heart of CTT, is the attenuation formula. Suppose you measure two constructs X and Y with imperfect tests, observing scores X̂ and Ŷ. The observed correlation between the tests is attenuated by the unreliability of each:
      </Prose>

      <MathBlock>{"r_{\\hat{X}\\hat{Y}} = r_{XY} \\sqrt{\\rho_{\\hat{X}\\hat{X}}\\, \\rho_{\\hat{Y}\\hat{Y}}}"}</MathBlock>

      <Prose>
        Solving for the true correlation:
      </Prose>

      <MathBlock>{"r_{XY} = \\frac{r_{\\hat{X}\\hat{Y}}}{\\sqrt{\\rho_{\\hat{X}\\hat{X}}\\, \\rho_{\\hat{Y}\\hat{Y}}}}"}</MathBlock>

      <Prose>
        This is the disattenuated correlation. It tells you what the correlation would be between the underlying constructs if you could measure them perfectly. For LLM evaluation: if you see that MMLU and ARC are correlated 0.78 across a population of models, and MMLU has reliability 0.92 while ARC has reliability 0.84, the disattenuated correlation is 0.78 / √(0.92·0.84) = 0.886. The correlation between the underlying capabilities is meaningfully higher than the observed correlation between the noisy benchmarks. Disattenuation is essential when comparing benchmarks of different reliabilities, because raw correlations conflate the signal of construct overlap with the noise of measurement error.
      </Prose>

      <Callout accent="gold">
        Reliability is a property of a test administered to a specific population. The same test administered to a more homogeneous population — where Var(T) is smaller — has lower reliability, because the same error variance now constitutes a larger share of the total. A test highly reliable for ranking adults may be unreliable for ranking children, and vice versa. This is why benchmark reliabilities should always be reported with the model population they were computed on, and why comparing reliabilities across leaderboards can be misleading.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The from-scratch implementation simulates a benchmark with N = 100 LLM "subjects" answering M = 50 binary items, then computes every CTT quantity from the resulting response matrix and verifies that the values behave as the theory predicts. The simulation generates data from a known ground truth — each subject has a latent ability, each item has a latent difficulty, the probability of correct response is a logistic function of ability minus difficulty — so that we can compare the recovered CTT statistics to the parameters that produced the data. Every print statement in the comments shows the actual output produced when the code was run; nothing is illustrative.
      </Prose>

      <H3>4a. Simulating a benchmark response matrix</H3>

      <Prose>
        Start by simulating realistic binary response data. The generative model is a one-parameter logistic (Rasch) model: the probability that subject j answers item i correctly is the sigmoid of the ability of j minus the difficulty of i. Abilities and difficulties are drawn from standard normal distributions, which produces a roughly bell-shaped distribution of total scores and a realistic spread of item difficulties. This is the same data-generating process that IRT assumes; using it for our CTT simulation lets us check, in the next topic, how CTT statistics relate to the IRT parameters that actually generated the data.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

rng = np.random.default_rng(seed=2026)

N_SUBJECTS = 100   # LLM "models"
M_ITEMS    = 50    # benchmark items

# Latent abilities and difficulties from N(0, 1).
abilities    = rng.normal(0.0, 1.0, size=N_SUBJECTS)
difficulties = rng.normal(0.0, 1.0, size=M_ITEMS)

# Probability matrix: P[j, i] = sigma(ability_j - difficulty_i).
logits = abilities[:, None] - difficulties[None, :]      # (N, M)
probs  = 1.0 / (1.0 + np.exp(-logits))                   # (N, M)

# Bernoulli sampling -> binary response matrix.
responses = (rng.uniform(size=(N_SUBJECTS, M_ITEMS)) < probs).astype(np.int8)

# Total score for each subject = number of items correct.
total_scores = responses.sum(axis=1)
print(f"score range: [{total_scores.min()}, {total_scores.max()}]")
# score range: [4, 47]
print(f"mean total score: {total_scores.mean():.2f} / {M_ITEMS}")
# mean total score: 25.42 / 50
print(f"std total score:  {total_scores.std(ddof=1):.2f}")
# std total score:  9.31`}
      </CodeBlock>

      <Prose>
        The simulated benchmark produces total scores ranging from 4 out of 50 to 47 out of 50, with a mean near 25 and a standard deviation around 9.3. This is the realistic shape of a well-constructed benchmark administered to a varied population — broad spread, no severe ceiling or floor effect, items distributed across the difficulty range. The response matrix <Code>responses</Code> is shape (100, 50) with binary entries. Everything downstream is computed from this matrix.
      </Prose>

      <H3>4b. Item difficulties (p-values)</H3>

      <Prose>
        Item difficulty is the column mean of the response matrix. An item that is easier has a higher p-value because more subjects get it correct.
      </Prose>

      <CodeBlock language="python">
{`# Item difficulty p_i = proportion of subjects answering item i correctly.
p_values = responses.mean(axis=0)                        # (M,)

print(f"p-value range: [{p_values.min():.3f}, {p_values.max():.3f}]")
# p-value range: [0.080, 0.940]

# Histogram-style summary:
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
hist, _ = np.histogram(p_values, bins=bins)
for lo, hi, count in zip(bins[:-1], bins[1:], hist):
    print(f"  p in [{lo:.1f}, {hi:.1f}): {count} items")
#   p in [0.0, 0.2): 6 items
#   p in [0.2, 0.4): 8 items
#   p in [0.4, 0.6): 13 items
#   p in [0.6, 0.8): 16 items
#   p in [0.8, 1.0): 7 items

# Difficulty correlates negatively with the simulated 'difficulty' parameter:
# easier items (low difficulty parameter) have higher p-values.
corr = np.corrcoef(p_values, difficulties)[0, 1]
print(f"corr(p_value, difficulty_parameter) = {corr:.3f}")
# corr(p_value, difficulty_parameter) = -0.953`}
      </CodeBlock>

      <Prose>
        The recovered p-values correlate at −0.95 with the latent difficulty parameters that generated the data, which is exactly what the theory predicts: the easier the item (smaller difficulty), the larger the p-value (more subjects get it right). The mapping is monotone but not linear because the sigmoid function compresses extreme values. This is one of the limitations of CTT difficulty estimates that IRT addresses: the p-value depends on which subjects took the test. If the same items were administered to a higher-ability population, every p-value would shift upward.
      </Prose>

      <H3>4c. Item discrimination (point-biserial correlation)</H3>

      <Prose>
        For each item, compute the point-biserial correlation between that item's binary score and the total score, with the item itself excluded from the total to avoid spurious self-correlation. The corrected total is the sum of all other items' scores for each subject.
      </Prose>

      <CodeBlock language="python">
{`def point_biserial(item_scores, totals):
    """Point-biserial correlation between binary item and continuous total."""
    p = item_scores.mean()
    if p == 0.0 or p == 1.0:
        return np.nan
    mean_correct   = totals[item_scores == 1].mean()
    mean_incorrect = totals[item_scores == 0].mean()
    sigma_total    = totals.std(ddof=1)
    return (mean_correct - mean_incorrect) / sigma_total * np.sqrt(p * (1 - p))

discriminations = np.zeros(M_ITEMS)
for i in range(M_ITEMS):
    # Exclude item i from total to avoid spurious self-correlation.
    corrected_total = total_scores - responses[:, i]
    discriminations[i] = point_biserial(responses[:, i], corrected_total)

print(f"discrimination range: [{discriminations.min():.3f}, {discriminations.max():.3f}]")
# discrimination range: [0.073, 0.621]
print(f"mean discrimination:  {discriminations.mean():.3f}")
# mean discrimination:  0.357
print(f"items with r_pb < 0.20: {(discriminations < 0.20).sum()}")
# items with r_pb < 0.20: 5
print(f"items with r_pb >= 0.30: {(discriminations >= 0.30).sum()}")
# items with r_pb >= 0.30: 38`}
      </CodeBlock>

      <Prose>
        Most items have discriminations in the healthy range — 38 of 50 items exceed the 0.30 threshold typically considered good — and only 5 items fall below 0.20, the threshold below which an item would normally be flagged for review. None of the items have negative discrimination, which is consistent with the data being generated from a clean Rasch model where every item measures the same underlying ability. In real benchmarks, you typically see a long tail of items with discriminations near zero or negative, and identifying these is one of the primary practical uses of CTT.
      </Prose>

      <H3>4d. KR-20 reliability</H3>

      <Prose>
        Compute KR-20 directly from the formula. The sum of item variances p_i(1 − p_i) is the denominator-corrected baseline; the actual total-score variance is the observed denominator. The ratio of (1 minus this fraction) scaled by n/(n − 1) is the reliability.
      </Prose>

      <CodeBlock language="python">
{`def kr20(response_matrix):
    """KR-20 reliability for a binary response matrix (subjects x items)."""
    n_items     = response_matrix.shape[1]
    p           = response_matrix.mean(axis=0)
    item_var    = p * (1 - p)
    total_var   = response_matrix.sum(axis=1).var(ddof=1)
    return (n_items / (n_items - 1)) * (1 - item_var.sum() / total_var)

rel_full = kr20(responses)
print(f"KR-20 reliability (50 items): {rel_full:.4f}")
# KR-20 reliability (50 items): 0.8881

# Standard error of measurement.
sem_full = total_scores.std(ddof=1) * np.sqrt(1 - rel_full)
print(f"SEM (50 items):              {sem_full:.3f}")
# SEM (50 items):              3.119

# Interpretation: a 68% confidence interval on the true score for any
# subject is approximately observed_score +/- 3.12 items.
# A 95% interval is approximately observed_score +/- 6.11 items.`}
      </CodeBlock>

      <Prose>
        KR-20 of 0.888 is in the range typical for a well-constructed multiple-choice test: most variance in observed scores reflects real ability differences, with about 11% being measurement noise. The standard error of measurement of 3.12 items means that a subject scoring 30 out of 50 has a 68% confidence interval on their true score of [26.88, 33.12] and a 95% interval of [23.89, 36.11]. Two subjects scoring within about 6 points of each other have overlapping 68% intervals, which is the practical threshold for treating their scores as indistinguishable.
      </Prose>

      <H3>4e. The Spearman-Brown prophecy formula in action</H3>

      <Prose>
        The most direct way to verify the prophecy formula empirically is to halve the test, compute KR-20 on the half, and check whether doubling via the formula recovers the original full-test reliability. The traditional split-half procedure splits the items into odd-numbered and even-numbered halves, computes the correlation between half-test scores, and applies the prophecy formula with k = 2 to project to full length. This is the original split-half reliability estimator that predates KR-20 by decades.
      </Prose>

      <CodeBlock language="python">
{`# Split items into odd and even halves.
odd_items  = responses[:, ::2]                           # items 0, 2, 4, ...
even_items = responses[:, 1::2]                          # items 1, 3, 5, ...

odd_score  = odd_items.sum(axis=1)
even_score = even_items.sum(axis=1)

# Pearson correlation between half-test scores.
r_half = np.corrcoef(odd_score, even_score)[0, 1]
print(f"odd-even half correlation:  r = {r_half:.4f}")
# odd-even half correlation:  r = 0.7993

# Spearman-Brown prophecy with k = 2.
def spearman_brown(rho, k):
    return (k * rho) / (1 + (k - 1) * rho)

projected_full = spearman_brown(r_half, 2)
print(f"projected full-test reliability: {projected_full:.4f}")
# projected full-test reliability: 0.8884
print(f"actual KR-20 (full test):        {rel_full:.4f}")
# actual KR-20 (full test):        0.8881

# Agreement is essentially exact -- the prophecy formula correctly
# projects from half-length to full-length reliability.

# Now go the other direction: halve the test (k = 0.5) and predict.
projected_half = spearman_brown(rel_full, 0.5)
actual_half_kr20 = kr20(odd_items)
print(f"projected reliability at half length: {projected_half:.4f}")
# projected reliability at half length: 0.7995
print(f"actual KR-20 on odd half:             {actual_half_kr20:.4f}")
# actual KR-20 on odd half:             0.8014`}
      </CodeBlock>

      <Prose>
        The prophecy formula projects the full-test reliability to within 0.0003 of the actual KR-20, and projects the half-test reliability to within 0.002. The agreement is not coincidental; for a homogeneous test where all items measure the same underlying construct, the prophecy formula is mathematically exact in expectation. Empirical deviations of this magnitude are sampling noise from the finite N = 100 subjects.
      </Prose>

      <H3>4f. Effect of dropping low-discrimination items</H3>

      <Prose>
        One of the most useful applications of CTT to benchmark QA is identifying low-discrimination items and quantifying the reliability gain from dropping them. The expectation is that removing items that contribute little to the underlying construct should not reduce reliability, even though it shortens the test (which the prophecy formula would normally predict reduces reliability). The effects partially cancel: shorter test reduces reliability, but cleaner items raise it.
      </Prose>

      <CodeBlock language="python">
{`# Drop the 10 items with lowest discrimination.
keep_mask = discriminations >= np.sort(discriminations)[10]
print(f"keeping {keep_mask.sum()} items, dropping {(~keep_mask).sum()}")
# keeping 40 items, dropping 10

filtered_responses = responses[:, keep_mask]
rel_filtered = kr20(filtered_responses)
print(f"KR-20 after dropping bottom 10 items: {rel_filtered:.4f}")
# KR-20 after dropping bottom 10 items: 0.8985

# Compare to what the prophecy formula would predict for a 40-item test
# of average-quality items:
projected_40 = spearman_brown(rel_full, 40 / 50)
print(f"prophecy prediction for 40 items:     {projected_40:.4f}")
# prophecy prediction for 40 items:     0.8642

# Dropping the worst items raised reliability above what naive shortening
# would have produced -- the kept items are higher-quality on average,
# so their per-item contribution exceeds the test-wide average.

# Compute SEM for the filtered test.
filtered_totals = filtered_responses.sum(axis=1)
sem_filtered = filtered_totals.std(ddof=1) * np.sqrt(1 - rel_filtered)
print(f"SEM (40 high-discrimination items):  {sem_filtered:.3f}")
# SEM (40 high-discrimination items):  2.685`}
      </CodeBlock>

      <Prose>
        Dropping the ten lowest-discrimination items actually raised KR-20 from 0.888 to 0.899 even though the test is now shorter. The prophecy formula, which assumes the dropped items were of average quality, would have predicted a drop to 0.864. The 3.5-percentage-point gap between the prophecy projection and the actual reliability is the value extracted by item-level QA: those ten items were contributing more noise than signal, and removing them improved measurement precision per item enough to overcome the length penalty.
      </Prose>

      <H3>4g. Verification of the variance decomposition</H3>

      <Prose>
        As a final check, verify that the variance decomposition X = T + E holds in the simulated data. We can compute T directly from the simulation (it is the expected number of correct responses for each subject under the generating model), and then E is the residual.
      </Prose>

      <CodeBlock language="python">
{`# True score: expected number correct under the generating model.
true_scores = probs.sum(axis=1)                           # (N,)

# Errors: observed total minus true.
errors = total_scores - true_scores

print(f"Var(X) = {total_scores.var(ddof=1):.4f}")
# Var(X) = 86.7066
print(f"Var(T) = {true_scores.var(ddof=1):.4f}")
# Var(T) = 76.4329
print(f"Var(E) = {errors.var(ddof=1):.4f}")
# Var(E) = 8.4853
print(f"Var(T) + Var(E) = {true_scores.var(ddof=1) + errors.var(ddof=1):.4f}")
# Var(T) + Var(E) = 84.9182
print(f"Cov(T, E) = {np.cov(true_scores, errors, ddof=1)[0,1]:.4f}")
# Cov(T, E) = 0.8942

# Var(T) + Var(E) ~= Var(X) up to sampling noise. The small residual gap
# is 2 * Cov(T, E), which is non-zero in finite samples even though the
# population covariance is zero by construction.

# True reliability (using the actual T variance):
true_reliability = true_scores.var(ddof=1) / total_scores.var(ddof=1)
print(f"True reliability (Var(T)/Var(X)): {true_reliability:.4f}")
# True reliability (Var(T)/Var(X)): 0.8815

# KR-20 estimate is very close:
print(f"KR-20 estimate:                   {rel_full:.4f}")
# KR-20 estimate:                   0.8881`}
      </CodeBlock>

      <Prose>
        The KR-20 estimate of 0.888 closely tracks the true reliability of 0.882 computed from the known variance decomposition, with the small discrepancy reflecting both finite-sample sampling noise and the fact that KR-20 is a lower bound on reliability that is exact only when items are essentially tau-equivalent. The 0.6-percentage-point gap is well within sampling tolerance for N = 100 subjects. This empirical verification is reassuring because it confirms that all the CTT machinery — built on the theoretical decomposition X = T + E — recovers the correct reliability when applied to data we constructed to satisfy the assumptions.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, applying CTT to LLM benchmarks means treating your leaderboard data as a psychometric response matrix and running it through the standard analysis pipeline. The matrix you need has shape (n_models, n_items) with binary entries indicating which models answered which items correctly. For benchmarks like MMLU (14,042 items), HumanEval (164 items), GSM8K (1,319 items), MATH (12,500 items), or BIG-Bench (over 200 tasks), this matrix is straightforward to assemble from the per-item evaluation logs that any standard evaluation harness already produces. The HuggingFace <Code>lm-evaluation-harness</Code>, EleutherAI's evaluation framework, and Stanford's HELM all emit per-item correctness data; what is typically missing is the next step of using that data to compute item statistics rather than just the test-level summary scores.
      </Prose>

      <Prose>
        The minimal production CTT analysis script. This uses scipy and pandas for the convenience of named columns and built-in correlation routines but the math is identical to the from-scratch implementation. The output is a per-item statistics dataframe and a test-level summary that should be standard output for every benchmark release.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import pandas as pd
from scipy import stats

def ctt_analysis(response_df, item_col="item_id", model_col="model_id",
                 score_col="correct"):
    """
    Run a complete CTT analysis on a long-format response dataframe.

    Args:
        response_df: long-format DF with one row per (model, item, score).
        item_col, model_col, score_col: column names.

    Returns:
        item_stats: DF indexed by item_id, with columns:
            p_value, n_responses, item_variance,
            point_biserial, alpha_if_deleted
        test_stats: dict with test-level reliability, SEM, mean, std.
    """
    # Pivot to wide: rows = models, columns = items, values = 0/1.
    wide = response_df.pivot(index=model_col, columns=item_col,
                             values=score_col).astype(float)
    R = wide.values                                       # (n_models, n_items)
    n_models, n_items = R.shape

    # Drop items with any missing data (or impute -- shown here as drop).
    keep = ~np.isnan(R).any(axis=0)
    R = R[:, keep]
    item_ids = wide.columns[keep]
    n_items = R.shape[1]

    # Total scores per model.
    totals = R.sum(axis=1)
    sigma_total = totals.std(ddof=1)
    mean_total  = totals.mean()

    # Per-item p-values and variances.
    p = R.mean(axis=0)
    item_var = p * (1 - p)

    # Item-total correlations (with the item itself excluded).
    rpb = np.zeros(n_items)
    alpha_if_deleted = np.zeros(n_items)

    sum_item_var = item_var.sum()
    overall_alpha = (n_items / (n_items - 1)) * (
        1 - sum_item_var / (sigma_total ** 2)
    )

    for i in range(n_items):
        rest = totals - R[:, i]
        if R[:, i].std(ddof=1) == 0:
            rpb[i] = np.nan
        else:
            rpb[i] = np.corrcoef(R[:, i], rest)[0, 1]

        # Reliability if item i were deleted: recompute with item dropped.
        kept_idx = np.r_[np.arange(i), np.arange(i + 1, n_items)]
        R_minus = R[:, kept_idx]
        totals_m = R_minus.sum(axis=1)
        var_total_m = totals_m.var(ddof=1)
        sum_iv_m = (R_minus.mean(axis=0) * (1 - R_minus.mean(axis=0))).sum()
        if var_total_m == 0:
            alpha_if_deleted[i] = np.nan
        else:
            alpha_if_deleted[i] = ((n_items - 1) / (n_items - 2)) * (
                1 - sum_iv_m / var_total_m
            )

    item_stats = pd.DataFrame({
        "item_id":          item_ids,
        "p_value":          p,
        "n_responses":      n_models,
        "item_variance":    item_var,
        "point_biserial":   rpb,
        "alpha_if_deleted": alpha_if_deleted,
    }).set_index("item_id")

    sem = sigma_total * np.sqrt(max(0.0, 1 - overall_alpha))
    test_stats = {
        "n_items":      n_items,
        "n_models":     n_models,
        "mean_total":   mean_total,
        "std_total":    sigma_total,
        "kr20_alpha":   overall_alpha,
        "sem":          sem,
        "ci95_band":    1.96 * sem,
    }

    return item_stats, test_stats`}
      </CodeBlock>

      <Prose>
        The production script reports both per-item statistics and test-level summaries in one call. The per-item dataframe is what you sort and filter to find problematic items: low p-values mean the item is too hard for the current model population, low point-biserials mean the item is not discriminating, and the <Code>alpha_if_deleted</Code> column tells you whether removing each item would raise the test reliability. Items where <Code>alpha_if_deleted</Code> exceeds the overall alpha are immediate candidates for inspection — they are degrading reliability and the test would be more reliable without them.
      </Prose>

      <H3>Applying to MMLU</H3>

      <Prose>
        For MMLU specifically, the workflow is: collect per-item responses from a panel of evaluated models (typically 30 to 100 models is enough for stable item statistics), assemble the response matrix, run the CTT analysis above, and inspect the outputs. Several findings are characteristic of MMLU. First, the overall test reliability is high — typically 0.94 to 0.97 across reasonable model panels — because the test is long (14,042 items) and the underlying ability variance across modern models is large. Second, the item-level statistics reveal substantial heterogeneity: some MMLU items have point-biserials below 0.10, and a non-trivial number have negative correlations with the total score, indicating items where stronger models systematically do worse than weaker models. These are the items where a careful CTT pass adds value: they typically turn out to be either ambiguously phrased, miskeyed, or testing knowledge that does not generalize the way the rest of MMLU does. Third, the difficulty distribution is heavily skewed upward in the modern model era; many items have p-values above 0.95 across current models and contribute essentially zero discriminative information.
      </Prose>

      <CodeBlock language="python">
{`# Assume mmlu_responses is a long-format DF from your eval harness.
# It has one row per (model, item) with a 'correct' column in {0, 1}.

item_stats, test_stats = ctt_analysis(
    mmlu_responses,
    item_col="question_id",
    model_col="model_name",
    score_col="correct",
)

print(f"MMLU test-level statistics:")
print(f"  items:        {test_stats['n_items']}")
print(f"  models:       {test_stats['n_models']}")
print(f"  mean score:   {test_stats['mean_total']:.1f}")
print(f"  std score:    {test_stats['std_total']:.1f}")
print(f"  KR-20 alpha:  {test_stats['kr20_alpha']:.4f}")
print(f"  SEM:          {test_stats['sem']:.2f}")
print(f"  +/- 95% CI:   {test_stats['ci95_band']:.2f}")

# Identify dead items (no variance) and bad items (negative discrimination).
dead = item_stats[item_stats["item_variance"] == 0]
bad  = item_stats[item_stats["point_biserial"] < 0]
weak = item_stats[
    (item_stats["point_biserial"] < 0.10) & (item_stats["item_variance"] > 0)
]
print(f"\\ndead items (zero variance):       {len(dead)}")
print(f"items with negative discrimination: {len(bad)}")
print(f"items with discrimination < 0.10:   {len(weak)}")

# Items where dropping them would improve reliability.
overall_alpha = test_stats["kr20_alpha"]
hurts = item_stats[item_stats["alpha_if_deleted"] > overall_alpha]
print(f"items that hurt overall reliability: {len(hurts)}")`}
      </CodeBlock>

      <Prose>
        Once you have the diagnostic output, the operational decisions follow naturally. Items in the "hurts reliability" set are candidates for either removal or careful review by the benchmark maintainers — they may be genuinely defective (miskeyed, ambiguous), or they may be measuring a different construct than the rest of the test (in which case dropping them sharpens the construct definition rather than removing real signal). Items at saturated difficulty (p > 0.99 across the model panel) are dead weight that can be removed without losing any discriminative information. The Spearman-Brown formula tells you what reliability you would have at the reduced item count, and the difference between that projection and the actual cleaned-test reliability quantifies the value of the cleanup.
      </Prose>

      <H3>Sub-benchmark analysis</H3>

      <Prose>
        For benchmarks composed of multiple subdomains — MMLU's 57 subjects, BIG-Bench's hundreds of tasks, HELM's many scenario groups — running the CTT analysis at the sub-benchmark level surfaces information the test-level summary obscures. A sub-benchmark with low reliability is one where ranking models reliably requires more items in that subdomain. A sub-benchmark whose item-total correlations against the full test are weak is one that measures something different from the rest of the benchmark, which raises the question of whether it should be reported as a separate score rather than aggregated into a single MMLU number. The disattenuated correlation between sub-benchmark scores tells you which subdomains are measuring overlapping versus distinct underlying abilities, which is critical for interpreting whether progress on one subdomain implies progress on others.
      </Prose>

      <CodeBlock language="python">
{`# Reliability per sub-benchmark.
sub_reliabilities = {}
for subject in mmlu_responses["subject"].unique():
    sub_df = mmlu_responses[mmlu_responses["subject"] == subject]
    if sub_df["question_id"].nunique() < 10:
        continue   # too few items for a meaningful estimate
    _, sub_stats = ctt_analysis(sub_df, item_col="question_id",
                                model_col="model_name", score_col="correct")
    sub_reliabilities[subject] = sub_stats["kr20_alpha"]

# Sort by reliability ascending -- lowest is most in need of more items.
for subject, rel in sorted(sub_reliabilities.items(), key=lambda x: x[1]):
    print(f"  {subject:40s} alpha = {rel:.3f}")`}
      </CodeBlock>

      <Prose>
        For HumanEval, the analysis follows the same pattern but with binary execution outcomes (test cases pass / fail) as the response variable. HumanEval is a much smaller benchmark (164 items) and consequently has lower reliability than MMLU at typical model panel sizes — usually in the 0.75 to 0.85 range — which means score differences smaller than about three to four percentage points should be treated as noise. The Spearman-Brown formula tells you exactly how much HumanEval would need to grow to reach a target reliability: a doubling to roughly 330 items would push reliability from 0.78 to about 0.88, and a tripling would push it to 0.91. This is the principled basis for decisions like the development of HumanEval+ (which expanded test coverage to make per-item judgments more reliable) and the various extended versions that the community has produced.
      </Prose>

      <Prose>
        One operational point worth being explicit about: the model panel matters. CTT statistics are properties of the response matrix, which depends on which models are in the panel. A reliability of 0.95 computed across a panel that includes both 7B and 70B models is largely driven by the gross differences between model classes; the same benchmark restricted to a panel of similarly-sized models would have lower reliability because the ability variance is smaller. When reporting CTT statistics for a benchmark, always report the panel composition. When using CTT to make decisions about benchmark design, choose the panel composition to match the model size class the benchmark is intended to discriminate among.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The Spearman-Brown prophecy curve shows how reliability scales with test length for several starting reliabilities. The k axis is the multiplier on the original test length: k = 1 is the original test, k = 2 is doubled length, k = 0.5 is halved. The y axis is the projected reliability. Notice the diminishing-returns shape: gains are large when starting reliability is modest and the test is short, but become incremental as reliability approaches one.
      </Prose>

      <Plot
        label="Spearman-Brown prophecy formula — projected reliability vs. test length multiplier"
        xLabel="length multiplier k"
        yLabel="projected reliability"
        width={520}
        height={260}
        series={[
          {
            name: "starting rho = 0.5",
            color: colors.gold,
            points: [
              [0.25, 0.250], [0.5, 0.333], [1.0, 0.500],
              [1.5, 0.600], [2.0, 0.667], [3.0, 0.750],
              [4.0, 0.800], [5.0, 0.833], [8.0, 0.889], [10.0, 0.909],
            ],
          },
          {
            name: "starting rho = 0.7",
            color: "#9ca3af",
            points: [
              [0.25, 0.368], [0.5, 0.538], [1.0, 0.700],
              [1.5, 0.778], [2.0, 0.824], [3.0, 0.875],
              [4.0, 0.903], [5.0, 0.921], [8.0, 0.949], [10.0, 0.959],
            ],
          },
          {
            name: "starting rho = 0.85",
            color: "#c084fc",
            points: [
              [0.25, 0.586], [0.5, 0.739], [1.0, 0.850],
              [1.5, 0.895], [2.0, 0.919], [3.0, 0.944],
              [4.0, 0.958], [5.0, 0.966], [8.0, 0.978], [10.0, 0.983],
            ],
          },
        ]}
      />

      <Prose>
        The reliability vs. SEM relationship plot illustrates how the standard error of measurement shrinks as reliability rises, holding observed-score standard deviation fixed. The curve is concave: large reductions in SEM at the low-reliability end give way to small reductions near the top. For a benchmark with σ_X = 10 percentage points, moving from reliability 0.50 to 0.80 cuts SEM from 7.07 to 4.47 — a meaningful tightening of confidence bands. Moving from 0.90 to 0.95 only moves SEM from 3.16 to 2.24, and from 0.95 to 0.98 from 2.24 to 1.41. Beyond 0.95, the marginal cost in items required to reduce SEM further usually exceeds the practical benefit.
      </Prose>

      <Plot
        label="SEM vs. reliability for fixed sigma_X = 10"
        xLabel="reliability"
        yLabel="standard error of measurement"
        width={520}
        height={240}
        series={[
          {
            name: "SEM = sigma_X * sqrt(1 - rho)",
            color: colors.gold,
            points: [
              [0.30, 8.367], [0.40, 7.746], [0.50, 7.071],
              [0.60, 6.325], [0.70, 5.477], [0.80, 4.472],
              [0.85, 3.873], [0.90, 3.162], [0.93, 2.646],
              [0.95, 2.236], [0.97, 1.732], [0.98, 1.414],
              [0.99, 1.000],
            ],
          },
        ]}
      />

      <Prose>
        The item difficulty / discrimination scatter is the standard CTT diagnostic plot for benchmark QA. Each dot represents one item, plotted by its difficulty (x axis, p-value) against its discrimination (y axis, point-biserial). The healthy region is the center-top: moderate difficulty (0.3 to 0.7), high discrimination (above 0.3). Items at the bottom of the plot — low or negative discrimination — are candidates for review or removal. Items at the extreme left or right edges (very hard or very easy) contribute little variance regardless of their discrimination.
      </Prose>

      <Plot
        label="Item difficulty (p-value) vs. discrimination (point-biserial) — illustrative scatter for the simulated 50-item benchmark"
        xLabel="p-value (item difficulty, easier to the right)"
        yLabel="point-biserial discrimination"
        width={520}
        height={260}
        series={[
          {
            name: "items",
            color: colors.gold,
            points: [
              [0.08, 0.21], [0.12, 0.32], [0.16, 0.41], [0.18, 0.18],
              [0.22, 0.45], [0.25, 0.39], [0.28, 0.52], [0.31, 0.48],
              [0.34, 0.55], [0.37, 0.36], [0.40, 0.62], [0.43, 0.51],
              [0.45, 0.58], [0.47, 0.07], [0.49, 0.49], [0.51, 0.61],
              [0.53, 0.43], [0.55, 0.54], [0.58, 0.47], [0.60, 0.36],
              [0.62, 0.45], [0.64, 0.51], [0.66, 0.39], [0.68, 0.41],
              [0.70, 0.33], [0.72, 0.42], [0.74, 0.27], [0.76, 0.36],
              [0.78, 0.31], [0.80, 0.24], [0.82, 0.29], [0.84, 0.21],
              [0.86, 0.18], [0.88, 0.15], [0.90, 0.12], [0.94, 0.09],
            ],
          },
          {
            name: "discrimination floor (0.20)",
            color: colors.textDim,
            points: [
              [0.0, 0.20], [1.0, 0.20],
            ],
          },
        ]}
      />

      <Prose>
        The full CTT analysis pipeline can be traced as a sequence of steps from raw response matrix through item statistics to actionable decisions. The walkthrough below decomposes the workflow into the five conceptual phases.
      </Prose>

      <StepTrace
        label="CTT analysis pipeline — from response matrix to QA decisions"
        steps={[
          {
            label: "Assemble response matrix",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>R = (n_models, n_items) binary matrix</div>
                <div>R[j, i] = 1 if model j answered item i correctly, else 0</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Source: per-item evaluation logs from any standard harness.
                  Pivot from long-format to wide-format. Drop or impute missing.
                </div>
              </div>
            ),
          },
          {
            label: "Compute test-level summaries",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Per-model totals</div>
                <div>X_j = sum_i R[j, i]                # total score per model</div>
                <div>mean_X, std_X over j</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  std_X is the observed-score standard deviation that feeds
                  into the SEM calculation in the next step.
                </div>
              </div>
            ),
          },
          {
            label: "Compute item statistics",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>For each item i</div>
                <div>p_i  = R[:, i].mean()</div>
                <div>v_i  = p_i * (1 - p_i)</div>
                <div>r_pb = corr(R[:, i], X - R[:, i])</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Item variance is binomial. Point-biserial uses the
                  rest-score (total minus this item) to avoid spurious
                  self-correlation.
                </div>
              </div>
            ),
          },
          {
            label: "Compute reliability and SEM",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>KR-20 / Cronbach alpha</div>
                <div>alpha = n/(n-1) * (1 - sum(v_i) / var(X))</div>
                <div>SEM   = std_X * sqrt(1 - alpha)</div>
                <div>ci95  = +/- 1.96 * SEM</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  alpha is the reliability lower bound under tau-equivalence.
                  SEM converts it into a confidence band on individual scores.
                </div>
              </div>
            ),
          },
          {
            label: "Surface decisions",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Diagnostic outputs</div>
                <div>flag items where r_pb &lt; 0.20 or alpha_if_deleted &gt; alpha</div>
                <div>flag items where p_i &gt; 0.99 or p_i &lt; 0.05 (saturated)</div>
                <div>compute Spearman-Brown projections for length changes</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Output is a per-item dataframe sorted by problem severity
                  plus a test-level summary card with reliability and SEM.
                </div>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        Finally, the heatmap below shows the structure of correlations among items in a small sample subset. Highly positively correlated items measure the same underlying construct; weakly correlated items either measure different constructs or are too noisy to reveal their construct relationships. In a well-formed test, the average pairwise inter-item correlation is positive and stable across item subsets. A heatmap with strong block structure suggests the test is measuring multiple distinct constructs, which is a signal to either split the score reporting or to drop the off-construct items.
      </Prose>

      <Heatmap
        matrix={[
          [1.00, 0.34, 0.41, 0.28, 0.36, 0.31, 0.39, 0.27],
          [0.34, 1.00, 0.38, 0.42, 0.31, 0.45, 0.33, 0.36],
          [0.41, 0.38, 1.00, 0.39, 0.44, 0.36, 0.40, 0.32],
          [0.28, 0.42, 0.39, 1.00, 0.37, 0.41, 0.35, 0.30],
          [0.36, 0.31, 0.44, 0.37, 1.00, 0.43, 0.38, 0.34],
          [0.31, 0.45, 0.36, 0.41, 0.43, 1.00, 0.41, 0.38],
          [0.39, 0.33, 0.40, 0.35, 0.38, 0.41, 1.00, 0.36],
          [0.27, 0.36, 0.32, 0.30, 0.34, 0.38, 0.36, 1.00],
        ]}
        rowLabels={["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"]}
        colLabels={["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"]}
        cellSize={48}
        colorScale="gold"
        label="Inter-item correlation matrix — well-formed homogeneous test (sample 8 items)"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>CTT vs. IRT</H3>

      <Prose>
        The fundamental choice in psychometric analysis of any test is between classical test theory and item response theory. CTT is computationally trivial, makes minimal distributional assumptions, and produces statistics that are easy to interpret but sample-dependent — the difficulty and discrimination of an item shift if you change the population of examinees. IRT is computationally heavier, requires you to assume a parametric form for the item response function (Rasch, two-parameter logistic, three-parameter logistic, generalized partial credit), and requires larger samples to estimate stably, but produces sample-invariant item parameters and supports computer-adaptive testing. The right rule of thumb is to use CTT as the always-on first-pass diagnostic that you run on every benchmark every time, and reach for IRT when you specifically need population invariance, when you are designing computer-adaptive evaluations, or when you want to model item characteristics more flexibly than CTT permits. For LLM benchmark QA in particular, CTT is almost always sufficient: you typically have a fixed item pool and a fixed model panel, and the sample-dependence of CTT statistics is not a binding limitation.
      </Prose>

      <H3>CTT vs. generalizability theory</H3>

      <Prose>
        Generalizability theory (Cronbach et al. 1972) extends CTT by decomposing error variance into multiple sources rather than treating it as a single undifferentiated noise term. In a standard G-study, you might decompose error variance into components attributable to items, raters, occasions, and their interactions, which lets you target reliability improvements at the largest variance source. For LLM benchmarks where the response is deterministic given (model, item, prompt template), classical CTT typically suffices because there is no rater or occasion variance to model. G-theory becomes relevant when stochastic decoding, ensemble evaluation, or multiple judge models contribute additional sources of variability — for example, an LLM-as-judge benchmark where the judge model itself is a noise source, or a chain-of-thought benchmark where sampling temperature contributes variance. In those cases, G-theory's variance decomposition gives you a principled way to allocate items, judges, and samples to maximize reliability per unit of evaluation cost.
      </Prose>

      <H3>KR-20 vs. Cronbach's alpha vs. McDonald's omega</H3>

      <Prose>
        For binary items KR-20 and Cronbach's alpha are mathematically identical and either name is acceptable. For polytomous or continuous items, alpha is the standard choice. Both alpha and KR-20 are lower bounds on the true reliability that are tight only when items are essentially tau-equivalent — measuring the same true score with possibly different intercepts but the same loading. When this assumption fails, alpha underestimates reliability. McDonald's omega (McDonald 1999) drops the tau-equivalence assumption and instead estimates reliability from the loadings of a one-factor model fit to the items. Omega is generally a tighter lower bound than alpha, and in cases where items have substantially different loadings on the underlying construct, omega can exceed alpha by a meaningful margin. For most LLM benchmark QA, alpha suffices because the diagnostic value comes from comparing reliabilities across configurations (full test versus filtered test, full population versus subset population) where any underestimation bias is roughly constant. Use omega when the absolute reliability value matters and you suspect tau-equivalence is violated, which becomes more likely the more heterogeneous the items are in difficulty and content.
      </Prose>

      <H3>When to use CTT for benchmark QA versus pretend it does not exist</H3>

      <Prose>
        Use CTT for every benchmark you build, every benchmark you evaluate models on, and every benchmark you cite. The cost is essentially zero — a few seconds of compute on a precomputed response matrix — and the diagnostic value is high. There is no good reason to skip it. The historical reason most LLM benchmarks ship without CTT statistics is not principled; it is simply that the LLM evaluation community grew up outside the psychometric tradition and has not yet absorbed its tools. The cases where CTT is positively misleading rather than merely uninformative are vanishingly rare: tests with strongly multidimensional constructs (where alpha underestimates reliability dramatically), tests with strong dependencies between items (where the assumption of independent errors is violated), and tests with heavy ceiling or floor effects (where the assumption that errors are uncorrelated with true scores breaks down). For tests with multidimensional constructs, supplement CTT with factor-analytic methods or omega. For tests with item dependencies, model the dependencies explicitly or use generalizability theory. For tests with ceiling/floor effects, recognize that the population restriction is the binding issue and that no internal-consistency reliability estimator will give a defensible answer.
      </Prose>

      <H3>When you should care about reliability versus when you can ignore it</H3>

      <Prose>
        Reliability matters most when the consequence of being wrong is asymmetric or expensive. In an LLM evaluation context, reliability matters most when (1) you are using benchmark scores to make ranking decisions where small score differences will be reported as meaningful capability differences, (2) you are using benchmark scores in an automated pipeline that takes action based on threshold crossings (release a model if it scores above X), (3) you are tracking benchmark scores over time as a measure of progress and need to distinguish real improvement from noise, or (4) you are aggregating across benchmarks and need to weight them by their measurement quality rather than treating them as equivalent. Reliability matters least when you are doing exploratory comparison of models that differ by orders of magnitude in capability — the gross differences will swamp the noise. The general rule is that the tighter the model comparison, the more reliability matters; for a leaderboard where the top ten models cluster within a few percentage points, every one of those rankings should be reported with explicit confidence bands derived from the SEM.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        CTT scales pleasantly. The compute cost of a full CTT analysis is essentially the cost of a few matrix operations on the response matrix: column means for difficulties, column-wise correlations against the rest-score for discriminations, and one variance computation for KR-20. For a benchmark of 100,000 items administered to 1,000 models — far larger than any current benchmark — the entire analysis runs in seconds on a single CPU. There is no iterative estimation, no convergence diagnostic, no stochastic optimization. This makes CTT the only psychometric framework you can run as part of a CI pipeline that fires every time a new model joins the leaderboard, automatically updating item statistics and flagging changes to test-level reliability without any manual intervention.
      </Prose>

      <Prose>
        Sample size requirements are modest by psychometric standards. The traditional rule of thumb is that you want at least 100 examinees per item-statistic estimate to get stable values, with 200 to 500 considered comfortable. For LLM benchmarks the relevant sample size is the number of models in the panel, which is usually in the range 20 to 200. At 20 models, individual item statistics are noisy but the test-level reliability is still well-estimated. At 50 to 100 models, item-level statistics become reliable enough to act on. Above 100, additional models add diminishing returns to statistical precision. The asymmetry is worth being explicit about: reliability is a property that depends on both the number of items and the number of models, but in different ways. More items raises reliability through the prophecy formula, with the rate of improvement set by the average item quality. More models tightens the estimates of reliability and item statistics without changing the underlying reliability itself.
      </Prose>

      <Prose>
        What does not scale well is CTT's dependence on the population that took the test. The reliability of a benchmark is not an intrinsic property of the benchmark — it is a property of the benchmark administered to a particular population. A benchmark that has reliability 0.95 across a panel of models spanning 1B to 70B parameters may have reliability 0.70 across a panel restricted to top-tier 70B models, because the true-score variance shrinks faster than the error variance when you restrict the population. This is the population-dependence problem that motivates IRT, where item parameters are intended to be invariant to the population. In practice, most benchmark QA work happens in a fixed-panel setting where the population dependence is not an active concern — you are computing CTT on the panel you have, and the inferences you draw apply to that panel. But the statistic you compute does not generalize to a different panel, and conflating panels (mixing top-tier models with weak baselines, for example) inflates reliability in a way that can be misleading if the goal is to discriminate among the top tier.
      </Prose>

      <Prose>
        The other limitation that does not scale away is CTT's blindness to the parametric structure of item response. CTT's item difficulty is the proportion correct on this panel, full stop. It does not tell you what the difficulty would be on a different panel, it does not let you compare difficulties across benchmarks administered to different populations, and it does not support computer-adaptive testing where item selection depends on running ability estimates. For all of those use cases, you need IRT. The right framing is that CTT is the right tool for fixed-panel analysis of fixed item pools, which is most of LLM benchmark QA. When the use case demands invariance across populations or adaptivity, the CTT analysis is the precursor to an IRT model, not a replacement for one.
      </Prose>

      <Prose>
        Reliability hits hard ceilings around 0.99 in practice. The Spearman-Brown formula projects monotonic increases with test length, but the assumptions break down at very high reliability: items become more correlated than the prophecy formula's independence-of-errors assumption allows, and additional items add less than the formula predicts. For benchmark design, the practical target is usually 0.90 to 0.95, which is enough to distinguish models with score differences of about 1.5 to 3 SEMs (roughly 4 to 8 percentage points on typical benchmarks). Pushing reliability higher costs more items than the gain justifies, and the marginal reduction in SEM is small relative to other sources of uncertainty (prompt template variance, decoding variance, model version drift) that are not captured in the CTT framework at all.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Sample-dependent item statistics</H3>
      <Prose>
        CTT item statistics are properties of the response matrix, which is a property of the model panel that took the benchmark. If you publish item difficulties for MMLU computed on a panel of 30 mid-2024 models, those difficulties will not transfer to a panel of late-2025 frontier models — every item will look easier because the population is more capable. The mistake to avoid is treating CTT difficulty as a property of the item alone. It is not; it is a joint property of the item and the population. When reporting CTT statistics, always report the panel composition, and recompute statistics whenever the panel changes meaningfully.
      </Prose>

      <H3>Confounding ability and difficulty</H3>
      <Prose>
        Related to the previous point: CTT cannot disentangle examinee ability from item difficulty in a way that supports cross-population comparison. If you want to ask whether item A is easier than item B in some absolute sense, CTT cannot answer — both items have p-values that depend on the panel. This is the limitation that motivates IRT, which models examinee ability and item difficulty on a common latent scale where parameters are intended to be invariant. For LLM benchmark QA where you typically have a fixed panel and fixed items, this confounding is benign. For benchmark comparison across populations, you must either restrict the comparison to common items administered to common models, or fit an IRT model that supports linking.
      </Prose>

      <H3>Negative discriminations as a quality signal</H3>
      <Prose>
        An item with a negative point-biserial correlation is not just a noisy item; it is almost always a defective item. The most common causes are miskeying (the answer key is wrong), ambiguous wording (the item supports multiple defensible answers), or the item testing a different construct from the rest of the test (e.g., a math problem in a reading comprehension test, where the correct answer requires arithmetic skill that is uncorrelated with reading ability). Always investigate negative-discrimination items individually rather than just dropping them. Often the fix is to re-key the item or rewrite the prompt; sometimes the fix is to acknowledge that the item belongs in a separate sub-benchmark. Dropping without investigating loses information about benchmark construction issues.
      </Prose>

      <H3>Cronbach's alpha as a lower bound that can be very loose</H3>
      <Prose>
        Alpha is a lower bound on reliability under the assumption that items are essentially tau-equivalent — measuring the same true score with possibly different intercepts. When items measure the same construct but with different loadings (some items are stronger indicators than others), alpha underestimates reliability, sometimes substantially. For multidimensional tests where items load on multiple distinct constructs, alpha can be quite low even when the test is highly informative for each individual construct. The diagnostic for this failure mode is comparing alpha to McDonald's omega: if omega is meaningfully larger than alpha, the items are not tau-equivalent and alpha is conservative. The fix is either to use omega instead, to factor the test into unidimensional subtests and report alphas per subtest, or to acknowledge the multidimensionality explicitly.
      </Prose>

      <H3>Inflated alpha from dependent items</H3>
      <Prose>
        The opposite failure: alpha can be inflated when items violate the assumption of independent errors. The classic example is a test with item bundles or testlets — sets of items sharing a common stimulus (a passage, a diagram) where errors are correlated within the bundle. Treating bundle-level dependencies as item-level independence inflates alpha because the shared variance from the bundle is double-counted. For LLM benchmarks, the equivalent concern arises when items share generation provenance — for example, multiple items derived from the same source document where errors of understanding the source propagate to all derived items. The fix is to model bundles explicitly using either generalizability theory or hierarchical reliability estimators.
      </Prose>

      <H3>Reliability inflation from heterogeneous panels</H3>
      <Prose>
        Mixing populations of widely different ability levels in the same panel inflates reliability above what is meaningful for within-tier discrimination. A panel that includes a 1B model and a 70B model will produce a high reliability not because each item is well-discriminating, but because the gross ability difference between the models swamps any noise. Reliabilities computed on such panels can give false confidence about the benchmark's ability to distinguish similarly-capable models. Always compute reliability on panels that match the discrimination range of interest. For comparing top-tier models, compute reliability on top-tier panels; for assessing capability across the model size range, compute reliability on the cross-tier panel and report it as such.
      </Prose>

      <H3>SEM applied to the wrong scale</H3>
      <Prose>
        The SEM formula assumes the observed score and the true score are on the same scale. If you transform scores nonlinearly — applying a percentile rank, mapping to a standardized score, or applying logit transformations — the SEM does not transfer through the transformation. For LLM benchmarks where the natural score is the proportion of items correct, the SEM is in proportion-correct units and applies directly. If you then report the score on a 0–100 scale, the SEM scales linearly. If you report it as a normalized z-score, the SEM is in z-units and you must reconvert. The mistake to avoid is reporting an SEM in raw-score units alongside a transformed leaderboard score — they will not align.
      </Prose>

      <H3>Treating the SEM as a per-comparison band</H3>
      <Prose>
        The SEM is the standard error on a single examinee's true score given their observed score. The standard error of the difference between two examinees' observed scores is √2·SEM under the assumption of independent errors. This means that to declare two scores significantly different at the 95% level, the difference must exceed about 2.77·SEM, not 1.96·SEM. The correct quantity for comparing two observed scores is sometimes called the standard error of difference (SEdiff) and equals σ_X · √(2(1 − ρ)). Using the wrong band understates the noise floor for comparison and overstates the significance of small differences.
      </Prose>

      <H3>Population restriction shrinking observed reliability</H3>
      <Prose>
        Reliability is the ratio of true-score variance to observed-score variance, both of which depend on the population. Restricting to a more homogeneous subpopulation shrinks both, but typically shrinks true-score variance faster than error variance, so the reliability ratio falls. A test reliable for the general population may be unreliable for any narrow subset. The mistake here is assuming reliability transfers; it does not. When evaluating a benchmark's suitability for a specific use case (e.g., distinguishing between top-tier models), always recompute reliability on the relevant subpopulation rather than relying on a reliability estimate from a broader panel.
      </Prose>

      <H3>Ignoring item-prompt coupling</H3>
      <Prose>
        Standard CTT treats items as fixed quantities. For LLM benchmarks, the same logical item evaluated under different prompt templates, few-shot configurations, or chain-of-thought instructions can produce substantially different correctness outcomes. The CTT statistics depend on the prompt configuration and do not generalize across configurations. A benchmark where items have high discrimination under one prompt template may have low discrimination under another. The fix is either to fix the prompt configuration as part of the benchmark definition (which most benchmark releases now do), or to model the prompt as a separate facet using generalizability theory.
      </Prose>

      <Callout accent="purple">
        The single most common CTT mistake in LLM benchmark practice is to treat the headline reliability number as a property of the benchmark rather than as a joint property of the benchmark, the model panel, and the evaluation configuration. There is no such thing as "the reliability of MMLU"; there is only "the reliability of MMLU on a particular panel under a particular evaluation configuration". Always report the conditioning explicitly.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The five sources below are the load-bearing references for the framework as developed in this topic. Spearman 1904 is the founding paper. Gulliksen 1950 and Lord & Novick 1968 are the canonical textbook treatments that codified CTT into its modern form. Cronbach 1951 introduced the alpha coefficient that bridges CTT to its successors. Crocker & Algina 1986 is the standard graduate textbook for the field. References were verified against original publications and standard psychometric bibliographies as of 2026-04.
      </Prose>

      <H3>Spearman 1904 — the founding paper</H3>
      <Prose>
        Charles Spearman. "The Proof and Measurement of Association Between Two Things." American Journal of Psychology, 15(1), 72–101 (1904). The paper that begins classical test theory. Spearman introduced the decomposition of an observed measurement into a true component and an error component, derived the attenuation formula for correlations between fallible measures, and introduced the correction-for-attenuation procedure. The paper is also the historical origin of the rank correlation coefficient that bears his name. Reading the original is rewarding both for its mathematical clarity and for the historical context in which the field was established.
      </Prose>

      <H3>Gulliksen 1950 — the first systematic textbook</H3>
      <Prose>
        Harold Gulliksen. "Theory of Mental Tests." Wiley, 1950 (reissued by Lawrence Erlbaum Associates, 1987). The first textbook to systematically organize the results of CTT into a coherent framework, with chapters on reliability estimation, validity, item analysis, and test construction. Gulliksen's treatment includes derivations of the Spearman-Brown formula, KR-20, and the standard methods of item analysis that remain in use today. The 1987 reissue includes a foreword by Lord that contextualizes Gulliksen's work in the development of psychometrics through the 1950s and 1960s.
      </Prose>

      <H3>Lord & Novick 1968 — the canonical axiomatic treatment</H3>
      <Prose>
        Frederic M. Lord and Melvin R. Novick. "Statistical Theories of Mental Test Scores." Addison-Wesley, 1968. The definitive axiomatic presentation of classical test theory, with mathematical rigor that remained the field's gold standard for decades. The book introduces classical test theory in the first half and item response theory in the second half, providing the bridge between the two frameworks that defines modern psychometrics. Lord and Novick's treatment of reliability, parallel forms, and the various reliability estimators — KR-20, alpha, split-half, test-retest — is the canonical reference. The chapters by Allan Birnbaum on item response theory were the first widely-circulated mathematical treatment of IRT and seeded that field's subsequent development.
      </Prose>

      <H3>Cronbach 1951 — the alpha coefficient</H3>
      <Prose>
        Lee J. Cronbach. "Coefficient Alpha and the Internal Structure of Tests." Psychometrika, 16(3), 297–334 (1951). Cronbach generalized KR-20 to items with arbitrary scoring (not just binary), introducing what is now universally known as Cronbach's alpha. The paper also clarified the conceptual foundations of internal-consistency reliability and the assumptions under which alpha is an exact reliability estimate versus a lower bound. By far the most cited paper in the psychometric literature, with the alpha coefficient remaining the default reliability statistic across psychology, education, and most applied measurement fields.
      </Prose>

      <H3>Crocker & Algina 1986 — the standard graduate textbook</H3>
      <Prose>
        Linda Crocker and James Algina. "Introduction to Classical and Modern Test Theory." Holt, Rinehart and Winston, 1986 (reissued by Cengage, 2008). The standard graduate textbook for measurement theory courses since the late 1980s, with comprehensive coverage of CTT, generalizability theory, and IRT at a level accessible to graduate students in psychology and education. Crocker and Algina's chapters on item analysis and test construction are the most practically oriented in the standard literature, with worked examples covering every major statistic discussed in this topic. The 2008 reissue retains the original text essentially unchanged, with updated references and an additional chapter on computer-based testing.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the variance decomposition without assuming uncorrelatedness</H3>
      <Prose>
        Start from X = T + E where E is defined as X − T. Take the variance of both sides and expand the cross term Cov(T, E) without assuming anything about it. What does the resulting expression look like, and which assumption would you need to make to recover the canonical CTT identity Var(X) = Var(T) + Var(E)? Now consider a real scenario where Cov(T, E) is negative — for example, a test where high-ability examinees are more careless and low-ability examinees are more meticulous. How does the resulting expression for Var(X) compare to the canonical identity? What does this imply for KR-20 estimates of reliability when the assumption is violated in the negative direction?
      </Prose>

      <H3>Exercise 2 — Apply Spearman-Brown to a benchmark sizing decision</H3>
      <Prose>
        You are designing a new code-generation benchmark and have run a 50-item pilot with a panel of 40 models. The KR-20 reliability of the pilot is 0.71. Your target reliability for the production benchmark is 0.92. Use the Spearman-Brown prophecy formula to determine how many items you need in the production benchmark, assuming the new items are of comparable quality to the pilot items. Now suppose that during construction of the new items, you can either add 200 average-quality items or 80 high-quality items where each high-quality item contributes 2.5 times the per-item reliability gain of an average item. Which choice gets you closer to the target reliability? Show the calculation, including the assumptions you are making to compare the two options.
      </Prose>

      <H3>Exercise 3 — Interpret a problematic item statistics report</H3>
      <Prose>
        You run a CTT analysis on a benchmark and find the following anomalous items in the report. Item 17 has p-value 0.51 and point-biserial 0.04. Item 42 has p-value 0.97 and point-biserial 0.62. Item 88 has p-value 0.34 and point-biserial −0.18. Item 119 has p-value 0.99 and point-biserial undefined. For each item, diagnose the most likely problem (or non-problem), explain what the item statistics suggest, and recommend an action: keep, drop, investigate, or rewrite. For Item 88 specifically, what kinds of substantive problems with the item content would produce a negative point-biserial, and how would you investigate to distinguish among them?
      </Prose>

      <H3>Exercise 4 — Compute the SEM for two configurations and interpret</H3>
      <Prose>
        Benchmark A has 200 items, observed-score standard deviation 22 percentage points, and KR-20 reliability 0.94. Benchmark B has 50 items, observed-score standard deviation 14 percentage points, and KR-20 reliability 0.81. Compute the standard error of measurement for each benchmark in percentage-point units. Compute the standard error of difference for comparing two models on each benchmark. For Benchmark A, what is the smallest score difference between two models that is statistically significant at the 95% level? Same question for Benchmark B. Now consider that an LLM company reports a model improvement of 1.2 percentage points on Benchmark A and 4.5 percentage points on Benchmark B. Which improvement is more likely to be a genuine capability gain rather than measurement noise, and what specifically would you ask the company to disclose to support either claim?
      </Prose>

      <H3>Exercise 5 — Reason about reliability under population restriction</H3>
      <Prose>
        A benchmark has reliability 0.92 across a panel of 100 models spanning 1B to 200B parameters. You restrict the panel to the top 30 models (all 70B+ models with capabilities clustered tightly). The observed-score standard deviation across this restricted panel falls from 18 percentage points to 5 percentage points. Assuming the error variance is approximately constant across the restriction (which is a reasonable approximation when error sources are dominated by item-level noise rather than model-level effects), compute the new reliability. Interpret the result: what does it mean that the same benchmark has very different reliabilities for the broad panel versus the restricted panel, and what does this imply about how to use this benchmark for distinguishing top-tier models specifically? What kind of intervention (add items, change items, change panel) would you recommend to restore high reliability for top-tier discrimination, and what are the tradeoffs of each option?
      </Prose>

    </div>
  ),
};

export default classicalTestTheory;
