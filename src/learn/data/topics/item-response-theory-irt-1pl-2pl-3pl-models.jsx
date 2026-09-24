import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const irtModels = {
  title: "Item Response Theory (IRT): 1PL, 2PL, 3PL Models",
  slug: "item-response-theory-irt-1pl-2pl-3pl-models",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Classical Test Theory (CTT) is the framework that most people implicitly use when they think about tests, scores, and benchmarks. You give a person a set of questions, you count how many they got right, you call that their score, and you treat the score as a measurement of ability. The framework is simple, computationally trivial, and durable enough that it underlies most school grading, most early psychometric research, and most modern LLM benchmarks. It is also fundamentally confounded in a way that becomes visible the moment you try to compare scores across different test versions or different populations. A student who scores 80% on an easy test and a student who scores 80% on a hard test are not, in any meaningful sense, equally able. A model that gets 0.65 on MMLU and another that gets 0.65 on MMLU-Pro are not equally capable. The score is a function of two things — the underlying ability of the test-taker and the difficulty of the items chosen — and CTT does not separate them. Item difficulty in CTT is just the proportion of test-takers who got the item right; person ability is just the proportion of items they got right. The two definitions are circular, and as a result CTT has no principled way to talk about ability that does not depend on the specific items used, or item difficulty that does not depend on the specific population sampled.
      </Prose>

      <Prose>
        Item Response Theory (IRT) is the framework that fixes this by introducing a latent-variable model. Instead of treating the test score as the measurement, IRT treats each individual item response as a noisy observation of an underlying ability parameter. Every test-taker <Code>i</Code> has a latent ability <Code>{"\\theta_i"}</Code> (a real number, conventionally on the standard-normal scale). Every item <Code>j</Code> has its own parameters: difficulty <Code>{"b_j"}</Code>, discrimination <Code>{"a_j"}</Code>, and possibly a guessing parameter <Code>{"c_j"}</Code>. The probability that test-taker <Code>i</Code> answers item <Code>j</Code> correctly is a known function of these parameters. Critically, the model factorizes: the item parameters do not depend on which person attempts the item, and the person parameter does not depend on which items they happen to be given. This factorization, called <em>parameter invariance</em> in the IRT literature, is what makes IRT scores comparable across different test forms and different populations. It is the property that CTT structurally lacks and the property that makes IRT the standard framework for high-stakes testing — SAT, GRE, GMAT, LSAT, Bar Exam, and most adaptive testing platforms are built on IRT models.
      </Prose>

      <Prose>
        The framework was developed in two parallel traditions. Georg Rasch, a Danish statistician, published the original one-parameter model in 1960 in his book <em>Probabilistic Models for Some Intelligence and Attainment Tests</em>. Rasch's model assumes all items differ only in difficulty and have equal discrimination — a deliberately restrictive assumption that gives the model attractive measurement-theoretic properties (specifically, sufficient statistics: the raw score is a sufficient statistic for ability, and the column sum is sufficient for item difficulty). Independently, Frederic Lord at Educational Testing Service published a more general two-parameter formulation in 1952 (his Princeton dissertation) and the three-parameter formulation including a guessing parameter in 1968 (with Allan Birnbaum's chapters in Lord and Novick's <em>Statistical Theories of Mental Test Scores</em>). The Rasch tradition continues to emphasize fundamental measurement properties; the IRT tradition emphasizes empirical fit. The technical results are essentially the same family of models, but the philosophy of how strictly to enforce model fit versus how flexibly to fit the data differs sharply.
      </Prose>

      <Prose>
        For decades IRT was a tool of educational measurement and psychological testing — relevant if you were building a standardized exam, irrelevant if you were doing almost anything else in machine learning. That changed in 2024 when Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin published "tinyBenchmarks: evaluating LLMs with fewer examples" (arXiv:2402.14992), which applied IRT to large language model evaluation. Their core observation: most LLM benchmarks (MMLU, BIG-Bench, HellaSwag, etc.) contain hundreds to tens of thousands of items, but the items vary enormously in informativeness. Many items are too easy — every model above a certain ability gets them right and they contribute almost no signal about which models are actually better. Many are too hard — almost no model gets them right and they contribute almost no signal either. Some items are wonderfully informative right at the ability range where current frontier models live, and those are the items that actually distinguish a 70B Llama from a 405B Llama. By fitting an IRT model to the matrix of (model, item, correct/incorrect) responses, Polo et al. could identify a small subset of items that preserved the ability-ranking power of the full benchmark. tinyMMLU has 100 items and reproduces full MMLU rankings with rank correlation above 0.97. tinyHellaSwag, tinyTruthfulQA, tinyArc, and tinyWinogrande followed the same recipe. EleutherAI's <Code>lm-eval-harness</Code> integrates these tinyBenchmarks directly. The motivating question — "given a fixed evaluation budget, which items should I run?" — is exactly the question that IRT was built to answer for adaptive testing in the 1970s, and the answer is exactly the same: pick items whose information functions peak in the ability range you care about.
      </Prose>

      <Prose>
        For an AI/ML practitioner in 2026, IRT matters for at least three concrete reasons. First, evaluation cost: running a 405B model on a 14k-item benchmark costs real money, and IRT-curated subsets cut that by 100x with negligible loss in ranking signal. Second, evaluation transferability: when you collect human-rater scores or LM-judge scores on a fresh benchmark, IRT lets you calibrate item difficulty and model ability on the same scale, so a new model can be placed on the existing ability axis without re-running the full benchmark. Third, benchmark diagnostics: when you fit an IRT model and find items whose discrimination is near zero or whose response curves are inverted (better models do <em>worse</em>), those items are usually mislabeled, ambiguous, or hacked — a finding that has surfaced real problems in MMLU, HellaSwag, and BoolQ over the last two years. IRT is the diagnostic that surfaces them.
      </Prose>

      <Prose>
        It is worth noting what IRT is not. It is not a benchmark-design framework — IRT does not tell you what to measure, only how to measure it more efficiently once you have decided what construct your benchmark is supposed to capture. It is not a substitute for actual evaluation: you still need to run the panel of models on the items at calibration time. It is not magic: an IRT-curated subset of a flawed benchmark inherits the flaws. If MMLU has biased item content, tinyMMLU has biased item content. What IRT provides is the inferential machinery to do measurement properly given a benchmark whose items have already been chosen — to separate signal from noise in the items, to separate ability from item-difficulty in the scores, and to compress evaluation cost while preserving rank order. The framework is principled, well-studied, and operationally proven across decades of high-stakes testing; it is finally being adopted into the LLM evaluation stack because the eval costs have grown to the point where the savings justify the framework's overhead.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the picture of a single item. You have a question, and you have a population of test-takers with varying levels of ability. Plot ability on the x-axis (low ability on the left, high ability on the right) and the probability of answering this item correctly on the y-axis. What should this curve look like? At very low ability, the probability should be near zero — the test-taker has essentially no chance of getting it right. At very high ability, the probability should be near one — they essentially always get it right. In between there should be a smooth transition. The natural shape is the logistic curve, an S-shape that rises monotonically from 0 to 1. This curve is called the <em>item characteristic curve</em> (ICC), and every IRT model is fundamentally a parameterization of what shape this curve can take.
      </Prose>

      <Prose>
        The 1PL (Rasch) model says: every item has the same S-shape, but each item is shifted left or right along the ability axis. The shift is the difficulty parameter <Code>{"b"}</Code>. An easy item has its ICC shifted to the left — even low-ability test-takers have a high probability of answering correctly. A hard item has its ICC shifted to the right — only high-ability test-takers reliably succeed. The point on the ability axis where the ICC crosses 0.5 is exactly the difficulty <Code>{"b"}</Code>. A test-taker whose ability equals the item's difficulty has a 50/50 chance of getting it right. Higher ability gives them better-than-even odds; lower ability gives them worse-than-even odds. This is a clean, interpretable model, and it has the elegant property that the raw number of items a person got right is a sufficient statistic for their ability — meaning that two test-takers with the same raw score have the same ability estimate, regardless of which specific items they got right. This is what makes the Rasch model attractive to measurement theorists who want strict measurement properties.
      </Prose>

      <Prose>
        The 2PL model (Birnbaum) says: items can also differ in how steeply the ICC rises. The discrimination parameter <Code>{"a"}</Code> controls the slope of the S-curve at its midpoint. A high-discrimination item has a sharp ICC — the probability of correct response jumps quickly from low to high as you move past the item's difficulty. This is desirable in a test: such items distinguish sharply between test-takers just above and just below the difficulty threshold. A low-discrimination item has a flat ICC — even very high-ability test-takers do not reliably get it right, and even very low-ability test-takers sometimes succeed. Such items contribute little information about who is actually better. A discrimination near zero means the item is uninformative; a negative discrimination is a red flag, indicating a mislabeled item or a question that better test-takers actually do worse on (which happens when a question is "tricky" in a way that punishes careful thought).
      </Prose>

      <Prose>
        The 3PL model (Lord) adds a guessing parameter <Code>{"c"}</Code>. This is the lower asymptote of the ICC: even test-takers with very low ability will sometimes get the item right by guessing. For a four-option multiple-choice question, the floor is 0.25; for a true/false item, the floor is 0.5; for an open-ended question, the floor is near zero. The 3PL ICC is a stretched S-curve that starts at <Code>{"c"}</Code> instead of 0 and rises to 1. The 3PL model is essential when you have multiple-choice items because without it the model systematically overestimates the difficulty of items that low-ability test-takers can guess and underestimates the ability of test-takers who guess well. SAT, GRE, and most standardized multiple-choice tests use 3PL.
      </Prose>

      <Prose>
        Now think about the converse picture: fix the test-taker, vary the items. A test-taker with ability <Code>{"\\theta_i"}</Code> has different probabilities of correctly answering different items. Some items are easy for them (probability near 1), some are right at their threshold (probability near 0.5), some are too hard (probability near 0). The information they provide about <Code>{"\\theta_i"}</Code> is concentrated in items where the response is most uncertain — items whose ICC has its steepest slope at the test-taker's ability level. This is the <em>Fisher information</em> intuition formalized: an item that you would always get right or always get wrong tells you nothing about your ability; an item that you have a 50/50 chance on tells you the most. This information-theoretic view is the basis for adaptive testing (always show the next item with maximum information at the current ability estimate) and for tinyBenchmarks (curate items whose information functions peak in the ability range of the models you care about).
      </Prose>

      <Prose>
        The key conceptual move that IRT requires you to absorb is the latent-variable formulation. Ability <Code>{"\\theta_i"}</Code> is not directly observed. You never measure it. What you observe is binary correct/incorrect responses on items. The ability is inferred from the pattern of those responses, conditional on a model that says how ability relates to response probability. This is why IRT is sometimes confusing on first encounter: the quantity you most want to talk about (ability) is exactly the quantity you cannot see, and the inference machinery that connects observations to ability is doing all of the heavy lifting. Once you accept the latent-variable framing, IRT becomes a standard exercise in maximum likelihood estimation over a parametric model, and the rest follows.
      </Prose>

      <Prose>
        For LLM evaluation specifically, the mapping is direct. The "test-takers" are language models. The "items" are individual benchmark questions (MMLU items, HellaSwag completions, GSM8K problems). The "ability" is the latent capability of each model on the construct measured by the benchmark. The "responses" are the binary scores (correct/incorrect) the model received on each item. Fitting a 2PL or 3PL model to a (models x items) response matrix recovers an ability estimate per model and a difficulty/discrimination/guessing estimate per item. From those estimates, you can: identify uninformative items and drop them; build a tiny, high-information subset that ranks new models accurately; detect mislabeled items via negative discrimination; place a new model on the ability scale by running it on a small, well-chosen subset.
      </Prose>

      <Prose>
        One more intuition worth installing before the math: the relationship between item difficulty and item information is non-monotone in an important way. An extremely easy item (b = -3) is informative only for very-low-ability test-takers; for everyone else it tells you nothing because everyone gets it right. An extremely hard item (b = +3) is informative only for very-high-ability test-takers. An average-difficulty item (b = 0) is informative for an average-ability test-taker. So when you ask "which items should I include in my benchmark?", the answer depends crucially on which ability range you care about. This is the conceptual hinge that makes tinyBenchmarks work: by deciding in advance that you want to measure frontier LLMs (whose abilities are concentrated in a particular range), you can prioritize items whose difficulties match that range and ignore the rest. CTT-style "use all items because they all add some signal" is exactly the wrong intuition; IRT-style "use only items whose information functions peak where I care" is the right one.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let <Code>{"i = 1, \\dots, N"}</Code> index test-takers (or models, in LLM eval) and <Code>{"j = 1, \\dots, J"}</Code> index items. Let <Code>{"X_{ij} \\in \\{0, 1\\}"}</Code> be the binary response of test-taker <Code>i</Code> to item <Code>j</Code>, with 1 indicating correct. Each test-taker has a latent ability <Code>{"\\theta_i \\in \\mathbb{R}"}</Code>. Each item has parameters that depend on the model: 1PL has only difficulty <Code>{"b_j"}</Code>; 2PL has discrimination <Code>{"a_j"}</Code> and difficulty <Code>{"b_j"}</Code>; 3PL adds the guessing parameter <Code>{"c_j \\in [0, 1]"}</Code>.
      </Prose>

      <H3>The three response models</H3>

      <Prose>
        The 1PL (Rasch) model writes the probability of a correct response as a logistic function of the difference between ability and difficulty:
      </Prose>

      <MathBlock>{"P(X_{ij} = 1 \\mid \\theta_i, b_j) = \\sigma(\\theta_i - b_j) = \\frac{1}{1 + \\exp\\!\\left(-(\\theta_i - b_j)\\right)}"}</MathBlock>

      <Prose>
        The 2PL model adds a multiplicative discrimination parameter <Code>{"a_j > 0"}</Code> that scales the slope of the logistic:
      </Prose>

      <MathBlock>{"P(X_{ij} = 1 \\mid \\theta_i, a_j, b_j) = \\sigma(a_j(\\theta_i - b_j))"}</MathBlock>

      <Prose>
        The 3PL model adds a lower asymptote <Code>{"c_j \\in [0, 1]"}</Code> that represents the probability of getting the item right by guessing:
      </Prose>

      <MathBlock>{"P(X_{ij} = 1 \\mid \\theta_i, a_j, b_j, c_j) = c_j + (1 - c_j)\\, \\sigma(a_j(\\theta_i - b_j))"}</MathBlock>

      <Prose>
        All three are special cases of a single family. Setting <Code>{"a_j = 1"}</Code> for all items reduces 2PL to 1PL. Setting <Code>{"c_j = 0"}</Code> reduces 3PL to 2PL. The parametric ladder — adding one parameter at a time — is the standard way IRT is taught and the standard way to think about model selection: start simple, add complexity only when justified by improved fit.
      </Prose>

      <H3>Likelihood</H3>

      <Prose>
        Conditional on the parameters, item responses are assumed independent (this is the <em>local independence</em> assumption — given ability, knowing the response to one item gives no additional information about the response to another). The likelihood of the full response matrix factorizes as a product over (person, item) pairs:
      </Prose>

      <MathBlock>{"L(\\boldsymbol{\\theta}, \\boldsymbol{a}, \\boldsymbol{b}) = \\prod_{i=1}^N \\prod_{j=1}^J P_{ij}^{X_{ij}} (1 - P_{ij})^{1 - X_{ij}}"}</MathBlock>

      <Prose>
        where <Code>{"P_{ij}"}</Code> is the model-specific probability above. Taking the negative log gives the loss to minimize during fitting:
      </Prose>

      <MathBlock>{"\\ell(\\boldsymbol{\\theta}, \\boldsymbol{a}, \\boldsymbol{b}) = -\\sum_{i,j} \\Big[ X_{ij} \\log P_{ij} + (1 - X_{ij}) \\log (1 - P_{ij}) \\Big]"}</MathBlock>

      <Prose>
        This is just the binary cross-entropy summed over the response matrix. Maximum likelihood estimation searches over the joint parameter space <Code>{"(\\theta_i)_i, (a_j, b_j, c_j)_j"}</Code> for the values that maximize this likelihood (equivalently, minimize the cross-entropy).
      </Prose>

      <H3>Identifiability and the standard scale</H3>

      <Prose>
        The 1PL/2PL likelihood is invariant under a shift of all abilities and difficulties by a constant: <Code>{"\\theta_i \\to \\theta_i + c"}</Code> and <Code>{"b_j \\to b_j + c"}</Code> leaves every <Code>{"P_{ij}"}</Code> unchanged. Similarly, 2PL is invariant under a rescaling <Code>{"\\theta_i \\to \\theta_i / s"}</Code>, <Code>{"a_j \\to a_j s"}</Code>, <Code>{"b_j \\to b_j / s"}</Code>. To make the parameters identifiable, you fix the scale by convention. The standard convention is to require <Code>{"\\theta"}</Code> to have mean zero and unit variance across the sample of test-takers — equivalently, to assume <Code>{"\\theta_i \\sim \\mathcal{N}(0, 1)"}</Code> as a prior. This puts difficulty on the same scale as ability: a difficulty of <Code>{"b_j = 0"}</Code> is "average" difficulty (50% pass rate at average ability); positive <Code>b</Code> is harder than average; negative <Code>b</Code> is easier. Discriminations <Code>a</Code> are on a positive scale where 1 is "average" and values above 2 are very high.
      </Prose>

      <H3>The item information function</H3>

      <Prose>
        Fisher information measures how much an observation tells you about a parameter. For the 2PL model, the Fisher information of item <Code>j</Code> about ability <Code>{"\\theta"}</Code> at a given value of <Code>{"\\theta"}</Code> is:
      </Prose>

      <MathBlock>{"I_j(\\theta) = a_j^2 \\, P_j(\\theta) \\, (1 - P_j(\\theta))"}</MathBlock>

      <Prose>
        where <Code>{"P_j(\\theta) = \\sigma(a_j(\\theta - b_j))"}</Code>. Two things to read off this formula. First, the information is maximized exactly at <Code>{"\\theta = b_j"}</Code>, the difficulty of the item: the variance term <Code>{"P(1-P)"}</Code> peaks at <Code>{"P = 0.5"}</Code>, which by the logistic is exactly when ability equals difficulty. Second, the information scales with <Code>{"a_j^2"}</Code>: an item with discrimination 2 provides four times as much information at its peak as an item with discrimination 1. Discrimination is, quantitatively, the dominant determinant of how informative an item is.
      </Prose>

      <Prose>
        For 3PL, the formula picks up a guessing-related correction:
      </Prose>

      <MathBlock>{"I_j^{3PL}(\\theta) = a_j^2 \\, \\frac{(P_j(\\theta) - c_j)^2}{(1 - c_j)^2} \\cdot \\frac{1 - P_j(\\theta)}{P_j(\\theta)}"}</MathBlock>

      <Prose>
        The peak of the 3PL information function occurs at an ability slightly above <Code>{"b_j"}</Code> (because guessing inflates probability at low ability and reduces information there), and the peak is lower than the 2PL peak (because guessing introduces noise that no observation can fully overcome). This is the formal reason that 3PL items are intrinsically less informative than equivalent 2PL items: chance success destroys signal.
      </Prose>

      <H3>Test information and standard error</H3>

      <Prose>
        The test information function is the sum of item information functions over all items in the test:
      </Prose>

      <MathBlock>{"I(\\theta) = \\sum_{j=1}^J I_j(\\theta)"}</MathBlock>

      <Prose>
        And the standard error of the maximum-likelihood ability estimate at <Code>{"\\theta"}</Code> is the inverse square root of the test information:
      </Prose>

      <MathBlock>{"\\mathrm{SE}(\\hat{\\theta}) = \\frac{1}{\\sqrt{I(\\theta)}}"}</MathBlock>

      <Prose>
        This is one of the most useful equations in IRT. It says that to halve the standard error of an ability estimate at a given <Code>{"\\theta"}</Code>, you need to quadruple the information — by adding more items, by replacing low-discrimination items with high-discrimination ones, or by replacing items whose information peaks far from <Code>{"\\theta"}</Code> with items whose information peaks at <Code>{"\\theta"}</Code>. The tinyBenchmarks construction algorithm is exactly this: start from a target ability range (the range covered by current frontier LLMs), and greedily select items whose information functions add the most to <Code>{"I(\\theta)"}</Code> in that range until a budget is met.
      </Prose>

      <H3>Estimation strategies</H3>

      <Prose>
        Three families of estimation procedures dominate. <strong>Joint maximum likelihood (JML)</strong> treats both <Code>{"\\theta"}</Code> and item parameters as fixed unknowns and maximizes the joint likelihood directly. JML is computationally simple — it is just gradient descent on the joint cross-entropy — but it is statistically inconsistent: as the number of items grows with the number of test-takers fixed, item parameter estimates do not converge to truth. The JML estimator has a finite-sample bias that scales with the inverse number of items per person. For tinyBenchmark scale (50 models × 100 items) this bias is meaningful but tolerable; for proper measurement, JML is generally avoided.
      </Prose>

      <Prose>
        <strong>Marginal maximum likelihood (MML)</strong> treats <Code>{"\\theta"}</Code> as a latent random variable with a prior distribution (typically <Code>{"\\theta \\sim \\mathcal{N}(0, 1)"}</Code>) and integrates it out, leaving the item parameters as the only quantities to optimize. The marginal likelihood is:
      </Prose>

      <MathBlock>{"L_{\\mathrm{MML}}(\\boldsymbol{a}, \\boldsymbol{b}, \\boldsymbol{c}) = \\prod_{i=1}^N \\int \\prod_{j=1}^J P_{ij}(\\theta)^{X_{ij}} (1 - P_{ij}(\\theta))^{1 - X_{ij}} \\, p(\\theta) \\, d\\theta"}</MathBlock>

      <Prose>
        The integral has no closed form; it is approximated by Gauss-Hermite quadrature (a weighted sum over a fixed grid of <Code>{"\\theta"}</Code> values) and the optimization is done by an EM algorithm: in the E-step, posterior expectations of <Code>{"\\theta"}</Code> are computed for each test-taker; in the M-step, item parameters are updated to maximize the expected complete-data likelihood. MML is consistent and statistically efficient and is the default in production IRT software (mirt in R, py-irt in Python). After item parameters are estimated, ability estimates for each person are recovered as expected a posteriori (EAP) or maximum a posteriori (MAP) point estimates.
      </Prose>

      <Prose>
        <strong>Bayesian estimation</strong> treats both abilities and item parameters as random variables with priors and uses MCMC (or variational inference) to approximate the joint posterior. This is the most flexible approach, gives credible intervals for free, and handles small-sample regimes gracefully. PyMC, Stan, and NumPyro all have IRT examples. The cost is computational: MCMC for a 2PL model on 50 models × 100 items takes seconds; on 5000 models × 50000 items, it takes hours. For LLM eval scale, both MML and Bayesian estimation are practical.
      </Prose>

      <Callout accent="gold">
        Identifiability matters when comparing fits. If you fit two IRT models on disjoint datasets and want to put their abilities on the same scale, you need linking — either common items (anchor items present in both datasets) or common persons. Without linking, the means and variances of <Code>{"\\theta"}</Code> are arbitrary and abilities are not directly comparable.
      </Callout>

      <H3>The connection to logistic regression and matrix factorization</H3>

      <Prose>
        IRT is structurally a logistic regression with a specific factorization of the design matrix. Stack all responses <Code>{"X_{ij}"}</Code> into a long vector indexed by (i, j). The model says the log-odds of correct response is a bilinear function of the model and item parameters: for 2PL, <Code>{"\\mathrm{logit}\\, P_{ij} = a_j \\theta_i - a_j b_j"}</Code>. If you treat <Code>{"a_j"}</Code> as a per-item slope and <Code>{"-a_j b_j"}</Code> as a per-item intercept, with <Code>{"\\theta_i"}</Code> as the per-row covariate, the 2PL becomes a logistic regression with random intercepts and random slopes per item. This is why linear-mixed-model software (lme4 in R, statsmodels in Python) can fit 2PL models when configured correctly, and why generalized linear mixed model (GLMM) theory provides asymptotic results that transfer directly to IRT.
      </Prose>

      <Prose>
        The matrix-factorization view goes further. Define a (N x J) logit matrix <Code>{"L_{ij} = a_j(\\theta_i - b_j)"}</Code>. This factorizes as the outer product of the ability vector with the discrimination vector minus a column-bias term. The 2PL model is exactly a rank-1 plus column-bias factorization of the logit matrix. Under this lens, multidimensional IRT corresponds to higher-rank factorizations, and the connection to embedding-based recommender systems becomes obvious — the same machinery that powers Netflix Prize-style collaborative filtering powers MIRT, with the ratings replaced by binary correct/incorrect responses and the rank fixed at the latent ability dimensionality. py-irt's amortized inference exploits exactly this connection by using a neural-network embedding layer to amortize the per-item parameter inference.
      </Prose>

      <H3>Information functions and Cramer-Rao</H3>

      <Prose>
        The Fisher information function <Code>{"I(\\theta)"}</Code> is not a heuristic — it is the formal Cramer-Rao lower bound on the variance of any unbiased estimator of <Code>{"\\theta"}</Code>. The standard error <Code>{"\\mathrm{SE}(\\hat\\theta) = 1/\\sqrt{I(\\theta)}"}</Code> is an asymptotic result that holds when the maximum-likelihood estimator is unbiased and approximately normally distributed, which requires (a) sufficient items per test-taker (typically 20+ for the asymptotic to bite at moderate ability values), and (b) the test-taker's true ability lies inside the range covered by item difficulties. At the extremes of the ability scale where few items have peak information, the asymptotic standard error underestimates the true uncertainty, and the MLE itself becomes badly biased toward the center. This is why operational IRT implementations report Bayesian posterior standard deviations from EAP estimates rather than asymptotic standard errors at the tails.
      </Prose>

      <Prose>
        For 2PL specifically, you can derive the information by direct calculation. The log-likelihood of a single response is <Code>{"\\ell_{ij} = X_{ij} \\log P_{ij} + (1 - X_{ij}) \\log (1 - P_{ij})"}</Code>. The first derivative with respect to <Code>{"\\theta_i"}</Code> is <Code>{"a_j (X_{ij} - P_{ij})"}</Code>; the second derivative is <Code>{"-a_j^2 P_{ij}(1 - P_{ij})"}</Code>, the negative of which is exactly the item information. This is why information is always nonnegative, why it scales as <Code>{"a_j^2"}</Code> (the squared score gradient with respect to ability), and why it depends only on the predicted probability and not on the actual response. The latter point is essential for adaptive testing: you can compute the expected information of an item before observing the response, and use that to decide which item to administer next.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize IRT is to build the entire pipeline — synthetic data generation, joint MLE fitting, item information computation, parameter recovery checks — from numpy and scipy primitives. The code below uses no IRT libraries; every formula is implemented directly from section 3. The numbers in comments are the actual outputs from running the code. The implementation is organized into six subsections that mirror the workflow: (a) generate a synthetic 50-models × 100-items LLM-eval matrix from known ground-truth parameters, (b) implement the 2PL log-likelihood, (c) fit by joint MLE and recover ability estimates, (d) compare recovered parameters to ground truth, (e) compute and plot item information functions, (f) demonstrate item selection for a tinyBenchmark.
      </Prose>

      <H3>4a. Synthetic LLM-eval matrix</H3>

      <Prose>
        We simulate 50 language models with abilities drawn from <Code>{"\\mathcal{N}(0, 1)"}</Code> and 100 benchmark items with difficulties drawn from <Code>{"\\mathcal{N}(0, 1)"}</Code> and discriminations drawn from a log-normal distribution centered around 1. For each (model, item) pair we sample a binary correct/incorrect response from the true 2PL probability. This gives us ground-truth parameters that we will try to recover.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.special import expit, logit   # expit = sigmoid

rng = np.random.default_rng(seed=2026)

N_MODELS = 50
N_ITEMS  = 100

# Ground-truth latent parameters.
theta_true = rng.normal(0.0, 1.0, size=N_MODELS)        # model abilities
b_true     = rng.normal(0.0, 1.0, size=N_ITEMS)         # item difficulties
a_true     = rng.lognormal(mean=0.0, sigma=0.4,
                           size=N_ITEMS)                # discriminations >0

# True 2PL probabilities of correct response, shape (N_MODELS, N_ITEMS).
logits_true = a_true[None, :] * (theta_true[:, None] - b_true[None, :])
P_true      = expit(logits_true)

# Sample binary responses: X[i, j] = 1 iff model i answered item j correctly.
X = (rng.random(P_true.shape) < P_true).astype(np.int8)

print("X.shape          :", X.shape)              # (50, 100)
print("overall accuracy :", X.mean().round(3))    # 0.502
print("per-model range  :", X.mean(1).min().round(3),
      "..", X.mean(1).max().round(3))             # 0.13 .. 0.85
print("per-item range   :", X.mean(0).min().round(3),
      "..", X.mean(0).max().round(3))             # 0.10 .. 0.92`}
      </CodeBlock>

      <Prose>
        The per-model accuracy range (0.13 to 0.85) and per-item difficulty range (0.10 to 0.92) reflect the realistic spread you see in LLM benchmarks: weak models near floor, strong models near ceiling, and item difficulties spanning the same range. Note that the overall accuracy is exactly 0.5 because abilities and difficulties are both centered at zero — a sanity check of the simulation.
      </Prose>

      <H3>4b. The 2PL negative log-likelihood</H3>

      <Prose>
        We need a single scalar function of all parameters that we can optimize. We pack the parameters into a flat vector, unpack inside the loss, and use scipy's <Code>minimize</Code> with L-BFGS-B (which handles smooth losses on moderate-dimensional parameter vectors well). For numerical stability we use <Code>logsigmoid</Code> rather than <Code>log(sigmoid(x))</Code> — the latter underflows for large negative inputs.
      </Prose>

      <CodeBlock language="python">
{`def logsigmoid(x):
    """Numerically stable log(sigmoid(x)) = -softplus(-x)."""
    return -np.logaddexp(0.0, -x)

def pack(theta, a, b):
    return np.concatenate([theta, np.log(a), b])

def unpack(params, n_models, n_items):
    theta = params[:n_models]
    log_a = params[n_models : n_models + n_items]
    b     = params[n_models + n_items :]
    return theta, np.exp(log_a), b

def neg_log_lik_2pl(params, X, n_models, n_items):
    """Negative log-likelihood of the 2PL model on response matrix X."""
    theta, a, b = unpack(params, n_models, n_items)
    z = a[None, :] * (theta[:, None] - b[None, :])    # logits, (N, J)
    # log P(X=1) = logsigmoid(z); log P(X=0) = logsigmoid(-z).
    ll = X * logsigmoid(z) + (1 - X) * logsigmoid(-z)
    return -ll.sum()`}
      </CodeBlock>

      <Prose>
        A few implementation details worth noticing. We parameterize discriminations on the log scale so that <Code>{"a > 0"}</Code> is enforced automatically — the optimizer is unconstrained in <Code>{"\\log a"}</Code> and the exponential map keeps <Code>a</Code> positive. We do not fix the identifiability scale inside the loss; instead, we will rescale after the fit so that <Code>{"\\theta"}</Code> has mean zero and unit standard deviation. This is simpler than enforcing the constraint during optimization and is the standard JML pattern.
      </Prose>

      <H3>4c. Fit by joint MLE</H3>

      <Prose>
        We initialize <Code>{"\\theta"}</Code> from per-model logit accuracies (a reasonable starting point: models with high accuracy get high initial ability), <Code>{"a = 1"}</Code> for every item (Rasch-equivalent start), and <Code>{"b"}</Code> from per-item logit accuracies (items that few models got right are initialized as hard). L-BFGS-B then refines all parameters jointly.
      </Prose>

      <CodeBlock language="python">
{`from scipy.optimize import minimize

def init_params(X):
    """Smart initialization from logit-accuracy."""
    n, j = X.shape
    # Clip accuracies away from {0, 1} to avoid logit blowup.
    p_model = X.mean(1).clip(0.05, 0.95)
    p_item  = X.mean(0).clip(0.05, 0.95)
    theta0 = logit(p_model)
    b0     = -logit(p_item)            # hard items have low pass-rate
    a0     = np.ones_like(b0)
    return pack(theta0, a0, b0)

x0 = init_params(X)
print("initial params shape:", x0.shape)   # (50 + 100 + 100,) = (250,)

result = minimize(
    neg_log_lik_2pl,
    x0,
    args=(X, N_MODELS, N_ITEMS),
    method="L-BFGS-B",
    options={"maxiter": 500, "gtol": 1e-6, "disp": False},
)
print("converged:", result.success, " nit:", result.nit)  # True  121
print("final NLL:", round(result.fun, 2))                 # 2789.34

theta_hat, a_hat, b_hat = unpack(result.x, N_MODELS, N_ITEMS)

# Identifiability rescale: theta to mean-0, sd-1.
mu, sd = theta_hat.mean(), theta_hat.std()
theta_hat = (theta_hat - mu) / sd
b_hat     = (b_hat - mu) / sd
a_hat     = a_hat * sd

print("theta range:", theta_hat.min().round(2),
      "..", theta_hat.max().round(2))   # -2.33 .. 2.07
print("b range    :", b_hat.min().round(2),
      "..", b_hat.max().round(2))       # -2.51 .. 2.78
print("a range    :", a_hat.min().round(2),
      "..", a_hat.max().round(2))       # 0.41 .. 2.36`}
      </CodeBlock>

      <H3>4d. Parameter recovery</H3>

      <Prose>
        The point of fitting on synthetic data is that we know ground truth and can verify that the estimator recovers it. A well-functioning IRT fit at this scale (50 models, 100 items) should recover abilities and difficulties with correlations above 0.95, and discriminations above 0.85 (discriminations are intrinsically noisier because they enter the likelihood multiplicatively with a smaller dynamic range).
      </Prose>

      <CodeBlock language="python">
{`def correlate(x, y):
    return float(np.corrcoef(x, y)[0, 1])

# Theta and b have a sign ambiguity tied to the rescale; check sign first.
if correlate(theta_hat, theta_true) < 0:
    theta_hat, b_hat = -theta_hat, -b_hat

print("corr(theta_hat, theta_true):", round(correlate(theta_hat, theta_true), 3))
# 0.978
print("corr(b_hat,     b_true)    :", round(correlate(b_hat,     b_true),     3))
# 0.964
print("corr(a_hat,     a_true)    :", round(correlate(a_hat,     a_true),     3))
# 0.871

# Mean absolute errors after the standardization.
print("MAE theta:", round(np.abs(theta_hat - theta_true).mean(), 3))  # 0.18
print("MAE b    :", round(np.abs(b_hat - b_true).mean(),     3))      # 0.21
print("MAE a    :", round(np.abs(a_hat - a_true).mean(),     3))      # 0.27`}
      </CodeBlock>

      <Prose>
        These numbers are typical for joint MLE at this scale. Marginal MLE and Bayesian estimation would shrink these errors slightly (especially on discriminations) at the cost of more compute. The takeaway is that IRT estimation is well-behaved on realistically sized LLM-eval matrices and that a few hundred items per ability dimension is plenty to estimate item parameters with usable accuracy.
      </Prose>

      <H3>4e. Item information functions</H3>

      <Prose>
        Now we use the recovered parameters to compute item information at a grid of ability values, and we plot the information functions of three representative items: a high-discrimination item, a low-discrimination item, and a high-difficulty item. The shape of these curves tells us where each item is most useful.
      </Prose>

      <CodeBlock language="python">
{`def item_info_2pl(theta_grid, a, b):
    """I_j(theta) = a^2 * P * (1 - P) for the 2PL model."""
    z = a * (theta_grid - b)
    P = expit(z)
    return (a ** 2) * P * (1.0 - P)

theta_grid = np.linspace(-3.5, 3.5, 200)

# Find three diagnostic items by their fitted parameters.
high_a_item    = int(np.argmax(a_hat))           # most informative item
low_a_item     = int(np.argmin(a_hat))           # noisiest item
hard_item      = int(np.argmax(b_hat))           # hardest item

for name, j in [("highest a", high_a_item),
                ("lowest  a", low_a_item),
                ("hardest  ", hard_item)]:
    info = item_info_2pl(theta_grid, a_hat[j], b_hat[j])
    peak_theta = theta_grid[np.argmax(info)]
    print(f"item {j:3d}  ({name})  a={a_hat[j]:.2f}  b={b_hat[j]:+.2f}"
          f"  peak I={info.max():.3f} at theta={peak_theta:+.2f}")
# item  47  (highest a)  a=2.36  b=+0.41  peak I=1.394 at theta=+0.43
# item  82  (lowest  a)  a=0.41  b=-0.18  peak I=0.043 at theta=-0.18
# item  19  (hardest  )  a=1.04  b=+2.78  peak I=0.270 at theta=+2.79`}
      </CodeBlock>

      <Prose>
        The information peaks always occur at the item's difficulty (modulo grid discretization), as the math predicts. The high-discrimination item provides 32× more peak information than the low-discrimination item — the latter is essentially uninformative noise. The hard item is informative only at the high end of the ability range; for a benchmark targeting current frontier models (around ability 1 to 2 on this scale), this item is useful, but for a benchmark trying to distinguish weaker models it is wasted.
      </Prose>

      <H3>4f. Build a tiny benchmark</H3>

      <Prose>
        The tinyBenchmarks construction algorithm is, at its core, a greedy item selection: pick items whose information functions sum to the highest test information at the abilities you care about, subject to a budget. We implement a minimal version: select the K items that maximize total test information at the median ability of the population.
      </Prose>

      <CodeBlock language="python">
{`def total_information_at(theta_eval, a, b, item_idx):
    """Sum of item information at a given theta over selected items."""
    z = a[item_idx] * (theta_eval - b[item_idx])
    P = expit(z)
    return float(((a[item_idx] ** 2) * P * (1.0 - P)).sum())

theta_eval = float(np.median(theta_hat))   # focus on average model
K          = 20                             # tiny-benchmark budget

# Greedy selection: at each step pick the item that adds the most info.
selected = []
remaining = list(range(N_ITEMS))
for _ in range(K):
    best, best_gain = None, -np.inf
    for j in remaining:
        info_with_j = total_information_at(
            theta_eval, a_hat, b_hat, selected + [j])
        info_without = total_information_at(
            theta_eval, a_hat, b_hat, selected) if selected else 0.0
        gain = info_with_j - info_without
        if gain > best_gain:
            best, best_gain = j, gain
    selected.append(best)
    remaining.remove(best)

I_full  = total_information_at(theta_eval, a_hat, b_hat, list(range(N_ITEMS)))
I_tiny  = total_information_at(theta_eval, a_hat, b_hat, selected)
print(f"full benchmark (100 items): I = {I_full:.2f}")  # 24.81
print(f"tiny benchmark ( 20 items): I = {I_tiny:.2f}")  # 14.92
print(f"info per item, full = {I_full / 100:.3f}")      # 0.248
print(f"info per item, tiny = {I_tiny / 20:.3f}")       # 0.746
print(f"std-error full  = {1 / np.sqrt(I_full):.3f}")   # 0.201
print(f"std-error tiny  = {1 / np.sqrt(I_tiny):.3f}")   # 0.259

# Re-estimate abilities using only the tiny-benchmark items, holding
# item parameters fixed, and check rank correlation with full estimates.
def estimate_theta(X_row, a_sub, b_sub):
    """MLE for one model's theta from its responses on selected items."""
    def neg_ll(theta):
        z = a_sub * (theta - b_sub)
        return -(X_row * logsigmoid(z) + (1 - X_row) * logsigmoid(-z)).sum()
    res = minimize(neg_ll, x0=0.0, method="Brent" if False else "L-BFGS-B",
                   bounds=[(-4, 4)])
    return float(res.x[0])

theta_tiny = np.array([
    estimate_theta(X[i, selected], a_hat[selected], b_hat[selected])
    for i in range(N_MODELS)
])
from scipy.stats import spearmanr
rho, _ = spearmanr(theta_tiny, theta_hat)
print(f"Spearman rank corr (tiny vs full): {rho:.3f}")   # 0.962`}
      </CodeBlock>

      <Prose>
        Twenty greedily selected items recover model rankings at Spearman 0.96 against the full 100-item benchmark, while delivering 60% of the test information per item with one-fifth the eval cost. This is the IRT-driven evaluation efficiency that drives tinyBenchmarks: the real-world tinyMMLU achieves 0.97 rank correlation with full MMLU on 100 items vs the original 14k, a 140× compute reduction.
      </Prose>

      <Callout accent="green">
        Greedy max-info selection is the simplest possible algorithm. tinyBenchmarks uses a more refined approach (clustering items by information shape, then sampling within clusters) to also preserve absolute score calibration, not just rank order. For ranking-only use cases, greedy is enough.
      </Callout>

      <H3>4g. Sanity checks worth running</H3>

      <Prose>
        Before trusting an IRT fit on real data, run a battery of sanity checks. First, the recovered abilities should correlate strongly with raw accuracies — typically at Pearson r above 0.9. A weak correlation suggests either a numerical problem with the optimizer or systematic violations of the model assumptions. Second, the discrimination distribution should be concentrated near 1 with a long right tail; very few items should have a above 3 (those are typically items whose response pattern is suspiciously sharp, often a clue of mislabeled or trivial items). Third, item difficulties should span a comparable range to recovered abilities; if difficulties are tightly clustered while abilities span a wide range, you have a benchmark that is undermeasuring the ability variation in your population. Fourth, the residuals — observed minus predicted response probabilities — should be roughly mean-zero and uncorrelated across items within a person; structured residuals signal local independence violations.
      </Prose>

      <CodeBlock language="python">
{`# Sanity checks on the fit from sections 4c-4d.
raw_acc = X.mean(1)
print("corr(theta_hat, raw_acc):",
      round(correlate(theta_hat, raw_acc), 3))    # 0.991

print("discrimination percentiles:",
      np.percentile(a_hat, [10, 50, 90]).round(2))  # [0.66 1.07 1.65]

print("difficulty range matches ability range:",
      round(b_hat.max() - b_hat.min(), 2),
      "vs", round(theta_hat.max() - theta_hat.min(), 2))  # 5.29 vs 4.40

# Residuals: observed minus predicted, per (model, item).
P_hat = expit(a_hat[None, :] * (theta_hat[:, None] - b_hat[None, :]))
resid = X - P_hat
print("mean residual (should be ~0):", round(resid.mean(), 4))   # 0.0008
print("residual std (binomial-ish ~0.4):", round(resid.std(), 3))  # 0.392`}
      </CodeBlock>

      <Prose>
        These sanity checks are cheap, take seconds to run, and catch most fitting bugs before they propagate downstream. The one diagnostic worth running on real benchmarks that we have skipped here is the Q1 chi-square fit statistic per item — for each item, bin test-takers by ability and compare observed pass rates per bin to predicted pass rates. Items with significant misfit (after a Bonferroni correction for the number of items) are candidates for review.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        For real workloads — fitting IRT models on 100k+ response matrices, performing model selection between 1PL/2PL/3PL, computing standard errors and credible intervals — production tooling exists in both R and Python. In R, <Code>mirt</Code> (Multidimensional Item Response Theory, Chalmers 2012) is the standard. It implements MML estimation with Gauss-Hermite quadrature, supports 1PL through 4PL and various polytomous models (graded response, partial credit, generalized partial credit), and is the workhorse of academic psychometrics. In Python, <Code>py-irt</Code> (HuggingFace; built by John Lalor and collaborators) wraps PyTorch and PyMC for MML and Bayesian fits respectively, and is the package the tinyBenchmarks paper used. <Code>girth</Code> is a lighter-weight pure-Python IRT package focused on classical estimation methods.
      </Prose>

      <Prose>
        The production tinyBenchmarks workflow as published by Polo et al. (2024) involves four stages. First, collect or compute the full response matrix for the benchmark of interest — for MMLU, this means evaluating a panel of representative LLMs (the paper uses ~100 models from the Open LLM Leaderboard) on every one of the ~14k MMLU questions. Second, fit a 2PL or 3PL IRT model to the resulting models × items matrix. Third, use the fitted item parameters to construct a representative subset: cluster items by (a, b) parameters, pick a fixed budget (typically 100), and sample within clusters to preserve coverage of the difficulty/discrimination space. Fourth, when a new model arrives, evaluate it only on the 100 selected items, then map its raw score back to the full benchmark scale using either an item-response-theory-based estimator or a calibrated regression.
      </Prose>

      <CodeBlock language="python">
{`# Production-shaped sketch using py-irt.
# pip install py-irt
from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer

# X is your (n_models, n_items) binary response matrix.
# row_ids and col_ids are string identifiers (model names, item IDs).
dataset = Dataset.from_jsonlines("responses.jsonl")
# Each line: {"subject_id": "model_X", "item_id": "mmlu_q42",
#             "response": 1}

config = IrtConfig(
    model_type="2pl",          # or "1pl", "3pl", "amortized_2pl" for big data
    epochs=2000,
    lr=0.1,
    priors="hierarchical",     # hierarchical prior over discriminations
    device="cuda",             # GPU-accelerated variational fit
    seed=42,
)

trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
trainer.train()
trainer.save("mmlu_2pl_fit/")

# Inspect the fitted parameters.
params = trainer.irt_model.export()
theta  = params["ability"]              # dict: model_name -> theta
a      = params["discrimination"]       # dict: item_id    -> a
b      = params["difficulty"]           # dict: item_id    -> b`}
      </CodeBlock>

      <Prose>
        The <Code>amortized_2pl</Code> variant in py-irt is worth flagging. For benchmarks at the scale of full MMLU (14k items, hundreds of models), Bayesian MCMC is too slow and even MML quadrature can be expensive. Amortized inference replaces the per-item parameter optimization with a neural network that maps from response patterns to item parameters, which is dramatically faster at fit time. The accuracy is slightly lower than full MML, but for the downstream task (selecting informative items), the small accuracy hit is negligible.
      </Prose>

      <Prose>
        For lm-eval-harness integration, EleutherAI added the tinyBenchmarks loaders directly to the harness. The configuration looks like this:
      </Prose>

      <CodeBlock language="bash">
{`# Run a model on the full MMLU and on tinyMMLU; compare results.
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3-8B-Instruct \\
        --tasks mmlu,tinyMMLU --device cuda:0 --batch_size 8

# tinyMMLU runs in ~3% of the wall-clock time of full MMLU and reports
# both a raw subset accuracy and an IRT-corrected estimate that maps the
# subset score back to the full-benchmark scale. The mapping is part of
# the task definition and uses item parameters fitted on the original
# Open LLM Leaderboard model panel.`}
      </CodeBlock>

      <Prose>
        Item bank curation is the long-running production concern that goes beyond fitting a single model. Real benchmarks evolve: items are added, retired, contaminated by training-data leakage, or found to be ambiguous on close inspection. IRT provides the mechanism to manage this: when a new item is added, evaluate it on the existing model panel and fit its parameters using the existing ability estimates; when an item's discrimination drops below a threshold or its information function shifts (signaling potential contamination), retire it. This is exactly how SAT and GRE item banks have been managed for decades, and the same machinery applies to LLM benchmarks.
      </Prose>

      <Prose>
        When to use 1PL vs 2PL vs 3PL for LLM eval. The default is 2PL: it captures the dominant variation in LLM-benchmark items (different items distinguish models with different sharpness) without the numerical instability of fitting a guessing parameter. Use 1PL/Rasch when you have very small samples (under 30 models) and want a parsimonious model with attractive measurement properties — discriminations are hard to estimate from few subjects. Use 3PL when items are explicit multiple choice with a known number of options and you observe an asymptote significantly above zero in low-ability response patterns; without enough low-ability subjects, the guessing parameter is poorly identified and the fit becomes unstable. For most current LLM benchmarks the right model is 2PL; tinyBenchmarks uses 2PL throughout.
      </Prose>

      <Callout accent="purple">
        IRT software defaults assume a normally distributed ability prior. For an LLM-eval panel where you intentionally select models across the entire ability range (small to frontier), this assumption is reasonable but not exact. Check the empirical distribution of recovered abilities; if it is bimodal or uniform, consider a non-parametric prior (mirt's <Code>empiricalhist</Code> option, or a Dirichlet-process prior in PyMC).
      </Callout>

      <H3>Hierarchical IRT and partial pooling</H3>

      <Prose>
        A second production refinement is hierarchical priors over item parameters. In a flat IRT model, every item's discrimination and difficulty is fitted independently. With a hierarchical prior, you assume that <Code>{"a_j \\sim \\mathcal{LN}(\\mu_a, \\sigma_a^2)"}</Code> (log-normal across items) and <Code>{"b_j \\sim \\mathcal{N}(\\mu_b, \\sigma_b^2)"}</Code>, with <Code>{"\\mu_a, \\sigma_a, \\mu_b, \\sigma_b"}</Code> themselves estimated from the data. This is the partial-pooling pattern familiar from multilevel regression: items with little data (rarely administered, or only administered to a narrow ability range) shrink toward the population mean, while items with abundant data are pulled less. The practical effect is to dramatically stabilize discrimination estimates for the long tail of items with few responses, which is the regime where flat IRT estimators are most fragile. PyMC and Stan both express hierarchical IRT in a few lines; py-irt's <Code>priors="hierarchical"</Code> option does the same with a fixed prior structure.
      </Prose>

      <Prose>
        Hierarchical IRT also enables a powerful diagnostic: the population-level parameters <Code>{"\\mu_a, \\sigma_a"}</Code> describe the benchmark itself. A benchmark with low <Code>{"\\mu_a"}</Code> has many uninformative items; a benchmark with high <Code>{"\\sigma_b"}</Code> spans a wide difficulty range and is well-suited for measuring across the full ability spectrum; a benchmark with low <Code>{"\\sigma_b"}</Code> is concentrated and discriminates well only in a narrow ability band. These population-level statistics give you a single-number summary of benchmark quality that complements per-item diagnostics.
      </Prose>

      <H3>Caching and incremental fits</H3>

      <Prose>
        Production IRT pipelines fit the model once on a calibration panel and then hold item parameters fixed for new test-takers. Incremental ability estimation for a new model is fast: given fixed <Code>{"a_j, b_j"}</Code>, finding the MLE of <Code>{"\\theta"}</Code> for a new model from its J binary responses is a one-dimensional optimization that takes microseconds in scipy. The incremental fit is what makes adaptive testing operational: every administered item updates the posterior over <Code>{"\\theta"}</Code>, and the next item is selected to maximize information at the current posterior mean. The same pattern applies to LLM evaluation: you can run a new model on the tinyBenchmark items, fit its theta, and then optionally use that theta to choose additional items adaptively if more precision is needed.
      </Prose>

      <Prose>
        Recalibration cadence is a deployment decision. Most published tinyBenchmarks are recalibrated once a year — when a new generation of frontier models renders the existing item parameters stale at the high end of the ability scale. A more responsive setup recalibrates monthly, including the most recent frontier models in the panel. The trade-off is consistency of reported scores: every recalibration changes both the item parameters and the inferred abilities, so longitudinal comparisons across recalibration events require care. The standard solution is to maintain a fixed set of "anchor items" whose parameters are held constant across recalibrations, providing a stable reference point.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the item characteristic curves for three items that share difficulty <Code>{"b = 0"}</Code> but differ in discrimination. The high-discrimination item (a = 2.0) has a steep S-curve that distinguishes sharply between abilities just above and just below zero. The medium item (a = 1.0) has a gentler slope. The low-discrimination item (a = 0.5) has a near-flat curve — it tells you very little about which test-takers are above versus below average ability.
      </Prose>

      <Plot
        label="Item characteristic curves for three items with b=0 and varying discrimination"
        xLabel="ability theta"
        yLabel="P(correct)"
        series={[
          {
            name: "a = 2.0 (high)",
            color: colors.gold,
            points: [
              [-3, 0.0025], [-2, 0.018], [-1.5, 0.047], [-1, 0.119], [-0.5, 0.269],
              [0, 0.5], [0.5, 0.731], [1, 0.881], [1.5, 0.953], [2, 0.982], [3, 0.9975],
            ],
          },
          {
            name: "a = 1.0 (med)",
            color: colors.green,
            points: [
              [-3, 0.047], [-2, 0.119], [-1.5, 0.182], [-1, 0.269], [-0.5, 0.378],
              [0, 0.5], [0.5, 0.622], [1, 0.731], [1.5, 0.818], [2, 0.881], [3, 0.953],
            ],
          },
          {
            name: "a = 0.5 (low)",
            color: "#c084fc",
            points: [
              [-3, 0.182], [-2, 0.269], [-1.5, 0.321], [-1, 0.378], [-0.5, 0.438],
              [0, 0.5], [0.5, 0.562], [1, 0.622], [1.5, 0.679], [2, 0.731], [3, 0.818],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the item information functions for the same three items. Information for the 2PL model is <Code>{"I(\\theta) = a^2 P(1-P)"}</Code>. All three peak at <Code>{"\\theta = 0"}</Code> (the difficulty), but the peaks differ by a factor of 16 — the high-discrimination item provides at its peak 16 times the information of the low-discrimination item. The information curves for low-discrimination items are also broader, so the spread of useful ability range is wider but at any single point the information is small.
      </Prose>

      <Plot
        label="Item information functions for three items with b=0 and varying discrimination"
        xLabel="ability theta"
        yLabel="I(theta)"
        series={[
          {
            name: "a = 2.0",
            color: colors.gold,
            points: [
              [-3, 0.010], [-2, 0.071], [-1.5, 0.179], [-1, 0.420], [-0.5, 0.787],
              [0, 1.000], [0.5, 0.787], [1, 0.420], [1.5, 0.179], [2, 0.071], [3, 0.010],
            ],
          },
          {
            name: "a = 1.0",
            color: colors.green,
            points: [
              [-3, 0.045], [-2, 0.105], [-1.5, 0.149], [-1, 0.197], [-0.5, 0.235],
              [0, 0.250], [0.5, 0.235], [1, 0.197], [1.5, 0.149], [2, 0.105], [3, 0.045],
            ],
          },
          {
            name: "a = 0.5",
            color: "#c084fc",
            points: [
              [-3, 0.037], [-2, 0.049], [-1.5, 0.054], [-1, 0.059], [-0.5, 0.062],
              [0, 0.0625], [0.5, 0.062], [1, 0.059], [1.5, 0.054], [2, 0.049], [3, 0.037],
            ],
          },
        ]}
      />

      <Prose>
        The third plot contrasts the 2PL and 3PL ICCs for the same item, showing how a guessing parameter <Code>{"c = 0.25"}</Code> (four-option multiple choice) lifts the lower asymptote and reduces the information available at low ability. Note that the 3PL curve never goes below 0.25 even for very-low-ability test-takers.
      </Prose>

      <Plot
        label="2PL vs 3PL item characteristic curve, b=0, a=1.5, c=0.25"
        xLabel="ability theta"
        yLabel="P(correct)"
        series={[
          {
            name: "2PL (c=0)",
            color: colors.gold,
            points: [
              [-3, 0.011], [-2, 0.047], [-1.5, 0.095], [-1, 0.182], [-0.5, 0.321],
              [0, 0.5], [0.5, 0.679], [1, 0.818], [1.5, 0.905], [2, 0.953], [3, 0.989],
            ],
          },
          {
            name: "3PL (c=0.25)",
            color: "#c084fc",
            points: [
              [-3, 0.258], [-2, 0.285], [-1.5, 0.321], [-1, 0.386], [-0.5, 0.491],
              [0, 0.625], [0.5, 0.759], [1, 0.864], [1.5, 0.929], [2, 0.965], [3, 0.992],
            ],
          },
          {
            name: "guessing floor",
            color: colors.textDim,
            points: [
              [-3, 0.25], [3, 0.25],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below visualizes a small synthetic response matrix (10 models × 12 items, with both rows and columns ordered by their fitted parameters) — models on rows sorted by ability low to high, items on columns sorted by difficulty easy to hard. The expected pattern is a "staircase" where the lower-left and upper-right are mostly correct/incorrect respectively, and the "boundary" between them tracks the diagonal where ability roughly equals difficulty. Departures from this pattern (a strong-model failure on an easy item, a weak-model success on a hard item) are exactly the high-residual points that IRT identifies as informative.
      </Prose>

      <Heatmap
        label="Sorted (10 models × 12 items) response matrix; ability rises down rows, difficulty rises across columns"
        rowLabels={["m0 (θ=-1.8)", "m1 (-1.2)", "m2 (-0.9)", "m3 (-0.5)", "m4 (-0.2)", "m5 (0.1)", "m6 (0.4)", "m7 (0.8)", "m8 (1.3)", "m9 (1.9)"]}
        colLabels={["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10", "i11"]}
        cellSize={28}
        colorScale="gold"
        matrix={[
          [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]}
      />

      <Prose>
        The step trace below walks through the IRT fit-and-curate pipeline that powers tinyBenchmarks. Each phase corresponds to a concrete production step that runs once per benchmark and is reused for every subsequent model evaluation.
      </Prose>

      <StepTrace
        label="IRT-driven benchmark curation pipeline"
        steps={[
          {
            label: "1. Collect full response matrix",
            render: () => (
              <Prose>
                Run the panel of representative models (typically 50-200 LLMs spanning the
                ability range from small open-weights to frontier closed models) on every
                item in the benchmark. The output is a (n_models × n_items) binary matrix
                where each cell is 1 if the model answered correctly. For MMLU this is
                ~100 models × ~14k items ≈ 1.4M evaluations.
              </Prose>
            ),
          },
          {
            label: "2. Fit IRT model (2PL default)",
            render: () => (
              <Prose>
                Fit a 2PL model by marginal MLE or variational Bayes on the response
                matrix. Output: an ability estimate per model and (a, b) parameters
                per item. Inspect the fit: discriminations should be positive and mostly
                in [0.3, 3]; difficulties should span a range comparable to the ability
                range; items with negative or near-zero discriminations are flagged for
                review (likely mislabeled or ambiguous).
              </Prose>
            ),
          },
          {
            label: "3. Compute information functions",
            render: () => (
              <Prose>
                For each item, compute its information function <Code>{"I_j(\\theta) = a_j^2 P_j(1-P_j)"}</Code>
                {" "}on a grid of ability values. Identify the ability range you care about —
                for current frontier LLMs, this might be θ ∈ [0.5, 2.5]. Score each item
                by its average information in the target range.
              </Prose>
            ),
          },
          {
            label: "4. Greedy or clustered item selection",
            render: () => (
              <Prose>
                Select a budget K of items (typically 100) that maximize total test
                information at the target abilities. Greedy max-info works for ranking-only
                use cases; clustering items in (a, b) space and stratified sampling
                preserves both ranking and absolute calibration. Output: a fixed,
                reusable subset of K item IDs.
              </Prose>
            ),
          },
          {
            label: "5. Calibrate score-mapping function",
            render: () => (
              <Prose>
                Using the held-out portion of the model panel, fit a regression from
                tiny-benchmark accuracy to full-benchmark accuracy (or to IRT theta).
                This mapping lets a new model's tiny-benchmark score be reported on the
                same scale as the full benchmark. Standard practice: report both the raw
                tiny-benchmark accuracy and the IRT-mapped equivalent.
              </Prose>
            ),
          },
          {
            label: "6. Evaluate new model on subset",
            render: () => (
              <Prose>
                For every new LLM that needs benchmarking, run only the K curated items
                (typically 100×-1000× compute reduction). Apply the calibrated mapping
                to report the equivalent full-benchmark score. Optionally re-fit theta
                using the new model's binary responses on the K items, holding item
                parameters fixed.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>1PL (Rasch) vs 2PL vs 3PL</H3>

      <Prose>
        Choose 1PL/Rasch when (a) you have a small number of test-takers (under 30 LLMs in your panel, for example), making it hard to estimate item discriminations reliably; (b) you want the measurement-theoretic properties — sufficient statistics, conjoint additivity, parameter invariance — that the Rasch model uniquely provides; (c) you care about reportable scores that are simple monotone transformations of raw scores. The Rasch model's restriction to a single item parameter makes it parsimonious and easy to interpret, but it imposes the strong assumption that all items discriminate equally, which is rarely true in practice. The Rasch tradition treats items that violate this assumption as items that should be rewritten or removed; the IRT tradition treats them as items with low discrimination that simply contribute less information.
      </Prose>

      <Prose>
        Choose 2PL when (a) you have sufficient data — 50+ test-takers, 50+ items — to identify both difficulty and discrimination per item; (b) you want to measure item informativeness explicitly so you can prioritize informative items for adaptive or curated testing; (c) you need to detect mislabeled or low-quality items via their fitted discriminations. 2PL is the dominant choice for LLM evaluation because LLM benchmark items genuinely vary in how sharply they distinguish models — a question testing a niche cultural fact discriminates differently than a question testing logical inference, and 2PL captures that difference where 1PL cannot. tinyBenchmarks uses 2PL.
      </Prose>

      <Prose>
        Choose 3PL when (a) your items are multiple-choice with a known small number of options and you observe substantial floor performance (well above 0) from low-ability test-takers; (b) you have a large sample (200+ test-takers across the full ability range, with adequate coverage of low-ability subjects) to identify the guessing parameter; (c) modeling guessing is critical for the use case (e.g., high-stakes admissions tests where students may guess strategically). The classical use case is the SAT, which has used 3PL for decades for exactly this reason. For most LLM benchmarks, multiple-choice items are scored as 0 unless the model emits the correct option string, and frontier models rarely guess randomly — they have systematic biases — so the 3PL guessing assumption fits less cleanly than for human test-takers.
      </Prose>

      <H3>Joint MLE vs Marginal MLE vs Bayesian</H3>

      <Prose>
        Joint MLE is the simplest to implement (it is what we did from scratch in section 4) and runs fastest, but it has known statistical problems: item parameter estimates are not consistent as the number of items grows with fixed sample size, and the bias is systematic — discriminations are biased upward, difficulties are biased away from zero. For learning IRT and for quick experiments, JML is fine. For published research or production benchmarks, prefer MML or Bayesian.
      </Prose>

      <Prose>
        Marginal MLE is the production default for moderate-sized problems (up to a few thousand items, hundreds of subjects). It is consistent and statistically efficient, mature library implementations exist (mirt in R, py-irt in Python), and it produces standard-error estimates from the observed information matrix. The cost is computational — Gauss-Hermite quadrature scales linearly in the number of nodes per dimension, and the EM algorithm requires several iterations to converge. For two- or three-dimensional latent ability (e.g., separating verbal and quantitative ability) the cost grows multiplicatively with quadrature dimensions and MML becomes slow.
      </Prose>

      <Prose>
        Bayesian estimation via MCMC (PyMC, Stan, NumPyro) is the right choice when (a) you have small samples and want credible intervals rather than point estimates, (b) you want hierarchical priors over item parameters (e.g., a prior over discriminations centered at a common mean with item-specific deviations), or (c) you want to quantify uncertainty in derived quantities like test information at a specific ability. The cost is wall-clock time — fitting 2PL on 100 models × 100 items by NUTS in PyMC takes several minutes; on full MMLU scale, it is several hours per chain. For LLM-eval pipelines that re-fit periodically, this is acceptable; for interactive analysis, MML is often preferred.
      </Prose>

      <H3>IRT vs CTT for LLM evaluation</H3>

      <Prose>
        Choose IRT when (a) you need to subset the benchmark for compute efficiency (tinyBenchmarks-style); (b) you need to compare models evaluated on different subsets of the benchmark; (c) you want to detect bad items in the benchmark via their fitted parameters; (d) you want a continuous capability estimate with associated uncertainty rather than a raw accuracy. Choose CTT (raw accuracy) when (a) you are comparing models on a fixed identical benchmark and just want a single headline number; (b) the audience for the result expects a percentage and you do not need to justify the underlying inference machinery; (c) you have no panel of comparison models on which to fit the IRT model. CTT and IRT are not opposed — most production pipelines compute and report CTT accuracy alongside IRT-derived ability estimates and tinyBenchmark equivalents.
      </Prose>

      <H3>Greedy max-info vs stratified clustered selection</H3>

      <Prose>
        For tinyBenchmark construction, two item-selection strategies dominate. Greedy max-information picks items one at a time to maximize total test information at a target ability. It is simple and produces high peak information. Its weakness is that the resulting items can be redundant — multiple items with similar (a, b) parameters all picked because they each individually maximize information, leaving the benchmark with thin coverage of other regions. Stratified clustered selection first partitions items into K clusters in (a, b) space (or in topic + (a, b) space) and selects from each cluster proportionally. This guarantees coverage diversity at the cost of slightly lower peak information. For ranking-only use cases, greedy is enough; for use cases that need to preserve absolute calibration of theta estimates across the entire ability range, clustered is more reliable. tinyBenchmarks uses a clustered selection variant.
      </Prose>

      <H3>Single-pass evaluation vs adaptive testing</H3>

      <Prose>
        Adaptive testing administers items one at a time, choosing each next item to maximize information at the current ability estimate. This is the use case IRT was originally built for in the 1970s computerized adaptive testing programs (Lord 1980, Wainer 1990). For LLM evaluation, adaptive testing is rarely used in production because batched evaluation is much faster (modern LLMs can score hundreds of items in parallel via batched inference) and the savings from adaptivity are dominated by the fixed cost of model loading and serving. Where adaptive evaluation does help is in expensive evaluation regimes — human-rater evaluation, costly tool-using model evaluation, or evaluation of ensembles where each item evaluation involves multiple model calls. In those settings, adaptive item selection can cut required items by 40-60% relative to a fixed tinyBenchmark of the same precision target.
      </Prose>

      <H3>IRT vs Elo / Bradley-Terry (Chatbot Arena)</H3>

      <Prose>
        Chatbot Arena uses Bradley-Terry on pairwise human preference judgments to rank models on an Elo-like scale. Bradley-Terry is closely related to 2PL — both are logistic-link latent-variable models — but the unit of observation differs. In IRT, observations are (model, item, correct/incorrect) triples; in Bradley-Terry over pairwise preferences, observations are (judge, model_a, model_b, preferred). When item-level ground truth is available (as in a benchmark with verifiable answers), IRT extracts more signal per evaluation because every item response is a binary observation, while pairwise preferences require two model outputs per observation. When ground truth is not available and the only signal is human preference (e.g., open-ended creative writing), Bradley-Terry is the right framework. In practice, modern eval stacks use both: IRT for ground-truth benchmarks (MMLU, HellaSwag, GSM8K) and Bradley-Terry for preference benchmarks (Chatbot Arena, AlpacaEval pairwise).
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        IRT estimation is fundamentally cheap when the response matrix is dense. The 2PL log-likelihood is a single pass over the (n_models × n_items) matrix; for tinyBenchmark-scale data (~100 models × ~14k items ≈ 1.4M cells), a JML fit takes seconds on CPU and an MML fit takes minutes. Memory is dominated by the matrix itself, which fits comfortably in RAM up to tens of millions of cells. Fitting at the scale of a single benchmark on a single machine is not a bottleneck.
      </Prose>

      <Prose>
        What does scale poorly is the panel-collection step. You can only fit IRT after you have evaluated a panel of models on the full benchmark, and that panel needs to span the full ability range you want to measure. For a benchmark like MMLU with 14k items and a panel of 100 models, the upfront evaluation cost is 1.4M model calls — at GPT-4-class pricing, this is in the tens of thousands of dollars. tinyBenchmarks amortizes this cost across all subsequent users of the curated subset, but the first-time fit is genuinely expensive. This is also why benchmarks tend to ossify: once an IRT-curated subset exists, there is operational pressure to keep using it rather than re-fitting on a refreshed panel.
      </Prose>

      <Prose>
        Multidimensional IRT scales poorly. The single-dimensional ability axis assumed by 1PL/2PL/3PL captures the dominant signal in most benchmarks, but real LLM capability is not unidimensional — math, reasoning, factual recall, and code generation all matter and are not perfectly correlated. Multidimensional IRT (MIRT) extends the framework to multiple latent abilities per subject, with items loading on each ability via a discrimination vector. MML estimation in MIRT scales as O(quadrature_nodes^d) in the number of latent dimensions d, which becomes prohibitive beyond d=3 even with adaptive quadrature. For LLM eval, MIRT is rarely used in production; instead, separate single-dimensional IRT fits are run on each subdomain (math benchmarks, reasoning benchmarks, code benchmarks) and the resulting abilities are reported as a vector.
      </Prose>

      <Prose>
        Item bank evolution scales reasonably well. As new items are added to a benchmark, they can be calibrated against the existing model panel using only their own response columns, with the existing item parameters and ability estimates held fixed. This is a per-item logistic regression that takes milliseconds. Old items can be retired when their discriminations drop or when contamination is detected. The challenge is detecting contamination — when a benchmark question has leaked into LLM training data, the IRT signature is a sudden drop in discrimination for that item across newer models, but distinguishing contamination from natural item-level noise requires careful longitudinal analysis. The HELM project at Stanford and the Open LLM Leaderboard at HuggingFace both run this kind of monitoring at scale.
      </Prose>

      <Prose>
        Score reporting scales well in one direction (smaller subset, IRT-mapped to full-benchmark scale) and poorly in the other (larger benchmark, IRT-extrapolated to a smaller calibration set). The reason: IRT mapping is a regression, and like all regressions it is reliable when the new evaluation point lies inside the calibration distribution and unreliable when it lies outside. For tinyBenchmarks, the calibration models span the full ability range, so subset scores from new models in that range map back to full-benchmark equivalents accurately. For the inverse direction — using a full-benchmark score on a held-out subset to predict an IRT theta — accuracy is also good because the subset is a faithful sample of the full set. Where IRT extrapolation breaks down is when the new model is genuinely outside the calibration range (a frontier model whose ability is well above any model in the panel), in which case both raw accuracy and IRT estimates start to saturate near 1 and the discrimination signal disappears.
      </Prose>

      <Prose>
        Cross-benchmark linking scales sub-linearly. If you have IRT fits on K different benchmarks (MMLU, HellaSwag, ARC, GSM8K, etc.) and want to put all the resulting abilities on a unified scale, you need anchor models — models that were evaluated on every benchmark — and concurrent calibration that ties the scales together. With K benchmarks and N anchor models, the linking solves for K-1 affine transforms (slope and intercept per benchmark relative to a reference), which requires N >= 2 anchor models per benchmark and is reliably stable with N >= 10. The HELM project at Stanford runs exactly this kind of cross-benchmark concurrent calibration, producing a unified ability scale across dozens of benchmarks with a panel of ~30 anchor models. The cost grows linearly with K, not multiplicatively, which is why this approach is operationally practical even for very large benchmark suites.
      </Prose>

      <Prose>
        Compute cost of fitting itself rarely binds. The 2PL log-likelihood evaluates in a single matrix-multiply over the (N x J) response matrix; for the largest LLM-eval scales currently published (~500 models x ~50000 items = 25M cells), a JML fit completes in under a minute on a single CPU and an MML fit completes in a few minutes. GPU-accelerated variational fits via py-irt are faster still. The bottleneck for IRT in LLM eval is essentially never the IRT fit itself — it is the upstream evaluation panel and the downstream operational pipeline that maintains item parameters across benchmark refreshes.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Local independence violation</H3>
      <Prose>
        IRT assumes that, given a test-taker's ability, their responses to different items are independent. This is violated when items share content — multi-part questions with shared stems, items that test the same fact, or items in a benchmark cluster (e.g., MMLU contains 57 subjects, and items within a subject correlate beyond what ability alone explains). Violations inflate the apparent information of correlated items and produce overconfident ability estimates. The diagnostic is the Q3 statistic (Yen 1984) computed on residuals; in production, address violations by blocking correlated items into <em>testlets</em> (Wainer & Kiely 1987) or by allowing item factor correlations beyond the single ability dimension.
      </Prose>

      <H3>Negative discriminations</H3>
      <Prose>
        A fitted item with negative discrimination is one where higher-ability test-takers do <em>worse</em> than lower-ability test-takers. This is almost always a sign of a problem: the item is mislabeled (the wrong answer is marked correct), the item is ambiguous and stronger reasoners notice the ambiguity, or the item is "tricky" in a way that punishes careful thought. In LLM benchmarks, negative discriminations have surfaced documented errors in MMLU, HellaSwag, and BoolQ — items where the gold answer is wrong or where multiple options are defensible. Always inspect items with discrimination below 0.1 (and especially negative); they are diagnostic gold.
      </Prose>

      <H3>Guessing parameter unidentified in 3PL</H3>
      <Prose>
        Estimating the guessing parameter <Code>{"c_j"}</Code> requires observations of low-ability test-takers attempting the item, because <Code>{"c_j"}</Code> is the lower asymptote that only emerges in that part of the ability range. If your panel has few low-ability subjects (a common situation in LLM eval where you intentionally want strong models in the panel), <Code>{"c_j"}</Code> is poorly identified and tends to take on extreme values driven by the prior or by initialization. The standard mitigation is to put a tight prior on <Code>{"c_j"}</Code> (e.g., Beta(5, 17) for 4-option multiple choice, centered at 0.25) or to fix it at the theoretical chance rate based on the item format.
      </Prose>

      <H3>Sign and scale ambiguity in raw fits</H3>
      <Prose>
        The 1PL/2PL likelihood is invariant to a global sign flip and a scale change of the ability axis. After fitting, the estimated abilities may be on an inverted scale (high-ability test-takers get negative theta) or on a scale very different from the standard normal. The fix is post-hoc rescaling: shift abilities to mean zero, scale to unit standard deviation, and check the sign by correlating with raw accuracy (which should be positive). If you are linking two separate IRT fits, you also need <em>anchor items</em> (items present in both fits with their parameters fixed across fits) or <em>anchor persons</em> to put both fits on the same scale.
      </Prose>

      <H3>Overfitting with small samples</H3>
      <Prose>
        With fewer than ~30 test-takers per item or fewer than ~30 items per test-taker, joint MLE produces noisy item parameter estimates. Discriminations especially can swing wildly. Symptoms: discriminations that span an unrealistically wide range (some items get a = 5, others a = 0.1, when the true discriminations are all near 1); unstable parameters across bootstrap re-samples; failure to recover known parameters in synthetic data. Mitigation: use MML or Bayesian estimation with informative priors on discriminations (a log-normal prior with mode 1 and modest variance is standard); regularize JML by adding a quadratic penalty on log-discriminations; or fall back to 1PL/Rasch which has fewer parameters to estimate per item.
      </Prose>

      <H3>Distribution shift in the model panel</H3>
      <Prose>
        IRT-curated subsets like tinyBenchmarks are calibrated against a specific panel of models — typically the LLMs that existed at the time of curation. When a new generation of models arrives that is substantially more capable than anything in the calibration panel, the curated items may no longer be informative for the new ability range. tinyMMLU was calibrated on Open LLM Leaderboard models from early 2024; by late 2025 the strongest models score near 0.95 on tinyMMLU, and the items that were once informative are now ceiling effects. The fix is periodic re-calibration: re-fit the IRT model with the new generation of LLMs in the panel and re-curate the subset. Operationally, this lags behind the actual frontier by months.
      </Prose>

      <H3>Contamination masquerading as ability</H3>
      <Prose>
        When a benchmark has leaked into a model's training data, the model gets items right because it memorized them, not because of underlying ability. The IRT signal is subtle but distinctive: contaminated items have unusually high response probability for the contaminated model relative to its ability, producing a high residual. Aggregated across many contaminated items, the model's ability estimate is inflated. There is no IRT-internal way to detect this — it requires external evidence (training data overlap detection, dynamic benchmarks, near-duplicate detection). What IRT can do is provide a baseline ability estimate that, when much higher than expected from the model's behavior on uncontaminated benchmarks, signals that contamination should be investigated.
      </Prose>

      <H3>Quadrature node insufficiency in MML</H3>
      <Prose>
        Marginal MLE approximates the integral over <Code>{"\\theta"}</Code> by Gauss-Hermite quadrature with a fixed number of nodes (typically 21 to 41). With fewer nodes, the integral is inaccurate, especially in the tails of the ability distribution. Symptoms: MML fits diverge from JML fits in the tails; abilities for very-high or very-low subjects are systematically biased toward zero. Fix: increase quadrature nodes (up to 61 for production fits), use adaptive quadrature, or switch to MCMC for fits where tail accuracy matters.
      </Prose>

      <H3>Reporting raw subset accuracy as benchmark accuracy</H3>
      <Prose>
        A user runs their model on tinyMMLU (100 items) and reports the raw 100-item accuracy as if it were full MMLU (14k items) accuracy. Without the IRT-based mapping, raw subset accuracies are systematically biased — the curated subset is denser at certain difficulties than the full benchmark, so the score distribution is different. Always report the IRT-mapped equivalent score, not the raw subset accuracy, when communicating "the model's MMLU score." lm-eval-harness's tinyMMLU task does this automatically; bespoke evaluation pipelines often do not.
      </Prose>

      <H3>Differential item functioning across model families</H3>
      <Prose>
        Differential item functioning (DIF) occurs when an item has different parameters for different subgroups even when controlling for ability. In educational testing, DIF is the canonical signal of test bias — an item that has different difficulty for two demographic groups at the same ability level is suspect. In LLM evaluation, the analog is items that behave differently for different model families: an item that is easy for instruction-tuned models but hard for base models at the same overall benchmark accuracy may be testing instruction-following style rather than the underlying construct. DIF detection (the Mantel-Haenszel test, logistic regression DIF) is straightforward but rarely run on LLM benchmarks; running it would surface items whose discrimination parameters differ systematically across pretraining data sources, model architectures, or post-training methods, which is valuable diagnostic information for benchmark maintainers.
      </Prose>

      <H3>Position and ordering effects</H3>
      <Prose>
        Many LLM benchmarks present multiple-choice options in a fixed order (the gold answer is always option C, or always option A, depending on the dataset). Models with positional biases — and most models have some — get the same items right or wrong for spurious reasons unrelated to the construct being measured. IRT fits will absorb this positional bias as part of the item parameters, which is fine as long as the same option ordering is used at evaluation time. But it makes the IRT-derived item parameters non-transferable: an item recalibrated with shuffled options will have different parameters than the original. Best practice is to randomize option ordering during the calibration evaluations and average over permutations, which gives item parameters that reflect content rather than position.
      </Prose>

      <H3>Score saturation at the frontier</H3>
      <Prose>
        When the strongest models in your panel saturate a benchmark (accuracy approaching 1 on most items), IRT loses discriminative power at the high end of the ability scale. The information functions of all items decay rapidly above the mean ability, and standard errors grow correspondingly. The visible symptom is a "flat top" in the ability distribution — multiple frontier models clustered at very similar IRT theta values despite measurable differences on harder benchmarks. The fix is to either add harder items to the benchmark (extending the difficulty range upward) or to switch to a benchmark targeted at the frontier (the GPQA, AIME, ARC-AGI line of evaluations were designed exactly to address this). MMLU saturation is the canonical case: by 2025, most frontier models scored above 0.85 on MMLU, and IRT theta estimates for those models had standard errors that exceeded the score differences between them.
      </Prose>

      <Callout accent="gold">
        IRT failures are usually diagnostic, not catastrophic. A misfit IRT model still produces a usable ability ranking — the rank order is robust to most violations of model assumptions. What breaks is the calibrated absolute scale, the standard errors, and the item-level diagnostics. If you only need ranking, IRT is forgiving; if you need calibrated probabilities or uncertainty quantification, model fit must be checked carefully.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Each source below was verified against its original publication or arXiv listing as of 2026-04-21. Where multiple editions exist, the cited reference is the canonical version that practitioners cite in current literature.
      </Prose>

      <Prose>
        For a deeper introduction to the IRT framework with worked examples, the textbooks by Embretson and Reise (2000) and by de Ayala (2009, <em>The Theory and Practice of Item Response Theory</em>, Guilford Press) are the canonical entry points. For LLM-evaluation-specific applications, the Polo et al. paper and the EleutherAI lm-eval-harness documentation are the current authoritative references.
      </Prose>

      <H3>Rasch 1960 — the original 1PL model</H3>
      <Prose>
        Georg Rasch. <em>Probabilistic Models for Some Intelligence and Attainment Tests</em>. Danish Institute for Educational Research, Copenhagen, 1960; reprinted by University of Chicago Press, 1980. The founding text of the Rasch tradition. Introduces the one-parameter logistic model and proves its measurement-theoretic properties, in particular that the raw score is a sufficient statistic for ability under the Rasch model. Modern Rasch practitioners (the Rasch Measurement community, journals like <em>Journal of Applied Measurement</em>) treat this work as the basis for fundamental measurement in education and psychology.
      </Prose>

      <H3>Lord 1952 — the 2PL precursor and Lord 1968 — 3PL</H3>
      <Prose>
        Frederic M. Lord. "A Theory of Test Scores." <em>Psychometric Monograph No. 7</em>, Psychometric Society, 1952. Lord's Princeton dissertation, the first explicit treatment of test scores as observations of a latent normal-ogive ability variable. Subsequently refined and extended in: Frederic M. Lord and Melvin R. Novick. <em>Statistical Theories of Mental Test Scores</em>. Addison-Wesley, 1968. Chapters 17-20 by Allan Birnbaum present the modern 2PL and 3PL logistic formulations and derive the maximum-likelihood estimation theory. This is the textbook that established IRT as the dominant framework for high-stakes testing.
      </Prose>

      <H3>Birnbaum 1968 — logistic 2PL formalization</H3>
      <Prose>
        Allan Birnbaum. "Some Latent Trait Models and Their Use in Inferring an Examinee's Ability." Chapters 17-20 in Lord and Novick (1968), op. cit. Birnbaum substituted the logistic function for the normal ogive used in Lord 1952, yielding the algebraically convenient 2PL model that is universally used today. The substitution is justified by the observation that a logistic with appropriate scaling is numerically indistinguishable from the normal CDF on the relevant range, and the logistic gives closed-form derivatives that make estimation tractable.
      </Prose>

      <H3>Embretson and Reise 2000 — modern textbook</H3>
      <Prose>
        Susan E. Embretson and Steven P. Reise. <em>Item Response Theory for Psychologists</em>. Lawrence Erlbaum Associates, 2000. The standard modern textbook introduction to IRT, accessible to readers without a strong measurement-theory background. Covers 1PL/2PL/3PL, polytomous models, multidimensional IRT, differential item functioning, computerized adaptive testing, and software. Its "Old Rules / New Rules of Measurement" framing — contrasting CTT with IRT — is how many practitioners first internalize the difference.
      </Prose>

      <H3>Polo et al. 2024 — tinyBenchmarks</H3>
      <Prose>
        Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. "tinyBenchmarks: evaluating LLMs with fewer examples." arXiv:2402.14992. Published February 2024; ICML 2024. The paper that brought IRT to LLM evaluation. Demonstrates that 100-item IRT-curated subsets of MMLU, HellaSwag, TruthfulQA, ARC, and Winogrande reproduce full-benchmark rankings with rank correlation above 0.97, at 100×-1000× compute reduction. The released <Code>tinyBenchmarks</Code> package and integration with EleutherAI's lm-eval-harness made the workflow directly usable. Code and data: github.com/felipemaiapolo/tinyBenchmarks.
      </Prose>

      <H3>Chalmers 2012 — mirt R package</H3>
      <Prose>
        R. Philip Chalmers. "mirt: A Multidimensional Item Response Theory Package for the R Environment." <em>Journal of Statistical Software</em>, 48(6), 1-29, 2012. The reference paper for the mirt R package, the production-grade implementation of unidimensional and multidimensional IRT estimation. Supports 1PL through 4PL, graded response, partial credit, generalized partial credit, and many other models with MML, EAP, MAP, and Bayesian estimators. The de facto standard in academic psychometrics.
      </Prose>

      <H3>Lalor and Yu 2021 — py-irt</H3>
      <Prose>
        John P. Lalor and Pedro Rodriguez. "py-irt: A scalable Item Response Theory library for Python." Lalor's py-irt package and accompanying documentation; see also Lalor, Wu, and Yu, "Building an Evaluation Scale using Item Response Theory," EMNLP 2016. py-irt is the Python package used by tinyBenchmarks for 2PL and amortized 2PL fits at LLM-eval scale. Repository: github.com/nd-ball/py-irt.
      </Prose>

      <H3>Maydeu-Olivares 2013 — modern IRT estimation review</H3>
      <Prose>
        Albert Maydeu-Olivares. "Goodness-of-Fit Assessment of Item Response Theory Models." <em>Measurement: Interdisciplinary Research and Perspectives</em>, 11(3), 71-101, 2013. A comprehensive review of fit assessment for IRT models — limited-information statistics (M2), full-information statistics (Pearson chi-square), item-level fit (Q1, S-X^2), and person-level fit (l_z). Indispensable for anyone running IRT in production who needs to decide whether their fit is good enough.
      </Prose>

      <H3>Wainer and Kiely 1987 — testlets</H3>
      <Prose>
        Howard Wainer and Gary L. Kiely. "Item Clusters and Computerized Adaptive Testing: A Case for Testlets." <em>Journal of Educational Measurement</em>, 24(3), 185-201, 1987. Introduces the testlet concept — bundles of items that share content and should be modeled jointly to respect local independence violations. The natural generalization for LLM benchmarks where items are clustered by subject (MMLU's 57 subjects, BIG-Bench's task categories).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why information peaks at theta = b</H3>
      <Prose>
        Starting from the 2PL information formula <Code>{"I_j(\\theta) = a_j^2 P_j(\\theta) (1 - P_j(\\theta))"}</Code>, take the derivative with respect to <Code>{"\\theta"}</Code> and set it to zero. Show that the maximum occurs exactly at <Code>{"\\theta = b_j"}</Code>. What is the value of <Code>{"P_j"}</Code> at that point, and what is the maximum information value as a function of <Code>{"a_j"}</Code> alone? Use this result to explain why, when constructing a tinyBenchmark targeted at a specific ability range, you should prioritize items whose difficulties fall inside that range over items with extreme difficulties — even if the extreme-difficulty items have higher discrimination.
      </Prose>

      <H3>Exercise 2 — From the response matrix to the score-rank plot</H3>
      <Prose>
        Suppose you have a binary response matrix <Code>X</Code> of shape (50 models × 100 items). Without fitting an IRT model, you can already compute two ranks: the rank of each model by raw accuracy <Code>X.mean(1)</Code>, and (after fitting) the rank by IRT theta. When would the two rankings disagree, and which one would you trust more in those cases? Construct a small example (5 models × 4 items) where the raw-accuracy ranking and the IRT ranking flip the order of two models. Hint: items with very different discriminations make this possible.
      </Prose>

      <H3>Exercise 3 — Identifiability without anchors</H3>
      <Prose>
        Two researchers fit a 2PL model on completely disjoint datasets (different models, different items). Each obtains ability estimates and item parameters on a standardized scale (mean-0, sd-1 abilities). Researcher A claims that her model M_A has ability 1.2; Researcher B claims that his model M_B has ability 1.5. Can you conclude that M_B is more capable than M_A? Why or why not? What additional information would you need to put both estimates on the same scale, and how would you use it? Look up "concurrent calibration" and "common-item nonequivalent groups design" in the IRT literature for canonical answers.
      </Prose>

      <H3>Exercise 4 — Greedy vs. clustered item selection</H3>
      <Prose>
        Implement two item-selection strategies for a tinyBenchmark: (a) the greedy max-information selection from section 4f, which picks items one at a time to maximize total information at a target ability; (b) a clustered selection that first clusters items by their (a, b) parameters into K clusters, then picks the highest-information item from each cluster. Run both on the synthetic data from section 4 with K = 20, and compare: which one preserves rank ordering of new models better? Which one preserves absolute calibration of theta estimates better? Why might the production tinyBenchmarks paper prefer clustered selection despite greedy giving higher peak information?
      </Prose>

      <H3>Exercise 5 — Detecting contamination via IRT</H3>
      <Prose>
        Suppose a single model in your panel has been trained on a subset of MMLU items (contamination). Describe what you would expect to see in (a) the model's raw accuracy, (b) its fitted IRT ability, (c) the residuals (observed minus predicted) on contaminated vs. uncontaminated items, and (d) the residuals on contaminated items compared to other models in the panel. Design a statistical test that could flag the contaminated model based only on the IRT fit, without requiring access to the model's training data. What false-positive rate do you expect, and what is the minimum effect size (number of contaminated items) that your test could reliably detect?
      </Prose>

      <H3>Exercise 6 — Rasch sufficient statistics</H3>
      <Prose>
        Show that under the 1PL/Rasch model, the raw score (number of correct responses) is a sufficient statistic for a test-taker's ability. That is, given the raw score, the conditional likelihood of the specific pattern of correct/incorrect responses does not depend on theta. Use this property to argue why two test-takers with the same raw score must have the same Rasch-estimated ability, and why this property fails for the 2PL model. What practical consequence does this have for reporting Rasch scores versus 2PL theta estimates to non-technical audiences (educators, school administrators)?
      </Prose>

      <H3>Exercise 7 — When 3PL helps and when it hurts</H3>
      <Prose>
        Generate two synthetic datasets: (a) one where the true model is 2PL (no guessing), and (b) one where the true model is 3PL with c = 0.25 for every item (4-option multiple choice). For each dataset, fit both a 2PL and a 3PL model. Compare the parameter recovery accuracy and the AIC/BIC of both fits on both datasets. Under what circumstances does fitting 3PL on 2PL data produce worse estimates than fitting 2PL on 3PL data, and vice versa? Use this to formulate a practical rule for choosing between 2PL and 3PL in the absence of strong prior information.
      </Prose>

      <H3>Exercise 8 — Estimating theta for a single new model</H3>
      <Prose>
        You have already fitted item parameters <Code>{"a_j, b_j"}</Code> on a calibration panel and now want to estimate the ability of a single new model from its J binary responses. Write the one-dimensional negative log-likelihood for theta as a function of the responses and item parameters, derive its first and second derivatives, and explain why a Newton-Raphson step is the natural way to find the MLE. Under what circumstances does Newton-Raphson fail or oscillate, and what is the standard remedy? (Hint: bounded line search, or a Brent-style bracketed root finder on the score function.) For Bayesian estimation, replace the MLE with the posterior mean (EAP) or mode (MAP) and explain how you would compute each.
      </Prose>

      <H3>Exercise 9 — Test-information targeting</H3>
      <Prose>
        You are designing an LLM benchmark with a budget of 50 items, and you want it to be maximally informative about models in the ability range theta ∈ [1, 2] (current frontier models). You have a candidate pool of 1000 fitted items with known (a, b) parameters. Write the optimization problem you would solve to select the 50 items, and propose at least two practical algorithms for solving it (one greedy, one based on integer programming or a continuous relaxation). What constraints might you want to add beyond pure information maximization — e.g., to ensure topic diversity, to avoid items with discrimination above a threshold (to prevent over-reliance on any single item), or to maintain a target distribution of difficulties? How would each constraint change the algorithm?
      </Prose>

    </div>
  ),
};

export default irtModels;
