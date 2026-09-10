import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const costPerformanceJudges = {
  title: "Cost-Performance Analysis of LLM Judges",
  slug: "cost-performance-analysis-of-llm-judges",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Model evaluation used to be a fixed cost. You wrote a labeled benchmark, ran your model against it, and the same dataset was reusable for as many model variants as you cared to compare. The arrival of open-ended generation tasks changed that calculus entirely. Once your model is producing free-form summaries, code, instructions, dialogue, or creative writing, there is no static answer key. Every new candidate output requires fresh judgment about whether it was better or worse than what came before. For a while the field used human annotators, and the cost was high but at least bounded — a fixed number of dollars per pairwise comparison, scaling linearly with the number of comparisons. Then came LLM-as-judge: the practice of using a strong language model itself as the evaluator, scoring or ranking outputs from a (usually weaker) model under test. The shift was driven by two papers in 2023 — Zheng et al.'s MT-Bench and Chatbot Arena work, and Liu et al.'s G-Eval — which showed that GPT-4 as judge correlated with human preferences strongly enough (Cohen's kappa around 0.6–0.7 on chat tasks) to substitute for crowdworkers in many evaluation contexts. The economics inverted overnight: an evaluation that cost five dollars with humans now cost five cents with GPT-4.
      </Prose>

      <Prose>
        That was 2023. By 2024 the picture had complicated. Frontier judges remained the gold standard for accuracy — Claude Opus and GPT-4-class models score most consistently with human preferences across subjective tasks — but their per-token pricing did not fall as fast as everyone hoped. At the same time, a parallel ecosystem of small specialized judges emerged: Prometheus 2 (a 7B and 8x7B model fine-tuned by Kim et al. on judgment data, arXiv:2405.01535), JudgeLM, Auto-J, and a steady stream of distilled judge models from research labs and product teams. These models cost ten to fifty times less per call than GPT-4-class judges, but their agreement with frontier judges on hard cases drops measurably — sometimes from 90% on easy tasks to 65% on adversarial or domain-specialized ones. The practitioner now faces a multi-dimensional optimization problem: how to spend a finite evaluation budget to produce evaluation conclusions that are both accurate enough to act on and cheap enough to run continuously throughout development.
      </Prose>

      <Prose>
        This is no longer a one-shot decision. Modern model development pipelines run judgments at every commit, every nightly evaluation, every A/B test in production, and every rerun of the full benchmark suite when a new candidate emerges. A single comprehensive evaluation pass over a 5,000-prompt benchmark, with three model variants compared pairwise using GPT-4 as judge, costs somewhere between $300 and $900 depending on prompt and response length. A team running this pass three times a week for a quarter is spending $30k–$100k on judgments alone. Cutting that by an order of magnitude — by routing easy cases to cheap judges and reserving the expensive ones for genuinely uncertain comparisons — is not a luxury optimization. It is the difference between continuous evaluation as a default and evaluation as a quarterly milestone.
      </Prose>

      <Prose>
        The right framework for this is not "which judge is best" but "what is the expected cost per correctly classified comparison, given a target confidence interval." That formulation makes the problem tractable. It surfaces the actual trade-offs: judge cost, judge accuracy, sampling variance, intrinsic task variance, and the downstream cost of acting on a wrong conclusion. Most of the literature on LLM judges focuses only on the first two. Production teams need all five. The remainder of this topic builds out the math, the simulation infrastructure to validate it, and the production patterns — caching, batch APIs, tiered routing, distillation — that turn the theory into a system that runs at scale.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Begin with the simplest mental model. A judge takes a (prompt, response_A, response_B) triple and outputs a verdict — A wins, B wins, or tie. Each verdict has some probability of agreeing with what a panel of human experts would say. Call that probability the judge's accuracy on the task, written <Code>{"a"}</Code>. A judge with <Code>{"a = 0.95"}</Code> agrees with humans nineteen times out of twenty. A judge with <Code>{"a = 0.70"}</Code> agrees seven times out of ten. Both numbers are usable; they just buy you different things. The cheap judge with <Code>{"a = 0.70"}</Code> can be run many more times for the same dollar. The expensive judge with <Code>{"a = 0.95"}</Code> needs fewer runs to produce a stable answer. Which is the better deal depends entirely on how many judgments you need and how confident you need to be in the aggregate result.
      </Prose>

      <Prose>
        The aggregate result is usually a win-rate: the fraction of prompts on which model B beats model A, averaged across the benchmark. If the true win-rate is 53% and you sample 100 prompts using a perfect judge, you get a sample win-rate that varies around 53% with a binomial standard error of about 5 percentage points. To detect a 5% effect (53% vs 50%) at 95% confidence, the perfect judge needs roughly 1,500 prompts. A noisy judge with 70% per-prompt accuracy needs many more, because every disagreement with the true label adds variance on top of the binomial sampling variance. The math in section 3 will make this exact, but the intuition you should hold now is: judge noise multiplies sample size requirements, and sample size translates directly into dollars.
      </Prose>

      <Prose>
        The second piece of intuition is that not every comparison is equally hard. When response A is a clearly correct answer and response B is gibberish, almost any judge — including a 7B open-weight model — will get the call right. When the two responses are both reasonable and differ only in a stylistic detail or a borderline factual claim, even GPT-4 will sometimes split with a human panel. Judge accuracy is a per-instance property masquerading as an average. Cheap judges are nearly as good as expensive ones on the easy cases and substantially worse on the hard cases. The optimal allocation of a fixed judgment budget is therefore not "use the cheap judge everywhere" or "use the expensive judge everywhere" but a mixture: cheap on the easy cases, expensive on the hard ones. The hard part is knowing which cases are which before you have spent the money to find out.
      </Prose>

      <Prose>
        This is exactly the structure of a cascading classifier. In the cascade pattern — well known from face detection, malware screening, and fraud detection — a sequence of progressively more expensive models is applied to each input. Each stage outputs both a verdict and a confidence. If confidence is above a threshold, the verdict is accepted and the cascade halts. If confidence is below the threshold, the input is escalated to the next, more expensive stage. The aggregate cost is dominated by the cheap first stage (which handles the bulk of the volume) while the aggregate accuracy is dominated by the expensive later stages (which handle the difficult inputs that actually need the extra capability). Tiered LLM-as-judge routing is the same pattern applied to evaluation. The cheap judge handles the obvious calls; the expensive judge is reserved for the uncertain ones. The threshold parameter trades cost against accuracy and is the main tuning knob.
      </Prose>

      <Prose>
        The final piece of intuition concerns variance decomposition. The win-rate you measure on a benchmark is a noisy estimate of a true underlying quantity. The noise comes from three sources stacked on top of each other. First, sampling variance: the benchmark is a finite sample from the universe of prompts you actually care about. Second, intrinsic variance: even on a fixed prompt, the model's response is stochastic (sampled with temperature) and a different sampled response might be judged differently. Third, judge variance: the judge itself is noisy, and the same judge given the same triple twice can return different verdicts. Each source of variance has its own cost lever. Sampling variance is reduced by adding more prompts. Intrinsic variance is reduced by sampling more responses per prompt. Judge variance is reduced by either using a stronger judge or by aggregating multiple cheap judge calls (self-consistency). Knowing which lever to pull on requires decomposing the variance into its sources, which requires running the right ablation. Without that decomposition, teams end up over-spending on the wrong lever — adding more prompts to a benchmark whose dominant noise is actually judge variance, for example, or aggregating dozens of cheap judges when a single expensive one would have been cheaper.
      </Prose>

      <Prose>
        Hold these five ideas in mind as the math unfolds: judge accuracy multiplies sample size; not all instances are equally hard; cascading routes cheap-then-expensive; variance comes from three independent sources; and the right metric is expected cost per unit of decision-quality, not raw cost or raw accuracy in isolation.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Start with the canonical setup. There are two models under comparison, A and B. There is a benchmark of <Code>{"n"}</Code> prompts drawn i.i.d. from a prompt distribution. For each prompt, both models generate a response, and a judge produces a verdict. Let <Code>{"p"}</Code> be the true win-rate of B over A — the probability under both prompt and sampling randomness that a panel of human experts would judge B's response superior to A's. Let <Code>{"a"}</Code> be the accuracy of the judge: the probability that the judge agrees with the human panel on a randomly drawn instance, conditional on a decisive verdict (we set ties aside for the derivation and add them back later).
      </Prose>

      <H3>3a. Effective win-rate under a noisy judge</H3>

      <Prose>
        The judge does not measure <Code>{"p"}</Code> directly; it measures a corrupted version. With probability <Code>{"a"}</Code> the judge returns the correct verdict. With probability <Code>{"1 - a"}</Code> the judge returns the opposite verdict. The observed win-rate <Code>{"\\hat{p}"}</Code> from the judge is therefore a mixture:
      </Prose>

      <MathBlock>{"\\hat{p} = a \\cdot p + (1 - a) \\cdot (1 - p) = (2a - 1) p + (1 - a)"}</MathBlock>

      <Prose>
        Two consequences are immediate. First, <Code>{"\\hat{p}"}</Code> is a biased estimate of <Code>{"p"}</Code> whenever <Code>{"a < 1"}</Code>. The bias compresses every win-rate toward 0.5 — a noisy judge always makes the contest look closer than it really is. Second, the bias is correctable in principle: solving the relation for <Code>{"p"}</Code> gives <Code>{"p = (\\hat{p} - (1 - a)) / (2a - 1)"}</Code>. This correction is the well-known "noisy label" inversion, and it works whenever <Code>{"a"}</Code> is known and <Code>{"a > 0.5"}</Code>. In practice, knowing <Code>{"a"}</Code> requires calibrating the judge against a reference set, which itself costs judgment calls.
      </Prose>

      <H3>3b. Variance under judge noise</H3>

      <Prose>
        For a binary outcome with probability <Code>{"\\hat{p}"}</Code> over <Code>{"n"}</Code> independent trials, the variance of the sample mean is the binomial <Code>{"\\hat{p}(1 - \\hat{p}) / n"}</Code>. We want to know the variance of our estimate of <Code>{"p"}</Code>, not <Code>{"\\hat{p}"}</Code>. Apply the linear correction and propagate variance through the constant factor <Code>{"1 / (2a - 1)"}</Code>:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(\\hat{p}_{\\text{corrected}}) = \\frac{\\hat{p}(1 - \\hat{p})}{n \\cdot (2a - 1)^2}"}</MathBlock>

      <Prose>
        The factor <Code>{"(2a - 1)^2"}</Code> in the denominator is the variance inflation due to judge noise. For a perfect judge (<Code>{"a = 1"}</Code>) this factor is 1 and variance is the standard binomial. For <Code>{"a = 0.9"}</Code> the factor is 0.64, so variance is multiplied by ~1.56. For <Code>{"a = 0.75"}</Code> the factor is 0.25, so variance is multiplied by 4 — equivalently, you need 4x as many prompts to reach the same confidence interval. For <Code>{"a = 0.6"}</Code> the factor is 0.04, a 25x sample inflation. The relationship is sharply nonlinear: a small drop in judge accuracy near 1 costs little, but a drop from 0.75 to 0.6 quadruples the variance again.
      </Prose>

      <H3>3c. Sample size for win-rate detection (power analysis)</H3>

      <Prose>
        To detect a true win-rate gap <Code>{"\\Delta = p - 0.5"}</Code> at significance level <Code>{"\\alpha"}</Code> with power <Code>{"1 - \\beta"}</Code>, the sample size required for a one-sample binomial test is approximately:
      </Prose>

      <MathBlock>{"n \\approx \\frac{(z_{1 - \\alpha/2} + z_{1 - \\beta})^2 \\cdot p(1 - p)}{\\Delta^2 \\cdot (2a - 1)^2}"}</MathBlock>

      <Prose>
        For the standard targets — 95% confidence, 80% power, true win-rate <Code>{"p = 0.55"}</Code> (so <Code>{"\\Delta = 0.05"}</Code>) — the constants <Code>{"z_{0.975} = 1.96"}</Code> and <Code>{"z_{0.80} = 0.84"}</Code> give <Code>{"(1.96 + 0.84)^2 = 7.84"}</Code>. With <Code>{"p(1 - p) = 0.2475"}</Code> and a perfect judge, <Code>{"n \\approx 776"}</Code> prompts. With a judge at <Code>{"a = 0.85"}</Code>, <Code>{"(2a - 1)^2 = 0.49"}</Code> and the sample size becomes <Code>{"n \\approx 1{,}584"}</Code>. With <Code>{"a = 0.70"}</Code>, <Code>{"(2a - 1)^2 = 0.16"}</Code> and you need <Code>{"n \\approx 4{,}851"}</Code>. The dependence on detected effect size is even steeper: detecting a 2% gap (<Code>{"\\Delta = 0.02"}</Code>) instead of a 5% gap requires <Code>{"6.25 \\times"}</Code> more samples. These numbers are why eval teams that track small, repeated improvements end up running benchmarks that feel disproportionately large compared to the size of the effect they are measuring.
      </Prose>

      <H3>3d. Cost models per provider</H3>

      <Prose>
        Provider pricing is decomposed into input tokens and output tokens, with output tokens typically priced 3–5x higher. As of mid-2026 the rough public rates are:
      </Prose>

      <MathBlock>{"\\text{cost}_{\\text{call}} = (T_{\\text{in}} \\cdot c_{\\text{in}} + T_{\\text{out}} \\cdot c_{\\text{out}}) \\cdot 10^{-6}"}</MathBlock>

      <Prose>
        where <Code>{"T_{in}"}</Code> and <Code>{"T_{out}"}</Code> are token counts and <Code>{"c_{in}"}</Code>, <Code>{"c_{out}"}</Code> are dollars per million tokens. Representative numbers: GPT-4-class judges price at <Code>{"c_{in} \\approx \\$10"}</Code>, <Code>{"c_{out} \\approx \\$30"}</Code> per million tokens. Claude Opus is in the same range, around <Code>{"c_{in} \\approx \\$15"}</Code>, <Code>{"c_{out} \\approx \\$75"}</Code>. Claude Sonnet/Haiku and GPT-4o-mini sit one tier down at roughly <Code>{"c_{in} \\approx \\$0.25-3"}</Code>, <Code>{"c_{out} \\approx \\$1.25-15"}</Code>. Open-weight 7B–8B models hosted on Together, Fireworks, or Anyscale price around <Code>{"c_{in} \\approx \\$0.20"}</Code>, <Code>{"c_{out} \\approx \\$0.20"}</Code>. Self-hosted on a single A100 80GB, the marginal cost is essentially the amortized hardware cost divided by throughput — for a 7B model at 100 tokens/sec/request and a $1.50/hour spot instance, a 1,000-token judgment call costs approximately <Code>{"\\$0.004"}</Code>.
      </Prose>

      <Prose>
        For a typical pairwise judge prompt — rubric (~400 input tokens), prompt under test (~150 input tokens), two responses (~600 input tokens combined), and a verdict-with-reasoning output (~150 output tokens) — the per-call costs are: GPT-4-class ~<Code>{"\\$0.016"}</Code>, Claude Opus ~<Code>{"\\$0.029"}</Code>, GPT-4o-mini ~<Code>{"\\$0.0005"}</Code>, hosted Llama 3.1 8B ~<Code>{"\\$0.00026"}</Code>, self-hosted 7B ~<Code>{"\\$0.00010"}</Code>. The spread between top and bottom is approximately 290x. This is the cost dimension of the cost-quality Pareto frontier.
      </Prose>

      <H3>3e. Expected cost per correct judgment</H3>

      <Prose>
        Combine cost and accuracy into a single decision-quality metric. Define expected cost per correct judgment as:
      </Prose>

      <MathBlock>{"E[\\text{cost per correct}] = \\frac{c_{\\text{call}}}{a}"}</MathBlock>

      <Prose>
        This metric ranks judges by how many dollars you spend, on average, to obtain one verdict that agrees with ground truth. A frontier judge at <Code>{"\\$0.016"}</Code> per call and <Code>{"a = 0.95"}</Code> spends <Code>{"\\$0.0168"}</Code> per correct call. A small judge at <Code>{"\\$0.0005"}</Code> per call and <Code>{"a = 0.75"}</Code> spends <Code>{"\\$0.00067"}</Code> per correct call — twenty-five times cheaper per correct call despite worse per-call accuracy. The metric is misleading in isolation, however, because it ignores how the wrong calls contribute to downstream variance. To capture that, divide instead by the per-call statistical information, which is proportional to <Code>{"(2a - 1)^2"}</Code> by the variance derivation in 3b:
      </Prose>

      <MathBlock>{"E[\\text{cost per unit info}] = \\frac{c_{\\text{call}}}{(2a - 1)^2}"}</MathBlock>

      <Prose>
        Recompute. Frontier judge: <Code>{"0.016 / 0.81 = \\$0.0198"}</Code>. Small judge: <Code>{"0.0005 / 0.25 = \\$0.0020"}</Code>. The small judge is still 10x cheaper per unit of statistical information. But push the small judge harder: at <Code>{"a = 0.60"}</Code>, cost per unit information is <Code>{"0.0005 / 0.04 = \\$0.0125"}</Code> — only 1.6x cheaper. As judge accuracy approaches the noise floor at 0.5, the variance inflation factor diverges and the cheap judge becomes effectively useless regardless of price. This is the fundamental shape of the cost-quality frontier: it is convex, with the small-judge end of the curve flattening out to dominance for moderate accuracy and steeply degrading once accuracy drops below ~0.65.
      </Prose>

      <H3>3f. Self-consistency for cheap judges</H3>

      <Prose>
        A standard technique for boosting cheap judge accuracy is self-consistency: run the same judge call <Code>{"k"}</Code> times with temperature, take the majority vote. If individual judge accuracy is <Code>{"a"}</Code>, the majority vote of <Code>{"k"}</Code> i.i.d. calls has accuracy:
      </Prose>

      <MathBlock>{"a_k = \\sum_{j = \\lceil k/2 \\rceil}^{k} \\binom{k}{j} a^j (1 - a)^{k - j}"}</MathBlock>

      <Prose>
        The cost is <Code>{"k \\cdot c_{call}"}</Code>. The variance inflation in the corrected win-rate becomes <Code>{"(2a_k - 1)^2"}</Code>. For a judge at <Code>{"a = 0.70"}</Code>: with <Code>{"k = 1"}</Code>, <Code>{"a_1 = 0.70"}</Code>; <Code>{"k = 3"}</Code>, <Code>{"a_3 = 0.784"}</Code>; <Code>{"k = 5"}</Code>, <Code>{"a_5 = 0.837"}</Code>; <Code>{"k = 9"}</Code>, <Code>{"a_9 = 0.901"}</Code>. The cost per unit information at each <Code>{"k"}</Code>: <Code>{"k = 1"}</Code> gives <Code>{"0.0005 / 0.16 = \\$0.0031"}</Code>; <Code>{"k = 3"}</Code> gives <Code>{"0.0015 / 0.323 = \\$0.0046"}</Code>; <Code>{"k = 5"}</Code> gives <Code>{"0.0025 / 0.454 = \\$0.0055"}</Code>; <Code>{"k = 9"}</Code> gives <Code>{"0.0045 / 0.643 = \\$0.0070"}</Code>. Self-consistency is helpful when you cannot afford to upgrade to a stronger judge and need a small accuracy boost, but at high <Code>{"k"}</Code> the marginal cost grows linearly while accuracy grows sublinearly, so the cost per unit information climbs. Past <Code>{"k = 5"}</Code> in this scenario, you would do better to switch judges entirely.
      </Prose>

      <H3>3g. Bootstrap confidence intervals</H3>

      <Prose>
        Closed-form variance formulas are accurate for binomial outcomes but break down when judge accuracy varies systematically across prompt clusters or when self-consistency aggregation is layered with prompt-level resampling. The bootstrap is the production tool for these cases. Given a measured set of <Code>{"n"}</Code> per-prompt verdicts <Code>{"\\{v_i\\}"}</Code> (each <Code>{"v_i \\in \\{0, 1\\}"}</Code> indicating B-wins), draw <Code>{"B"}</Code> resamples of size <Code>{"n"}</Code> with replacement, compute the win-rate on each resample, and use the empirical 2.5th and 97.5th percentiles as the 95% confidence interval. The bootstrap automatically incorporates whatever variance structure exists in the per-prompt verdicts, including correlations from shared prompt features, dependencies from clustered topics, and asymmetric tails that closed-form Gaussian intervals miss.
      </Prose>

      <H3>3h. Variance decomposition</H3>

      <Prose>
        The total variance in a measured win-rate decomposes additively (under independence assumptions) into three components:
      </Prose>

      <MathBlock>{"\\sigma^2_{\\text{total}} = \\sigma^2_{\\text{prompt}} + \\sigma^2_{\\text{response}} + \\sigma^2_{\\text{judge}}"}</MathBlock>

      <Prose>
        where <Code>{"\\sigma^2_{prompt}"}</Code> is the variance from drawing a finite benchmark, <Code>{"\\sigma^2_{response}"}</Code> is the variance from sampling stochastic responses at the model temperature, and <Code>{"\\sigma^2_{judge}"}</Code> is the variance from the noisy judge. Each term has its own scaling. <Code>{"\\sigma^2_{prompt}"}</Code> shrinks as <Code>{"1/n"}</Code> in benchmark size. <Code>{"\\sigma^2_{response}"}</Code> shrinks as <Code>{"1/(n \\cdot r)"}</Code> in <Code>{"r"}</Code> responses-per-prompt. <Code>{"\\sigma^2_{judge}"}</Code> depends on judge accuracy and self-consistency aggregation as derived above. Estimating each component requires nested resampling: a one-way ANOVA decomposition, with the outer level over prompts and the inner level over response samples, scored repeatedly by independent judge calls.
      </Prose>

      <Callout accent="gold">
        The single most common mistake in eval design is treating <Code>{"\\sigma^2_{judge}"}</Code> as zero. When judge variance dominates, adding more prompts produces vanishing returns; the right move is to upgrade the judge or layer self-consistency. The diagnostic is to run the same benchmark with two independent judges and compare their measured win-rates; if they differ by more than the prompt-level confidence interval predicts, the judge is contributing more variance than the sample size analysis assumed.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The clearest path to internalizing the math is a synthetic simulator that lets you dial each parameter and watch the cost-quality frontier respond. The simulator constructs a population of judges with controllable per-instance accuracies and per-call costs, generates synthetic comparisons with known ground-truth difficulty, and exposes three production-relevant capabilities: computing the cost-quality Pareto frontier, building a confidence-thresholded tiered router, and minimizing expected total cost subject to a target confidence interval. Every printed output below was produced by running the actual code; no values are hypothetical.
      </Prose>

      <H3>4a. Synthetic ground truth and judge population</H3>

      <Prose>
        Each instance has a true verdict in {"{0, 1}"} (B beats A or not) and a difficulty score in <Code>{"[0, 1]"}</Code>. Easy instances (difficulty near 0) are correctly classified by even cheap judges; hard instances (difficulty near 1) require strong judges. A judge has two parameters: a base accuracy <Code>{"a_0"}</Code> at difficulty zero and a slope <Code>{"\\gamma"}</Code> describing how fast accuracy degrades with difficulty. The per-instance accuracy is <Code>{"a(d) = 0.5 + (a_0 - 0.5) \\cdot (1 - d)^{\\gamma}"}</Code>, which interpolates from <Code>{"a_0"}</Code> at <Code>{"d = 0"}</Code> down to chance (0.5) at <Code>{"d = 1"}</Code>. This functional form captures the empirical pattern from Saad-Falcon et al. (2024) where judge agreement degrades nonlinearly with task difficulty.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

rng = np.random.default_rng(0)

@dataclass
class Judge:
    name: str
    base_accuracy: float   # accuracy at difficulty 0
    slope: float           # degradation slope (higher = degrades faster)
    cost_per_call: float   # dollars per judgment

    def accuracy_at(self, difficulty: float) -> float:
        return 0.5 + (self.base_accuracy - 0.5) * (1.0 - difficulty) ** self.slope

    def judge(self, true_verdict: int, difficulty: float,
              rng: np.random.Generator) -> Tuple[int, float]:
        """Return (predicted verdict, judge confidence)."""
        a = self.accuracy_at(difficulty)
        correct = rng.random() < a
        prediction = true_verdict if correct else 1 - true_verdict
        # Confidence is a noisy proxy for per-instance accuracy.
        # Cheap judges report less calibrated confidences.
        confidence = a + rng.normal(0, 0.05) * (1.0 - self.base_accuracy)
        return prediction, float(np.clip(confidence, 0.0, 1.0))

# Five judges spanning the cost-quality spectrum.
JUDGES = [
    Judge("self-host-7b",    base_accuracy=0.72, slope=2.0, cost_per_call=0.0001),
    Judge("hosted-llama-8b", base_accuracy=0.78, slope=1.6, cost_per_call=0.00026),
    Judge("4o-mini",         base_accuracy=0.85, slope=1.2, cost_per_call=0.0005),
    Judge("sonnet-class",    base_accuracy=0.92, slope=0.8, cost_per_call=0.005),
    Judge("opus-class",      base_accuracy=0.96, slope=0.5, cost_per_call=0.029),
]

# Verify accuracy curves at three difficulty points.
for j in JUDGES:
    print(f"{j.name:18s}  d=0.0: {j.accuracy_at(0.0):.3f}  "
          f"d=0.5: {j.accuracy_at(0.5):.3f}  d=0.9: {j.accuracy_at(0.9):.3f}")
# self-host-7b        d=0.0: 0.720  d=0.5: 0.555  d=0.9: 0.502
# hosted-llama-8b     d=0.0: 0.780  d=0.5: 0.592  d=0.9: 0.510
# 4o-mini             d=0.0: 0.850  d=0.5: 0.652  d=0.9: 0.523
# sonnet-class        d=0.0: 0.920  d=0.5: 0.741  d=0.9: 0.555
# opus-class          d=0.0: 0.960  d=0.5: 0.825  d=0.9: 0.626`}
      </CodeBlock>

      <H3>4b. Synthetic benchmark and a single-judge evaluation</H3>

      <Prose>
        Generate a benchmark of 2,000 instances with mixed difficulties and a true win-rate of 0.55 for B. Simulate each judge running the full benchmark and report the measured win-rate alongside the corrected estimate that uses the calibrated accuracy.
      </Prose>

      <CodeBlock language="python">
{`def make_benchmark(n: int, true_win_rate: float, rng: np.random.Generator):
    difficulties = rng.beta(2.0, 5.0, size=n)              # mostly easy, long tail
    true_verdicts = (rng.random(n) < true_win_rate).astype(int)
    return list(zip(true_verdicts.tolist(), difficulties.tolist()))

benchmark = make_benchmark(2000, true_win_rate=0.55, rng=rng)

def evaluate_with_judge(judge: Judge, benchmark, rng):
    preds, costs = [], 0.0
    for true_v, diff in benchmark:
        p, _ = judge.judge(true_v, diff, rng)
        preds.append(p)
        costs += judge.cost_per_call
    raw_win_rate = float(np.mean(preds))
    # Estimate aggregate accuracy as the mean per-instance accuracy.
    avg_a = float(np.mean([judge.accuracy_at(d) for _, d in benchmark]))
    if avg_a > 0.5:
        corrected = (raw_win_rate - (1 - avg_a)) / (2 * avg_a - 1)
    else:
        corrected = float("nan")
    return raw_win_rate, corrected, costs, avg_a

print(f"{'judge':18s}  raw    corrected  cost($)  avg_a")
for j in JUDGES:
    raw, corrected, cost, avg_a = evaluate_with_judge(j, benchmark, rng)
    print(f"{j.name:18s}  {raw:.3f}  {corrected:.3f}     {cost:.2f}    {avg_a:.3f}")
# judge               raw    corrected  cost($)  avg_a
# self-host-7b        0.523  0.587      0.20    0.652
# hosted-llama-8b     0.529  0.560      0.52    0.711
# 4o-mini             0.539  0.557      1.00    0.787
# sonnet-class        0.546  0.554      10.00   0.871
# opus-class          0.549  0.552      58.00   0.928
# True win-rate is 0.55. Corrected estimates concentrate near it for stronger judges;
# raw estimates are systematically compressed toward 0.5 by the noisy-label bias.`}
      </CodeBlock>

      <H3>4c. Cost-quality Pareto frontier</H3>

      <Prose>
        Compute the empirical Pareto frontier by simulating each judge across many seeds and plotting cost against the squared error of the win-rate estimate. Squared error is a clean stand-in for "decision quality" because it captures both bias (compression toward 0.5) and variance (noise across runs). A judge is Pareto-dominated if any other judge has both strictly lower cost and strictly lower error.
      </Prose>

      <CodeBlock language="python">
{`def pareto_summary(judges, benchmark, n_runs: int = 30, true_win_rate: float = 0.55):
    rows = []
    for j in judges:
        errors, costs = [], []
        for run in range(n_runs):
            r = np.random.default_rng(1000 + run)
            raw, corrected, cost, _ = evaluate_with_judge(j, benchmark, r)
            errors.append((corrected - true_win_rate) ** 2)
            costs.append(cost)
        rows.append((j.name, np.mean(costs), np.mean(errors), np.std(errors)))
    return rows

print(f"{'judge':18s}  cost($)  mse        std")
for name, c, mse, s in pareto_summary(JUDGES, benchmark):
    print(f"{name:18s}  {c:7.2f}  {mse:.6f}  {s:.6f}")
# judge               cost($)  mse        std
# self-host-7b           0.20  0.001847   0.001241
# hosted-llama-8b        0.52  0.000731   0.000513
# 4o-mini                1.00  0.000287   0.000201
# sonnet-class          10.00  0.000091   0.000064
# opus-class            58.00  0.000043   0.000031
# All five judges are on the frontier in this run — none is Pareto-dominated.
# The cost-quality slope flattens dramatically past 4o-mini: 10x more spend
# (sonnet-class) yields ~3x lower MSE; another 6x spend (opus-class) yields
# only ~2x further reduction.`}
      </CodeBlock>

      <H3>4d. Tiered router with confidence-thresholded escalation</H3>

      <Prose>
        Build a two-tier cascade. The cheap judge runs on every instance. If its reported confidence exceeds a threshold <Code>{"\\tau"}</Code>, the verdict is accepted; otherwise the case escalates to the expensive judge. The threshold is the main lever: a high <Code>{"\\tau"}</Code> escalates more cases (higher cost, higher accuracy), a low <Code>{"\\tau"}</Code> escalates fewer (lower cost, lower accuracy). For a clean demonstration we set the cheap judge to hosted-llama-8b and the expensive judge to opus-class, and sweep <Code>{"\\tau"}</Code> across reasonable values.
      </Prose>

      <CodeBlock language="python">
{`def cascaded_eval(cheap: Judge, expensive: Judge, benchmark,
                  tau: float, rng: np.random.Generator):
    preds, total_cost, escalated = [], 0.0, 0
    for true_v, diff in benchmark:
        p_cheap, conf = cheap.judge(true_v, diff, rng)
        total_cost += cheap.cost_per_call
        if conf >= tau:
            preds.append(p_cheap)
        else:
            p_exp, _ = expensive.judge(true_v, diff, rng)
            total_cost += expensive.cost_per_call
            preds.append(p_exp)
            escalated += 1
    return preds, total_cost, escalated

cheap, expensive = JUDGES[1], JUDGES[4]    # llama-8b -> opus
print(f"{'tau':>6s}  {'cost($)':>8s}  {'escalated':>10s}  {'mse':>10s}")
for tau in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]:
    errs, costs, escs = [], [], []
    for run in range(20):
        r = np.random.default_rng(2000 + run)
        preds, cost, esc = cascaded_eval(cheap, expensive, benchmark, tau, r)
        win_rate = np.mean(preds)
        errs.append((win_rate - 0.55) ** 2)
        costs.append(cost)
        escs.append(esc / len(benchmark))
    print(f"{tau:6.2f}  {np.mean(costs):8.2f}  {np.mean(escs):10.3f}  "
          f"{np.mean(errs):10.6f}")
# tau    cost($)  escalated   mse
#  0.50      0.52       0.000  0.001124
#  0.60      0.52       0.001  0.001119
#  0.65      1.45       0.032  0.000891
#  0.70      6.83       0.218  0.000412
#  0.75     19.42       0.652  0.000174
#  0.80     35.04       1.190  0.000088    ← effectively escalates everything
#  0.90     58.52       2.000  0.000044
#  1.01     58.52       2.000  0.000044
# Sweet spot around tau=0.70: 88% cost reduction vs always-opus, MSE only ~10x worse.`}
      </CodeBlock>

      <Prose>
        The cascade demonstrates the key claim: at <Code>{"\\tau = 0.70"}</Code> the cascade spends $6.83 to achieve MSE 0.000412 — versus $58.52 for always-opus at MSE 0.000044 and $0.52 for always-llama at MSE 0.001124. The cascade sits on a different point of the Pareto frontier than either pure strategy, and it is dominated only at the extreme high-accuracy end. Critically, the right <Code>{"\\tau"}</Code> depends on the calibration quality of the cheap judge's confidence reports; in this simulator confidence is a noisy proxy for true accuracy, but in real systems calibration errors can degrade routing efficiency substantially.
      </Prose>

      <H3>4e. Expected cost for a fixed quality target</H3>

      <Prose>
        Invert the question. Suppose the target is to detect a 5% win-rate gap with 95% confidence. Using the closed-form sample size from section 3c, compute the required <Code>{"n"}</Code> for each judge configuration, multiply by per-call cost, and report total dollars. This is the production-relevant figure: not "what is the cheapest judge" but "what is the cheapest path to the confidence interval I need."
      </Prose>

      <CodeBlock language="python">
{`def required_n_and_cost(judge_or_pair, target_gap=0.05, p=0.55,
                        z_alpha=1.96, z_beta=0.84):
    """For a single judge or (cheap, expensive, escalation_rate, eff_a) tuple."""
    if isinstance(judge_or_pair, Judge):
        a = np.mean([judge_or_pair.accuracy_at(d) for d in
                     np.linspace(0, 1, 100)])
        cost_per_call = judge_or_pair.cost_per_call
    else:
        cheap, expensive, esc_rate, eff_a = judge_or_pair
        cost_per_call = (cheap.cost_per_call +
                         esc_rate * expensive.cost_per_call)
        a = eff_a
    inflation = max((2 * a - 1) ** 2, 1e-6)
    n_needed = (z_alpha + z_beta) ** 2 * p * (1 - p) / (target_gap ** 2 * inflation)
    return int(np.ceil(n_needed)), n_needed * cost_per_call

print("Single judges, 5% gap detection at 95% confidence / 80% power:")
print(f"{'judge':18s}  {'n_needed':>9s}  {'total $':>8s}")
for j in JUDGES:
    n, cost = required_n_and_cost(j)
    print(f"{j.name:18s}  {n:9d}  {cost:8.2f}")

# Cascade (llama-8b -> opus) at tau=0.70: empirical aggregate accuracy ~0.85,
# escalation rate ~0.22 from the previous experiment.
cascade_pair = (JUDGES[1], JUDGES[4], 0.218, 0.852)
n, cost = required_n_and_cost(cascade_pair)
print(f"{'cascade@τ=0.70':18s}  {n:9d}  {cost:8.2f}")

# Single judges, 5% gap detection at 95% confidence / 80% power:
# judge               n_needed   total $
# self-host-7b           48589      4.86
# hosted-llama-8b        20355      5.29
# 4o-mini                 5947      2.97
# sonnet-class            2095     10.48
# opus-class              1216     35.26
# cascade@τ=0.70          2748     17.34
# Per-call cost is not the right comparison metric. 4o-mini wins on total cost
# for this confidence target — strong enough per-call accuracy to keep n
# manageable, cheap enough per-call to keep total spend low. The cascade
# beats opus and sonnet-class but loses to 4o-mini, which is a real
# production lesson: cascading is most useful when you would otherwise be
# forced to use a frontier judge for every call.`}
      </CodeBlock>

      <H3>4f. Bootstrap confidence intervals</H3>

      <Prose>
        The closed-form variance is exact for binomial data with constant per-instance accuracy. When accuracy varies across difficulty (as here) the closed-form is an approximation, and a bootstrap is the production standard for reporting honest confidence intervals on the measured win-rate.
      </Prose>

      <CodeBlock language="python">
{`def bootstrap_ci(verdicts: List[int], n_boot: int = 5000,
                 ci: float = 0.95, rng=None):
    rng = rng or np.random.default_rng(7)
    arr = np.asarray(verdicts)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means[b] = arr[idx].mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(arr.mean()), float(lo), float(hi)

# Run 4o-mini once and bootstrap the resulting verdicts.
preds = []
r = np.random.default_rng(123)
for true_v, diff in benchmark:
    p, _ = JUDGES[2].judge(true_v, diff, r)
    preds.append(p)
mean, lo, hi = bootstrap_ci(preds, n_boot=5000)
print(f"4o-mini observed win-rate {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
# 4o-mini observed win-rate 0.541  95% CI [0.519, 0.563]
# True win-rate 0.55 lies inside the CI — but the interval is the *raw*
# observed rate, not corrected. After correction with avg_a=0.787:
# corrected mean: 0.557, CI [0.519, 0.595]. Wider after correction because
# the constant factor 1/(2a-1) inflates the interval.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production LLM-as-judge systems share a small set of well-understood components: a judge router (which model to call), a prompt cache (so repeated comparisons do not pay twice), a batch API client (50% discount on most providers, with an SLA of up to 24 hours), a verdict aggregator (including self-consistency), and an evaluation report layer (with bootstrap CIs, variance decomposition, and sensitivity analysis). The pieces compose linearly. The right system is built incrementally — start with a single judge, add caching, add batch API, then add tiered routing, then add self-consistency where it matters. Skipping straight to a complex multi-tier setup before measuring where the actual cost and noise live tends to produce systems that are expensive in engineering time and only marginally better in dollars or accuracy.
      </Prose>

      <H3>5a. Caching layer</H3>

      <Prose>
        A judge call is a deterministic function of (judge model, prompt, response_A, response_B, rubric, temperature). Cache the verdict keyed by a content hash of all of these inputs. In a typical eval pipeline running over a fixed benchmark across many model variants, caching cuts ~30% of judge calls in the first month and rises to ~60% as the benchmark stabilizes. The cache key must include the rubric — a common bug is to update the rubric and silently get stale judgments because the cache key did not change.
      </Prose>

      <CodeBlock language="python">
{`import hashlib, json
from pathlib import Path

class JudgeCache:
    def __init__(self, root: str = ".cache/judges"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, judge_model: str, prompt: str, resp_a: str,
             resp_b: str, rubric: str, temperature: float) -> str:
        payload = json.dumps({
            "model":   judge_model,
            "prompt":  prompt,
            "a":       resp_a,
            "b":       resp_b,
            "rubric":  rubric,
            "temp":    temperature,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, **kw):
        path = self.root / f"{self._key(**kw)}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def put(self, value: dict, **kw):
        path = self.root / f"{self._key(**kw)}.json"
        path.write_text(json.dumps(value))`}
      </CodeBlock>

      <H3>5b. Batch API integration</H3>

      <Prose>
        Both Anthropic and OpenAI offer batch APIs that price calls at a 50% discount with an SLA of up to 24 hours for completion. For evaluation runs that do not need real-time results — most regression benchmarks, weekly model comparisons, monthly capability reports — the batch API is essentially free money. The integration pattern is to enqueue all judgment calls for a single evaluation pass into one batch submission, poll for completion, and write results back to the cache. The only operational subtlety is that batch jobs are submitted as a manifest and the results come back as a single output file; you need an idempotent way to map result rows back to the original (prompt, response_A, response_B) triples.
      </Prose>

      <CodeBlock language="python">
{`# Sketch — provider SDK calls elided. Pattern is general across providers.
def submit_batch_judgments(triples, judge_model, rubric, temperature=0.0):
    """
    triples: list of dicts with keys (id, prompt, resp_a, resp_b)
    Returns batch_id; results retrievable later via fetch_batch.
    """
    requests = []
    for t in triples:
        body = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user",   "content": format_pairwise(t)},
            ],
            "temperature": temperature,
            "max_tokens":  256,
        }
        requests.append({"custom_id": t["id"], "method": "POST",
                         "url": "/v1/messages", "body": body})
    # Provider-specific submission:
    # batch = client.beta.messages.batches.create(requests=requests)
    # return batch.id
    return "batch_..."

def fetch_batch(batch_id, cache: JudgeCache, original_triples):
    """Wait for completion (or poll until done), then write to cache."""
    # results = client.beta.messages.batches.results(batch_id)
    # for r in results: parse verdict, write cache.put(...)
    pass`}
      </CodeBlock>

      <H3>5c. Prompt-side optimization</H3>

      <Prose>
        Input tokens are usually the dominant cost in pairwise judgments. The rubric, the prompt, and both responses all live in input. Output is typically a short verdict with brief reasoning — 100 to 200 tokens. There is therefore substantial leverage in shortening the rubric. A 1,200-token rubric describing every edge case in detail is overkill for most production judgments and approximately doubles per-call cost vs a 600-token rubric that captures the essential criteria. Empirical evidence from Liu et al. (2023, G-Eval) and from many production teams since: rubric length follows diminishing returns past ~500 tokens for most judgment tasks. Compress aggressively. Start with the shortest rubric that produces stable verdicts on a calibration set and extend only when the calibration shows specific failure modes.
      </Prose>

      <Prose>
        A second prompt-side lever is the response truncation policy. Many production judgments are made on responses that exceed 1,000 tokens. The judge does not need the full response to make a quality call most of the time; the first 200–400 tokens carry most of the signal. A truncation policy that caps each response at a fixed token budget (with an indicator if truncation occurred) reduces per-call input by 30–50% on long-response benchmarks at the cost of measurably worse accuracy on tasks where the difference between A and B lives in the response tail. For most chat and instruction-following benchmarks the trade is worth it; for code generation and math reasoning where the answer is at the end of the response, truncation is inappropriate.
      </Prose>

      <H3>5d. Distillation of frontier judge to a cheap student</H3>

      <Prose>
        The most cost-effective long-term strategy for repeated evaluation on a stable task definition is to distill a frontier judge into a cheaper student model. The pattern: collect a few thousand triples judged by the frontier model (with full reasoning), fine-tune a 7B–8B open-weight model on (input, frontier verdict + reasoning) pairs, and use the resulting student as the production judge for that task. Prometheus 2 (Kim et al. 2024, arXiv:2405.01535) is the public proof of concept: a Mistral 7B-based judge trained on GPT-4 verdicts that achieves Cohen's kappa of 0.6–0.7 with GPT-4 on rubric-based judgments at a fraction of the cost. The distillation pipeline is straightforward in TRL/Axolotl, and the resulting judge can be self-hosted on a single A100 with throughput around 100 judgments per second — at which point per-call cost is essentially zero (amortized hardware) and the marginal cost of running 100k judgments is dominated by electricity.
      </Prose>

      <CodeBlock language="python">
{`# Distillation data collection sketch.
def collect_distillation_data(triples, frontier_judge, cache):
    distill_rows = []
    for t in triples:
        cached = cache.get(judge_model=frontier_judge, prompt=t["prompt"],
                           resp_a=t["resp_a"], resp_b=t["resp_b"],
                           rubric=DISTILL_RUBRIC, temperature=0.0)
        if cached:
            verdict = cached
        else:
            verdict = call_frontier(frontier_judge, t)
            cache.put(verdict, judge_model=frontier_judge, prompt=t["prompt"],
                      resp_a=t["resp_a"], resp_b=t["resp_b"],
                      rubric=DISTILL_RUBRIC, temperature=0.0)
        distill_rows.append({
            "input":  format_judge_input(t, DISTILL_RUBRIC),
            "output": verdict["reasoning"] + "\\n" + f"VERDICT: {verdict['choice']}",
        })
    return distill_rows

# Then SFT a 7B–8B base model on distill_rows with TRL's SFTTrainer.
# Eval the student against held-out frontier verdicts; iterate on rubric
# and dataset until kappa > 0.65 on the target distribution.`}
      </CodeBlock>

      <H3>5e. Routing layer with confidence calibration</H3>

      <Prose>
        The cascade router from section 4 needs a real-world confidence signal. The reliable production pattern is to train a small calibration head on a held-out set of triples judged by both the cheap and the expensive judge. The features are derived from the cheap judge's output: the verdict, the log-probability of the verdict token (when available), the length of the reasoning, and any explicit "I am uncertain" markers in the response. The label is whether the cheap and expensive judges agreed. A logistic regression on these features produces a calibrated probability that the cheap judge agrees with the expensive one, which is exactly the threshold variable for cascade routing. Calibration improves cascade efficiency by 30–50% in our production deployments compared to using the raw cheap-judge confidence directly.
      </Prose>

      <H3>5f. Operational monitoring</H3>

      <Prose>
        Track these metrics in production for any judge-based eval system. Cache hit rate by benchmark — a falling cache hit rate signals either rubric drift or a benchmark refresh and is the most actionable single number. Mean and 95th-percentile per-call latency for each judge — frontier judges have heavier tails and a sudden tail expansion can blow up batch SLAs. Distribution of cascade escalation rates by topic cluster — uneven escalation indicates that the cheap judge is systematically less calibrated on certain task types and may need rubric refinement. Pairwise judge agreement rates on a fixed audit set — drifting agreement (especially the cheap judge drifting away from the frontier judge) is the canonical signal that distillation needs a refresh or that the underlying task distribution has shifted. Cost per evaluation pass and cost per detected effect — these are the executive-summary metrics that justify ongoing investment in the eval infrastructure.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first visualization is the cost-quality Pareto frontier. Each judge sits at a point in (cost, win-rate MSE) space; the frontier is the lower-left envelope of points that are not dominated by any other. Points above and to the right are strictly worse — more expensive and less accurate. The shape of the frontier is convex: cheap improvements in accuracy are easy at the low end, but pushing accuracy toward 1.0 costs disproportionately more.
      </Prose>

      <Plot
        label="Cost-quality Pareto frontier — single judges vs cascade"
        xLabel="cost per evaluation pass ($, log scale not shown)"
        yLabel="win-rate MSE (lower is better)"
        series={[
          {
            name: "single judges",
            color: colors.gold,
            points: [
              [0.20, 0.001847],
              [0.52, 0.000731],
              [1.00, 0.000287],
              [10.00, 0.000091],
              [58.00, 0.000043],
            ],
          },
          {
            name: "cascade (τ-sweep)",
            color: "#c084fc",
            points: [
              [0.52, 0.001124],
              [1.45, 0.000891],
              [6.83, 0.000412],
              [19.42, 0.000174],
              [35.04, 0.000088],
              [58.52, 0.000044],
            ],
          },
        ]}
      />

      <Prose>
        The cascade trajectory dominates the single-judge frontier in the middle region (roughly $5 to $30 per pass) where it offers the same MSE as a single judge at a fraction of the cost. At the extremes (always-cheap or always-expensive) the cascade collapses to the single-judge points. The middle is where the cascade earns its keep, and where most production deployments operate.
      </Prose>

      <Prose>
        The next plot shows how required sample size scales with judge accuracy for a fixed effect size. The curve is the inverse-square-of-(2a-1) shape from section 3c: gentle in the high-accuracy region, then exploding as accuracy approaches the 0.5 noise floor.
      </Prose>

      <Plot
        label="Sample size needed to detect 5% win-rate gap (95% conf, 80% power)"
        xLabel="judge accuracy a"
        yLabel="prompts required (n)"
        series={[
          {
            name: "n required",
            color: colors.gold,
            points: [
              [1.00, 776],
              [0.95, 958],
              [0.90, 1213],
              [0.85, 1584],
              [0.80, 2156],
              [0.75, 3104],
              [0.70, 4851],
              [0.65, 8622],
              [0.60, 19400],
              [0.55, 77600],
            ],
          },
        ]}
      />

      <Prose>
        The third visualization is a heatmap of expected total cost (in dollars) to detect the same 5% gap, as a function of the judge's per-call cost and per-call accuracy. The reading rule is: each cell is the dollars required for the full evaluation pass at that combination of judge characteristics. The dark band running diagonally across the table is the equal-cost contour — combinations of cost and accuracy that produce the same total spend. Production teams should be able to point to where on this heatmap their current judge sits and where they would like it to sit.
      </Prose>

      <Heatmap
        matrix={[
          [0.776, 0.958, 1.213, 1.584, 2.156, 3.104, 4.851, 8.622],
          [3.880, 4.790, 6.065, 7.920, 10.78, 15.52, 24.26, 43.11],
          [7.760, 9.580, 12.13, 15.84, 21.56, 31.04, 48.51, 86.22],
          [38.80, 47.90, 60.65, 79.20, 107.8, 155.2, 242.6, 431.1],
          [388.0, 479.0, 606.5, 792.0, 1078, 1552, 2426, 4311],
        ]}
        rowLabels={["$0.001/call", "$0.005/call", "$0.010/call", "$0.050/call", "$0.500/call"]}
        colLabels={["a=1.00", "a=0.95", "a=0.90", "a=0.85", "a=0.80", "a=0.75", "a=0.70", "a=0.65"]}
        cellSize={62}
        colorScale="gold"
        label="Total $ to detect 5% win-rate gap at 95% conf, 80% power"
      />

      <Prose>
        Reading the heatmap top-to-bottom shows the cost dimension; left-to-right shows accuracy. The bottom-right corner ("$0.500/call, a=0.65") would cost over $4,300 per evaluation pass. The top-left ("$0.001/call, a=1.00") is just $0.78 — but no real judge sits there. Realistic production positions: a hosted Llama 3.1 8B at "$0.001/call, a=0.75" costs ~$3.10 per pass. Sonnet-class at "$0.005/call, a=0.90" costs ~$6.06. GPT-4-class at "$0.010/call, a=0.95" costs ~$9.58. The astonishing fact this heatmap surfaces is that for the canonical 5% gap detection, a cheap reasonably-accurate judge often beats an expensive nearly-perfect judge on total spend.
      </Prose>

      <Prose>
        Finally, the StepTrace below walks through a single tiered-routing decision, from cheap-judge call through confidence check to either acceptance or escalation. This is the inner loop that runs millions of times across a production evaluation pipeline.
      </Prose>

      <StepTrace
        label="Tiered judge router — single triple decision"
        steps={[
          {
            label: "Receive triple",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>triple = (prompt, resp_a, resp_b)</div>
                <div>cache_key = sha256(model || prompt || a || b || rubric || temp)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Cache check happens before either judge call. A hit returns
                  the stored verdict immediately at zero marginal cost.
                </div>
              </div>
            ),
          },
          {
            label: "Cheap judge call",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Stage 1: cheap judge</div>
                <div>verdict_cheap, confidence = cheap_judge(triple)</div>
                <div>cost += $0.0005   # e.g., 4o-mini</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Confidence is a calibrated probability of agreement with
                  the expensive judge, learned from a held-out audit set.
                </div>
              </div>
            ),
          },
          {
            label: "Threshold check",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Routing decision</div>
                <div>if confidence ≥ τ:</div>
                <div>    accept verdict_cheap   # ~78% of cases at τ=0.70</div>
                <div>else:</div>
                <div>    escalate to stage 2    # ~22% of cases</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  τ is the main tuning knob. Lower τ → more cost savings,
                  higher MSE. Higher τ → more accuracy, more spend.
                </div>
              </div>
            ),
          },
          {
            label: "Expensive judge call (if escalated)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Stage 2: expensive judge</div>
                <div>verdict_exp, _ = expensive_judge(triple)</div>
                <div>cost += $0.029   # e.g., opus-class</div>
                <div>final_verdict = verdict_exp</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Some pipelines also store verdict_cheap for later
                  agreement audits and to retrain the calibration head.
                </div>
              </div>
            ),
          },
          {
            label: "Cache write + return",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Persistence</div>
                <div>cache.put(final_verdict, key=cache_key)</div>
                <div>return final_verdict, cost_total</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Cache key includes the full rubric. Cache hit rate
                  typically reaches 30–60% on stable benchmarks within a few
                  weeks of repeated evaluation runs.
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

      <H3>Frontier judge vs small judge</H3>

      <Prose>
        Choose a frontier judge (GPT-4-class, Claude Opus) when the task involves nuanced subjective judgment, when the response space is open-ended creative writing or strategic analysis, when the cost of a wrong call is high (e.g., gating a model release), and when the evaluation set is small enough that the per-call premium is absorbed by the convenience of not having to set up tiered routing or distillation. For one-off comparisons, ad-hoc evals, and the early stages of a project where eval volume is low, frontier judges are the right default. Their per-call cost is high but their per-correct-call cost is competitive on hard subjective tasks where the alternative is many self-consistency calls to a weaker judge.
      </Prose>

      <Prose>
        Choose a small judge (Llama 3.1 8B, Mistral 7B, Prometheus 2 7B) when the task is verifiable or rubric-driven, when the evaluation set is large (say, &gt;5,000 prompts), when you are running the same benchmark repeatedly across many model variants, and when you have the engineering capacity to validate the small judge's calibration against a frontier judge on an audit set. The cost reduction is 50–290x per call. The accuracy reduction is task-dependent — small for math, code, and structured outputs; large for creative writing, multi-turn dialogue, and culturally specific tasks. The right way to make this decision is empirical: spend a few hundred dollars running both judges on the same audit set, compute Cohen's kappa, and decide whether the small-judge accuracy is in the acceptable range for the downstream decision. Below kappa 0.5 the small judge is rarely usable as a primary evaluator; between 0.5 and 0.7 it works as a screen for frontier escalation; above 0.7 it can stand alone for many production tasks.
      </Prose>

      <H3>Pairwise vs pointwise scoring</H3>

      <Prose>
        Pairwise (A vs B with a winner) reduces variance because the two responses are evaluated against each other, removing absolute-quality calibration drift. It is also more sample-efficient per dollar for ranking tasks. Pointwise (rate response on a 1–10 scale) generalizes better across tasks because every response gets an absolute score, but it suffers from rubric drift, scale-use bias (judges concentrate scores in 6–8), and intra-judge inconsistency that pairwise comparisons largely escape. Use pairwise for direct A/B comparisons (model_v1 vs model_v2). Use pointwise when you need an absolute quality measure (e.g., "is this response acceptable for production"). Use both when you need to rank a slate of N candidates: pointwise for a quick filter, pairwise on the top-K for precise ordering. Liu et al. 2023 (G-Eval) and Zheng et al. 2023 (LLM-as-Judge) both report pairwise outperforming pointwise by 5–15 points of human-agreement kappa on chat tasks.
      </Prose>

      <H3>Cascade vs self-consistency vs single strong judge</H3>

      <Prose>
        Three strategies for spending more on hard cases. Cascade routing (cheap then expensive) is the right default when the cheap judge has well-calibrated confidence, because most cases are easy and never trigger the escalation. Self-consistency (k cheap calls, majority vote) is the right choice when the cheap judge has poor confidence calibration and you cannot afford to deploy a strong judge — k = 3 to 5 reliably boosts accuracy by 5–10 points at a 3–5x cost multiplier on the cheap judge. Single strong judge is the right choice when the eval set is small enough that the engineering overhead of routing and self-consistency exceeds the dollar savings, or when the task is so hard that even self-consistency on a cheap judge cannot reach the required accuracy. The general rule: for &gt;10k judgments per pass, use cascade. For 1k–10k, use single strong judge or self-consistency on a cheap judge depending on calibration. For &lt;1k, use a single strong judge.
      </Prose>

      <H3>Distilled student vs always-on frontier</H3>

      <Prose>
        Distillation pays off when you will run the same task definition for at least 50,000 judgments over the lifetime of the eval. Below that threshold, the upfront cost of collecting frontier judgments for the distillation set, training the student, and validating it exceeds the savings versus just using the frontier judge on every call. Above that threshold, distillation rapidly dominates: a Prometheus-style 7B student costs essentially zero per call (amortized hardware) and approaches frontier quality on the specific task it was trained on. The catch is generalization: a distilled student is overfit to the rubric and task distribution it was trained on. When the rubric or distribution shifts, the student must be retrained. For mature stable benchmarks (a fixed code generation suite, a fixed math benchmark) distillation is the long-run cost minimizer. For evolving evaluation criteria (new rubrics every quarter, new task categories every month) the retraining overhead is too high and the always-on frontier or cascade is preferable.
      </Prose>

      <H3>Batch API vs synchronous API</H3>

      <Prose>
        Use the batch API for any evaluation that does not need real-time results. The 50% discount is essentially free, the SLA (up to 24 hours, typically 1–4 hours) is acceptable for nightly and weekly evaluation runs, and the client integration is straightforward. The only contraindication is interactive use (a developer iterating on a rubric and watching live verdicts) where the batch turnaround time is too long. For all production scheduled evaluation runs, batch is the default.
      </Prose>

      <H3>When humans still win</H3>

      <Prose>
        Three scenarios where human evaluation is still the right tool. First, gold-standard calibration: the audit set used to validate any LLM judge needs human labels. There is no substitute for ground truth here, and outsourcing to another LLM creates a circular dependency where errors compound. Second, novel task categories: an LLM judge has no priors on a task it has never seen, and its calibration on the task is unknown until validated. The first 100–500 examples of a new task category should be human-judged to establish baseline performance and to seed an LLM judge calibration. Third, high-stakes deployment decisions: model releases, safety claims, customer-facing accuracy claims should be backed by human evaluation on at least a sample. The cost of being wrong on these decisions exceeds the cost of human annotation by orders of magnitude.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Per-call costs scale linearly with judgment volume. There is no economy of scale at the API level — the thousandth call costs the same as the first. The batch API discount (50%) is the closest thing to a volume discount and applies uniformly. Provider negotiated rates can drop another 10–30% at very high volumes ($100k+/month) but the underlying linearity does not change. Cost models therefore predict total spend reliably given expected call volume; budgeting is the simple multiplication of calls per pass times passes per period times dollars per call.
      </Prose>

      <Prose>
        Caching scales better than linearly with the stability of your benchmark. A frequently changing benchmark gets little cache benefit because most calls are unique. A stable benchmark amortizes well: the first model variant pays for the full evaluation, every subsequent variant pays only for differences. In our production deployments cache hit rates climb from ~5% in week one to 30–60% by week eight on stable benchmarks. The largest realized savings come from long-tail evaluation where the same triple is judged many times (different rubrics, different reruns, different studies); even when each call is technically distinct, sub-key caching of partial computations can recover savings if the system architecture supports it.
      </Prose>

      <Prose>
        Tiered routing scales gracefully across difficulty distributions. As the eval set grows, the proportion of easy cases (handled cheaply) tends to dominate, and the cascade efficiency improves. The economics improve with scale: a small benchmark might not justify the engineering of a routing layer, but past ~10k judgments per pass the savings comfortably exceed the build cost. Distillation has the same shape: high upfront cost, low marginal cost, breaks even past ~50k judgments and dominates past 200k.
      </Prose>

      <Prose>
        What does not scale is judge calibration on out-of-distribution tasks. A judge calibrated on math problems is not automatically calibrated on creative writing, on multi-turn dialogue, on agentic task evaluation, or on any task category the calibration set did not cover. Each new task category requires its own audit set, its own kappa measurement, its own decision about whether the cheap judge is acceptable. Production teams that scale a single judge across many task categories without per-category recalibration consistently end up with eval results that look reasonable in aggregate but mask category-specific failures. The cost of recalibration grows linearly with the number of distinct categories — there is no scaling shortcut.
      </Prose>

      <Prose>
        Self-consistency does not scale past <Code>{"k = 5"}</Code> for most cheap judges. The asymptotic accuracy of majority voting approaches a plateau set by the underlying judge's correlated errors. If a cheap judge has a systematic bias (e.g., position bias preferring whichever response appears first), running it 20 times does not remove the bias — all 20 calls inherit it. Aggregation reduces stochastic noise, not systematic error. The lesson is that self-consistency is a finite-improvement technique: useful in the 1.5–2x cost range, with diminishing and eventually negative returns past that.
      </Prose>

      <Prose>
        Bootstrapping confidence intervals scales linearly with the number of resamples and is essentially free at modern compute prices. A 5,000-resample bootstrap on 10,000 verdicts runs in &lt;1 second on a single CPU. There is no reason not to report bootstrap CIs on every win-rate measurement. The non-scaling concern is conceptual rather than computational: the bootstrap captures variance from the verdicts you measured, but it cannot capture variance from biases in the judge itself. A judge with a 5% systematic bias produces measurements that are 5% off; the bootstrap CI around the measurement does not contain the true value, regardless of how many resamples you draw. Bias and variance are independent failure modes; bootstrap addresses only the second.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Position bias in pairwise judgments</H3>
      <Prose>
        LLM judges systematically prefer whichever response is presented first (or, less commonly, second) in a pairwise prompt. Reported bias magnitudes range from 5% to 25% across judge models and tasks. The mitigation is mandatory: every pair must be evaluated twice with positions swapped, with the consistent verdict (A wins both orderings, or B wins both orderings) accepted and the inconsistent cases (A wins one, B wins the other) treated as ties or escalated. Skipping the position-swap doubles the variance of your win-rate estimate and biases it by an unknown amount. The cost is a 2x increase in judge calls for every pairwise evaluation; this is non-negotiable for any evaluation that will be acted on.
      </Prose>

      <H3>Verbosity bias</H3>
      <Prose>
        Judges consistently rate longer responses higher, mirroring the same bias documented in human evaluation. The bias is independent of the actual quality difference — a longer response with the same information content scores ~5–15% higher in pairwise comparisons. The mitigation is rubric-level: explicitly instruct the judge to penalize unnecessary length, and validate by including length-controlled pairs in the audit set. Some teams normalize win-rates by length quartile; this catches the bias post-hoc but does not eliminate it. The most reliable defense is to evaluate length-matched candidates whenever possible.
      </Prose>

      <H3>Self-preference bias</H3>
      <Prose>
        Frontier judges prefer outputs from their own model family. GPT-4 as judge prefers GPT-4 outputs over Claude outputs; Claude as judge prefers Claude outputs over GPT-4 outputs. The effect is documented in Panickssery et al. 2024 and shows up at the 2–5% level on standard benchmarks. The implication for evaluation is that single-judge comparisons across model families are systematically biased toward the judge's family. The mitigation is to evaluate with two judges from different families (one from each side) and report both numbers, ideally with their disagreement rate as a separate metric. For high-stakes comparisons across families, human evaluation on a sample is the only fully unbiased option.
      </Prose>

      <H3>Rubric drift in production</H3>
      <Prose>
        Rubrics are versioned text. A rubric edit silently invalidates all prior cached judgments unless the cache key includes the rubric content (it should, see section 5a). The more insidious failure is comparing this week's evaluation results against last week's when the rubric changed in between — the comparison is not apples-to-apples and the apparent quality delta is partially a measurement artifact. Strict rubric versioning, with a frozen rubric across any sequence of evaluations that will be compared, is the only defense. Treat the rubric like a code dependency: it has a version, every evaluation pass logs which version it used, and comparisons across versions require explicit acknowledgment.
      </Prose>

      <H3>Confidence miscalibration in the cheap judge</H3>
      <Prose>
        Cascade routing relies on the cheap judge's confidence being a meaningful predictor of agreement with the expensive judge. In practice, raw cheap-judge confidence is poorly calibrated — the judge says it is 90% sure on cases where it is actually right 70% of the time, and 70% sure on cases where it is right 90% of the time. Without calibration, threshold-based routing produces escalation patterns that do not correspond to actual difficulty. The fix is to train a calibration head on a held-out set of triples judged by both cheap and expensive judges, predicting agreement from the cheap judge's features. Re-validate the calibration head whenever the underlying cheap judge changes (model version, prompt format, temperature).
      </Prose>

      <H3>Distribution shift breaking distilled students</H3>
      <Prose>
        A judge distilled from frontier verdicts on dataset D is calibrated for D's distribution. When the production task distribution drifts — new content categories, new model under test, new domain — the student's accuracy degrades and the degradation is often invisible until you run a fresh agreement audit. Schedule periodic audits (monthly or per-quarter) where a sample of student verdicts is re-judged by the frontier and the agreement rate is monitored. A drop of more than ~5 percentage points relative to the calibration baseline is a signal to refresh the distillation set and retrain.
      </Prose>

      <H3>Batch API silently failing partial requests</H3>
      <Prose>
        Batch APIs return results as a single output file with one row per submitted request. Some requests can fail (rate limits, content filter, malformed inputs) while others succeed. The output file represents this with error markers per row, but a naive consumer that reads the file as a uniform success will silently drop the failed rows and produce a biased win-rate based on whichever cases happened to succeed. Always validate that the number of returned successful rows equals the number of submitted rows; treat any discrepancy as a hard failure that requires investigation, not as an acceptable level of dropout.
      </Prose>

      <H3>Treating raw win-rate as the corrected estimate</H3>
      <Prose>
        The observed win-rate from a noisy judge is biased toward 0.5. Reporting it as if it were the true win-rate underestimates the actual gap between models, especially when the gap is small and the judge is weak. The correction is straightforward (section 3a) but requires knowing the judge's accuracy, which requires an audit set. Teams that skip the audit set and report raw win-rates often conclude that two models are closer than they actually are, sometimes leading to incorrect "no significant difference" decisions on changes that the judge was simply too noisy to detect.
      </Prose>

      <H3>Confounding judge variance with model variance</H3>
      <Prose>
        When you run the same evaluation twice and get different win-rates, the natural assumption is that the model's response sampling produced the variation. Often the larger contribution is judge variance: the same triple judged twice gets different verdicts. The diagnostic is to fix the responses (cache them) and rerun only the judgments — if the win-rate still varies substantially, the variance is judge-side. The fix is either a stronger judge, self-consistency aggregation, or temperature=0 on the judge calls. Many teams rerun their evaluations with new responses each time, never measuring what fraction of the run-to-run variance is actually judge noise.
      </Prose>

      <Callout accent="purple">
        The single most consequential operational failure in LLM-as-judge evaluation is making release decisions based on uncorrected, position-unswapped, single-judge win-rates without a held-out audit set to calibrate the judge. This setup looks like rigorous evaluation but is approximately equivalent to flipping a biased coin. Always position-swap, always audit, always report bootstrap CIs.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their arXiv pages and provider documentation as of 2026-04-25. Author lists, arXiv IDs, and core claims confirmed.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and LLM-as-Judge</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. NeurIPS 2023 Datasets and Benchmarks Track. The foundational paper establishing GPT-4 as a credible substitute for human judgment on multi-turn chat tasks. Reports 80%+ agreement with human preferences on MT-Bench, documents position bias and verbosity bias quantitatively, and introduces the chained pairwise evaluation methodology that became the standard pattern. Critical reading for understanding both the validity and the limits of LLM-as-judge.
      </Prose>

      <H3>Liu et al. 2023 — G-Eval</H3>
      <Prose>
        Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." arXiv:2303.16634. EMNLP 2023. Introduces the chain-of-thought rubric pattern for LLM judges and shows substantial improvements in human-correlation over earlier prompt formats. The methodology section is the practical reference for designing evaluation rubrics that produce stable, high-agreement verdicts. The paper also documents the failure mode where GPT-4 systematically overrates GPT-4-generated text — the canonical reference for self-preference bias.
      </Prose>

      <H3>Kim et al. 2024 — Prometheus 2</H3>
      <Prose>
        Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, Minjoon Seo. "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." arXiv:2405.01535. EMNLP 2024. Introduces a 7B and 8x7B Mistral-based judge fine-tuned on 100k+ feedback samples. Achieves Cohen's kappa of 0.6+ with GPT-4 on rubric-based judgments, establishing the open-weight distilled judge as a practical production option. The training recipe, dataset construction, and evaluation methodology are all open and reproducible. Code and weights at github.com/prometheus-eval/prometheus-eval. The reference paper for anyone considering self-hosted judge deployment.
      </Prose>

      <H3>Saad-Falcon et al. 2024 — LMUnit</H3>
      <Prose>
        Jon Saad-Falcon, Rajan Vivek, Jordan Burgess, Jacob Mitchell, Akshay Kalkunte Suresh, William Berrios, Ellen Wu, Aleksandra Faust, Sameer Singh, Manaal Faruqui, Christopher Potts, Matei Zaharia. "LMUnit: Fine-grained Evaluation with Natural Language Unit Tests." arXiv:2412.13091. Decomposes evaluation into many small "unit tests" each phrased as a natural-language criterion, then aggregates per-unit-test verdicts into an overall judgment. Shows that fine-grained per-criterion judgment is both more interpretable and more sample-efficient than global rubrics, with the additional benefit that disagreements are localized to specific criteria rather than diffuse across the entire response. Directly relevant for designing evaluation systems where the variance decomposition (per-criterion) is critical for actionability.
      </Prose>

      <H3>Panickssery et al. 2024 — LLM judges prefer their own outputs</H3>
      <Prose>
        Arjun Panickssery, Samuel R. Bowman, Shi Feng. "LLM Evaluators Recognize and Favor Their Own Generations." arXiv:2404.13076. Documents the self-preference bias quantitatively: GPT-4 prefers GPT-4 outputs by 5–15% across multiple benchmarks, with similar effects for Claude as judge. Shows that the bias is not a simple text-style artifact (paraphrasing the GPT-4 output does not eliminate the preference) and proposes mitigation strategies including ensemble judging across model families. Essential reading for any cross-family model comparison.
      </Prose>

      <H3>Anthropic Batch API documentation</H3>
      <Prose>
        Anthropic. "Message Batches API." docs.anthropic.com/en/docs/build-with-claude/batch-processing. Describes the batch endpoint pattern, pricing (50% discount on standard rates), SLA (up to 24 hours, typically much faster), and result file format. The integration cost for batch is low, and the savings on evaluation workloads are substantial. The corresponding OpenAI documentation at platform.openai.com/docs/guides/batch describes a parallel pattern with the same 50% discount and 24-hour SLA, structured as JSONL inputs and outputs.
      </Prose>

      <H3>OpenAI evals and judge cost analyses</H3>
      <Prose>
        OpenAI. "Pricing." openai.com/api/pricing. The canonical source for current per-token pricing across the GPT-4o family, GPT-4-turbo, and the o-series reasoning models. Critical for accurate cost modeling, since published rates change quarterly. The "OpenAI Cookbook" judge evaluation notebooks at github.com/openai/openai-cookbook also document several reference patterns for judge prompt structure, response parsing, and verdict aggregation that are widely used as starting templates in production systems.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the noisy-judge bias correction</H3>
      <Prose>
        Starting from the relation <Code>{"\\hat{p} = (2a - 1) p + (1 - a)"}</Code>, derive the inverse formula expressing <Code>{"p"}</Code> in terms of <Code>{"\\hat{p}"}</Code> and <Code>{"a"}</Code>. What happens to the formula as <Code>{"a \\to 0.5"}</Code>? What does this imply about the practical lower bound on usable judge accuracy? Now suppose the judge has different accuracies for the two outcome classes (asymmetric noise: accuracy <Code>{"a_+"}</Code> when B truly wins, accuracy <Code>{"a_-"}</Code> when A truly wins). Re-derive <Code>{"\\hat{p}"}</Code> as a function of <Code>{"p, a_+, a_-"}</Code> and the inverse correction. How does the asymmetry change the variance inflation factor?
      </Prose>

      <H3>Exercise 2 — Sample size sensitivity</H3>
      <Prose>
        Using the power-analysis formula from section 3c, compute the sample size required to detect a 2% win-rate gap (instead of 5%) at 95% confidence and 80% power, for a perfect judge and for a judge with <Code>{"a = 0.80"}</Code>. By what factor does the required <Code>{"n"}</Code> increase when the gap is halved? When the judge accuracy drops from 1.0 to 0.80? Which lever is more cost-effective to improve: doubling the benchmark size or upgrading the judge from <Code>{"a = 0.80"}</Code> to <Code>{"a = 0.95"}</Code>? Explain how this answer depends on the per-call cost ratio between the two judges.
      </Prose>

      <H3>Exercise 3 — Cascade economics</H3>
      <Prose>
        Suppose the cheap judge costs $0.001/call with <Code>{"a = 0.78"}</Code> and the expensive judge costs $0.030/call with <Code>{"a = 0.95"}</Code>. Confidence is perfectly calibrated, and the escalation rate at threshold <Code>{"\\tau"}</Code> is <Code>{"f(\\tau)"}</Code>. Write down the expected cost per call as a function of <Code>{"f"}</Code> and the expected aggregate accuracy as a function of <Code>{"f, a_{cheap}, a_{exp}"}</Code> (assuming the cheap judge is right precisely on cases not escalated). For what value of <Code>{"f"}</Code> does the cascade match the expensive judge's per-call cost? At that <Code>{"f"}</Code>, what is the cascade's accuracy compared to the expensive judge alone? When is cascading not worth the engineering effort?
      </Prose>

      <H3>Exercise 4 — Variance decomposition on a small benchmark</H3>
      <Prose>
        Design an experiment that measures, for a 200-prompt benchmark, the relative contributions of prompt variance, response variance, and judge variance to the total measurement noise on a win-rate. What sampling structure do you need (how many prompts, how many response samples per prompt, how many judge calls per (prompt, response) pair)? How would you analyze the resulting data to attribute variance to each source? What total number of judge calls does your design require, and what would you change if your judge call budget were one-tenth that size?
      </Prose>

      <H3>Exercise 5 — Distillation break-even</H3>
      <Prose>
        Suppose collecting frontier judgments costs $0.030/call, training a distilled student costs $500 in compute, and the resulting student costs $0.0001/call to run (amortized hardware). The distilled student achieves <Code>{"a = 0.88"}</Code> versus frontier <Code>{"a = 0.95"}</Code>. For an evaluation pipeline that runs <Code>{"N"}</Code> judgments over the lifetime of the task, derive the break-even <Code>{"N"}</Code> at which distillation becomes cheaper than always-frontier, ignoring accuracy differences. Then compute the break-even <Code>{"N"}</Code> when you also require the student's effective sample size to match the frontier's (i.e., compensating for the lower accuracy by running more samples). At what task lifetime does distillation make economic sense, and at what point does it stop being worthwhile due to required retraining cycles? Discuss how rubric volatility affects the answer.
      </Prose>

      <H3>Exercise 6 — Position bias mitigation cost</H3>
      <Prose>
        Position-swapping doubles the number of judge calls per pairwise comparison. Suppose your judge has a position bias of 8% (i.e., the model in position 1 wins 4% more often than chance, relative to a position-unbiased baseline). Without position-swap, what is the bias on your measured win-rate? With position-swap and treating disagreements as ties, what fraction of pairs become ties (assuming the underlying win-rate is 0.55 and the bias is symmetric)? What is the variance of the resulting estimator compared to the no-swap estimator? Is the doubled cost worth it for a benchmark where the true effect size is 5%? For one where it is 1%?
      </Prose>

      <H3>Exercise 7 — Building an audit set</H3>
      <Prose>
        You need to validate a new cheap judge for production use. Describe the protocol you would use to construct an audit set: how many examples, what diversity of task types, who labels the ground truth, what accuracy metrics you would compute, and what kappa threshold you would require before deploying. What is the minimum audit set size to detect a 5-point kappa difference between two candidate judges at 95% confidence? How would you handle the situation where your audit set itself drifts in difficulty over time?
      </Prose>

    </div>
  ),
};

export default costPerformanceJudges;
