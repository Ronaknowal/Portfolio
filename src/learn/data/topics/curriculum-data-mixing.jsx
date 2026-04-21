import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const curriculumDataMixing = {
  title: "Curriculum Learning & Data Mixing Strategies",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A pre-training corpus is never a single thing. It is a weighted sum: some fraction of filtered web crawl, some fraction of code from GitHub, some fraction of books, some fraction of scientific papers, some fraction of math derivations scraped from textbooks and arXiv LaTeX. Those fractions are not dictated by the natural availability of the data — there is vastly more web text than code in the universe, but modern LLMs do not train at that ratio, because the marginal token of code teaches the model more than the marginal token of boilerplate web prose. The ratio is a hyperparameter. Getting it wrong is among the most expensive errors you can make at pre-training scale. Llama 2 trained with roughly 4.5% code in its mixture. Llama 3 pushed code and math to a combined ~17%. The architecture barely moved between the two. The gap on HumanEval, MATH, and GSM8k moved by double digits. What changed was the mix.
      </Prose>

      <Prose>
        Curriculum is the other half. A mix tells you <em>what</em> the model sees. A curriculum tells you <em>when</em> the model sees it. At the scale of modern pre-training — trillions of tokens, each seen roughly once — the order of the bulk of training matters less than you would expect, because every concept is encountered many times across the corpus regardless of shuffling. But the edges matter a lot. The first few thousand steps, when the learning rate is warming up and the optimizer is still finding its footing, are sensitive to data pathology: a burst of repetitive or low-quality tokens early can poison the first representations the model builds. And the final 5–10% of training, when the learning rate is decaying, is sensitive in the other direction: a shift to a higher-quality, curated mixture at the end pushes the model's final weights toward exactly the distribution it will be evaluated on. Llama 3 reported that annealing their 8B on a curated mix for the final 40M tokens improved GSM8k by 24 points and MATH by 6.4 points — a change that costs no extra FLOPs, only a reshuffling of the schedule.
      </Prose>

      <Prose>
        The intellectual lineage is longer than the recent surge suggests. Bengio, Louradour, Collobert, and Weston formalized "curriculum learning" at ICML 2009, arguing by analogy to how humans and animals are taught — start with simple examples, graduate to harder ones. For a decade the idea lived in computer vision and reinforcement learning, where it worked well on small models and small datasets. Transferring it to language model pre-training at scale took most of another decade and produced a more complicated picture: naive difficulty-based curricula don't help much past ~1B parameters, but two specific variants — mixture optimization (choosing <em>how much</em> of each domain) and cooldown-on-curated-data (choosing <em>what</em> to end on) — produce consistent, reproducible gains. DoReMi (Xie et al., NeurIPS 2023) and DoGE (Fan et al., 2023) automated the mixture search. Ye et al. (2024) proposed "data mixing laws" that extrapolate mixture performance the way Chinchilla extrapolates compute. Llama 3 and DeepSeek-V3 published enough of their cooldown recipes that the practice has moved from folklore to engineering.
      </Prose>

      <Prose>
        This topic is about treating the mix and the curriculum as first-class pre-training hyperparameters — as load-bearing as learning rate, batch size, and model width, and decided by the same mix of theory and systematic search. Ignore them and you leave on the table gains that no amount of architecture tuning will recover. The rest of the document walks through the math, builds working implementations of the five central algorithms in roughly fifty lines of Python each, surveys the production recipes, and lays out the failure modes that reliably catch first-timers.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Three ideas, taken together, cover most of what curriculum-and-mixing is about. Each is simple on its own; the interactions are where the interesting decisions live.
      </Prose>

      <Prose>
        <strong>Idea 1 — the corpus is a weighted sum of domains.</strong> Write <Code>P(x) = Σ w_i · P(x | domain_i)</Code>. The weights <Code>w_i</Code> determine what the model learns. Double the code weight and the model gets better at code, usually at some cost to natural-language fluency. Halve it and the model writes clean prose but cannot implement a binary search. There is no abstract "correct" set of weights — the correct weights depend on what you want the model to be good at, which domains you have data for, and how the domains interact during training. But the choice is continuous and consequential: the difference between a mediocre model and a frontier model on any given benchmark is often a handful of percentage-point shifts in the mix.
      </Prose>

      <Prose>
        <strong>Idea 2 — order matters at the edges, not in the middle.</strong> Run a pre-training job and divide it into three phases: the warmup (first 1–2% of steps), the bulk (middle 90%), and the cooldown (last 5–10%). Within the bulk, random shuffling of the mixed corpus works well and staged curricula usually do not improve on it in blind A/B tests. The statistical reason is that every concept is encountered many times during the bulk and the order-of-exposure signal is dominated by the signal from sheer repetition. But at the warmup, the first few thousand batches set the initial representations and the model is in a regime where individual examples have outsized effect; high-quality, well-behaved data here pays off. And at the cooldown, the learning rate is small enough that the final examples disproportionately shape the final weights; curated data here pushes the model toward the evaluation distribution.
      </Prose>

      <Prose>
        <strong>Idea 3 — cooldown sharpens; it does not replace training.</strong> A cooldown on curated data is not a substitute for training on a good mixture — it is a refinement on top of one. Models trained on a bad general mix and then cooled down on curated data do not recover; the cooldown sharpens whatever the bulk phase built. Think of it as the last 5% of a polishing sequence rather than a remediation step. The standard recipe at the frontier labs — Llama 3, DeepSeek-V3, Gemma — is to cool down on a carefully filtered subset of the main corpus that excludes benchmark-adjacent content (to avoid eval leakage) and upsamples high-quality domains (textbooks, math derivations, deduplicated long-form code, reasoning-heavy web text).
      </Prose>

      <Prose>
        The mental model to carry through the rest of the topic: <strong>a pre-training run is a distribution over tokens unrolled in time</strong>. Data mixing chooses the distribution. Curriculum chooses how the distribution shifts across time. The mass of the run lives in the bulk where neither choice has much leverage; the leverage lives at the boundaries, where a few percent of the tokens shape a disproportionate amount of the final model's behavior.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Start with the mixture. Let <Code>D = {"{D_1, ..., D_k}"}</Code> be <Code>k</Code> domains (web, code, math, books, papers), and let <Code>w = (w_1, ..., w_k)</Code> be a probability simplex over them — non-negative, sums to one. The sampling distribution over tokens during a training step is a mixture:
      </Prose>

      <MathBlock>
        {"P(x) = \\sum_{i=1}^{k} w_i \\cdot P(x \\mid D_i), \\qquad w_i \\geq 0, \\qquad \\sum_{i=1}^{k} w_i = 1"}
      </MathBlock>

      <Prose>
        During training, the per-step loss is a weighted average of per-domain losses. If <Code>L_i(θ)</Code> is the expected loss of model parameters <Code>θ</Code> on domain <Code>D_i</Code>, the effective training objective is:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(\\theta; w) = \\sum_{i=1}^{k} w_i \\cdot L_i(\\theta)"}
      </MathBlock>

      <Prose>
        This is where the choice of <Code>w</Code> exerts its influence. A gradient descent step on <Code>ℒ(θ; w)</Code> with respect to <Code>θ</Code> moves the model along a direction that is a weighted average of per-domain gradients. Upweight code and the step is pulled harder toward reducing code loss; the tradeoff with other domains is exactly what the weights encode.
      </Prose>

      <Prose>
        <strong>Temperature sampling.</strong> A recurring problem is that the natural distribution of data across domains is skewed: there is enormously more English web text than Hindi web text, but a multilingual model is supposed to serve both. Temperature sampling softens the natural distribution. Given natural proportions <Code>p_i</Code>, the temperature-adjusted weights are:
      </Prose>

      <MathBlock>
        {"w_i(\\alpha) = \\frac{p_i^{\\alpha}}{\\sum_{j=1}^{k} p_j^{\\alpha}}"}
      </MathBlock>

      <Prose>
        At <Code>α = 1</Code> the mixture matches the natural distribution. At <Code>α = 0</Code> it is uniform across domains regardless of their natural sizes. Values in between interpolate — <Code>α = 0.3</Code> is a standard choice for multilingual pretraining (mBART, XLM-R use similar settings) because it meaningfully boosts low-resource languages without completely flattening the distribution. The choice of <Code>α</Code> is a dial that trades head-coverage (more <Code>α</Code>) against tail-coverage (less <Code>α</Code>).
      </Prose>

      <Prose>
        <strong>DoReMi regret.</strong> Xie et al. (2023) frame mixture search as distributionally robust optimization. Let <Code>L_ref(D_i)</Code> be the per-domain loss of a small reference model trained on the uniform mixture, and let <Code>L_target(D_i)</Code> be a target loss level — in the cleanest version, the loss achieved by a specialist reference model trained only on domain <Code>i</Code>. The regret on domain <Code>i</Code> is how much the main model underperforms the specialist:
      </Prose>

      <MathBlock>
        {"\\text{regret}(D_i) = \\max\\big(0, L_{\\text{ref}}(D_i) - L_{\\text{target}}(D_i)\\big)"}
      </MathBlock>

      <Prose>
        The DoReMi update rule shifts weight toward high-regret domains: domains where the reference model is doing <em>worse</em> than a specialist would have done. The intuition is that those are the domains with room to improve, so we should spend proportionally more training tokens there. The actual DoReMi algorithm uses Group DRO with an exponentiated-gradient update over domain weights, but the essential move is the same: measure regret, upweight. The paper's headline result is that the derived weights outperform the default Pile mix by 6.5 average percentage points on downstream accuracy, using a 280M proxy model (~1% of main-run compute) to set weights for an 8B training run.
      </Prose>

      <Prose>
        <strong>Data mixing laws.</strong> Ye et al. (2024, arXiv:2403.16952) proposed a scaling-law-style relationship that lets you extrapolate performance on new mixtures from a small set of training runs. For a held-out validation domain, they observe that per-domain validation loss <Code>L(w)</Code> as a function of the training mixture <Code>w</Code> is well-fit by a functional form that is log-linear in the domain weights:
      </Prose>

      <MathBlock>
        {"L(w) \\approx c_0 + \\sum_{i=1}^{k} c_i \\cdot \\exp(-t_i \\cdot w_i)"}
      </MathBlock>

      <Prose>
        Where <Code>c_0, c_i, t_i</Code> are fit coefficients. The practical consequence is that a handful of small-scale training runs at different mixture points give you a fitted surface you can optimize analytically, without running the full grid. Combined with scaling laws for model size and training tokens, this gives you a three-axis predictor — model size, training tokens, mixture weights — from which you can choose the mixture that minimizes predicted loss at the scale you actually intend to train. Ye et al. show this recovers 48%-faster-to-baseline mixtures on a 1B model / 100B token target relative to the default RedPajama weights.
      </Prose>

      <Prose>
        <strong>Curriculum as a schedule.</strong> Formally, a curriculum is a time-dependent mixture <Code>w(t)</Code> where <Code>t</Code> indexes the training step. Constant mixtures are the special case <Code>w(t) = w</Code>. The cooldown schedule is a piecewise function: <Code>w(t) = w_bulk</Code> for <Code>t &lt; T_cd</Code> and <Code>w(t) = w_curated</Code> for <Code>t ≥ T_cd</Code>, with <Code>T_cd</Code> usually the last 5–10% of steps. A smooth cooldown interpolates linearly between the two mixtures across the cooldown window. The learning rate schedule <Code>η(t)</Code> typically decays across the cooldown as well, reducing the update magnitude so that the curated data sharpens without destabilizing the representations built during the bulk phase.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Five short programs, each under sixty lines, that implement the machinery of this topic: a mixture sampler, temperature reweighting, DoReMi-style regret updates, a combined learning-rate and mix schedule with cooldown, and a replay-buffer mixer for domain-adaptive pretraining. Every output shown as a comment was produced by running the code; the numbers are verbatim.
      </Prose>

      <H3>4a. Mixture sampler</H3>

      <Prose>
        The most basic primitive. Given a set of domain corpora and a weight vector, draw samples such that the empirical distribution over domains matches the weights. This is what sits underneath every training loop's data loader: at each step, you pick a domain according to the mixture, then pick a document (or batch) from within that domain. The implementation is five lines of Python with the standard library.
      </Prose>

      <CodeBlock language="python">
{`import random
from collections import Counter

def mixture_sampler(corpora, weights, n):
    """Weighted sampling across domain corpora.
    corpora: dict[name -> list[str]]    — domain -> list of documents
    weights: dict[name -> float]         — will be L1-normalized internally
    returns: list of (domain, sample) tuples of length n
    """
    total = sum(weights.values())
    probs = {d: w / total for d, w in weights.items()}
    domains = list(probs.keys())
    pvec    = [probs[d] for d in domains]
    out = []
    for _ in range(n):
        d = random.choices(domains, weights=pvec, k=1)[0]
        out.append((d, random.choice(corpora[d])))
    return out

corpora = {
    "web":    ["the quick brown fox"] * 10,
    "code":   ["def f(x): return x*2"] * 10,
    "math":   ["let epsilon > 0 ..."]  * 10,
    "books":  ["It was the best of times"] * 10,
    "papers": ["We report a novel method"] * 10,
}
weights = {"web": 0.50, "code": 0.17, "math": 0.10, "books": 0.13, "papers": 0.10}

random.seed(42)
draws = mixture_sampler(corpora, weights, 10000)
got   = Counter(d for d, _ in draws)
print("domain   target   empirical")
for d, w in weights.items():
    print(f"{d:7s}  {w:6.3f}   {got[d]/10000:6.4f}")

# Actual output (verified by running this code):
# domain   target   empirical
# web       0.500   0.5060
# code      0.170   0.1646
# math      0.100   0.0999
# books     0.130   0.1303
# papers    0.100   0.0992`}
      </CodeBlock>

      <Prose>
        Two details worth flagging. First, the sampler samples <em>with</em> replacement from each domain. In production pretraining you want sampling without replacement across an epoch, which requires per-domain shuffled indices, but for illustrating mixing behavior the with-replacement version is cleaner. Second, at ten thousand draws, the empirical proportions hit target to within roughly half a percentage point, which is what you would expect from a multinomial with that sample size. Real pre-training corpora are sampled for trillions of tokens, so mixture noise is far below any other source of variance. The mixture you declare is, for all practical purposes, the mixture you get.
      </Prose>

      <H3>4b. Temperature sampling</H3>

      <Prose>
        The natural distribution of training data across domains or languages is usually heavily skewed. If English is 70% of your multilingual crawl and Hindi is 10%, training at the natural rate means the model sees Hindi only sporadically and does poorly there. Temperature sampling smooths this out by raising each natural probability to a power <Code>α ∈ (0, 1]</Code> and renormalizing.
      </Prose>

      <CodeBlock language="python">
{`def temperature_reweight(freqs, alpha):
    """Apply p^alpha / sum(p_j^alpha) to natural frequencies.
    freqs: dict[name -> count or natural weight]
    alpha: float in (0, 1]. alpha=1 -> natural; alpha=0 -> uniform.
    """
    total = sum(freqs.values())
    probs = {k: v / total for k, v in freqs.items()}
    adj   = {k: p ** alpha for k, p in probs.items()}
    z     = sum(adj.values())
    return {k: v / z for k, v in adj.items()}

natural = {"en": 70, "es": 20, "hi": 10}   # 70/20/10 by raw count

for alpha in (1.0, 0.5, 0.3, 0.1):
    rw = temperature_reweight(natural, alpha)
    parts = "  ".join(f"{k}={rw[k]:.4f}" for k in natural)
    print(f"alpha={alpha:>4}  {parts}")

# Actual output (verified):
# alpha= 1.0  en=0.7000  es=0.2000  hi=0.1000
# alpha= 0.5  en=0.5229  es=0.2795  hi=0.1976
# alpha= 0.3  en=0.4455  es=0.3060  hi=0.2485
# alpha= 0.1  en=0.3696  es=0.3261  hi=0.3043`}
      </CodeBlock>

      <Prose>
        Read the output line by line. At <Code>α = 1.0</Code> the mixture is the natural one: 70/20/10. At <Code>α = 0.5</Code> — square-root reweighting, the XLM-R default — English drops to 52%, Hindi climbs to nearly 20%. At <Code>α = 0.3</Code>, which is used in several multilingual papers for aggressive rebalancing, English is 45% and Hindi is 25%. At <Code>α = 0.1</Code> the mixture is almost uniform: 37/33/30. The lower the temperature, the more bandwidth you allocate to tail languages at the cost of head languages. No setting is right in the abstract; the choice trades off what your downstream evaluation cares about.
      </Prose>

      <Callout accent="gold">
        A practical heuristic: pick <Code>α</Code> so that the most-underrepresented domain you care about gets at least 5% of the effective mixture. Below that threshold, gradient noise on the minority domain is high enough that the model often fails to converge on it.
      </Callout>

      <H3>4c. DoReMi-style regret update</H3>

      <Prose>
        The DoReMi idea formalized in code. Given a reference model's per-domain losses and a target loss level per domain (from specialist training runs, or set by hand), iteratively shift weight toward high-regret domains until the weights stabilize. The real DoReMi paper uses Group DRO with exponentiated gradient; the version here uses a simpler convex combination step that exhibits the same qualitative behavior.
      </Prose>

      <CodeBlock language="python">
{`def doremi_weights(ref_losses, target_losses, n_steps=10, lr=0.5, eps=1e-6):
    """Iterative regret-based mixture update.

    ref_losses[d]    — per-domain loss of a proxy model on the current mix
    target_losses[d] — loss we would be satisfied with (e.g., specialist's loss)

    At each step:
      1. Compute regret_d = max(0, ref_d - target_d)
      2. Normalize regrets to a target distribution over domains
      3. Take a convex-combination step toward the target distribution
    """
    domains = list(ref_losses.keys())
    w = {d: 1.0 / len(domains) for d in domains}
    history = [w.copy()]
    for _ in range(n_steps):
        regret = {d: max(0.0, ref_losses[d] - target_losses[d]) for d in domains}
        total  = sum(regret.values()) + eps
        target_w = {d: regret[d] / total for d in domains}
        w = {d: (1 - lr) * w[d] + lr * target_w[d] for d in domains}
        z = sum(w.values())
        w = {d: x / z for d, x in w.items()}   # renormalize for safety
        history.append(w.copy())
    return w, history

ref_losses    = {"web": 2.20, "code": 3.80, "math": 4.10, "books": 2.60, "papers": 3.00}
target_losses = {"web": 2.00, "code": 2.50, "math": 2.80, "books": 2.30, "papers": 2.60}

final_w, history = doremi_weights(ref_losses, target_losses, n_steps=10, lr=0.5)
print("iter " + " ".join(f"{d:>7s}" for d in ref_losses))
for i, h in enumerate(history):
    print(f"{i:>4} " + " ".join(f"{h[d]:7.4f}" for d in ref_losses))

# Actual output (verified):
# iter     web    code    math   books  papers
#    0  0.2000  0.2000  0.2000  0.2000  0.2000
#    1  0.1286  0.2857  0.2857  0.1429  0.1571
#    2  0.0929  0.3286  0.3286  0.1143  0.1357
#    3  0.0750  0.3500  0.3500  0.1000  0.1250
#    4  0.0661  0.3607  0.3607  0.0929  0.1196
#    5  0.0616  0.3661  0.3661  0.0893  0.1170
#    6  0.0594  0.3687  0.3687  0.0875  0.1156
#    7  0.0583  0.3701  0.3701  0.0866  0.1150
#    8  0.0577  0.3708  0.3708  0.0862  0.1146
#    9  0.0574  0.3711  0.3711  0.0859  0.1145
#   10  0.0573  0.3713  0.3713  0.0858  0.1144`}
      </CodeBlock>

      <Prose>
        The update converges in roughly six or seven iterations to a fixed point where web is at 5.7% and code and math are each at 37%. The logic is visible in the numbers: web has the smallest excess loss (0.20 nats above target), so the algorithm downweights it aggressively; code and math have the largest excess losses (1.30 nats each), so they receive the bulk of the mass. Books and papers land in between.
      </Prose>

      <Prose>
        Two things not to miss. First, this is a degenerate-looking final mixture — 74% of the corpus is code and math combined, with only 5.7% web. That is the point at which a practitioner would intervene: regret-based methods will happily starve a domain that is "easy" (low regret) even if that domain carries capabilities you care about. Production DoReMi adds a floor — no domain gets less than some minimum weight, often 2–5% — to prevent exactly this collapse. Second, the convergence is fast enough that you can run DoReMi in a single training pass of a proxy model, computing regret on a held-out validation set every few hundred steps and updating the sampling weights accordingly. That is close to how the paper's actual algorithm works.
      </Prose>

      <Heatmap
        label="doremi weight evolution (rows = iteration, cols = domain)"
        matrix={[
          [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
          [0.1286, 0.2857, 0.2857, 0.1429, 0.1571],
          [0.0929, 0.3286, 0.3286, 0.1143, 0.1357],
          [0.0750, 0.3500, 0.3500, 0.1000, 0.1250],
          [0.0661, 0.3607, 0.3607, 0.0929, 0.1196],
          [0.0616, 0.3661, 0.3661, 0.0893, 0.1170],
          [0.0594, 0.3687, 0.3687, 0.0875, 0.1156],
          [0.0583, 0.3701, 0.3701, 0.0866, 0.1150],
          [0.0577, 0.3708, 0.3708, 0.0862, 0.1146],
          [0.0574, 0.3711, 0.3711, 0.0859, 0.1145],
          [0.0573, 0.3713, 0.3713, 0.0858, 0.1144],
        ]}
        rowLabels={["0","1","2","3","4","5","6","7","8","9","10"]}
        colLabels={["web","code","math","books","papers"]}
        colorScale="gold"
        cellSize={42}
      />

      <H3>4d. Cooldown learning-rate and mix schedule</H3>

      <Prose>
        The standard three-phase schedule combines a learning-rate shape (linear warmup, cosine decay to a floor, linear cooldown to near-zero) with a data-mix shape (general mixture in warmup and bulk, gradual shift to a curated mixture during cooldown). Both schedules share the same timeline and transition at the same steps.
      </Prose>

      <CodeBlock language="python">
{`import math

def schedule_lr(step, total_steps, peak=3e-4, min_lr=3e-5,
                warmup_frac=0.02, cooldown_frac=0.10):
    """Linear warmup -> cosine decay -> linear cooldown."""
    warmup = int(warmup_frac * total_steps)
    cd_start = int((1 - cooldown_frac) * total_steps)
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step < cd_start:
        t = (step - warmup) / max(cd_start - warmup, 1)
        return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * t))
    # Cooldown: linear from min_lr to 10% of min_lr.
    t = (step - cd_start) / max(total_steps - cd_start, 1)
    return min_lr * (1 - t) + (min_lr * 0.1) * t

def schedule_mix(step, total_steps, cooldown_frac=0.10):
    """General mixture during bulk; linear blend to curated during cooldown."""
    cd_start = int((1 - cooldown_frac) * total_steps)
    general = {"web": 0.60, "code": 0.15, "math": 0.10, "books": 0.08, "papers": 0.07}
    curated = {"web": 0.25, "code": 0.25, "math": 0.20, "books": 0.15, "papers": 0.15}
    if step < cd_start:
        return general
    t = (step - cd_start) / max(total_steps - cd_start, 1)
    return {d: (1 - t) * general[d] + t * curated[d] for d in general}

TOTAL = 10000
for s in [0, 100, 1000, 5000, 8999, 9000, 9500, 9999]:
    lr  = schedule_lr(s, TOTAL)
    mix = schedule_mix(s, TOTAL)
    parts = " ".join(f"{mix[d]:6.3f}" for d in ("web","code","math","books","papers"))
    print(f"{s:>5}  lr={lr:.3e}   {parts}")

# Actual output (verified):
#     0  lr=0.000e+00   0.600  0.150  0.100  0.080  0.070
#   100  lr=1.500e-04   0.600  0.150  0.100  0.080  0.070
#  1000  lr=2.945e-04   0.600  0.150  0.100  0.080  0.070
#  5000  lr=1.458e-04   0.600  0.150  0.100  0.080  0.070
#  8999  lr=3.000e-05   0.600  0.150  0.100  0.080  0.070
#  9000  lr=3.000e-05   0.600  0.150  0.100  0.080  0.070
#  9500  lr=1.650e-05   0.425  0.200  0.150  0.115  0.110
#  9999  lr=3.027e-06   0.250  0.250  0.200  0.150  0.150`}
      </CodeBlock>

      <Prose>
        Notice the structure of the output. At step 100 the LR is climbing through warmup and the mix is still general. At step 1000 the LR has peaked and cosine decay has begun. At step 5000 — roughly halfway — the LR is about half of peak, and the mix is unchanged. At step 8999, the last bulk step, the LR has decayed to the floor and the mix is still fully general. At step 9000, the cooldown kicks in: the LR starts its linear descent toward 10% of floor, and the mix begins its linear shift toward curated. By step 9999, web has dropped from 60% to 25% while code, math, books, and papers have all grown; the LR is nearly an order of magnitude below <Code>min_lr</Code>. That final 10% of training is where the curated mix imprints on the model without re-destabilizing the representations built in the bulk.
      </Prose>

      <Plot
        label="lr schedule: warmup -> cosine decay -> cooldown"
        width={520}
        height={220}
        xLabel="step"
        yLabel="learning rate (× 1e-4)"
        series={[
          { name: "lr", points: [[0, 0.0], [100, 1.5], [500, 3.0], [1000, 2.95], [3000, 2.20], [5000, 1.46], [7000, 0.77], [8999, 0.30], [9000, 0.30], [9500, 0.165], [9999, 0.03]] },
        ]}
      />

      <Plot
        label="mix evolution across a full run (% per domain)"
        width={520}
        height={240}
        xLabel="step"
        yLabel="% of batch"
        series={[
          { name: "web",    points: [[0, 60], [8999, 60], [9500, 42.5], [9999, 25]] },
          { name: "code",   points: [[0, 15], [8999, 15], [9500, 20.0], [9999, 25]] },
          { name: "math",   points: [[0, 10], [8999, 10], [9500, 15.0], [9999, 20]] },
          { name: "books",  points: [[0,  8], [8999,  8], [9500, 11.5], [9999, 15]] },
          { name: "papers", points: [[0,  7], [8999,  7], [9500, 11.0], [9999, 15]] },
        ]}
      />

      <StepTrace
        label="three-phase schedule, step by step"
        steps={[
          {
            label: "phase 1 — warmup (0 - 2% of steps)",
            render: () => (
              <TokenStream tokens={[{ label: "LR ramp 0 -> peak", color: colors.gold }, " ", { label: "general mix", color: colors.green }]} />
            ),
          },
          {
            label: "phase 2 — bulk (2 - 90%)",
            render: () => (
              <TokenStream tokens={[{ label: "cosine decay peak -> min", color: colors.gold }, " ", { label: "general mix (constant)", color: colors.green }]} />
            ),
          },
          {
            label: "phase 3 — cooldown (last 10%)",
            render: () => (
              <TokenStream tokens={[{ label: "LR min -> ~0", color: colors.gold }, " ", { label: "mix blend general -> curated", color: "#c084fc" }]} />
            ),
          },
        ]}
      />

      <H3>4e. Replay buffer mixer for domain-adaptive pretraining</H3>

      <Prose>
        Continued pre-training on a narrow domain — medical literature, legal text, code of a specific language — risks catastrophic forgetting: the model learns the new domain at the cost of the general capabilities it spent hundreds of billions of tokens acquiring. The standard mitigation is a replay buffer: at each step, with probability <Code>r ∈ [0.1, 0.2]</Code>, draw from a general-purpose corpus instead of the domain corpus. This keeps a trickle of general data flowing through the model and anchors the general distribution against the pull of the domain shift.
      </Prose>

      <CodeBlock language="python">
{`import random
from collections import Counter

def replay_mixer(domain_corpus, general_corpus, replay_frac, n):
    """Interleave domain samples with a replay_frac of general samples."""
    out = []
    for _ in range(n):
        if random.random() < replay_frac:
            out.append(("general", random.choice(general_corpus)))
        else:
            out.append(("domain",  random.choice(domain_corpus)))
    return out

random.seed(0)
domain_corpus  = ["MED: patient presents with..."] * 50
general_corpus = ["the cat sat on the mat",
                  "a journey of a thousand miles"] * 50

for frac in (0.0, 0.10, 0.15, 0.20, 0.30):
    draws = replay_mixer(domain_corpus, general_corpus, frac, 10000)
    c = Counter(src for src, _ in draws)
    print(f"replay_frac={frac:0.2f}  "
          f"domain={c['domain']/10000:.4f}  general={c['general']/10000:.4f}")

# Actual output (verified):
# replay_frac=0.00  domain=1.0000  general=0.0000
# replay_frac=0.10  domain=0.8992  general=0.1008
# replay_frac=0.15  domain=0.8475  general=0.1525
# replay_frac=0.20  domain=0.8039  general=0.1961
# replay_frac=0.30  domain=0.6992  general=0.3008`}
      </CodeBlock>

      <Prose>
        The exact replay fraction is a hyperparameter you cannot derive from first principles; it depends on how far the domain is from the base model's training distribution. Adapting Llama-2 to general English code (Code Llama) used a relatively low replay rate because code is already well-represented in the base. Adapting to a narrow medical subfield, or to a low-resource language, requires higher replay — 20% is a reasonable default, 30% is not unusual for aggressive shifts. A replay fraction of zero is continued pre-training without replay, which is how catastrophic forgetting experiments are usually staged. The line between "continued pre-training" and "specialization" is, operationally, mostly about where you set this dial.
      </Prose>

      <Prose>
        That is the complete toolkit. Mixture sampling chooses how much of each domain enters the batch. Temperature reweights natural frequencies when they are too skewed. DoReMi automates the search for good weights. The cooldown schedule shifts both learning rate and mixture at the end of training. The replay mixer protects general capabilities during domain adaptation. Every frontier pre-training recipe is some combination of these five pieces, scaled up to production engineering.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATIONS
          ====================================================================== */}
      <H2>5. Production implementations</H2>

      <Prose>
        The machinery is the same at production scale; what changes is the corpus engineering around it. Real systems spend most of their code on the things the from-scratch implementations ignore: streaming data from sharded storage, deduplication across and within domains, preserving shuffle state across checkpoints, running the proxy-model DoReMi loop as its own training job, and packaging the cooldown corpus as a versioned artifact that can be audited for benchmark leakage.
      </Prose>

      <H3>RedPajama and the published mixes</H3>

      <Prose>
        RedPajama was among the first open efforts to reproduce Llama-scale training data with published mixture weights. The RedPajama-Data-1T recipe allocated roughly 67% to deduplicated CommonCrawl web, 15% to C4, 4.5% each to GitHub code, books, and arXiv, and smaller fractions to Wikipedia and StackExchange. The RedPajama-2 release extended the corpus to 30T tokens and published quality-filtered variants at several aggressiveness levels. The exact weights are a useful reference point because they represent a public consensus estimate of what frontier-scale mixtures looked like around 2023. More recent open mixtures — DCLM, FineWeb — shift weight toward aggressively quality-filtered web, sometimes dropping below 50% of the corpus by token count while still dominating by information value.
      </Prose>

      <CodeBlock language="python">
{`# RedPajama-1T published mixture (approximate; from the Together release notes)
redpajama_1t = {
    "commoncrawl":   0.670,   # deduplicated and filtered CC
    "c4":            0.150,   # Google's cleaned CC variant
    "github":        0.045,   # code
    "books":         0.045,   # Books3 and similar
    "arxiv":         0.045,   # latex-extracted papers
    "stackexchange": 0.020,
    "wikipedia":     0.025,
}
assert abs(sum(redpajama_1t.values()) - 1.0) < 1e-6`}
      </CodeBlock>

      <H3>FineWeb and staged corpus building</H3>

      <Prose>
        FineWeb (HuggingFace, 2024) is a 15T-token filtered web-only corpus, and its published build pipeline is the clearest public demonstration of a modern staged quality filter. The stages are roughly: (1) raw CommonCrawl WARC files, (2) language identification and English filtering, (3) URL-based filtering against denylists, (4) Gopher-style quality heuristics (average word length, stopword ratio, repeated n-gram density), (5) MinHash-based near-duplicate removal, (6) model-based quality classification with a small educational-content classifier (the FineWeb-Edu variant). Each stage drops 30–70% of the remaining tokens; the final corpus is roughly 5% of the raw crawl by token count.
      </Prose>

      <Prose>
        The mix-relevant observation is that FineWeb deliberately does not try to be a full pretraining corpus on its own — it is designed to be blended with code, math, and book corpora in a typical mixture. The same quality-filtering principles appear in DCLM, Dolma, and Common Corpus. A production pretraining data pipeline chains tools like DataTrove (the HuggingFace-internal processing library FineWeb is built on) to orchestrate these stages reproducibly across thousands of CPUs.
      </Prose>

      <H3>Llama 3 annealing recipe</H3>

      <Prose>
        The Llama 3 technical report (Meta, 2024, arXiv:2407.21783) contains the clearest public description of a cooldown (called "annealing" in the report). The schedule: during the final 40M tokens of the 8B pre-training run, the learning rate is annealed linearly to zero while the data mixture is shifted to upsample high-quality domains, with context length held at 128K. The annealing data mix is deliberately built to exclude any content derivable from standard benchmarks, so that post-annealing evaluation measures genuine generalization rather than eval leakage. The reported effect: +24 percentage points on GSM8k, +6.4 percentage points on MATH, relative to the pre-annealing checkpoint of the same model. The 405B model showed negligible benefit from the same procedure — a data point suggesting that annealing's effect saturates at scale, though the authors do not spell out a mechanism.
      </Prose>

      <Prose>
        The report also describes using annealing as a <em>measurement</em> tool: take a 50%-trained checkpoint, anneal it on a candidate small dataset (30% weight) mixed with the default corpus (70%), and observe the delta on downstream evaluations. This is presented as a cheaper alternative to running scaling-law experiments for every candidate dataset. It is one of the few examples of the cooldown phase being used as a probe of data quality rather than only as a final sharpening step.
      </Prose>

      <H3>DeepSeek-V3 cooldown</H3>

      <Prose>
        The DeepSeek-V3 technical report (arXiv:2412.19437) describes pre-training on 14.8T tokens and confirms a cooldown phase in which no synthetic OpenAI-model outputs were included — a detail flagged in the report specifically to differentiate from competitors whose pre-training corpora include distilled GPT-4-generated text. The cooldown data engineering is otherwise similar in shape to Llama 3's: a curated subset, high-quality upsampling, careful exclusion of benchmark-adjacent content. The transparency around the cooldown's <em>contents</em> — not just the schedule — is part of why DeepSeek's releases have become reference implementations for the open community.
      </Prose>

      <H3>FLAN T5 instruction mixes</H3>

      <Prose>
        FLAN (Wei et al., 2021; Longpre et al., 2023) and its successors — FLAN-T5, FLAN-PaLM — apply the same mixture logic at the instruction-tuning layer rather than at pre-training. The FLAN collection bundles roughly 1,800 tasks into a single instruction-tuning corpus and the ratio of tasks per template is a mixing hyperparameter. Longpre et al.'s 2023 paper on the public FLAN collection is largely a study of how instruction-mix design affects downstream zero-shot quality, using many of the same tools — temperature sampling, held-out validation, per-domain loss tracking — that appear at the pre-training layer. The transferability of mixture search tools across the pre-training / instruction-tuning boundary is an underdiscussed point in the literature; in practice, if you have a mixture-optimization setup for one, you probably have it for the other.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Pre-training mix evolution across four generations of frontier models, drawn from public reports. The numbers are approximate — not every lab publishes exact percentages — but the shape is what matters: web contracts, code and math grow, books and papers hold roughly flat. The contraction of web is not a vote against web text; it is a vote that the marginal token of filtered web prose teaches less than the marginal token of structured code or a mathematical derivation.
      </Prose>

      <Plot
        label="pre-training mix evolution across generations (illustrative %)"
        width={520}
        height={240}
        xLabel="model generation"
        yLabel="% of corpus"
        series={[
          { name: "web",          points: [[1, 75], [2, 65], [3, 55], [4, 50]] },
          { name: "code",         points: [[1,  4], [2,  8], [3, 15], [4, 17]] },
          { name: "math",         points: [[1,  2], [2,  4], [3,  8], [4, 10]] },
          { name: "books+papers", points: [[1, 15], [2, 17], [3, 15], [4, 13]] },
        ]}
      />

      <Prose>
        Below, the complete three-stage schedule as a step-by-step trace — warmup, bulk, cooldown — with the canonical data-mix shift at the boundary between bulk and cooldown.
      </Prose>

      <StepTrace
        label="cooldown data-mix shift, in detail"
        steps={[
          {
            label: "step 1 — bulk training, step 8500 of 10000",
            render: () => (
              <div>
                <TokenStream
                  label="batch composition (general mix)"
                  tokens={[
                    { label: "web×6", color: colors.gold },
                    { label: "code×2", color: colors.green },
                    { label: "math", color: "#c084fc" },
                    { label: "books", color: "#60a5fa" },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "step 2 — cooldown begins, step 9000",
            render: () => (
              <div>
                <TokenStream
                  label="LR drops to min; mix still general"
                  tokens={[
                    { label: "web×6", color: colors.gold },
                    { label: "code×2", color: colors.green },
                    { label: "math", color: "#c084fc" },
                    { label: "books", color: "#60a5fa" },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "step 3 — cooldown midway, step 9500",
            render: () => (
              <div>
                <TokenStream
                  label="linear blend: 50% general / 50% curated"
                  tokens={[
                    { label: "web×4", color: colors.gold },
                    { label: "code×2", color: colors.green },
                    { label: "math", color: "#c084fc" },
                    { label: "books", color: "#60a5fa" },
                    { label: "papers", color: "#f87171" },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "step 4 — cooldown end, step 9999",
            render: () => (
              <div>
                <TokenStream
                  label="batch composition (curated mix)"
                  tokens={[
                    { label: "web×2", color: colors.gold },
                    { label: "code×2", color: colors.green },
                    { label: "math×2", color: "#c084fc" },
                    { label: "books", color: "#60a5fa" },
                    { label: "papers", color: "#f87171" },
                  ]}
                />
                <Prose>
                  The final ~5% of training imprints the curated distribution on top of the general representations. LR is now two orders of magnitude below peak; the curated data sharpens without destabilizing.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The DoReMi weight evolution from section 4c, rendered as a heatmap. Each row is one iteration; darker color indicates higher weight. Web decays from 20% to 5.7% over ten iterations while code and math climb to 37% each. The pattern is characteristic of regret-based methods: weight collapses onto the one or two domains where the proxy model has the most headroom, which is exactly why production systems add floor constraints to prevent starvation of easy-but-important domains.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Given a specific pretraining project, the choice is rarely "which algorithm is best in theory" — it is "which combination of techniques best fits the compute budget, the domain coverage, and the evaluation profile I actually care about." The table below lines up the distinctions that show up in those decisions.
      </Prose>

      <CodeBlock>
{`                      grid search      DoReMi / DoGE     data-mixing laws   constant mix    cooldown
                      (manual)         (automated)       (scaling-law)      (no schedule)   (staged)
k (domains)           2-5 feasible     5-50 feasible     5-20 fit           any             any
proxy compute         10-50x grid pts  ~1% of main run   ~N small runs      0               0
code complexity       trivial          moderate (DRO)    moderate (fit)     trivial         moderate
sensitivity           high             medium            low (near surface) n/a             low
cold-start practical  yes              yes               no - needs data    yes             yes
eval-leakage risk     low              low               low                low             medium*
downstream delta      baseline         +5-7 pts vs uni   +48% steps faster  baseline        +5-25 pts math
canonical use         early research,  frontier labs     scaling-law-style  small models,   every frontier
                      ablations        (Llama, Gemini)   planning           debugging       pretrain run today

* cooldown eval-leakage risk is specifically about curated corpus overlapping with benchmark content;
  controlled by corpus hygiene, not the algorithm itself.`}
      </CodeBlock>

      <Prose>
        A short decision tree, applied to real projects:
      </Prose>

      <Prose>
        <strong>If you have two to five domains and plenty of compute for small proxy runs:</strong> grid search. Train eight to sixteen proxy models at corner and interior points of the mixture simplex, evaluate on a held-out benchmark suite, and pick the best. This is what labs did before 2023 and it still works; the main cost is engineer time to orchestrate the sweeps. Grid search is also the gold-standard baseline that every more-sophisticated method is compared against.
      </Prose>

      <Prose>
        <strong>If you have many domains (ten or more) or expect the mixture space to be high-dimensional:</strong> DoReMi or DoGE. Grid search becomes exponential in the number of domains. DoReMi scales linearly because it walks the simplex rather than gridding it, and the per-domain loss signal is informative enough that the optimization converges fast. Combine with a floor constraint on every domain (say 2–5% minimum) to prevent degenerate mixes.
      </Prose>

      <Prose>
        <strong>If you are planning a large training run and want to extrapolate from small runs:</strong> fit data-mixing laws. Run a handful of small-scale mixtures, fit the functional form to the measured losses, and use the fitted surface to choose the best mixture at the scale you actually intend to train. Combined with standard model-size and token-count scaling laws, this gives you a principled three-axis predictor. The extrapolation is not exact — the functional form breaks down far from the fit points — but it substantially shrinks the search space.
      </Prose>

      <Prose>
        <strong>If you are adapting a base model to a new domain:</strong> constant mix with replay. Run the domain-adaptive corpus with 10–20% general-data replay to prevent catastrophic forgetting. This is what Code Llama, Meditron, and the long list of continued-pretraining recipes converge on. Replay fraction scales with domain distance: closer domains (code on top of a code-aware base) need less, farther domains (medicine, law, low-resource languages) need more.
      </Prose>

      <Prose>
        <strong>If you are running a frontier pretraining job today:</strong> all of the above, plus cooldown. The frontier recipe in 2025 is: DoReMi-derived weights for the bulk phase, a handcrafted curated mixture for the last 5–10% of steps, linear LR decay to near-zero across the cooldown, and careful corpus hygiene to keep benchmark data out of the curated subset. No frontier lab ships a production pretraining run without a cooldown; the expected value is too high and the cost is too small.
      </Prose>

      <Prose>
        <strong>If you are not sure the mix matters at all:</strong> run a single A/B with one alternative. Train two small models — say 500M parameters for 50B tokens — on two mixtures that differ by a single deliberate shift (halve the web, double the code, or the inverse). Evaluate on a benchmark suite that includes domains you varied. The delta you measure is, at approximately-correct scale, the delta you would get from the same shift at full scale. This is the smallest useful ablation in mixture research and the one that most teams skip because it feels obvious.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Each technique in this topic has a scale regime where it pays off and a regime where it stops helping. Knowing which axis you are on matters more than knowing the algorithms in the abstract.
      </Prose>

      <Prose>
        <strong>Grid search over domains.</strong> Feasible for two to five domains; exponential past that. With five domains and four weight levels per domain, a full grid is 1024 configurations; with ten domains it is a million. The combinatorial explosion is not the real bottleneck — you can usually use a Latin-hypercube sample instead of a full grid — but the per-configuration cost (a proxy model training run) is large enough that you cannot afford to explore densely past about twenty configurations. Grid search is a small-<Code>k</Code> tool.
      </Prose>

      <Prose>
        <strong>DoReMi and regret-based search.</strong> Scale linearly in the number of domains, because each iteration touches each domain once. DoReMi has been demonstrated cleanly on corpora with twenty-plus domains (The Pile has twenty-two) and scales beyond that with diminishing but positive returns. The harder scaling question is corpus size: the per-domain loss signal is only as accurate as the validation set for that domain, and small domains with few held-out tokens produce noisy regret estimates. In practice, every domain needs at least a few hundred thousand held-out tokens for DoReMi's signal to be trustworthy. Below that threshold, the algorithm works but the results are noisy enough to be confounded with the model-initialization noise floor.
      </Prose>

      <Prose>
        <strong>Cooldown.</strong> The cooldown's measurable benefit saturates at some point between 8B and 405B parameters. Llama 3 reports +24 points on GSM8k from cooldown at 8B but negligible gains at 405B. The hypothesis — not formally tested in the paper — is that large models during the bulk phase have already converged on the representations that cooldown would sharpen, so the sharpening step has no remaining headroom. At smaller scales the bulk phase is still building representations and the cooldown rounds them off. For a practitioner at the 1B–70B scale, the cooldown is effectively free and uniformly beneficial. Above that, run the ablation; you may be spending compute on a no-op.
      </Prose>

      <Prose>
        <strong>Curriculum by difficulty.</strong> Helps on small models (where early exposure to well-formed examples sets up good representations) and stops helping past roughly 1B parameters (where the law of large numbers washes out the order-of-exposure signal). The mixture-curriculum (shifting <em>domain</em> composition across time) continues to help at scale because it is changing a first-order property of the training distribution rather than a second-order reordering. Difficulty-ordering was one of Bengio's original 2009 pitches and has been the most consistently disappointing form of curriculum at LLM scale.
      </Prose>

      <Prose>
        <strong>Replay fraction.</strong> Scales with domain distance, not with model size. For any given base-and-target pair, the right replay fraction is approximately constant across model scales — a 100M model adapting to medicine needs about the same 15–20% replay as a 100B model doing the same adaptation. The exception is very small models that are still building general capabilities during the adaptation; below roughly 100M, the replay fraction has to be higher (30%+) because the base capabilities are fragile.
      </Prose>

      <Prose>
        <strong>Corpus size and mixing-law fit.</strong> Ye et al.'s mixing laws fit best on small proxy runs and extrapolate usefully to 10×–100× the fit scale. Past that, the functional form starts to miss. If you intend to train a 400B model, fitting a law on 1B proxy runs will give you a starting point but you should expect to correct the extrapolation with a mid-scale calibration run. This is analogous to compute-scaling laws, which also require calibration past 10–100× extrapolation.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Eight things that reliably go wrong. The pattern across all of them is the same: the mix and the curriculum are second-order effects that interact with everything else in pre-training, and failures show up as subtle, domain-specific regressions rather than as obvious training-time signals.
      </Prose>

      <Prose>
        <strong>1. Overcooked cooldown.</strong> The cooldown runs too long or at too high a weight toward the curated mix, and the model overfits to the curated subset's distribution. Symptom: benchmark gains at the cooldown endpoint, but the model starts producing "benchmark-shaped" outputs on open-ended prompts — overly formal prose, excess mathematical notation, a tendency to structure every answer as a step-by-step derivation. Fix: keep the cooldown under 10% of total tokens and keep at least some fraction (25–40%) of general data in the curated mix. The goal is sharpening, not a final epoch on curated data alone.
      </Prose>

      <Prose>
        <strong>2. Catastrophic forgetting during domain adaptation.</strong> Continued pre-training on a narrow domain with zero replay erodes general capabilities faster than the new capabilities accumulate. Symptom: the model scores higher on the domain benchmark but scores lower on broad suites (MMLU, HellaSwag) — often by more than the domain gain. Fix: set replay fraction to at least 10%, monitor general benchmarks during the adaptation run, stop if general scores degrade by more than a threshold (say, 3 percentage points).
      </Prose>

      <Prose>
        <strong>3. DoReMi degenerate weights.</strong> Regret-based methods can collapse onto one or two high-regret domains and starve everything else, as in the section 4c example where web dropped to 5.7% of the corpus. Symptom: proxy-model-optimal mixtures that allocate 70%+ to code and math. Fix: add per-domain floor constraints (no domain below 2–5%), cap the maximum weight any single domain can receive (usually 40–50%), and validate the derived weights on a broad downstream suite, not just per-domain perplexity.
      </Prose>

      <Prose>
        <strong>4. Bad difficulty metric in curriculum design.</strong> You decide to order training examples by difficulty and score difficulty by perplexity under a small model. The small model's perplexity is dominated by surface features (sequence length, vocabulary overlap with training) that correlate weakly with what a human would call "difficulty." Result: the ordering shuffles examples by surface complexity rather than conceptual complexity, and the curriculum provides no benefit. Symptom: negligible delta versus a shuffled baseline. Fix: if you must order by difficulty, use an external difficulty metric (grade-level readability, domain-annotated difficulty) rather than a proxy-model perplexity score; or, more realistically, accept that difficulty curricula do not help at pretraining scale and spend the compute elsewhere.
      </Prose>

      <Prose>
        <strong>5. Cooldown corpus leaks benchmarks.</strong> The curated cooldown corpus overlaps with held-out benchmark content — a paper in the papers slice quotes a GSM8k problem, a coding textbook reprints a LeetCode problem that ended up in HumanEval, a Wikipedia dump includes a transcription of MMLU questions. The cooldown sharpens the model on the benchmark directly and the reported gains are contaminated. Symptom: cooldown produces unusually large gains on specific benchmarks but no transfer gains to out-of-distribution evaluations. Fix: run an n-gram overlap check (or an embedding-based similarity check) between every cooldown corpus and every benchmark you intend to report; remove overlapping segments. This is the same hygiene that Llama 3 describes applying to its annealing corpus.
      </Prose>

      <Prose>
        <strong>6. Perplexity-only weights miss downstream capability.</strong> You optimize the mixture for average per-domain perplexity and the resulting model has lower perplexity than the baseline but performs worse on downstream reasoning benchmarks. Perplexity is a surface metric; it rewards fluent continuation, not reasoning quality. Symptom: mixture-optimized model is slightly better on perplexity, slightly worse on GSM8k, HumanEval, MMLU. Fix: include at least some downstream benchmarks (not just perplexity) in the mixture-optimization objective; or, weight domains that correlate with downstream capability (code, math) above their natural optimum even if perplexity-only optimization would downweight them.
      </Prose>

      <Prose>
        <strong>7. Small-to-large mixture non-transfer.</strong> A mixture optimized on a 280M proxy model looks great on the proxy's evaluations and performs noticeably worse when applied to a 70B training run. The proxy is too small for its per-domain loss landscape to reflect the large model's. Symptom: DoReMi-at-scale gains that are smaller than the paper's 6.5-point delta, or absent. Fix: run at least two proxy scales (say 280M and 1B), check that the derived weights are stable across scales, and distrust weights that change sharply as the proxy scales up. Ye et al.'s mixing laws explicitly model this scale dependence and are a more principled version of the same correction.
      </Prose>

      <Prose>
        <strong>8. Code-heavy mixture hurts natural language.</strong> You push code to 25% of the corpus because code-heavy models perform well on reasoning benchmarks. The model gets better at code, slightly worse at prose generation — prose outputs are shorter, more formal, occasionally drift into pseudo-code mid-sentence. Symptom: wins on HumanEval and GSM8k, losses on MT-Bench and open-ended generation quality. Fix: cap code weight around 17–20% (the Llama 3 regime) unless the product specifically targets coding; if pushing higher, monitor open-ended generation quality as a first-class eval.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Every citation below was cross-checked against arXiv, conference proceedings, or official publisher pages during the preparation of this topic. Dates, arXiv ids, and venue details reflect the verified records.
      </Prose>

      <Prose>
        <strong>1.</strong> Bengio, Yoshua; Louradour, Jérôme; Collobert, Ronan; Weston, Jason. "Curriculum Learning." <em>Proceedings of the 26th Annual International Conference on Machine Learning (ICML 2009)</em>, pages 41–48. DOI: 10.1145/1553374.1553380. The paper that formalized curriculum learning as a training strategy. Argues by analogy to how humans and animals are taught — start with simple examples, graduate to harder — and shows empirical gains on shape recognition and language modeling at small scale.
      </Prose>

      <Prose>
        <strong>2.</strong> Xie, Sang Michael; Pham, Hieu; Dong, Xuanyi; Du, Nan; Liu, Hanxiao; Lu, Yifeng; Liang, Percy; Le, Quoc V.; Ma, Tengyu; Yu, Adams Wei. "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining." arXiv:2305.10429 (May 2023), published at NeurIPS 2023. Introduces Domain Reweighting with Minimax Optimization. Trains a 280M proxy with Group DRO over Pile domains; resulting weights improve average few-shot downstream accuracy of a Llama-style 8B model by 6.5 points over Pile defaults and reach baseline 2.6× faster.
      </Prose>

      <Prose>
        <strong>3.</strong> Fan, Simin; Pagliardini, Matteo; Jaggi, Martin. "DoGE: Domain Reweighting with Generalization Estimation." arXiv:2310.15393 (October 2023). Proposes a bi-level optimization where domain weights are chosen to upweight domains whose gradients align with target-domain gradients. Extends DoReMi's framing to out-of-domain target tasks and cross-domain generalization.
      </Prose>

      <Prose>
        <strong>4.</strong> Ye, Jiasheng; Liu, Peiju; Sun, Tianxiang; Zhan, Jun; Zhou, Yunhua; Qiu, Xipeng. "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance." arXiv:2403.16952 (March 2024). Shows that per-domain validation loss under a mixture is well-fit by a log-linear functional form and can be extrapolated from small-scale runs. Combined with scaling laws, enables prediction of large-scale mixture performance from proxy experiments. Recovers a 1B-on-100B-tokens mixture that reaches RedPajama baseline 48% faster.
      </Prose>

      <Prose>
        <strong>5.</strong> Muennighoff, Niklas; Rush, Alexander M.; Barak, Boaz; Le Scao, Teven; Piktus, Aleksandra; Tazi, Nouamane; Pyysalo, Sampo; Wolf, Thomas; Raffel, Colin. "Scaling Data-Constrained Language Models." arXiv:2305.16264 (May 2023), NeurIPS 2023. Studies repetition and epoch count when training data is limited; shows that up to four epochs of repetition is nearly as good as unique data for fixed compute, and proposes a modified scaling law for compute optimality under data constraints. Relevant to mixing decisions because it governs how small domains can be safely repeated relative to large ones.
      </Prose>

      <Prose>
        <strong>6.</strong> Llama Team, Meta AI. "The Llama 3 Herd of Models." arXiv:2407.21783 (July 2024). Describes the Llama 3 pretraining pipeline, including the annealing phase: final 40M tokens with learning rate linearly annealed to zero, data mix shifted to upsample high-quality domains, benchmark-adjacent content excluded from the annealing corpus. Reports +24 points on GSM8k and +6.4 points on MATH from annealing at 8B; negligible benefit at 405B. Describes using annealing as a measurement tool for data quality.
      </Prose>

      <Prose>
        <strong>7.</strong> DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437 (December 2024). Describes pretraining on 14.8T tokens with a cooldown phase that explicitly excludes synthetic outputs from other frontier models. Confirms the Llama-3-shaped cooldown as standard practice at the 2024 frontier.
      </Prose>

      <Callout accent="gold">
        Secondary but worth flagging: Longpre, Shayne; et al. "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning." <em>ICML 2023</em>. The clearest empirical study of how instruction-data mixture design affects downstream zero-shot quality; the mixture-optimization tools transfer almost verbatim from pretraining to instruction-tuning. Also: Penedo, Guilherme; et al. "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale." arXiv:2406.17557 (June 2024). The reference public description of a modern staged web-filtering pipeline; useful companion reading for understanding what the "web" slice of a modern mixture actually contains.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five short problems. Work them before reading the answers; the point is to surface confusions you might not know you had.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> You are designing a DoReMi target mixture for a model that will be evaluated primarily on code and math reasoning but must still handle general dialogue competently. You have a reference model trained on a uniform mixture over {"{web, code, math, books, papers}"} with per-domain losses <Code>{"{2.0, 3.5, 4.0, 2.5, 2.9}"}</Code>. What target losses would you set, and why? What floor constraints would you add?
      </Prose>

      <Callout accent="green">
        Set target losses tighter on code and math (say 2.6 and 3.0, a 0.9–1.0 nat improvement over the reference) and looser on web, books, papers (say 1.9, 2.3, 2.7 — small improvements). The regrets will be: web 0.10, code 0.90, math 1.00, books 0.20, papers 0.20. Naively, the derived weights would be roughly {"{web: 0.04, code: 0.37, math: 0.42, books: 0.08, papers: 0.08}"} — code and math would dominate. Because general dialogue competence matters, add a floor of at least 15% on web and 5% on books and papers; this prevents the model from starving the domain that produces the most dialogue-style text. The resulting mixture will be something like web 15%, code 35%, math 35%, books 8%, papers 7% — aggressively reasoning-tilted but still trained on enough web to preserve open-ended prose fluency.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> You have a trilingual corpus with natural token proportions en=70%, es=20%, hi=10%. You want to train a multilingual model where Hindi competence matters roughly as much as English competence. Derive the <Code>α</Code> you would use for temperature sampling, show the resulting weights, and state the tradeoff you are accepting.
      </Prose>

      <Callout accent="green">
        You want Hindi's effective weight to be roughly comparable to English's. At <Code>α = 0.1</Code> the weights become approximately 37% en, 33% es, 30% hi — very close to uniform. At <Code>α = 0.3</Code> they become 45%, 31%, 25%. At <Code>α = 0.5</Code>, 52%, 28%, 20%. For "roughly as much as English" you want <Code>α</Code> in the 0.1 to 0.3 range — 0.2 is a reasonable default. The tradeoff: you are training English at roughly half its natural rate, so your English-only evaluation performance will be a few points lower than a model trained at natural proportions. The Hindi performance, in exchange, will be substantially higher, potentially by 10–20 points on Hindi-specific benchmarks. Whether the tradeoff is worth it depends on what you evaluate on and who uses the model; there is no setting of <Code>α</Code> that improves every language simultaneously.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> Cooldown-on-curated-data produces measurable gains at the end of training. Why does the same data not help if you put it at the beginning of training, or if you train on it for equal duration without an LR decay?
      </Prose>

      <Callout accent="green">
        Two reasons, both about how the LR schedule interacts with the data. First, at the start of training the model is in a representation-building phase where the loss landscape is dominated by basic features (token embeddings, low-level n-gram statistics). Exposing it to curated high-reasoning data early means those reasoning-heavy examples are processed with an undertrained representation and the gradients they produce are noisy; most of the signal from the curated data is wasted. Second, the cooldown works because the LR is small. With a small LR, each curated example sharpens the existing representations without overwriting them; with a high LR (as in the bulk phase), the same curated data would move the weights far enough to partially erase the general representations built on the general mix. If you trained on curated data for equal duration without the LR decay, you would get closer to what happens if you used the curated data as your full pretraining mix — a model that is good on curated-distribution content and worse on everything else. The cooldown's power is the combination: high-quality data <em>and</em> small updates.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> You add code from 10% to 25% of your pre-training mixture and benchmark scores move as follows: HumanEval +6 points, GSM8k +4 points, MMLU −2 points, MT-Bench conversation quality −0.3 (on a 10-point scale). What is the failure signal and what would you do about it?
      </Prose>

      <Callout accent="green">
        The MMLU and MT-Bench regressions are the signal. Two and a half times more code means two and a half times fewer tokens of whatever got displaced — almost certainly web and books in that particular delta. MMLU tests broad knowledge that lives mostly in the displaced domains; MT-Bench tests open-ended generation that is also displaced. The failure mode is "code-heavy hurts natural language" from section 9 (#8). Options: (1) roll back partway, to 17–20% code, and accept smaller HumanEval/GSM8k gains in exchange for recovering MMLU and MT-Bench; (2) hold code at 25% but compensate by improving web quality or increasing web's effective training via repetition or quality filtering, to recover the lost MMLU signal with a smaller share; (3) accept the tradeoff if and only if the product use case is code-first (a coding assistant) and open-ended generation is a secondary concern. Option 1 is the default at most labs; Llama 3's 17% combined code-and-math figure is roughly the empirical sweet spot.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> Design an evaluation plan that would catch mixture regressions between two pretraining runs. Your constraint: you can afford to evaluate each candidate on 2,000 examples total across all benchmarks.
      </Prose>

      <Callout accent="green">
        Split the budget across four benchmark categories that are diagnostic of orthogonal capabilities. (1) Broad knowledge and reasoning: 500 examples from MMLU, sampled evenly across subjects. (2) Mathematical reasoning: 500 examples from GSM8k and MATH combined. (3) Code generation: 400 examples from HumanEval or MBPP. (4) Open-ended generation quality: 300 prompts from MT-Bench or a similar judge-based eval. (5) Long-tail / multilingual coverage: 300 examples from a suite that includes the domains you explicitly mixed for (multilingual content if you have it, legal/medical/etc. if you mixed them in). The rationale: MMLU catches general knowledge regressions, math and code catch the reasoning axis, MT-Bench catches fluency regressions that perplexity misses, and the long-tail split catches starvation of domains that might have been downweighted by a regret-based method. A regression on any of the five is a blocking signal; regressions on multiple are a clear sign the new mixture is worse in ways that will not show up in a single metric.
      </Callout>

      <Prose>
        The mix and the curriculum do not get their own sections in every pretraining paper, but every lab that has trained a frontier model has paid close attention to them. The gains are quiet, distributed across domains, and usually ascribed in public writeups to "better data" without the specifics. This topic is the specifics. The next one picks up from the cooldown endpoint and looks at what happens after pre-training ends — how the base model gets turned into a chat model, an instruction follower, or a reasoner, and how many of the same mixture-design tools reappear at those later stages.
      </Prose>
    </div>
  ),
};

export default curriculumDataMixing;
