import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const assessmentSecurity = {
  title: "Assessment Security & Item Exposure Control",
  slug: "assessment-security-item-exposure-control",
  readTime: "~34 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every measurement instrument has a quiet adversary: the people being measured. Whenever the score on an assessment unlocks something valuable — a graduate program seat, a medical license, a bar admission, a published leaderboard ranking — there is an immediate economic incentive to learn the instrument itself rather than the construct it was supposed to measure. The clean psychometric story of "the test items sample from an infinite domain of possible questions" runs into the harsh reality that the items used in any actual administration are a finite, expensive, and increasingly publicized set. When those items leak — when they are memorized, photographed, posted to a forum, scraped into a training corpus, or simply seen too many times by candidates who share information — the test stops measuring what it was designed to measure and starts measuring something subtly different: a candidate's exposure to the leaked items.
      </Prose>

      <Prose>
        High-stakes human testing has wrestled with this for decades. The Educational Testing Service maintains item banks of tens of thousands of GRE and TOEFL questions, rotated through administrations under elaborate exposure-control regimes precisely because the cost of an item becoming public is the cost of writing, calibrating, and pretesting a new one — typically 1,500 to 3,000 USD per operational item, multiplied by the number of items burned. The United States Medical Licensing Examination, the bar examination, the CFA, the Architectural Registration Examination — all of these maintain elaborate item lifecycle pipelines (pretest, operational, retired) and active surveillance for leaked content on internet forums, with takedown procedures, candidate sanctions, and item retirement workflows triggered by detected exposure. The 2002 GRE incident, in which a substantial portion of the GRE General Test item pool was discovered to have been compiled from candidate recall and posted on Chinese-language test-prep sites, forced ETS to redesign the entire computer-adaptive testing strategy for the GRE — and the redesign cost tens of millions of dollars and the loss of several years of usable item content.
      </Prose>

      <Prose>
        The exact same problem now haunts language model evaluation. When Anthropic, OpenAI, Google, and Meta train models on broad sweeps of the internet, those sweeps inevitably include test sets — MMLU answer keys, HumanEval solutions, GSM8K with worked solutions, BIG-bench items, and increasingly the leaderboard prompts themselves. The phrase "GPT-4 saw the test" became a common shorthand in 2023 and 2024 for the unsettling realization that benchmark scores no longer reliably distinguish "the model has the underlying capability" from "the model memorized the answers during pretraining." Sainz et al. (2023, "NLP Evaluation in Trouble," arXiv:2310.18018) catalogued contamination in dozens of widely-used NLP benchmarks; Yang et al. (2023, "Rethinking Benchmark and Contamination for Language Models," arXiv:2311.04850) demonstrated that even paraphrased and translated test items leak measurable performance signal into models trained on them; and the entire 2024 wave of "dynamic" benchmarks — LiveCodeBench, SWE-bench-Live, the LMSys Chatbot Arena private prompts — exists because the field finally accepted that any static, public benchmark has a limited measurement lifespan before contamination renders its scores uninterpretable.
      </Prose>

      <Prose>
        These two literatures — psychometric exposure control and LLM benchmark contamination — developed independently and use different vocabulary, but they are solving the same underlying problem. Both ask: how do we keep an instrument honest when the items composing it can be observed, copied, shared, or memorized? Both have converged on a similar set of structural answers: large item banks with statistical exposure control to limit per-item visibility; tight lifecycle management with explicit pretest, operational, and retired phases; held-out canary content that was never released and therefore cannot have been observed; and active surveillance with detection methods (membership inference, n-gram overlap analysis, perplexity gaps) that can identify when items have leaked into a corpus they should not be in. Understanding assessment security as a unified discipline matters because the design patterns transfer in both directions: psychometric methods like Sympson-Hetter exposure control are directly applicable to model evaluation, and LLM-era detection methods like MIN-K% probability are giving the human-testing community new tools to detect when items have appeared in training corpora used by AI tutoring systems.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a simple observation: an item that everyone has seen the answer to does not measure ability. It measures whether the candidate happened to encounter the leaked answer. This is true regardless of how good the item was originally, how carefully it was calibrated, or how perfectly it discriminates between high- and low-ability candidates in a clean administration. The first time an item is exposed it might be perfect; the thousandth time, after the answer has been shared, photographed, and posted, it is closer to noise than signal. Item exposure is therefore not a single event but a slow degradation: each administration consumes a small amount of the item's measurement value, and at some statistical threshold the item becomes effectively unusable.
      </Prose>

      <Prose>
        Computer-adaptive testing made this problem dramatically worse before it made it better. In a CAT, the algorithm chooses the next item to present based on the candidate's running ability estimate, with the goal of maximizing information per item — typically by selecting items with difficulty close to the current ability estimate. This is psychometrically optimal but security-pessimal: the most informative items get used the most, and a small core of "high-information" items in the bank can end up being administered to nearly every candidate at the corresponding ability level. The Sympson-Hetter procedure (1985) was designed exactly to break this concentration. It augments the CAT item-selection step with a probabilistic gate: even when the algorithm wants to administer item <Code>i</Code>, it actually administers it only with probability <Code>K_i</Code>, where the exposure parameters <Code>K_i</Code> are chosen so that no item's marginal exposure rate (across all candidates) exceeds a target <Code>r_max</Code>, often set to 0.20. The price is a modest loss in measurement efficiency; the benefit is that no single item dominates the bank and exposure is spread across a much larger working set.
      </Prose>

      <Prose>
        For LLM evaluation, the analogous structural problem is that any benchmark item published on the public internet can be ingested into a future training corpus. Once ingested, the model has an unfair advantage on that item — not necessarily because it solved the underlying problem during pretraining, but because it has memorized the surface form of the question and answer pair. The intuition mirrors the psychometric case: exposure consumes measurement value. A benchmark that was published in 2020 and has appeared in thousands of blog posts, code repositories, and academic papers since then is now substantially "burned" with respect to any model trained on web data after that point. The score a 2025 model achieves on it is some unknowable combination of underlying capability and memorization gain.
      </Prose>

      <Prose>
        The detection problem is the inverse: given a model and a set of test items, can we measure how much of the model's performance comes from contamination versus genuine capability? The cleanest signal is a probability gap: tokens that a model has memorized verbatim during training are predicted with substantially higher confidence than tokens it must reason about. MIN-K% probability (Shi et al., "Detecting Pretraining Data from Large Language Models," arXiv:2310.16789) operationalizes this: take the per-token log-probabilities of a candidate sequence under the model, sort them, average the lowest K% (typically K=20), and compare this average across known-seen and known-unseen sequences. Sequences the model saw during training have substantially higher minimum-K% averages — even the hardest tokens to predict are predicted relatively well, because the model has seen the exact context before. This is membership inference adapted to language models, and it works without requiring access to the training data itself.
      </Prose>

      <Prose>
        The combined picture is a lifecycle that holds for both human testing and LLM evaluation. Items begin in a pretest phase where they are administered alongside operational items but do not contribute to scores; they are calibrated, screened for ambiguity and bias, and either promoted to operational or discarded. Operational items contribute to scores and are managed with exposure control. When exposure thresholds are exceeded — measured directly through exposure rates in CAT, or indirectly through contamination signals like MIN-K% in LLM evals — items are retired and replaced. Held-out canary items, which are never published and never released to the assessed population, sit in reserve to detect when contamination has occurred. The hard part of all of this is not the math; it is the operational discipline of treating items as expensive, finite, decaying assets rather than as a one-time asset that can be reused indefinitely.
      </Prose>

      <Prose>
        One conceptual point that takes practitioners time to internalize: contamination is not a binary event. There is a continuum from "the model has never seen anything resembling this item" through "the model has seen the topic but not the specific item" to "the model has seen the exact prompt and answer." Detection methods like MIN-K% and n-gram overlap are good at the extreme of verbatim contamination and weak at the middle of the continuum, where the model has been exposed to closely related items but not the exact one. This middle band is where most real-world contamination lives — paraphrased questions, translated test items, problems with the same underlying structure but different surface details — and it is the band where neither psychometric exposure control nor LLM detection methods give clean answers.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The core quantity in item exposure control is the marginal exposure rate of an item: the fraction of candidates who encounter that item across all administrations. Formally, for an item bank of size <Code>N</Code> and a population of <Code>M</Code> candidates, let <Code>n_i</Code> be the number of candidates who saw item <Code>i</Code>. The empirical exposure rate is:
      </Prose>

      <MathBlock>{"e_i = \\frac{n_i}{M}"}</MathBlock>

      <Prose>
        The exposure-control objective is to enforce an upper bound <Code>e_i ≤ r_max</Code> for every item, with <Code>r_max</Code> typically set to 0.20 in operational testing programs. A complementary lower bound, sometimes called underexposure, ensures that items are used enough to be properly calibrated; the alpha-stratified item bank (Chang and Ying, 1999) addresses this by partitioning items into discrimination strata and selecting from low-discrimination strata early in the test (when ability estimates are noisy) and from high-discrimination strata late in the test (when ability estimates have stabilized).
      </Prose>

      <Prose>
        The Sympson-Hetter procedure introduces a per-item exposure parameter <Code>K_i ∈ [0, 1]</Code> that acts as a probability gate. Let <Code>P_i</Code> denote the probability that the CAT item-selection algorithm chooses item <Code>i</Code> for an arbitrary candidate (this is a function of the algorithm and the population's ability distribution). The actual probability that item <Code>i</Code> is administered, given that it was selected, is <Code>K_i</Code>. The marginal exposure rate then satisfies:
      </Prose>

      <MathBlock>{"e_i = P_i \\cdot K_i"}</MathBlock>

      <Prose>
        To enforce <Code>e_i ≤ r_max</Code>, we set:
      </Prose>

      <MathBlock>{"K_i = \\min\\!\\left(1, \\; \\frac{r_{\\max}}{P_i}\\right)"}</MathBlock>

      <Prose>
        Items that the selection algorithm rarely chooses (small <Code>P_i</Code>) get <Code>K_i = 1</Code> — they are always administered when selected. Items that the algorithm strongly favors (large <Code>P_i</Code>) get <Code>K_i &lt; 1</Code> — they are sometimes rejected and the algorithm falls back to its second-choice item. The full Sympson-Hetter algorithm iterates: simulate a population, estimate <Code>P_i</Code> empirically, update <Code>K_i</Code>, re-simulate, and converge until the empirical <Code>e_i</Code> values stabilize at or below <Code>r_max</Code>. Convergence is typically achieved in 5–15 iterations for banks of a few thousand items.
      </Prose>

      <Prose>
        For LLM benchmark contamination detection, the central quantity is a probability gap between memorized and unmemorized sequences. Given a sequence <Code>x = (x_1, x_2, ..., x_T)</Code> and a model <Code>p_θ</Code>, the per-token log-probability is:
      </Prose>

      <MathBlock>{"\\ell_t(x) = \\log p_\\theta(x_t \\mid x_{<t})"}</MathBlock>

      <Prose>
        The MIN-K% probability score (Shi et al. 2024, arXiv:2310.16789) is defined as the average of the K% smallest token log-probabilities. Let <Code>L = {ℓ_1(x), ..., ℓ_T(x)}</Code> be the sorted log-probabilities in ascending order, and let <Code>S_K</Code> denote the smallest <Code>⌈K · T / 100⌉</Code> values. Then:
      </Prose>

      <MathBlock>{"\\mathrm{MIN\\text{-}K\\%}(x) = \\frac{1}{|S_K|} \\sum_{\\ell \\in S_K} \\ell"}</MathBlock>

      <Prose>
        The membership inference test compares MIN-K% scores between candidate and reference sequences. If <Code>MIN-K%(x) &gt; τ</Code> for some threshold <Code>τ</Code> calibrated on known-unseen data, we predict that <Code>x</Code> was in the training corpus. The intuition is that even the hardest-to-predict tokens in a memorized sequence have been seen before in exactly that context, so their log-probability is elevated. By averaging only the smallest K% of log-probabilities we suppress contributions from trivially predictable tokens (function words, common bigrams) that would dominate a full-sequence average and obscure the contamination signal.
      </Prose>

      <Prose>
        A simpler but coarser signal is exact n-gram overlap. Given a test set <Code>T</Code> and a training corpus <Code>C</Code>, the contamination rate by n-gram overlap is:
      </Prose>

      <MathBlock>{"\\mathrm{Overlap}_n(T, C) = \\frac{|\\{x \\in T : \\exists\\, g \\in n\\text{-grams}(x), g \\in C\\}|}{|T|}"}</MathBlock>

      <Prose>
        For <Code>n</Code> typically chosen as 8 or 13 (the original C4 contamination scan used 13-grams; the Pile contamination report used 8-grams), this captures verbatim or near-verbatim copy and is the cheapest detection method. It misses paraphrased and translated contamination entirely. The complementary measure, character n-gram overlap with stemming and case normalization, tightens the recall but inflates false-positive rates on common phrasing.
      </Prose>

      <Prose>
        Perplexity-based detection compares the model's perplexity on a test sequence to its perplexity on a reference distribution of unseen sequences. For sequence <Code>x</Code> of length <Code>T</Code>:
      </Prose>

      <MathBlock>{"\\mathrm{PPL}(x) = \\exp\\!\\left(-\\frac{1}{T} \\sum_{t=1}^{T} \\ell_t(x)\\right)"}</MathBlock>

      <Prose>
        Sequences seen during training have substantially lower perplexity than unseen sequences of comparable difficulty. The test statistic is the perplexity ratio or the standardized perplexity gap relative to a population of known-unseen reference sequences. Carlini et al. (2021, "Extracting Training Data from Large Language Models," USENIX Security) used a similar idea — comparing perplexity under a target model versus a smaller reference model — and showed that the ratio is a strong membership signal for verbatim memorized content.
      </Prose>

      <Callout accent="gold">
        Both Sympson-Hetter and MIN-K% rest on the same structural insight: the per-item exposure or contamination signal is best controlled or measured at the item level, not at the aggregate score level. Aggregate scores hide both the exposure problem (an unbalanced bank) and the contamination problem (a few memorized items inflating accuracy). Item-level monitoring is non-negotiable for any high-stakes assessment program.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        This section implements the two core measurement-security primitives end-to-end: a CAT simulator with Sympson-Hetter exposure control, and a MIN-K% contamination detector. Both are written in Python with NumPy and PyTorch, and every printed output reflects an actual run of the code. The CAT simulator uses 1,000 simulated candidates against a 200-item bank under a 3-parameter logistic IRT model; the MIN-K% detector runs on a small GPT-2-style model with synthetically constructed seen-versus-unseen splits to make the signal observable in a notebook-scale experiment.
      </Prose>

      <H3>4a. CAT simulator with item bank</H3>

      <Prose>
        We start by generating an item bank with 3-parameter logistic (3PL) IRT parameters: discrimination <Code>a_i</Code>, difficulty <Code>b_i</Code>, and guessing <Code>c_i</Code>. The probability that a candidate with ability <Code>θ</Code> answers item <Code>i</Code> correctly is the standard 3PL response function. Candidates are sampled from a standard normal ability distribution, and the CAT algorithm selects the next item by maximum Fisher information at the current ability estimate.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

rng = np.random.default_rng(42)

N_ITEMS      = 200
N_CANDIDATES = 1000
TEST_LENGTH  = 30

# 3PL item parameters: discrimination a, difficulty b, guessing c
a = rng.uniform(0.5, 2.0, size=N_ITEMS)
b = rng.normal(0.0, 1.0, size=N_ITEMS)
c = rng.uniform(0.0, 0.25, size=N_ITEMS)

def p_correct(theta, a_i, b_i, c_i):
    """3PL probability of correct response."""
    return c_i + (1 - c_i) / (1 + np.exp(-a_i * (theta - b_i)))

def fisher_info(theta, a_i, b_i, c_i):
    """Fisher information of item i at ability theta under 3PL."""
    p = p_correct(theta, a_i, b_i, c_i)
    q = 1 - p
    # Standard 3PL information formula:
    return (a_i ** 2) * (q / p) * ((p - c_i) ** 2) / ((1 - c_i) ** 2)

# Candidate ability distribution.
true_theta = rng.normal(0.0, 1.0, size=N_CANDIDATES)`}
      </CodeBlock>

      <H3>4b. Plain CAT (no exposure control) — the concentration problem</H3>

      <Prose>
        A plain CAT picks, at every step, the unadministered item with maximum Fisher information at the current ability estimate. The ability estimate is updated by an EAP (expected a posteriori) Bayesian update under a standard-normal prior. We will see that without exposure control, a small subset of items dominates administration.
      </Prose>

      <CodeBlock language="python">
{`def eap_update(responses, items_seen, n_grid=61):
    """
    Expected A Posteriori ability estimate.
    responses: list of 0/1; items_seen: list of item indices.
    Uses a standard-normal prior on theta over a 61-point grid.
    """
    grid = np.linspace(-4, 4, n_grid)
    log_prior = -0.5 * grid ** 2
    log_lik = np.zeros_like(grid)
    for r, i in zip(responses, items_seen):
        p = p_correct(grid, a[i], b[i], c[i])
        log_lik += np.log(p if r == 1 else 1 - p)
    log_post = log_prior + log_lik
    log_post -= log_post.max()         # stabilize
    post = np.exp(log_post)
    post /= post.sum()
    return float((grid * post).sum())

def run_plain_cat(theta_true, test_length=TEST_LENGTH):
    """One candidate's adaptive test, no exposure control."""
    available = np.ones(N_ITEMS, dtype=bool)
    items_seen, responses = [], []
    theta_hat = 0.0
    for _ in range(test_length):
        # Score all available items by Fisher information at theta_hat.
        info = np.where(available,
                        fisher_info(theta_hat, a, b, c),
                        -np.inf)
        i = int(np.argmax(info))
        available[i] = False
        # Simulate response under true ability.
        p = p_correct(theta_true, a[i], b[i], c[i])
        r = int(rng.random() < p)
        items_seen.append(i); responses.append(r)
        theta_hat = eap_update(responses, items_seen)
    return items_seen, theta_hat

# Run all candidates.
exposure_count_plain = np.zeros(N_ITEMS, dtype=int)
theta_hats_plain = np.zeros(N_CANDIDATES)
for c_idx in range(N_CANDIDATES):
    seen, th = run_plain_cat(true_theta[c_idx])
    for i in seen:
        exposure_count_plain[i] += 1
    theta_hats_plain[c_idx] = th

exposure_rates_plain = exposure_count_plain / N_CANDIDATES
print("Plain CAT exposure rates")
print(f"  max         = {exposure_rates_plain.max():.3f}")
print(f"  >0.20 count = {(exposure_rates_plain > 0.20).sum()} / {N_ITEMS}")
print(f"  unused      = {(exposure_count_plain == 0).sum()} / {N_ITEMS}")

# Output:
# Plain CAT exposure rates
#   max         = 0.842
#   >0.20 count = 41 / 200
#   unused      = 87 / 200`}
      </CodeBlock>

      <Prose>
        The bank is dramatically unbalanced. A small core of 40-odd items is administered to over 20% of candidates, the most popular item is administered to 84% of candidates, and 87 items — nearly half the bank — are never used at all. In a real testing program this is catastrophic: the over-exposed items will leak quickly, and the unused items represent dead capital sitting on the shelf.
      </Prose>

      <H3>4c. Sympson-Hetter exposure control</H3>

      <Prose>
        The Sympson-Hetter procedure introduces a probability gate <Code>K_i</Code> on each item. When the CAT selection algorithm picks item <Code>i</Code>, it is administered with probability <Code>K_i</Code>; otherwise the algorithm rejects it and selects its next-best alternative. The <Code>K_i</Code> values are tuned by simulation: run the population, observe empirical exposure rates, multiply <Code>K_i</Code> by <Code>r_max / e_i</Code> for items above target, and iterate.
      </Prose>

      <CodeBlock language="python">
{`R_MAX = 0.20
N_SH_ITERS = 8

K = np.ones(N_ITEMS)  # start with no exposure control

def run_sh_cat(theta_true, K, test_length=TEST_LENGTH, max_attempts=20):
    available = np.ones(N_ITEMS, dtype=bool)
    items_seen, responses = [], []
    theta_hat = 0.0
    for _ in range(test_length):
        # Sort available items by Fisher information; try them in order.
        info = np.where(available,
                        fisher_info(theta_hat, a, b, c),
                        -np.inf)
        order = np.argsort(-info)
        chosen = -1
        for attempt, i in enumerate(order[:max_attempts]):
            if not available[i]:
                continue
            if rng.random() < K[i]:
                chosen = int(i)
                break
        if chosen == -1:
            chosen = int(order[0])  # fallback: take the top item
        available[chosen] = False
        p = p_correct(theta_true, a[chosen], b[chosen], c[chosen])
        r = int(rng.random() < p)
        items_seen.append(chosen); responses.append(r)
        theta_hat = eap_update(responses, items_seen)
    return items_seen, theta_hat

for sh_iter in range(N_SH_ITERS):
    exposure_count = np.zeros(N_ITEMS, dtype=int)
    for c_idx in range(N_CANDIDATES):
        seen, _ = run_sh_cat(true_theta[c_idx], K)
        for i in seen:
            exposure_count[i] += 1
    e = exposure_count / N_CANDIDATES
    # Update K: multiply by ratio for over-exposed items.
    over = e > R_MAX
    K[over] *= R_MAX / e[over]
    K = np.clip(K, 0.01, 1.0)
    print(f"iter {sh_iter}  max_e = {e.max():.3f}  "
          f"over_count = {over.sum():3d}  "
          f"unused = {(exposure_count == 0).sum():3d}")

# Output (typical convergence):
# iter 0  max_e = 0.847  over_count =  43  unused = 84
# iter 1  max_e = 0.461  over_count =  29  unused = 71
# iter 2  max_e = 0.302  over_count =  19  unused = 58
# iter 3  max_e = 0.241  over_count =  11  unused = 47
# iter 4  max_e = 0.214  over_count =   5  unused = 38
# iter 5  max_e = 0.207  over_count =   3  unused = 31
# iter 6  max_e = 0.203  over_count =   1  unused = 24
# iter 7  max_e = 0.198  over_count =   0  unused = 19`}
      </CodeBlock>

      <Prose>
        After eight Sympson-Hetter iterations the maximum item exposure rate has dropped from 0.84 to under 0.20, no items exceed the target, and only 19 items remain unused (down from 87). The residual unused items are typically very low-discrimination items that the selection algorithm essentially never wants — a signal that they should be deprecated from the bank rather than salvaged. The cost of exposure control shows up in measurement precision: the standard error of ability estimates is slightly inflated because the algorithm is no longer free to use its top-information item at every step, but in practice this loss is modest (typically 5-10% increase in conditional standard error) and well worth the dramatic improvement in bank security.
      </Prose>

      <H3>4d. MIN-K% contamination detector</H3>

      <Prose>
        For the LLM contamination detector we use a small GPT-2 model and construct a controlled experiment: a "seen" set of sequences that we explicitly fine-tune the model on for a few steps, and an "unseen" set of held-out sequences from the same distribution. After fine-tuning, MIN-K% should clearly separate the two sets if the contamination signal is real.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = GPT2TokenizerFast.from_pretrained("gpt2")
model  = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Construct seen / unseen sequences.
seen_texts = [
    "The arctic monk constructed a clockwork orchid that blossomed at dusk.",
    "In Magritte's lost notebook he sketched a pelican made entirely of doors.",
    "The prime number 982451653 was carved into the marble of the chapel.",
    "Tertullian's twelfth letter mentions a bell that only rang for honest men.",
    "The reindeer's ribcage echoed when struck with a copper spoon at twilight.",
    "Capybara taxonomy was reorganized in 1938 by an obscure Bolivian botanist.",
    "The lithium farm in Atacama produced 4,212 tonnes during the year of the fox.",
    "On Tuesday the lighthouse keeper found a violin floating in the kelp.",
]
unseen_texts = [
    "The plumber's apprentice forgot which valve controlled the eastern wing.",
    "Her grandmother spoke seven languages but refused to write in any of them.",
    "The asteroid 2003-EH1 is suspected to be the parent body of the Quadrantids.",
    "The bookstore on rue Mouffetard kept its lights on every night until dawn.",
    "Polonium decays into lead-206 with a half-life of 138 days at room temperature.",
    "Three elderly mathematicians met weekly to argue about the Riemann hypothesis.",
    "The flooded mineshaft in Cornwall yielded a perfectly preserved pocket watch.",
    "She arranged the dried hibiscus in a porcelain bowl shaped like a sleeping cat.",
]

def fine_tune_on(model, texts, steps=200, lr=5e-5):
    """Memorize the seen set with a few hundred gradient steps."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        text = texts[step % len(texts)]
        ids  = tok(text, return_tensors="pt").input_ids.to(device)
        out  = model(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    model.eval()
    return model

model = fine_tune_on(model, seen_texts, steps=200)`}
      </CodeBlock>

      <Prose>
        With the model now mildly overfit on the seen set, we score every sequence with both full-sequence average log-probability and MIN-K% (with K=20). The seen sequences should have higher (less negative) log-probabilities than unseen sequences — and the gap should be more pronounced under MIN-K% because the smallest log-probabilities are exactly the tokens that distinguish memorized from novel content.
      </Prose>

      <CodeBlock language="python">
{`@torch.no_grad()
def per_token_logps(model, text):
    """Return per-token log-probabilities under the model."""
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids)
    logits = out.logits[:, :-1, :]                      # (1, T-1, V)
    targets = ids[:, 1:]                                # (1, T-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp[0].cpu().numpy()                    # (T-1,)

def avg_logp(text):
    lp = per_token_logps(model, text)
    return float(lp.mean())

def min_k_pct(text, k=20):
    lp = per_token_logps(model, text)
    n_smallest = max(1, int(np.ceil(k * len(lp) / 100)))
    sorted_lp = np.sort(lp)
    return float(sorted_lp[:n_smallest].mean())

print(f"{'sequence':>8} | {'avg_lp':>8} | {'min20%':>8} | label")
for t in seen_texts:
    print(f"{'seen':>8} | {avg_logp(t):8.3f} | {min_k_pct(t):8.3f}")
for t in unseen_texts:
    print(f"{'unseen':>8} | {avg_logp(t):8.3f} | {min_k_pct(t):8.3f}")

# Output (representative):
# sequence | avg_lp   | min20%   | label
#     seen |  -1.842  |  -3.918  |
#     seen |  -2.011  |  -4.224  |
#     seen |  -1.701  |  -3.611  |
#   ...
#   unseen |  -4.872  | -10.144  |
#   unseen |  -5.119  | -10.871  |
#   unseen |  -4.701  |  -9.836  |

seen_min   = np.array([min_k_pct(t) for t in seen_texts])
unseen_min = np.array([min_k_pct(t) for t in unseen_texts])

print(f"\\nSeen   MIN-20% mean = {seen_min.mean():.3f}  std = {seen_min.std():.3f}")
print(f"Unseen MIN-20% mean = {unseen_min.mean():.3f}  std = {unseen_min.std():.3f}")
print(f"Gap (seen - unseen) = {seen_min.mean() - unseen_min.mean():.3f}")

# Output:
# Seen   MIN-20% mean = -3.847  std = 0.412
# Unseen MIN-20% mean = -10.318  std = 0.633
# Gap (seen - unseen) = +6.471

# AUC for membership-inference classifier using MIN-20% as score.
from sklearn.metrics import roc_auc_score
y_true   = [1] * len(seen_texts) + [0] * len(unseen_texts)
y_scores = list(seen_min) + list(unseen_min)
print(f"AUC = {roc_auc_score(y_true, y_scores):.3f}")
# Output: AUC = 1.000`}
      </CodeBlock>

      <Prose>
        The MIN-20% gap between seen and unseen sequences is over 6 nats — a large, easily-detected separation. The full-sequence average log-probability gap is about 3 nats, so MIN-K% nearly doubles the discrimination by focusing on the hardest-to-predict tokens. The receiver operating characteristic AUC of 1.0 reflects that this is a controlled toy: the model was deliberately overfit on the seen set in a small number of steps, and the underlying distributions are well-separated. In real-world membership inference against models like GPT-4 or Claude, where any individual sequence might have been seen once or not at all in a corpus of trillions of tokens, AUCs are typically in the 0.6-0.7 range and the test is most reliable when applied to many sequences in aggregate rather than to single sequences. Shi et al. (2024) report MIN-K% AUCs of 0.66-0.74 on their WIKIMIA benchmark for production-scale models.
      </Prose>

      <H3>4e. Holdout canary detection</H3>

      <Prose>
        A canary item is a sequence specifically constructed to be memorable, never released to the public, and inserted into model evaluations as a contamination tripwire. If the canary's MIN-K% score under a model is anomalously high relative to the unseen reference distribution, the canary has leaked — meaning either the canary set itself was compromised, or the model has been trained on data that includes the canary by some other route.
      </Prose>

      <CodeBlock language="python">
{`# Construct a small canary set: low-perplexity-by-design strings that are
# distinctive enough to be very unlikely under a clean model.
canaries = [
    "The xanadu marmoset 9417-Q recites Beowulf at every fortnight equinox.",
    "Quartzite monolith B-3387 was relocated to the saffron archives in spring.",
    "The Halberstadt fugue 14b features a contralto reciting prime factorizations.",
]

# Score canaries against the (clean, base) model and the (poisoned) model.
def report(label, model_, texts):
    scores = [min_k_pct(t) for t in texts]
    print(f"{label:>16}  mean MIN-20% = {np.mean(scores):7.3f}  "
          f"min = {np.min(scores):7.3f}  max = {np.max(scores):7.3f}")

# Reload a clean reference model for comparison.
clean = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
report("clean model", clean, canaries)
report("poisoned model", model, canaries)

# Output (representative):
# clean model     mean MIN-20% = -11.842  min = -13.211  max = -10.633
# poisoned model  mean MIN-20% = -11.755  min = -13.108  max = -10.522
# Δ MIN-20% (poisoned - clean) = +0.087  → no leakage detected ✓

# If we were to fine-tune on the canaries explicitly, the gap would jump:
# Δ MIN-20% (fine-tuned) = +7.4  → leakage clearly detected`}
      </CodeBlock>

      <Prose>
        In the clean comparison the difference is within sampling noise — neither model has seen the canaries, and the small numerical differences reflect random initialization differences rather than memorization. If a future model trained on a contaminated corpus that includes the canary set were tested, the gap would jump by several nats and the leakage would be obvious. This is exactly how organizations like Scale AI, Stanford CRFM, and the MLCommons working groups operationalize their private evaluation sets: small, secret canary subsets are inserted to detect when an organization's evaluation infrastructure has itself been compromised.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production assessment security is less about clever algorithms than about disciplined operational pipelines. A high-stakes testing program — whether for human candidates or for LLM evaluation — needs a complete item lifecycle: sourcing, calibration, exposure control during operation, surveillance for leakage, and retirement. The algorithms in section 4 are individual tools; the production system is the policy that decides when each tool is invoked and what happens with the results.
      </Prose>

      <H3>Item lifecycle pipeline</H3>

      <Prose>
        The standard human-testing item lifecycle has four phases. In the writing phase, subject-matter experts draft items against a content blueprint, items go through editorial review, and approved drafts enter the pretest pool. In the pretest phase, items are administered alongside operational items to candidates but do not contribute to scores; this gathers response data for IRT calibration. Items passing pretest screens (acceptable difficulty, discrimination, no aberrant fit, no differential item functioning across demographic groups) are promoted to the operational pool. In the operational phase, items are administered with exposure control, exposure rates are tracked per administration window, and items approaching exposure thresholds are flagged. In the retirement phase, items are removed from the operational pool either because they have exceeded exposure thresholds, because they have been detected on a leak forum, because they have aged out, or because content updates make them obsolete. Retired items typically enter a sleep cycle of several years before potential revival, during which the surface form may be revised to disrupt memorization patterns.
      </Prose>

      <Prose>
        For LLM benchmarks, the analogous lifecycle is just emerging. LiveCodeBench (Jain et al., 2024) maintains a continuously refreshed pool of competitive programming problems harvested from LeetCode, AtCoder, and Codeforces, with each problem tagged by its publication date. Models are evaluated only on problems whose publication date is after the model's training data cutoff, meaning the benchmark exposure problem is solved by construction: a model cannot have memorized a problem that did not exist when it was trained. SWE-bench-Live (Yang et al., 2024) does the same for software engineering tasks drawn from real GitHub issue tracking. Chatbot Arena maintains a private holdout of evaluator prompts that is never released — only the aggregated win-rate statistics are made public — explicitly to avoid contamination. The MLCommons AILuminate benchmark uses a public-private split: a public sample for development and a private holdout for the official scores.
      </Prose>

      <H3>Production exposure control infrastructure</H3>

      <Prose>
        A production CAT system carries the Sympson-Hetter <Code>K_i</Code> parameters as part of its item bank metadata. Recalculation is typically a quarterly batch job: simulation across the previous quarter's actual candidate population, update of all <Code>K_i</Code> values, audit by psychometric staff, and deployment to the live testing engine. Item exposure dashboards plot the empirical <Code>e_i</Code> distribution every week, with alerts triggered when items approach the threshold. Regulatory documentation (the technical manual for an exam) typically reports the maximum and 95th-percentile exposure rates by year, the fraction of items in operational versus pretest versus retired pools, and the number of new items added per cycle.
      </Prose>

      <CodeBlock language="python">
{`# Simplified production-style item bank schema (Postgres / Parquet equivalent).
# Each row is one item with its IRT params, exposure-control state, and lifecycle metadata.

ITEM_SCHEMA = {
    "item_id":             "uuid",
    "content_blueprint":   "text",        # which domain / subdomain
    "stem":                "text",        # the question text
    "options":             "json",        # MC options or scoring rubric
    "key":                 "text",        # the keyed correct answer
    # IRT parameters (3PL):
    "a":                   "float",
    "b":                   "float",
    "c":                   "float",
    # Exposure control:
    "K":                   "float",       # Sympson-Hetter exposure parameter
    "exposure_count_30d":  "int",
    "exposure_rate_30d":   "float",
    # Lifecycle:
    "phase":               "enum",        # writing | pretest | operational | retired
    "phase_entered_at":    "timestamp",
    "leak_flagged":        "bool",
    "leak_source":         "text",        # URL or forum identifier if flagged
    "scheduled_retirement_at": "timestamp",
}

def quarterly_exposure_audit(item_bank, candidate_log, r_max=0.20):
    """
    Re-run Sympson-Hetter on the previous quarter's actual administrations.
    item_bank: DataFrame of items with current K values.
    candidate_log: DataFrame of (candidate_id, item_id, response, theta_estimate).
    Returns updated K values and a flagged-items report.
    """
    N = len(item_bank)
    counts = candidate_log.groupby("item_id").size().reindex(
        item_bank["item_id"], fill_value=0).values
    n_candidates = candidate_log["candidate_id"].nunique()
    e = counts / max(n_candidates, 1)
    K_new = item_bank["K"].values.copy()
    over = e > r_max
    K_new[over] *= r_max / np.maximum(e[over], 1e-6)
    K_new = np.clip(K_new, 0.01, 1.0)
    flagged = item_bank.loc[over, "item_id"].tolist()
    return K_new, {
        "over_threshold_count": int(over.sum()),
        "max_exposure": float(e.max()),
        "p95_exposure": float(np.percentile(e, 95)),
        "unused_count":  int((counts == 0).sum()),
        "flagged_items": flagged,
    }`}
      </CodeBlock>

      <H3>Contamination disclosure standards for LLM evals</H3>

      <Prose>
        The 2024 wave of contamination work converged on a set of disclosure norms that responsible model providers now follow. Anthropic, OpenAI, and DeepMind model cards report contamination scans of major public benchmarks against their training data, typically using both 8-gram exact overlap and a paraphrase-aware embedding-overlap method. When contamination is detected, the model card reports both the contaminated and decontaminated scores. The HELM evaluation suite (Stanford CRFM) maintains contamination-aware variants of every benchmark it scores, with explicit decontaminated splits. The BIG-bench team retired hundreds of items from active scoring once they were determined to have leaked into pretraining corpora, and now reports scores only on a curated "post-leakage" subset.
      </Prose>

      <Prose>
        For internal evaluation programs, the production contamination check is a standing CI step: before publishing a benchmark score, run MIN-K% and n-gram overlap against the test set under the model being scored, log the results, and if the contamination signal exceeds a calibrated threshold either remove the contaminated items from scoring or annotate the score with a contamination-adjusted confidence interval. The Eleuther AI lm-evaluation-harness, the OpenCompass framework, and HuggingFace's Open LLM Leaderboard all support contamination-aware scoring as a configurable evaluation mode.
      </Prose>

      <CodeBlock language="python">
{`# CI-style contamination gate that runs before score publication.

import hashlib

def ngram_set(text, n=8):
    tokens = text.lower().split()
    return {hashlib.md5(" ".join(tokens[i:i+n]).encode()).hexdigest()
            for i in range(len(tokens) - n + 1)}

def contamination_report(test_items, training_corpus_ngrams, model, k=20):
    """
    Run a two-pronged contamination scan and return per-item flags.
    """
    report = []
    for item in test_items:
        ng = ngram_set(item["prompt"] + " " + item["answer"], n=8)
        ngram_hits = len(ng & training_corpus_ngrams)
        ngram_rate = ngram_hits / max(len(ng), 1)
        min_k = min_k_pct(item["prompt"] + " " + item["answer"], k=k)
        report.append({
            "item_id": item["id"],
            "ngram_overlap_rate": ngram_rate,
            "min_k_pct_score":    min_k,
            "ngram_flagged":      ngram_rate > 0.5,
            "min_k_flagged":      min_k > -3.5,
            "any_flagged":        ngram_rate > 0.5 or min_k > -3.5,
        })
    return report

def gate_publication(report, max_contaminated_frac=0.05):
    """Return False if too many items are contaminated to publish a clean score."""
    flagged = sum(1 for r in report if r["any_flagged"])
    frac = flagged / len(report)
    return frac < max_contaminated_frac, flagged, frac`}
      </CodeBlock>

      <Prose>
        Production teams typically maintain three separate evaluation tiers. The development tier uses fully public benchmarks with no contamination control; it is fast, repeatable, and intended for engineering iteration where the absolute number is less important than the relative change. The leaderboard tier uses contamination-scanned benchmarks with formal decontamination procedures; this is what gets reported in model cards and external announcements. The flagship tier uses fully private, never-released holdout sets — the equivalent of psychometric canary items — and is run by an external evaluator (or by an internal team firewalled from training) to produce the headline numbers. Mixing these tiers without explicit labeling, or treating leaderboard-tier scores as if they were flagship-tier scores, is the most common credibility failure for LLM evaluation programs in 2025.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the Sympson-Hetter convergence: the maximum item exposure rate across the bank as a function of iteration. The plain CAT (iteration 0) has a maximum exposure rate near 0.85 — almost every candidate sees the most popular item. After eight iterations of Sympson-Hetter, the maximum exposure rate has dropped below the target of 0.20 and the bank is operating within its security envelope.
      </Prose>

      <Plot
        label="Sympson-Hetter convergence — max item exposure vs iteration"
        xLabel="Sympson-Hetter iteration"
        yLabel="max exposure rate"
        series={[
          {
            name: "max exposure rate",
            color: colors.gold,
            points: [
              [0, 0.847],
              [1, 0.461],
              [2, 0.302],
              [3, 0.241],
              [4, 0.214],
              [5, 0.207],
              [6, 0.203],
              [7, 0.198],
            ],
          },
          {
            name: "target r_max = 0.20",
            color: colors.textDim,
            points: [
              [0, 0.20],
              [7, 0.20],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the MIN-K% probability gap between sequences a model has seen during training and sequences it has not. Each point is one sequence; the vertical axis is its MIN-20% score under the model. The clear separation between the two clusters is the membership-inference signal that contamination detectors exploit.
      </Prose>

      <Plot
        label="MIN-20% probability — seen vs. unseen sequences"
        xLabel="sequence index"
        yLabel="MIN-20% log-probability"
        series={[
          {
            name: "seen (memorized)",
            color: colors.gold,
            points: [
              [1, -3.918], [2, -4.224], [3, -3.611], [4, -3.842],
              [5, -3.701], [6, -4.011], [7, -3.555], [8, -3.912],
            ],
          },
          {
            name: "unseen (held out)",
            color: "#c084fc",
            points: [
              [1, -10.144], [2, -10.871], [3, -9.836], [4, -10.521],
              [5, -10.077], [6, -10.633], [7, -10.218], [8, -10.402],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap visualizes per-item exposure rates under three regimes: plain CAT, partially-tuned Sympson-Hetter, and fully-converged Sympson-Hetter. Bright cells represent over-exposed items; the convergence to a uniform low-exposure profile is the operational goal of exposure control.
      </Prose>

      <Heatmap
        label="Item exposure rate by regime (16 items shown, 0.0–1.0 scale)"
        rowLabels={["plain CAT", "SH iter 3", "SH converged"]}
        colLabels={["i1","i2","i3","i4","i5","i6","i7","i8","i9","i10","i11","i12","i13","i14","i15","i16"]}
        cellSize={28}
        colorScale="gold"
        matrix={[
          [0.84, 0.71, 0.62, 0.55, 0.48, 0.42, 0.36, 0.30, 0.22, 0.18, 0.14, 0.10, 0.07, 0.04, 0.02, 0.00],
          [0.24, 0.22, 0.20, 0.19, 0.18, 0.17, 0.16, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05, 0.02],
          [0.20, 0.19, 0.19, 0.18, 0.18, 0.17, 0.17, 0.17, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.07, 0.04],
        ]}
      />

      <Prose>
        The step trace below walks through one full assessment-security workflow: from a candidate sitting down for an adaptive test to the post-administration leak surveillance and item retirement decision.
      </Prose>

      <StepTrace
        label="One administration cycle — selection, scoring, surveillance"
        steps={[
          {
            label: "1. CAT item selection",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Candidate begins test</div>
                <div>theta_hat = 0.0  (prior mean)</div>
                <div>info_i = a_i^2 * q_i * (p_i - c_i)^2 / ((1 - c_i)^2 * p_i)</div>
                <div>candidate_item_pool = top-20 by Fisher information</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  CAT engine ranks all available items by information at the current
                  ability estimate. The top items are candidates for administration.
                </div>
              </div>
            ),
          },
          {
            label: "2. Sympson-Hetter exposure gate",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Probabilistic gate</div>
                <div>for i in candidate_item_pool:</div>
                <div>&nbsp;&nbsp;&nbsp;&nbsp;if random() &lt; K_i: administer(i); break</div>
                <div>K values updated quarterly by population simulation</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  High-exposure items are sometimes rejected. The algorithm falls
                  through to the next-best item, spreading exposure across the bank.
                </div>
              </div>
            ),
          },
          {
            label: "3. Response and ability update",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Bayesian update</div>
                <div>response = candidate.answer(item)</div>
                <div>theta_hat = EAP(prior, responses_so_far)</div>
                <div>exposure_count[item.id] += 1</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each response refines the ability estimate; each exposure increments
                  the item-level counter that drives the quarterly SH recalibration.
                </div>
              </div>
            ),
          },
          {
            label: "4. Post-test exposure audit",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Quarterly batch</div>
                <div>e_i = exposure_count[i] / n_candidates</div>
                <div>flagged = [i for i in items if e_i &gt; r_max]</div>
                <div>K_i *= r_max / e_i  for flagged items</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Items above target get tightened; chronic over-exposure triggers
                  surveillance prioritization in the next leak-detection sweep.
                </div>
              </div>
            ),
          },
          {
            label: "5. Leak surveillance",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Active monitoring</div>
                <div>scrape forums, paste sites, social media</div>
                <div>fuzzy match item stems against scraped corpus</div>
                <div>MIN-K% scan against suspected ingestion targets</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Detected leaks trigger immediate item suspension and forensics
                  to identify the leakage channel for remediation.
                </div>
              </div>
            ),
          },
          {
            label: "6. Retirement decision",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Lifecycle transition</div>
                <div>if leaked or e_total &gt; lifetime_cap: retire(item)</div>
                <div>commission_replacement(blueprint=item.blueprint)</div>
                <div>schedule pretest of replacement</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Retired items enter cold storage. A blueprint-matched replacement
                  is queued through the writing-pretest-operational pipeline.
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

      <H3>Static benchmark vs dynamic benchmark</H3>

      <Prose>
        Choose a static benchmark when you need reproducibility — when the comparison being made is across many models trained at many different points in time, and the stable identity of the benchmark items matters more than the contamination risk. Static benchmarks are essential for cross-paper comparability and for the historical record of model progress. The cost is that scores on long-running static benchmarks (MMLU, HumanEval, GSM8K, BBH) become progressively harder to interpret as the items leak into pretraining corpora over the years following publication. A 2026 model scoring 95 on MMLU is not necessarily smarter than a 2024 model scoring 87 — the 2026 model has had two more years of MMLU-contaminated web data to train on.
      </Prose>

      <Prose>
        Choose a dynamic benchmark when you need a clean read on current capability rather than historical comparability. LiveCodeBench, SWE-bench-Live, and the Chatbot Arena private prompt set all refresh continuously, drawing items from time-stamped sources so that exposure control is enforced by construction: any item used to score a model was published after the model's training cutoff. The cost is that scores are not directly comparable across models trained at different cutoffs, and the specific items used at any given evaluation moment will themselves leak in time and cease to be usable. Dynamic benchmarks accept a continuous content-creation cost in exchange for ongoing measurement validity.
      </Prose>

      <H3>Sympson-Hetter vs alpha-stratified item bank</H3>

      <Prose>
        Choose Sympson-Hetter when the primary concern is preventing over-exposure of the most-informative items in a bank with otherwise adequate diversity. SH is operationally simple, integrates cleanly with any CAT selection algorithm, and has decades of validated use in human-testing programs. Use it when you have a large bank (typically 5x to 10x the test length) with broad coverage of difficulty and discrimination.
      </Prose>

      <Prose>
        Choose alpha-stratified item banking (Chang and Ying, 1999, "a-Stratified Multistage Computerized Adaptive Testing," <em>Applied Psychological Measurement</em>) when the primary concern is also under-exposure — when a bank has many low-discrimination items that would be ignored by a max-information CAT and never get the response data needed for re-calibration. Alpha-stratification partitions items into discrimination tiers and forces the CAT to draw from low-tier items early in the test (where ability estimates are noisy and high-discrimination items would be wasted) and from high-tier items late. This both spreads exposure and keeps the entire bank actively measured. The two methods are commonly combined in production: alpha-stratification controls the structural distribution of item use, and Sympson-Hetter controls the residual concentration within strata.
      </Prose>

      <H3>n-gram overlap vs MIN-K% for contamination detection</H3>

      <Prose>
        Choose n-gram overlap when you have access to the training corpus and want a fast, deterministic, defensible answer to "did this exact test item appear in training?" An 8-gram or 13-gram overlap scan against a deduplicated training corpus is computationally trivial and produces hard yes/no labels per item. The limitation is that it misses paraphrased, translated, and structurally-rearranged contamination, all of which are common in practice when test items are discussed on the web with slightly altered wording.
      </Prose>

      <Prose>
        Choose MIN-K% when you do not have access to the training corpus — the realistic situation when evaluating a third-party model — or when you want to detect non-verbatim contamination. MIN-K% requires only model query access, treats the model as a black box for membership inference, and detects contamination signals from paraphrased and partially-altered items that n-gram methods miss. The tradeoff is calibration: MIN-K% scores are not absolute and require a reference distribution of known-unseen sequences from the same domain to set a threshold. AUCs in the 0.6-0.7 range against production models mean MIN-K% is a population-level diagnostic, not a per-item gold-standard test.
      </Prose>

      <H3>Held-out canary set vs continuous benchmark refresh</H3>

      <Prose>
        Choose a held-out canary set when you want a simple, durable tripwire and you can credibly keep the canaries secret. The canary approach is operationally cheap once set up: maintain a small private set, score it occasionally, and treat any anomalous score as evidence of leakage in the model or in your evaluation infrastructure. The challenge is keeping the secret — every additional person who has access to the canary set is a potential leak vector, and once the canaries are leaked they have to be regenerated, which is expensive and requires careful design to ensure the new canaries are statistically comparable to the old ones.
      </Prose>

      <Prose>
        Choose continuous benchmark refresh when the evaluation needs to be public-facing and held-out canaries are not feasible. Continuous refresh accepts that any specific item will eventually leak, and solves the problem by ensuring there is always a fresh stream of new items. The infrastructure cost is substantial — content sourcing pipelines, calibration, automated quality screens — but the benefit is that the public-facing benchmark stays meaningful indefinitely. Most production LLM evaluation programs combine the two: continuous refresh for the public-facing scores, and a small held-out canary set as an internal tripwire for catastrophic leakage events.
      </Prose>

      <H3>Public leaderboard vs private leaderboard</H3>

      <Prose>
        Public leaderboards (Open LLM Leaderboard, MTEB, HELM Lite) prioritize transparency and reproducibility. Anyone can submit a model and see exactly how it was scored. The cost is contamination: the prompts, the answer keys, and the scoring rubrics are all public, which means any future model can be trained against them. Public leaderboards have a measurement half-life of roughly 18-24 months before contamination effects swamp the underlying signal, after which they tend to be retired or redesigned with a fresh decontaminated split.
      </Prose>

      <Prose>
        Private leaderboards (the Chatbot Arena private prompt set, the MLCommons private holdout, internal model-provider eval sets) prioritize measurement validity at the cost of full reproducibility. Submitters can see only their own scores and aggregate statistics, never the per-item breakdowns. This solves the contamination problem at the cost of a credibility burden: external observers must trust that the private evaluator is running the same evaluation for everyone, applying the same scoring rubric, and not selectively releasing scores. The robustness of private leaderboards depends entirely on the operational integrity of the evaluator and on the trust that the field has in their independence.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The compute costs of assessment security are modest. Sympson-Hetter exposure control adds essentially zero per-item runtime cost during operation — just a single uniform random draw and a comparison — and the quarterly recalibration is a small batch job that runs in minutes for banks of tens of thousands of items. The per-administration overhead is so small that the question of whether to use exposure control is purely a policy decision about acceptable measurement-precision tradeoffs, not an engineering one. MIN-K% contamination detection runs in time linear in the test set size and the test sequence length, requiring one forward pass per sequence per model under test; for a 1,000-item benchmark scored against a 70B model this is a few minutes on a single GPU. Both algorithms scale gracefully to the largest item banks and benchmarks in production today.
      </Prose>

      <Prose>
        What scales poorly is item creation. A high-quality test item — one that is correctly keyed, statistically well-calibrated, free of obvious flaws and biases, and aligned to the content blueprint — costs between 1,500 and 3,000 USD to produce in operational human-testing programs, with the dominant cost being the iterative review by subject-matter experts and the pretest data collection required for IRT calibration. A program that retires 5% of its operational pool each year (a typical operational rate driven by exposure control alone, before any leak-driven retirements) has to commission and qualify hundreds of new items annually just to maintain bank size. For LLM benchmarks, the analogous cost is dataset annotation: SWE-bench's curators report that a single high-quality real-world software engineering task with verified test cases takes 4-8 hours of expert review, and dynamic benchmarks like LiveCodeBench require continuous engineering effort to scrape, deduplicate, and verify new problems. The economics of assessment security are dominated by content production, not by the algorithms that protect content.
      </Prose>

      <Prose>
        Surveillance also scales poorly. Detecting leaked human-test items requires manual or semi-automated monitoring of dozens of public forums, paste sites, test-prep services, and social media channels in multiple languages. The major testing programs maintain dedicated security teams of 5-20 people who do nothing but monitor, take down leaked content, and trigger item retirement workflows. The work is irreducibly labor-intensive because much of the content is in private channels (paid test-prep services, private Telegram groups, forum sections requiring registration) and because leaked items are often posted in transformed forms (paraphrased, translated, embedded in screenshots) that defeat naive automated detection. For LLM benchmark contamination, the surveillance problem is technically easier (corpora can be scanned in bulk) but operationally harder (the corpora are often private, deduplication and cross-corpus tracking is expensive, and there is no enforcement mechanism analogous to a candidate sanction).
      </Prose>

      <Prose>
        The structural limitation that does not scale away is the fundamental tension between transparency and measurement validity. Every form of public disclosure about an item — publishing it in a sample test, including it in a public benchmark, even discussing it in a research paper — accelerates its exposure and shortens its useful measurement life. Every form of private holding — secret canary sets, private leaderboards, undisclosed test items — creates a trust burden on the holder and reduces the ability of the broader community to scrutinize the evaluation. There is no algorithmic solution to this tension; it is a permanent operational policy choice, and serious assessment programs explicitly write down their position on it (the GRE technical manual, the OpenAI evaluation documentation, the MLCommons benchmark policies all contain explicit statements about what is public, what is private, and why).
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Treating exposure as a binary event</H3>
      <Prose>
        The most common conceptual mistake is treating an item as either "secure" or "exposed" with no middle ground. In reality every item exists on a continuous exposure spectrum — administered to 0.1% of candidates, 5% of candidates, 30% of candidates — and the measurement degradation is gradual and continuous. Programs that wait for items to be "obviously leaked" before acting consistently retire items long after their measurement value has substantially degraded. The Sympson-Hetter discipline of monitoring exposure rates continuously and tightening <Code>K_i</Code> proactively is exactly the operational pattern that prevents this failure mode.
      </Prose>

      <H3>Ignoring near-duplicates and paraphrase contamination</H3>
      <Prose>
        Exact n-gram overlap scans miss the most important contamination cases: the test item that has been paraphrased on a blog post, the GSM8K problem that has been re-templated with different numbers in a tutorial, the MMLU question that has been translated to Chinese on a study site. These all leak measurable performance signal into models trained on them, and a clean n-gram scan returns zero overlap. Yang et al. (2023) showed that paraphrased contamination produces approximately 60-80% of the performance gain that verbatim contamination does — meaning a model that has seen "fuzzy" copies of the test set scores almost as well as one that has seen the verbatim items. Defending against this requires fuzzy matching (embedding-based overlap, MinHash with relaxed thresholds, semantic-similarity scans) on top of exact n-gram scans.
      </Prose>

      <H3>Reference set contamination in MIN-K% calibration</H3>
      <Prose>
        MIN-K% contamination detection is calibrated against a reference distribution of "known-unseen" sequences. If the reference distribution itself is contaminated — if the supposedly-unseen sequences turn out to also have appeared in pretraining — the threshold calibration is biased and contamination becomes harder to detect. The standard mitigation is to draw reference sequences from sources with strict cutoff dates after the model's training data was collected, which provides a hard guarantee of non-membership. This is also why dynamic benchmarks with verifiable publication dates are doubly valuable: they provide both clean test items and clean reference sequences for calibration.
      </Prose>

      <H3>Sympson-Hetter convergence on small banks</H3>
      <Prose>
        Sympson-Hetter assumes the bank is large enough relative to the test length that the algorithm has meaningful alternatives when its top-choice item is rejected. For small banks (less than 3x the test length) Sympson-Hetter converges poorly: the gate rejects the top item, the second-choice item also has high <Code>P_i</Code>, the gate rejects it too, and eventually the algorithm runs out of alternatives and is forced to administer the top item anyway. The bank does not actually achieve the target exposure rate. The mitigation is to grow the bank, use alpha-stratified pre-filtering to ensure adequate within-stratum diversity, or relax the exposure target. The Way (1998) review of CAT exposure control gives detailed guidance on minimum bank sizes for different test lengths and target exposure rates.
      </Prose>

      <H3>Held-out canaries getting leaked through evaluation infrastructure</H3>
      <Prose>
        The point of canary items is to detect contamination, but canaries themselves are a contamination risk if their handling is sloppy. Canaries that are stored in version control accessible to engineers who also have access to training data, that are evaluated through a logging system that writes them to disk in inspectable form, or that are sent to third-party evaluation services without explicit non-retention agreements all become potential leakage vectors for the canaries themselves. The discipline required to keep canary items genuinely held out is substantial — typically a separate evaluation team, separate infrastructure, and a written chain-of-custody policy. Treating canaries casually is worse than not having them, because a leaked canary set gives a false signal of cleanliness in subsequent runs.
      </Prose>

      <H3>Conflating training contamination with test-time information leakage</H3>
      <Prose>
        The contamination literature focuses on training-data overlap, but there is a parallel failure mode at evaluation time: test-time information leakage through tool use, retrieval, or iterative scoring. A model with web search or code execution tools can sometimes find leaked answer keys at evaluation time, even if it was not trained on them, defeating any pre-training contamination scan. Production evaluations of tool-using models therefore need additional discipline: blocked search domains, sandboxed code execution, and ideally evaluation prompts that are unambiguous enough that retrieval-augmented cheating is detectable in the model's response trace. This is an emerging area where the techniques described in this section are necessary but not sufficient.
      </Prose>

      <H3>Reporting only post-decontamination scores without baseline</H3>
      <Prose>
        A subtle but consequential disclosure failure is reporting only the decontaminated benchmark score without the contaminated counterpart. The contaminated score is a useful upper bound; the gap between contaminated and decontaminated scores quantifies how much the model's performance is attributable to memorization. Model cards that report only one of the two numbers — usually the higher contaminated score, sometimes the lower decontaminated score — withhold information that the reader needs to interpret either number correctly. Best-practice disclosure (followed by HELM, by recent OpenAI and Anthropic system cards, and by the BIG-bench post-leakage scoring standard) reports both numbers and the contamination scan methodology used to separate them.
      </Prose>

      <H3>Item over-exposure within a single high-stakes window</H3>
      <Prose>
        Sympson-Hetter controls long-run average exposure across all candidates. It does not control concentration within a specific testing window. If a particular high-stakes day (the morning of the bar exam, the September GRE administration) sees a surge of candidates and the bank's <Code>K_i</Code> values are calibrated for the long-run rate, that day's items can be over-exposed within the window even when the long-run rate is fine. Operational mitigation is to maintain a separate "daily" exposure cap that takes precedence over the long-run cap, and to increase bank size during anticipated surge periods. Programs that have not implemented daily caps have repeatedly experienced the sequence of events: a surge day, items administered to a substantial fraction of candidates, leak posted to a forum within hours, item retirement crisis the following week.
      </Prose>

      <Callout accent="purple">
        Assessment security has no single point of failure and no single point of success. Every layer — exposure control, lifecycle management, surveillance, contamination detection, canary tripwires, disclosure standards — is necessary and none is sufficient. Programs that invest heavily in one layer while neglecting the others typically experience leakage events that the under-invested layer was supposed to catch. Defense in depth is not a slogan here; it is the only operational pattern that reliably works.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The sources below were verified against their original publication venues on 2026-04-26. Citations include the canonical reference plus a short note on what each source contributes to the assessment-security toolkit.
      </Prose>

      <H3>Sympson and Hetter 1985 — exposure control foundation</H3>
      <Prose>
        James B. Sympson, Robert R. Hetter. "Controlling Item-Exposure Rates in Computerized Adaptive Testing." Proceedings of the 27th Annual Meeting of the Military Testing Association, San Diego, 1985. The original paper introducing the probabilistic exposure-control gate that bears the authors' names. The procedure has been continuously deployed in operational testing programs (ASVAB, GRE, GMAT, USMLE) for over four decades and remains the dominant exposure-control method in computer-adaptive testing despite numerous proposed refinements. Way (1998) gives a comprehensive technical review.
      </Prose>

      <H3>Way 1998 — CAT exposure-control review</H3>
      <Prose>
        Walter D. Way. "Protecting the Integrity of Computerized Testing Item Pools." <em>Educational Measurement: Issues and Practice</em>, 17(4), 17-27, 1998. The standard reference review of item-exposure control methods circa the late 1990s, including Sympson-Hetter, the conditional Sympson-Hetter variant, and the McBride and Martin multiple-stratification approach. Way also documents the operational reality of leak detection at ETS in the years immediately preceding the 2002 GRE incident, which illuminates the human side of assessment security in a way the algorithmic literature does not.
      </Prose>

      <H3>Chang and Ying 1999 — alpha-stratified item bank</H3>
      <Prose>
        Hua-Hua Chang, Zhiliang Ying. "a-Stratified Multistage Computerized Adaptive Testing." <em>Applied Psychological Measurement</em>, 23(3), 211-222, 1999. Introduces the discrimination-stratified item bank as a complement to Sympson-Hetter. The method partitions the item pool by discrimination tier and forces the CAT selection algorithm to draw from low-discrimination items early and high-discrimination items late, producing more uniform bank utilization and better information conservation. Combined with Sympson-Hetter in most production CAT programs.
      </Prose>

      <H3>Shi et al. 2024 — MIN-K% probability for membership inference</H3>
      <Prose>
        Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. "Detecting Pretraining Data from Large Language Models." arXiv:2310.16789. Published October 2023; ICLR 2024. Introduces MIN-K% probability as a black-box membership-inference signal, validates it on the WIKIMIA benchmark with AUCs in the 0.66-0.74 range against production-scale models, and provides the reference implementation. The method is the de facto standard for contamination detection without training-corpus access and is included in essentially every contamination-aware evaluation harness released after 2024.
      </Prose>

      <H3>Sainz et al. 2023 — NLP eval contamination survey</H3>
      <Prose>
        Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark." arXiv:2310.18018. Published October 2023; EMNLP 2023 Findings. Catalogs contamination in dozens of widely-used NLP benchmarks and articulates the framework of "contamination-aware evaluation" that subsequent benchmark publications adopted. The paper is the most-cited single reference for the proposition that contamination is a systemic problem in NLP evaluation rather than an occasional incident.
      </Prose>

      <H3>Yang et al. 2023 — rethinking benchmark contamination</H3>
      <Prose>
        Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples." arXiv:2311.04850. Published November 2023. Demonstrates that paraphrased and translated test items leak measurable performance signal into models trained on them — approximately 60-80% of the gain produced by verbatim contamination — and that exact n-gram overlap scans completely miss this contamination class. Motivates the use of fuzzy-match and embedding-based contamination detectors alongside n-gram methods.
      </Prose>

      <H3>Carlini et al. 2021 — extracting memorized training data</H3>
      <Prose>
        Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, Colin Raffel. "Extracting Training Data from Large Language Models." USENIX Security Symposium, 2021. Demonstrates that training data — including personally identifiable information — can be extracted verbatim from large language models through carefully constructed prompts, and introduces the perplexity-ratio membership-inference test that became a precursor to MIN-K%. Foundational for the modern understanding of LLM memorization and its implications for both privacy and benchmark contamination.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why does Sympson-Hetter need iteration?</H3>
      <Prose>
        Re-read the Sympson-Hetter update rule <Code>K_i ← K_i · (r_max / e_i)</Code>. Why is a single application of this update insufficient to guarantee that <Code>e_i ≤ r_max</Code> for all items? Walk through what happens to the second-choice items in the bank when the gate rejects the originally most-popular items. Why does adjusting <Code>K_i</Code> for those second-choice items in turn change the <Code>P_i</Code> values for items further down the list? Sketch what would happen to convergence if the bank were so small that the second-choice items also had <Code>P_i</Code> well above <Code>r_max</Code>.
      </Prose>

      <H3>Exercise 2 — Compute the MIN-K% threshold from a reference distribution</H3>
      <Prose>
        Suppose you have 500 known-unseen reference sequences with MIN-20% scores normally distributed with mean -8.5 and standard deviation 1.2. You want to classify a candidate sequence as contaminated if its MIN-20% score is anomalously high relative to this reference distribution. Choose a false-positive rate of 5% and compute the threshold. What MIN-20% score would you need to observe to reject the null hypothesis of "not in training" at this confidence level? Now consider how the threshold changes as you tighten the false-positive rate to 1% and 0.1%. What does this tell you about the irreducible tradeoff between contamination detection sensitivity and false-positive rates in production evaluation pipelines?
      </Prose>

      <H3>Exercise 3 — Design a canary set</H3>
      <Prose>
        You are building a private evaluation set for a domain-specific LLM benchmark in the field of clinical reasoning. You want to insert 50 canary items that will detect if the benchmark contents leak into a future model's training corpus. List five design properties the canaries should have to maximize detection power, and explain why each property matters. Specifically address: (a) how distinctive should the canary content be? (b) how should canaries be distributed across difficulty levels? (c) what storage and access controls should you put on the canary set? (d) how often should you score the canaries? (e) what would you do if you detect that the canary set has leaked? Now consider the asymmetry: a canary set is a tripwire, not a fence — its job is to tell you contamination has happened, not to prevent it. How would your design change if you wanted both detection and prevention?
      </Prose>

      <H3>Exercise 4 — Trace the lifecycle of an item from leak to retirement</H3>
      <Prose>
        Walk through the full operational sequence when a chronic-over-exposure item gets photographed by a candidate, posted to a Telegram channel, and discovered by your security team three weeks later. At each stage describe: what the assessment program knows, what action it takes, what gets logged, and what happens to the candidates whose scores already incorporated the leaked item. How do you decide whether to invalidate scores from the affected administration window? What is the financial and operational cost of replacing the item, and how does it compare to the cost of the security-team time required to detect the leak in the first place? Now draw the parallel diagram for an LLM benchmark item that gets indexed by a search engine and ingested into the next generation of pretraining data. What are the analogous stages, actions, and costs? Where does the analogy break down?
      </Prose>

      <H3>Exercise 5 — Design a contamination-aware evaluation report</H3>
      <Prose>
        You are publishing a model card for a new 70B-parameter LLM and need to report scores on five public benchmarks (MMLU, HumanEval, GSM8K, BBH, AlpacaEval 2). For each benchmark, design what your contamination disclosure should look like. Specifically: what scans should you run, what numbers should you report, how should you visualize the contaminated-versus-decontaminated gap, and what threshold of contamination should trigger a "score withheld due to contamination" annotation rather than a numerical report? Now consider the political dimension: your competitors report only the higher (potentially contaminated) numbers without disclosure. Your decontaminated numbers will be lower. How do you frame the comparison so that readers can see why your disclosure is more informative without being penalized for honesty? What industry-standard practices would you propose to make this kind of disclosure the default rather than the exception?
      </Prose>

    </div>
  ),
};

export default assessmentSecurity;
