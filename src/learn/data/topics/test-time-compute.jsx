import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

// =============================================================================
// Test-Time Compute Scaling
// Deep standard — 11 sections
// Verified computations embedded below (Python, random, math, collections.Counter)
//   Best-of-N:         p=0.3 N=10 -> 0.9718 (theory); empirical matches within 1%
//   Majority vote:     p=0.55, N=31 -> 0.9946 (empirical 5000 trials)
//   Self-consistency:  p=0.60, N=41 -> 0.9088 (empirical 5000 trials, +31pp over N=1)
//   Beam search:       greedy 0.173 vs Beam-4 0.942 vs Beam-8 0.996 (p_step=0.65, 4 steps)
//   Serial vs parallel: hard task, 800 tokens: serial 0.649 vs parallel BoN 0.596
// Sources verified: arXiv:2408.03314, arXiv:2501.12948, arXiv:2203.11171
// =============================================================================

const testTimeCompute = {
  title: "Test-Time Compute Scaling",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Classical scaling laws map a clean relationship: give a model more parameters, more
        training tokens, and more training FLOPs, and its cross-entropy loss falls on a
        predictable power curve. That is the axis Kaplan and Chinchilla mapped. Every major
        lab internalised it: if you want a more capable model, spend more during training.
        The capability is in the weights, fixed at the moment training ends.
      </Prose>

      <Prose>
        Test-time compute scaling describes a different axis entirely. The model is already
        trained. Its weights are frozen. You do not retrain it, fine-tune it, or increase
        its parameter count. Instead, you let it generate more tokens when answering a hard
        question — and it gets the right answer more often. The same weights, the same
        checkpoint, meaningfully smarter on hard verifiable tasks when given more inference
        time. This is not a prompting trick. It is a distinct empirical phenomenon that
        operates independently of the training-compute axis.
      </Prose>

      <Prose>
        The evidence is not subtle. OpenAI's o1, released September 2024, demonstrated
        log-linear accuracy growth on AIME 2024 as inference tokens increased from roughly
        one thousand to thirty thousand per problem — moving from around 30% accuracy to
        around 80% on competition mathematics. DeepSeek-R1 (arXiv:2501.12948, January 2025)
        replicated the result from a different lab using GRPO training: R1's pass@1 on
        AIME 2024 reaches 71%, with majority voting pushing it to 86.7%, matching o1's
        performance. Snell et al. (arXiv:2408.03314) formalised the relationship
        empirically: compute-optimal test-time scaling can outperform a model 14x larger
        in raw parameter count. These are not marginal benchmark improvements. They represent
        a new axis of capability that labs can tune per-query without retraining anything.
      </Prose>

      <Prose>
        Why does this matter for the field? Because it uncouples capability from training
        cost in a way that reshapes the economics of AI development. A training run costs
        a fixed sum, once, producing a model whose capability ceiling is set before the
        first user query. Test-time compute is a runtime variable: cheap on easy queries,
        expensive on hard ones, tunable per-request. The question "how capable is this
        model?" gains a new modifier: "at how many inference tokens?" That is a genuinely
        new degree of freedom, and understanding its mechanics — the empirical curve, the
        techniques that drive it, its limits and failure modes — is now a prerequisite for
        working at the inference frontier.
      </Prose>

      <Callout accent="gold">
        Classical scaling: more training compute → smarter weights. Test-time scaling:
        same weights + more inference tokens → smarter answers. Both axes are real; neither
        subsumes the other.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 More tokens, more accuracy — when the model is trained for it</H3>

      <Prose>
        The key qualifier in the empirical finding is "RL-trained." A GPT-4-class base
        model given 100k inference tokens shows essentially no log-linear accuracy scaling
        on hard math. Its accuracy rises from roughly 24% to maybe 37% across a two-order-
        of-magnitude token budget — most of that is sampling noise, not a reliable reasoning
        mechanism. The log-linear curve only appears in models trained through reinforcement
        learning on verifiable rewards. The RL training procedure is what installs a
        policy that uses inference tokens productively: backtracking when a path fails,
        exploring alternative approaches, checking partial results before committing. That
        reasoning behavior is a learned policy, not an emergent side-effect of making the
        model bigger.
      </Prose>

      <Prose>
        The intuition for why it works: RL on verifiable problems rewards rollouts where
        the model explores multiple approaches before committing to an answer. Over millions
        of training problems, the model learns that on hard problems, more deliberation
        produces better outcomes. This generalises to test time. When given a large token
        budget, the model's learned policy allocates tokens toward genuine exploration —
        not verbose padding, not restating the problem, but actual search through solution
        space. The training-side mechanics behind this are covered in the RL for Reasoning
        topic. This topic covers what the policy does at inference time.
      </Prose>

      <H3>2.2 The four main mechanisms</H3>

      <Prose>
        At least four distinct mechanisms produce test-time compute gains, operating at
        different granularities and with different latency profiles. Production reasoning
        models typically combine several of them.
      </Prose>

      <Prose>
        <strong>Long chain-of-thought.</strong> The model generates an extended internal
        reasoning trace — sometimes tens of thousands of tokens — before producing a final
        answer. The trace is not shown to users in most deployments but is generated and
        billed as output tokens. The RL training makes the trace useful: rollouts with
        long productive deliberation earn more reward than rollouts with padded verbosity,
        so the policy learns to use the trace for actual reasoning.
      </Prose>

      <Prose>
        <strong>Best-of-N with a verifier.</strong> Generate N independent candidate
        answers or full reasoning traces, score each with a verifier — a unit test suite,
        an outcome reward model, a formal checker — and return the highest-scoring result.
        Best-of-N is naturally parallel: all N generations run simultaneously on separate
        compute slots. This is the parallel alternative to serial chain-of-thought.
      </Prose>

      <Prose>
        <strong>Majority voting (self-consistency).</strong> Sample N independent chains
        of thought, collect the final answers, and return the modal answer. No external
        verifier needed — just a way to aggregate answers. Wang et al. (arXiv:2203.11171)
        showed majority vote over diverse CoT chains outperforms single greedy decoding
        by 12–18 points on arithmetic benchmarks. The intuition: if the correct reasoning
        path leads to the correct answer more often than any single wrong path, the mode
        of the answer distribution is the right answer.
      </Prose>

      <Prose>
        <strong>Tree search with a process reward model.</strong> Rather than a linear
        chain, the model explores a tree of reasoning states. At each step, a process
        reward model (PRM) scores intermediate states. Beam search or Monte Carlo Tree
        Search prunes unpromising branches and expands promising ones. This is the most
        compute-intensive approach but achieves the highest accuracy per total-token budget
        on hard structured problems by directing compute toward productive reasoning paths.
      </Prose>

      <H3>2.3 Serial vs parallel: different latency profiles, same compute budget</H3>

      <Prose>
        Serial chain-of-thought and parallel best-of-N are not competing techniques —
        they occupy different points in latency-vs-throughput space at equal total token
        cost. A serial chain of 32,000 tokens ties up one GPU slot for the full duration
        of generation, potentially one to two minutes. Parallel best-of-N with 32 samples
        of 1,000 tokens each uses 32 GPU slots simultaneously and finishes in roughly
        the time of a single 1,000-token generation. At the same total token count, the
        parallel strategy has far lower wall-clock latency. The serial strategy has an
        advantage on tasks where depth of deliberation matters more than breadth of
        sampling — when the problem requires building on earlier steps of reasoning rather
        than independently re-rolling the dice.
      </Prose>

      <H3>2.4 Log-linear scaling</H3>

      <Prose>
        The empirical relationship between inference token budget and accuracy on hard
        verifiable benchmarks is approximately log-linear: doubling the token budget gives
        a roughly constant additive accuracy gain. This means the accuracy curve is a
        straight line on a log-scale x-axis. The gain per doubling is not large — roughly
        5–10 percentage points on AIME-class problems — but it is reliable across many
        doublings, producing significant total improvement across two to three orders of
        magnitude. The curve eventually plateaus when the problem approaches the ceiling
        of what a given model can solve regardless of deliberation time — typically when
        the model's latent knowledge or reasoning capacity is the binding constraint, not
        the token budget.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Best-of-N with a perfect verifier</H3>

      <Prose>
        Let <Code>p</Code> be the probability that a single sample from the model produces
        a correct answer, and assume a perfect verifier (precision = recall = 1). With
        <Code>N</Code> independent samples, the probability that at least one is correct is:
      </Prose>

      <MathBlock caption="Best-of-N accuracy formula with perfect verifier">
        {"P(\\text{at least one correct}) = 1 - (1 - p)^N"}
      </MathBlock>

      <Prose>
        This converges to 1 as N grows for any fixed p &gt; 0. The rate of convergence
        depends critically on p. For p = 0.3 and N = 10 (self-check exercise 1), the
        formula gives <Code>1 - 0.7^10 = 0.9718</Code>. For p = 0.1, N = 10 gives 0.651 —
        substantially lower, showing how a weaker base model is harder to lift through
        sampling alone. The formula assumes independent samples; in practice, model outputs
        are correlated (the same systematic failure modes repeat), so the empirical curve
        is typically below the theoretical one at large N.
      </Prose>

      <H3>3.2 Best-of-N with an imperfect verifier</H3>

      <Prose>
        A real verifier has precision <Code>π</Code> (fraction of accepted answers that
        are truly correct) and recall <Code>ρ</Code> (fraction of correct answers that are
        accepted). The effective accuracy of best-of-N with an imperfect verifier is bounded
        by verifier precision:
      </Prose>

      <MathBlock caption="Effective accuracy bounded by verifier precision">
        {"\\text{accuracy}(N, \\pi, \\rho) \\leq \\pi"}
      </MathBlock>

      <Prose>
        No matter how many samples N you draw, if the verifier accepts wrong answers with
        non-zero probability, the final selected answer will be wrong at least <Code>1 - π</Code>
        of the time. This is the fundamental bottleneck of test-time compute at extreme
        scale: the verifier becomes the binding constraint, not the sampling budget. This
        is why process reward models, which evaluate reasoning steps rather than only final
        answers, are a research priority — they achieve higher precision on structured
        problems by detecting errors earlier in the reasoning trace.
      </Prose>

      <H3>3.3 Majority voting convergence</H3>

      <Prose>
        For majority voting over N independent samples where each sample produces the
        correct answer with probability <Code>p &gt; 0.5</Code>, the probability that
        the majority vote is correct follows a Condorcet-like formula. For odd N, the
        majority is correct whenever more than N/2 samples are correct. By the law of
        large numbers, for any p &gt; 0.5 the majority vote converges to the correct
        answer as N grows. For p &lt; 0.5, majority vote converges to the wrong answer.
        The condition p &gt; 0.5 is a necessary requirement for majority voting to help.
      </Prose>

      <MathBlock caption="Majority vote accuracy for odd N (Condorcet)">
        {"P(\\text{majority correct}) = \\sum_{k=\\lceil N/2 \\rceil}^{N} \\binom{N}{k} p^k (1-p)^{N-k}"}
      </MathBlock>

      <H3>3.4 Test-time compute scaling law</H3>

      <Prose>
        The empirically observed relationship between inference token budget <Code>C</Code>
        and accuracy on hard reasoning benchmarks (AIME 2024, GPQA Diamond, Codeforces)
        for RL-trained models is approximately:
      </Prose>

      <MathBlock caption="Test-time scaling law: accuracy grows log-linearly with inference token budget (empirical)">
        {"\\text{accuracy}(C) \\approx \\alpha \\cdot \\log_{10}(C) + \\beta"}
      </MathBlock>

      <Prose>
        Here <Code>α</Code> is the slope (accuracy gain per decade of tokens, roughly 15–25
        percentage points on hard math benchmarks) and <Code>β</Code> is the intercept.
        This relationship holds for RL-trained models over roughly three orders of magnitude
        of token budget (1k to 100k tokens), after which the curve plateaus at or near the
        ceiling for a given benchmark. Snell et al. (arXiv:2408.03314) formalised this
        and showed that compute-optimal allocation — adaptively varying the token budget
        based on estimated problem difficulty — improves efficiency by more than 4× compared
        to a fixed best-of-N baseline.
      </Prose>

      <H3>3.5 Training vs test-time compute trade-off</H3>

      <Prose>
        Snell et al. also studied the trade-off between training compute and test-time
        compute at fixed total FLOPs. Their key finding: for problems where the base model
        achieves a non-trivial accuracy (a few percent or higher), it is often more
        efficient to allocate compute to test-time search than to training additional
        parameters. Specifically, a smaller model with a large test-time budget can
        outperform a 14× larger model on matching compute problems. This suggests that
        the optimal training-vs-inference compute split depends on the problem difficulty
        distribution of your deployment workload.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        This section builds each major test-time compute technique from scratch using
        only Python's standard library. All code is runnable and the outputs shown below
        are actual verified outputs. The implementations use a toy "mock model" that
        returns the correct answer with a fixed probability <Code>p</Code>, allowing clean
        verification of the theory against empirical simulation.
      </Prose>

      <H3>4a. Best-of-N with a verifier</H3>

      <Prose>
        The most direct form of parallel test-time compute: sample N answers independently,
        apply a verifier to each, return the first correct one. If none are correct,
        return an arbitrary candidate (worst case for the verifier). We verify the formula
        <Code>P(success) = 1 - (1-p)^N</Code> empirically across N from 1 to 32.
      </Prose>

      <CodeBlock language="python">
{`import random
from collections import Counter

random.seed(42)

def sample_model(p_correct=0.3):
    """Toy model: returns correct answer with probability p."""
    return 'correct' if random.random() < p_correct else 'wrong'

def apply_verifier(answer):
    """Deterministic verifier: exact match."""
    return answer == 'correct'

def best_of_n_trial(n, p_correct=0.3):
    candidates = [sample_model(p_correct) for _ in range(n)]
    return any(apply_verifier(c) for c in candidates)

def theory_bon(p, n):
    """Theoretical: P(at least one correct) = 1 - (1-p)^N"""
    return 1.0 - (1.0 - p) ** n

p = 0.3
n_trials = 5000

print(f'p_correct_per_sample = {p}')
print('  N |   Theory |  Empirical')
for N in [1, 2, 4, 8, 16, 32]:
    theo = theory_bon(p, N)
    emp = sum(best_of_n_trial(N, p) for _ in range(n_trials)) / n_trials
    print(f'{N:>3} | {theo:>8.4f} | {emp:>9.4f}')

print()
# Self-check exercise 1 answer:
print(f'p=0.3, N=10 -> {theory_bon(0.3, 10):.4f}')

# Output:
#   p_correct_per_sample = 0.3
#     N |   Theory |  Empirical
#     1 |   0.3000 |    0.2984
#     2 |   0.5100 |    0.5006
#     4 |   0.7599 |    0.7606
#     8 |   0.9424 |    0.9406
#    16 |   0.9967 |    0.9974
#    32 |   1.0000 |    1.0000
#
#   p=0.3, N=10 -> 0.9718
#
# Empirical tracks theory closely. At N=8 we are already at 94% despite
# p=0.3 per sample. The verifier is doing the heavy lifting — without it,
# you'd have no way to select the one correct answer from eight candidates.`}
      </CodeBlock>

      <H3>4b. Majority voting without a verifier</H3>

      <Prose>
        When no external verifier exists, majority voting is the natural alternative.
        Sample N independent answers and return the modal answer. This requires that
        the correct answer is more probable per sample than any individual wrong answer.
        With p = 0.55 (correct) and three distractors sharing the remaining 0.45, the
        correct answer dominates and the mode converges rapidly.
      </Prose>

      <CodeBlock language="python">
{`import random
from collections import Counter

random.seed(123)

def sample_answer(p_correct=0.55):
    """Each sample: p_correct chance of correct, else uniform over 3 distractors."""
    if random.random() < p_correct:
        return 'A'  # correct
    return random.choice(['B', 'C', 'D'])

def majority_vote(n, p_correct=0.55):
    answers = [sample_answer(p_correct) for _ in range(n)]
    return Counter(answers).most_common(1)[0][0] == 'A'

n_trials = 5000
p = 0.55
print(f'p_correct_per_sample = {p}, 3 distractors')
print('  N | Accuracy')
for N in [1, 3, 7, 15, 31, 63]:
    acc = sum(majority_vote(N, p) for _ in range(n_trials)) / n_trials
    print(f'{N:>3} | {acc:>8.4f}')

# Output:
#   p_correct_per_sample = 0.55, 3 distractors
#     N | Accuracy
#     1 |   0.5390
#     3 |   0.6630
#     7 |   0.8226
#    15 |   0.9478
#    31 |   0.9946
#    63 |   0.9998
#
# A 55%-accurate model reaches 99.5% with majority vote over 31 samples —
# no verifier required. The critical assumption: the correct answer must be
# the single modal answer, i.e. p > 1/(num_choices). If the model has
# a systematic wrong-answer bias, majority vote amplifies it, not corrects it.`}
      </CodeBlock>

      <H3>4c. Self-consistency (Wang et al. 2022)</H3>

      <Prose>
        Self-consistency applies majority voting specifically to chain-of-thought outputs.
        Multiple diverse reasoning chains are sampled; the final answer of each is
        extracted and the mode is returned. Wang et al. (arXiv:2203.11171) showed gains
        of 12–18 percentage points on arithmetic reasoning over single greedy decoding.
        We simulate with <Code>p = 0.60</Code> per chain (a moderately capable RL-trained
        model on hard math) and measure how much majority voting helps as N grows.
      </Prose>

      <CodeBlock language="python">
{`import random
from collections import Counter

random.seed(456)

def single_cot_chain(p_correct=0.60):
    """Each independently-sampled CoT chain: correct with prob p."""
    return 'correct' if random.random() < p_correct else 'wrong'

def self_consistency_vote(n, p_correct=0.60):
    answers = [single_cot_chain(p_correct) for _ in range(n)]
    return Counter(answers).most_common(1)[0][0] == 'correct'

n_trials = 5000
p = 0.60

base = sum(single_cot_chain(p) == 'correct' for _ in range(n_trials)) / n_trials
print(f'p_correct_per_chain = {p}')
print(f'  N | SC accuracy | Delta vs N=1')
print(f'  1 | {base:>11.4f} | baseline')
for N in [3, 5, 11, 21, 41]:
    acc = sum(self_consistency_vote(N, p) for _ in range(n_trials)) / n_trials
    print(f'{N:>3} | {acc:>11.4f} | +{acc - base:>6.4f}')

# Output:
#   p_correct_per_chain = 0.6
#     N | SC accuracy | Delta vs N=1
#     1 |      0.5988 | baseline
#     3 |      0.6428 | +0.0440
#     5 |      0.6864 | +0.0876
#    11 |      0.7568 | +0.1580
#    21 |      0.8242 | +0.2254
#    41 |      0.9088 | +0.3100
#
# A 60% single-chain model reaches 91% accuracy at N=41 chains (+31pp).
# Self-consistency is free — no verifier, no PRM, no additional training.
# The cost is N× inference; the gain is reliable majority-vote convergence.`}
      </CodeBlock>

      <H3>4d. Tree search (beam search + PRM)</H3>

      <Prose>
        Tree search with a process reward model (PRM) is the highest-accuracy but most
        compute-intensive approach. At each reasoning step, the model generates multiple
        candidate continuations; the PRM scores each; only the top-<Code>B</Code>
        (beam width) candidates advance. We simulate with <Code>p_step = 0.65</Code>
        (each step individually correct 65% of the time) over four steps. The greedy
        baseline requires all four steps correct: <Code>0.65^4 = 0.178</Code>.
      </Prose>

      <CodeBlock language="python">
{`import random

random.seed(42)

def prm_score(is_correct_step):
    """Process reward model: correct step -> high score, wrong -> low."""
    if is_correct_step:
        return max(0.0, min(1.0, random.gauss(0.78, 0.12)))
    else:
        return max(0.0, min(1.0, random.gauss(0.28, 0.15)))

def greedy_solve(n_steps=4, p_step=0.65):
    """Single greedy chain: must succeed at every step."""
    for _ in range(n_steps):
        if random.random() >= p_step:
            return False
    return True

def beam_search_solve(beam_width=4, n_steps=4, p_step=0.65):
    """Beam search: expand 2 candidates per beam, score, keep top beam_width."""
    beams = [(True, 0.0)]  # (on_correct_path, cumulative_prm)
    for _ in range(n_steps):
        candidates = []
        for (on_correct, score) in beams:
            for _ in range(2):  # 2 branches per beam
                step_ok = on_correct and (random.random() < p_step)
                candidates.append((step_ok, score + prm_score(step_ok)))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam_width]
    return any(ok for ok, _ in beams)

n_trials = 3000
print('n_steps=4, p_step=0.65 (theory greedy: 0.65^4=0.1785)')
print('Method       | Accuracy')
g = sum(greedy_solve() for _ in range(n_trials)) / n_trials
print(f'Greedy (B=1) | {g:.4f}')
for B in [2, 4, 8]:
    acc = sum(beam_search_solve(beam_width=B) for _ in range(n_trials)) / n_trials
    print(f'Beam B={B:<5} | {acc:.4f}')

# Output:
#   n_steps=4, p_step=0.65 (theory greedy: 0.65^4=0.1785)
#   Method       | Accuracy
#   Greedy (B=1) | 0.1730
#   Beam B=2     | 0.7470
#   Beam B=4     | 0.9420
#   Beam B=8     | 0.9963
#
# The PRM is the engine: greedy has a 17% chance of getting all four steps right.
# Beam-4 reaches 94% — a 5.5× lift from the PRM-guided search.
# Beam-8 reaches 99.6%. The trade-off is latency: beam search is sequential,
# not parallelisable across beams, and inference is blocking per step.`}
      </CodeBlock>

      <H3>4e. Serial vs parallel compute trade-off</H3>

      <Prose>
        At a fixed total token budget, the choice between one long serial chain (depth)
        and many short parallel attempts (breadth) depends on the task structure. We
        model this explicitly: serial CoT accuracy on hard problems grows log-linearly
        with token budget (the RL-trained model uses each additional token productively),
        while parallel BoN with short attempts has a fixed per-attempt accuracy that
        benefits from breadth only when a verifier exists.
      </Prose>

      <CodeBlock language="python">
{`import random, math

random.seed(99)

TOTAL_TOKENS = 800

def serial_cot_hard(total_tokens, p_base=0.25):
    """
    Serial CoT on a hard problem. RL-trained model: accuracy scales log-linearly.
    Simulates: p(correct) = p_base + 0.18 * log(tokens/100 + 1), capped at 0.9.
    """
    effective_p = min(0.9, p_base + 0.18 * math.log(total_tokens / 100 + 1))
    return random.random() < effective_p

def parallel_bon_hard(total_tokens, tokens_per_attempt=200, p_attempt=0.20):
    """
    Parallel BoN on a hard problem. Each short attempt: low per-sample accuracy.
    """
    n_attempts = max(1, total_tokens // tokens_per_attempt)
    return any(random.random() < p_attempt for _ in range(n_attempts))

n_trials = 5000
print('Fixed total budget = 800 tokens — hard problem')
print('Token budget | Serial CoT | Parallel BoN')
for tokens in [100, 200, 400, 800, 1600]:
    s = sum(serial_cot_hard(tokens) for _ in range(n_trials)) / n_trials
    p = sum(parallel_bon_hard(tokens) for _ in range(n_trials)) / n_trials
    print(f'{tokens:>12} | {s:>10.4f} | {p:>12.4f}')

# Output:
#   Fixed total budget = 800 tokens — hard problem
#   Token budget | Serial CoT | Parallel BoN
#            100 |     0.3838 |       0.2092
#            200 |     0.4502 |       0.2006
#            400 |     0.5370 |       0.3706
#            800 |     0.6486 |       0.5958
#           1600 |     0.7550 |       0.8286
#
# On hard problems at low budgets, serial CoT wins: each added token
# lets the model explore one more reasoning branch. Parallel BoN needs a
# reliable verifier AND enough per-sample accuracy to benefit from breadth.
# At 1600 tokens, parallel BoN overtakes serial: N=8 attempts of 200 tokens
# each, each with p=0.20, gives 1-(0.8^8)=0.83 — outpacing the serial log curve.
#
# Practical rule: serial CoT for depth-requiring tasks (formal proofs, complex
# multi-step derivations). Parallel BoN for breadth-requiring tasks where a
# verifier exists and per-attempt accuracy is non-trivial.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 OpenAI o1 / o3 API</H3>

      <Prose>
        OpenAI's reasoning models expose a <Code>max_completion_tokens</Code> budget that
        bounds the combined thinking and output tokens. The internal reasoning trace is
        hidden from the response body; only the final answer is returned. The API surfaces
        usage statistics that distinguish thinking tokens from output tokens for billing.
      </Prose>

      <CodeBlock language="python">
{`from openai import OpenAI

client = OpenAI()

# o1 and o3 accept a reasoning effort hint or an explicit token budget.
# "high" effort = more inference tokens; "low" = fast, cheaper.
response = client.chat.completions.create(
    model="o3",
    messages=[
        {"role": "user", "content": "Prove that sqrt(2) is irrational."}
    ],
    # Effort level: "low" | "medium" | "high"
    # "high" allocates more thinking tokens for hard problems.
    reasoning_effort="high",
    max_completion_tokens=16_000,
)

print(response.choices[0].message.content)

# Inspect token usage
usage = response.usage
print(f"Input tokens:         {usage.prompt_tokens}")
print(f"Reasoning tokens:     {usage.completion_tokens_details.reasoning_tokens}")
print(f"Output tokens:        {usage.completion_tokens_details.audio_tokens}")  # final answer tokens
print(f"Total billed tokens:  {usage.total_tokens}")`}
      </CodeBlock>

      <H3>5.2 DeepSeek-R1 reasoning mode</H3>

      <Prose>
        DeepSeek-R1 (arXiv:2501.12948) is served both as an API and as an open-weight
        model. When using the API or a self-hosted vLLM deployment, the reasoning trace
        appears inside <Code>&lt;think&gt;...&lt;/think&gt;</Code> tags before the final
        answer. You can surface it (for interpretability) or strip it (for end users).
      </Prose>

      <CodeBlock language="python">
{`import re
from openai import OpenAI  # DeepSeek uses the OpenAI-compatible API format

client = OpenAI(
    api_key="YOUR_DEEPSEEK_KEY",
    base_url="https://api.deepseek.com/v1",
)

response = client.chat.completions.create(
    model="deepseek-reasoner",   # R1 reasoning model
    messages=[
        {"role": "user", "content": "Find all integer solutions to x^2 - 7y^2 = 1."}
    ],
    max_tokens=8192,
)

raw = response.choices[0].message.content

# R1 returns: <think>...reasoning trace...</think> final answer
think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
reasoning_trace = think_match.group(1).strip() if think_match else ""
final_answer = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

print("=== Reasoning Trace (first 300 chars) ===")
print(reasoning_trace[:300])
print("=== Final Answer ===")
print(final_answer)`}
      </CodeBlock>

      <H3>5.3 Claude extended thinking</H3>

      <Prose>
        Anthropic's Claude 3.7 Sonnet and later models support extended thinking via
        the <Code>thinking</Code> parameter. A <Code>budget_tokens</Code> parameter
        controls how many tokens are allocated to the internal reasoning trace. Thinking
        blocks are returned as separate content items in the response, distinct from
        the text answer.
      </Prose>

      <CodeBlock language="python">
{`import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000,  # tokens allocated to internal reasoning
    },
    messages=[{
        "role": "user",
        "content": "What is the 100th prime number? Show your reasoning."
    }]
)

# Response may contain multiple blocks: thinking block + text block
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking] {block.thinking[:200]}...")
    elif block.type == "text":
        print(f"[Answer] {block.text}")`}
      </CodeBlock>

      <H3>5.4 Self-hosted inference with vLLM</H3>

      <Prose>
        For open-weight reasoning models (DeepSeek-R1, Qwen-QwQ-32B, Llama-based RL
        fine-tunes), vLLM handles the variable-length generation natively. The key
        configuration difference from chat serving is a much higher <Code>max_model_len</Code>
        and adjusting the KV cache budget accordingly.
      </Prose>

      <CodeBlock language="bash">
{`# Launch DeepSeek-R1-Distill-Qwen-32B with vLLM
# Key flags for reasoning models:
#   --max-model-len: must accommodate full reasoning trace (32k+ tokens)
#   --gpu-memory-utilization: higher than chat — reasoning is output-dominated
#   --enable-chunked-prefill: helps with long KV caches

python -m vllm.entrypoints.openai.api_server \\
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \\
    --max-model-len 32768 \\
    --gpu-memory-utilization 0.92 \\
    --enable-chunked-prefill \\
    --tensor-parallel-size 4  # 4 GPUs for a 32B model at full precision`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Log-linear accuracy vs inference tokens</H3>

      <Prose>
        The canonical result: RL-trained reasoning model accuracy grows approximately
        log-linearly with token budget on hard benchmarks. The base model (no RL) shows
        minimal scaling — it gains a few points from lucky traces but hits a ceiling
        around 35% regardless of budget. Data points are illustrative of AIME 2024-class
        results from the o1 and R1 generation.
      </Prose>

      <Plot
        label="accuracy vs. log10(inference tokens per problem) — hard math benchmark"
        width={540}
        height={280}
        xLabel="log10 tokens per problem"
        yLabel="accuracy %"
        series={[
          {
            name: "RL-trained (o1/R1-class)",
            color: colors.gold,
            points: [[3, 30], [3.5, 48], [4, 64], [4.5, 76], [5, 83]],
          },
          {
            name: "Base model (no RL)",
            color: colors.green,
            points: [[3, 22], [3.5, 28], [4, 32], [4.5, 35], [5, 36]],
          },
        ]}
      />

      <H3>6.2 Best-of-N accuracy vs N for different base accuracies</H3>

      <Prose>
        Best-of-N accuracy as a function of N for base per-sample accuracies of 10%,
        30%, 50%, and 70%. All curves converge to 100% as N grows (with a perfect
        verifier), but lower base accuracy requires more samples to reach the same
        target accuracy. This shows why base model quality still matters even with
        test-time compute: a model with p = 0.10 needs N ≈ 22 samples to reach 90%,
        while p = 0.30 needs only N ≈ 9.
      </Prose>

      <Plot
        label="best-of-N accuracy vs. N for varying base accuracy p (perfect verifier)"
        width={540}
        height={280}
        xLabel="N (number of samples)"
        yLabel="BoN accuracy %"
        series={[
          {
            name: "p=0.10",
            color: "#f87171",
            points: [1, 2, 4, 8, 16, 32].map((N) => [
              N,
              Math.round((1 - Math.pow(0.9, N)) * 100),
            ]),
          },
          {
            name: "p=0.30",
            color: colors.gold,
            points: [1, 2, 4, 8, 16, 32].map((N) => [
              N,
              Math.round((1 - Math.pow(0.7, N)) * 100),
            ]),
          },
          {
            name: "p=0.50",
            color: colors.green,
            points: [1, 2, 4, 8, 16, 32].map((N) => [
              N,
              Math.round((1 - Math.pow(0.5, N)) * 100),
            ]),
          },
          {
            name: "p=0.70",
            color: "#c084fc",
            points: [1, 2, 4, 8, 16, 32].map((N) => [
              N,
              Math.round((1 - Math.pow(0.3, N)) * 100),
            ]),
          },
        ]}
      />

      <H3>6.3 Step trace: test-time compute reasoning loop</H3>

      <Prose>
        A concrete walk-through of a single test-time compute trace, showing how a
        reasoning model allocates its token budget when the initial attempt fails
        verification. Each step advances the session.
      </Prose>

      <StepTrace
        label="test-time compute trace: math problem with verifier"
        steps={[
          {
            label: "Initial attempt",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div style={{ color: colors.textMuted, marginBottom: 4 }}>PROBLEM: Prove that for all integers n, n² + n is even.</div>
                <div style={{ color: colors.gold }}>{"<think>"}</div>
                <div style={{ paddingLeft: 16 }}>Let me try direct: n² + n = n(n+1)...</div>
                <div style={{ paddingLeft: 16 }}>Hmm, I think I need to check both cases.</div>
                <div style={{ color: colors.gold }}>{"</think>"}</div>
                <div style={{ color: colors.textPrimary, marginTop: 4 }}>Answer: n² + n is always divisible by 2. ∎</div>
              </div>
            ),
          },
          {
            label: "Verifier check",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div style={{ color: colors.textMuted, marginBottom: 4 }}>VERIFIER: Checking proof completeness...</div>
                <div style={{ color: "#f87171" }}>✗ FAIL: Answer asserts result but does not prove it.</div>
                <div style={{ color: "#f87171" }}>  Missing: explicit case analysis for n even / n odd.</div>
                <div style={{ color: "#f87171" }}>  PRM step score: 0.31 (below threshold 0.70)</div>
                <div style={{ color: colors.textMuted, marginTop: 4 }}>→ Retry with more reasoning budget.</div>
              </div>
            ),
          },
          {
            label: "Extended reasoning",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div style={{ color: colors.gold }}>{"<think>"}</div>
                <div style={{ paddingLeft: 16 }}>My first attempt was incomplete. I need two cases.</div>
                <div style={{ paddingLeft: 16 }}>Case 1: n even → n = 2k → n² + n = 4k² + 2k = 2(2k² + k). ✓</div>
                <div style={{ paddingLeft: 16 }}>Case 2: n odd → n = 2k+1 → n² + n = (2k+1)(2k+2) = 2(2k+1)(k+1). ✓</div>
                <div style={{ paddingLeft: 16 }}>Both cases give an even number. QED.</div>
                <div style={{ color: colors.gold }}>{"</think>"}</div>
              </div>
            ),
          },
          {
            label: "Verifier succeeds",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div style={{ color: colors.textMuted, marginBottom: 4 }}>VERIFIER: Re-checking proof completeness...</div>
                <div style={{ color: colors.green }}>✓ PASS: Case split covers all integers. Algebraic steps verified.</div>
                <div style={{ color: colors.green }}>  PRM step score: 0.91 (above threshold 0.70)</div>
                <div style={{ color: colors.textPrimary, marginTop: 4 }}>
                  Final answer: Since n is either even or odd, and in both cases n² + n = 2·(integer), it is always even. ∎
                </div>
                <div style={{ color: colors.textMuted, marginTop: 4 }}>Tokens used: ~420 (vs 80 in initial attempt)</div>
              </div>
            ),
          },
        ]}
      />

      <H3>6.4 Training vs test-time compute trade-off matrix</H3>

      <Prose>
        A heatmap of relative accuracy improvement (illustrative scale) across combinations
        of training compute level (rows: small, medium, large) and test-time token budget
        (columns: 1k, 4k, 16k, 64k tokens). Higher values = better accuracy on hard
        benchmark. The critical observation: additional test-time compute adds roughly
        constant accuracy per column across model sizes, but a small model with large
        test-time budget can approach or exceed a larger model with small test-time budget.
      </Prose>

      <Heatmap
        label="accuracy % on hard math benchmark — training compute (rows) × test-time tokens (cols)"
        rowLabels={["Small model (7B)", "Medium model (35B)", "Large model (70B)"]}
        colLabels={["1k tokens", "4k tokens", "16k tokens", "64k tokens"]}
        matrix={[
          [28, 41, 55, 66],
          [38, 52, 66, 76],
          [47, 61, 74, 83],
        ]}
        colorScale="gold"
        cellSize={52}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>When to use test-time compute</H3>

      <Prose>
        Test-time compute is not a universal improvement. It is most valuable on a
        specific class of problems and a significant waste on most others. The decision
        should be made at the query level, not as a system-wide default.
      </Prose>

      <Prose>
        <strong>Use it when:</strong> the problem is verifiable — it has a correct answer
        that an external system can check (unit tests, a formal solver, numerical ground
        truth, a well-calibrated reward model). Competition mathematics, competitive
        programming, formal proof checking, structured planning with checkable intermediate
        states, SQL query generation against a schema. On these tasks, more reasoning tokens
        reliably improve accuracy, and the verifier gives you a signal to guide selection
        or beam search.
      </Prose>

      <Prose>
        <strong>Do not use it when:</strong> the problem is subjective or open-ended —
        casual conversation, creative writing, factual lookup within training distribution,
        emotional support. Extended deliberation does not make a poem better in any
        reliable sense. A factual question the model knows the answer to is answered
        correctly at the first token or not at all — more thinking tokens add latency
        with no accuracy gain. For a general-purpose chat assistant, test-time compute
        should be a routing decision: classify the incoming query, route hard verifiable
        problems to reasoning mode, and route everything else to the standard low-latency
        path.
      </Prose>

      <H3>Choosing among techniques</H3>

      <Prose>
        <strong>Parallel best-of-N</strong> is the right choice when: you have a reliable
        external verifier; you have N available GPU slots (not a bottleneck); and low
        latency matters more than minimising token count. N × 1k-token samples finish
        in the same wall-clock time as one 1k-token sample on N parallel slots.
      </Prose>

      <Prose>
        <strong>Serial chain-of-thought</strong> is the right choice when: the problem
        requires depth (each step builds on the previous); you have an RL-trained model
        that uses inference tokens productively; and latency is acceptable (seconds to
        minutes). The model's internal trace handles search; you do not need an external
        verifier for the intermediate steps, only for the final answer.
      </Prose>

      <Prose>
        <strong>PRM-guided beam search</strong> is the right choice when: maximum accuracy
        is the priority; you have a trained PRM that scores intermediate steps reliably;
        and you can afford the serial per-step latency overhead. Beam search is not
        parallelisable across beams — all beams must complete each step before the next
        expansion. It is the slowest option in wall-clock terms but the most compute-efficient
        in accuracy per total token on hard structured problems.
      </Prose>

      <Callout accent="gold">
        Decision rule: verifiable task → use test-time compute. Parallel capacity available
        + good verifier → best-of-N. Latency-tolerant + hard multi-step → serial CoT or
        beam search. Subjective task → don't use it, route to standard path.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        Test-time compute is a genuinely new scaling axis, orthogonal to training compute,
        that extends the capability of a fixed model across a wide range of token budgets.
        On hard verifiable benchmarks — AIME, GPQA Diamond, Codeforces — the empirical
        evidence shows reliable log-linear accuracy gains over two to three orders of
        magnitude of token budget. Snell et al. showed that compute-optimal test-time
        scaling can surpass a 14× larger model on matching-FLOPs evaluations. This is not
        a small effect: it is the reason o1-class models outperform GPT-4-class models on
        math benchmarks despite having fewer parameters in many cases.
      </Prose>

      <Prose>
        The scaling also extends across different technique combinations. Best-of-N,
        majority voting, and PRM-guided search are all additive: combining a larger N with
        a better verifier and a beam search produces higher accuracy than any single
        technique alone, up to the ceiling of the verifier's precision.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        The log-linear curve is specific to problems where the model's latent knowledge is
        sufficient to solve the problem if given enough exploration time. For problems
        beyond the model's knowledge horizon — questions that require information not
        present in training — more reasoning tokens produce more elaborate wrong answers,
        not correct ones. The reasoning trace can be long, coherent, and confidently wrong.
      </Prose>

      <Prose>
        Verifier quality is the binding constraint at extreme compute. Once N is large
        enough that at least one correct answer is almost certainly in the sample, the
        question is entirely whether the verifier can identify it. A process reward model
        trained on insufficient data or in-distribution only will misrank candidates at
        test-time, converting a high-N sampling budget into garbage selection. At very
        large N, a 5% verifier error rate is devastating: with 100 candidates, roughly
        five wrong answers will be scored above the correct one if the PRM makes systematic
        errors, and the returned answer is likely wrong regardless of how many correct
        samples were generated.
      </Prose>

      <Prose>
        Subjective tasks do not benefit. Accuracy on creative writing quality, helpfulness
        in conversation, and emotional calibration in support contexts does not improve
        reliably with extended reasoning. These are not verifiable tasks — there is no
        ground truth — and the RL training signal that makes serial CoT productive is not
        available for them. Applying test-time compute to these tasks wastes tokens and
        adds latency without measurable quality improvement.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Best-of-N without a verifier degenerates to random selection</H3>

      <Prose>
        If you sample N answers but have no way to score them, you must pick one at random.
        Random selection from N independent draws with per-sample accuracy p gives expected
        accuracy exactly p — the same as a single draw. The entire compute budget of N samples
        produces zero benefit over one sample when there is no selection signal. This failure
        is obvious in theory but common in practice: teams implement best-of-N sampling
        without implementing a verifier, believing the diversity of outputs will somehow
        help. It does not. Best-of-N requires a reliable scorer.
      </Prose>

      <H3>9.2 Verifier gaming (Goodhart's Law for reasoning)</H3>

      <Prose>
        A learned verifier — a reward model trained to score answers — can be "hacked" by
        the model. With enough samples, the model finds outputs that score highly on the
        verifier without being genuinely correct. This is Goodhart's Law applied to
        test-time compute: when the measure becomes the target, it ceases to be a good
        measure. The mitigation is to use ground-truth verifiers where possible (unit
        tests, symbolic checkers) rather than learned reward models. For tasks where only
        learned verifiers are available, monitor the divergence between verifier score and
        actual correctness on a held-out evaluation set.
      </Prose>

      <H3>9.3 Verbose collapse</H3>

      <Prose>
        Without careful RL training, models can learn that longer outputs are rewarded
        (because the grader sees more content and is more likely to find something correct)
        rather than that accurate outputs are rewarded. This produces "verbose collapse":
        reasoning traces that are long but structurally hollow — restating the problem,
        listing approaches without following them, repeating the same reasoning step
        in different words. Verbose collapse wastes compute with no accuracy gain and
        can actually harm accuracy by confusing the model's own reasoning. DeepSeek-R1
        mitigated this with a length penalty term and by using outcome-only rewards rather
        than step-level rewards that could incentivise verbosity.
      </Prose>

      <H3>9.4 Wasting compute on easy problems</H3>

      <Prose>
        An RL-trained model given a large token budget on a trivial problem will often
        use it — because the policy learned to deliberate on problems presented in a
        reasoning context. A model that always uses 32,000 tokens regardless of problem
        difficulty wastes roughly 10–100× the necessary compute on problems that could
        be solved at 300 tokens. Adaptive budget allocation — predicting query difficulty
        before solving it and allocating tokens accordingly — is an active research problem.
        In production, a simple heuristic (route one-shot factual questions to a standard
        path, route math/code to reasoning mode) captures most of the gain.
      </Prose>

      <H3>9.5 Latency explosion from serial chains</H3>

      <Prose>
        A serial reasoning trace of 32,000 tokens at 50 tokens/second takes over ten
        minutes. A 64,000-token trace takes over twenty. Users waiting for a math solution
        can tolerate two minutes; waiting twenty minutes for a web search suggestion is
        unusable. The failure mode is applying serial chain-of-thought to latency-sensitive
        paths because it improves accuracy on benchmarks, without accounting for the
        user experience of near-real-time interaction. Serving systems for reasoning models
        need aggressive timeout policies and graceful fallback to standard generation
        when reasoning chains approach time limits.
      </Prose>

      <H3>9.6 Tree search with poor exploration-exploitation balance</H3>

      <Prose>
        Beam search with a PRM can get stuck in a local optimum: the top-scoring beams
        all represent slight variations of the same wrong reasoning path, while the correct
        path was pruned early because its initial PRM score was below the beam threshold.
        Diversity-promoting techniques (DVTS — Diverse Verifier Tree Search, used in the
        HuggingFace search-and-learn project) explicitly diversify beams at each step to
        avoid this collapse. Beeching et al. (HuggingFace, December 2024) showed DVTS
        outperforms standard beam search precisely because it maintains coverage of
        qualitatively different reasoning paths rather than converging prematurely to a
        local optimum.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>OpenAI o1 System Card (September 2024)</H3>

      <Prose>
        The foundational public disclosure of test-time compute scaling in a production
        model. Released September 12, 2024, the system card documents o1's approach to
        RL training on verifiable rewards and the resulting log-linear accuracy-vs-token-
        budget behavior on AIME 2024 and GPQA Diamond. Available at
        <Code>openai.com/index/openai-o1-system-card</Code> and arXiv:2412.16720.
      </Prose>

      <H3>DeepSeek-R1 (arXiv:2501.12948, January 2025)</H3>

      <Prose>
        The DeepSeek-AI technical report documenting DeepSeek-R1-Zero (pure RL, no SFT)
        and DeepSeek-R1 (cold-start data + RL). The paper demonstrates that RL with
        GRPO on verifiable rewards produces emergent reasoning behaviors — self-correction,
        backtracking, structured verification — without being trained on reasoning
        demonstrations. R1 achieves 71% pass@1 on AIME 2024, 86.7% with majority voting,
        matching OpenAI o1. The paper is the primary reference for the DeepSeek-style
        training pipeline described in the RL for Reasoning topic.
      </Prose>

      <H3>Snell et al. 2024 (arXiv:2408.03314)</H3>

      <Prose>
        "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model
        Parameters." Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. August 2024.
        Formalises the compute-optimal test-time scaling strategy: adaptively allocating
        token budgets per problem based on estimated difficulty improves efficiency by
        over 4× compared to fixed best-of-N. On matching-FLOPs evaluations, compute-
        optimal test-time scaling outperforms a 14× larger model. The key empirical
        contribution is showing that the optimal training-vs-test-time compute split
        depends on problem difficulty distribution.
      </Prose>

      <H3>Wang et al. 2022 (arXiv:2203.11171)</H3>

      <Prose>
        "Self-Consistency Improves Chain of Thought Reasoning in Language Models." Xuezhi
        Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha
        Chowdhery, Denny Zhou. ICLR 2023. The foundational paper on majority voting over
        diverse CoT chains. Demonstrated +17.9pp on GSM8K, +11.0pp on SVAMP, +12.2pp
        on AQuA over single greedy decoding. Introduced the self-consistency framework
        that now underlies most production majority-vote implementations.
      </Prose>

      <H3>Beeching et al. 2024 (HuggingFace blog, December 16, 2024)</H3>

      <Prose>
        "Scaling test-time compute with open models." Edward Beeching, Lewis Tunstall,
        Sasha Rush. HuggingFace blog post with accompanying code at
        <Code>github.com/huggingface/search-and-learn</Code>. Demonstrates best-of-N,
        beam search, and Diverse Verifier Tree Search (DVTS) on open-weight models.
        Key finding: models as small as 1–3B parameters can outperform 70B models with
        sufficient test-time compute. DVTS explicitly maintains beam diversity to avoid
        the exploration-exploitation collapse of standard beam search.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1</H3>

      <Prose>
        Derive the best-of-N accuracy formula for a perfect verifier from first principles
        (start from the complementary probability that all N samples are wrong). Then
        compute the accuracy for p = 0.3, N = 10. What is the minimum N such that
        accuracy exceeds 99% for p = 0.3?
      </Prose>

      <Callout accent="green">
        Answer: P(at least one correct) = 1 - (1-p)^N. For p=0.3, N=10:
        1 - 0.7^10 = 1 - 0.0282 = 0.9718. For 99%: solve 0.7^N &lt; 0.01 →
        N &gt; log(0.01)/log(0.7) = 12.9, so N = 13.
      </Callout>

      <H3>Exercise 2</H3>

      <Prose>
        Why does verifier quality bound the accuracy gain from test-time compute at large
        N? Give a concrete example with numbers. If a verifier has 92% precision (8% of
        accepted answers are wrong), what is the maximum achievable accuracy from best-of-N
        regardless of how large N is?
      </Prose>

      <Callout accent="green">
        Answer: As N → ∞, nearly every sample is available to the verifier. The bottleneck
        shifts entirely to the verifier's precision. Maximum accuracy = verifier precision
        = 0.92. Even with N = 1,000,000 samples, 8% of returned answers are wrong. The
        selection step, not the generation step, becomes the binding constraint.
      </Callout>

      <H3>Exercise 3</H3>

      <Prose>
        Design a test-time compute strategy for a general-purpose chat assistant. The
        assistant handles: (a) casual conversation, (b) factual lookup, (c) Python
        debugging, (d) essay writing, (e) competitive math problems. Which tasks should
        use test-time compute and in what form? What routing signal would you use?
      </Prose>

      <Callout accent="green">
        Answer: (a) No — subjective, first answer is as good as the 10,000th. (b) No
        for in-distribution facts; maybe for complex retrieval-requiring questions.
        (c) Yes — best-of-N with unit tests as verifier; low latency via parallelism.
        (d) No — no verifier for quality. (e) Yes — serial CoT (long deliberation) or
        beam search with a math PRM. Routing signal: classify intent (verifiable vs
        open-ended) and complexity (simple vs multi-step) at the system prompt level.
      </Callout>

      <H3>Exercise 4</H3>

      <Prose>
        Compare self-consistency (majority voting over CoT chains, no verifier) and
        PRM-guided beam search (verifier at each step). On what tasks does self-consistency
        outperform beam search? When does beam search dominate? What does each approach
        require that the other does not?
      </Prose>

      <Callout accent="green">
        Answer: Self-consistency requires only a model that produces the correct answer
        as the modal answer — no verifier, no PRM, no additional infrastructure. It
        performs well when p &gt; 0.5 per sample and sampling is cheap. Beam search
        requires a trained PRM and is serial (blocking per step), but directs compute
        toward productive paths rather than sampling blindly. Beam search dominates
        on long multi-step problems where early error detection is valuable. Self-
        consistency dominates on tasks where all-or-nothing final answer generation
        is feasible and latency is a concern.
      </Callout>

      <H3>Exercise 5</H3>

      <Prose>
        Predict what happens when you apply test-time compute scaling to a non-verifiable
        task (e.g., "write a compelling opening paragraph for an essay on urban planning").
        Will the log-linear accuracy curve appear? Why or why not? What failure mode would
        you expect to observe?
      </Prose>

      <Callout accent="green">
        Answer: The log-linear curve will not appear. There is no verifier signal to guide
        beam search or best-of-N selection, and no ground-truth reward for RL training on
        this task. Without a verifier, best-of-N degenerates to random selection (accuracy
        = p regardless of N). Serial CoT may produce longer, more elaborate text but not
        reliably better text by any objective measure. The failure mode is verbose collapse:
        the model produces a long internal deliberation that circles around stylistic choices
        without a signal to distinguish better from worse, then returns an answer not
        meaningfully different from its first attempt.
      </Callout>

    </div>
  ),
};

export default testTimeCompute;
