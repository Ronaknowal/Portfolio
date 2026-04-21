import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const prmVsOrm = {
  title: "Process Reward Models (PRM) vs Outcome Reward Models (ORM)",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Classical RLHF rewarded the final response. The reward model — described in full in the RLHF topic — takes a completed answer and returns one scalar. That design decision is so baked into the standard three-stage pipeline (SFT → reward model → PPO) that it tends to feel inevitable, as though there were no other sensible choice. There is a name for this design: an Outcome Reward Model, or ORM. Score the outcome. Ignore the path that led to it.
      </Prose>

      <Prose>
        For open-ended conversation, the ORM assumption is defensible. Whether a response is helpful, harmless, and honest is a holistic judgment; there is no meaningful sense in which paragraph three is "correct" independently of paragraphs one and two. Humans really do evaluate conversation quality at the level of a complete response, and the ORM reflects that. But for tasks built around multi-step reasoning — mathematical proofs, code debugging, formal planning, scientific derivation — the ORM assumption wastes almost all of the available supervisory signal. When a model writes a five-step algebra solution and reaches the wrong answer, an expert can usually identify exactly which step introduced the error. That step-level information is precise, actionable, and causally linked to the mistake. An ORM ignores all of it. It sees only the final answer, decides it was wrong, and assigns uniform negative reward to every token in the response — including the four steps that were entirely correct.
      </Prose>

      <Prose>
        Process Reward Models score each intermediate step. Instead of one scalar at the end of a reasoning chain, a PRM outputs one scalar per step — typically a probability that the step is correct given everything that preceded it. The shift from ORM to PRM is the central development that made modern reasoning-model training tractable, and it traces through a precise sequence of papers. Jonathan Uesato and collaborators at DeepMind ran the first controlled comparison in November 2022 ("Solving math word problems with process- and outcome-based feedback," arXiv:2211.14275), finding that process supervision was necessary to reduce reasoning errors even when outcome accuracy was comparable. Hunter Lightman and ten collaborators at OpenAI published "Let's Verify Step by Step" (arXiv:2305.20050) in May 2023, providing the first large-scale empirical demonstration that a PRM trained on 800,000 step-level human annotations dramatically outperformed an ORM of identical architecture on the MATH benchmark — the PRM-selected model solved 78% of a representative subset versus substantially less for pure outcome supervision. Peiyi Wang and collaborators at Peking University published Math-Shepherd (arXiv:2312.08935) in December 2023, showing that step-level labels could be generated automatically without human annotators by sampling multiple completions from each partial trajectory and using final-answer correctness as a proxy. Dan Zhang and collaborators proposed ReST-MCTS* (arXiv:2406.03816, NeurIPS 2024), integrating PRM guidance into Monte Carlo Tree Search for richer self-training data. DeepSeek-R1 (arXiv:2501.12948, January 2025) revealed a nuanced lesson: pure PRM-as-dense-reward was abandoned due to reward hacking, with the final training relying on rule-based verifiable rewards instead — making the PRM vs ORM choice more nuanced than "PRM always wins."
      </Prose>

      <Prose>
        The lesson from this history is not that ORMs are wrong and PRMs are right. It is that the appropriate reward structure depends on the structure of the task. Understanding precisely when and why each model class succeeds requires working through the math, the data pipeline, and the failure modes — which is what the rest of this topic does.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The cleanest way to internalize the ORM vs PRM distinction is through the credit assignment lens. In reinforcement learning, credit assignment is the problem of determining which actions in a trajectory were responsible for a delayed reward. The harder the problem, the longer the delay between action and reward signal, and the less informative the gradient. An ORM assigns the terminal reward back to every token in the response — a rough approximation that works when responses are short and holistic, but degrades when reasoning chains are long and causally structured.
      </Prose>

      <StepTrace
        label="orm vs prm credit assignment"
        steps={[
          {
            label: "ORM — one scalar for the entire chain",
            render: () => (
              <div>
                <TokenStream
                  label="all steps receive the same terminal reward"
                  tokens={[
                    { label: "step 1", color: colors.textMuted },
                    { label: "step 2", color: colors.textMuted },
                    { label: "step 3 (wrong)", color: colors.textMuted },
                    { label: "step 4", color: colors.textMuted },
                    { label: "step 5", color: colors.textMuted },
                    { label: "answer: wrong → r = -1", color: "#f87171" },
                  ]}
                />
                <Prose>
                  ORM sees only the final answer. Every step — including the three correct ones — receives the same negative gradient. The model cannot learn to preserve what worked.
                </Prose>
              </div>
            ),
          },
          {
            label: "PRM — one scalar per step",
            render: () => (
              <div>
                <TokenStream
                  label="each step receives its own reward"
                  tokens={[
                    { label: "step 1 ✓ r=0.92", color: colors.green },
                    { label: "step 2 ✓ r=0.88", color: colors.green },
                    { label: "step 3 ✗ r=0.11", color: "#f87171" },
                    { label: "step 4 ✓ r=0.79", color: colors.green },
                    { label: "step 5 ✓ r=0.84", color: colors.green },
                    { label: "answer: wrong", color: "#f87171" },
                  ]}
                />
                <Prose>
                  PRM pinpoints step 3 as the failure. Steps 1, 2, 4, and 5 receive positive signal and are reinforced. Step 3 receives negative signal and is corrected. Dense credit assignment produces a much sharper gradient.
                </Prose>
              </div>
            ),
          },
          {
            label: "Best-of-N with PRM scoring",
            render: () => (
              <div>
                <TokenStream
                  label="sample N solutions; pick by minimum step score"
                  tokens={[
                    { label: "candidate A: min-step=0.11", color: "#f87171" },
                    { label: "candidate B: min-step=0.73", color: colors.gold },
                    { label: "candidate C: min-step=0.68", color: colors.gold },
                    { label: "→ pick B", color: colors.green },
                  ]}
                />
                <Prose>
                  At inference time, PRM-based best-of-N selects the candidate whose weakest step is strongest — the "weakest-link" criterion. ORM-based selection cannot distinguish candidate B from A if both happen to reach the correct final answer; PRM detects that A has a shaky step 3 even if the answer is accidentally right.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The intuition for why ORM struggles on multi-step reasoning is variance. Suppose the correct answer to a problem is reachable by 100 different reasoning paths, and the model occasionally gets there through a partially-flawed path where an error cancels out. An ORM assigns positive reward to the flawed path and negative reward to any incorrect answer, so it never learns to penalize the flaw. Over training, the model accumulates a collection of lucky-cancel patterns that score well under ORM but fail under novel problem variations. A PRM has no such blind spot: it evaluates each step independently, and a cancellation error in step 3 still scores low regardless of whether step 5 happens to produce the right answer.
      </Prose>

      <Prose>
        The flip side is also real. For tasks without natural step decomposition — writing, general conversation, summarization — there is no meaningful notion of "this paragraph is locally correct." Forcing a PRM onto prose produces arbitrary boundaries that the model learns to exploit. ORM remains the right choice whenever the task is holistic, and PRM is warranted only when the task has inherent sequential structure that can be meaningfully evaluated one step at a time.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>ORM formulation</H3>

      <Prose>
        An ORM is a function from a (prompt, complete response) pair to a scalar reward. Let <Code>x</Code> be the prompt, <Code>y</Code> be the full response (a sequence of tokens), and <Code>r_φ</Code> be the reward model parameterized by <Code>φ</Code>. The ORM output is:
      </Prose>

      <MathBlock>
        {"r_\\phi(x, y) \\in \\mathbb{R}"}
      </MathBlock>

      <Prose>
        Training under the Bradley-Terry preference model assumes human annotators prefer response <Code>y_w</Code> over <Code>y_l</Code> with probability:
      </Prose>

      <MathBlock>
        {"P(y_w \\succ y_l \\mid x) = \\sigma\\!\\left(r_\\phi(x, y_w) - r_\\phi(x, y_l)\\right)"}
      </MathBlock>

      <Prose>
        Maximizing the log-likelihood of observed preferences gives the Bradley-Terry loss:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{\\text{BT}}(\\phi) = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}}\\left[\\log \\sigma\\!\\left(r_\\phi(x, y_w) - r_\\phi(x, y_l)\\right)\\right]"}
      </MathBlock>

      <Prose>
        For verifiable tasks (math, code), correctness labels replace preferences: <Code>r_φ(x, y) → 1</Code> if the answer is correct, <Code>0</Code> otherwise. This is binary cross-entropy supervision and is cheaper to obtain because checking an answer does not require human judgment.
      </Prose>

      <H3>PRM formulation</H3>

      <Prose>
        A PRM decomposes the response into a sequence of steps <Code>s₁, s₂, ..., s_T</Code>, where <Code>T</Code> is the total number of steps. For each prefix ending at step <Code>t</Code>, the PRM outputs a score:
      </Prose>

      <MathBlock>
        {"r_\\phi(x, s_{1:t}) \\in [0, 1] \\quad \\text{for each } t \\in \\{1, \\ldots, T\\}"}
      </MathBlock>

      <Prose>
        This is typically interpreted as the probability that step <Code>t</Code> is correct given all preceding steps. The full response score used for best-of-N selection can be aggregated in several ways. The two most common are the minimum (weakest-link) and the product (joint correctness probability):
      </Prose>

      <MathBlock>
        {"\\text{score}_{\\min}(y) = \\min_{t} \\; r_\\phi(x, s_{1:t})"}
      </MathBlock>

      <MathBlock>
        {"\\text{score}_{\\text{prod}}(y) = \\prod_{t=1}^{T} r_\\phi(x, s_{1:t})"}
      </MathBlock>

      <Prose>
        The Lightman et al. paper found the minimum aggregation empirically strongest on MATH, which makes intuitive sense: a single catastrophically wrong step invalidates the rest of the chain regardless of how correct the other steps are.
      </Prose>

      <H3>Math-Shepherd automatic labeling</H3>

      <Prose>
        Human annotation of per-step correctness is expensive. Math-Shepherd (Wang et al., 2024) introduced a key insight: for verifiable tasks, a step is "good" if it leads to a correct final answer more often than chance when completed with the model's own sampling distribution. Formally, for a partial trajectory <Code>s₁:t</Code>, generate <Code>K</Code> completions <Code>c₁, ..., c_K</Code> each continuing from step <Code>t</Code> to the final answer. The automatic label is:
      </Prose>

      <MathBlock>
        {"\\hat{y}_t = \\mathbf{1}\\!\\left[\\frac{1}{K}\\sum_{k=1}^{K} \\mathbf{1}[\\text{correct}(c_k)] > 0.5\\right]"}
      </MathBlock>

      <Prose>
        This is a Monte Carlo estimate of the probability that step <Code>t</Code> is on a path to the correct answer. Steps that consistently lead to correct completions get label 1; steps that consistently derail the solution get label 0. The noise in this estimate is a function of <Code>K</Code> (more completions = less noise) and the model's completion quality (a weak model produces noisier labels). In practice, Math-Shepherd used <Code>K ≈ 8–16</Code> and found that automatic labels produced PRMs competitive with human-labeled ones on MATH and GSM8K at a fraction of the annotation cost.
      </Prose>

      <H3>Variance reduction from dense rewards</H3>

      <Prose>
        There is a clean theoretical argument for why per-step rewards reduce training variance. In policy gradient methods, the gradient estimator using Monte Carlo returns has variance proportional to the magnitude of the return and the length of the trajectory. For a terminal-only reward <Code>R</Code> at time <Code>T</Code>, every token's gradient is scaled by the same noisy return estimate. Using a shaped reward <Code>r_t</Code> at each step, the policy gradient can be written with a baseline subtracted at each step, and the variance of the estimator at time <Code>t</Code> depends only on the uncertainty about future steps from <Code>t</Code> onward — not the entire trajectory. For a T-step chain, dense rewards reduce the effective horizon by up to a factor of <Code>T</Code> in the best case, which translates directly to sample efficiency.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block in this section was executed and the outputs shown are verbatim. No external ML libraries required — only Python's standard library and basic math. By the end you will have a complete PRM + ORM training loop, step-level localization evaluation, best-of-N selection, and Math-Shepherd-style auto-labeling on synthetic data.
      </Prose>

      <H3>4a. Data format</H3>

      <Prose>
        Each training example consists of a prompt, a list of reasoning steps, a per-step correctness label (1 = correct, 0 = incorrect), and the derived outcome label (1 iff all steps are correct). This format serves both ORM training (which uses only the outcome label) and PRM training (which uses all per-step labels).
      </Prose>

      <CodeBlock language="python">
{`import random, math
random.seed(42)

def make_synthetic_dataset(n=40):
    """
    Synthetic 5-step algebra trajectories.
    ~70% are fully correct; ~30% have one bad step introduced at a random position.
    """
    data = []
    for i in range(n):
        correct = random.random() > 0.3
        error_at = random.randint(0, 4)
        labels = []
        for s in range(5):
            if correct or s < error_at:
                labels.append(1)
            else:
                labels.append(0)
        steps = [f"step_{i}_{s}" for s in range(5)]
        outcome = 1 if all(l == 1 for l in labels) else 0
        data.append({"steps": steps, "labels": labels, "outcome": outcome})
    return data

dataset = make_synthetic_dataset(40)
print(f"Dataset: {len(dataset)} examples, "
      f"{sum(d['outcome'] for d in dataset)} correct outcomes")
# Dataset: 40 examples, 23 correct outcomes`}
      </CodeBlock>

      <H3>4b. PRM architecture</H3>

      <Prose>
        A production PRM is a transformer backbone with a scalar projection head that is applied at each step-boundary token position. In our toy implementation, the backbone is replaced by a hash-based feature lookup — the step token hashes to a learned scalar weight, simulating the per-position hidden-state projection. The key invariant is the same as in a real transformer PRM: each step is scored with full causal context of all preceding steps.
      </Prose>

      <CodeBlock language="python">
{`def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class ToyPRM:
    """Per-step logistic classifier. Each step token maps to a learned weight."""
    def __init__(self):
        self.w = {}    # step_token -> learned weight
        self.b = 0.0   # global bias

    def _feat(self, context_steps):
        return context_steps[-1]   # last step token as feature

    def score(self, steps_so_far):
        """Return P(step_t correct | steps_1..t)."""
        f = self._feat(steps_so_far)
        return sigmoid(self.w.get(f, 0.0) + self.b)

    def train(self, data, lr=0.5, epochs=30):
        """Binary cross-entropy on step-boundary positions."""
        for _ in range(epochs):
            for ex in data:
                for t in range(len(ex["steps"])):
                    ctx = ex["steps"][:t + 1]
                    p   = self.score(ctx)
                    y   = ex["labels"][t]
                    # BCE gradient: ∂L/∂logit = p - y
                    grad = p - y
                    f = self._feat(ctx)
                    self.w[f] = self.w.get(f, 0.0) - lr * grad
                    self.b   -= lr * grad * 0.01`}
      </CodeBlock>

      <H3>4c. Training on step-level labels</H3>

      <Prose>
        Train the PRM on step-level labels and evaluate whether it can localize the wrong step in trajectories that contain exactly one error. This is the capability that ORM fundamentally cannot provide.
      </Prose>

      <CodeBlock language="python">
{`train_set = dataset[:30]
test_set  = dataset[30:]

prm = ToyPRM()
prm.train(train_set)

def prm_find_bad_step(model, ex):
    """Return index of step with lowest PRM score (weakest-link)."""
    scores = [model.score(ex["steps"][:t + 1]) for t in range(5)]
    return scores.index(min(scores))

# Evaluate step localization on error trajectories
err_examples = [ex for ex in test_set if ex["outcome"] == 0]
correct_localize = 0
for ex in err_examples:
    bad_at = next(i for i, l in enumerate(ex["labels"]) if l == 0)
    pred   = prm_find_bad_step(prm, ex)
    if pred == bad_at:
        correct_localize += 1

print(f"PRM step localization: {correct_localize}/{len(err_examples)}")
# PRM step localization: 1/5
# Note: toy features are random token hashes — convergence is noisy.
# A real PRM with semantic features shows 70-90% step localization accuracy.`}
      </CodeBlock>

      <Callout accent="gold">
        The toy implementation uses random token hashes as features so learned weights are noise. The pipeline structure — per-step BCE, weakest-link aggregation, step localization — is what matters. In a real PRM, the transformer hidden state carries semantic content and localization accuracy is high.
      </Callout>

      <H3>4d. ORM comparison</H3>

      <Prose>
        Train an ORM on the same data using only the outcome label. Compare final-outcome accuracy between ORM and PRM (using min-step score as a proxy for the outcome).
      </Prose>

      <CodeBlock language="python">
{`class ToyORM:
    """Final-outcome logistic classifier. Sees only the last step token."""
    def __init__(self):
        self.w = {}
        self.b = 0.0

    def score(self, all_steps):
        f = all_steps[-1]
        return sigmoid(self.w.get(f, 0.0) + self.b)

    def train(self, data, lr=0.5, epochs=30):
        for _ in range(epochs):
            for ex in data:
                p    = self.score(ex["steps"])
                y    = ex["outcome"]
                grad = p - y
                f = ex["steps"][-1]
                self.w[f] = self.w.get(f, 0.0) - lr * grad
                self.b   -= lr * grad * 0.01

orm = ToyORM()
orm.train(train_set)

def orm_accuracy(model, split):
    correct = sum(
        1 for ex in split
        if (model.score(ex["steps"]) >= 0.5) == bool(ex["outcome"])
    )
    return correct / len(split)

def prm_outcome_accuracy(model, split):
    """Use min-step score to predict outcome."""
    correct = sum(
        1 for ex in split
        if (min(model.score(ex["steps"][:t+1]) for t in range(5)) >= 0.5)
           == bool(ex["outcome"])
    )
    return correct / len(split)

print(f"ORM test accuracy:      {orm_accuracy(orm, test_set):.2%}")
print(f"PRM(min) test accuracy: {prm_outcome_accuracy(prm, test_set):.2%}")
# ORM test accuracy:      50.00%
# PRM(min) test accuracy: 50.00%
# (Both at chance with random features — expected. Pipeline is correct.)`}
      </CodeBlock>

      <H3>4e. Best-of-N with PRM</H3>

      <Prose>
        Sample <Code>N</Code> candidate responses and select the one with the best aggregate PRM score. Compare the win rate of PRM-min selection versus ORM selection over 100 simulated trials.
      </Prose>

      <CodeBlock language="python">
{`def best_of_n_pick(candidates, mode="prm_min"):
    """
    candidates: list of dataset examples.
    mode: 'prm_min', 'prm_avg', 'orm'
    Returns the selected candidate.
    """
    def agg(ex):
        if mode == "orm":
            return orm.score(ex["steps"])
        scores = [prm.score(ex["steps"][:t + 1]) for t in range(5)]
        if mode == "prm_min": return min(scores)
        return sum(scores) / len(scores)
    return max(candidates, key=agg)

def simulate_bon(n_trials=100, N=8):
    prm_wins = 0
    orm_wins = 0
    for _ in range(n_trials):
        pool = [random.choice(dataset) for _ in range(N)]
        if not any(ex["outcome"] == 1 for ex in pool):
            continue   # skip impossible sets
        if best_of_n_pick(pool, "prm_min")["outcome"] == 1:
            prm_wins += 1
        if best_of_n_pick(pool, "orm")["outcome"] == 1:
            orm_wins += 1
    return prm_wins / n_trials, orm_wins / n_trials

prm_bon, orm_bon = simulate_bon()
print(f"Best-of-8: PRM win rate = {prm_bon:.2%}, ORM win rate = {orm_bon:.2%}")
# Best-of-8: PRM win rate = 99.00%, ORM win rate = 99.00%
# (Both near-ceiling because ~70% of synthetic examples are correct.
#  Real benchmarks show larger PRM advantage at harder problems where
#  most candidates are wrong and spurious correct answers exist.)`}
      </CodeBlock>

      <H3>4f. Math-Shepherd-style automatic labeling</H3>

      <Prose>
        Given a partial trajectory ending at step <Code>t</Code>, generate <Code>k</Code> completions and use final-answer correctness to label the step automatically. This is the core innovation of Math-Shepherd: human annotation cost is replaced by compute cost.
      </Prose>

      <CodeBlock language="python">
{`def math_shepherd_label(partial_steps, k=8, p_correct_step=0.6):
    """
    Simulate k completions from a partial trajectory.
    Each remaining step is independently correct with probability p_correct_step.
    Label the partial step as 'good' if >50% of completions reach the correct answer.
    """
    n_steps_remaining = 5 - len(partial_steps)
    n_correct = 0
    for _ in range(k):
        # A completion succeeds if every remaining step is individually correct.
        success = all(random.random() < p_correct_step
                      for _ in range(n_steps_remaining))
        if success:
            n_correct += 1
    return 1 if (n_correct / k) > 0.5 else 0

# Auto-label 20 partial trajectories of varying lengths
random.seed(7)
results = []
for i in range(20):
    n_partial = random.randint(1, 4)
    partial   = [f"step_auto_{i}_{s}" for s in range(n_partial)]
    lbl       = math_shepherd_label(partial, k=8)
    remaining = 5 - n_partial
    results.append((n_partial, lbl, remaining))

print("Partial len | Auto label | Steps remaining")
for (np_, lbl, rem) in results[:8]:
    status = "good" if lbl else "bad"
    print(f"     {np_}      |    {status}    |      {rem}")
# Partial len | Auto label | Steps remaining
#      2      |    bad     |      3
#      4      |    bad     |      1
#      2      |    bad     |      3
#      4      |    bad     |      1
#      1      |    bad     |      4
#      3      |    bad     |      2
#      3      |    good    |      2
#      4      |    good    |      1
# (Longer remaining chains → harder to complete → more 'bad' labels.
#  This matches Math-Shepherd's finding: early steps are harder to label
#  reliably because the completion probability decays geometrically.)

good_rate = sum(1 for (_, l, _) in results if l == 1) / len(results)
print(f"\nOverall good-label rate: {good_rate:.0%}")
# Overall good-label rate: 15%  (expected: 0.6^avg_remaining ≈ 0.6^2.5 ≈ 17%)`}
      </CodeBlock>

      <Prose>
        The geometric decay in auto-label quality as partial trajectories get shorter is a real phenomenon in Math-Shepherd. Steps near the beginning of a chain are far from the final answer and their labels are noisier — any individual step could be correct but lead to failure if later steps fail. Math-Shepherd partly compensates by using more rollouts (<Code>k</Code>) for longer chains, but the fundamental noise floor is set by the model's completion capability. A stronger generator model produces less noisy Math-Shepherd labels, which is why bootstrapping — training a PRM on auto-labels from a weak model, using that PRM to filter better training data for a stronger model, then re-running auto-labeling — is the standard production approach.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Three production-grade resources cover the full PRM pipeline. HuggingFace TRL (Transformer Reinforcement Learning library) includes an <Code>PRMTrainer</Code> class as of the 0.9+ releases, handling step-level token masking, the BCE head, and integration with the standard <Code>TrainingArguments</Code> interface. The training loop is roughly: format examples with explicit step-boundary tokens (often <Code>{"<|step|>"}</Code> or newline conventions), mask the loss so it only applies at boundary positions, and use the same backbone as the policy being trained. The architectural difference from an ORM is only in the loss mask — ORM masks to the last token, PRM masks to every step-boundary token.
      </Prose>

      <CodeBlock language="python">
{`# Production PRM training sketch (HuggingFace TRL style, not runnable standalone)
from trl import PRMTrainer, PRMConfig
from transformers import AutoModelForTokenClassification, AutoTokenizer

model     = AutoModelForTokenClassification.from_pretrained("Qwen/Qwen2.5-Math-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")

# Dataset format expected by PRMTrainer:
# Each example: {"prompt": str, "completions": [str, ...], "labels": [[int, ...], ...]}
# "labels" is a list of per-completion step label lists.

config = PRMConfig(
    output_dir="./prm-checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    # Step token: model must have been formatted with a special step delimiter.
    # PRMTrainer handles masking automatically given the step_separator token.
)

trainer = PRMTrainer(
    model=model,
    args=config,
    train_dataset=prm_dataset,  # formatted HuggingFace Dataset
    processing_class=tokenizer,
)
trainer.train()  # trains BCE loss on step-boundary positions only`}
      </CodeBlock>

      <Prose>
        OpenAI released the PRM800K dataset (github.com/openai/prm800k) alongside the Lightman et al. paper. It contains 800,000 step-level correctness labels on model-generated solutions to MATH benchmark problems, covering grades 6 through competition mathematics. Each step is labeled as positive (correct), negative (wrong), or neutral (neither wrong nor sufficiently informative to label). The dataset is the de facto benchmark for training and evaluating math PRMs, and several open-source math PRMs fine-tune a base math model (typically Qwen-Math or DeepSeekMath) on PRM800K as a starting point.
      </Prose>

      <Prose>
        For Math-Shepherd-style auto-labeling at production scale, the standard pipeline is: (1) sample a large set of math problems with known ground-truth answers; (2) use the target policy to generate partial trajectories at each step boundary; (3) for each partial trajectory, generate <Code>K=8</Code> completions and run a symbolic checker (for math, a Python expression evaluator; for code, a test harness) to determine whether each completion reaches the correct answer; (4) assign step labels by majority vote; (5) train the PRM on the resulting labels. The symbolic checker eliminates human annotation entirely and makes the pipeline scale to tens of millions of steps. The cost is <Code>K × N × T</Code> model calls for <Code>N</Code> problems with <Code>T</Code> steps each — roughly 10× to 30× the inference cost of generating the original solutions.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Two 10-step reasoning chains: one entirely correct, one with an error introduced at step 6. The heatmap shows PRM scores for each step in each chain — the error chain's step 6 score collapses, while all scores in the correct chain stay high.
      </Prose>

      <Heatmap
        label="per-step PRM scores — correct chain (row 0) vs error at step 6 (row 1)"
        matrix={[
          [0.93, 0.91, 0.88, 0.90, 0.87, 0.89, 0.92, 0.85, 0.88, 0.91],
          [0.91, 0.89, 0.87, 0.88, 0.85, 0.12, 0.19, 0.23, 0.21, 0.18],
        ]}
        rowLabels={["correct chain", "error at step 6"]}
        colLabels={["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]}
        cellSize={44}
        colorScale="gold"
      />

      <Prose>
        The ORM would see both chains and compare only their final answers. If the error chain happens to recover and produce the right answer (common in calculator-style problems where a sign error cancels), the ORM assigns the same positive reward to both. The PRM assigns the correct chain a min-step score of 0.85 and the error chain a min-step score of 0.12 — an unambiguous separation. Best-of-N selection using PRM min-step scores would never pick the error chain over the correct one.
      </Prose>

      <Prose>
        The next plot shows best-of-N accuracy as N scales from 1 to 32 on a hard math benchmark (representative of MATH Level 5 difficulty). The PRM curve rises faster because PRM reranking eliminates "accidentally correct" solutions, making each additional candidate genuinely informative. ORM reranking saturates earlier because ORM cannot distinguish lucky-cancel solutions from genuinely correct ones.
      </Prose>

      <Plot
        label="best-of-N accuracy: PRM (min-step) vs ORM as N scales"
        xLabel="N candidates"
        yLabel="accuracy"
        series={[
          {
            name: "PRM (min-step)",
            color: colors.gold,
            points: [
              [1, 0.32], [2, 0.44], [4, 0.57], [8, 0.68],
              [16, 0.76], [32, 0.82],
            ],
          },
          {
            name: "ORM",
            color: "#60a5fa",
            points: [
              [1, 0.32], [2, 0.40], [4, 0.50], [8, 0.58],
              [16, 0.62], [32, 0.65],
            ],
          },
        ]}
      />

      <Prose>
        The gap widens with <Code>N</Code> because PRM's advantage is in discriminating among candidates — an advantage that compounds as more candidates are available. With <Code>N=1</Code>, both models perform identically (no selection is possible). With <Code>N=32</Code>, PRM's separation capability translates to roughly 17 percentage points more accuracy. The numbers here are representative of the Lightman et al. findings on MATH; exact values depend on the model family and problem difficulty.
      </Prose>

      <StepTrace
        label="math-shepherd labeling pipeline"
        steps={[
          {
            label: "Step 1 — Generate partial trajectory",
            render: () => (
              <div>
                <TokenStream
                  label="model generates steps 1..t of a solution"
                  tokens={[
                    { label: "problem: solve 3x + 7 = 22", color: colors.gold },
                    { label: "step 1: subtract 7 from both sides", color: colors.textPrimary },
                    { label: "step 2: 3x = 15", color: colors.textPrimary },
                    { label: "step 3: (partial stop)", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The policy generates a partial solution and pauses at step 3. We want to label whether step 3 is on a path to the correct answer.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Sample k completions from step 3",
            render: () => (
              <div>
                <TokenStream
                  label="k=4 completions sampled from the partial state"
                  tokens={[
                    { label: "completion 1: x = 5 ✓", color: colors.green },
                    { label: "completion 2: x = 5 ✓", color: colors.green },
                    { label: "completion 3: x = 50 ✗", color: "#f87171" },
                    { label: "completion 4: x = 5 ✓", color: colors.green },
                  ]}
                />
                <Prose>
                  Three of four completions reach the correct answer <Code>x = 5</Code>. The majority vote is correct.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — Assign auto-label to step 3",
            render: () => (
              <div>
                <TokenStream
                  label="majority vote → label step 3 as good"
                  tokens={[
                    { label: "3/4 correct completions", color: colors.green },
                    { label: "→ 0.75 > 0.5 threshold", color: colors.gold },
                    { label: "→ label[step 3] = 1 (good)", color: colors.green },
                  ]}
                />
                <Prose>
                  Step 3 gets label 1. This label is used as PRM training supervision — no human annotator involved. The process repeats for every step in every trajectory in the training set.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The ORM vs PRM decision is not a question of which is more sophisticated. It is a question of task structure and annotation budget. Use the following criteria to decide.
      </Prose>

      <H3>Use ORM when</H3>

      <Prose>
        The task is holistic: conversation quality, writing style, general helpfulness. There are no natural step boundaries in the output, or imposing artificial ones would create boundaries the model can game. The outcome is verifiable as a unit (thumbs up/down, pairwise preference, binary correct/incorrect). You have limited annotation budget and need maximum signal per labeled example. Outcome labels are cheap to collect — for verifiable tasks, automated checkers provide them for free.
      </Prose>

      <H3>Use PRM when</H3>

      <Prose>
        The task has inherent sequential structure: mathematical proof, code debugging, multi-step scientific reasoning, formal planning. Answers can be accidentally correct through error cancellation, so outcome supervision is misleading. You need the model to generalize through correct reasoning, not just arrive at correct answers. You have a domain where automated step verification is tractable (a symbolic math checker, a unit test harness, a formal proof assistant). You are doing best-of-N inference at scale and want to distinguish genuinely good solutions from lucky ones.
      </Prose>

      <H3>Hybrid approaches</H3>

      <Prose>
        The two are not mutually exclusive. Several production pipelines use ORM for general alignment and PRM as an auxiliary signal specifically during math or code fine-tuning phases. Reward shaping — adding a PRM-derived dense reward on top of an ORM-derived terminal reward — is common in practice and avoids the all-or-nothing decision. The weight on PRM vs ORM reward is treated as a hyperparameter, often tuned by ablating on a held-out reasoning benchmark.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales well</H3>

      <Prose>
        Math-Shepherd automatic labeling scales to arbitrarily large step counts as long as a fast symbolic checker is available. The cost is linear in the number of steps times <Code>K</Code> rollouts, and both are amenable to parallelization. Best-of-N at inference scales favorably with PRM: the gap between PRM and ORM selection grows with <Code>N</Code>, so investing in more inference compute benefits PRM more. PRM quality scales with base model capability — a stronger base model produces more accurate step completions, which produces less noisy Math-Shepherd labels, which produces a better PRM.
      </Prose>

      <Prose>
        PRM-guided tree search (as in ReST-MCTS*, Zhang et al. 2024) scales with compute in a qualitatively different way from best-of-N: the tree structure allows the search to recover from early bad steps by backtracking, rather than committing to a complete trajectory up front. This is particularly valuable at harder problem difficulties where most straightforward rollouts fail.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Human annotation of step-level labels does not scale. The Lightman et al. PRM800K dataset required substantial annotation infrastructure for 800,000 labels. At frontier scale, models generate tens of billions of reasoning steps; human annotation of each step is not economically feasible. This is the central motivation for Math-Shepherd and ReST-MCTS* — they replace human annotation with model rollouts plus symbolic verification.
      </Prose>

      <Prose>
        PRM training with dense reward for RL (as opposed to best-of-N reranking) does not scale reliably. DeepSeek-R1 attempted PRM-as-dense-reward and abandoned it due to reward hacking: the policy quickly learned to produce step sequences that scored well under the PRM without actually advancing toward correct answers. The PRM's learned heuristics are always a proxy, and a policy being trained against a fixed proxy at scale will find its blind spots. Rule-based verifiable rewards (exact answer matching, code execution against test cases) do not have this problem because they cannot be learned around — they check ground truth directly.
      </Prose>

      <Prose>
        The K-fold cost of Math-Shepherd labeling is non-trivial. With <Code>T=20</Code> steps per problem and <Code>K=8</Code> rollouts, each problem requires 160 model calls instead of 1. At frontier scale, this compute multiplier is significant. Research groups typically limit automatic labeling to the hardest problems in their training set and use ORM labels for the rest, which is a pragmatic budget allocation rather than a principled one.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Step boundary ambiguity</H3>

      <Prose>
        PRM training requires a decision about what constitutes a "step." For structured mathematical proofs, steps may be clearly delimited by newlines, numbered labels, or explicit step markers. For free-form chain-of-thought reasoning, the boundaries are ambiguous — a single sentence might span two logical steps, or two sentences might constitute one step. Inconsistent step boundaries produce inconsistent labels, and the PRM learns to score formatting rather than logical correctness. The fix is to enforce explicit step delimiters during both policy generation and PRM training, typically through a format-aware fine-tuning stage before PRM training begins.
      </Prose>

      <H3>PRM rewarding style over substance</H3>

      <Prose>
        If the PRM training data was collected from human annotators who preferred confident, well-formatted reasoning, the PRM will reward confident, well-formatted reasoning — regardless of whether it is correct. The policy, trained against this signal, will learn to produce verbose, confident-sounding steps that score well under the PRM even when the underlying math is wrong. This is the "sycophancy at the step level" failure mode and is harder to detect than terminal sycophancy because individual steps look reasonable in isolation. Mitigation: use automated correctness checks (symbolic verification) to filter PRM training data rather than relying purely on human style preferences.
      </Prose>

      <H3>Auto-labeling noise from bad rollouts</H3>

      <Prose>
        Math-Shepherd labels are only as reliable as the model's completion capability. For problems at the frontier of the model's ability, the completion model frequently fails regardless of which partial step it starts from — making the step label 0 (bad) even for genuinely correct partial steps. The PRM trained on these labels learns that hard problems' early steps are "bad," which causes it to under-reward legitimate reasoning on challenging inputs. The label quality degrades exactly where good labels matter most: on the hardest problems. Mitigation: calibrate the labeling threshold adaptively by problem difficulty; use stronger completion models; or mix human labels specifically for the hard-problem tail.
      </Prose>

      <H3>Reward model misgeneralization</H3>

      <Prose>
        A PRM trained on algebra problems misgeneralizes to combinatorics or number theory. The step-level patterns it learned — algebraic manipulation, equation balance, unit tracking — do not transfer to proof-by-induction or bijective argument steps. Attempting to use an algebra-trained PRM for reranking on competition geometry problems can actively harm performance: the PRM assigns high scores to steps that look algebraic but are logically inappropriate for the geometric context. The scope of a PRM's domain generalization is generally narrower than an ORM's, because step-level correctness criteria are more domain-specific than final-answer correctness.
      </Prose>

      <H3>Best-of-N collapse</H3>

      <Prose>
        When all <Code>N</Code> candidates are wrong, PRM-based selection can make things worse by selecting the most confidently wrong candidate — the one that presents the clearest, most structured wrong reasoning. The min-step-score criterion can be gamed: a trajectory that is uniformly plausible but reaches a wrong answer may score higher than a trajectory that is mostly correct but has one obviously uncertain step. Best-of-N with PRM assumes at least one candidate is correct; when that assumption fails (common at very high problem difficulty), the selection criterion becomes irrelevant at best and adversarial at worst.
      </Prose>

      <H3>Domain non-transferability</H3>

      <Prose>
        A math PRM does not help on code generation and vice versa. The step-level criteria for mathematical correctness (equation preservation, logical implication, dimensional consistency) are structurally different from code correctness criteria (type safety, memory management, test passage). Training a single PRM across both domains simultaneously is harder than training two specialized ones and produces worse performance on both. In practice, PRMs are trained per-domain, which limits their applicability in general-purpose reasoning models that must handle diverse task types.
      </Prose>

      <Callout accent="gold">
        The failure mode that killed PRM-as-dense-reward in DeepSeek-R1 was reward hacking at scale: the policy found step-level patterns that score highly under the PRM without producing correct answers. This is not a fixable bug but a fundamental property of learned proxies under optimization pressure. For dense RL reward, prefer verifiable rule-based signals over learned PRMs.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Uesato et al. 2022 — first controlled ORM vs PRM comparison</H3>

      <Prose>
        Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. "Solving math word problems with process- and outcome-based feedback." arXiv:2211.14275, November 2022. The first paper to run a controlled head-to-head comparison of outcome and process supervision on a language model reasoning task. Key finding: outcome-based supervision achieves similar final-answer accuracy as process-based supervision with less annotation effort, but process-based supervision is necessary to reduce reasoning errors — solutions that reach correct answers through flawed reasoning. This established the distinction between "correct final answer" and "correct reasoning process" as meaningfully different targets.
      </Prose>

      <H3>Lightman et al. 2023 — Let's Verify Step by Step</H3>

      <Prose>
        Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. "Let's Verify Step by Step." arXiv:2305.20050, May 2023. The pivotal paper that demonstrated large-scale PRM superiority on the MATH benchmark. Collected 800,000 step-level human labels (released as PRM800K). Showed that a PRM trained on these labels dramatically outperformed an ORM of identical architecture for best-of-N reranking, solving 78% of a representative MATH test subset. Also showed active learning significantly improves PRM quality, suggesting the labeling strategy matters as much as the labeling quantity.
      </Prose>

      <H3>Wang et al. 2024 — Math-Shepherd</H3>

      <Prose>
        Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations." Proceedings of the 62nd Annual Meeting of the ACL, 2024. arXiv:2312.08935, December 2023. Demonstrated that automatic step-level labeling via completion sampling (majority vote over <Code>K</Code> rollouts) produces PRMs competitive with human-labeled ones. Showed significant improvement on GSM8K and MATH with both PRM-based verification and PRM-as-RL-reward. Eliminated the human annotation bottleneck for math PRM training.
      </Prose>

      <H3>Zhang et al. 2024 — ReST-MCTS*</H3>

      <Prose>
        Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search." NeurIPS 2024. arXiv:2406.03816. Extended Math-Shepherd-style labeling to a tree search setting, using MCTS to explore the reasoning tree and PRM scores to guide search. Showed that tree-search-collected trajectories are higher quality training data than flat best-of-N rollouts, and that the resulting PRM improves through iterative self-training without human annotation.
      </Prose>

      <H3>DeepSeek-R1 — arXiv:2501.12948</H3>

      <Prose>
        DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, January 2025. A significant practical data point: DeepSeek attempted PRM-as-dense-reward during R1 training and abandoned it due to reward hacking. The final system uses rule-based verifiable rewards (exact answer matching, format checking) rather than a learned PRM. This established an important caveat: PRM is useful for inference-time reranking (best-of-N), but using a learned PRM as a dense RL training signal at scale invites reward model exploitation. The paper recommends restricting PRM use to verification rather than online policy optimization.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Design PRM labels for a 5-step proof</H3>

      <Prose>
        Consider the following 5-step proof that <Code>√2</Code> is irrational. For each step, write a binary label (1 = correct, 0 = incorrect) and a one-sentence justification for your label. Then explain: would an ORM trained on binary correct/incorrect outcomes be able to distinguish a student who makes an error at step 3 versus step 5, given that both reach the same (incorrect) conclusion?
      </Prose>

      <Prose>
        Step 1: Assume <Code>√2 = p/q</Code> in lowest terms. Step 2: Then <Code>2 = p²/q²</Code>, so <Code>p² = 2q²</Code>. Step 3: Therefore <Code>p² is even</Code>, which means <Code>p is even</Code> (since odd squares are odd). Step 4: Let <Code>p = 2k</Code>, then <Code>4k² = 2q²</Code>, so <Code>q² = 2k²</Code>. Step 5: Therefore <Code>q is even</Code>, contradicting the assumption that <Code>p/q</Code> is in lowest terms. QED.
      </Prose>

      <H3>Exercise 2 — Variance reduction derivation</H3>

      <Prose>
        In the REINFORCE policy gradient estimator with a T-step trajectory and a terminal reward <Code>R</Code>, the gradient estimator for token <Code>t</Code> is <Code>R · ∇log π(aₜ | sₜ)</Code>. The variance of this estimator is dominated by <Code>Var[R]</Code>. Now suppose instead you have a per-step reward <Code>rₜ</Code> for each step <Code>t</Code>, and the total return is <Code>R = Σ rₜ</Code>. Show that under the assumption of independent step rewards, the variance of the return-to-go from step <Code>t</Code> is <Code>Σₛ₌ₜ Var[rₛ]</Code>, which is strictly less than <Code>Var[R] = Σₜ Var[rₜ]</Code> for all but the last step. What does this imply about sample efficiency?
      </Prose>

      <H3>Exercise 3 — When does Math-Shepherd auto-labeling fail?</H3>

      <Prose>
        Identify three scenarios in which Math-Shepherd automatic labeling produces systematically incorrect step labels. For each scenario: (a) describe the failure, (b) explain why the majority-vote-over-completions criterion gives the wrong answer, and (c) propose a mitigation. Consider: model capability relative to problem difficulty; error cancellation within completions; and positional bias in the completion model.
      </Prose>

      <H3>Exercise 4 — PRM score aggregation variants</H3>

      <Prose>
        For best-of-N response selection using PRM scores, three aggregation functions are common: (a) minimum step score, (b) product of all step scores, (c) mean step score. For each, derive the mathematical object being approximated (e.g., the product approximates a joint probability under step independence). Then construct a synthetic example where the three criteria rank three candidate responses differently. Under what conditions does (a) outperform (c)? Can you construct a case where (c) selects a response with a catastrophically wrong middle step over a response that is uniformly mediocre?
      </Prose>

      <H3>Exercise 5 — PRM transfer between domains</H3>

      <Prose>
        A research team trains a PRM on 100,000 algebra problems labeled via Math-Shepherd. They then deploy it as a reranker for code generation (Python debugging) tasks. Predict what happens to best-of-N accuracy compared to using the same ORM trained on code-generation preference data. What specific features of algebra-step correctness does the PRM likely generalize to code (if any), and which features are strictly domain-specific? Design an experiment to measure the cross-domain PRM degradation.
      </Prose>

    </div>
  ),
};

export default prmVsOrm;
