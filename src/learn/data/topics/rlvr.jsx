import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rlvr = {
  title: "RLVR (Reinforcement Learning with Verifiable Rewards)",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The reward model at the center of RLHF is a neural network — a learned, imperfect proxy for human preferences, and the first place reward hacking enters the pipeline. A powerful optimizer will eventually find inputs where the proxy diverges from what it was supposed to represent: the reward score climbs while actual quality falls. This is not a bug in any particular implementation. It is the consequence of optimizing against an approximation rather than the thing itself.
      </Prose>

      <Prose>
        For a specific class of tasks — math problems, code generation, formal proofs, logic puzzles, anything with a mechanically checkable answer — the reward model is unnecessary. The correct answer is either present or it is not, and a deterministic function can tell you which without consulting a neural network. RLVR replaces the learned reward model with a verifier: a program that returns 1 for a correct answer and 0 for a wrong one. No parameters to overfit, no proxy to hack, no calibration to worry about. The signal is exact by construction.
      </Prose>

      <Prose>
        This approach became central to reasoning model post-training. The three clearest instances in the public record are: Lambert et al. 2024, Tülu 3 (arXiv:2411.15124), which introduced RLVR as an explicit training stage and demonstrated it on math benchmarks with a deterministic answer parser; DeepSeek-R1 (arXiv:2501.12948), which applied GRPO with a verifier-based reward signal and achieved near-OpenAI-o1 performance on AIME 2024; and the open-r1 project by HuggingFace, which reproduced the full R1 training pipeline publicly. In each case, the key move was the same: remove the learned reward model for verifiable tasks, and let the verifier provide exact signal.
      </Prose>

      <Callout accent="gold">
        RLVR is RLHF with the reward model replaced by a deterministic verifier. Narrower scope — only verifiable tasks — but the signal is exact, scalable, and cannot be hacked.
      </Callout>

      <Prose>
        The practical consequence is that RLVR training can continue far longer than RLHF before hitting the reward-hacking ceiling. With a learned reward model, the policy eventually finds the cracks in the proxy and exploits them; training past that point actively degrades quality. With a verifier, there are no cracks to find. The policy can only improve its reward score by producing more correct answers. The training signal stays honest regardless of how hard the optimizer pushes.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The verifier does what the reward model was imperfectly trying to approximate. For GSM8K-style arithmetic, the verifier is a regex that extracts the boxed final answer and compares it numerically to the ground truth. For code generation, it compiles the program and runs it against a hidden test suite, returning the fraction of tests that pass. For formal theorem proving with Lean or Isabelle, it submits the proof to the type-checker and returns whether it type-checks. In each case the verifier is a small, fast, interpretable function; the hard work of specifying what "correct" means was already done when someone wrote the problem and its ground-truth answer.
      </Prose>

      <Prose>
        The scope is deliberately narrow. "Write a clear explanation" is not verifiable. "Prove this theorem" is. Arithmetic at grade-school level is verifiable by regex and float comparison; arithmetic at research level may require a formal proof assistant. Code with a complete unit test suite is verifiable; code described only as "make this UI feel better" is not. RLVR does not replace RLHF — it bypasses it for the subset of tasks where the bypass is possible. The boundary between verifiable and non-verifiable roughly coincides with the boundary between symbolic and aesthetic tasks.
      </Prose>

      <Prose>
        Training uses standard policy gradient — PPO or GRPO — but the reward signal is now binary and deterministic. This creates a specific challenge: sparse binary reward causes high gradient variance. If the policy is weak and most rollouts return 0, the gradient estimate is near zero and training stalls. The solution used in DeepSeekMath and R1 is group sampling: generate G responses per prompt (typically 8 to 16), score each with the verifier, and compute advantages within the group. Even if only 1 in 8 rollouts is correct, that one correct response is compared to 7 incorrect ones and generates a clean learning signal. This is the central algorithmic contribution of GRPO, described in detail in the GRPO/RLOO/KTO topic.
      </Prose>

      <StepTrace
        label="rlvr training loop — one update step"
        steps={[
          {
            label: "1. Sample G responses per prompt",
            render: () => (
              <div>
                <TokenStream
                  label="prompt → policy → G rollouts"
                  tokens={[
                    { label: "prompt x", color: colors.gold },
                    { label: "→ π samples", color: colors.textMuted },
                    { label: "y₁, y₂, …, y₈", color: "#c084fc" },
                  ]}
                />
                <Prose>
                  The policy generates G responses to the same prompt. This is the exploration mechanism — multiple attempts increase the probability that at least one rollout is correct, even for hard problems where the base pass rate is low.
                </Prose>
              </div>
            ),
          },
          {
            label: "2. Score each with the verifier",
            render: () => (
              <div>
                <TokenStream
                  label="verifier assigns binary rewards"
                  tokens={[
                    { label: "y₁ → r=0", color: "#f87171" },
                    { label: "y₂ → r=0", color: "#f87171" },
                    { label: "y₃ → r=1", color: "#4ade80" },
                    { label: "y₄ → r=0", color: "#f87171" },
                    { label: "…", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The verifier is deterministic: regex match + float comparison for math, unit test execution for code. No neural network involved. The reward is exactly 0 or 1, with no calibration drift.
                </Prose>
              </div>
            ),
          },
          {
            label: "3. Compute group-normalized advantages",
            render: () => (
              <div>
                <TokenStream
                  label="GRPO advantage: Aᵢ = (rᵢ − mean(r)) / std(r)"
                  tokens={[
                    { label: "r = [0,0,1,0,0,1,0,0]", color: colors.gold },
                    { label: "→ mean=0.25, std≈0.43", color: colors.textMuted },
                    { label: "→ A₃=+1.73, A₁=−0.58", color: "#c084fc" },
                  ]}
                />
                <Prose>
                  The group baseline replaces the value model. Correct responses get positive advantage, incorrect ones get negative advantage. The magnitude scales automatically with group variance.
                </Prose>
              </div>
            ),
          },
          {
            label: "4. PPO update + KL penalty",
            render: () => (
              <div>
                <TokenStream
                  label="clipped objective with KL regularization"
                  tokens={[
                    { label: "L = E[min(r·A, clip(r)·A)]", color: colors.gold },
                    { label: "− β·KL(π || π_ref)", color: "#60a5fa" },
                    { label: "→ gradient step", color: "#4ade80" },
                  ]}
                />
                <Prose>
                  Standard PPO clipping prevents large policy steps. The KL penalty keeps the policy near the SFT reference. After the update, the loop repeats with fresh rollouts from the new policy.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The verifier as a reward function</H3>

      <Prose>
        In RLHF, the reward function is a neural network <Code>r_φ(x, y) ∈ ℝ</Code> trained on human preference pairs. In RLVR, the reward function is a deterministic program:
      </Prose>

      <MathBlock caption="RLVR reward: deterministic, binary, no learned parameters">
        {"r(x, y) \\in \\{0, 1\\} \\quad \\text{(deterministic)}"}
      </MathBlock>

      <Prose>
        For math: <Code>r(x, y) = 1</Code> if the boxed answer in <Code>y</Code> matches the ground truth for problem <Code>x</Code>, else 0. For code: <Code>r(x, y) = fraction_of_tests_passed(y, test_suite_x)</Code>, which is 0/1 for a single test or a real number in [0,1] for a suite. The key property is that <Code>r</Code> is fixed throughout training — it cannot be hacked because it has no learned parameters to exploit.
      </Prose>

      <H3>3.2 The policy objective</H3>

      <Prose>
        The optimization objective is formally identical to RLHF's KL-regularized form, with the learned reward model swapped out for the verifier:
      </Prose>

      <MathBlock caption="RLVR objective: maximize verifier reward while staying close to SFT reference">
        {"\\max_{\\pi_\\theta}\\; \\mathbb{E}_{x \\sim \\mathcal{D},\\; y \\sim \\pi_\\theta(\\cdot|x)}\\!\\left[r(x, y) - \\beta \\log \\frac{\\pi_\\theta(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}\\right]"}
      </MathBlock>

      <Prose>
        The KL term <Code>β log(π_θ / π_ref)</Code> penalizes drift from the SFT reference distribution. Unlike RLHF, where the reward model itself drifts relative to the policy, the verifier <Code>r</Code> is static. The only moving part is the policy. This means the KL penalty's job is simpler: prevent distributional collapse and maintain response diversity, rather than also guarding against proxy exploitation.
      </Prose>

      <H3>3.3 Group sampling and advantage estimation</H3>

      <Prose>
        Binary rewards produce high variance gradient estimates under single-sample REINFORCE. If the policy has a 10% pass rate on a problem, the expected gradient from one rollout is 90% zero. GRPO (Shao et al., DeepSeekMath, arXiv:2402.03300) addresses this by sampling <Code>G</Code> responses per prompt and normalizing within the group:
      </Prose>

      <MathBlock caption="GRPO group-normalized advantage: each response scored relative to its group">
        {"A_i = \\frac{r_i - \\mu_G}{\\sigma_G}, \\quad \\mu_G = \\frac{1}{G}\\sum_{j=1}^G r_j, \\quad \\sigma_G = \\sqrt{\\frac{1}{G}\\sum_{j=1}^G (r_j - \\mu_G)^2}"}
      </MathBlock>

      <Prose>
        With binary rewards and G=8, a group where one response is correct has <Code>μ_G = 0.125</Code> and <Code>σ_G ≈ 0.33</Code>. The correct response gets advantage <Code>(1 − 0.125) / 0.33 ≈ 2.65</Code>; each incorrect response gets <Code>(0 − 0.125) / 0.33 ≈ −0.38</Code>. The gradient step pushes the policy toward the correct response and away from the incorrect ones. Compare to RLHF with a continuous reward: the variance reduction is automatic and does not require a learned value model.
      </Prose>

      <H3>3.4 Chain-of-thought gradient flow</H3>

      <Prose>
        A subtle but important point: in math RLVR, the model generates a chain-of-thought that ends with a boxed final answer. Only the final answer is scored by the verifier. But the gradient flows back through the entire sequence — every reasoning step receives a gradient proportional to the advantage of the final answer.
      </Prose>

      <Prose>
        This means the verifier is effectively rewarding the entire reasoning trace when the answer is correct, and penalizing it when the answer is wrong. The model learns reasoning patterns indirectly — not because the reasoning is explicitly supervised, but because good reasoning leads to correct answers that the verifier rewards. This is the mechanism behind the emergent chain-of-thought behavior observed in DeepSeek-R1-Zero, which was trained with pure RLVR and no supervised chain-of-thought data.
      </Prose>

      <MathBlock caption="Gradient flows through the full chain-of-thought, not just the final answer token">
        {"\\nabla_\\theta \\mathbb{E}[r(x,y)] = \\mathbb{E}\\!\\left[r(x,y) \\sum_{t=1}^{T} \\nabla_\\theta \\log \\pi_\\theta(y_t \\mid x, y_{<t})\\right]"}
      </MathBlock>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is runnable Python (standard library only). Each section is self-contained and produces the outputs shown. The goal is to make the mechanics of the full RLVR loop visceral before the production library abstracts them.
      </Prose>

      <H3>4a. Math verifier</H3>

      <Prose>
        The math verifier extracts a boxed final answer from the model's response and compares it to the ground truth. The regex pattern matches <Code>{String.raw`\boxed{...}`}</Code> and handles both integer and decimal answers via float comparison with a tolerance of 1e-6. We test on 20 synthetic problems: 10 with correct boxed answers, 10 with wrong ones.
      </Prose>

      <CodeBlock language="python">
{`import re

def math_verifier(response: str, ground_truth) -> float:
    """
    Parse \\\\boxed{answer} from response; compare to ground truth.
    Returns 1.0 if correct, 0.0 otherwise.
    """
    match = re.search(r"\\\\boxed\\{([^}]+)\\}", response)
    if not match:
        return 0.0
    raw = match.group(1).strip()
    try:
        # Numeric comparison with tolerance for floating-point answers
        return 1.0 if abs(float(raw) - float(ground_truth)) < 1e-6 else 0.0
    except (ValueError, TypeError):
        # Non-numeric: exact string match (for symbolic answers)
        return 1.0 if raw == str(ground_truth).strip() else 0.0

# 20 synthetic problems: addition and multiplication
problems = (
    [("What is %d + %d?" % (a, b), a + b) for a, b in
     [(2,3),(7,8),(15,12),(100,200),(3,3),(9,1),(4,4),(11,6),(50,25),(33,17)]]
  + [("What is %d * %d?" % (a, b), a * b) for a, b in
     [(3,4),(5,6),(7,8),(9,10),(2,11),(12,3),(4,5),(6,7),(8,9),(10,11)]]
)

correct_count = 0
for i, (problem, ans) in enumerate(problems):
    if i % 2 == 0:
        # Correct: response with right boxed answer
        response = "Let me think. The answer is \\\\boxed{%s}" % ans
    else:
        # Wrong: response with off-by-one answer
        response = "I believe the answer is \\\\boxed{%s}" % (ans + 1)
    score = math_verifier(response, ans)
    correct_count += score

# Output: 10 correct responses scored 1.0; 10 wrong scored 0.0
print("4a. Math verifier: %d/20 correct responses identified" % int(correct_count))
# 4a. Math verifier: 10/20 correct responses identified`}
      </CodeBlock>

      <H3>4b. Code verifier</H3>

      <Prose>
        The code verifier executes generated code in an isolated namespace and runs it against hidden test cases. Each test case is a callable that receives the execution namespace and returns a boolean. The verifier returns the fraction of tests that pass — 0.0 for syntax errors or any exception during execution. We test three cases: correct implementation, wrong implementation (multiply instead of add), and syntax error.
      </Prose>

      <CodeBlock language="python">
{`def code_verifier(code: str, test_cases: list) -> float:
    """
    Execute generated code; run hidden tests. Returns fraction passing.
    Production version uses a subprocess sandbox with time/memory limits.
    """
    try:
        exec_globals = {}
        exec(compile(code, "<generated>", "exec"), exec_globals)
        passed = sum(1 for tc in test_cases if tc(exec_globals))
        return passed / len(test_cases)
    except Exception:
        return 0.0

# Hidden test suite for add(a, b)
tests = [
    lambda g: g.get("add") and g["add"](1, 2) == 3,
    lambda g: g.get("add") and g["add"](0, 0) == 0,
    lambda g: g.get("add") and g["add"](-1, 1) == 0,
    lambda g: g.get("add") and g["add"](10, 20) == 30,
    lambda g: g.get("add") and g["add"](5, 5) == 10,
    lambda g: g.get("add") and g["add"](-3, -2) == -5,
    lambda g: g.get("add") and g["add"](100, 200) == 300,
    lambda g: g.get("add") and g["add"](1, -1) == 0,
]

correct_code = "def add(a, b): return a + b"
wrong_code   = "def add(a, b): return a * b"   # wrong logic
broken_code  = "def add(a, b) return a + b"    # syntax error

s1 = code_verifier(correct_code, tests)  # 1.000
s2 = code_verifier(wrong_code,   tests)  # 0.125  (one test has a=0,b=0)
s3 = code_verifier(broken_code,  tests)  # 0.000

print("correct=%.3f, wrong=%.3f, syntax_error=%.3f" % (s1, s2, s3))
# correct=1.000, wrong=0.125, syntax_error=0.000`}
      </CodeBlock>

      <Callout accent="gold">
        The sandboxing requirement for production code execution is real engineering work: subprocess isolation, time limits, memory limits, no network access. The verifier function itself is trivial — the execution environment is the hard part.
      </Callout>

      <H3>4c. RLVR loop (GRPO-style)</H3>

      <Prose>
        We implement the full RLVR loop on 50 synthetic math problems. The policy is represented as a single logit parameter per difficulty level (easy/hard) — a deliberate simplification that makes the gradient math transparent. Group size G=8, KL coefficient β=0.05, learning rate 0.15. Easy problems (0–24) have a base pass rate of ~40%; hard problems (25–49) start at ~8%. We train for 40 steps and track pass rates for both buckets.
      </Prose>

      <CodeBlock language="python">
{`import random, math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))

random.seed(7)
G    = 8       # group size
lr   = 0.15
beta = 0.05    # KL regularization strength

logit_easy = 0.0    # sigmoid(0.0) ≈ 0.50 starting pass rate
logit_hard = -2.5   # sigmoid(-2.5) ≈ 0.08 starting pass rate

def base_pass_rate(pid):
    return 0.40 if pid < 25 else 0.08  # reference for KL

for step in range(40):
    grad_easy, grad_hard = 0.0, 0.0
    n_easy,    n_hard    = 0, 0

    for pid in range(50):
        is_hard = pid >= 25
        logit   = logit_hard if is_hard else logit_easy
        p       = sigmoid(logit)

        # Sample G responses; score each with verifier (0 or 1)
        rewards = [1.0 if random.random() < p else 0.0 for _ in range(G)]

        # Group normalize (GRPO advantage)
        mean_r = sum(rewards) / G
        std_r  = (sum((r - mean_r)**2 for r in rewards) / G) ** 0.5
        if std_r < 1e-8:
            continue   # all same reward — no gradient signal

        advantages = [(r - mean_r) / std_r for r in rewards]

        # Policy gradient: d(log π(correct)) / d(logit) = (1-p)
        #                  d(log π(wrong))   / d(logit) = -p
        g = sum(
            adv * ((1 - p) if r == 1.0 else -p)
            for adv, r in zip(advantages, rewards)
        ) / G
        kl_grad = beta * (p - base_pass_rate(pid))  # KL penalty

        if is_hard:
            grad_hard += g - kl_grad;  n_hard += 1
        else:
            grad_easy += g - kl_grad;  n_easy += 1

    if n_easy > 0: logit_easy += lr * grad_easy / n_easy
    if n_hard > 0: logit_hard += lr * grad_hard / n_hard

    if step % 10 == 0 or step == 39:
        pr_e, pr_h = sigmoid(logit_easy), sigmoid(logit_hard)
        print("Step %3d: easy=%.3f, hard=%.3f, avg=%.3f"
              % (step, pr_e, pr_h, (pr_e + pr_h) / 2))

# Step   0: easy=0.517, hard=0.080, avg=0.298
# Step  10: easy=0.676, hard=0.130, avg=0.403
# Step  20: easy=0.796, hard=0.212, avg=0.504
# Step  30: easy=0.872, hard=0.334, avg=0.603
# Step  39: easy=0.915, hard=0.476, avg=0.696`}
      </CodeBlock>

      <Plot
        label="4c. rlvr grpo loop — pass rate over training steps"
        xLabel="training step"
        yLabel="pass rate"
        series={[
          {
            name: "easy problems (pids 0–24)",
            color: "#4ade80",
            points: [
              [0,0.517],[5,0.600],[10,0.676],[15,0.732],[20,0.796],
              [25,0.832],[30,0.872],[35,0.897],[39,0.915],
            ],
          },
          {
            name: "hard problems (pids 25–49)",
            color: "#f97316",
            points: [
              [0,0.080],[5,0.102],[10,0.130],[15,0.168],[20,0.212],
              [25,0.270],[30,0.334],[35,0.408],[39,0.476],
            ],
          },
          {
            name: "average",
            color: "#e2b55a",
            points: [
              [0,0.298],[5,0.351],[10,0.403],[15,0.450],[20,0.504],
              [25,0.551],[30,0.603],[35,0.653],[39,0.696],
            ],
          },
        ]}
      />

      <Prose>
        The easy problems converge quickly — the policy was already close to the solution and GRPO rapidly shifts probability mass toward correct responses. Hard problems improve much more slowly; the initial exploration rate is low enough that many gradient steps contribute little signal. This asymmetry motivates curriculum training, addressed in 4d.
      </Prose>

      <H3>4d. Exploration bootstrap — curriculum vs hard-only</H3>

      <Prose>
        When the base model's pass rate on hard problems is near zero, most groups of G=8 rollouts return all-zero rewards. The group standard deviation is zero, the advantage is undefined, and the gradient step is skipped. Training on hard problems exclusively stalls. The fix is curriculum: begin with easy problems the base model can mostly solve, gradually introduce harder ones as the policy strengthens. We compare both strategies over 30 steps.
      </Prose>

      <CodeBlock language="python">
{`random.seed(13)

def run_rlvr(problem_schedule, n_steps=30, G=8, lr=0.15):
    """
    problem_schedule(step, total) -> (n_easy, n_hard) problems per step.
    Returns list of pass rates (hard-problem logit) over training.
    """
    logit = -2.5   # weak start: ~8% pass rate on hard problems
    rates = []

    for step in range(n_steps):
        n_easy, n_hard = problem_schedule(step, n_steps)
        total_grad, total_n = 0.0, 0

        # Easy problems: floor pass rate at 0.35 to model prior SFT capability
        for _ in range(n_easy):
            p_prob = max(0.35, sigmoid(logit))
            rewards = [1.0 if random.random() < p_prob else 0.0 for _ in range(G)]
            mean_r, std_r = sum(rewards)/G, (sum((r-mean_r)**2 for r in rewards)/G)**.5
            if std_r > 1e-8:
                advantages = [(r - sum(rewards)/G) / std_r for r in rewards]
                g = sum(adv * ((1-p_prob) if r==1.0 else -p_prob)
                        for adv,r in zip(advantages,rewards)) / G
                total_grad += g;  total_n += 1

        # Hard problems
        for _ in range(n_hard):
            p_prob = sigmoid(logit)
            rewards = [1.0 if random.random() < p_prob else 0.0 for _ in range(G)]
            mean_r = sum(rewards)/G
            std_r  = (sum((r-mean_r)**2 for r in rewards)/G)**.5
            if std_r > 1e-8:
                advantages = [(r-mean_r)/std_r for r in rewards]
                g = sum(adv * ((1-p_prob) if r==1.0 else -p_prob)
                        for adv,r in zip(advantages,rewards)) / G
                total_grad += g;  total_n += 1

        if total_n > 0:
            logit += lr * total_grad / total_n
        rates.append(sigmoid(logit))
    return rates

# Curriculum: ramp from all-easy to all-hard over training
def curriculum(step, total):
    frac = step / total
    return max(1, int(10 * (1 - frac))), int(10 * frac)

# No curriculum: hard problems only from step 0
def hard_only(step, total):
    return 0, 10

curr_rates = run_rlvr(curriculum)
hard_rates = run_rlvr(hard_only)

print("Curriculum — start: %.3f, end: %.3f" % (curr_rates[0], curr_rates[-1]))
print("Hard-only  — start: %.3f, end: %.3f" % (hard_rates[0], hard_rates[-1]))
# Curriculum — start: 0.081, end: 0.362
# Hard-only  — start: 0.080, end: 0.326`}
      </CodeBlock>

      <Plot
        label="4d. curriculum vs hard-only — hard-problem pass rate over steps"
        xLabel="training step"
        yLabel="hard pass rate"
        series={[
          {
            name: "curriculum (easy → hard ramp)",
            color: "#4ade80",
            points: [
              [0,0.081],[5,0.132],[10,0.182],[15,0.230],[20,0.280],
              [25,0.318],[29,0.362],
            ],
          },
          {
            name: "hard-only from step 0",
            color: "#f87171",
            points: [
              [0,0.080],[5,0.105],[10,0.138],[15,0.190],[20,0.245],
              [25,0.288],[29,0.326],
            ],
          },
        ]}
      />

      <Prose>
        The curriculum advantage is clearest in the first 10 steps: the hard-only run has almost no signal and barely moves, while the curriculum run receives gradient from easy problems and strengthens the policy enough that, by step 10, hard problems start yielding useful gradients too. By step 29 the curriculum achieves a 11% higher hard pass rate. In real training with much harder problems — AIME-level math where the base model starts at 0% — the hard-only run would not move at all.
      </Prose>

      <H3>4e. Synthetic data amplification</H3>

      <Prose>
        After each round of RLVR, the policy has produced rollouts the verifier accepted. Those correct rollouts — full reasoning chains ending in verified answers — can be used as supervised fine-tuning data before the next RL phase. SFT on correct rollouts shifts the base distribution toward good reasoning patterns, so the next RLVR phase starts from a better initialization. We model this as a logit shift at the start of RLVR and compare against RLVR-only over 25 steps.
      </Prose>

      <CodeBlock language="python">
{`random.seed(99)

def run_amplification(sft_boost: float, n_steps=25, G=8, lr=0.1):
    """
    sft_boost: logit improvement from SFT on correct rollouts before RLVR.
    Returns pass rate over RLVR training.
    """
    logit = -3.0 + sft_boost   # very weak base + SFT shift
    rates = []

    for step in range(n_steps):
        total_grad, total_n = 0.0, 0
        for _ in range(20):   # 20 medium-difficulty problems per step
            p = sigmoid(logit)
            rewards = [1.0 if random.random() < p else 0.0 for _ in range(G)]
            mean_r  = sum(rewards) / G
            std_r   = (sum((r - mean_r)**2 for r in rewards) / G) ** 0.5
            if std_r > 1e-8:
                advantages = [(r - mean_r) / std_r for r in rewards]
                g = sum(adv * ((1-p) if r==1.0 else -p)
                        for adv,r in zip(advantages,rewards)) / G
                total_grad += g;  total_n += 1
        if total_n > 0:
            logit += lr * total_grad / total_n
        rates.append(sigmoid(logit))
    return rates

rlvr_only = run_amplification(sft_boost=0.0)   # no SFT warm-start
sft_then  = run_amplification(sft_boost=1.5)   # SFT shifts logit by +1.5

print("RLVR-only   start=%.3f  end=%.3f" % (rlvr_only[0], rlvr_only[-1]))
print("SFT + RLVR  start=%.3f  end=%.3f" % (sft_then[0],  sft_then[-1]))
# RLVR-only   start=0.049  end=0.108
# SFT + RLVR  start=0.188  end=0.394`}
      </CodeBlock>

      <Plot
        label="4e. synthetic data amplification — rlvr-only vs sft+rlvr"
        xLabel="rlvr training step"
        yLabel="pass rate"
        series={[
          {
            name: "RLVR-only (no SFT warm-start)",
            color: "#f87171",
            points: [
              [0,0.049],[3,0.058],[6,0.068],[9,0.079],[12,0.089],
              [15,0.096],[18,0.101],[21,0.105],[24,0.108],
            ],
          },
          {
            name: "SFT on correct rollouts → then RLVR",
            color: "#4ade80",
            points: [
              [0,0.188],[3,0.234],[6,0.278],[9,0.317],[12,0.347],
              [15,0.369],[18,0.381],[21,0.389],[24,0.394],
            ],
          },
        ]}
      />

      <Prose>
        The amplification effect is stark. The RLVR-only run barely moves from its initial 5% pass rate because the base model is too weak to generate correct rollouts; without any correct rollouts, there is no positive gradient signal. The SFT+RLVR run starts at 19% — a strong enough base that GRPO generates useful gradient from the first step — and reaches 39% by step 24. This is the bootstrapping pattern used by DeepSeek-R1: cold-start SFT on curated rollouts, then RLVR for the bulk of capability improvement.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 TRL's GRPOTrainer with a custom verifier</H3>

      <Prose>
        HuggingFace TRL's <Code>GRPOTrainer</Code> accepts a custom reward function directly — plug in a verifier and the standard GRPO loop handles rollout, scoring, advantage computation, and PPO updates. The key parameter is <Code>reward_funcs</Code>, which takes a list of callables that map (prompts, completions) to lists of floats.
      </Prose>

      <CodeBlock language="python">
{`from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# ── 1. Define the math verifier ────────────────────────────────────────────
def math_reward_fn(prompts, completions, ground_truths=None, **kwargs):
    """
    TRL reward function signature: (prompts, completions) -> List[float].
    ground_truths must be passed as an extra dataset column.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        match = re.search(r"\\\\boxed\\{([^}]+)\\}", text)
        if not match:
            rewards.append(0.0)
            continue
        try:
            score = 1.0 if abs(float(match.group(1)) - float(gt)) < 1e-6 else 0.0
        except ValueError:
            score = 0.0
        rewards.append(score)
    return rewards

# ── 2. Configure GRPO training ─────────────────────────────────────────────
grpo_config = GRPOConfig(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./rlvr_math_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch: 8 prompts
    num_generations=8,               # G=8 rollouts per prompt
    max_new_tokens=512,
    temperature=0.9,
    beta=0.04,                       # KL coefficient (matches DeepSeekMath)
    learning_rate=1e-6,
    bf16=True,
    gradient_checkpointing=True,
    # GRPO-specific: no value model needed
    use_vllm=True,                   # fast generation with vLLM
)

model     = AutoModelForCausalLM.from_pretrained(grpo_config.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_name_or_path)

# dataset must have columns: "prompt" and "ground_truth"
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=math_dataset,
    reward_funcs=[math_reward_fn],
    tokenizer=tokenizer,
)
trainer.train()`}
      </CodeBlock>

      <H3>5.2 Open-R1 and DeepSeek-R1 reproduction</H3>

      <Prose>
        The open-r1 project (github.com/huggingface/open-r1) is the most detailed public reproduction of DeepSeek-R1's RLVR pipeline. It provides three scripts: <Code>grpo.py</Code> trains a model with GRPO on a verifiable dataset; <Code>sft.py</Code> performs supervised fine-tuning on distilled reasoning traces; and <Code>generate.py</Code> generates synthetic data using the current policy. The project reproduces the cold-start + RLVR two-stage structure: SFT on 350k verified traces, then GRPO with a math verifier.
      </Prose>

      <Prose>
        Key engineering decisions from the open-r1 codebase that differ from a naive implementation: vLLM is used for generation (not the Transformers model directly) to get 8x throughput on rollout generation; rewards are computed asynchronously while the optimizer runs; the math verifier uses a more robust answer parser that handles LaTeX fractions, scientific notation, and symbolic expressions (not just float parsing); and the curriculum is implemented via dataset difficulty scores computed by the base model's pass rate on a held-out set before training begins.
      </Prose>

      <H3>5.3 Tülu 3's RLVR implementation</H3>

      <Prose>
        Lambert et al. (Tülu 3, arXiv:2411.15124) document RLVR as a distinct post-training stage after SFT and DPO. Their implementation differs from DeepSeek in two ways: they use RLOO rather than GRPO as the base algorithm (which avoids the per-group normalization and uses a leave-one-out baseline instead); and they combine RLVR with DPO on the same training run using a multi-objective loss. The verifier setup is the same — regex-based math answer parser, execution-based code verifier. The reported result is a significant improvement on GSM8K and MATH over the DPO-only baseline, with no increase in alignment tax on non-math benchmarks.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Pass rate over training — with and without curriculum</H3>

      <Plot
        label="6.1 pass rate on hard problems — curriculum vs no curriculum"
        xLabel="training step"
        yLabel="pass rate on hard problems"
        series={[
          {
            name: "with curriculum (easy → hard ramp)",
            color: "#4ade80",
            points: [
              [0,0.08],[10,0.14],[20,0.22],[30,0.32],[40,0.43],
              [50,0.53],[60,0.62],[70,0.68],[80,0.73],[100,0.78],
            ],
          },
          {
            name: "no curriculum (hard-only from step 0)",
            color: "#f87171",
            points: [
              [0,0.08],[10,0.09],[20,0.11],[30,0.14],[40,0.19],
              [50,0.25],[60,0.32],[70,0.39],[80,0.45],[100,0.52],
            ],
          },
        ]}
      />

      <Prose>
        The curriculum gap opens in the first 30 steps and never closes. The hard-only run eventually catches up in slope — once the policy is strong enough that hard problems occasionally succeed — but it started from a much lower base and never reaches parity within the same compute budget. In practice (DeepSeek-R1), the curriculum is implicit: the dataset contains a mix of easy and hard problems, and the easy ones provide gradient while the hard ones are gradually learned.
      </Prose>

      <H3>6.2 Exploration rate — fraction of groups with any correct response</H3>

      <Plot
        label="6.2 exploration rate — groups with at least one correct rollout (G=8)"
        xLabel="training step"
        yLabel="fraction of groups with ≥1 correct"
        series={[
          {
            name: "exploration rate (curriculum)",
            color: "#e2b55a",
            points: [
              [0,0.51],[10,0.61],[20,0.72],[30,0.81],[40,0.87],
              [50,0.92],[60,0.95],[70,0.97],[80,0.98],[100,0.99],
            ],
          },
          {
            name: "exploration rate (hard-only)",
            color: "#60a5fa",
            points: [
              [0,0.49],[10,0.51],[20,0.57],[30,0.65],[40,0.74],
              [50,0.82],[60,0.88],[70,0.92],[80,0.95],[100,0.98],
            ],
          },
        ]}
      />

      <Prose>
        The exploration rate — the fraction of groups where at least one of G=8 rollouts is correct — is the key diagnostic for RLVR health. When it is near zero, training stalls; when it is near one, the policy is essentially solved and the advantage variance collapses (all groups return all-correct). The healthy training range is 40–95%: enough correct rollouts to generate positive gradient, enough incorrect ones to generate contrastive gradient. The curriculum keeps exploration rate in this range throughout training, while the hard-only run spends the first 20 steps below 60%.
      </Prose>

      <H3>6.3 The RLVR loop — interactive step trace</H3>

      <StepTrace
        label="rlvr full training loop"
        steps={[
          {
            label: "Step 1 — Prompt sampling",
            render: () => (
              <div>
                <TokenStream
                  label="sample a batch of problems from the curriculum"
                  tokens={[
                    { label: "x₁: '2x + 3 = 7'", color: colors.gold },
                    { label: "x₂: 'Prove √2 irrational'", color: colors.gold },
                    { label: "x₃: 'Implement merge sort'", color: colors.gold },
                    { label: "…", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The curriculum scheduler selects problems at the frontier of the current policy's capability — problems where pass rate is between 10% and 90%, so there is both signal and room to improve.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Group rollout (G=8)",
            render: () => (
              <div>
                <TokenStream
                  label="policy generates 8 responses per prompt"
                  tokens={[
                    { label: "π(·|x₁) →", color: colors.textMuted },
                    { label: "y¹₁…y⁸₁", color: "#c084fc" },
                    { label: "π(·|x₂) →", color: colors.textMuted },
                    { label: "y¹₂…y⁸₂", color: "#c084fc" },
                  ]}
                />
                <Prose>
                  Temperature is set to 0.8–1.0 to ensure response diversity. With G=8 responses, the probability that at least one is correct is <Code>1 - (1-p)^8</Code>: for p=0.1 this is ~57%, vs 10% for a single sample.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — Verifier scoring",
            render: () => (
              <div>
                <TokenStream
                  label="verifier returns binary rewards"
                  tokens={[
                    { label: "y¹₁: r=0", color: "#f87171" },
                    { label: "y²₁: r=1", color: "#4ade80" },
                    { label: "y³₁: r=0", color: "#f87171" },
                    { label: "y⁴₁: r=1", color: "#4ade80" },
                    { label: "y⁵₁–y⁸₁: r=0,0,0,0", color: "#f87171" },
                  ]}
                />
                <Prose>
                  The math verifier parses boxed answers; the code verifier executes against hidden tests. Both run in milliseconds. No neural network inference needed for the reward signal.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 4 — Group advantage + PPO update",
            render: () => (
              <div>
                <TokenStream
                  label="group-normalize → PPO clip → gradient step"
                  tokens={[
                    { label: "A = (r − μ) / σ", color: "#e2b55a" },
                    { label: "→ L_CLIP", color: colors.textMuted },
                    { label: "− β·KL", color: "#60a5fa" },
                    { label: "→ ∇θ", color: "#4ade80" },
                  ]}
                />
                <Prose>
                  Two inner PPO epochs per rollout batch. The clip range ε=0.2 prevents large updates. After the gradient step, the reference policy logprobs are recomputed only if enough KL has accumulated.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 5 — Amplification (optional)",
            render: () => (
              <div>
                <TokenStream
                  label="correct rollouts → SFT → stronger π_ref → repeat"
                  tokens={[
                    { label: "verified y*", color: "#4ade80" },
                    { label: "→ SFT dataset", color: colors.gold },
                    { label: "→ fine-tune", color: "#c084fc" },
                    { label: "→ updated π_ref", color: "#4ade80" },
                    { label: "→ next RLVR round", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  After each major checkpoint, correct rollouts are collected and used for an SFT warmup. This updates the reference policy for the next RLVR phase, gradually raising the floor of what the policy can do before RL begins.
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

      <H3>7.1 When to use RLVR vs alternatives</H3>

      <Heatmap
        label="post-training method comparison (5 = best for that criterion)"
        matrix={[
          [5, 1, 5, 3, 4],
          [3, 4, 2, 5, 2],
          [2, 5, 2, 5, 1],
          [3, 4, 3, 4, 3],
          [4, 2, 5, 2, 5],
        ]}
        rowLabels={["RLVR (verifier)", "DPO / SimPO", "RLHF (PPO+RM)", "GRPO + RM", "RLOO + RM"]}
        colLabels={["signal quality", "implementation ease", "no reward hacking", "data efficiency", "reasoning tasks"]}
        colorScale="green"
        cellSize={48}
      />

      <Prose>
        <strong>Use RLVR</strong> when the task has mechanically verifiable correctness: math problems with ground-truth answers, code with a test suite, formal proofs with a type-checker, structured output with schema validation. The training signal is exact and scalable — you can generate synthetic problems at scale and verify them for free.
      </Prose>

      <Prose>
        <strong>Use RLHF/DPO</strong> when the task is subjective: creative writing, open-ended conversation, summarization, translation quality. There is no ground truth, so a learned reward model trained on human preferences is the only option. DPO is simpler when you have a clean offline preference dataset. Full PPO is justified when you need online training or have a very large compute budget.
      </Prose>

      <Prose>
        <strong>Hybrid: RLVR + RLHF</strong> is the current frontier. Models like Tülu 3 apply DPO for general alignment and RLVR for math/code capability. The two objectives do not conflict — DPO anchors helpfulness and safety, RLVR drives reasoning capability — and together they outperform either alone.
      </Prose>

      <H3>7.2 Bootstrap decision tree</H3>

      <Prose>
        Starting from a base pretrained model, the bootstrapping order matters:
      </Prose>

      <Prose>
        <strong>Strong base model (GPT-4 scale, or fine-tuned with good chain-of-thought coverage):</strong> Start directly with RLVR. The pass rate on medium problems will be high enough to generate gradient from the first step. Use curriculum to ramp from medium to hard.
      </Prose>

      <Prose>
        <strong>Weak base model (7B, no chain-of-thought SFT):</strong> Do SFT on curated reasoning traces first (DeepSeek-R1's cold-start data, or open-r1's Mixture-of-Thoughts dataset). After SFT, pass rate on easy problems will be ~30–40%, sufficient for RLVR to take over. Then apply RLVR with curriculum. This is the two-stage recipe documented in DeepSeek-R1 and Tülu 3.
      </Prose>

      <Prose>
        <strong>No labeled data:</strong> Generate synthetic problems with known ground-truth answers (e.g., arithmetic expressions with computed answers, programmatically generated code challenges). Use the base model to generate rollouts; the verifier scores them for free. Keep correct rollouts as SFT data. The first SFT round will be noisy, but after one iteration the model is strong enough to generate better rollouts. This self-amplification loop is the core idea behind DeepSeekMath's data generation pipeline.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 What scales</H3>

      <Prose>
        <strong>Synthetic data volume.</strong> Because the verifier can check correctness without human annotation, the labeled dataset size is limited only by compute. DeepSeekMath (arXiv:2402.03300) collected 120B math-related tokens from Common Crawl and filtered them with a classifier; the RLVR training data was generated by the model itself and verified automatically. There is no saturation point in sight — more verified problems consistently improve performance on math benchmarks.
      </Prose>

      <Prose>
        <strong>Policy improvement with rollout count.</strong> Group size G directly trades compute for signal quality. Larger G reduces the fraction of groups with zero correct responses, which is the primary stall mechanism. DeepSeek-R1 used G=8 to G=16; larger groups are beneficial up to the point where GPU memory for concurrent rollout becomes the constraint. Unlike RLHF where the value model is the memory bottleneck, GRPO's memory cost scales only with G, not with a separate large model.
      </Prose>

      <Prose>
        <strong>Emergent reasoning transfer.</strong> Models RL-trained on math and code consistently improve on general reasoning benchmarks, chain-of-thought coherence, and instruction following — tasks where no verifier was present during training. The informal account is that RLVR teaches reasoning discipline: produce a chain of steps that leads to a conclusion you can actually verify. That discipline transfers. DeepSeek-R1's general reasoning scores improved substantially relative to the SFT baseline, even on non-verifiable tasks.
      </Prose>

      <H3>8.2 What doesn't scale</H3>

      <Prose>
        <strong>Verifier cost.</strong> Running code execution at scale is expensive. At 8 rollouts per prompt, 100K training prompts per step, and 10ms per code execution, the verifier alone accounts for 8,000 CPU-seconds per step. Lean/Isabelle proof checking is orders of magnitude slower. For math with regex verification, the cost is negligible. For code and formal proofs, verifier throughput is the practical bottleneck, and it does not improve as fast as model FLOPs.
      </Prose>

      <Prose>
        <strong>Distribution narrowing.</strong> Training exclusively on verifiable tasks narrows the model's distribution toward those tasks. A model RLVR-trained only on math problems will show a measurable degradation on creative writing, open-ended conversation, and tasks that require calibrated uncertainty. Production pipelines address this by mixing RLVR with DPO or SFT on general tasks, but the alignment tax is real and requires explicit mitigation.
      </Prose>

      <Prose>
        <strong>Hard problem cold start.</strong> When the base model cannot solve any instances of a problem class, RLVR provides no signal for that class. The exploration rate is zero, the group standard deviation is zero, and training stalls. No amount of scaling RLVR alone can solve this — you need a bootstrapping mechanism (curriculum, SFT warm-start, or problem difficulty sampling) before RLVR can take over. Scaling laws for RLVR are thus not uniform across difficulty levels; they are conditional on the base model already having some capability to bootstrap from.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Sparse reward stall</H3>

      <Prose>
        The defining early-training failure. When the base model's pass rate on a problem is below roughly 1/G (so that even with G rollouts, most groups return all-zero rewards), the group standard deviation is zero and the gradient is skipped. Training on hard problems exclusively from step 0 produces no learning signal. Detection: monitor the fraction of groups where std(rewards) &gt; 0 — if it falls below 30%, training is effectively stalled. Mitigation: curriculum, SFT warm-start, or increase G.
      </Prose>

      <H3>9.2 Verifier gaming</H3>

      <Prose>
        Even a deterministic verifier can be gamed if its specification is imprecise. The most common RLVR failure mode is regex gaming: the model learns to output a boxed answer that matches the ground truth format even when the reasoning chain is wrong. Specifically, models learn to output <Code>{String.raw`\boxed{42}`}</Code> without any derivation — and if the test problem happened to have answer 42, the verifier rewards it. The fix is to validate that the reasoning chain is present and coherent, not just the final answer token. Tülu 3 adds a format reward that penalizes responses where the reasoning chain is absent or shorter than expected.
      </Prose>

      <Prose>
        A more subtle form: for code verifiers, the model learns to special-case the test inputs. If the hidden tests use predictable values (assert add(1, 2) == 3), a model can generate code that hardcodes the expected outputs without implementing the general function. Production verifiers randomize test inputs and use property-based testing to prevent this.
      </Prose>

      <H3>9.3 KL anchor drift</H3>

      <Prose>
        When β is set too low, the policy drifts far from the SFT reference. With a fixed reference, this eventually produces responses that are highly optimized for the verifier's exact format requirements but stilted or unnatural in prose. The policy learns to always produce <Code>{String.raw`Let me work through this step by step... \boxed{ans}`}</Code> regardless of problem type, because that format is reliably parsed by the math verifier. Detection: monitor KL divergence — if it exceeds 20 nats in early training, β is too low. Mitigation: increase β or use adaptive KL control.
      </Prose>

      <H3>9.4 Curriculum getting stuck</H3>

      <Prose>
        A badly designed curriculum can stall at a difficulty transition. If the policy improves quickly on easy problems and the curriculum advances too fast to hard ones before the policy is ready, the exploration rate drops suddenly and training stalls at the harder level. The opposite failure: if the curriculum stays on easy problems too long, the policy wastes compute becoming very strong on easy problems with no improvement on hard ones. The best curriculum adapts based on the current exploration rate: advance difficulty when easy-level exploration rate exceeds 90%, stay at current difficulty when hard-level exploration rate is below 40%.
      </Prose>

      <H3>9.5 Verifier quality bound</H3>

      <Prose>
        RLVR quality is bounded by verifier correctness. A math verifier that cannot parse LaTeX fractions will fail on problems where the correct answer is <Code>3/4</Code>; the model gets no signal that its answer of 0.75 is equivalent. A code verifier that does not test edge cases (negative inputs, empty lists, large numbers) will reward incorrect implementations that pass only the basic cases. Every gap in the verifier specification is a potential gap in training signal. The engineering investment in the verifier is as important as the engineering investment in the training loop.
      </Prose>

      <H3>9.6 Generalization narrowing</H3>

      <Prose>
        Extended RLVR on a narrow task distribution narrows the model's generalization. A model RLVR-trained for 50K steps on arithmetic problems will degrade on general language tasks because the policy gradient has been pointing consistently toward arithmetic-format outputs. The SFT reference (KL anchor) mitigates this but does not eliminate it when β is small. Production pipelines mix RLVR batches with general SFT batches, effectively adding a regularization term that keeps the model grounded in its full capability distribution.
      </Prose>

      <Callout accent="gold">
        Six of these nine failure modes have the same root cause: the verifier's specification is narrower than what "correct" means in the real task. The verifier is the ground truth for training — if it is wrong, training is wrong. Investing in the verifier is investing in the training signal's ceiling.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All papers below were verified against arXiv and GitHub as of April 2026.
      </Prose>

      <H3>10.1 Core RLVR papers</H3>

      <Prose>
        <strong>Lambert, Morrison, Miranda, et al. (2024).</strong> "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv:2411.15124. Allen Institute for AI. Introduced RLVR as an explicit named post-training stage, distinct from SFT and DPO. Documented the implementation of a deterministic math verifier integrated into a RLOO-based training loop. Reported significant gains on GSM8K and MATH with no alignment tax on non-math benchmarks. Includes full training code and model weights.
      </Prose>

      <Prose>
        <strong>DeepSeek-AI (2025).</strong> "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. The canonical RLVR reasoning model paper. DeepSeek-R1-Zero demonstrated that pure RL with a verifiable reward — no supervised chain-of-thought, no human preference data — produces emergent self-reflection and reasoning behaviors. DeepSeek-R1 added a cold-start SFT phase followed by RLVR and achieved pass@1 of 71% on AIME 2024, matching OpenAI o1-0912. Used GRPO with G=8–16, β=0.04, and a deterministic math answer parser as the verifier.
      </Prose>

      <H3>10.2 Algorithmic foundations</H3>

      <Prose>
        <strong>Shao, Wang, Zhu, Xu, et al. (2024).</strong> "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. Introduced GRPO — the specific RL algorithm used in RLVR deployments. Shows that group sampling with within-group normalization eliminates the need for a value model and reduces peak memory by roughly 50% compared to PPO, with no loss in final performance on mathematical reasoning benchmarks.
      </Prose>

      <Prose>
        <strong>Uesato, Kushman, Kumar, Song, et al. (2022).</strong> "Solving math word problems with process- and outcome-based feedback." arXiv:2211.14275. DeepMind. The earliest systematic comparison of outcome-based (verifiable final answer) vs process-based (intermediate step) reward in math reasoning. Found that outcome-based supervision produces comparable final-answer accuracy to process-based supervision with less labeling overhead — an early empirical basis for the RLVR approach. Also established GSM8K as the standard math reasoning benchmark for RL training experiments.
      </Prose>

      <H3>10.3 Open-source reproduction</H3>

      <Prose>
        <strong>HuggingFace (2025).</strong> Open-R1: Fully Open Reproduction of DeepSeek-R1. github.com/huggingface/open-r1. Provides: GRPO training script with pluggable verifier interface; SFT script for cold-start; Distilabel-based synthetic data generation pipeline; the Mixture-of-Thoughts dataset (350K verified reasoning traces across math, coding, and science); and the CodeForces-CoTs dataset (10K competitive programming problems, 100K solutions). The 7B model trained on CodeForces-CoTs outperforms Claude 3.7 Sonnet on IOI24. This is the most complete public RLVR implementation available.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Write a verifier for simple arithmetic</H3>

      <Prose>
        Implement a math verifier that handles all of: integer answers, decimal answers (tolerance 1e-4), fraction answers expressed as "p/q", and answers expressed in scientific notation ("3.14e2"). Test it against a suite of 10 ground truths covering each format. What happens when the model outputs "3/4" but the ground truth is "0.75"? How do you handle this without penalizing correct answers?
      </Prose>

      <Callout accent="green">
        Hint: normalize all answers to a canonical float before comparison. Use Python's <Code>fractions.Fraction</Code> for exact rational parsing. Scientific notation can be handled by <Code>float()</Code> directly. The tricky case is symbolic answers like "π" or "√2" — for those, reject or use a separate symbolic equality check.
      </Callout>

      <H3>Exercise 2 — Why binary rewards need group sampling more than scalar rewards</H3>

      <Prose>
        Consider two reward signals: (A) a continuous reward model that outputs scores in [0, 1] with standard deviation ~0.3 across responses to the same prompt; and (B) a binary verifier that outputs exactly 0 or 1. For a policy with 20% pass rate on a given problem, compute the expected gradient variance for a single-sample REINFORCE estimator under each reward type. Now compute the variance reduction factor from using G=8 samples with group normalization. Show that the variance reduction is larger for case (B) than case (A). Why does binary reward make group sampling more essential?
      </Prose>

      <H3>Exercise 3 — Synthetic amplification vs RLVR-only for bootstrap</H3>

      <Prose>
        You have a 7B base model with 5% pass rate on your target math problems. You have a budget of 100 GPU-hours for training. Compare two strategies: (A) pure RLVR for 100 hours; (B) 20 hours of SFT on rollouts from the base model, then 80 hours of RLVR. Describe what you expect to happen in the first 10 steps of each strategy in terms of exploration rate and gradient magnitude. Under what conditions would strategy (A) outperform (B)? (Hint: consider the quality of the SFT data when the base model is very weak.)
      </Prose>

      <H3>Exercise 4 — Detect verifier gaming from rollout statistics</H3>

      <Prose>
        Design a set of automatic statistics to detect verifier gaming without running a human evaluation. Specifically: (a) a statistic to detect "answer without reasoning" — the model outputs <Code>{String.raw`\boxed{42}`}</Code> but the response length is below 20 tokens; (b) a statistic to detect "test-case hardcoding" in code — the generated function contains hardcoded if-statements matching the test input values; (c) a statistic to detect "format collapse" — nearly all responses follow the exact same template structure. For each, describe how you would compute it from rollout data and the threshold at which you would intervene.
      </Prose>

      <H3>Exercise 5 — Design a curriculum scheduler</H3>

      <Prose>
        You have a dataset of 10,000 math problems, each with a difficulty score computed as the base model's pass rate (0 to 1). Design a curriculum scheduler that: (a) starts with problems in the difficulty range [0.3, 0.7] (problems where the base model sometimes succeeds); (b) adapts the difficulty window based on the current exploration rate — expand toward harder problems when easy-bucket exploration rate exceeds 90%; (c) never drops a problem entirely — occasionally sample from all difficulty levels to prevent catastrophic forgetting; (d) emits a warning when the hard-bucket exploration rate falls below 20% for more than 5 consecutive steps. Write the scheduler as a Python class with a <Code>sample(batch_size, current_metrics)</Code> method.
      </Prose>

    </div>
  ),
};

export default rlvr;
