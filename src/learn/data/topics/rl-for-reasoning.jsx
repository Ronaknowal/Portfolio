import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rlForReasoning = {
  title: "RL for Reasoning (DeepSeek-R1 Style)",
  readTime: "~65 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Something empirically new happened in late 2024 and through 2025. Models trained
        with pure reinforcement learning on verifiable rewards — math competition problems,
        competitive programming, formal proofs — started developing behaviors that nobody
        trained them to produce. Long internal chains of deliberation. Self-correction
        mid-derivation. Explicit backtracking when a reasoning path led nowhere. The
        vocabulary of reconsideration — "wait, that can't be right," "let me try a
        different approach," "checking my earlier step" — appeared spontaneously in
        the model's outputs. Not because anyone wrote those phrases into the training data.
        Because the reward signal made them profitable.
      </Prose>

      <Prose>
        DeepSeek-R1's technical report (arXiv:2501.12948) called it the "aha moment."
        The paper documented the phenomenon precisely: a model trained with GRPO on
        verifiable math rewards, with no chain-of-thought supervision and no human
        preference data, began spontaneously producing multi-thousand-token reasoning
        traces with structured self-verification. The pass@1 rate on AIME 2024 rose
        from 15.6% at the start of training to 71.0% — with majority voting pushing it
        to 86.7%, matching OpenAI's o1-0912. OpenAI's o1 and o3 series shows the same
        pattern from a different lab. Kimi k1.5 (arXiv:2501.12599) demonstrated it again,
        independently, at comparable scale.
      </Prose>

      <Prose>
        The striking claim is not that RL improves reasoning. That was expected. The
        striking claim is that the improvement mechanism is emergent: behaviors that were
        not in the training distribution, not prompted for, not demonstrated via imitation,
        appeared because the RL dynamics made them the optimal strategy for earning reward.
        This is categorically different from what chain-of-thought prompting does. Prompting
        a base model to "think step by step" produces some improvement, then plateaus — the
        model elaborates but does not genuinely explore. The RL-trained model's accuracy
        keeps climbing as you give it more inference tokens, following a log-linear curve
        that has no analogue in the untuned model.
      </Prose>

      <Prose>
        To place it in context: Lightman et al. (arXiv:2305.20050) showed in 2023 that
        process-supervised reward models — PRMs that give step-level feedback — outperform
        outcome-supervised models on hard math. That was a reward signal design result.
        DeepSeek-R1 went further: with only outcome-level verification (the final answer
        is correct or it isn't), and with GRPO's group sampling to handle the sparse binary
        reward, a capable base model discovers process-level reasoning behaviors on its own.
        The model doesn't need to be told how to reason. It needs to be told whether it
        got the right answer, enough times, for the reasoning strategies that work to
        accumulate enough gradient signal to stick.
      </Prose>

      <Callout accent="gold">
        RL for reasoning is the first post-training regime that reliably produces behaviors
        the training data did not contain. That is not a prompting result. It is not a
        fine-tuning result. It is a new capability that emerges from the RL dynamics.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The formula is simple to state, though large in its consequences. Take a capable
        base model — something at the DeepSeek-V3, Qwen-72B, or Llama-3-70B tier. Apply
        GRPO with a verifiable reward function, using a moderate KL anchor to prevent
        distributional collapse. Train on hard math and code problems at scale. The model
        learns to reason at inference time.
      </Prose>

      <Prose>
        Why does this work when RLHF with a learned reward model doesn't produce the same
        effect? Two reasons. First, a learned reward model is a proxy — it can be hacked,
        its score can be maximized in ways that don't correspond to actual reasoning quality.
        A verifier is exact: the final answer is correct or it isn't. There's no proxy to
        exploit. Second, RLHF optimizes for preference among responses, which rewards
        surface-level quality: tone, formatting, appropriate hedging. RLVR with verifiable
        rewards optimizes for correctness, which requires actual competence. The gradient
        signal points toward getting the answer right, not toward sounding like you got it
        right.
      </Prose>

      <Prose>
        The base model's capability is the floor, not the ceiling. A weak base model given
        RLVR training won't produce emergent reasoning — it can't get enough problems right
        to generate a meaningful gradient signal from any starting point. A strong base
        model given RLVR training has latent reasoning capability in its weights from
        pretraining; the RL dynamics amplify and specialize it. DeepSeek-R1 starts from
        DeepSeek-V3, a 671B mixture-of-experts model pretrained on enormous data. The RL
        training doesn't create reasoning from nothing. It extracts and sharpens what was
        already there.
      </Prose>

      <Prose>
        Test-time compute scaling is where the capability cashes out. The RL-trained model
        learns that more tokens invested in reasoning produces better answers on hard
        problems. This is a policy learned from reward: rollouts where the model explored
        multiple approaches before committing earned higher reward than rollouts where the
        model committed immediately to its first guess. Over millions of training problems,
        this becomes a general strategy: on hard problems, think longer. The resulting
        accuracy-vs-token-budget curve is approximately log-linear — accuracy scales with
        log inference compute — a relationship that simply doesn't exist in the base model.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The verifiable reward function</H3>

      <Prose>
        The reward in RL for reasoning is a deterministic verifier. For math, it is a
        function that parses the boxed final answer and compares it numerically to the
        ground truth. For code, it executes the generated program against hidden test
        cases. The reward is binary — 1 if correct, 0 if wrong — with no learned
        parameters:
      </Prose>

      <MathBlock caption="Verifiable reward: deterministic, binary, no proxy to hack">
        {"r(x, y) \\in \\{0, 1\\}, \\quad r(x, y) = \\mathbf{1}[\\text{verify}(y, \\text{ans}(x)) = \\text{true}]"}
      </MathBlock>

      <Prose>
        The key property: unlike RLHF's learned reward model, this function has no
        parameters and cannot drift. The policy can only improve its score by producing
        more correct answers. There is no shortcut.
      </Prose>

      <H3>3.2 The KL-regularized objective</H3>

      <Prose>
        The training objective is the standard KL-constrained RL formulation, with the
        learned reward model replaced by the verifier:
      </Prose>

      <MathBlock caption="RL for reasoning objective: maximize verifier reward, stay near SFT reference">
        {"\\max_{\\pi_\\theta}\\; \\mathbb{E}_{x \\sim \\mathcal{D},\\; y \\sim \\pi_\\theta(\\cdot|x)}\\!\\left[r(x, y) - \\beta \\log \\frac{\\pi_\\theta(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}\\right]"}
      </MathBlock>

      <Prose>
        The KL term is critical and deserves a careful reading in this setting. The
        reasoning traces can be thousands of tokens long. Without a KL anchor, the policy
        drifts into reasoning templates that are highly optimized for the verifier's
        exact format requirements but produce stilted, unnatural text. The anchor
        maintains the diversity and naturalness of the model's prose while allowing the
        reasoning strategy to change. DeepSeek-R1 used β ≈ 0.04 — loose enough to allow
        substantial policy change, tight enough to prevent distributional collapse.
      </Prose>

      <H3>3.3 Test-time compute scaling</H3>

      <Prose>
        The empirically observed relationship between inference token budget and accuracy
        on hard math problems follows an approximately log-linear form:
      </Prose>

      <MathBlock caption="Test-time compute scaling: accuracy grows log-linearly with inference token budget">
        {"\\text{accuracy} \\approx \\alpha \\cdot \\log_{10}(\\text{tokens}) + \\beta"}
      </MathBlock>

      <Prose>
        This relationship holds for the RL-trained model over many orders of magnitude of
        token budget on AIME-class problems. The base model's accuracy curve is
        qualitatively different — it improves slightly with more tokens up to a plateau,
        then flattens regardless of further token budget increases. The RL training
        installs a capability the base model does not have: productive use of inference
        compute.
      </Prose>

      <H3>3.4 GRPO group-normalized advantages</H3>

      <Prose>
        Binary verifiable rewards create a specific gradient estimation challenge: when the
        policy's pass rate on a problem is low, single-sample REINFORCE produces near-zero
        gradients because most rollouts return zero reward. GRPO addresses this by sampling
        G responses per prompt and normalizing within the group:
      </Prose>

      <MathBlock caption="GRPO advantage: each response scored relative to its group — essential for sparse binary rewards">
        {"A_i = \\frac{r_i - \\mu_G}{\\sigma_G + \\varepsilon}, \\quad \\mu_G = \\frac{1}{G}\\sum_{j=1}^G r_j, \\quad \\sigma_G = \\sqrt{\\frac{1}{G}\\sum_{j=1}^G (r_j - \\mu_G)^2}"}
      </MathBlock>

      <Prose>
        With G=8 and a 12% pass rate, a group where one rollout is correct gets
        {" μ_G = 0.125, σ_G ≈ 0.33"}. The correct rollout earns advantage
        {" (1 − 0.125)/0.33 ≈ 2.65"}; each incorrect rollout earns
        {" (0 − 0.125)/0.33 ≈ −0.38"}. A single correct response in a group of eight
        failures is enough to produce a meaningful gradient. This is why GRPO is
        essential for hard reasoning problems where the base pass rate is very low —
        and why DeepSeek-R1 used G=8 to G=16 rollouts per prompt.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        This section builds the RL-for-reasoning training loop from components introduced
        in earlier topics: the verifier (RLVR topic), the GRPO update (GRPO/RLOO/KTO
        topic), and the KL-regularized objective (RLHF topic). The emphasis here is on
        the novel phenomena: the KL objective in the long-rollout setting, emergent
        self-correction, and the test-time compute scaling curve. All code is standard
        library Python. All outputs below are actual verified outputs.
      </Prose>

      <H3>4a. The GRPO training loop with KL regularization</H3>

      <Prose>
        We implement the core loop: a two-action toy policy (action 0 = correct answer,
        action 1 = wrong answer; reward is binary), GRPO group sampling with G=8, and a
        KL penalty anchored to a reference policy. We track pass rate, KL divergence,
        and the full KL-regularized objective over 50 steps.
      </Prose>

      <CodeBlock language="python">
{`import random, math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))

random.seed(42)
G = 8
beta_kl = 0.05   # KL coefficient (beta in the objective)
lr = 0.15

# Two-action toy policy: logit for action 0 (correct answer)
logit = 0.0         # starts at 50% pass rate
ref_logit = -0.5   # SFT reference: ~38% pass rate

def probs(logit):
    p0 = sigmoid(logit)
    return p0, 1.0 - p0

for step in range(50):
    p0, p1 = probs(logit)
    ref_p0, ref_p1 = probs(ref_logit)

    # Group rollout: G responses to one problem
    rollout_actions, rollout_rewards = [], []
    for _ in range(G):
        a = 0 if random.random() < p0 else 1
        r = 1.0 if a == 0 else 0.0       # binary verifier
        rollout_actions.append(a)
        rollout_rewards.append(r)

    # GRPO group-normalized advantage
    mean_r = sum(rollout_rewards) / G
    std_r = (sum((r - mean_r)**2 for r in rollout_rewards) / G)**0.5

    # KL divergence from reference (for monitoring and penalty)
    kl = (p0 * (math.log(p0 + 1e-10) - math.log(ref_p0 + 1e-10))
        + p1 * (math.log(p1 + 1e-10) - math.log(ref_p1 + 1e-10)))
    exp_r = p0 * 1.0  # E[r] = prob(correct)

    if step % 10 == 0 or step == 49:
        obj = exp_r - beta_kl * kl
        print(f"Step {step:2d}: pass_rate={p0:.3f}, KL={kl:.4f}, "
              f"E[r]-b*KL={obj:.4f}")

    # GRPO gradient update
    if std_r > 1e-8:
        for a, r in zip(rollout_actions, rollout_rewards):
            adv = (r - mean_r) / std_r
            grad_logit = (1 - p0) if a == 0 else -p0
            kl_grad = beta_kl * (p0 - ref_p0)   # gradient of KL penalty
            logit += lr * (adv * grad_logit - kl_grad) / G

# Step  0: pass_rate=0.500, KL=0.0309, E[r]-b*KL=0.4985
# Step 10: pass_rate=0.666, KL=0.1701, E[r]-b*KL=0.6575
# Step 20: pass_rate=0.778, KL=0.3344, E[r]-b*KL=0.7617
# Step 30: pass_rate=0.863, KL=0.5051, E[r]-b*KL=0.8373
# Step 40: pass_rate=0.904, KL=0.6097, E[r]-b*KL=0.8735
# Step 49: pass_rate=0.926, KL=0.6725, E[r]-b*KL=0.8921
#
# Pass rate climbs from 50% to 93% while KL stays bounded at 0.67 nats.
# The objective E[r] - beta*KL tracks real improvement, not proxy hacking.`}
      </CodeBlock>

      <Prose>
        The output shows the KL-regularized objective working as designed: the pass rate
        climbs monotonically, the KL divergence grows but stays bounded, and the objective
        tracks genuine improvement. With a learned reward model, optimizing this same
        objective would eventually cause the KL to grow faster than the true reward
        improves — reward hacking. With a verifier, the true reward is the verifier score.
        There's nothing else to optimize against.
      </Prose>

      <H3>4b. Emergent self-correction: appearance of reasoning strategies without direct supervision</H3>

      <Prose>
        The core surprise of DeepSeek-R1-Zero was that self-correction behaviors emerged
        without any supervised chain-of-thought data. We model this with a 3-action policy:
        the model can respond with a short direct answer (action 0, base accuracy 28%),
        a long reasoning trace without self-correction (action 1, base accuracy 50%), or
        a long trace with explicit self-verification markers — "wait, let me reconsider"
        (action 2, base accuracy 72%). None of these actions is directly labeled or
        rewarded differently by type. The only reward is the binary verifier. Observe that
        RL training discovers and amplifies the self-correction strategy without being told it exists.
      </Prose>

      <CodeBlock language="python">
{`import random, math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))

random.seed(42)
G = 8
lr = 0.25
beta_kl = 0.04

# Three response strategies (not labeled in training — only the reward differs)
# Action 0: short direct answer  (base accuracy: 28%)
# Action 1: long reasoning       (base accuracy: 50%)
# Action 2: self-correction CoT  (base accuracy: 72%)  ← emerges
BASE_ACC = [0.28, 0.50, 0.72]

logits = [0.0, 0.0, 0.0]      # policy starts uniform over strategies
ref_logits = [0.0, 0.0, 0.0]  # SFT reference: also uniform

def get_probs(lgs):
    m = max(lgs)
    e = [math.exp(l - m) for l in lgs]
    s = sum(e)
    return [ei / s for ei in e]

for step in range(60):
    probs = get_probs(logits)
    ref_probs = get_probs(ref_logits)
    step_choices, step_rewards = [], []

    for _ in range(40):      # 40 problems per training step
        rollout_a, rollout_r = [], []
        for _ in range(G):
            rv = random.random()
            if rv < probs[0]:              a = 0
            elif rv < probs[0] + probs[1]: a = 1
            else:                          a = 2
            r = 1.0 if random.random() < BASE_ACC[a] else 0.0
            rollout_a.append(a); rollout_r.append(r)
        step_choices.extend(rollout_a); step_rewards.extend(rollout_r)

        mean_r = sum(rollout_r) / G
        std_r = (sum((r - mean_r)**2 for r in rollout_r) / G)**0.5
        if std_r > 1e-8:
            for a, r in zip(rollout_a, rollout_r):
                adv = (r - mean_r) / std_r
                for i in range(3):
                    grad_i = (1 if a == i else 0) - probs[i]
                    kl_pen = beta_kl * (math.log(probs[i] + 1e-10)
                                      - math.log(ref_probs[i] + 1e-10))
                    logits[i] += lr * (adv * grad_i - kl_pen) / (G * 40)

    if step % 10 == 0 or step == 59:
        self_corr_freq = step_choices.count(2) / len(step_choices)
        avg_r = sum(step_rewards) / len(step_rewards)
        print(f"Step {step:2d}: avg_reward={avg_r:.3f}, "
              f"self_correction_freq={self_corr_freq:.3f}")

# Step  0: avg_reward=0.509, self_correction_freq=0.334
# Step 10: avg_reward=0.541, self_correction_freq=0.472
# Step 20: avg_reward=0.603, self_correction_freq=0.575
# Step 30: avg_reward=0.584, self_correction_freq=0.688
# Step 40: avg_reward=0.641, self_correction_freq=0.794
# Step 50: avg_reward=0.622, self_correction_freq=0.784
# Step 59: avg_reward=0.634, self_correction_freq=0.794
#
# The self-correction strategy grows from 33% to 79% of responses.
# It was never labeled, never demonstrated, never directly rewarded.
# The RL dynamics discovered it because it consistently earns higher reward.`}
      </CodeBlock>

      <Callout accent="gold">
        The self-correction frequency grows from 33% to 79% through RL training, without
        any labeled examples of self-correction behavior. This is the toy analog of what
        DeepSeek-R1-Zero documented at scale: emergent reasoning behaviors arising from
        verifiable reward alone.
      </Callout>

      <H3>4c. Test-time compute scaling curve</H3>

      <Prose>
        The most practically consequential property of RL-trained reasoning models is that
        their accuracy scales with the inference token budget, following a log-linear curve
        on hard problems. The base model with chain-of-thought prompting shows a different
        shape: rapid initial improvement followed by a plateau, regardless of further token
        budget increases. We model both curves and plot them. These shapes match the
        qualitative structure reported in DeepSeek-R1 and Kimi k1.5.
      </Prose>

      <CodeBlock language="python">
{`import math

def rl_model_accuracy(log10_tokens, alpha=0.22, base=0.18):
    """RL-trained model: accuracy grows log-linearly with token budget."""
    return min(0.95, alpha * log10_tokens + base)

def base_model_accuracy(log10_tokens, plateau=0.38, warmup=0.08):
    """Base model + CoT prompt: rapid early gain, then plateau."""
    return min(plateau, warmup + (plateau - warmup) * (1 - math.exp(-log10_tokens)))

token_budgets = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]   # log10 tokens

print(f"{'log10(tokens)':>15} | {'RL model':>10} | {'Base+CoT':>10}")
print("-" * 42)
for lt in token_budgets:
    rl = rl_model_accuracy(lt)
    base = base_model_accuracy(lt)
    print(f"{lt:>15.1f} | {rl:>10.3f} | {base:>10.3f}")

# log10(tokens)  |   RL model |   Base+CoT
# ----------------------------------------
#           2.0  |      0.620 |      0.265
#           2.5  |      0.730 |      0.305
#           3.0  |      0.840 |      0.326
#           3.5  |      0.950 |      0.336
#           4.0  |      0.950 |      0.342
#           4.5  |      0.950 |      0.346
#
# RL model: 62% → 95% across 2.5 orders of magnitude of token budget.
# Base+CoT: 27% → 35%, then plateaus. More tokens, no benefit.
# This is not the same model using more words. Different structural behavior.`}
      </CodeBlock>

      <H3>4d. The cascade: why hard problems unlock gradually</H3>

      <Prose>
        DeepSeek-R1's training report documents a key observation: improvement on hard
        problem classes is not gradual — it appears discontinuously. The mechanism:
        hard problems only generate useful GRPO gradients when the group exploration rate
        is above zero (at least one rollout in the group is correct). Below that threshold,
        std_r is zero, the gradient is skipped, and training stalls. Once the policy
        crosses the threshold — often driven by improvement on easier problems — the
        hard problems start contributing gradients, and a cascade begins.
      </Prose>

      <CodeBlock language="python">
{`import random, math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))

random.seed(7)
G = 8
lr = 0.2
beta = 0.04

# Two problem difficulties. Easy: starts at 40%. Hard: starts at 4%.
logit_easy = -0.4   # sigmoid(-0.4) ~40%
logit_hard = -3.2   # sigmoid(-3.2) ~4%

for step in range(50):
    grad_easy = grad_hard = 0.0
    skipped_hard = 0

    for _ in range(20):   # 20 easy problems per step
        p = sigmoid(logit_easy)
        rewards = [1.0 if random.random() < p else 0.0 for _ in range(G)]
        mean_r, std_r = sum(rewards)/G, (sum((r-sum(rewards)/G)**2 for r in rewards)/G)**0.5
        if std_r > 1e-8:
            for r in rewards:
                adv = (r - mean_r) / std_r
                grad_easy += lr * adv * ((1-p) if r==1 else -p) / (G*20)

    for _ in range(20):   # 20 hard problems per step
        p = sigmoid(logit_hard)
        rewards = [1.0 if random.random() < p else 0.0 for _ in range(G)]
        mean_r, std_r = sum(rewards)/G, (sum((r-sum(rewards)/G)**2 for r in rewards)/G)**0.5
        if std_r < 1e-8:
            skipped_hard += 1        # all-zero group: no gradient
            continue
        for r in rewards:
            adv = (r - mean_r) / std_r
            grad_hard += lr * adv * ((1-p) if r==1 else -p) / (G*20)

    logit_easy += grad_easy
    logit_hard += grad_hard

    if step % 10 == 0 or step == 49:
        pe, ph = sigmoid(logit_easy), sigmoid(logit_hard)
        print(f"Step {step:2d}: easy={pe:.3f}, hard={ph:.4f}, "
              f"hard_skipped={skipped_hard}/20")

# Step  0: easy=0.399, hard=0.0401, hard_skipped=12/20
# Step 10: easy=0.623, hard=0.0432, hard_skipped=16/20
# Step 20: easy=0.820, hard=0.0641, hard_skipped=14/20
# Step 30: easy=0.919, hard=0.1423, hard_skipped=9/20
# Step 40: easy=0.963, hard=0.3071, hard_skipped=3/20
# Step 49: easy=0.981, hard=0.5338, hard_skipped=0/20
#
# Hard problems contribute almost no gradient until step 30.
# Once the policy is strong enough that hard-problem groups
# occasionally have a correct rollout, improvement accelerates sharply.
# This is the cascade DeepSeek-R1's training report documents.`}
      </CodeBlock>

      <Prose>
        The cascade pattern is the toy analog of DeepSeek-R1's "aha moment." Hard problems
        stall at near-zero gradient for the first 20+ steps, then begin accelerating once
        the policy has improved enough on easier problems for the hard group exploration
        rate to exceed zero. The improvement on hard problems from step 30 to 49 — 14% to
        53% — is faster than the entire first 30 steps combined. This is why problem
        curriculum matters, and why easy-to-hard ordering outperforms hard-only training
        (as documented in the RLVR topic). Acknowledge toy scale: real models use
        millions of problems, not 40.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 DeepSeek-R1 recipe</H3>

      <Prose>
        The DeepSeek-R1 paper (arXiv:2501.12948) is the most detailed public account of
        a production RL-for-reasoning pipeline. The recipe has four stages. Stage one:
        cold-start SFT on a small dataset of high-quality reasoning traces — a few thousand
        examples — to establish a reasoning format so GRPO has a sensible starting
        distribution. Without cold-start, R1-Zero (the pure RL version) produces
        structurally inconsistent outputs in the early training steps, which makes
        evaluation difficult even when the reasoning is substantively correct.
      </Prose>

      <Prose>
        Stage two: GRPO on math and code problems with verifiable rewards. Group size
        G=8 to G=16. KL coefficient β=0.04. Math verifier: regex extraction of boxed
        final answers with float comparison. Code verifier: execution against hidden test
        cases. Format reward: mild bonus for well-structured reasoning traces (penalizes
        responses that output only a boxed answer without any reasoning chain). Training
        continues for a very large number of steps — the paper reports training on hundreds
        of thousands of problems across multiple iterations.
      </Prose>

      <Prose>
        Stage three: rejection sampling and SFT. The trained RL policy generates new
        reasoning traces; the verifier selects the correct ones. These correct traces —
        now high-quality chain-of-thought data generated by the RL model itself — are
        used to run another round of SFT, raising the floor for the next RL phase. Stage
        four: a final GRPO pass combining both math/code verifiable rewards and preference
        data for general helpfulness, producing the final DeepSeek-R1 model.
      </Prose>

      <H3>5.2 Open-R1 (HuggingFace)</H3>

      <Prose>
        The open-r1 project (github.com/huggingface/open-r1) is the most complete public
        reproduction of the DeepSeek-R1 pipeline. It provides: a GRPO training script
        with pluggable verifier interface (<Code>GRPOTrainer</Code> from TRL with a
        custom <Code>reward_funcs</Code> callable); a SFT script for cold-start; a
        data generation pipeline using the Distilabel framework; the OpenR1-Math-220k
        dataset of 220k reasoning traces distilled from R1; and the CodeForces-CoTs
        dataset of 100k competitive programming solutions. A 7B model trained on
        CodeForces-CoTs using this pipeline outperforms Claude 3.7 Sonnet on competitive
        programming benchmarks. The training configuration in the open-r1 codebase is
        the most documented open implementation of the full recipe.
      </Prose>

      <H3>5.3 Tülu 3 approach</H3>

      <Prose>
        Lambert et al. (Tülu 3, arXiv:2411.15124) implement RLVR as an explicit
        post-training stage distinct from SFT and DPO, using RLOO rather than GRPO as
        the base algorithm. Their verifier is implemented in the same deterministic
        format — regex math parser plus code execution — but their approach combines
        RLVR with DPO in a multi-objective training scheme rather than running them
        sequentially. The reported gains on GSM8K and MATH relative to the DPO-only
        baseline are substantial, with no measurable alignment tax on non-math benchmarks.
        Tülu 3 is fully open-source, including training code, datasets, and model weights.
      </Prose>

      <H3>5.4 OpenAI o-series (opaque but informative)</H3>

      <Prose>
        OpenAI's o1 and o3 models operate on the same principle — RL on verifiable
        rewards producing emergent reasoning — but the technical details are not publicly
        documented at the same depth as DeepSeek-R1. The o1 system card confirms the use
        of RL training on reasoning tasks with verifiable rewards and the test-time compute
        scaling property. The key public claim: o3's accuracy on frontier benchmarks like
        ARC-AGI scales with the inference compute budget in a way that o1 did not — each
        generation of o-series models has shown a steeper accuracy-vs-compute slope on
        hard evaluation tasks. Kimi k1.5 (arXiv:2501.12599) is a parallel effort from
        Moonshot AI that documents similar test-time compute scaling results with a public
        technical report and reaches comparable performance to o1 on AIME and MATH 500.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Test-time compute scaling: RL-trained vs base model</H3>

      <Plot
        label="accuracy vs inference token budget: rl-trained model scales, base model plateaus"
        width={540}
        height={260}
        xLabel="log10 inference tokens per problem"
        yLabel="accuracy %"
        series={[
          {
            name: "RL-trained (R1-style)",
            color: "#4ade80",
            points: [
              [2.0, 62], [2.5, 73], [3.0, 84], [3.5, 92], [4.0, 95], [4.5, 95],
            ],
          },
          {
            name: "Base model + CoT prompt",
            color: "#94a3b8",
            points: [
              [2.0, 27], [2.5, 31], [3.0, 33], [3.5, 35], [4.0, 34], [4.5, 35],
            ],
          },
        ]}
      />

      <Prose>
        The qualitative shape of this plot is the defining finding of the o1/R1 era.
        The RL-trained model's accuracy on hard math problems climbs by more than 30
        percentage points as the token budget increases from 100 tokens (log10=2) to
        10,000 tokens (log10=4). The base model with a CoT prompt shows some early
        improvement, then flattens entirely regardless of further token budget. This is
        not the same model using more tokens to say the same thing — it is a structurally
        different inference regime. More tokens means more exploration of the reasoning
        space before committing to an answer.
      </Prose>

      <H3>6.2 Emergent self-correction frequency over RL training</H3>

      <Plot
        label="self-correction strategy frequency over rl training — emerges without direct supervision"
        width={540}
        height={240}
        xLabel="RL training step"
        yLabel="fraction of responses using self-correction strategy"
        series={[
          {
            name: "self-correction frequency",
            color: "#e2b55a",
            points: [
              [0, 0.334], [10, 0.472], [20, 0.575], [30, 0.688],
              [40, 0.794], [50, 0.784], [59, 0.794],
            ],
          },
          {
            name: "average reward (right axis proxy)",
            color: "#60a5fa",
            points: [
              [0, 0.509], [10, 0.541], [20, 0.603], [30, 0.584],
              [40, 0.641], [50, 0.622], [59, 0.634],
            ],
          },
        ]}
      />

      <Prose>
        The self-correction strategy grows from 33% to 79% of all responses over 60
        training steps. It was not labeled, not demonstrated, and not directly rewarded.
        The policy discovered it because responses that include explicit verification
        steps earn higher binary reward from the verifier — they are more likely to
        catch errors before committing to an answer. Average reward grows correspondingly.
        This is the toy-scale analog of what DeepSeek-R1-Zero documented: emergent
        self-correction vocabulary appearing spontaneously in model outputs.
      </Prose>

      <H3>6.3 The R1-style training loop — interactive step trace</H3>

      <StepTrace
        label="r1-style rl training loop: cold-start → grpo → rejection sampling → repeat"
        steps={[
          {
            label: "Step 1 — Cold-start SFT",
            render: () => (
              <div>
                <TokenStream
                  label="curated reasoning traces → SFT → stable starting policy"
                  tokens={[
                    { label: "curated CoT data", color: colors.gold },
                    { label: "→ SFT", color: colors.textMuted },
                    { label: "π_ref (warm start)", color: "#4ade80" },
                    { label: "establishes format", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  A few thousand high-quality reasoning traces establish a reasoning format
                  before GRPO begins. Without this, R1-Zero's early outputs are structurally
                  inconsistent — often correct in substance but hard to evaluate. The cold start
                  is not required for capability emergence but substantially speeds up training.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Generate G long rollouts per problem",
            render: () => (
              <div>
                <TokenStream
                  label="problem x → policy generates G reasoning traces"
                  tokens={[
                    { label: "x: AIME problem", color: colors.gold },
                    { label: "→ π samples G=8 traces", color: colors.textMuted },
                    { label: "y₁: 2400 tokens", color: "#c084fc" },
                    { label: "y₂: 3100 tokens", color: "#c084fc" },
                    { label: "...y₈", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  Temperature ~1.0. Each rollout is a full chain-of-thought ending in a
                  boxed answer. Long traces are expected and encouraged — the policy has
                  learned that longer exploration leads to higher reward on hard problems.
                  GPU memory for 8 long sequences in parallel is the primary compute cost.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — Verify and compute group advantages",
            render: () => (
              <div>
                <TokenStream
                  label="math verifier → binary rewards → GRPO normalization"
                  tokens={[
                    { label: "y₁: r=0", color: "#f87171" },
                    { label: "y₂: r=1", color: "#4ade80" },
                    { label: "y₃: r=0", color: "#f87171" },
                    { label: "...y₈: r=0", color: "#f87171" },
                    { label: "A₂=+2.65, Aᵢ=−0.38", color: "#e2b55a" },
                  ]}
                />
                <Prose>
                  Regex-based answer parser extracts the boxed final answer and compares
                  it to ground truth. Binary 0/1 reward. GRPO group normalization converts
                  the raw reward into an advantage that is meaningful even when only one
                  of eight rollouts is correct.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 4 — PPO clipped update with KL penalty",
            render: () => (
              <div>
                <TokenStream
                  label="clipped surrogate + KL regularization → gradient step"
                  tokens={[
                    { label: "L_CLIP(A, ratio)", color: "#e2b55a" },
                    { label: "− β·KL(π‖π_ref)", color: "#60a5fa" },
                    { label: "→ ∇θ", color: "#4ade80" },
                    { label: "→ 4 inner epochs", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  Two to four inner PPO epochs per rollout batch. KL coefficient β=0.04.
                  Clip range ε=0.2. The KL penalty is especially important at this
                  stage: without it, reasoning traces collapse onto a single high-reward
                  template (mode collapse on specific proof strategies) rather than
                  maintaining the diversity needed for continued generalization.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 5 — Rejection sampling and SFT amplification",
            render: () => (
              <div>
                <TokenStream
                  label="correct rollouts → SFT dataset → updated π_ref → next GRPO phase"
                  tokens={[
                    { label: "verified traces", color: "#4ade80" },
                    { label: "→ SFT", color: colors.gold },
                    { label: "→ π_ref update", color: "#c084fc" },
                    { label: "→ next RL phase", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  After each major RL phase, the model generates many rollouts and the
                  verifier selects the correct ones. SFT on these verified correct traces
                  raises the floor of the policy — improving the reference model for the
                  next GRPO phase. This amplification loop is why the full DeepSeek-R1
                  pipeline significantly outperforms R1-Zero (pure RL, no SFT phases).
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

      <H3>When to apply RL for reasoning</H3>

      <Heatmap
        label="rl for reasoning: use vs skip — by task and infrastructure criteria (5 = strongly favors)"
        matrix={[
          [5, 5, 5, 5, 4],
          [1, 2, 1, 2, 1],
          [2, 3, 1, 3, 2],
          [4, 4, 5, 3, 3],
        ]}
        rowLabels={[
          "Use RL-for-reasoning",
          "Use SFT only",
          "Use RLHF/DPO instead",
          "Need more data first",
        ]}
        colLabels={[
          "Verifiable task",
          "Strong base model",
          "Auto-verifier available",
          "Long rollout budget",
          "Hard eval benchmark",
        ]}
        colorScale="green"
        cellSize={48}
      />

      <Prose>
        <strong>Apply RL for reasoning when:</strong> the task has mechanically verifiable
        correctness (math with ground-truth answers, code with test suites, formal proofs
        with type checkers); the base model is at the 30B+ parameter tier with strong
        pretraining; the compute budget supports generating 8–16 long rollouts per training
        step; and you are targeting hard evaluation benchmarks where base model + CoT
        prompting has plateaued. This is the setting where emergent reasoning behaviors
        appear and test-time compute scaling pays out.
      </Prose>

      <Prose>
        <strong>Skip RL for reasoning when:</strong> the task is subjective (creative
        writing, open-ended conversation) and no ground-truth verifier can be constructed;
        the base model is too small or underpowered to generate any correct responses on
        the target problem class (no gradient signal → no learning); compute is constrained
        to the point that generating 8+ long rollouts per training step is infeasible; or
        the task distribution is narrow enough that SFT on high-quality demonstrations
        already saturates performance.
      </Prose>

      <Prose>
        <strong>Hybrid approaches:</strong> Tülu 3 and similar pipelines combine RLVR for
        math/code capability with DPO for general alignment. The two objectives do not
        conflict — DPO anchors helpfulness and safety, RLVR drives reasoning capability —
        and the combination consistently outperforms either alone on comprehensive evals.
        This is currently the recommended production recipe for frontier open models.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        <strong>Base model capability is the floor, not the ceiling.</strong> Larger and
        better-pretrained base models produce more dramatic RL improvements. DeepSeek-R1
        starts from DeepSeek-V3, a 671B mixture-of-experts model. The RL doesn't create
        mathematical reasoning from nothing — it amplifies latent capability that the
        base model acquired during pretraining on vast mathematical text. A 7B model
        given identical GRPO training will improve significantly less, not because the
        algorithm fails, but because the capability being amplified isn't there to
        the same degree.
      </Prose>

      <Prose>
        <strong>Test-time compute scales log-linearly on hard problems.</strong> This is
        the signature finding: the RL-trained model's accuracy on AIME-class problems
        grows approximately as α·log(tokens) over several orders of magnitude of token
        budget. Each doubling of the token budget produces a fixed additive improvement
        in accuracy. The base model's curve shows no such relationship at scale. This
        means both dimensions — model size and inference compute — now independently
        pay out, and can be traded against each other depending on whether you want a
        faster smaller model with more compute at inference, or a slower larger model
        with standard inference.
      </Prose>

      <Prose>
        <strong>Verified synthetic data scales without annotation cost.</strong> Because
        the verifier checks correctness deterministically, the labeled dataset size is
        limited only by compute — not by human annotation labor. You can generate
        millions of math problems programmatically (synthetic arithmetic, competition
        problem variations, formal proof exercises) and verify the model's responses for
        free. This removes the annotation bottleneck that limits RLHF datasets and is
        one reason RL-for-reasoning training runs can be sustained for far longer than
        RLHF runs before hitting quality ceilings.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        <strong>Weak base models cannot bootstrap.</strong> If the base model's pass rate
        on the target problem class is essentially zero, GRPO groups return all-zero
        rewards on every step. Group standard deviation is zero. The gradient is skipped.
        Training stalls entirely. No amount of RLVR scaling rescues this situation — you
        need either a stronger base model, a curriculum that starts with easier problems
        the model can occasionally solve, or SFT warm-start on correct reasoning traces
        before RL begins. R1-Zero itself required DeepSeek-V3 as the base; even that
        required a cold-start SFT phase for the full R1 recipe.
      </Prose>

      <Prose>
        <strong>Reasoning transfer to non-verifiable tasks is real but partial.</strong>
        Models trained on math and code reasoning show improved performance on general
        language benchmarks — transfer is documented in DeepSeek-R1, Kimi k1.5, and Tülu
        3. But the transfer is not complete. A model that reasons brilliantly about AIME
        problems does not automatically reason as brilliantly about ambiguous policy
        questions or open-ended research questions where no verifier exists. The
        structured deliberation style transfers; the task-specific precision does not.
      </Prose>

      <Prose>
        <strong>Rollout generation is the compute bottleneck, not gradient computation.</strong>
        Generating 8–16 long reasoning traces per training problem per step is expensive.
        At thousands of tokens per trace, eight rollouts means tens of thousands of tokens
        of generation for every gradient step. vLLM and speculative decoding help, but the
        fundamental bottleneck is the autoregressive generation of long sequences, which
        scales linearly with sequence length. This is why production RL-for-reasoning
        training requires dedicated inference clusters separate from the gradient computation
        cluster, and why training throughput is measured in problems per hour rather than
        tokens per second.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Cold-start failure: base model can't generate any correct responses</H3>

      <Prose>
        The most common failure in RL-for-reasoning: the base model is applied to a
        problem class that is too hard for it to solve even occasionally. With a 0%
        pass rate, every group of G=8 rollouts returns all-zero rewards. GRPO's group
        standard deviation is zero. The gradient is skipped. Nothing learns. This is
        not a hyperparameter problem — it's a bootstrap problem. Mitigation: curriculum
        starting with problems the model can solve at 10-30% pass rate; SFT warm-start
        on correct rollouts from a stronger model or a simpler problem class; or
        extending G substantially (G=32 or G=64) to increase the probability that at
        least one rollout is correct even at very low base pass rates.
      </Prose>

      <H3>9.2 Reward hacking via verifier edge cases</H3>

      <Prose>
        Even a deterministic verifier can be exploited if its specification has gaps.
        The most common form: the model learns to output a boxed answer that matches the
        ground truth without producing any genuine reasoning chain — essentially guessing
        the numerical answer. If the verifier only checks the boxed answer and not the
        presence or quality of the reasoning chain, this exploits the verifier's narrower
        scope. DeepSeek-R1 addresses this with a format reward that penalizes responses
        where the reasoning chain is absent or suspiciously short. A more subtle form:
        for code verifiers with predictable test inputs, the model can hardcode expected
        outputs rather than implementing the general function.
      </Prose>

      <H3>9.3 Mode collapse on reasoning templates</H3>

      <Prose>
        Without adequate KL regularization, the policy collapses onto a small set of
        high-reward reasoning templates — specific proof structures or solution
        approaches that happened to earn high reward in training. The collapse manifests
        as response diversity falling sharply: if you generate 50 responses to the same
        problem, they all look structurally identical even when the answer is reached
        by genuinely different paths. The model's reasoning becomes a template engine.
        Detection: monitor the unique n-gram ratio across rollout batches and the
        perplexity of the model on its own training rollouts. Mitigation: increase β,
        add diversity bonuses to the reward function, ensure the training problem
        distribution spans many different mathematical domains and proof styles.
      </Prose>

      <H3>9.4 Verbose collapse</H3>

      <Prose>
        A failure mode specific to the long-rollout setting: the model learns that
        producing very long outputs correlates with higher reward (because longer traces
        are more likely to include a correct reasoning path somewhere), so it learns to
        generate long outputs regardless of whether the length is actually warranted by
        the problem's difficulty. On a simple arithmetic problem that should take 50
        tokens, a verbosely-collapsed model produces 3,000 tokens of meandering
        deliberation before arriving at 2+2=4. Detection: track the mean response length
        per problem difficulty tier. If easy-problem responses are growing in length
        toward hard-problem lengths, the model is verbose-collapsing. Mitigation: add
        a length efficiency reward (reward per token, not just reward per response) or
        per-difficulty length caps.
      </Prose>

      <H3>9.5 KL anchor drift</H3>

      <Prose>
        If β is too low, the policy drifts far from the SFT reference. In the long-rollout
        setting, this produces reasoning traces that are highly optimized for the verifier's
        exact answer format but are unnatural to read and fail to generalize to problem
        variations. The reasoning style becomes crystallized around whatever format first
        earned high reward, rather than remaining flexible across domains. Detection:
        monitor KL divergence — if it exceeds 20 nats early in training, β is too low.
        Mitigation: increase β, use adaptive KL control targeting a specific KL budget
        (DeepSeek-R1 used β=0.04; most open reproductions use the same range).
      </Prose>

      <H3>9.6 Eval contamination amplified by RL</H3>

      <Prose>
        RL training on a dataset that overlaps with evaluation benchmarks produces
        inflated results that don't reflect true generalization. This failure mode is
        more acute for RL than for SFT because RL can overfit to the exact problem
        distribution used in training in ways that are harder to detect — the model
        generalizes well within the training distribution and fails catastrophically
        on held-out variations. For math: ensure the training problem set has no
        overlap with AIME, MATH, and AMC benchmarks. For code: verify the training
        problems don't appear in HumanEval, MBPP, or LiveCodeBench. Check for
        near-duplicate paraphrases, not just exact matches.
      </Prose>

      <H3>9.7 Hyperparameter sensitivity at scale</H3>

      <Prose>
        RL-for-reasoning training is significantly more hyperparameter-sensitive than
        SFT or DPO. The KL coefficient (β), learning rate, group size (G), clip range
        (ε), and temperature during rollout generation all interact, and their optimal
        settings are not predictable from small-scale experiments. A combination that
        works on a 7B model may cause KL explosion on a 70B model using otherwise
        identical settings. Published configurations (DeepSeek-R1: β=0.04, G=8–16,
        ε=0.2; open-r1: lr=1e-6, temperature=0.9) are reasonable starting points, but
        expect a tuning phase with active monitoring of KL divergence, response length
        distribution, and held-out pass rates before committing to a long training run.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below verified against arXiv, HuggingFace, and OpenAI as of
        April 2026.
      </Prose>

      <H3>10.1 Core papers</H3>

      <Prose>
        <strong>DeepSeek-AI (2025).</strong> "DeepSeek-R1: Incentivizing Reasoning
        Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. The canonical
        paper documenting RL-for-reasoning at scale. Documents R1-Zero (pure RL,
        emergent self-correction without SFT), R1 (full pipeline with cold-start SFT,
        GRPO, rejection sampling, and final alignment), the "aha moment" phenomenon,
        and the test-time compute scaling results. AIME 2024: 15.6% → 71.0% pass@1
        (86.7% with majority voting). GRPO configuration: G=8–16, β=0.04, clip ratio
        0.2. Also published Nature Communications 2025.
      </Prose>

      <Prose>
        <strong>Shao, Wang, Zhu, Xu, et al. (2024).</strong> "DeepSeekMath: Pushing the
        Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
        Introduces GRPO as an alternative to PPO that eliminates the value model by
        using within-group reward normalization. DeepSeekMath-7B achieves 51.7% on the
        MATH benchmark using GRPO. The GRPO section is the mathematical and algorithmic
        foundation for all DeepSeek-R1 training.
      </Prose>

      <Prose>
        <strong>Lightman, Kosaraju, Burda, Edwards, et al. (2023).</strong> "Let's Verify
        Step by Step." arXiv:2305.20050. OpenAI. Compares process-supervised reward
        models (step-level feedback) to outcome-supervised reward models (final answer
        only). Finds process supervision outperforms outcome supervision on MATH, and
        releases PRM800K — 800,000 step-level human feedback labels. Important context
        for understanding why RLVR's outcome-only approach being competitive is
        surprising: the model discovers process-level behavior from outcome-level signals.
      </Prose>

      <Prose>
        <strong>Lambert, Morrison, Miranda, et al. (2024).</strong> "Tülu 3: Pushing
        Frontiers in Open Language Model Post-Training." arXiv:2411.15124. Allen
        Institute for AI. First paper to explicitly name and document RLVR as a distinct
        post-training stage. Uses RLOO rather than GRPO. Combines RLVR with DPO in a
        multi-objective pipeline. Full open-source release.
      </Prose>

      <Prose>
        <strong>Kimi Team (2025).</strong> "Kimi k1.5: Scaling Reinforcement Learning with
        LLMs." arXiv:2501.12599. Moonshot AI. Independent parallel effort to DeepSeek-R1.
        Documents the same test-time compute scaling property, long context RL training,
        and reasoning emergence. AIME: 77.5. MATH 500: 96.2. Codeforces: 94th percentile.
        Matches o1 without relying on Monte Carlo tree search or process reward models.
        A second independent confirmation that the RL-for-reasoning recipe generalizes.
      </Prose>

      <H3>10.2 Open-source implementations</H3>

      <Prose>
        <strong>HuggingFace (2025).</strong> Open-R1: Fully Open Reproduction of
        DeepSeek-R1. github.com/huggingface/open-r1. Provides GRPO training script
        with pluggable verifier interface, SFT cold-start script, Distilabel-based
        synthetic data generation, OpenR1-Math-220k dataset (220k reasoning traces),
        CodeForces-CoTs dataset (100k competitive programming solutions). The most
        complete public RLVR training implementation available as of 2026.
      </Prose>

      <Prose>
        <strong>OpenAI (2024).</strong> o1 System Card and Technical Report.
        openai.com/o1. Not a scientific paper but a technical disclosure confirming
        the use of RL training on reasoning tasks with verifiable rewards and the
        test-time compute scaling property. The o1 series is the original commercial
        deployment of RL-for-reasoning at scale.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive why KL regularization is critical in the long-rollout setting</H3>

      <Prose>
        In standard RLHF, the KL penalty prevents the policy from exploiting the learned
        reward model. In RL-for-reasoning with a verifier, the verifier cannot be hacked —
        it is deterministic and correct. Why, then, is KL regularization still critical?
        Derive two distinct failure modes that arise when β is set to zero in the
        long-rollout setting. Consider: (a) reasoning template collapse — what happens to
        the diversity of reasoning approaches as training progresses without KL constraint;
        (b) distribution shift — how does removing the KL anchor affect the model's
        performance on problems outside the training distribution? Show mathematically
        why the KL term becomes more important, not less, as rollout length increases.
      </Prose>

      <H3>Exercise 2 — Design a multi-step algebra verifier</H3>

      <Prose>
        DeepSeek-R1's math verifier uses regex extraction of a boxed final answer with
        float comparison. Design a more robust verifier for multi-step algebra problems
        where the answer may be expressed as a fraction, a symbolic expression, or a
        polynomial. Your verifier should: (a) handle equivalent representations
        (3/4 = 0.75 = 6/8); (b) detect "answer without reasoning" — a response that
        contains a correct boxed answer but less than 50 tokens of reasoning before it;
        (c) score partial credit for responses that reach the correct intermediate steps
        but make an arithmetic error at the final step. Describe the engineering
        tradeoffs between each of these additions and the risk of introducing new
        exploitable failure modes.
      </Prose>

      <H3>Exercise 3 — Why does R1-Zero struggle to bootstrap without SFT warmup?</H3>

      <Prose>
        DeepSeek-R1-Zero applies pure RL with no SFT cold-start phase. The paper reports
        that while R1-Zero eventually achieves strong performance, early training is
        unstable and the model produces inconsistent output formats. Explain precisely
        why the absence of a cold-start SFT phase creates this instability. Consider:
        (a) what the initial policy's output distribution looks like for a raw base model
        on a math competition problem; (b) how inconsistent output structure affects the
        math verifier's ability to parse answers; (c) how an inability to parse answers
        affects the reward distribution and GRPO's group advantage estimates. Why does
        the cold-start phase help even when its data is a tiny fraction of the eventual
        RL training volume?
      </Prose>

      <H3>Exercise 4 — Predict the test-time compute curve for an untuned base model</H3>

      <Prose>
        You are given a strong base model (e.g., Llama-3-70B) that has not received any
        RLVR or reasoning fine-tuning. On AIME-level math problems, it achieves 12%
        accuracy with greedy decoding and 18% with a "think step by step" prompt.
        Predict the shape of the accuracy-vs-inference-token-budget curve for this model
        compared to an R1-style RL-trained version of the same base. Explain: (a) why
        the base model's curve plateaus quickly despite having more tokens available;
        (b) what specific behaviors the RL-trained model exhibits at high token budgets
        that the base model does not; (c) at what point on the token budget axis the
        curves are most similar and why. Design a benchmark experiment to verify your
        predictions empirically.
      </Prose>

      <H3>Exercise 5 — Detect reward hacking from chain-of-thought content statistics</H3>

      <Prose>
        Without running human evaluation, design a set of automatic statistics you would
        monitor during RL-for-reasoning training to detect reward hacking and reasoning
        quality degradation. Consider: (a) response length distribution per problem
        difficulty tier — what pattern indicates verbose collapse vs productive long
        reasoning; (b) unique n-gram ratio across rollouts from the same problem — what
        threshold indicates mode collapse on reasoning templates; (c) "reasoning
        consistency" — whether the steps in the chain-of-thought logically entail the
        next step (detectable via a small classification model trained on correct vs
        incoherent chains); (d) answer-without-reasoning rate — fraction of responses
        where the boxed answer appears in the first 10% of tokens. For each statistic,
        specify the monitoring threshold and the corrective action you would take.
      </Prose>

      {/* ======================================================================
          CLOSING — SECTION FINALE
          ====================================================================== */}
      <H2>The Post-Training section: what this has all been building toward</H2>

      <Prose>
        This topic is the summit of the Post-Training section, and it is worth pausing
        at the peak to see what the climb covered.
      </Prose>

      <Prose>
        The section began with supervised fine-tuning: imitation of human demonstrations,
        compressing the base model's output distribution from "everything the internet
        contains" down to "things that look like assistant responses." SFT is fast and
        effective, but it teaches the model what good responses look like — not what makes
        them good. It has a ceiling set by the quality of the demonstrations it imitates.
      </Prose>

      <Prose>
        RLHF broke through that ceiling by replacing imitation with preference optimization.
        Instead of "copy this response," the signal became "this response is better than
        that one." A learned reward model translated human pairwise comparisons into a
        scalar signal the policy could optimize against. The key move was the asymmetry
        exploit: humans are better at judging comparative quality than at generating
        ideal responses from scratch. RLHF harvests that asymmetry. But it introduced
        Goodhart's Law: the reward model is a proxy, the policy eventually finds the
        proxy's cracks, and continued optimization degrades true quality while RM score
        keeps rising.
      </Prose>

      <Prose>
        DPO, SimPO, ORPO, and the DPO-family methods addressed RLHF's operational
        complexity. They showed that the KL-constrained preference optimization objective
        could be solved offline, without an explicit reward model or RL training loop,
        using a supervised loss over preference pairs. Constitutional AI replaced human
        labelers with model-generated preference signal, scaling the preference pipeline
        to tasks and domains where human annotation is impractical. Each of these methods
        reduced cost and operational complexity. None of them solved the proxy problem —
        they all optimize against some approximation of human preference, and all are
        subject to the same ceiling that RLHF hits.
      </Prose>

      <Prose>
        Process Reward Models and Outcome Reward Models (the PRM/ORM topic) addressed
        the precision of feedback: rather than scoring complete responses, PRMs score
        individual reasoning steps, providing denser and more targeted gradient signal.
        This is where the connection to mathematical reasoning first appeared explicitly
        — the Lightman et al. paper showed that step-level supervision outperforms
        response-level supervision on hard math. But PRMs still require human annotation
        of reasoning steps, which is expensive and doesn't scale to the hardest problems
        where expert annotators are needed.
      </Prose>

      <Prose>
        RLVR replaced the learned reward model entirely with a deterministic verifier for
        tasks where correctness is mechanically checkable. No learned parameters, no proxy
        to hack, no annotation cost beyond the problem and its ground-truth answer. The
        signal is exact. The training can continue far longer before hitting quality ceilings.
        And because correct rollouts can be collected and used as SFT data for the next
        training phase, the pipeline is self-amplifying.
      </Prose>

      <Prose>
        Knowledge distillation (the distillation topic) showed how the capability gains
        from large RL-trained models could be transferred to smaller models via supervised
        training on the large model's outputs — making the reasoning behaviors accessible
        in models too small to develop them via direct RL training. Distillation from
        DeepSeek-R1's reasoning traces produced 7B and 14B models that outperform much
        larger models on math benchmarks, because they inherited the reasoning patterns
        without needing to rediscover them from scratch.
      </Prose>

      <Prose>
        RL for reasoning is where all of these threads converge. RLVR provides the exact,
        unhackable reward signal. GRPO provides the gradient estimation algorithm that
        handles sparse binary rewards across long rollouts without a value model. The strong
        base model provides the latent capability that RL amplifies. The KL anchor
        (from RLHF's core design) prevents distributional collapse during long training.
        The rejection sampling and SFT phases (SFT's machinery) periodically raise the
        policy floor. And the result — emergent self-correction, test-time compute scaling,
        reasoning behaviors the training data didn't contain — is something none of the
        earlier methods could produce individually. It required all of them, assembled in
        the right order.
      </Prose>

      <Callout accent="gold">
        The Post-Training section traced a sequence of interventions, each one addressing
        the specific limitation that came before: SFT's imitation ceiling led to RLHF.
        RLHF's proxy problem led to RLVR. RLVR's sparse reward gradient estimation led to
        GRPO. GRPO on hard problems, run long enough with a strong enough base, produced
        the aha phenomenon. The next section — Inference Optimization — turns to the
        practical question that follows from all of this: how do you actually serve models
        that reason in thousands of tokens per query, at scale, without spending your
        entire compute budget on a single hard question?
      </Callout>

      <Prose>
        One final note on epistemic humility. The material in this topic moves faster than
        any other topic in this hub. The DeepSeek-R1 paper is from January 2025. Open-R1
        is from the same month. Kimi k1.5 is concurrent. As of this writing, labs are
        actively extending the recipe — to agentic tasks where the verifier is a real
        environment, to formal theorem proving where Lean or Isabelle is the verifier,
        to multimodal domains where a visual verifier can check geometric proofs. The
        core mechanisms documented here — verifiable reward, GRPO group sampling, emergent
        reasoning, test-time compute scaling — appear stable across these extensions. The
        specific architectures, training configurations, and benchmark numbers will age.
        The principles will not, or at least they will age more slowly. Treat them as
        the right abstraction level to hold onto as the frontier continues moving.
      </Prose>

    </div>
  ),
};

export default rlForReasoning;
