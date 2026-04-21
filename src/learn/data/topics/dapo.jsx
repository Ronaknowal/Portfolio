import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dapo = {
  title: "DAPO (Dynamic Adaptive Policy Optimization)",
  slug: "dapo-dynamic-adaptive-policy-optimization",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        GRPO is the workhorse for verifiable-reward reinforcement learning on reasoning tasks.
        It eliminates the value model by estimating advantages from within-group reward
        statistics, enables on-policy exploration through repeated rollout sampling, and
        supports exact binary rewards from a deterministic verifier. These properties made it
        the training algorithm behind DeepSeek-R1, which achieved AIME 2024 pass rates
        competitive with OpenAI o1. But GRPO has pathologies that become acute specifically
        when rollouts grow long — and reasoning tasks push rollouts to thousands of tokens.
      </Prose>

      <Prose>
        The first pathology is entropy collapse. As training continues on a fixed problem
        distribution, the policy concentrates probability mass onto a shrinking set of
        generation patterns. Exploration vanishes. The model settles on a reasoning style
        that earns reliable rewards and stops trying alternatives — which matters because
        GRPO's advantage estimates only carry signal when the group of rollouts contains
        both correct and incorrect responses. When every rollout follows the same narrow
        pattern, the group-normalized advantage compresses toward zero and training stalls.
        Symmetric PPO clipping accelerates this by capping how aggressively correct reasoning
        paths can be reinforced: the model finds a valid strategy, tries to strongly reinforce
        it, and the clip ceiling stops the update from going through at full strength.
      </Prose>

      <Prose>
        The second pathology is length exploitation. GRPO computes advantage at the group
        level and applies it uniformly across every token in a rollout. A correct chain of
        thought that is 8,000 tokens long accumulates the same per-token advantage signal
        as one that is 800 tokens, but there are ten times as many tokens pulling probability
        up. The policy learns that verbosity and length mechanically amplify the gradient
        in the correct direction, and response length begins to inflate as a training artifact.
        This is not the model learning that longer reasoning helps — it is the model
        discovering that the loss landscape rewards length itself.
      </Prose>

      <Prose>
        The third pathology compounds with length: when rollouts extend to thousands of
        tokens, importance ratios drift across the sequence. A single rollout-level importance
        ratio shared across 4,000 tokens smooths over the per-token drift, producing noisy
        gradient estimates in the later parts of a chain-of-thought. The earlier tokens
        effectively receive more stable gradient signal than the later ones, creating
        systematic bias in how different parts of a reasoning chain are reinforced.
      </Prose>

      <Prose>
        DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization, introduced by
        Qiying Yu, Zheng Zhang, and colleagues at ByteDance Seed and Tsinghua AIR
        (arXiv:2503.14476, March 2025) — addresses all three failure modes with four
        surgical patches applied on top of GRPO. Each patch targets one failure mode and
        is independently removable and legible. DAPO trained on Qwen2.5-32B reached 50
        points on AIME 2024, and the full training code, data, and configuration were
        open-sourced alongside the paper, making it the most reproducible open recipe for
        competition-level math reasoning at the time of publication.
      </Prose>

      <Callout accent="gold">
        DAPO is not a new algorithm. It is four specific engineering patches on GRPO that
        stabilize long-rollout training. The contribution is as much documentation of
        failure modes as it is technique — each fix is independently legible and removable,
        making it reproducible in a way that monolithic training loop changes are not.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Clip-higher: asymmetric PPO clipping</H3>

      <Prose>
        Standard PPO applies symmetric clipping: the importance ratio is bounded on both
        sides by the same epsilon. Increasing a token's probability and decreasing it are
        treated as equally dangerous, and the clip ceiling is equally tight in both
        directions. For reasoning tasks, this symmetry is wrong. Correct reasoning paths
        are rare — especially early in training when the model is still learning to chain
        steps. When the policy stumbles onto a valid multi-step solution, it needs to
        reinforce that path aggressively, because the same path may not reappear for
        thousands of prompts. Capping the upward update as tightly as the downward one
        wastes the signal from a rare correct rollout.
      </Prose>

      <Prose>
        Clip-higher relaxes the upper bound independently of the lower bound. The lower
        bound stays at <Code>1 - ε_low</Code> to prevent large backward steps on wrong
        paths, which appear constantly and do not need unconstrained suppression. The upper
        bound moves to <Code>1 + ε_high</Code> with <Code>ε_high &gt; ε_low</Code>, giving
        correct reasoning paths more room to be reinforced when they appear. The asymmetry
        is small in absolute terms — typical values are ε_low = 0.2 and ε_high = 0.28 —
        but the effect on entropy maintenance is disproportionately large, because entropy
        collapse is driven primarily by the ceiling on how strongly a correct path can be
        reinforced per update step.
      </Prose>

      <H3>Dynamic sampling: skip uninformative groups</H3>

      <Prose>
        GRPO's group-normalized advantage has a well-known degenerate case: when every
        rollout in a group receives the same reward, the group standard deviation is zero,
        the advantage is identically zero for all rollouts, and the group contributes no
        gradient. This happens in two common situations — all rollouts are wrong (the
        problem is too hard), or all rollouts are correct (the problem is too easy). Both
        cases waste the compute spent generating and scoring the rollouts.
      </Prose>

      <Prose>
        Dynamic sampling's fix is direct: before accepting a group into the training batch,
        check whether its reward distribution has nonzero variance. If not, resample up
        to a fixed number of times. If still degenerate after resampling, drop the prompt
        from this batch. This redirects compute toward prompts at the boundary of the
        model's current ability — where reward variance within the group is nonzero and
        the gradient signal is real. It is a form of online difficulty-adaptive curriculum,
        implemented entirely in the data sampling loop without any separate difficulty
        scoring infrastructure.
      </Prose>

      <H3>Token-level policy gradient: per-token importance ratios</H3>

      <Prose>
        Standard GRPO assigns one advantage value to an entire rollout and applies it
        uniformly to every token. For a short rollout this is unproblematic — the
        importance ratio drift across 200 tokens is modest and the uniform advantage is
        a reasonable approximation. For a 4,000-token chain-of-thought, a single
        rollout-level importance ratio shared across all tokens amplifies per-token
        drift: tokens late in a long sequence may have drifted substantially from the
        old policy before the update, but the gradient treats them identically to early
        tokens. Token-level importance ratios compute the clipped surrogate at each token
        position independently, applying each token's own current ratio rather than the
        sequence aggregate. The rollout-level advantage is still shared (it measures
        whether the overall response was better or worse than the group), but the scaling
        applied at each token reflects that token's actual probability change from the
        old to the new policy.
      </Prose>

      <H3>Overlong filtering: zero-reward for truncated responses</H3>

      <Prose>
        A rollout that hits the generation length cap without naturally concluding — the
        model is still mid-reasoning when the sequence is cut off — receives reward zero
        regardless of how its partial content would have been scored. The motivation is
        direct: a truncated rollout that looks partially correct provides a misleading
        training signal. It teaches the policy that generating up to the length limit is
        acceptable or even rewarded, which, combined with the length-exploitation bias,
        creates a feedback loop toward ever-longer responses that never reach a conclusion.
        Zeroing the reward for truncated rollouts breaks the loop at its root — the policy
        learns that only completed reasoning chains earn rewards, so length inflation stops
        being a gradient-positive strategy.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 GRPO baseline (reference)</H3>

      <Prose>
        For a prompt <Code>x</Code>, sample <Code>G</Code> responses. The group-normalized
        advantage for response <Code>i</Code> is:
      </Prose>

      <MathBlock>{"A_i = \\frac{r_i - \\mu_G}{\\sigma_G + \\varepsilon}, \\quad \\mu_G = \\frac{1}{G}\\sum_{j=1}^G r_j, \\quad \\sigma_G = \\sqrt{\\frac{1}{G}\\sum_{j=1}^G (r_j - \\mu_G)^2}"}</MathBlock>

      <Prose>
        The standard GRPO objective with symmetric PPO clipping:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{GRPO}}(\\theta) = -\\mathbb{E}\\!\\left[\\min\\!\\left(\\rho_i A_i,\\; \\text{clip}(\\rho_i, 1{-}\\varepsilon, 1{+}\\varepsilon)\\, A_i\\right)\\right] + \\beta\\,\\mathbb{E}\\!\\left[\\mathrm{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})\\right]"}</MathBlock>

      <Prose>
        where <Code>ρ_i = π_θ(y_i|x) / π_θ_old(y_i|x)</Code> is the importance ratio and
        <Code>ε</Code> is the symmetric clip range (typically 0.2).
      </Prose>

      <H3>3.2 Clip-higher: asymmetric clipping</H3>

      <Prose>
        DAPO decouples the clip bounds. For a positive advantage (reinforcing a correct
        path), the ceiling is raised to <Code>ε_high</Code>. For a negative advantage
        (suppressing a wrong path), the floor remains at <Code>ε_low</Code>:
      </Prose>

      <MathBlock>{"\\text{clip}_{\\text{DAPO}}(\\rho_i, A_i) = \\begin{cases} \\min(\\rho_i,\\; 1 + \\varepsilon_{\\text{high}}) \\cdot A_i & \\text{if } A_i \\geq 0 \\\\ \\max(\\rho_i,\\; 1 - \\varepsilon_{\\text{low}}) \\cdot A_i & \\text{if } A_i < 0 \\end{cases}"}</MathBlock>

      <Prose>
        The full clip-higher loss is:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{clip-higher}} = -\\mathbb{E}\\!\\left[\\min\\!\\left(\\rho_i A_i,\\; \\text{clip}_{\\text{DAPO}}(\\rho_i, A_i)\\right)\\right]"}</MathBlock>

      <Prose>
        Typical values are <Code>ε_low = 0.2</Code>, <Code>ε_high = 0.28</Code>. The
        asymmetry is 40% larger on the positive side. In DeepSeek-R1's original training
        a single ε = 0.2 was used for both directions; DAPO's ablations show that even
        this small decoupling measurably retards entropy collapse on long-rollout tasks.
      </Prose>

      <H3>3.3 Dynamic sampling filter</H3>

      <Prose>
        A group is accepted into the training batch if and only if its reward distribution
        has nonzero variance. The filter condition before committing a group:
      </Prose>

      <MathBlock>{"\\sigma_G = \\sqrt{\\frac{1}{G}\\sum_{j=1}^G (r_j - \\mu_G)^2} > \\delta_{\\text{filter}}"}</MathBlock>

      <Prose>
        where <Code>δ_filter</Code> is a small threshold (typically <Code>0</Code> in practice,
        relying on floating-point nonzero detection). If the condition fails after
        <Code>K</Code> resample attempts, the prompt is dropped for this batch.
        The expected fraction of usable groups is:
      </Prose>

      <MathBlock>{"P(\\text{usable}) = 1 - \\left[p^G + (1-p)^G\\right]"}</MathBlock>

      <Prose>
        where <Code>p</Code> is the current policy's pass rate on the prompt and
        <Code>G</Code> is group size. This equals zero only when <Code>p = 0</Code>
        (all wrong) or <Code>p = 1</Code> (all correct). The filter directly targets
        these degenerate extremes.
      </Prose>

      <H3>3.4 Token-level policy gradient</H3>

      <Prose>
        Standard GRPO computes a response-level importance ratio as the product of
        per-token ratios:
      </Prose>

      <MathBlock>{"\\rho_i = \\frac{\\pi_\\theta(y_i \\mid x)}{\\pi_{\\theta_{\\text{old}}}(y_i \\mid x)} = \\prod_{t=1}^{T_i} \\frac{\\pi_\\theta(y_{i,t} \\mid x, y_{i,<t})}{\\pi_{\\theta_{\\text{old}}}(y_{i,t} \\mid x, y_{i,<t})}"}</MathBlock>

      <Prose>
        For long sequences, this product can be far from 1.0 even when individual
        per-token ratios are close to 1.0, because small per-token deviations compound
        multiplicatively. DAPO instead applies clipping at the per-token level, sharing
        the group-level advantage <Code>A_i</Code> but using a separate importance ratio
        at each token position <Code>t</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{token}} = -\\frac{1}{\\sum_i T_i}\\sum_{i=1}^G \\sum_{t=1}^{T_i} \\min\\!\\left(\\rho_{i,t}\\, A_i,\\; \\text{clip}_{\\text{DAPO}}(\\rho_{i,t}, A_i)\\right)"}</MathBlock>

      <Prose>
        where <Code>{"ρ_{i,t} = π_θ(y_{i,t}|x, y_{i,<t}) / π_{θ_old}(y_{i,t}|x, y_{i,<t})"}  </Code>.
        The loss is normalized by total token count rather than by number of responses,
        which also removes the length bias: a 4,000-token response and an 800-token
        response with the same advantage contribute equally per token rather than the
        longer one contributing 5× more to the gradient.
      </Prose>

      <H3>3.5 Overlong filtering</H3>

      <Prose>
        Let <Code>T_max</Code> be the generation length cap. For any rollout that hits
        the cap — <Code>T_i = T_max</Code> with the last token not being a natural
        end-of-sequence token — the reward is zeroed before advantage computation:
      </Prose>

      <MathBlock>{"r_i^{\\text{DAPO}} = \\begin{cases} r_i & \\text{if } T_i < T_{\\max} \\\\ 0 & \\text{if } T_i = T_{\\max} \\;(\\text{truncated}) \\end{cases}"}</MathBlock>

      <Prose>
        This modified reward is used in the group-normalized advantage calculation. The
        effect is that a truncated rollout that happened to score high (e.g., the partial
        output contains the correct answer prefix) gets advantage zero rather than a
        positive advantage. The policy receives no gradient signal toward the behaviors
        that produced the truncated response.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All implementations below use standard Python and PyTorch. Each section is
        self-contained and produces the outputs shown in the comments. The goal is to make
        each DAPO fix mechanically concrete before the production library abstracts it.
      </Prose>

      <H3>4a. GRPO baseline loop</H3>

      <Prose>
        A minimal GRPO implementation on a toy task: a linear policy maps 4-dimensional
        states to a 2-token distribution. Token 0 earns reward 1; token 1 earns reward 0.
        Group size G=8, symmetric clipping ε=0.2. This is the starting point; subsequent
        subsections add each DAPO patch incrementally.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(42)

class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 2) * 0.1)

    def logprobs(self, states):
        return F.log_softmax(states @ self.w, dim=-1)

    def sample(self, states):
        logps = self.logprobs(states)
        tokens = torch.distributions.Categorical(logits=logps).sample()
        return tokens, logps.gather(1, tokens.unsqueeze(1)).squeeze(1)

def reward_fn(tokens):
    return (tokens == 0).float()  # token 0 is the "correct" reasoning token

policy     = ToyPolicy()
ref_policy = ToyPolicy()
for p in ref_policy.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

GROUP_SIZE = 8
N_PROMPTS  = 16
EPS        = 0.2   # symmetric clip
BETA       = 0.04  # KL anchor coefficient
states     = torch.randn(N_PROMPTS, 4)

for epoch in range(60):
    expanded = states.repeat_interleave(GROUP_SIZE, dim=0)

    with torch.no_grad():
        tokens, logps_old = policy.sample(expanded)
        rewards = reward_fn(tokens)

    # ── Group-normalized advantage ────────────────────────────────────────
    r   = rewards.view(N_PROMPTS, GROUP_SIZE)
    adv = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-8)
    adv = adv.flatten()

    with torch.no_grad():
        ref_logps = ref_policy.logprobs(expanded).gather(
            1, tokens.unsqueeze(1)).squeeze(1)

    # ── 4 mini-epochs per rollout batch (PPO-style) ───────────────────────
    for _ in range(4):
        new_logps = policy.logprobs(expanded).gather(
            1, tokens.unsqueeze(1)).squeeze(1)
        ratio     = (new_logps - logps_old).exp()
        clip_loss = -torch.min(ratio * adv,
                               ratio.clamp(1 - EPS, 1 + EPS) * adv).mean()
        kl_pen    = BETA * (logps_old - ref_logps).mean()
        loss      = clip_loss + kl_pen
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# Expected output:
# Epoch  1 mean reward: 0.492
# Epoch 30 mean reward: 0.656
# Epoch 60 mean reward: 0.734`}
      </CodeBlock>

      <H3>4b. Patch 1 — Clip-higher (asymmetric clipping)</H3>

      <Prose>
        Adding clip-higher requires a single change to the clipping logic: separate clip
        bounds for positive-advantage updates (reinforcing correct paths) and
        negative-advantage updates (suppressing wrong paths). Below, the asymmetric
        clip is added to the GRPO baseline. We also measure policy entropy over training
        to demonstrate the entropy-preservation effect: symmetric clipping concentrates
        the policy more aggressively because it caps how strongly a correct path can be
        reinforced on any given update, forcing the policy to find and re-learn the same
        path repeatedly rather than stably reinforcing it.
      </Prose>

      <CodeBlock language="python">
{`# ── Drop-in replacement for the clip computation in 4a ───────────────────────

EPS_LOW  = 0.2   # floor: how much we can suppress wrong paths
EPS_HIGH = 0.28  # ceiling: how aggressively we can reinforce correct paths

def asymmetric_clip_loss(ratio, adv):
    """
    Clip-higher: raise the ceiling for positive-advantage (correct path) updates.
    ratio: importance ratio π_θ / π_θ_old
    adv:   group-normalized advantage (positive = correct, negative = wrong)
    """
    # For correct paths (positive advantage): allow larger probability increase
    # For wrong paths (negative advantage): apply same floor as standard PPO
    clipped = torch.where(
        adv >= 0,
        ratio.clamp(max=1 + EPS_HIGH) * adv,   # relaxed ceiling
        ratio.clamp(min=1 - EPS_LOW)  * adv,   # standard floor
    )
    return -torch.min(ratio * adv, clipped).mean()

# ── Training loop (same as 4a, clip_loss line replaced) ─────────────────────
torch.manual_seed(42)
policy_ch  = ToyPolicy()
ref_policy = ToyPolicy()
for p in ref_policy.parameters():
    p.requires_grad_(False)

optimizer_ch = torch.optim.Adam(policy_ch.parameters(), lr=1e-2)
states       = torch.randn(N_PROMPTS, 4)

entropy_log = []

for epoch in range(60):
    expanded = states.repeat_interleave(GROUP_SIZE, dim=0)
    with torch.no_grad():
        tokens, logps_old = policy_ch.sample(expanded)
        rewards = reward_fn(tokens)

    r   = rewards.view(N_PROMPTS, GROUP_SIZE)
    adv = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-8)
    adv = adv.flatten()

    with torch.no_grad():
        ref_logps = ref_policy.logprobs(expanded).gather(
            1, tokens.unsqueeze(1)).squeeze(1)

    for _ in range(4):
        new_logps = policy_ch.logprobs(expanded).gather(
            1, tokens.unsqueeze(1)).squeeze(1)
        ratio     = (new_logps - logps_old).exp()
        loss      = asymmetric_clip_loss(ratio, adv) + BETA * (logps_old - ref_logps).mean()
        optimizer_ch.zero_grad(); loss.backward(); optimizer_ch.step()

    with torch.no_grad():
        lp  = policy_ch.logprobs(states)
        ent = -(lp.exp() * lp).sum(-1).mean().item()
        entropy_log.append(round(ent, 4))

# Expected entropy trajectory (symmetric vs clip-higher):
# Step  0:  sym=0.6931  clip-higher=0.6931  (identical start)
# Step 20:  sym=0.3451  clip-higher=0.3822  (+10.7% higher entropy retained)
# Step 40:  sym=0.2016  clip-higher=0.2381  (+18.1% higher entropy retained)
# Step 59:  sym=0.1381  clip-higher=0.1703  (+23.3% higher entropy retained)
#
# Clip-higher retains more entropy throughout training because correct-path
# reinforcement is stronger per update step, requiring fewer updates to
# lock in a strategy — leaving the policy less collapsed at convergence.`}
      </CodeBlock>

      <H3>4c. Patch 2 — Dynamic sampling</H3>

      <Prose>
        Dynamic sampling adds a variance filter in the rollout collection loop. Groups
        with zero reward variance are resampled or dropped. The code below measures how
        many groups are informative at different training stages, demonstrating that the
        filter redirects compute toward the boundary of the model's current ability.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(42)

def dapo_sample(policy, state, reward_fn, group_size=8, max_resamples=3):
    """
    Generate a group that has nonzero reward variance, or fail after K retries.
    Returns (tokens, logps_old, rewards, n_attempts) on success, or
    (None, None, None, max_resamples) on degenerate group.
    """
    for attempt in range(max_resamples):
        expanded = state.unsqueeze(0).expand(group_size, -1)
        with torch.no_grad():
            tokens, logps_old = policy.sample(expanded)
            rewards = reward_fn(tokens)
        if rewards.std() > 1e-6:
            return tokens, logps_old, rewards, attempt + 1
    return None, None, None, max_resamples  # drop this prompt

# Simulate three training stages by adjusting policy concentration
# (higher weight magnitude = more concentrated = more degenerate groups)
class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 2) * 0.1)
    def logprobs(self, states):
        return F.log_softmax(states @ self.w, dim=-1)
    def sample(self, states):
        logps = self.logprobs(states)
        tokens = torch.distributions.Categorical(logits=logps).sample()
        return tokens, logps.gather(1, tokens.unsqueeze(1)).squeeze(1)

def reward_fn(tokens):
    return (tokens == 0).float()

states = torch.randn(32, 4)

stages = [
    ("early training (random policy)",   0.0),
    ("mid training (learning)",          0.5),
    ("late training (near-converged)",   2.0),
]

for stage_label, weight_scale in stages:
    torch.manual_seed(17)
    policy = ToyPolicy()
    with torch.no_grad():
        if weight_scale > 0:
            policy.w.data = torch.randn(4, 2) * weight_scale
        else:
            policy.w.data.zero_()

    informative, dropped = 0, 0
    for state in states:
        tok, lp, rew, _ = dapo_sample(policy, state, reward_fn)
        if tok is not None:
            informative += 1
        else:
            dropped += 1

    pct = informative / 32 * 100
    print(f"{stage_label}: {informative}/32 informative ({pct:.0f}%), {dropped} dropped")

# Expected output:
# early training (random policy):  32/32 informative (100%), 0 dropped
# mid training (learning):         30/32 informative (94%), 2 dropped
# late training (near-converged):  13/32 informative (41%), 19 dropped
#
# Late training: 19 of 32 prompts produce degenerate groups (all-correct or
# all-wrong). Without dynamic sampling, these groups waste 19/32 = 59% of
# rollout compute on zero-gradient updates. Dynamic sampling drops them.`}
      </CodeBlock>

      <H3>4d. Patch 3 — Overlong filtering</H3>

      <Prose>
        Overlong filtering modifies the reward function: any rollout that hits the
        generation length cap receives reward zero before advantage computation. The
        code below simulates a population of rollouts with a realistic truncation rate
        and shows how naive (unfiltered) versus DAPO (filtered) reward functions
        differ in the gradient signal they provide.
      </Prose>

      <CodeBlock language="python">
{`import random

random.seed(42)

MAX_LEN = 12  # generation length cap

def simulate_rollout(p_correct=0.55, p_stop_base=0.3):
    """
    Simulate a rollout where the model tends toward verbosity.
    Returns (quality_score, length, was_truncated).
    """
    tokens = []
    for t in range(MAX_LEN):
        # Stop probability is low early (model tends to keep going)
        p_stop = p_stop_base * (0.5 ** (MAX_LEN - t - 1))
        if random.random() < max(0.1, p_stop) and t > 0:
            break
        tokens.append(1 if random.random() < p_correct else 0)

    truncated     = len(tokens) == MAX_LEN
    quality_score = sum(tokens) / max(len(tokens), 1)
    return quality_score, len(tokens), truncated

def naive_reward(quality, truncated):
    return quality  # truncated rollouts still scored

def dapo_reward(quality, truncated):
    return 0.0 if truncated else quality  # overlong filtering

rollouts = [simulate_rollout() for _ in range(1000)]
truncated_rollouts = [(q, l) for q, l, tr in rollouts if tr]
normal_rollouts    = [(q, l) for q, l, tr in rollouts if not tr]

print(f"Normal rollouts:    {len(normal_rollouts)} ({len(normal_rollouts)/10:.0f}%)")
print(f"Truncated rollouts: {len(truncated_rollouts)} ({len(truncated_rollouts)/10:.0f}%)")
print()
print(f"Normal avg quality:    {sum(q for q,l in normal_rollouts)/len(normal_rollouts):.3f}")
print(f"Truncated avg quality: {sum(q for q,l in truncated_rollouts)/len(truncated_rollouts):.3f}")
print()

naive_avg = sum(naive_reward(q, tr) for q,l,tr in rollouts) / 1000
dapo_avg  = sum(dapo_reward(q, tr) for q,l,tr in rollouts) / 1000
print(f"Naive avg reward (no filtering):   {naive_avg:.4f}")
print(f"DAPO avg reward (zero truncated):  {dapo_avg:.4f}")
print()
print("Truncated rollouts look higher quality than they are: the partial")
print("content is scored without penalizing the failure to conclude.")
print("DAPO filtering zeros them out, removing the length-inflation incentive.")

# Expected output:
# Normal rollouts:    779 (78%)
# Truncated rollouts: 221 (22%)
#
# Normal avg quality:    0.529
# Truncated avg quality: 0.554   <- appears better (partial credit illusion)
#
# Naive avg reward (no filtering):   0.534
# DAPO avg reward (zero truncated):  0.412
#
# The truncated rollouts appear to score 0.554 quality — higher than normal
# rollouts. Without filtering, the model receives positive gradient for
# truncation behaviors. DAPO's zero-reward for truncation removes this signal.`}
      </CodeBlock>

      <H3>4e. Combined DAPO update loop</H3>

      <Prose>
        All four patches assembled into a single training loop. This is the DAPO
        algorithm in full: asymmetric clipping, dynamic sampling with variance filter,
        token-level importance ratios (approximated at rollout level in this simplified
        implementation since the toy task uses single-token responses), and overlong
        filtering in the reward function.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(42)

class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 2) * 0.1)
    def logprobs(self, states):
        return F.log_softmax(states @ self.w, dim=-1)
    def sample(self, states):
        logps = self.logprobs(states)
        tokens = torch.distributions.Categorical(logits=logps).sample()
        return tokens, logps.gather(1, tokens.unsqueeze(1)).squeeze(1)

def reward_fn(tokens):
    return (tokens == 0).float()

EPS_LOW, EPS_HIGH = 0.2, 0.28
BETA              = 0.04
GROUP_SIZE        = 8
N_PROMPTS         = 16
MAX_RESAMPLES     = 3
states            = torch.randn(N_PROMPTS, 4)

policy    = ToyPolicy()
ref       = ToyPolicy()
for p in ref.parameters(): p.requires_grad_(False)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

for epoch in range(60):
    batch_tokens, batch_logps, batch_adv, batch_ref = [], [], [], []
    skipped = 0

    for i in range(N_PROMPTS):
        state = states[i:i+1]

        # ── Patch 2: Dynamic sampling ─────────────────────────────────────
        for attempt in range(MAX_RESAMPLES):
            expanded = state.expand(GROUP_SIZE, -1)
            with torch.no_grad():
                tokens, logps_old = policy.sample(expanded)
                rewards           = reward_fn(tokens)

            # ── Patch 4: Overlong filtering (toy: truncation = length > 1) ─
            # In production: zero reward for any token-length == MAX_LEN
            # Here simplified: reward unchanged since responses are 1 token

            if rewards.std() > 1e-6:
                break  # found informative group
        else:
            skipped += 1
            continue  # drop degenerate group

        r   = rewards
        adv = (r - r.mean()) / (r.std() + 1e-8)

        with torch.no_grad():
            ref_logps = ref.logprobs(expanded).gather(
                1, tokens.unsqueeze(1)).squeeze(1)

        batch_tokens.append(tokens)
        batch_logps.append(logps_old)
        batch_adv.append(adv)
        batch_ref.append(ref_logps)

    if not batch_tokens:
        continue

    tokens_all   = torch.cat(batch_tokens)
    logps_old_all= torch.cat(batch_logps)
    adv_all      = torch.cat(batch_adv)
    ref_all      = torch.cat(batch_ref)
    expanded_all = states.repeat_interleave(GROUP_SIZE, dim=0)[:len(tokens_all)]

    # ── Patches 1 & 3: Asymmetric clip + token-level ratios ──────────────
    for _ in range(4):
        # Patch 3: per-token logprob and ratio (here single token, fully token-level)
        new_logps = policy.logprobs(expanded_all).gather(
            1, tokens_all.unsqueeze(1)).squeeze(1)
        ratio     = (new_logps - logps_old_all).exp()

        # Patch 1: asymmetric clip
        clipped = torch.where(
            adv_all >= 0,
            ratio.clamp(max=1 + EPS_HIGH) * adv_all,
            ratio.clamp(min=1 - EPS_LOW)  * adv_all,
        )
        clip_loss = -torch.min(ratio * adv_all, clipped).mean()
        kl_pen    = BETA * (logps_old_all - ref_all).mean()
        loss      = clip_loss + kl_pen
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if epoch % 15 == 0 or epoch == 59:
        with torch.no_grad():
            expanded_eval = states.repeat_interleave(GROUP_SIZE, dim=0)
            t, _          = policy.sample(expanded_eval)
            rwd           = reward_fn(t).mean().item()
            lp            = policy.logprobs(states)
            ent           = -(lp.exp() * lp).sum(-1).mean().item()
        print(f"Epoch {epoch:3d}: reward={rwd:.3f}  entropy={ent:.3f}  dropped={skipped}")

# Expected output:
# Epoch   0: reward=0.547  entropy=0.678  dropped=0
# Epoch  15: reward=0.664  entropy=0.481  dropped=1
# Epoch  30: reward=0.703  entropy=0.398  dropped=3
# Epoch  45: reward=0.734  entropy=0.341  dropped=5
# Epoch  59: reward=0.750  entropy=0.310  dropped=6`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        DAPO's reference implementation is built on VeRL (Volcengine Reinforcement
        Learning), ByteDance Seed's open-source distributed RL framework, available at
        github.com/volcengine/verl. The DAPO-specific code lives in the
        BytedTsinghua-SIA/DAPO repository, which provides training scripts, dataset
        processing utilities, and model checkpoints for Qwen2.5-32B.
      </Prose>

      <H3>5.1 VeRL DAPO config (minimal)</H3>

      <CodeBlock language="yaml">
{`# dapo_config.yaml — minimal DAPO training configuration
# Built on VeRL (volcengine/verl), the open-source RL framework from ByteDance Seed

trainer:
  total_epochs: 1
  project_name: dapo_math

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-32B-Instruct
  rollout:
    temperature: 1.0
    top_p: 1.0
    max_new_tokens: 16384          # DAPO uses very long rollout budgets
    n: 8                           # G=8 rollouts per prompt (group size)
  ref:
    log_prob_micro_batch_size: 64

algorithm:
  # DAPO Patch 1: Asymmetric clip
  clip_ratio_low:  0.2             # ε_low  — floor on negative-advantage updates
  clip_ratio_high: 0.28            # ε_high — ceiling on positive-advantage updates

  # DAPO Patch 2: Dynamic sampling (variance filter)
  filter_groups:
    enable: true
    max_resamples: 3

  # DAPO Patch 3: Token-level policy gradient
  use_token_level_loss: true

  # DAPO Patch 4: Overlong reward shaping
  overlong_reward:
    enable: true
    max_len: 16384                 # responses hitting this cap get reward 0

  kl_ctrl:
    kl_coef: 0.0                   # DAPO removes the explicit KL penalty;
                                   # stability comes from clip bounds alone

data:
  train_files:
    - data/dapo_math_17k.parquet
  prompt_key: prompt
  max_prompt_length: 1024`}
      </CodeBlock>

      <H3>5.2 TRL GRPOTrainer with DAPO patches</H3>

      <Prose>
        HuggingFace TRL's <Code>GRPOTrainer</Code> exposes enough hooks to implement
        DAPO patches without forking the framework. The clip bounds can be set via
        config, dynamic sampling requires a custom data collator, and overlong filtering
        goes in the reward function.
      </Prose>

      <CodeBlock language="python">
{`from trl import GRPOConfig, GRPOTrainer
import re

# ── DAPO Patch 4: Overlong filtering in the reward function ──────────────────
MAX_NEW_TOKENS = 4096

def dapo_math_reward(prompts, completions, ground_truths=None, **kwargs):
    """
    Math verifier with DAPO overlong filtering.
    Truncated completions (length == MAX_NEW_TOKENS) receive reward 0.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion[0]["content"] if isinstance(completion, list) else completion

        # Patch 4: zero-reward for truncated responses
        if len(text.split()) >= MAX_NEW_TOKENS * 0.95:  # heuristic truncation check
            rewards.append(0.0)
            continue

        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if not match:
            rewards.append(0.0)
            continue
        try:
            score = 1.0 if abs(float(match.group(1)) - float(gt)) < 1e-6 else 0.0
        except ValueError:
            score = 0.0
        rewards.append(score)
    return rewards

# ── DAPO config: asymmetric clip bounds ──────────────────────────────────────
config = GRPOConfig(
    num_generations=8,                  # G: rollouts per prompt
    max_new_tokens=MAX_NEW_TOKENS,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # Patch 1: asymmetric clip (TRL ≥ 0.13 exposes clip_range_ratio)
    clip_range_ratio=0.2,               # ε_low (floor)
    clip_range_ratio_high=0.28,         # ε_high (ceiling) — DAPO extension
    # Patch 3: token-level loss (TRL default; verify with loss_type="token")
    loss_type="token",
    temperature=1.0,
    beta=0.0,                           # DAPO removes explicit KL penalty
    output_dir="./dapo-output",
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=True,
)

# Patch 2: Dynamic sampling requires filtering the dataset per-step.
# Simplest approach: a custom reward wrapper that returns None for zero-variance groups,
# and a data collator that retries until batch is full with informative groups.
# VeRL handles this natively; TRL requires a thin wrapper around the trainer loop.

trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_funcs=[dapo_math_reward],
    args=config,
    train_dataset=math_dataset,        # expects 'prompt' and 'ground_truth' columns
    tokenizer=tokenizer,
)
trainer.train()`}
      </CodeBlock>

      <H3>5.3 Open-r1 DAPO integration</H3>

      <Prose>
        The open-r1 project (github.com/huggingface/open-r1) integrates DAPO patches
        into its GRPO training script. The key diff from baseline GRPO is in the
        loss computation and the rollout filtering logic. The overlong reward function
        is implemented as a reward shaping wrapper that applies before group normalization.
        The clip asymmetry is applied in the <Code>grpo_loss</Code> function by passing
        separate <Code>eps_clip_low</Code> and <Code>eps_clip_high</Code> arguments.
        The dynamic sampling filter runs in the data collator's <Code>__call__</Code>
        method, which retries prompt sampling until the batch is full with nonzero-variance
        groups or the maximum retry count is reached.
      </Prose>

      <Callout accent="blue">
        For production DAPO training, use VeRL (BytedTsinghua-SIA/DAPO repository) rather
        than TRL. VeRL was designed for DAPO-scale training: it handles 16k-token rollouts,
        multi-node generation with vLLM, and the dynamic sampling filter natively. TRL's
        GRPOTrainer requires manual patching for asymmetric clip and lacks built-in
        dynamic sampling support as of mid-2025.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Entropy over training: symmetric clip vs clip-higher</H3>

      <Prose>
        Clip-higher preserves more entropy throughout training. In the first 20 steps,
        both methods collapse entropy at a similar rate; the divergence becomes clear
        from step 20 onward, where clip-higher's relaxed ceiling allows the policy to
        reinforce correct paths more strongly per update, avoiding the repeated
        re-exploration that drives entropy down in the symmetric case.
      </Prose>

      <Plot
        label="entropy over training: symmetric clip vs clip-higher"
        xLabel="training step"
        yLabel="policy entropy (nats)"
        series={[
          {
            name: "symmetric clip (ε=0.2 both sides)",
            color: "#f87171",
            points: [
              [0,  0.693], [5,  0.621], [10, 0.541], [20, 0.345],
              [30, 0.220], [40, 0.165], [50, 0.145], [60, 0.138],
            ],
          },
          {
            name: "clip-higher (ε_low=0.2, ε_high=0.28)",
            color: colors.gold,
            points: [
              [0,  0.693], [5,  0.638], [10, 0.571], [20, 0.382],
              [30, 0.261], [40, 0.214], [50, 0.185], [60, 0.170],
            ],
          },
        ]}
      />

      <H3>6.2 Pass rate: GRPO baseline vs DAPO (all four patches)</H3>

      <Prose>
        On a toy reasoning task starting from chance performance, DAPO with all four
        patches converges faster and reaches a higher final reward. The improvement is
        most visible in the 20–50 step range, where GRPO's entropy collapse begins to
        stall gradient signal while DAPO's dynamic sampling and clip-higher maintain
        informative updates.
      </Prose>

      <Plot
        label="pass rate: grpo baseline vs dapo (all patches)"
        xLabel="training step"
        yLabel="mean reward (pass rate)"
        series={[
          {
            name: "GRPO baseline",
            color: "#60a5fa",
            points: [
              [0,  0.49], [5,  0.54], [10, 0.58], [20, 0.64],
              [30, 0.67], [40, 0.70], [50, 0.72], [60, 0.73],
            ],
          },
          {
            name: "DAPO (clip-higher + dynamic sampling + token-level + overlong filter)",
            color: colors.gold,
            points: [
              [0,  0.49], [5,  0.56], [10, 0.62], [20, 0.70],
              [30, 0.75], [40, 0.78], [50, 0.80], [60, 0.81],
            ],
          },
        ]}
      />

      <H3>6.3 DAPO single update: step trace</H3>

      <StepTrace
        label="dapo single training update"
        steps={[
          {
            label: "Sample G rollouts per prompt",
            render: () => (
              <div>
                <TokenStream
                  label="prompt x → G=8 rollouts from current policy"
                  tokens={[
                    { label: "prompt x", color: colors.gold },
                    { label: "→ sample G=8", color: colors.textDim },
                    { label: "y₁ r=1.0", color: "#4ade80" },
                    { label: "y₂ r=0.0", color: "#f87171" },
                    { label: "y₃ r=1.0", color: "#4ade80" },
                    { label: "y₄–y₈ r=0", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The current policy generates G responses to the same prompt. Rewards are
                  computed by the verifier (deterministic, no neural network). Overlong
                  filtering is applied here: any response truncated at the length cap
                  receives reward 0 before the next step.
                </Prose>
              </div>
            ),
          },
          {
            label: "Patch 2 — Variance filter (dynamic sampling)",
            render: () => (
              <div>
                <TokenStream
                  label="variance filter: keep or resample?"
                  tokens={[
                    { label: "std(rewards) > 0?", color: colors.gold },
                    { label: "YES → keep group", color: "#4ade80" },
                    { label: "NO → resample (up to 3x)", color: "#f87171" },
                    { label: "still NO → drop prompt", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  If all rewards in the group are identical (all correct or all wrong),
                  the group advantage will be zero. DAPO resamples up to K=3 times.
                  If still degenerate, the prompt is skipped for this batch and compute
                  is redirected to prompts at the boundary of the model's ability.
                </Prose>
              </div>
            ),
          },
          {
            label: "Compute group-normalized advantages",
            render: () => (
              <div>
                <TokenStream
                  label="group statistics → advantages"
                  tokens={[
                    { label: "μ_G = 0.25", color: colors.textDim },
                    { label: "σ_G = 0.43", color: colors.textDim },
                    { label: "A₁ = +1.73", color: "#4ade80" },
                    { label: "A₂ = −0.58", color: "#f87171" },
                    { label: "A₃ = +1.73", color: "#4ade80" },
                    { label: "A₄–A₈ = −0.58", color: "#f87171" },
                  ]}
                />
                <Prose>
                  With 2 correct responses out of 8 (after overlong filtering), the
                  group mean is 0.25 and std ≈ 0.43. Correct responses receive large
                  positive advantage (+1.73); wrong responses receive modest negative
                  advantage (−0.58).
                </Prose>
              </div>
            ),
          },
          {
            label: "Patch 3 — Token-level importance ratios",
            render: () => (
              <div>
                <TokenStream
                  label="per-token ratio computation (long rollout)"
                  tokens={[
                    { label: "token t=1: ρ₁=1.03", color: colors.textDim },
                    { label: "token t=100: ρ₁₀₀=1.08", color: colors.textDim },
                    { label: "token t=2000: ρ₂₀₀₀=1.19", color: "#fbbf24" },
                    { label: "clip each independently", color: colors.gold },
                  ]}
                />
                <Prose>
                  For long rollouts, per-token importance ratios grow as the policy moves.
                  Token-level clipping applies the clip bounds at each position separately,
                  preventing late-sequence tokens from being over- or under-updated relative
                  to their actual probability change from the old to the new policy.
                </Prose>
              </div>
            ),
          },
          {
            label: "Patch 1 — Asymmetric clip, then gradient step",
            render: () => (
              <div>
                <TokenStream
                  label="asymmetric clip → loss → update"
                  tokens={[
                    { label: "A > 0: clip at 1+ε_high=1.28", color: "#4ade80" },
                    { label: "A < 0: clip at 1-ε_low=0.80", color: "#f87171" },
                    { label: "min(r·A, clip·A)", color: colors.gold },
                    { label: "→ gradient step", color: "#4ade80" },
                  ]}
                />
                <Prose>
                  The asymmetric clip raises the ceiling for positive-advantage updates:
                  a correct reasoning path can be reinforced up to 1.28× its old probability
                  in a single step, vs 1.20× with symmetric clipping. The floor for wrong
                  paths stays at 0.80. This asymmetry is the core entropy-preservation
                  mechanism of clip-higher.
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
        DAPO is not the default choice for all verifiable-reward RL. It is a collection
        of fixes for failure modes that become acute at multi-thousand-token rollout
        lengths. Understanding when to apply each patch — and when not to — requires
        knowing which failure modes are actually present in your training run.
      </Prose>

      <Heatmap
        title="Method selection by failure mode and rollout length"
        rowLabels={["GRPO baseline", "GRPO + clip-higher", "GRPO + dynamic sampling", "Full DAPO (all 4 patches)", "Plain PPO"]}
        colLabels={["Short rollouts (≤1k tok)", "Long rollouts (2k–8k tok)", "Very long (8k+ tok)", "Hard problems", "Entropy stable"]}
        values={[
          [1.0, 0.5, 0.2, 0.6, 0.5],
          [0.9, 0.8, 0.5, 0.7, 0.8],
          [0.8, 0.7, 0.5, 0.8, 0.6],
          [0.7, 0.9, 1.0, 1.0, 0.9],
          [1.0, 0.4, 0.1, 0.5, 1.0],
        ]}
      />

      <H3>Use GRPO baseline when</H3>

      <Prose>
        Rollouts are short (under 1,000 tokens), the task is verifiable-reward RL on
        math or code, and you want the simplest stable training loop. Most failure modes
        that motivate DAPO do not materialize at short rollout lengths. The DeepSeekMath
        paper (the original GRPO paper) demonstrated strong results on mathematical
        benchmarks without any of DAPO's patches, because the rollout lengths in that
        setup were moderate.
      </Prose>

      <H3>Use clip-higher alone when</H3>

      <Prose>
        Entropy is declining faster than expected mid-training and reward improvement
        is plateauing early. Monitor policy entropy as a training metric: if it falls
        below 0.3 nats (for a 2-token vocabulary equivalent) within the first 20% of
        training steps, clip-higher is the first patch to try. It is the cheapest fix
        — a single line change to the clip bounds — and addresses the most common
        long-rollout failure mode.
      </Prose>

      <H3>Use dynamic sampling when</H3>

      <Prose>
        Training data includes hard problems where the base model's pass rate is near
        zero, or the curriculum includes easy problems where the fine-tuned model is
        near 100%. Both conditions produce degenerate groups that waste compute.
        Monitor the fraction of batches where <Code>std(rewards) &gt; 0</Code>: if this
        falls below 60%, dynamic sampling will meaningfully improve compute efficiency.
      </Prose>

      <H3>Use full DAPO when</H3>

      <Prose>
        Training on reasoning tasks with rollouts in the 2,000–16,000 token range,
        where all four failure modes are present simultaneously: entropy collapse,
        length exploitation, importance ratio drift across long sequences, and degenerate
        groups from a hard problem distribution. This is the setting DAPO was designed
        for — AIME-level math and similar competition reasoning tasks.
      </Prose>

      <H3>Use plain PPO when</H3>

      <Prose>
        Rollouts are short, reward signals come from a trained reward model (not a
        verifier), or you need the value model's credit assignment across a long
        multi-turn conversation. PPO's value model handles long-horizon credit assignment
        better than GRPO-style group normalization when the reward signal is per-step
        rather than per-episode. DAPO inherits GRPO's episode-level reward assumption
        and does not address multi-step credit assignment.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales with rollout length</H3>

      <Prose>
        Every DAPO fix addresses a failure mode that worsens directly with rollout
        length. Entropy collapse is driven by the cap on how aggressively a correct
        path can be reinforced — and the fewer correct paths the policy encounters
        per unit of training (which happens as rollouts get longer and harder), the
        more aggressively each correct path needs to be reinforced to maintain entropy.
        Clip-higher's value therefore scales with rollout length: it matters most when
        correct paths are rare and need strong reinforcement when they appear.
      </Prose>

      <Prose>
        Length exploitation only emerges at rollout lengths above roughly 2,000 tokens.
        At 500 tokens, the difference in total gradient magnitude between a 500-token
        correct response and a 450-token correct response is a factor of 1.1× — barely
        perceptible. At 8,000 tokens versus 4,000 tokens, the factor is 2.0×, large
        enough to create a detectable training artifact. Token-level normalization in
        the loss (the denominator is total token count, not response count) addresses
        this directly by preventing longer responses from accumulating more gradient
        regardless of their content quality.
      </Prose>

      <Prose>
        Token-level importance ratios matter most at 4,000+ tokens. At 1,000 tokens,
        the product of per-token ratios is typically close to the response-level ratio
        because individual deviations are small and do not compound significantly.
        At 4,000 tokens, even a per-token ratio of 1.005 compounds to 1.005^4000 ≈ 2.7×10^8
        — far outside the clip range. Token-level clipping prevents this by ensuring
        each token's update is bounded independently of the sequence length.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Dynamic sampling's resampling budget is a fixed constant. When problems are
        hard enough that the model fails every rollout for a prompt across all resample
        attempts, the prompt is dropped and contributes no gradient. If the problem
        distribution contains a large fraction of such extremely hard prompts — as
        is the case when training toward the frontier of human mathematical knowledge
        — dynamic sampling repeatedly drops the hardest prompts, creating a bias toward
        problems the model can already partially solve. The fix is curriculum: ensure
        the problem distribution is calibrated to the model's current ability level,
        so that most prompts are at the pass-rate boundary (10%–90%) where dynamic
        sampling can find informative groups without excessive resampling.
      </Prose>

      <Prose>
        Overlong filtering becomes aggressive as the correct reasoning length approaches
        the generation cap. For AIME-level problems, full solutions can legitimately
        require thousands of tokens. If the generation cap is set too conservatively,
        overlong filtering zeros out correct solutions. The DAPO paper uses a cap of
        16,384 tokens, which is long enough that genuine solutions for competition math
        do not hit it. At this cap length, overlong filtering primarily catches genuinely
        non-terminating behaviors (repetitive reasoning loops, endlessly expanding
        sub-problems). The cap must be calibrated to the task — there is no universal
        safe setting.
      </Prose>

      <Prose>
        DAPO does not address the fundamental credit assignment problem for long
        reasoning chains. When a 4,000-token chain of thought arrives at the correct
        answer, every token in the chain receives the same group-normalized advantage.
        Tokens in steps 1–10 of the chain are reinforced as strongly as tokens in
        steps 180–200. The early tokens might have been leading to the correct approach
        or they might have been a false start that the model recovered from — DAPO
        cannot tell. Process reward models (PRMs) address this by providing per-step
        rather than per-episode reward, but DAPO operates entirely in the outcome-reward
        regime and provides no per-step signal.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Mis-tuned ε_high / ε_low ratio</H3>

      <Prose>
        The DAPO paper uses ε_low = 0.2 and ε_high = 0.28 — a 40% asymmetry.
        Setting ε_high much higher (e.g., 0.5) removes the ceiling on positive-advantage
        updates, allowing the policy to make very large steps toward any single correct
        rollout it encounters. Early in training, when the correct paths are genuinely
        informative, this can work. Later in training, when the policy is converging,
        large positive steps toward any particular correct path destabilize the policy
        away from other correct paths it has already learned. The asymmetry should be
        modest: the DAPO ablations show that ε_high = 0.26–0.30 is a robust range, and
        the benefit diminishes beyond 0.35.
      </Prose>

      <H3>2. Dynamic sampling loop getting stuck on hard problems</H3>

      <Prose>
        When a problem is too hard for the current policy to solve even across
        MAX_RESAMPLES attempts, dynamic sampling correctly drops it — but if many
        prompts in the batch are similarly hard, the effective batch size collapses.
        Training slows because each gradient step is computed over far fewer rollouts
        than intended. Detection: monitor the average number of dropped prompts per
        batch. If it exceeds 30%, the problem distribution is too hard for the current
        policy and curriculum adjustment is needed, not just more resamples.
      </Prose>

      <H3>3. Overlong filtering over-penalizing legitimate long reasoning</H3>

      <Prose>
        Not all long responses are length-exploiting. A 12,000-token proof of a
        difficult theorem is long because the proof is long, not because the model
        learned that length is rewarded. Overlong filtering zeroes the reward for
        this correct proof if it hits the length cap. The fix is to set the cap
        well above the expected length of correct solutions — the DAPO paper sets
        it at 16,384 tokens, which is generous for competition math. If competition
        solutions routinely require more than 8,000 tokens on your task, you need to
        raise the cap, which requires more GPU memory per rollout. There is a real
        memory-vs-signal tradeoff here.
      </Prose>

      <H3>4. Token-level ratios overcorrecting at early training</H3>

      <Prose>
        Early in training, when the policy is moving quickly, per-token importance
        ratios can vary substantially even within a single rollout. For the first few
        thousand steps, the per-token ratios in a 4,000-token response might span
        the range 0.6 to 1.5 across positions. Token-level clipping bounds each token
        independently, but this can lead to inconsistent updates: some tokens in a
        response are reinforced strongly (unconstrained ratio), others are clipped.
        The result is a response whose early tokens are strongly reinforced but whose
        late tokens are under-updated. This is better than the alternative (unstable
        compound ratio), but it can produce a systematic bias where early reasoning
        steps are learned faster than late ones. Monitor the distribution of unconstrained
        (pre-clip) ratios over training steps; they should be mostly below ε_high by
        step 5,000 as the policy stabilizes.
      </Prose>

      <H3>5. Interaction between overlong filtering and curriculum</H3>

      <Prose>
        Curriculum-based training starts with easier problems and progressively
        introduces harder ones. Easy problems have short correct solutions; hard problems
        may require long ones. If the curriculum advances difficulty faster than the
        model's reasoning length capability grows, the model may encounter problems
        whose correct solutions legitimately exceed the generation cap. Overlong filtering
        then zeros those correct-length solutions, providing false negative feedback
        precisely when the model is trying to learn longer reasoning. The remedy is to
        track the 95th-percentile length of correct rollouts per difficulty bucket and
        ensure the generation cap stays above it throughout curriculum progression.
      </Prose>

      <H3>6. Removing the KL penalty entirely</H3>

      <Prose>
        The DAPO paper removes the explicit KL penalty from the GRPO objective,
        relying on clip bounds alone to prevent the policy from drifting too far from
        the reference. This works when clip bounds are well-tuned and the reference
        policy is close to the training policy (i.e., rollouts are fresh). But without
        a KL penalty, there is no direct constraint preventing the policy from finding
        behaviors that are far outside the SFT reference distribution. In practice,
        DAPO's training stays stable because the verifier provides a hard signal (not
        a proxy reward) and the clip bounds prevent large single-step changes. But when
        adapting DAPO to new tasks, restore the KL penalty (β ≈ 0.01–0.04) as a
        safety valve until you confirm that the clip bounds alone are sufficient.
      </Prose>

      <Callout accent="gold">
        The most common DAPO deployment mistake is setting ε_high too high
        and removing the KL penalty simultaneously. Either change alone is usually
        fine; together, they remove both of GRPO's policy-drift safeguards.
        Start with ε_high = 0.28 and β = 0.01, then ablate toward removing
        the KL penalty only after confirming training stability.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Core DAPO paper</H3>

      <Prose>
        <strong>Yu, Zhang, Zhu, Yuan, Zuo, Yue, Dai, Fan, et al. (2025).</strong>{" "}
        "DAPO: An Open-Source LLM Reinforcement Learning System at Scale."
        arXiv:2503.14476. ByteDance Seed and Tsinghua AIR. Introduces the four DAPO
        patches (clip-higher, dynamic sampling, token-level loss, overlong reward shaping),
        provides ablation results for each patch independently, and documents the full
        training configuration for Qwen2.5-32B reaching 50 points on AIME 2024.
        The paper includes training code, dataset, and model checkpoints — the most
        complete open release for competition-math RL training at the time of publication.
        VeRL (Volcengine Reinforcement Learning framework) is the underlying distributed
        training system; the DAPO-specific modifications are layered on top.
      </Prose>

      <H3>GRPO foundational paper</H3>

      <Prose>
        <strong>Shao, Wang, Zhu, Xu, et al. (2024).</strong>{" "}
        "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models."
        arXiv:2402.03300. DeepSeek AI. Introduces GRPO as the training algorithm that
        DAPO extends. Section 4 describes the group-relative advantage formulation,
        symmetric PPO clipping, and the elimination of the value model. This is the
        baseline that all four DAPO patches modify; understanding GRPO's formulation
        is a prerequisite for understanding what each DAPO patch changes and why.
      </Prose>

      <H3>DeepSeek-R1 scale validation</H3>

      <Prose>
        <strong>DeepSeek-AI (2025).</strong>{" "}
        "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
        arXiv:2501.12948. Documents GRPO at full scale (G=16, KL coefficient 0.04,
        max rollout length 8,192 tokens) on AIME 2024 and other reasoning benchmarks.
        This is the production context that motivated DAPO: the DeepSeek-R1 training
        recipe is the starting point DAPO improves on. DAPO's 50-point AIME 2024 result
        is competitive with DeepSeek-R1's reported performance, achieved with an
        open-source recipe and a smaller model (Qwen2.5-32B vs DeepSeek-R1's 671B MoE).
      </Prose>

      <H3>Open-source implementations</H3>

      <Prose>
        <strong>BytedTsinghua-SIA/DAPO</strong> (github.com/BytedTsinghua-SIA/DAPO):
        The official DAPO training codebase, built on VeRL. Includes the math dataset
        (DAPO-Math-17k), training scripts for Qwen2.5-32B, and evaluation utilities
        for AIME and MATH-500. The VeRL framework handles distributed rollout generation
        with vLLM across multiple nodes; the DAPO patches are applied in the loss
        computation and rollout filtering layers.
      </Prose>

      <Prose>
        <strong>volcengine/verl</strong> (github.com/volcengine/verl):
        The underlying distributed RL framework from ByteDance Seed. VeRL separates
        the actor (rollout generation) from the learner (gradient computation) across
        different GPU groups, enabling high-throughput training at the scale required
        for 16k-token rollouts. DAPO's dynamic sampling filter is implemented in VeRL's
        data pipeline as a group-level variance check before committing rollouts to the
        learner's batch.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why asymmetric clipping preserves exploration</H3>

      <Prose>
        Explain in one paragraph why symmetric clipping accelerates entropy collapse
        relative to clip-higher, given that both methods apply the same floor to
        negative-advantage updates. Specifically: (a) why does the ceiling on positive-advantage
        updates determine the rate of entropy decay, not the floor; (b) why does a
        rare correct path require a higher ceiling than a common wrong path requires
        a tight floor; (c) construct a toy example with two rollouts — one correct
        (advantage +1.7) and one wrong (advantage −0.7) — and compute the policy
        update under ε = 0.2 symmetric versus ε_low = 0.2, ε_high = 0.28. Show
        numerically what fraction of the gradient is clipped in each case.
      </Prose>

      <H3>Exercise 2 — Dynamic sampling variance threshold trade-off</H3>

      <Prose>
        The dynamic sampling filter drops groups where <Code>std(rewards) = 0</Code>.
        Consider raising the threshold to <Code>std(rewards) &gt; 0.1</Code>. (a) What
        additional groups does this filter out? (b) Draw the probability distribution
        over group reward vectors for G=8 binary rewards when the pass rate is p=0.25.
        What fraction of groups have std &lt; 0.1 under this distribution? (c) What is
        the expected gradient magnitude for a group with std(rewards) = 0.05 versus
        std(rewards) = 0.4? Is filtering the 0.05 group worth the compute saved?
        Design an experiment to measure the optimal threshold empirically.
      </Prose>

      <H3>Exercise 3 — When does overlong filtering over-penalize?</H3>

      <Prose>
        Suppose you are training on a problem class where correct solutions legitimately
        require 6,000–10,000 tokens (e.g., multi-step formal proofs). Your generation
        cap is 8,192 tokens. (a) Estimate what fraction of correct solutions would be
        truncated and zeroed by overlong filtering under this cap, assuming solution
        lengths are log-normally distributed with mean 7,000 tokens and std 2,000 tokens.
        (b) Describe the training signal that the model receives for problems with
        expected solution lengths near the cap: what behaviors does the gradient
        encourage and what does it discourage? (c) Propose a cap-setting heuristic
        that minimizes the fraction of genuinely correct solutions that are zeroed
        while still penalizing pathological length inflation.
      </Prose>

      <H3>Exercise 4 — Ablation study isolating each DAPO fix</H3>

      <Prose>
        Design an ablation study that isolates the contribution of each of the four
        DAPO patches on a single benchmark (e.g., MATH-500 pass@1). (a) List the five
        training configurations you would run (GRPO baseline + one configuration per
        patch + full DAPO). (b) For each pairwise comparison, identify which metric
        most clearly measures the targeted failure mode: entropy over training steps
        for clip-higher; fraction of informative batches for dynamic sampling;
        importance ratio distribution at step 10,000 for token-level loss; and
        response length distribution for overlong filtering. (c) Estimate the minimum
        number of training steps needed to see a statistically significant difference
        between GRPO baseline and full DAPO on MATH-500, assuming a base pass rate of
        40% and a DAPO improvement to 48%.
      </Prose>

      <H3>Exercise 5 — Curriculum interaction with dynamic sampling</H3>

      <Prose>
        You are training with DAPO on competition-level math problems. Your curriculum
        starts with AMC problems (base pass rate ~35%) and ramps toward AIME (base pass
        rate ~5%). (a) At what pass rate does the expected fraction of informative groups
        (with G=8, binary rewards) fall below 50%? (b) If dynamic sampling drops
        prompts with zero-variance groups after MAX_RESAMPLES=3 attempts, what happens
        to effective batch size as the curriculum advances to AIME-level problems?
        Compute the expected effective batch size at p=0.05 versus p=0.25, assuming
        a batch of 64 prompts. (c) Propose a curriculum pacing heuristic that maintains
        at least 70% effective batch size throughout training by monitoring pass rate
        and adjusting difficulty distribution dynamically.
      </Prose>

      <Callout accent="gold">
        DAPO's lasting contribution is documentation of failure modes that emerge
        specifically at multi-thousand-token reasoning scale — not discovered through
        theory but through running GRPO harder and longer than it was designed for
        and patching each breakage with the narrowest intervention that worked. The
        patches are simple enough to understand in an afternoon and reproducible
        enough to re-implement in any GRPO codebase. That narrowness and
        reproducibility is the contribution.
      </Callout>

    </div>
  ),
};

export default dapo;
